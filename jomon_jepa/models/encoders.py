from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalPointFallbackEncoder(nn.Module):
    """
    Lightweight local-to-global point cloud encoder used if point-transformer-v3
    weights are unavailable or during fast CPU/WSL debugging.
    """

    def __init__(self, in_channels: int = 9, out_dim: int = 256):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_dim, nhead=4, dim_feedforward=out_dim * 2, batch_first=True, norm_first=True
        )
        self.patch_transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

    def forward(self, coord: torch.Tensor, color: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
        # Sort points along Morton/Z-order approximation (x + y + z) for local patch grouping
        B, N, _ = coord.shape
        sort_key = coord[:, :, 0] * 100.0 + coord[:, :, 1] * 10.0 + coord[:, :, 2]
        idx = torch.argsort(sort_key, dim=1)
        inv_idx = torch.argsort(idx, dim=1)

        feats = torch.cat([coord, color, normal], dim=-1)
        feats_sorted = torch.gather(feats, 1, idx.unsqueeze(-1).expand(-1, -1, feats.shape[-1]))
        h = self.point_mlp(feats_sorted)

        # Group into patches of 32 points for self-attention context
        patch_size = 32
        num_patches = N // patch_size
        h_patches = h[:, : num_patches * patch_size].view(B, num_patches, patch_size, -1)
        patch_summary = h_patches.mean(dim=2)
        patch_context = self.patch_transformer(patch_summary)
        h_enriched = (h_patches + patch_context.unsqueeze(2)).view(B, num_patches * patch_size, -1)

        if num_patches * patch_size < N:
            h_enriched = torch.cat([h_enriched, h[:, num_patches * patch_size :]], dim=1)

        return torch.gather(h_enriched, 1, inv_idx.unsqueeze(-1).expand(-1, -1, h_enriched.shape[-1]))


class PretrainedPTv3ObjectEncoder(nn.Module):
    """
    Wraps pretrained PointTransformerV3 (Sonata or Concerto) to extract per-point
    geometric & color features [B, N, latent_dim].
    """

    def __init__(self, backbone_type: str = "ptv3_sonata", latent_dim: int = 256):
        super().__init__()
        self.backbone_type = backbone_type
        self.latent_dim = latent_dim
        self.ptv3_model = None
        self.proj = None

        if backbone_type in ("ptv3_sonata", "ptv3_concerto", "ptv3_utonia"):
            try:
                import point_transformer_v3 as ptv3

                weight_map = {
                    "ptv3_sonata": "sonata",
                    "ptv3_concerto": "concerto",
                    "ptv3_utonia": "utonia",
                }
                weight_name = weight_map.get(backbone_type, "sonata")
                self.ptv3_model = ptv3.load(weight_name)

                # Probe exact upcasted feature dimension returned by .embed() (e.g., 1232 for Sonata, 1088 for Concerto/Utonia)
                probe_dev = next(self.ptv3_model.parameters()).device
                with torch.no_grad():
                    dummy_c = torch.zeros((16, 3), device=probe_dev)
                    dummy_col = torch.zeros((16, 3), device=probe_dev)
                    dummy_n = torch.zeros((16, 3), device=probe_dev)
                    in_feat_dim = self.ptv3_model.embed(dummy_c, dummy_col, dummy_n).shape[-1]

                self.proj = nn.Sequential(
                    nn.Linear(in_feat_dim, latent_dim),
                    nn.LayerNorm(latent_dim),
                    nn.GELU(),
                )
                print(f"[Encoder] Loaded pretrained PTv3 ({weight_name}, dim={in_feat_dim}) via Triton backend.")
            except Exception as e:
                print(f"[Encoder] Could not load pretrained PTv3 ({e}). Using fallback encoder.")
                self.ptv3_model = None

        if self.ptv3_model is None:
            self.fallback = HierarchicalPointFallbackEncoder(in_channels=9, out_dim=latent_dim)

    def set_backbone_trainable(self, trainable: bool):
        if self.ptv3_model is not None:
            for p in self.ptv3_model.parameters():
                p.requires_grad = trainable
        elif hasattr(self, "fallback"):
            for p in self.fallback.parameters():
                p.requires_grad = trainable

    def forward(self, coord: torch.Tensor, color: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
        if self.ptv3_model is not None:
            B, N, _ = coord.shape
            coord_list = [coord[b] for b in range(B)]
            color_list = [(color[b] * 255.0) for b in range(B)]
            normal_list = [normal[b] for b in range(B)]
            raw_feats = self.ptv3_model.embed(coord_list, color_list, normal_list)
            if isinstance(raw_feats, list):
                raw_feats = torch.stack(raw_feats, dim=0)
            else:
                raw_feats = raw_feats.view(B, N, -1)
            return self.proj(raw_feats.float())
        return self.fallback(coord, color, normal)


class SpatialAffectEncoder(nn.Module):
    """
    Encodes sparse/continuous surface Gaze [B, N, 1] and/or 5-Emotion maps [B, N, 5]
    anchored to 3D coordinates [B, N, 3] into point-wise embeddings [B, N, latent_dim].
    Used both as the JEPA target teacher and as conditional input when 'gaze' or 'emotion'
    is passed in input_modalities.
    """

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        # Input channels: coord (3) + gaze (1) + emotion (5) = 9
        self.net = nn.Sequential(
            nn.Linear(9, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(
        self,
        coord: torch.Tensor,
        gaze: Optional[torch.Tensor] = None,
        emotion: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = coord.shape
        if gaze is None:
            gaze = torch.zeros((B, N, 1), device=coord.device, dtype=coord.dtype)
        if emotion is None:
            emotion = torch.zeros((B, N, 5), device=coord.device, dtype=coord.dtype)
        x = torch.cat([coord, gaze, emotion], dim=-1)
        return self.net(x)


class FrozenConceptEncoder(nn.Module):
    """
    Encodes Japanese/English sentences into continuous semantic concept vectors [B, S, concept_dim]
    (SONAR / Multilingual SentenceTransformer concept space) projected to latent_dim.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        concept_dim: int = 768,
        latent_dim: int = 256,
    ):
        super().__init__()
        self.concept_dim = concept_dim
        self.st_model = None
        try:
            from sentence_transformers import SentenceTransformer

            self.st_model = SentenceTransformer(model_name)
            self.st_model.eval()
            for p in self.st_model.parameters():
                p.requires_grad = False
            self.concept_dim = self.st_model.get_sentence_embedding_dimension()
        except Exception as e:
            print(f"[ConceptEncoder] SentenceTransformer unavailable ({e}), using hash-embedding fallback.")
            self.vocab_fallback = nn.Embedding(8192, concept_dim)
            nn.init.normal_(self.vocab_fallback.weight, std=0.02)
            self.vocab_fallback.weight.requires_grad = False

        self.to_latent = nn.Sequential(
            nn.Linear(self.concept_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, self.concept_dim),
            nn.LayerNorm(self.concept_dim),
        )

    @torch.no_grad()
    def encode_texts(self, texts: List[str], device: torch.device) -> torch.Tensor:
        if self.st_model is not None:
            emb = self.st_model.encode(texts, convert_to_tensor=True, device=str(device))
            return emb.clone().detach().float()
        # Deterministic character n-gram fallback for offline testing
        out = []
        for t in texts:
            ids = torch.tensor([ord(ch) % 8192 for ch in (t or " ")[:64]], device=device)
            out.append(self.vocab_fallback(ids).mean(dim=0))
        return torch.stack(out, dim=0)

    @torch.no_grad()
    def encode_concept_sequences(
        self, concepts_batch: List[List[str]], max_len: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            concept_embs: [B, max_len, concept_dim]
            concept_mask: [B, max_len] (1 for valid concept, 0 for pad)
        """
        B = len(concepts_batch)
        concept_embs = torch.zeros((B, max_len, self.concept_dim), device=device)
        concept_mask = torch.zeros((B, max_len), device=device)

        flat_texts = []
        mapping = []
        for b_idx, seq in enumerate(concepts_batch):
            for s_idx, sent in enumerate(seq[:max_len]):
                flat_texts.append(sent)
                mapping.append((b_idx, s_idx))

        if flat_texts:
            flat_embs = self.encode_texts(flat_texts, device)
            for idx, (b_idx, s_idx) in enumerate(mapping):
                concept_embs[b_idx, s_idx] = flat_embs[idx]
                concept_mask[b_idx, s_idx] = 1.0

        return concept_embs, concept_mask