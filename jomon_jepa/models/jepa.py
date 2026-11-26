from typing import Dict, Optional
import torch
import torch.nn as nn


class Multimodal3DJEPAPredictor(nn.Module):
    """
    Pruned Cross-Modal 3D-JEPA Predictor:
    Only allocates modality embeddings and target query parameters for the active
    input and output modalities of this specialized model.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_queries: int = 32,
        num_layers: int = 4,
        num_heads: int = 8,
        has_spatial_input: bool = False,
        has_text_input: bool = False,
        predict_spatial: bool = True,
        predict_concept: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_queries = num_queries
        self.predict_spatial = predict_spatial
        self.predict_concept = predict_concept

        # Only create modality embeddings for active inputs
        self.mod_embed_object = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        if has_spatial_input:
            self.mod_embed_spatial = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        if has_text_input:
            self.mod_embed_text = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)

        # Only create target queries for active outputs
        if predict_spatial:
            self.spatial_queries = nn.Parameter(torch.randn(1, num_queries, latent_dim) * 0.02)
        if predict_concept:
            self.concept_queries = nn.Parameter(torch.randn(1, num_queries, latent_dim) * 0.02)

        # Variational latent head for temperature-controlled stochastic sampling
        self.posterior_mu = nn.Linear(latent_dim, latent_dim)
        self.posterior_logvar = nn.Linear(latent_dim, latent_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.predictor = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(latent_dim)

    def pool_point_patches(self, point_feats: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        B, N, D = point_feats.shape
        sort_key = coord[:, :, 0] * 100.0 + coord[:, :, 1] * 10.0 + coord[:, :, 2]
        idx = torch.argsort(sort_key, dim=1)
        sorted_feats = torch.gather(point_feats, 1, idx.unsqueeze(-1).expand(-1, -1, D))
        K = min(64, N)
        return sorted_feats.view(B, K, N // K, D).mean(dim=2)

    def sample_latent_perturbation(
        self, queries: torch.Tensor, temperature: float = 0.7, noise_scale: float = 0.15, stochastic: bool = True
    ) -> torch.Tensor:
        mu = self.posterior_mu(queries)
        if not stochastic or temperature <= 1e-4:
            return mu
        logvar = torch.clamp(self.posterior_logvar(queries), min=-6.0, max=2.0)
        std = torch.exp(0.5 * logvar) * (temperature * noise_scale)
        return mu + torch.randn_like(std) * std

    def forward(
        self,
        obj_point_feats: torch.Tensor,
        coord: torch.Tensor,
        cond_spatial_feats: Optional[torch.Tensor] = None,
        cond_text_latent: Optional[torch.Tensor] = None,
        temperature: float = 0.7,
        noise_scale: float = 0.15,
        stochastic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B = obj_point_feats.shape[0]

        context_tokens = [self.pool_point_patches(obj_point_feats, coord) + self.mod_embed_object]

        if cond_spatial_feats is not None and hasattr(self, "mod_embed_spatial"):
            context_tokens.append(self.pool_point_patches(cond_spatial_feats, coord) + self.mod_embed_spatial)

        if cond_text_latent is not None and hasattr(self, "mod_embed_text"):
            if cond_text_latent.ndim == 2:
                cond_text_latent = cond_text_latent.unsqueeze(1)
            context_tokens.append(cond_text_latent + self.mod_embed_text)

        memory = torch.cat(context_tokens, dim=1)

        query_list = []
        if self.predict_spatial:
            query_list.append(self.spatial_queries.expand(B, -1, -1))
        if self.predict_concept:
            query_list.append(self.concept_queries.expand(B, -1, -1))

        queries = torch.cat(query_list, dim=1)
        pred_latents = self.out_norm(self.predictor(tgt=queries, memory=memory))
        pred_latents = self.sample_latent_perturbation(
            pred_latents, temperature=temperature, noise_scale=noise_scale, stochastic=stochastic
        )

        out = {}
        offset = 0
        if self.predict_spatial:
            sp_tokens = pred_latents[:, offset : offset + self.num_queries]
            out["pred_spatial_tokens"] = sp_tokens
            out["pred_global_spatial"] = sp_tokens.mean(dim=1)
            offset += self.num_queries

        if self.predict_concept:
            cp_tokens = pred_latents[:, offset : offset + self.num_queries]
            out["pred_concept_tokens"] = cp_tokens
            out["pred_global_concept"] = cp_tokens.mean(dim=1)

        return out