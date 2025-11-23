import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import open3d as o3d
from dataset.utils import DEFAULT_QNA_ANSWER_COLOR_MAP, filter_data_on_condition

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
EMOTION_ORDER = ["面白い・気になる形だ", "美しい・芸術的だ", "不思議・意味不明", "不気味・不安・怖い", "何も感じない"]
RAW_DATA_DIR = r"D:\storage\jomon_kaen\jomon_kaen_dataset\japan"
MESH_DIR = r"D:\storage\jomon_kaen\pottery"
# TEST_GROUPS = ['G9']
TEST_GROUPS = ['NM0239(89)', 'UD0028(93)', 'NM0002(23)', 'NM0175(47)', 'NM0010(25)', 'NM0001(22)', 'SJ0504(55)', 'UD0308(80)']
BATCH_SIZE = 8
VOXEL_RESOLUTION = 80
MAX_EPOCHS = 1000
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
L1_WEIGHT = 0.001
NONZERO_EMO_TARGET = 0.005
NONZERO_GAZE_TARGET = 0.01
SAVE_DIR = r"D:\storage\jomon_kaen\dsvt_tidar"
EARLY_STOPPING_PATIENCE = 1000
MAX_COMMENT_LEN = 150
DRAFT_LEN = 8  # TiDAR block size
NUM_EMOTIONS = len(EMOTION_ORDER)


class DSVTLayer(nn.Module):
    def __init__(self, embed_dim, axis='x', tau=36):
        super().__init__()
        self.embed_dim = embed_dim
        self.axis = axis
        self.tau = tau
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, tokens, coords, batch_idx, D, H, W):
        N = tokens.shape[0]
        if N == 0: return tokens
        device = tokens.device
        if self.axis == 'x': sort_key = coords[:, 0]
        elif self.axis == 'y': sort_key = coords[:, 1]
        else: sort_key = coords[:, 2]
        composite = batch_idx * (D * H * W) + sort_key.long()
        _, indices = torch.sort(composite)
        tokens_sorted = tokens[indices]
        coords_sorted = coords[indices]
        batch_sorted = batch_idx[indices]
        S = (N + self.tau - 1) // self.tau
        padded_N = S * self.tau
        pad_len = padded_N - N
        if pad_len > 0:
            tokens_pad = torch.zeros(pad_len, self.embed_dim, device=device)
            tokens_sorted = torch.cat([tokens_sorted, tokens_pad], dim=0)
        tokens_sets = tokens_sorted.view(S, self.tau, self.embed_dim)
        tokens_norm = self.norm(tokens_sets)
        attn_out, _ = self.attn(tokens_norm, tokens_norm, tokens_norm)
        tokens_sets = tokens_sets + attn_out
        tokens_sets = tokens_sets + self.mlp(self.norm(tokens_sets))
        tokens_flat = tokens_sets.view(-1, self.embed_dim)[:N]
        unsort_indices = torch.argsort(indices)
        return tokens_flat[unsort_indices]

class DSVTBackbone(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64, resolution=80, tau=36, num_layers=6):
        super().__init__()
        self.resolution = resolution
        self.embed_dim = embed_dim
        self.tau = tau
        self.voxel_embed = nn.Linear(in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, resolution, resolution, resolution, embed_dim))
        self.encoder_layers = nn.ModuleList([
            DSVTLayer(embed_dim, axis=ax, tau=tau)
            for ax in ['x', 'y', 'z'] * (num_layers // 3 + 1)
        ][:num_layers])

    def forward(self, x):
        B, C, D, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(-1, C)
        mask = x_flat.abs().sum(dim=1) > 1e-6
        if mask.sum() == 0:
            return None, None, None, None, None, None, None
        coords = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        feats = x_flat[mask]
        batch_indices = coords // (D * H * W)
        rest = coords % (D * H * W)
        z = rest // (H * W)
        y = (rest % (H * W)) // W
        x_coord = rest % W
        tokens = self.voxel_embed(feats) + self.pos_embed[0, x_coord, y, z]
        pos = torch.stack([x_coord, y, z], dim=1).float()
        early_features = None
        for i, layer in enumerate(self.encoder_layers):
            tokens = layer(tokens, pos, batch_indices, D, H, W)
            if i == 2:
                early_features = tokens.clone()
        return tokens, coords, batch_indices, (x_coord, y, z), (D, H, W), B, early_features

class DSVTDecoder(nn.Module):
    def __init__(self, embed_dim=64, out_channels=6, resolution=80, tau=36, num_layers=6):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, out_channels))
        self.decoder_layers = nn.ModuleList([
            DSVTLayer(embed_dim, axis=ax, tau=tau)
            for ax in ['z', 'y', 'x'] * (num_layers // 3 + 1)
        ][:num_layers])

    def forward(self, tokens, coords, batch_indices, xyz, dims, B, early_features=None):
        x_coord, y, z = xyz
        D, H, W = dims
        pos = torch.stack([x_coord, y, z], dim=1).float()
        tokens_dec = tokens
        for i, layer in enumerate(self.decoder_layers):
            if i >= 3 and early_features is not None:
                tokens_dec = tokens_dec + early_features
            tokens_dec = layer(tokens_dec, pos, batch_indices, D, H, W)
        logits_sparse = self.head(tokens_dec)
        logits_dense = torch.zeros(B * D * H * W, self.out_channels, device=logits_sparse.device, dtype=logits_sparse.dtype)
        logits_dense[coords] = logits_sparse
        logits_dense = logits_dense.view(B, D, H, W, self.out_channels).permute(0, 4, 1, 2, 3)
        return logits_dense, tokens_dec


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0): super().__init__(); self.alpha, self.gamma = alpha, gamma
    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_weight = (1 - pt)**self.gamma
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return (focal_weight * alpha_t * bce).mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0): super().__init__(); self.smooth = smooth
    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        inputs_flat = probs.view(probs.size(0), probs.size(1), -1)
        targets_flat = targets.view(targets.size(0), targets.size(1), -1)
        inter = (inputs_flat * targets_flat).sum(dim=2)
        union = inputs_flat.sum(dim=2) + targets_flat.sum(dim=2)
        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

class NonZeroRegularization(nn.Module):
    def __init__(self, weight=1.0, target_sparsity=0.05): super().__init__(); self.weight, self.target_sparsity = weight, target_sparsity
    def forward(self, preds):
        mean_act = torch.sigmoid(preds).reshape(preds.size(0), -1).mean(dim=1)
        target = torch.full_like(mean_act, self.target_sparsity)
        return self.weight * F.mse_loss(mean_act, target)


class TiDARCaptionHead(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_layers=4, num_heads=4, max_len=150, draft_len=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.draft_len = draft_len
        self.embed_dim = embed_dim

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len + draft_len, embed_dim))
        self.visual_proj = nn.Linear(32, embed_dim)  # from DSVT global feat [32]

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            batch_first=True,
            dropout=0.1,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        self._register_masks()

    def _register_masks(self):
        L = self.max_len
        k = self.draft_len

        # Training mask: [L + k, L + k]
        mask = torch.zeros(L + k, L + k)
        # AR prefix (first L-k): causal
        mask[:L - k, :L - k] = torch.triu(torch.ones(L - k, L - k), diagonal=1)
        # AR prefix cannot see draft block
        mask[:L - k, L - k:] = 1
        # Draft block (last k): bidirectional + can see prefix
        mask[L - k:, L - k:] = 0
        mask[L - k:, :L - k] = 0
        self.register_buffer("train_mask", mask.bool())

        # Decoding base mask (for inference)
        total_len = L + k
        dec_mask = torch.triu(torch.ones(total_len, total_len), diagonal=1)
        self.register_buffer("dec_mask_base", dec_mask.bool())

    def forward(self, visual_feat, input_ids=None, labels=None, mode="train"):
        B = visual_feat.shape[0]
        device = visual_feat.device

        if mode == "train":
            assert input_ids is not None
            L = input_ids.shape[1]
            assert L == self.max_len
            tokens = self.token_embed(input_ids) + self.pos_embed[:, :L]
            visual = self.visual_proj(visual_feat).unsqueeze(1)  # [B, 1, D]
            x = torch.cat([visual, tokens], dim=1)  # [B, 1+L, D]

            full_mask = torch.zeros(1 + L, 1 + L, dtype=torch.bool, device=device)
            full_mask[1:, 1:] = self.train_mask[:L, :L]
            full_mask[1:, 0] = False  # tokens can see visual
            full_mask[0, :] = True    # visual sees no one

            output = self.decoder(x, x, tgt_mask=full_mask)
            logits = self.lm_head(self.norm(output[:, 1:]))
            return logits

        elif mode == "generate":
            return self._generate(visual_feat)

    @torch.no_grad()
    def _generate(self, visual_feat):
        B, _ = visual_feat.shape
        assert B == 1
        device = visual_feat.device
        visual = self.visual_proj(visual_feat)  # [1, D]

        input_ids = torch.full((1, self.max_len), self.vocab_size - 1, dtype=torch.long, device=device)
        input_ids[0, 0] = 1  # <sos>
        accepted_len = 1

        for step in range(20):
            if accepted_len >= self.max_len - 1: break
            end_draft = min(accepted_len + self.draft_len, self.max_len)
            draft_pos = torch.arange(accepted_len, end_draft, device=device)
            masked_ids = input_ids.clone()
            masked_ids[0, draft_pos] = self.vocab_size - 1  # <mask>

            tokens = self.token_embed(masked_ids) + self.pos_embed[:, :self.max_len]
            x = torch.cat([visual.unsqueeze(1), tokens], dim=1)  # [1, 1+L, D]

            L = self.max_len
            mask = self.dec_mask_base[:1 + L, :1 + L].clone()
            prefix_end = 1 + accepted_len
            draft_start = 1 + accepted_len
            draft_end = 1 + end_draft
            mask[draft_start:draft_end, :prefix_end] = False
            mask[draft_start:draft_end, draft_start:draft_end] = False
            mask[:prefix_end, draft_start:draft_end] = True

            output = self.decoder(x, x, tgt_mask=mask)
            logits = self.lm_head(self.norm(output[0, 1:]))
            draft_tokens = logits[accepted_len:end_draft].argmax(dim=-1)
            for i, tok in enumerate(draft_tokens):
                pos = accepted_len + i
                if pos >= self.max_len - 1: break
                input_ids[0, pos] = tok.item()
            accepted_len = end_draft
            if 2 in input_ids[0, :accepted_len]: break

        return input_ids[0]


from model.helper import SimpleTokenizer, voxel_grid_to_point_cloud

class DSVTWithTiDAR(pl.LightningModule):
    def __init__(self, vocab_size, tokenizer, max_comment_len=150, draft_len=8):
        super().__init__()
        self.save_hyperparameters(ignore=['tokenizer'])
        self.tokenizer = tokenizer
        self.draft_len = draft_len
        self.vocab_size = vocab_size

        self.backbone = DSVTBackbone(in_channels=3, embed_dim=32, resolution=VOXEL_RESOLUTION, tau=64, num_layers=6)
        self.decoder = DSVTDecoder(embed_dim=32, out_channels=6, resolution=VOXEL_RESOLUTION, tau=64, num_layers=6)

        self.global_attn_pool = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 1, 32))

        self.caption_head = TiDARCaptionHead(
            vocab_size=vocab_size,
            embed_dim=32,
            num_layers=3,
            num_heads=4,
            max_len=max_comment_len,
            draft_len=draft_len
        )

        self.focal_loss = FocalLoss(alpha=0.2, gamma=2.0)
        self.dice_loss = DiceLoss()
        self.nonzero_emo = NonZeroRegularization(weight=0.1, target_sparsity=NONZERO_EMO_TARGET)
        self.nonzero_gaze = NonZeroRegularization(weight=0.05, target_sparsity=NONZERO_GAZE_TARGET)

    def forward(self, x, input_ids=None, labels=None):
        B = x.shape[0]
        enc_out = self.backbone(x)
        if enc_out[0] is None:
            emo = torch.zeros(B, 5, VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION, device=x.device)
            gaze = torch.zeros(B, 1, VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION, device=x.device)
            caption_loss = torch.tensor(0.0, device=x.device)
            return emo, gaze, caption_loss

        tokens_enc, coords, batch_idx, xyz, dims, _, early_features = enc_out
        dec_logits, tokens_dec = self.decoder(tokens_enc, coords, batch_idx, xyz, dims, B, early_features)
        emo = dec_logits[:, :5]
        gaze = dec_logits[:, 5:6]

        global_feats = []
        for b in range(B):
            mask_b = (batch_idx == b)
            feat_b = torch.zeros(32, device=x.device) if not mask_b.any() else self._get_global_feat(tokens_enc[mask_b], tokens_dec[mask_b])
            global_feats.append(feat_b)
        global_feats = torch.stack(global_feats)

        caption_loss = torch.tensor(0.0, device=x.device)
        if input_ids is not None:
            # Prepare full-masked input for diffusion block
            masked_input = input_ids.clone()
            masked_input[:, -self.draft_len:] = self.vocab_size - 1  # <mask>
            logits = self.caption_head(global_feats, masked_input, mode="train")
            caption_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=0)
        return emo, gaze, caption_loss

    def _get_global_feat(self, enc_t, dec_t):
        fused = (enc_t + dec_t).unsqueeze(0)
        attn_out, _ = self.global_attn_pool(self.pool_query, fused, fused)
        return attn_out.squeeze(1).squeeze(0)

    def _shared_step(self, batch, stage):
        inputs, (emo_labels, gaze_labels, tokens) = batch
        labels = tokens.clone()
        emo_labels = emo_labels.float()
        gaze_labels = gaze_labels.float()

        emo_pred, gaze_pred, mlm_loss = self(inputs, input_ids=tokens, labels=labels)

        loss_emo_focal = self.focal_loss(emo_pred, emo_labels)
        loss_emo_dice = self.dice_loss(emo_pred, emo_labels)
        loss_emo_l1 = F.l1_loss(torch.sigmoid(emo_pred), emo_labels)
        loss_emo_reg = self.nonzero_emo(emo_pred)

        loss_gaze_focal = self.focal_loss(gaze_pred, gaze_labels)
        loss_gaze_dice = self.dice_loss(gaze_pred, gaze_labels)
        loss_gaze_l1 = F.l1_loss(torch.sigmoid(gaze_pred), gaze_labels)
        loss_gaze_reg = self.nonzero_gaze(gaze_pred)

        total = (0.5 * loss_emo_focal + 1.0 * loss_emo_dice +
                 L1_WEIGHT * loss_emo_l1 + loss_emo_reg +
                 1.0 * loss_gaze_focal + 0.5 * loss_gaze_dice +
                 L1_WEIGHT * loss_gaze_l1 + loss_gaze_reg + mlm_loss)

        self.log_dict({
            f"{stage}_loss": total,
            f"{stage}_emo_focal": loss_emo_focal,
            f"{stage}_gaze_focal": loss_gaze_focal,
            f"{stage}_mlm": mlm_loss,
        }, on_epoch=True, prog_bar=(stage == 'val'))
        return total

    def training_step(self, batch, batch_idx): return self._shared_step(batch, 'train')
    def validation_step(self, batch, batch_idx): return self._shared_step(batch, 'val')

    @torch.no_grad()
    def generate_caption_from_voxel(self, voxel_input, tokenizer, max_len=None):
        if max_len is None: max_len = self.hparams.max_comment_len
        enc_out = self.backbone(voxel_input)
        if enc_out[0] is None:
            return "", np.full(max_len, tokenizer.word_to_idx.get('<mask>', tokenizer.vocab_size - 1))
        tokens_enc, coords, batch_idx, xyz, dims, _, early_features = enc_out
        _, tokens_dec = self.decoder(tokens_enc, coords, batch_idx, xyz, dims, 1, early_features)
        global_feat = self._get_global_feat(tokens_enc, tokens_dec).unsqueeze(0)
        final_ids = self.caption_head._generate(global_feat)
        text = tokenizer.decode(final_ids.cpu().numpy())
        return text, final_ids.cpu().numpy()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, (emo_labels, gaze_labels, tokens) = batch
        B = inputs.shape[0]
        enc_out = self.backbone(inputs)
        if enc_out[0] is None:
            emo = torch.zeros(B, 5, VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION, device=inputs.device)
            gaze = torch.zeros(B, 1, VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION, device=inputs.device)
            captions = [""] * B
        else:
            tokens_enc, coords, batch_idx_coords, xyz, dims, _, early_features = enc_out
            dec_logits, tokens_dec = self.decoder(tokens_enc, coords, batch_idx_coords, xyz, dims, B, early_features)
            emo = dec_logits[:, :5]
            gaze = dec_logits[:, 5:6]
            captions = []
            for b in range(B):
                text, _ = self.generate_caption_from_voxel(inputs[b:b+1], self.tokenizer)
                captions.append(text)
        return emo, gaze, captions

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=LEARNING_RATE, weight_decay=0.01)


class SavePredictionCallback(pl.Callback):
    def __init__(self, save_dir="", save_every_n_epochs=10, max_samples_to_save=20, emotion_order=EMOTION_ORDER):
        super().__init__()
        self.save_dir = save_dir
        self.save_every_n_epochs = save_every_n_epochs
        self.max_samples_to_save = max_samples_to_save
        self.emotion_order = emotion_order
        os.makedirs(self.save_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.save_every_n_epochs != 0 or not trainer.is_global_zero: return
        print(f"\nSaving predictions for epoch {epoch + 1}...")
        epoch_dir = os.path.join(self.save_dir, f"epoch_{epoch + 1}")
        os.makedirs(epoch_dir, exist_ok=True)
        val_loader = trainer.datamodule.val_dataloader()
        val_data_paths = getattr(trainer.datamodule, 'val_data_paths', [])
        if not val_data_paths:
            print("Error: Could not retrieve validation data paths.")
            return
        pl_module.eval()
        sample_count = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if sample_count >= self.max_samples_to_save: break
                inputs, (emotion_labels, heatmaps, tokens) = batch
                inputs = inputs.to(pl_module.device)
                emotion_labels = emotion_labels.to(pl_module.device)
                heatmaps = heatmaps.to(pl_module.device)
                tokens = tokens.to(pl_module.device)

                enc_out = pl_module.backbone(inputs)
                if enc_out[0] is None:
                    emo_preds = torch.zeros_like(emotion_labels)
                    gaze_preds = torch.zeros_like(heatmaps)
                else:
                    tokens_enc, coords, batch_idx_coords, xyz, dims, _, early_features = enc_out
                    dec_logits, _ = pl_module.decoder(tokens_enc, coords, batch_idx_coords, xyz, dims, inputs.shape[0], early_features)
                    emo_preds = dec_logits[:, :5]
                    gaze_preds = dec_logits[:, 5:6]

                batch_size = inputs.size(0)
                for i in range(batch_size):
                    if sample_count >= self.max_samples_to_save: break
                    dataset_idx = batch_idx * trainer.datamodule.hparams.batch_size + i
                    if dataset_idx >= len(val_data_paths): break

                    pottery_id = val_data_paths[dataset_idx]['ID']
                    pottery_path = val_data_paths[dataset_idx]['processed_pottery_path']
                    sample_name = f"{sample_count}_{pottery_id}"

                    self.save_input_pottery(inputs[i], epoch_dir, sample_name, pottery_path)
                    self.save_emotions(emo_preds[i], emotion_labels[i], epoch_dir, sample_name, pottery_path, input_mask=inputs[i])
                    self.save_heatmap(gaze_preds[i], heatmaps[i], epoch_dir, sample_name, pottery_path, input_mask=inputs[i])

                    caption_text, final_ids = pl_module.generate_caption_from_voxel(
                        inputs[i:i+1], pl_module.tokenizer, max_len=MAX_COMMENT_LEN
                    )
                    self.save_caption(caption_text, final_ids, tokens[i].cpu().numpy(), pl_module.tokenizer, epoch_dir, sample_name)

                    sample_count += 1
        pl_module.train()
        print(f"Saved {sample_count} samples to {epoch_dir}")

    def save_input_pottery(self,
                           input_tensor,
                           epoch_dir,
                           sample_idx,
                           reference_path=None):
        if input_tensor.is_cuda:
            input_tensor = input_tensor.cpu()
        voxel_grid = input_tensor.permute(1, 2, 3, 0).numpy()
        points_indices = np.argwhere(voxel_grid.sum(axis=-1) > 0.01)
        if points_indices.shape[0] == 0:
            pcd = o3d.geometry.PointCloud()
        else:
            colors = voxel_grid[points_indices[:,
                                               0],
                                points_indices[:,
                                               1],
                                points_indices[:,
                                               2]]
            resolution = voxel_grid.shape[0]
            if reference_path and os.path.exists(reference_path):
                try:
                    ref_pcd = o3d.io.read_point_cloud(reference_path)
                    if ref_pcd.has_points():
                        min_bound = ref_pcd.get_min_bound()
                        max_bound = ref_pcd.get_max_bound()
                        scale = np.max(max_bound - min_bound)
                        normalized_points = points_indices / (resolution - 1)
                        points = normalized_points * scale + min_bound
                    else:
                        points = points_indices / (resolution - 1.0) - 0.5
                except Exception:
                    points = points_indices / (resolution - 1.0) - 0.5
            else:
                points = points_indices / (resolution - 1.0) - 0.5
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(
            os.path.join(epoch_dir,
                         f"sample_{sample_idx}_input.ply"),
            pcd)

    def save_emotions(self,
                      preds_tensor,
                      labels_tensor,
                      epoch_dir,
                      sample_idx,
                      reference_path=None,
                      input_mask=None):
        preds_prob = torch.sigmoid(preds_tensor)
        for i, emotion_name in enumerate(self.emotion_order):
            safe_emotion_name = f"emotion_{DEFAULT_QNA_ANSWER_COLOR_MAP[emotion_name]['name']}"
            target_color_rgb = DEFAULT_QNA_ANSWER_COLOR_MAP[emotion_name][
                'rgb']
            # Always save full masked volume (even if all near-zero)
            pred_pcd = voxel_grid_to_point_cloud(
                preds_prob[i],
                intensity_threshold=-1.0,  # <-- disable thresholding
                reference_pcd_path=reference_path,
                mask_tensor=input_mask,
                fixed_color_rgb=target_color_rgb,
            )
            o3d.io.write_point_cloud(
                os.path.join(
                    epoch_dir,
                    f"sample_{sample_idx}_{safe_emotion_name}_PRED.ply"),
                pred_pcd)
            gt_pcd = voxel_grid_to_point_cloud(
                labels_tensor[i].float(),
                intensity_threshold=-1.0,  # <-- disable thresholding
                reference_pcd_path=reference_path,
                mask_tensor=input_mask,
                fixed_color_rgb=target_color_rgb
            )
            o3d.io.write_point_cloud(
                os.path.join(
                    epoch_dir,
                    f"sample_{sample_idx}_{safe_emotion_name}_GT.ply"),
                gt_pcd)

    def save_heatmap(self,
                     heatmap_pred_tensor,
                     heatmap_gt_tensor,
                     epoch_dir,
                     sample_idx,
                     reference_path=None,
                     input_mask=None):
        # Save full masked heatmap (including near-zero values)
        pred_pcd = voxel_grid_to_point_cloud(
            torch.sigmoid(heatmap_pred_tensor),
            intensity_threshold=-1.0,
            reference_pcd_path=reference_path,
            mask_tensor=input_mask,
            colormap_name='jet')
        o3d.io.write_point_cloud(
            os.path.join(epoch_dir,
                         f"sample_{sample_idx}_heatmap_PRED.ply"),
            pred_pcd)
        gt_pcd = voxel_grid_to_point_cloud(
            heatmap_gt_tensor.float(),
            intensity_threshold=-1.0,  # <-- disable thresholding
            reference_pcd_path=reference_path,
            mask_tensor=input_mask,
            colormap_name='jet'
        )
        o3d.io.write_point_cloud(
            os.path.join(epoch_dir,
                         f"sample_{sample_idx}_heatmap_GT.ply"),
            gt_pcd)

    def save_caption(self, pred_caption, pred_ids, gt_tokens, tokenizer, epoch_dir, sample_idx):
        # Decode ground truth
        gt_words = []
        for idx in gt_tokens:
            if idx == 2:  # <eos>
                break
            if idx not in [0, 1]:  # skip <pad>, <sos>
                word = tokenizer.idx_to_word.get(int(idx), f"[UNK:{idx}]")
                gt_words.append(word)
        gt_text = "".join(gt_words)  # Japanese: no spaces

        with open(os.path.join(epoch_dir, f"sample_{sample_idx}_caption.txt"), "w", encoding="utf-8") as f:
            f.write(f"Ground Truth: {gt_text}\n")
            f.write(f"Prediction:   {pred_caption}\n")

from dataset.voxel_dataset import ExtendedVoxelDataset
from model.helper import SimpleTokenizer


class DSVTDataModule(pl.LightningDataModule):

    def __init__(self,
                 all_data_paths,
                 batch_size,
                 num_workers,
                 voxel_resolution,
                 test_groups):
        super().__init__()
        self.save_hyperparameters(ignore=['all_data_paths'])
        self.all_data_paths = all_data_paths
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.voxel_resolution = voxel_resolution
        self.test_groups = test_groups
        self.tokenizer = SimpleTokenizer(
            max_len=MAX_COMMENT_LEN)  # Needed for dataset, but unused

    def setup(self, stage=None):
        np.random.shuffle(self.all_data_paths)
        # train_paths = [
        #     p for p in self.all_data_paths
        #     if p['GROUP'] not in self.test_groups
        # ]
        # val_paths = [
        #     p for p in self.all_data_paths if p['GROUP'] in self.test_groups
        # ]
        train_paths = [p for p in self.all_data_paths if p['ID'] not in self.test_groups]
        val_paths = [p for p in self.all_data_paths if p['ID'] in self.test_groups]
        self.val_data_paths = val_paths
        # Build vocab (required by dataset, even if unused)
        comments = []
        for p in self.all_data_paths:
            if os.path.exists(p.get('TRANSCRIPT', '')):
                with open(p['TRANSCRIPT'], 'r', encoding='utf-8') as f:
                    comments.append(f.read().strip())
        self.tokenizer.build_vocab(comments)
        common_args = {
            'voxel_resolution': self.voxel_resolution,
            'tokenizer': self.tokenizer
        }
        self.train_dataset = ExtendedVoxelDataset(train_paths,
                                                  augment_color_p=0.5,
                                                  **common_args)
        self.val_dataset = ExtendedVoxelDataset(val_paths,
                                                augment_color_p=0.0,
                                                **common_args)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          persistent_workers=True)

if __name__ == "__main__":
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('high')

    all_data_paths, _ = filter_data_on_condition(
        root=RAW_DATA_DIR,
        pottery_path=MESH_DIR,
        preprocess=True,
        use_cache=True,
        mode=0,
        target_voxel_resolution=VOXEL_RESOLUTION,
        min_emotion_count=1,
        min_qa_size=1,
        limit=1000,
    )

    datamodule = DSVTDataModule(
        all_data_paths=all_data_paths,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        voxel_resolution=VOXEL_RESOLUTION,
        test_groups=TEST_GROUPS
    )
    datamodule.setup()

    model = DSVTWithTiDAR(
        vocab_size=datamodule.tokenizer.vocab_size,
        tokenizer=datamodule.tokenizer,
        max_comment_len=MAX_COMMENT_LEN,
        draft_len=DRAFT_LEN
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=10, every_n_epochs=20, save_last=True, mode='min'),
            pl.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, mode='min'),
            SavePredictionCallback(save_dir=SAVE_DIR, save_every_n_epochs=25, max_samples_to_save=200, emotion_order=EMOTION_ORDER)
        ],
        log_every_n_steps=10,
        precision='16-mixed' if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, datamodule=datamodule)

    print("\n=== Inference Example ===")
    model.eval()
    sample_input, _ = next(iter(datamodule.val_dataloader()))
    sample_input = sample_input[:1].to(model.device)
    with torch.no_grad():
        caption, _ = model.generate_caption_from_voxel(sample_input, datamodule.tokenizer)
    print("Generated caption:", caption)
