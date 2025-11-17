import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np

# Config
EMOTION_ORDER = ["面白い・気になる形だ", "美しい・芸術的だ", "不思議・意味不明", "不気味・不安・怖い", "何も感じない"]
RAW_DATA_DIR = "./src/jomon_kaen_dataset/japan"
MESH_DIR = "./src/pottery"
TEST_GROUPS = ['G9']
BATCH_SIZE = 8
VOXEL_RESOLUTION = 80
MAX_EPOCHS = 1000
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
L1_WEIGHT = 0.001
NONZERO_EMO_TARGET = 0.005
NONZERO_GAZE_TARGET = 0.01
SAVE_DIR = "dsvt_full"
EARLY_STOPPING_PATIENCE = 1000
NUM_EMOTIONS = len(EMOTION_ORDER)
MAX_COMMENT_LEN = 150
VOCAB_SIZE = 3000  # Will be set dynamically


# Stochastic Depth
class StochasticDepth(nn.Module):

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape,
                                               dtype=x.dtype,
                                               device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# # DSVT Layer with internal 3D CNN after grouping
# class DSVTLayerWCNN(nn.Module):

#     def __init__(self, embed_dim, axis='x', tau=36):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.axis = axis
#         self.tau = tau
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.attn = nn.MultiheadAttention(embed_dim,
#                                           num_heads=4,
#                                           batch_first=True)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.mlp = nn.Sequential(nn.Linear(embed_dim,
#                                            embed_dim * 2),
#                                  nn.GELU(),
#                                  nn.Linear(embed_dim * 2,
#                                            embed_dim))

#         if tau == 216:
#             self.local_shape = (6, 6, 6)
#         elif tau == 125:
#             self.local_shape = (5, 5, 5)
#         elif tau == 36:
#             self.local_shape = (3, 3, 4)
#         elif tau == 16:
#             self.local_shape = (2, 2, 4)
#         else:
#             self.local_shape = (1, 1, tau)
#         d_l, h_l, w_l = self.local_shape
#         assert d_l * h_l * w_l == tau, f"Local shape {self.local_shape} must multiply to tau={tau}"

#         self.local_cnn = nn.Sequential(
#             nn.Conv3d(embed_dim,
#                       embed_dim,
#                       kernel_size=3,
#                       padding=1,
#                       groups=embed_dim,
#                       bias=False),
#             nn.BatchNorm3d(embed_dim),
#             nn.Conv3d(embed_dim,
#                       embed_dim,
#                       kernel_size=1,
#                       bias=False),
#             nn.GELU())

#     def forward(self, tokens, coords, batch_idx, D, H, W):
#         N = tokens.shape[0]
#         if N == 0:
#             return tokens
#         device = tokens.device

#         if self.axis == 'x':
#             sort_key = coords[:, 0]
#         elif self.axis == 'y':
#             sort_key = coords[:, 1]
#         else:
#             sort_key = coords[:, 2]

#         composite = batch_idx * (D * H * W) + sort_key.long()
#         _, indices = torch.sort(composite)
#         tokens_sorted = tokens[indices]
#         coords_sorted = coords[indices]
#         batch_sorted = batch_idx[indices]

#         S = (N + self.tau - 1) // self.tau
#         padded_N = S * self.tau
#         pad_len = padded_N - N
#         if pad_len > 0:
#             tokens_pad = torch.zeros(pad_len, self.embed_dim, device=device)
#             tokens_sorted = torch.cat([tokens_sorted, tokens_pad], dim=0)

#         tokens_sets = tokens_sorted.view(S, self.tau, self.embed_dim)
#         d_l, h_l, w_l = self.local_shape
#         tokens_3d = tokens_sets.transpose(1,
#                                           2).reshape(S,
#                                                      self.embed_dim,
#                                                      d_l,
#                                                      h_l,
#                                                      w_l)
#         tokens_cnn = self.local_cnn(tokens_3d)
#         tokens_cnn = tokens_cnn.reshape(S,
#                                         self.embed_dim,
#                                         self.tau).transpose(1,
#                                                             2)
#         tokens_sets = tokens_sets + tokens_cnn

#         tokens_norm = self.norm1(tokens_sets)
#         attn_out, _ = self.attn(tokens_norm, tokens_norm, tokens_norm)
#         tokens_sets = tokens_sets + attn_out
#         tokens_sets = tokens_sets + self.mlp(self.norm2(tokens_sets))

#         tokens_flat = tokens_sets.view(-1, self.embed_dim)[:N]
#         unsort_indices = torch.argsort(indices)
#         return tokens_flat[unsort_indices]


class DSVTLayer(nn.Module):

    def __init__(self, embed_dim, axis='x', tau=36):
        super().__init__()
        self.embed_dim = embed_dim
        self.axis = axis
        self.tau = tau
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim,
                                          num_heads=4,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim,
                                           embed_dim * 2),
                                 nn.GELU(),
                                 nn.Linear(embed_dim * 2,
                                           embed_dim))

    def forward(self, tokens, coords, batch_idx, D, H, W):
        N = tokens.shape[0]
        if N == 0:
            return tokens
        device = tokens.device

        if self.axis == 'x':
            sort_key = coords[:, 0]
        elif self.axis == 'y':
            sort_key = coords[:, 1]
        else:
            sort_key = coords[:, 2]

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
        tokens_norm = self.norm1(tokens_sets)
        attn_out, _ = self.attn(tokens_norm, tokens_norm, tokens_norm)
        tokens_sets = tokens_sets + attn_out
        tokens_sets = tokens_sets + self.mlp(self.norm2(tokens_sets))

        tokens_flat = tokens_sets.view(-1, self.embed_dim)[:N]
        unsort_indices = torch.argsort(indices)
        return tokens_flat[unsort_indices]


class DSVTBackbone(nn.Module):

    def __init__(self,
                 in_channels=3,
                 embed_dim=64,
                 resolution=80,
                 tau=36,
                 num_layers=6):
        super().__init__()
        self.resolution = resolution
        self.embed_dim = embed_dim
        self.tau = tau
        self.voxel_embed = nn.Linear(in_channels, embed_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1,
                        resolution,
                        resolution,
                        resolution,
                        embed_dim))
        self.encoder_layers = nn.ModuleList(
            # [
        #     DSVTLayerWCNN(embed_dim,
        #                   axis=ax,
        #                   tau=tau) for ax in ['x', 'y', 'z']
        # ] + 
        [
            DSVTLayer(embed_dim,
                      axis=ax,
                      tau=tau)
            for ax in ['x', 'y', 'z'] * (num_layers // 3 + 1)
        ][:num_layers])

    def forward(self, x):
        B, C, D, H, W = x.shape
        device = x.device
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(-1, C)
        mask = x_flat.abs().sum(dim=1) > 1e-6
        coords = torch.nonzero(mask, as_tuple=False).squeeze(-1)

        if coords.numel() == 0:
            return None, None, None, None, None, None, None

        feats = x_flat[mask]
        batch_indices = coords // (D * H * W)
        rest = coords % (D * H * W)
        z = rest // (H * W)
        y = (rest % (H * W)) // W
        x_coord = rest % W

        tokens = self.voxel_embed(feats)
        pos_enc = self.pos_embed[0, x_coord, y, z]
        tokens = tokens + pos_enc
        pos = torch.stack([x_coord, y, z], dim=1).float()

        early_features = None
        for i, layer in enumerate(self.encoder_layers):
            tokens = layer(tokens, pos, batch_indices, D, H, W)
            if i == 2:
                early_features = tokens.clone()

        return tokens, coords, batch_indices, (x_coord, y, z), (D, H, W), B, early_features


class DSVTDecoder(nn.Module):

    def __init__(self,
                 embed_dim=64,
                 out_channels=6,
                 resolution=80,
                 tau=36,
                 num_layers=6):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.head = nn.Sequential(nn.LayerNorm(embed_dim),
                                  nn.Linear(embed_dim,
                                            out_channels))
        self.decoder_layers = nn.ModuleList([
            DSVTLayer(embed_dim,
                      axis=ax,
                      tau=tau)
            for ax in ['z', 'y', 'x'] * (num_layers // 3 + 1)
        ][:num_layers])

    def forward(self,
                tokens,
                coords,
                batch_indices,
                xyz,
                dims,
                B,
                early_features=None):
        x_coord, y, z = xyz
        D, H, W = dims
        pos = torch.stack([x_coord, y, z], dim=1).float()

        tokens_dec = tokens
        for i, layer in enumerate(self.decoder_layers):
            if i >= 3 and early_features is not None:
                tokens_dec = tokens_dec + early_features
            tokens_dec = layer(tokens_dec, pos, batch_indices, D, H, W)

        # Return tokens_dec BEFORE the head for captioning
        logits_sparse = self.head(tokens_dec)
        logits_dense = torch.zeros(B * D * H * W,
                                   self.out_channels,
                                   dtype=logits_sparse.dtype,
                                   device=logits_sparse.device)
        logits_dense[coords] = logits_sparse
        logits_dense = logits_dense.view(B,
                                         D,
                                         H,
                                         W,
                                         self.out_channels).permute(
                                             0,
                                             4,
                                             1,
                                             2,
                                             3)
        return logits_dense, tokens_dec


# GPT Caption Decoder
class GPTCaptioner(nn.Module):

    def __init__(self,
                 embed_dim,
                 vocab_size,
                 max_len,
                 num_layers=3,
                 nhead=4,
                 dropout=0.4,
                 layer_drop=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.token_embedding = nn.Embedding(vocab_size,
                                            embed_dim,
                                            padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len + 1, embed_dim)
        self.context_proj = nn.Linear(embed_dim, embed_dim)

        layers = []
        for _ in range(num_layers):
            layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                               nhead=nhead,
                                               dim_feedforward=64,
                                               dropout=dropout,
                                               batch_first=True,
                                               activation='gelu')
            sd = StochasticDepth(drop_prob=layer_drop)
            layers.append(nn.ModuleList([layer, sd]))
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def _build_causal_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(self, global_feat, target_tokens=None):
        B = global_feat.shape[0]
        device = global_feat.device
        context = self.context_proj(global_feat).unsqueeze(1)

        if self.training and target_tokens is not None:
            T = target_tokens.size(1)
            token_emb = self.token_embedding(target_tokens)
            input_emb = torch.cat([context, token_emb], dim=1)
            pos_ids = torch.arange(T + 1,
                                   device=device).unsqueeze(0).expand(B,
                                                                      -1)
            x = input_emb + self.pos_embedding(pos_ids)
            mask = self._build_causal_mask(T + 1).to(device)

            for layer, sd in self.layers:
                residual = x
                x = layer(x, src_mask=mask)
                x = sd(x - residual) + residual
            x = self.norm(x)
            logits = self.output_proj(x[:, :-1, :])
            return logits
        else:
            generated = []
            emb = context
            for t in range(self.max_len):
                T_curr = emb.size(1)
                pos_ids = torch.arange(T_curr,
                                       device=device).unsqueeze(0).expand(
                                           B,
                                           -1)
                x = emb + self.pos_embedding(pos_ids)
                mask = self._build_causal_mask(T_curr).to(device)
                for layer, sd in self.layers:
                    residual = x
                    x = layer(x, src_mask=mask)
                    x = sd(x - residual) + residual
                x = self.norm(x)
                logits = self.output_proj(x[:, -1, :])
                next_token = logits.argmax(dim=-1)
                generated.append(next_token)
                if (next_token == 2).all():
                    break
                next_emb = self.token_embedding(next_token).unsqueeze(1)
                emb = torch.cat([emb, next_emb], dim=1)
            generated = torch.stack(generated,
                                    dim=1) if generated else torch.zeros(
                                        B,
                                        0,
                                        device=device,
                                        dtype=torch.long)
            L = generated.size(1)
            if L < self.max_len:
                pad = torch.full((B,
                                  self.max_len - L),
                                 0,
                                 dtype=torch.long,
                                 device=device)
                generated = torch.cat([generated, pad], dim=1)
            logits_out = torch.zeros(B,
                                     self.max_len,
                                     self.vocab_size,
                                     device=device)
            logits_out.scatter_(2, generated.unsqueeze(2), 1.0)
            return logits_out


# Loss Modules
class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs,
                                                 targets,
                                                 reduction='none')
        pt = torch.exp(-bce)
        focal_weight = (1 - pt)**self.gamma
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return (focal_weight * alpha_t * bce).mean()


class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        inputs_flat = probs.view(probs.size(0), probs.size(1), -1)
        targets_flat = targets.view(targets.size(0), targets.size(1), -1)
        inter = (inputs_flat * targets_flat).sum(dim=2)
        union = inputs_flat.sum(dim=2) + targets_flat.sum(dim=2)
        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class NonZeroRegularization(nn.Module):

    def __init__(self, weight=1.0, target_sparsity=0.05):
        super().__init__()
        self.weight = weight
        self.target_sparsity = target_sparsity

    def forward(self, preds):
        mean_act = torch.sigmoid(preds).reshape(preds.size(0), -1).mean(dim=1)
        target = torch.full_like(mean_act, self.target_sparsity)
        return self.weight * F.mse_loss(mean_act, target)


class DSVTFullModel(pl.LightningModule):

    def __init__(self, vocab_size, max_comment_len=150):
        super().__init__()
        self.save_hyperparameters(ignore=["val_data_paths"])

        self.backbone = DSVTBackbone(in_channels=3,
                                     embed_dim=32,
                                     resolution=VOXEL_RESOLUTION,
                                     tau=64,
                                     num_layers=9)
        self.decoder = DSVTDecoder(embed_dim=32,
                                   out_channels=6,
                                   resolution=VOXEL_RESOLUTION,
                                   tau=64,
                                   num_layers=9)
        self.captioner = GPTCaptioner(embed_dim=32,
                                      vocab_size=vocab_size,
                                      max_len=max_comment_len,
                                      num_layers=12,
                                      nhead=4,
                                      dropout=0.4,
                                      layer_drop=0.4)

        # Attention pooling for fused features
        self.global_attn_pool = nn.MultiheadAttention(embed_dim=32,
                                                      num_heads=4,
                                                      batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 1, 32))
        self.fuse_proj = nn.Linear(32 * 2, 32)

        # Losses
        self.focal_loss = FocalLoss(alpha=0.2, gamma=2.0)
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        self.nonzero_emo = NonZeroRegularization(
            weight=0.1,
            target_sparsity=NONZERO_EMO_TARGET)
        self.nonzero_gaze = NonZeroRegularization(
            weight=0.05,
            target_sparsity=NONZERO_GAZE_TARGET)
        self.val_data_paths = []

    def forward(self, x, tokens=None):
        B = x.shape[0]
        enc_out = self.backbone(x)
        if enc_out[0] is None:
            emo = torch.zeros(B,
                              5,
                              VOXEL_RESOLUTION,
                              VOXEL_RESOLUTION,
                              VOXEL_RESOLUTION,
                              device=x.device)
            gaze = torch.zeros(B,
                               1,
                               VOXEL_RESOLUTION,
                               VOXEL_RESOLUTION,
                               VOXEL_RESOLUTION,
                               device=x.device)
            cap_logits = torch.zeros(B,
                                     self.hparams.max_comment_len,
                                     self.hparams.vocab_size,
                                     device=x.device)
            return emo, gaze, cap_logits

        tokens_enc, coords, batch_idx, xyz, dims, _, early_features = enc_out

        # Get both logits and decoder tokens
        dec_logits, tokens_dec = self.decoder(tokens_enc, coords, batch_idx, xyz, dims, B, early_features)
        emo = dec_logits[:, :5]
        gaze = dec_logits[:, 5:6]

        # Fuse encoder and decoder tokens per batch
        global_feats = []
        for b in range(B):
            mask_b = (batch_idx == b)
            if not mask_b.any():
                feat_b = torch.zeros(32, device=x.device)
            else:
                enc_t = tokens_enc[mask_b]
                dec_t = tokens_dec[mask_b]
                # Option 1: concatenate and project
                # fused = torch.cat([enc_t, dec_t], dim=-1)  # [N_b, 64]
                # fused = self.fuse_proj(fused)  # [N_b, 32]

                # Option 2: just sum
                fused = enc_t + dec_t

                # Apply attention pooling on fused tokens
                fused = fused.unsqueeze(0)  # [1, N_b, 32]
                query = self.pool_query
                attn_out, _ = self.global_attn_pool(query, fused, fused)
                feat_b = attn_out.squeeze(1).squeeze(0)  # [32]

                # enc_t = tokens_enc[mask_b].unsqueeze(0)
                # dec_t = tokens_dec[mask_b].unsqueeze(0)
                # query = self.pool_query.expand(1, -1, -1)
                # attn_out, _ = self.global_attn_pool(query, enc_t, dec_t)
                # feat_b = attn_out.squeeze(1).squeeze(0)
            global_feats.append(feat_b)
        global_feats = torch.stack(global_feats)  # [B, 32]

        cap_logits = self.captioner(global_feats,
                                    tokens if self.training else None)
        return emo, gaze, cap_logits

    def _shared_step(self, batch, stage):
        inputs, (emo_labels, gaze_labels, tokens) = batch
        emo_labels = emo_labels.float()
        gaze_labels = gaze_labels.float()

        emo_pred, gaze_pred, cap_pred = self(inputs, tokens if self.training else None)

        # Emotion loss
        loss_emo_focal = self.focal_loss(emo_pred, emo_labels)
        loss_emo_dice = self.dice_loss(emo_pred, emo_labels)
        loss_emo_l1 = F.l1_loss(torch.sigmoid(emo_pred), emo_labels)
        loss_emo_reg = self.nonzero_emo(emo_pred)

        # Gaze loss
        loss_gaze_focal = self.focal_loss(gaze_pred, gaze_labels)
        loss_gaze_dice = self.dice_loss(gaze_pred, gaze_labels)
        loss_gaze_l1 = F.l1_loss(torch.sigmoid(gaze_pred), gaze_labels)
        loss_gaze_reg = self.nonzero_gaze(gaze_pred)

        # Caption loss
        cap_target = tokens[:, :cap_pred.size(1)]
        if cap_target.size(1) < cap_pred.size(1):
            pad = torch.zeros(cap_target.size(0),
                              cap_pred.size(1) - cap_target.size(1),
                              dtype=torch.long,
                              device=cap_target.device)
            cap_target = torch.cat([cap_target, pad], dim=1)
        loss_cap = self.ce_loss(cap_pred.reshape(-1,
                                                 self.hparams.vocab_size),
                                cap_target.reshape(-1))

        total = (0.5 * loss_emo_focal + 1.0 * loss_emo_dice +
                 L1_WEIGHT * loss_emo_l1 + loss_emo_reg +
                 1.0 * loss_gaze_focal + 0.5 * loss_gaze_dice +
                 L1_WEIGHT * loss_gaze_l1 + loss_gaze_reg + loss_cap)

        self.log_dict(
            {
                f"{stage}_loss": total,
                f"{stage}_emo_focal": loss_emo_focal,
                f"{stage}_emo_dice": loss_emo_dice,
                f"{stage}_gaze_focal": loss_gaze_focal,
                f"{stage}_gaze_dice": loss_gaze_dice,
                f"{stage}_emo_l1": loss_emo_l1,
                f"{stage}_gaze_l1": loss_gaze_l1,
                f"{stage}_cap": loss_cap,
            },
            on_epoch=True,
            prog_bar=(stage == 'val'))
        return total

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')

    def on_validation_start(self):
        if hasattr(self.trainer.datamodule, 'val_paths'):
            self.val_data_paths = self.trainer.datamodule.val_paths
        elif hasattr(self.trainer.datamodule,
                     'val_dataset') and hasattr(
                         self.trainer.datamodule.val_dataset,
                         'data_paths'):
            self.val_data_paths = self.trainer.datamodule.val_dataset.data_paths
        else:
            self.val_data_paths = []

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, 'val')

    def configure_optimizers(self):
        gpt_params = list(self.captioner.parameters())
        backbone_params = (list(self.backbone.parameters()) +
                           list(self.decoder.parameters()) +
                           list(self.global_attn_pool.parameters()) +
                           list(self.fuse_proj.parameters()) +
                           [self.pool_query])

        optimizer = torch.optim.AdamW([{
            'params': backbone_params,
            'weight_decay': 0.01
        },
                                       {
                                           'params': gpt_params,
                                           'weight_decay': 0.1
                                       }],
                                      lr=LEARNING_RATE)

        linear_warmup_epochs = 30
        linear_warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=linear_warmup_epochs)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [linear_warmup,
             cosine_scheduler],
            milestones=[linear_warmup_epochs])
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }


# Data Module (unchanged – assumed to exist)
from dataset.voxel_dataset import ExtendedVoxelDataset
from dataset.utils import filter_data_on_condition
from model.helper import SavePredictionCallback, SimpleTokenizer


class FullDataModule(pl.LightningDataModule):

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
        self.tokenizer = SimpleTokenizer(max_len=MAX_COMMENT_LEN)

    def setup(self, stage=None):
        np.random.shuffle(self.all_data_paths)
        train_paths = [
            p for p in self.all_data_paths
            if p['GROUP'] not in self.test_groups
        ]
        val_paths = [
            p for p in self.all_data_paths if p['GROUP'] in self.test_groups
        ]

        all_comments = []
        for item in self.all_data_paths:
            comment_path = item.get('TRANSCRIPT', '')
            if os.path.exists(comment_path):
                try:
                    with open(comment_path, 'r', encoding='utf-8') as f:
                        all_comments.append(f.read().strip())
                except:
                    pass
        self.tokenizer.build_vocab(all_comments)
        print(f"Vocab size: {self.tokenizer.vocab_size}")

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


# Main
if __name__ == "__main__":
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('high')

    print("Loading data...")
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

    if len(all_data_paths) == 0:
        raise RuntimeError("No data found!")

    datamodule = FullDataModule(all_data_paths=all_data_paths,
                                batch_size=BATCH_SIZE,
                                num_workers=NUM_WORKERS,
                                voxel_resolution=VOXEL_RESOLUTION,
                                test_groups=TEST_GROUPS)
    datamodule.setup()

    model = DSVTFullModel(vocab_size=datamodule.tokenizer.vocab_size,
                          max_comment_len=MAX_COMMENT_LEN)

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        # devices="auto",
        devices=1,
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                         save_top_k=2,
                                         every_n_epochs=20,
                                         save_last=True,
                                         mode='min'),
            pl.callbacks.EarlyStopping(monitor='val_loss',
                                       patience=EARLY_STOPPING_PATIENCE,
                                       mode='min'),
            SavePredictionCallback(save_dir=SAVE_DIR,
                                   emotion_order=EMOTION_ORDER,
                                   save_every_n_epochs=1,
                                   max_samples_to_save=100)
        ],
        log_every_n_steps=10,
        precision='16-mixed' if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0)

    trainer.fit(model, datamodule=datamodule)
