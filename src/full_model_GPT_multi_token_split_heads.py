import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" 
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchinfo
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import open3d as o3d
from dataset.utils import filter_data_on_condition, DEFAULT_QNA_ANSWER_COLOR_MAP
from dataset.voxel_dataset import ExtendedVoxelDataset
from model.config import *
from model.helper import *
from model.loss import *
from model.layers import *


class ExtendedVoxelDataModule(pl.LightningDataModule):
    def __init__(self,
                 all_data_paths,
                 batch_size,
                 num_workers,
                 voxel_resolution,
                 max_comment_len,
                 test_groups,
                 augment_color_p=0.5,
                 color_jitter_std=0.05,
                 jitter_voxel_p=0.2,
                 prefetch_factor=2):
        super().__init__()
        self.save_hyperparameters(ignore=['all_data_paths'])
        self.all_data_paths = all_data_paths
        self.tokenizer = SimpleTokenizer(max_len=self.hparams.max_comment_len)
        self.train_paths = []
        self.val_paths = []

    def setup(self, stage=None):
        if stage in ('fit', None):
            all_comments = []
            for item in self.all_data_paths:
                comment_path = item.get('TRANSCRIPT', '')
                if os.path.exists(comment_path):
                    try:
                        with open(comment_path, 'r', encoding='utf-8') as f:
                            all_comments.append(f.read().strip())
                    except Exception as e:
                        print(f"WARNING: Failed to read {comment_path}")
            self.tokenizer.build_vocab(all_comments)
            print(
                f"Tokenizer vocabulary total size = {self.tokenizer.vocab_size} words."
            )
        np.random.shuffle(self.all_data_paths)
        for data_paths in self.all_data_paths:
            if (data_paths['GROUP'] in self.hparams.test_groups):
                self.val_paths.append(data_paths)
            else:
                self.train_paths.append(data_paths)
        print(
            f"Data split: {len(self.train_paths)} training, {len(self.val_paths)} validation."
        )
        if stage in ('fit', None):
            common_args = {
                'voxel_resolution': self.hparams.voxel_resolution,
                'tokenizer': self.tokenizer
            }
            self.train_dataset = ExtendedVoxelDataset(
                self.train_paths,
                augment_color_p=self.hparams.augment_color_p,
                color_jitter_std=self.hparams.color_jitter_std,
                jitter_voxel_p=self.hparams.jitter_voxel_p,
                **common_args)
            self.val_dataset = ExtendedVoxelDataset(self.val_paths,
                                                    augment_color_p=0.0,
                                                    **common_args)
        if stage in ('test', None):
            common_args = {
                'voxel_resolution': self.hparams.voxel_resolution,
                'tokenizer': self.tokenizer
            }
            self.test_dataset = ExtendedVoxelDataset(self.val_paths,
                                                     augment_color_p=0.0,
                                                     **common_args)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
            pin_memory=True,
            prefetch_factor=self.hparams.prefetch_factor)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
            pin_memory=True,
            prefetch_factor=self.hparams.prefetch_factor)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
            pin_memory=True,
            prefetch_factor=self.hparams.prefetch_factor)



class MeaningMakingModel(nn.Module):
    def __init__(self,
                 num_emotions,
                 vocab_size,
                 max_comment_len,
                 conv_dims: list,
                 resolution,
                 embed_dim=64,
                 hidden_dim=64,
                 num_experts=3,
                 gpt_num_layers=2,
                 gpt_nhead=8):
        super().__init__()
        self.depth = len(conv_dims) - 1
        final_size = resolution // (2**self.depth)
        flat_feature_size = conv_dims[-1] * (final_size**3)

        self.encoder_blocks = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(conv_dims[:-1], conv_dims[1:])):
            self.encoder_blocks.append(ConvConvEncoder(in_dim, out_dim))
        self.pool = nn.MaxPool3d(2)

        up_dims = [conv_dims[-1]] + conv_dims[1:][::-1]
        self.fusion_layer_idx = 2

        # Emotion Stream
        self.emo_decoder_blocks = nn.ModuleList()
        self.emo_skip_blocks = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(up_dims[:-1], up_dims[1:])):
            self.emo_decoder_blocks.append(UpSampleDecoder(in_dim, out_dim))
            skip_idx = self.depth - 1 - i
            if skip_idx >= 0:
                skip_channels = conv_dims[skip_idx + 1]
                self.emo_skip_blocks.append(SkipBlock(out_dim, skip_channels))
            else:
                self.emo_skip_blocks.append(
                    nn.Sequential(
                        nn.Conv3d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm3d(out_dim), nn.ReLU()))

        # Heatmap Stream
        self.heat_decoder_blocks = nn.ModuleList()
        self.heat_skip_blocks = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(up_dims[:-1], up_dims[1:])):
            self.heat_decoder_blocks.append(UpSampleDecoder(in_dim, out_dim))
            skip_idx = self.depth - 1 - i
            if skip_idx >= 0:
                skip_channels = conv_dims[skip_idx + 1]
                self.heat_skip_blocks.append(SkipBlock(out_dim, skip_channels))
            else:
                self.heat_skip_blocks.append(
                    nn.Sequential(
                        nn.Conv3d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm3d(out_dim), nn.ReLU()))

        fused_dim = up_dims[self.fusion_layer_idx] * 2
        self.fusion_conv_emo = nn.Sequential(
            nn.Conv3d(fused_dim, up_dims[self.fusion_layer_idx], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(up_dims[self.fusion_layer_idx]), nn.ReLU())
        self.fusion_conv_heat = nn.Sequential(
            nn.Conv3d(fused_dim, up_dims[self.fusion_layer_idx], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(up_dims[self.fusion_layer_idx]), nn.ReLU())

        self.heatmap_head = nn.ConvTranspose3d(up_dims[-1], 1, kernel_size=1)
        self.emotion_head = nn.ConvTranspose3d(up_dims[-1], num_emotions, kernel_size=1)
        self.emotion_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.heatmap_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))

        visual_context_size = flat_feature_size + num_emotions + 1
        self.context_projection = nn.Sequential(
            nn.Linear(visual_context_size, hidden_dim), nn.ReLU(), nn.Dropout(0.3))

        self.vocab_size = vocab_size
        self.max_comment_len = max_comment_len
        self.embed_dim = embed_dim

        # Token & position embeddings (note: +1 for prefix)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_comment_len + 1, embed_dim)

        # Project visual context to prefix token
        self.context_to_prefix = nn.Linear(hidden_dim, embed_dim)

        # GPT-style causal decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=gpt_nhead,
            dim_feedforward=256,
            dropout=0.3,
            batch_first=True,
            activation='gelu'
        )
        self.gpt_decoder = nn.TransformerEncoder(decoder_layer, num_layers=gpt_num_layers)
        self.output_projection = nn.Linear(embed_dim, vocab_size)

        # Auxiliary heads
        self.aux_emo_classifier = nn.Linear(flat_feature_size, num_emotions)
        self.aux_heat_regressor = nn.Linear(flat_feature_size, 1)
        self.aux_3tok_head = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 3 * vocab_size)
        )

        # Experts
        self.num_experts = num_experts
        self.bottleneck_dim = conv_dims[-1]
        self.experts = nn.ModuleList([
            ExpertBlock_PersonalityBlock(self.bottleneck_dim) for _ in range(self.num_experts)
        ])

    def _build_causal_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self,
                x,
                target_tokens=None,
                teacher_forcing_ratio=0.5,
                return_bottleneck=False,
                return_all_experts=False,
                expert_idx=None):
        batch_size = x.size(0)
        skip_features = []
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            skip_features.append(x)
            x = self.pool(x)
        bottleneck_features = x
        flat_features = bottleneck_features.view(batch_size, -1)
        if return_bottleneck:
            return flat_features

        expert_features = []
        for expert in self.experts:
            attn = expert(bottleneck_features)
            modulated = bottleneck_features * attn
            expert_features.append(modulated)

        if return_all_experts:
            outputs = []
            for feat in expert_features:
                emo, heat, tok, aux_3tok = self._decode_and_predict(feat, skip_features, target_tokens)
                outputs.append((emo, heat, tok, aux_3tok))
            return outputs, flat_features
        elif expert_idx is not None:
            emo, heat, tok, aux_3tok = self._decode_and_predict(expert_features[expert_idx], skip_features, target_tokens)
            return emo, heat, tok, aux_3tok, flat_features
        else:
            emo, heat, tok, aux_3tok = self._decode_and_predict(expert_features[0], skip_features, target_tokens)
            return emo, heat, tok, aux_3tok, flat_features

    def _decode_and_predict(self, bottleneck, skip_features, target_tokens):
        batch_size = bottleneck.size(0)
        emo_features = bottleneck
        heat_features = bottleneck
        for layer_idx in range(len(self.emo_decoder_blocks)):
            emo_features = self.emo_decoder_blocks[layer_idx](emo_features)
            heat_features = self.heat_decoder_blocks[layer_idx](heat_features)
            skip_idx = self.depth - 1 - layer_idx
            if skip_idx >= 0 and skip_idx < len(skip_features):
                skip_emo = skip_features[skip_idx]
                skip_heat = skip_features[skip_idx]
                if emo_features.shape[2:] != skip_emo.shape[2:]:
                    skip_emo = F.interpolate(skip_emo, size=emo_features.shape[2:], mode='trilinear', align_corners=True)
                if heat_features.shape[2:] != skip_heat.shape[2:]:
                    skip_heat = F.interpolate(skip_heat, size=heat_features.shape[2:], mode='trilinear', align_corners=True)
                emo_features = torch.cat([emo_features, skip_emo], dim=1)
                heat_features = torch.cat([heat_features, skip_heat], dim=1)
            emo_features = self.emo_skip_blocks[layer_idx](emo_features)
            heat_features = self.heat_skip_blocks[layer_idx](heat_features)
            if layer_idx == self.fusion_layer_idx - 1:
                fused_emo = torch.cat([emo_features, heat_features], dim=1)
                fused_heat = torch.cat([heat_features, emo_features], dim=1)
                emo_features = self.fusion_conv_emo(fused_emo)
                heat_features = self.fusion_conv_heat(fused_heat)

        emotion_preds = self.emotion_head(emo_features)
        heatmap_preds = self.heatmap_head(heat_features)
        emotion_context = self.emotion_pooling(emotion_preds.detach()).view(batch_size, -1)
        heatmap_context = self.heatmap_pooling(heatmap_preds.detach()).view(batch_size, -1)
        flat_features = bottleneck.view(batch_size, -1)
        visual_features = torch.cat([flat_features, emotion_context, heatmap_context], dim=1)
        visual_context = self.context_projection(visual_features)

        aux_3tok_logits = None
        if self.training and target_tokens is not None:
            aux_3tok_logits = self.aux_3tok_head(visual_context)
            aux_3tok_logits = aux_3tok_logits.view(-1, 3, self.vocab_size)

        if self.training and target_tokens is not None:
            token_preds = self._forward_train_gpt(visual_context, target_tokens)
        else:
            token_preds = self._forward_inference_gpt(visual_context)

        return emotion_preds, heatmap_preds, token_preds, aux_3tok_logits

    def _forward_train_gpt(self, visual_context, target_tokens):
        B, T = target_tokens.shape
        device = target_tokens.device

        prefix_emb = self.context_to_prefix(visual_context).unsqueeze(1)  # [B, 1, E]
        token_emb = self.token_embedding(target_tokens)                   # [B, T, E]
        input_emb = torch.cat([prefix_emb, token_emb], dim=1)             # [B, T+1, E]

        pos_ids = torch.arange(T + 1, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embedding(pos_ids)
        x = input_emb + pos_emb

        causal_mask = self._build_causal_mask(T + 1).to(device)
        decoded = self.gpt_decoder(x, mask=causal_mask)
        logits = self.output_projection(decoded[:, :-1, :])  # [B, T, V]
        return logits

    def _forward_inference_gpt(self, visual_context):
        B = visual_context.size(0)
        device = visual_context.device
        max_len = self.max_comment_len

        prefix_emb = self.context_to_prefix(visual_context).unsqueeze(1)  # [B, 1, E]
        generated_embs = prefix_emb
        generated_tokens = []

        for t in range(max_len):
            T_curr = generated_embs.size(1)
            pos_ids = torch.arange(T_curr, device=device).unsqueeze(0).expand(B, -1)
            x = generated_embs + self.pos_embedding(pos_ids)

            causal_mask = self._build_causal_mask(T_curr).to(device)
            decoded = self.gpt_decoder(x, mask=causal_mask)
            logits = self.output_projection(decoded[:, -1, :])
            next_token = logits.argmax(dim=-1)
            generated_tokens.append(next_token)

            if (next_token == 2).all():
                break

            next_emb = self.token_embedding(next_token).unsqueeze(1)
            generated_embs = torch.cat([generated_embs, next_emb], dim=1)

        generated_tokens = torch.stack(generated_tokens, dim=1)  # [B, L]
        L = generated_tokens.size(1)
        if L < max_len:
            pad = torch.full((B, max_len - L), 0, device=device, dtype=torch.long)
            generated_tokens = torch.cat([generated_tokens, pad], dim=1)

        logits_out = torch.zeros(B, max_len, self.vocab_size, device=device)
        logits_out.scatter_(2, generated_tokens.unsqueeze(2), 1.0)
        return logits_out



class MeaningMakingLightningModule(pl.LightningModule):
    def __init__(self,
                 model,
                 learning_rate=1e-3,
                 l1_weight=0.01,
                 teacher_forcing_ratio=0.3,
                 nonzero_emo_target=0.015,
                 nonzero_heat_target=0.01,
                 voxel_loss_weight=1.0,
                 emo_loss_weight=1.0,
                 heat_loss_weight=1.0,
                 global_loss_weight=0.2,
                 aux_3tok_weight=0.5):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        initialize_sparse_heads(self.model)
        self.emotion_criterion = CombinedSparseLoss(use_focal=True, use_dice=True, use_iou=True,
                                                    focal_weight=0.5, dice_weight=1.0, iou_weight=1.0)
        self.heatmap_criterion = CombinedSparseLoss(use_focal=True, use_dice=True, use_iou=True,
                                                    focal_weight=1.5, dice_weight=0.5, iou_weight=0.5)
        self.emo_sparsity = self.hparams.nonzero_emo_target
        self.heat_sparsity = self.hparams.nonzero_heat_target
        self.nonzero_reg_emo = NonZeroRegularization(weight=0.5, target_sparsity=self.hparams.nonzero_emo_target)
        self.nonzero_reg_heat = NonZeroRegularization(weight=0.05, target_sparsity=self.hparams.nonzero_heat_target)
        self.token_criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        self.voxel_loss_weight = self.hparams.voxel_loss_weight
        self.aux_3tok_weight = aux_3tok_weight
        self.val_data_paths = []

    def forward(self, x, target_tokens=None):
        return self.model(x, target_tokens)

    def _shared_step(self, batch, is_training=True):
        inputs, (emotion_labels, heatmaps, tokens) = batch
        emotion_labels = emotion_labels.float()
        heatmaps = heatmaps.float()
        if is_training:
            emotion_labels = emotion_labels * 0.98 + 0.01
            heatmaps = heatmaps * 0.98 + 0.01

        current_epoch = self.current_epoch if self.trainer else 0
        text_scale = 0.1 + 0.9 * min(current_epoch / 50.0, 1.0)

        if is_training:
            all_outputs, flat_features = self.model(inputs, target_tokens=tokens, return_all_experts=True)
        else:
            all_outputs, flat_features = self.model(inputs, target_tokens=None, return_all_experts=True)

        # Encoder auxiliary losses
        aux_emo_logits = self.model.aux_emo_classifier(flat_features)
        emo_any_threshold = 0.1 if is_training else 0.5
        emo_any = (emotion_labels > emo_any_threshold).float().amax(dim=(2, 3, 4))
        aux_emo_loss = F.binary_cross_entropy_with_logits(aux_emo_logits, emo_any)
        heat_max = heatmaps.amax(dim=(2, 3, 4))
        aux_heat_pred_raw = self.model.aux_heat_regressor(flat_features)
        aux_heat_loss = F.mse_loss(torch.sigmoid(aux_heat_pred_raw), heat_max)
        aux_loss = aux_emo_loss + aux_heat_loss

        # Expert losses + 3-token aux loss
        branch_losses = []
        aux_3tok_losses = []
        for emo_pred, heat_pred, tok_pred, aux_3tok_logits in all_outputs:
            loss_emo, _ = self.emotion_criterion(emo_pred, emotion_labels)
            loss_heat, _ = self.heatmap_criterion(heat_pred, heatmaps)

            pred_seq_len = tok_pred.size(1)
            target_seq_len = self.model.max_comment_len
            tok_target = tokens
            if pred_seq_len < target_seq_len:
                padding = torch.zeros(tok_pred.size(0), target_seq_len - pred_seq_len, tok_pred.size(2), device=tok_pred.device)
                tok_pred = torch.cat([tok_pred, padding], dim=1)
            elif pred_seq_len > target_seq_len:
                tok_pred = tok_pred[:, :target_seq_len, :]
            if tok_target.size(1) > target_seq_len:
                tok_target = tok_target[:, :target_seq_len]
            elif tok_target.size(1) < target_seq_len:
                pad = torch.zeros(tok_target.size(0), target_seq_len - tok_target.size(1), dtype=torch.long, device=tok_target.device)
                tok_target = torch.cat([tok_target, pad], dim=1)

            loss_tok = self.token_criterion(tok_pred.reshape(-1, self.model.vocab_size), tok_target.reshape(-1)) * text_scale
            nonzero_emo = self.nonzero_reg_emo(emo_pred)
            nonzero_heat = self.nonzero_reg_heat(heat_pred)
            l1_emo = F.l1_loss(torch.sigmoid(emo_pred), emotion_labels)
            l1_heat = F.l1_loss(torch.sigmoid(heat_pred), heatmaps)
            weighted_emo = self.hparams.emo_loss_weight * loss_emo
            weighted_heat = self.hparams.heat_loss_weight * loss_heat
            global_loss = self.hparams.global_loss_weight * (loss_emo + loss_heat + loss_tok)
            voxel_loss = (weighted_emo + weighted_heat + global_loss +
                          nonzero_emo + nonzero_heat +
                          self.hparams.l1_weight * 0.5 * (l1_emo + l1_heat))

            aux_3tok_loss = torch.tensor(0.0, device=flat_features.device)
            if is_training and aux_3tok_logits is not None:
                target_3 = tokens[:, 1:4]
                if target_3.size(1) < 3:
                    pad = torch.zeros(target_3.size(0), 3 - target_3.size(1), dtype=torch.long, device=target_3.device)
                    target_3 = torch.cat([target_3, pad], dim=1)
                aux_3tok_loss = F.cross_entropy(
                    aux_3tok_logits.reshape(-1, self.model.vocab_size),
                    target_3.reshape(-1),
                    ignore_index=0,
                    label_smoothing=0.1
                ) * text_scale * self.aux_3tok_weight

            aux_3tok_losses.append(aux_3tok_loss)
            expert_specific_loss = self.hparams.voxel_loss_weight * voxel_loss + loss_tok + aux_3tok_loss
            branch_losses.append(expert_specific_loss)

        branch_losses = torch.stack(branch_losses)
        best_idx = torch.argmin(branch_losses).item()
        if is_training:
            best_loss = branch_losses[best_idx]
            other_losses_detached = sum(loss.detach() for i, loss in enumerate(branch_losses) if i != best_idx)
            total_loss = best_loss + other_losses_detached + aux_loss
            self.log("train_best_expert", float(best_idx), on_step=True, on_epoch=False)
        else:
            total_loss = branch_losses[best_idx] + aux_loss

        emotion_preds, heatmap_preds, token_preds, _ = all_outputs[best_idx]
        loss_emo, _ = self.emotion_criterion(emotion_preds, emotion_labels)
        loss_heat, _ = self.heatmap_criterion(heatmap_preds, heatmaps)
        pred_seq_len = token_preds.size(1)
        target_seq_len = self.model.max_comment_len
        tok_target = tokens
        if pred_seq_len < target_seq_len:
            padding = torch.zeros(token_preds.size(0), target_seq_len - pred_seq_len, token_preds.size(2), device=token_preds.device)
            token_preds = torch.cat([token_preds, padding], dim=1)
        elif pred_seq_len > target_seq_len:
            token_preds = token_preds[:, :target_seq_len, :]
        if tok_target.size(1) > target_seq_len:
            tok_target = tok_target[:, :target_seq_len]
        elif tok_target.size(1) < target_seq_len:
            pad = torch.zeros(tok_target.size(0), target_seq_len - tok_target.size(1), dtype=torch.long, device=tok_target.device)
            tok_target = torch.cat([tok_target, pad], dim=1)
        loss_tok = self.token_criterion(token_preds.reshape(-1, self.model.vocab_size), tok_target.reshape(-1)) * text_scale
        nonzero_emo = self.nonzero_reg_emo(emotion_preds)
        nonzero_heat = self.nonzero_reg_heat(heatmap_preds)
        l1_emo = F.l1_loss(torch.sigmoid(emotion_preds), emotion_labels)
        l1_heat = F.l1_loss(torch.sigmoid(heatmap_preds), heatmaps)
        logged_aux_3tok_loss = aux_3tok_losses[best_idx] if is_training else torch.tensor(0.0, device=flat_features.device)

        return total_loss, loss_emo, loss_heat, loss_tok, l1_emo, l1_heat, emotion_preds, heatmap_preds, nonzero_emo, nonzero_heat, aux_loss, logged_aux_3tok_loss

    def training_step(self, batch, batch_idx):
        loss, l_emo, l_heat, l_tok, l1_emo, l1_heat, _, _, nz_emo, nz_heat, aux_loss, aux_3tok_loss = self._shared_step(batch, is_training=True)
        self.log_dict({
            'train_loss': loss,
            'train_emo': l_emo,
            'train_heat': l_heat,
            'train_tok': l_tok,
            'train_l1_emo': l1_emo,
            'train_l1_heat': l1_heat,
            'train_nonzero_emo': nz_emo,
            'train_nonzero_heat': nz_heat,
            'train_aux_loss': aux_loss,
            'train_aux_3tok_loss': aux_3tok_loss,
        }, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch[0].size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        loss, l_emo, l_heat, l_tok, l1_emo, l1_heat, emotion_preds, heatmap_preds, nz_emo, nz_heat, aux_loss, _ = self._shared_step(batch, is_training=False)
        inputs, (emotion_labels, heatmaps, tokens) = batch
        emotion_labels = emotion_labels.float()
        heatmaps = heatmaps.float()
        emotion_pred_binary = (torch.sigmoid(emotion_preds) > 0.5).float()
        emo_target_binary = (emotion_labels > 0.5).float()
        emo_iou = (emotion_pred_binary * emo_target_binary).sum() / ((emotion_pred_binary + emo_target_binary).clamp(0, 1).sum() + 1e-6)
        heat_pred_binary = (torch.sigmoid(heatmap_preds) > 0.5).float()
        heat_target_binary = (heatmaps > 0.1).float()
        heat_iou = (heat_pred_binary * heat_target_binary).sum() / ((heat_pred_binary + heat_target_binary).clamp(0, 1).sum() + 1e-6)
        self.log_dict({
            'val_loss': loss,
            'val_emo': l_emo,
            'val_heat': l_heat,
            'val_tok': l_tok,
            'val_l1_emo': l1_emo,
            'val_l1_heat': l1_heat,
            'val_nonzero_emo': nz_emo,
            'val_nonzero_heat': nz_heat,
            'val_aux_loss': aux_loss,
            'val_emo_iou_metric': emo_iou,
            'val_heat_iou_metric': heat_iou,
            'val_emo_pred_mean': torch.sigmoid(emotion_preds).mean(),
            'val_heat_pred_mean': torch.sigmoid(heatmap_preds).mean(),
        }, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch[0].size(0))
        if batch_idx == 0 and self.global_rank == 0:
            emo_mean = torch.sigmoid(emotion_preds).mean().item()
            heat_mean = torch.sigmoid(heatmap_preds).mean().item()
            if emo_mean < self.emo_sparsity:
                print(f"[Rank {self.global_rank}] WARNING: Emotion pred mean = {emo_mean:.6f}. Lower than data sparsity {self.emo_sparsity:.6f}")
            if heat_mean < self.heat_sparsity:
                print(f"[Rank {self.global_rank}] WARNING: Heatmap pred mean = {heat_mean:.6f}. Lower than data sparsity {self.heat_sparsity:.6f}")

    def configure_optimizers(self):
        params = [
            {'params': self.model.encoder_blocks.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.model.emo_decoder_blocks.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.model.heat_decoder_blocks.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.model.emo_skip_blocks.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.model.heat_skip_blocks.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.model.fusion_conv_emo.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.model.fusion_conv_heat.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.model.emotion_head.parameters(), 'lr': self.hparams.learning_rate * 1.5},
            {'params': self.model.heatmap_head.parameters(), 'lr': self.hparams.learning_rate * 1.5},
            {'params': self.model.token_embedding.parameters(), 'lr': self.hparams.learning_rate * 0.5},
            {'params': self.model.gpt_decoder.parameters(), 'lr': self.hparams.learning_rate},  # ← updated
            {'params': self.model.output_projection.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.model.context_projection.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.model.context_to_prefix.parameters(), 'lr': self.hparams.learning_rate},  # ← updated name
            {'params': self.model.aux_emo_classifier.parameters(), 'lr': self.hparams.learning_rate * 0.8},
            {'params': self.model.aux_heat_regressor.parameters(), 'lr': self.hparams.learning_rate * 0.8},
            {'params': self.model.aux_3tok_head.parameters(), 'lr': self.hparams.learning_rate * 0.8},
            {'params': self.model.experts.parameters(), 'lr': self.hparams.learning_rate * 1.5},
        ]
        optimizer = torch.optim.AdamW(params, weight_decay=0.01)
        linear_warmup_epochs = 30
        linear_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=linear_warmup_epochs)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS - linear_warmup_epochs, eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [linear_warmup, cosine_scheduler], milestones=[linear_warmup_epochs])
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}
        }

    def on_validation_start(self):
        if hasattr(self.trainer.datamodule, 'val_paths'):
            self.val_data_paths = self.trainer.datamodule.val_paths
        else:
            self.val_data_paths = []


if __name__ == "__main__":
    if not os.path.exists(RAW_DATA_DIR):
        print("Warning: Real data paths not found.")
    print("Starting Pottery Analysis Training")
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
        print("ERROR: No data files found after filtering. Please check paths and data.")
    else:
        pl.seed_everything(42)
        torch.set_float32_matmul_precision('high')
        datamodule = ExtendedVoxelDataModule(all_data_paths=all_data_paths,
                                             batch_size=BATCH_SIZE,
                                             num_workers=NUM_WORKERS,
                                             voxel_resolution=VOXEL_RESOLUTION,
                                             max_comment_len=MAX_COMMENT_LEN,
                                             test_groups=TEST_GROUPS,
                                             augment_color_p=AUGMENT_COLOR_P,
                                             color_jitter_std=COLOR_JITTER_STD,
                                             jitter_voxel_p=JITTER_VOXEL_P)
        datamodule.setup('fit')
        sample_batch = next(iter(datamodule.train_dataloader()))
        _, (emo_labels, heat_labels, _) = sample_batch
        actual_emo_sparsity = emo_labels.float().mean().item()
        actual_heat_sparsity = heat_labels.float().mean().item()
        print(f"Actual emotion sparsity: {actual_emo_sparsity:.6f}")
        print(f"Actual heatmap sparsity: {actual_heat_sparsity:.6f}")
        target_emo = min(0.02, max(0.005, actual_emo_sparsity * 2))
        target_heat = min(0.02, max(0.005, actual_heat_sparsity * 2))
        model = MeaningMakingModel(num_emotions=len(EMOTION_ORDER),
                                   vocab_size=datamodule.tokenizer.vocab_size,
                                   max_comment_len=MAX_COMMENT_LEN,
                                   conv_dims=CONV_DIMS,
                                   resolution=VOXEL_RESOLUTION,
                                   num_experts=NUM_EXPERTS)
        lightning_module = MeaningMakingLightningModule(
            model,
            learning_rate=LEARNING_RATE,
            l1_weight=L1_WEIGHT,
            teacher_forcing_ratio=TEACHER_FORCING_RATIO,
            nonzero_emo_target=target_emo,
            nonzero_heat_target=target_heat,
            voxel_loss_weight=VOXEL_LOSS_WEIGHT)
        prediction_saver = SavePredictionCallback(
            save_dir=SAVE_DIR,
            save_every_n_epochs=SAVE_EVERY_N_EPOCHS,
            max_samples_to_save=MAX_SAMPLES_TO_SAVE)
        trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, mode='min', verbose=True),
                ModelCheckpoint(monitor='val_loss', save_top_k=3, every_n_epochs=20, save_last=True, mode='min',
                                filename='model-{epoch:02d}-{val_loss:.4f}'),
                prediction_saver
            ],
            log_every_n_steps=10,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision='16-mixed' if torch.cuda.is_available() else 32,
            gradient_clip_val=1.0)
        print("Running torchinfo summary")
        try:
            torchinfo.summary(model, input_size=(BATCH_SIZE, 3, VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION))
        except Exception as e:
            print(f"torchinfo summary failed: {e}")
        if VISUALIZE_SAMPLES:
            show_n_samples(datamodule, NUM_SAMPLES)
        print("\nChecking Data Sparsity")
        print(f"Emotion target sparsity: {actual_emo_sparsity:.6f} ({actual_emo_sparsity*100:.4f}%)")
        print(f"Heatmap target sparsity: {actual_heat_sparsity:.6f} ({actual_heat_sparsity*100:.4f}%)")
        if emo_labels.sum() == 0:
            print("CRITICAL: Emotion labels in batch are ALL ZERO.")
        if heat_labels.sum() == 0:
            print("CRITICAL: Heatmap labels in batch are ALL ZERO.")
        trainer.fit(lightning_module, datamodule=datamodule)
        print("\n--- Training Finished")