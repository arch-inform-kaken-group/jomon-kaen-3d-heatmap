import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchinfo
import pytorch_lightning as pl
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
                f"Tokenizer vocaulary total size = {self.tokenizer.vocab_size} words."
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
                 num_experts=3):
        super().__init__()
        self.depth = len(conv_dims) - 1
        final_size = resolution // (2**self.depth)
        flat_feature_size = conv_dims[-1] * (final_size**3)
        self.encoder_blocks = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(conv_dims[:-1], conv_dims[1:])):
            self.encoder_blocks.append(ConvConvEncoder(in_dim, out_dim))
        self.pool = nn.MaxPool3d(2)

        up_dims = [conv_dims[-1]] + conv_dims[1:][::-1]
        self.decoder_blocks = nn.ModuleList()
        self.skip_conv_blocks = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(up_dims[:-1], up_dims[1:])):
            self.decoder_blocks.append(UpSampleDecoder(in_dim, out_dim))
            skip_idx = self.depth - 1 - i
            if skip_idx >= 0:
                skip_channels = conv_dims[skip_idx + 1]
                self.skip_conv_blocks.append(SkipBlock(out_dim, skip_channels))
            else:
                self.skip_conv_blocks.append(
                    nn.Sequential(
                        nn.Conv3d(out_dim,
                                  out_dim,
                                  kernel_size=3,
                                  padding=1,
                                  bias=False),
                        nn.BatchNorm3d(out_dim),
                        nn.ReLU()))

        self.heatmap_head = nn.ConvTranspose3d(up_dims[-1], 1, kernel_size=1)
        self.emotion_head = nn.ConvTranspose3d(up_dims[-1],
                                               num_emotions,
                                               kernel_size=1)

        self.emotion_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.heatmap_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))

        visual_context_size = flat_feature_size + num_emotions + 1
        self.context_projection = nn.Sequential(
            nn.Linear(visual_context_size,
                      hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3))

        # === TRANSFORMER DECODER REPLACEMENT ===
        self.vocab_size = vocab_size
        self.max_comment_len = max_comment_len
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.token_embedding = nn.Embedding(vocab_size,
                                            embed_dim,
                                            padding_idx=0)
        self.pos_embedding = nn.Embedding(max_comment_len, embed_dim)

        # Project visual context to match embed_dim for fusion
        self.context_to_embed = nn.Linear(hidden_dim, embed_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim,
                                                   nhead=8,
                                                   dim_feedforward=256,
                                                   dropout=0.3,
                                                   batch_first=True,
                                                   activation='gelu')
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer,
                                                         num_layers=2)
        self.output_projection = nn.Linear(embed_dim, vocab_size)

        # Auxiliary heads
        self.aux_emo_classifier = nn.Linear(flat_feature_size, num_emotions)
        self.aux_heat_regressor = nn.Linear(flat_feature_size, 1)

        # Experts
        self.num_experts = num_experts
        self.bottleneck_dim = conv_dims[-1]
        self.experts = nn.ModuleList([
            ExpertBlock_PersonalityBlock(self.bottleneck_dim)
            for _ in range(self.num_experts)
        ])

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
                emo, heat, tok = self._decode_and_predict(feat, skip_features, target_tokens)
                outputs.append((emo, heat, tok))
            return outputs
        elif expert_idx is not None:
            emo, heat, tok = self._decode_and_predict(expert_features[expert_idx], skip_features, target_tokens)
            return emo, heat, tok
        else:
            emo, heat, tok = self._decode_and_predict(expert_features[0], skip_features, target_tokens)
            return emo, heat, tok

    def _decode_and_predict(self, bottleneck, skip_features, target_tokens):
        decoder_out = bottleneck
        for i, (upsample_block, skip_block) in enumerate(zip(self.decoder_blocks, self.skip_conv_blocks)):
            decoder_out = upsample_block(decoder_out)
            skip_idx = self.depth - 1 - i
            if skip_idx >= 0 and skip_idx < len(skip_features):
                skip = skip_features[skip_idx]
                if decoder_out.shape[2:] != skip.shape[2:]:
                    skip = F.interpolate(skip,
                                         size=decoder_out.shape[2:],
                                         mode='trilinear',
                                         align_corners=True)
                decoder_out = torch.cat([decoder_out, skip], dim=1)
                decoder_out = skip_block(decoder_out)

        heatmap_preds = self.heatmap_head(decoder_out)
        emotion_preds = self.emotion_head(decoder_out)

        batch_size = emotion_preds.size(0)
        emotion_context = self.emotion_pooling(emotion_preds.detach()).view(
            batch_size,
            -1)
        heatmap_context = self.heatmap_pooling(heatmap_preds.detach()).view(
            batch_size,
            -1)
        flat_features = bottleneck.view(batch_size, -1)
        visual_features = torch.cat(
            [flat_features,
             emotion_context,
             heatmap_context],
            dim=1)
        visual_context = self.context_projection(
            visual_features)  # (B, hidden_dim)

        # Project to embedding space
        context_embed = self.context_to_embed(visual_context)  # (B, embed_dim)

        if self.training and target_tokens is not None:
            token_preds = self._forward_train(context_embed, target_tokens)
        else:
            token_preds = self._forward_inference(context_embed)
        return emotion_preds, heatmap_preds, token_preds

    def _forward_train(self, context_embed, target_tokens):
        B, T = target_tokens.shape
        device = target_tokens.device

        # Shift target: input = [BOS, w1, w2, ..., w_{T-1}]
        input_tokens = torch.cat([
            torch.ones(B, 1, dtype=torch.long, device=device) * 1,  # BOS = 1
            target_tokens[:, :-1]
        ], dim=1)

        token_emb = self.token_embedding(input_tokens)  # (B, T, embed_dim)
        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embedding(pos_ids)
        tgt = token_emb + pos_emb  # (B, T, embed_dim)

        # Memory = context repeated T times
        memory = context_embed.unsqueeze(1).expand(-1,
                                                   T,
                                                   -1)  # (B, T, embed_dim)

        # Causal mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            T,
            device=device)

        decoded = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        logits = self.output_projection(decoded)  # (B, T, vocab_size)
        return logits

    def _forward_inference(self, context_embed):
        B = context_embed.size(0)
        device = context_embed.device
        max_len = self.max_comment_len

        # Start with BOS
        generated = torch.ones(B,
                               1,
                               dtype=torch.long,
                               device=device) * 1  # (B, 1)

        for t in range(1, max_len):
            T_curr = generated.size(1)
            token_emb = self.token_embedding(
                generated)  # (B, T_curr, embed_dim)
            pos_ids = torch.arange(T_curr,
                                   device=device).unsqueeze(0).expand(B,
                                                                      -1)
            pos_emb = self.pos_embedding(pos_ids)
            tgt = token_emb + pos_emb  # (B, T_curr, embed_dim)

            memory = context_embed.unsqueeze(1).expand(
                -1,
                T_curr,
                -1)  # (B, T_curr, embed_dim)

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                T_curr,
                device=device)
            decoded = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
            logits = self.output_projection(decoded)  # (B, T_curr, vocab_size)

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == 2).all():  # EOS = 2
                break

        # Pad to max length
        if generated.size(1) < max_len:
            pad = torch.zeros(B,
                              max_len - generated.size(1),
                              dtype=torch.long,
                              device=device)
            generated = torch.cat([generated, pad], dim=1)

        # Convert to logits (one-hot style for compatibility)
        logits_out = torch.zeros(B, max_len, self.vocab_size, device=device)
        logits_out.scatter_(2, generated.unsqueeze(2), 1.0)
        return logits_out

class MeaningMakingLightningModule(pl.LightningModule):

    def __init__(self,
                 model,
                 learning_rate=1e-3,
                 l1_weight=0.01,
                 teacher_forcing_ratio=0.3,
                 nonzero_emo_target=0.015,
                 nonzero_heat_target=0.01,
                 voxel_loss_weight=1.0):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        initialize_sparse_heads(self.model)
        self.emotion_criterion = CombinedSparseLoss(use_focal=True,
                                                    use_dice=True,
                                                    use_iou=True,
                                                    focal_weight=0.5,
                                                    dice_weight=1.0,
                                                    iou_weight=1.0)
        self.heatmap_criterion = CombinedSparseLoss(use_focal=True,
                                                    use_dice=True,
                                                    use_iou=True,
                                                    focal_weight=1.5,
                                                    dice_weight=0.5,
                                                    iou_weight=0.5)
        self.emo_sparsity = nonzero_emo_target
        self.heat_sparsity = nonzero_heat_target
        self.nonzero_reg_emo = NonZeroRegularization(
            weight=0.5,
            target_sparsity=nonzero_emo_target)
        self.nonzero_reg_heat = NonZeroRegularization(
            weight=0.1,
            target_sparsity=nonzero_heat_target)
        self.token_criterion = nn.CrossEntropyLoss(ignore_index=0,
                                                   label_smoothing=0.1)
        self.voxel_loss_weight = voxel_loss_weight
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
            all_outputs = self.model(inputs,
                                     target_tokens=tokens,
                                     return_all_experts=True)
            branch_losses = []
            for emo_pred, heat_pred, tok_pred in all_outputs:
                loss_emo, _ = self.emotion_criterion(emo_pred, emotion_labels)
                loss_heat, _ = self.heatmap_criterion(heat_pred, heatmaps)
                pred_seq_len = tok_pred.size(1)
                target_seq_len = self.model.max_comment_len
                if pred_seq_len < target_seq_len:
                    padding = torch.zeros(tok_pred.size(0),
                                          target_seq_len - pred_seq_len,
                                          tok_pred.size(2),
                                          device=tok_pred.device)
                    tok_pred = torch.cat([tok_pred, padding], dim=1)
                elif pred_seq_len > target_seq_len:
                    tok_pred = tok_pred[:, :target_seq_len, :]
                tok_target = tokens[:, :target_seq_len]
                loss_tok = self.token_criterion(
                    tok_pred.reshape(-1,
                                     self.model.vocab_size),
                    tok_target.reshape(-1)) * text_scale
                nonzero_emo = self.nonzero_reg_emo(emo_pred)
                nonzero_heat = self.nonzero_reg_heat(heat_pred)
                l1_emo = F.l1_loss(torch.sigmoid(emo_pred), emotion_labels)
                l1_heat = F.l1_loss(torch.sigmoid(heat_pred), heatmaps)
                with torch.no_grad():
                    flat_features = self.model(inputs, return_bottleneck=True)
                aux_emo_logits = self.model.aux_emo_classifier(flat_features)
                emo_any = (emotion_labels > 0.1).float().amax(dim=(2, 3, 4))
                aux_emo_loss = F.binary_cross_entropy_with_logits(
                    aux_emo_logits,
                    emo_any)
                heat_max = heatmaps.amax(dim=(2, 3, 4))
                aux_heat_pred_raw = self.model.aux_heat_regressor(
                    flat_features)
                aux_heat_loss = F.mse_loss(torch.sigmoid(aux_heat_pred_raw),
                                           heat_max)
                aux_loss = aux_emo_loss + aux_heat_loss
                voxel_loss = loss_emo + loss_heat + nonzero_emo + nonzero_heat + self.hparams.l1_weight * 0.5 * (
                    l1_emo + l1_heat) + aux_loss
                total_loss = self.voxel_loss_weight * voxel_loss + loss_tok
                branch_losses.append(total_loss)
            branch_losses = torch.stack(branch_losses)
            best_idx = torch.argmin(branch_losses).item()
            total_loss = sum(loss if i == best_idx else loss.detach()
                             for i, loss in enumerate(branch_losses))
            emotion_preds, heatmap_preds, token_preds = all_outputs[best_idx]
            self.log("train_best_expert",
                     float(best_idx),
                     on_step=True,
                     on_epoch=False)
        else:
            all_outputs = self.model(inputs,
                                     target_tokens=None,
                                     return_all_experts=True)
            branch_losses = []
            for emo_pred, heat_pred, tok_pred in all_outputs:
                loss_emo, _ = self.emotion_criterion(emo_pred, emotion_labels)
                loss_heat, _ = self.heatmap_criterion(heat_pred, heatmaps)
                pred_seq_len = tok_pred.size(1)
                target_seq_len = self.model.max_comment_len
                if pred_seq_len < target_seq_len:
                    padding = torch.zeros(tok_pred.size(0),
                                          target_seq_len - pred_seq_len,
                                          tok_pred.size(2),
                                          device=tok_pred.device)
                    tok_pred = torch.cat([tok_pred, padding], dim=1)
                elif pred_seq_len > target_seq_len:
                    tok_pred = tok_pred[:, :target_seq_len, :]
                tokens_trimmed = tokens[:, :target_seq_len]
                loss_tok = self.token_criterion(
                    tok_pred.reshape(-1,
                                     self.model.vocab_size),
                    tokens_trimmed.reshape(-1)) * text_scale
                nonzero_emo = self.nonzero_reg_emo(emo_pred)
                nonzero_heat = self.nonzero_reg_heat(heat_pred)
                l1_emo = F.l1_loss(torch.sigmoid(emo_pred), emotion_labels)
                l1_heat = F.l1_loss(torch.sigmoid(heat_pred), heatmaps)
                with torch.no_grad():
                    flat_features = self.model(inputs, return_bottleneck=True)
                aux_emo_logits = self.model.aux_emo_classifier(flat_features)
                emo_any = (emotion_labels > 0.5).float().amax(dim=(2, 3, 4))
                aux_emo_loss = F.binary_cross_entropy_with_logits(
                    aux_emo_logits,
                    emo_any)
                heat_max = heatmaps.amax(dim=(2, 3, 4))
                aux_heat_pred_raw = self.model.aux_heat_regressor(
                    flat_features)
                aux_heat_loss = F.mse_loss(torch.sigmoid(aux_heat_pred_raw),
                                           heat_max)
                aux_loss = aux_emo_loss + aux_heat_loss
                voxel_loss = loss_emo + loss_heat + nonzero_emo + nonzero_heat + self.hparams.l1_weight * 0.5 * (
                    l1_emo + l1_heat) + aux_loss
                total_loss = self.voxel_loss_weight * voxel_loss + loss_tok
                branch_losses.append(total_loss)
            branch_losses = torch.stack(branch_losses)
            best_idx = torch.argmin(branch_losses).item()
            emotion_preds, heatmap_preds, token_preds = all_outputs[best_idx]
            # Recompute for logging
            loss_emo, _ = self.emotion_criterion(emotion_preds, emotion_labels)
            loss_heat, _ = self.heatmap_criterion(heatmap_preds, heatmaps)
            pred_seq_len = token_preds.size(1)
            target_seq_len = self.model.max_comment_len
            if pred_seq_len < target_seq_len:
                padding = torch.zeros(token_preds.size(0),
                                      target_seq_len - pred_seq_len,
                                      token_preds.size(2),
                                      device=token_preds.device)
                token_preds = torch.cat([token_preds, padding], dim=1)
            elif pred_seq_len > target_seq_len:
                token_preds = token_preds[:, :target_seq_len, :]
            tokens_trimmed = tokens[:, :target_seq_len]
            loss_tok = self.token_criterion(
                token_preds.reshape(-1,
                                    self.model.vocab_size),
                tokens_trimmed.reshape(-1)) * text_scale
            nonzero_emo = self.nonzero_reg_emo(emotion_preds)
            nonzero_heat = self.nonzero_reg_heat(heatmap_preds)
            l1_emo = F.l1_loss(torch.sigmoid(emotion_preds), emotion_labels)
            l1_heat = F.l1_loss(torch.sigmoid(heatmap_preds), heatmaps)
            with torch.no_grad():
                flat_features = self.model(inputs, return_bottleneck=True)
            aux_emo_logits = self.model.aux_emo_classifier(flat_features)
            emo_any = (emotion_labels > 0.5).float().amax(dim=(2, 3, 4))
            aux_emo_loss = F.binary_cross_entropy_with_logits(
                aux_emo_logits,
                emo_any)
            heat_max = heatmaps.amax(dim=(2, 3, 4))
            aux_heat_pred_raw = self.model.aux_heat_regressor(flat_features)
            aux_heat_loss = F.mse_loss(torch.sigmoid(aux_heat_pred_raw),
                                       heat_max)
            aux_loss = aux_emo_loss + aux_heat_loss
            voxel_loss = loss_emo + loss_heat + nonzero_emo + nonzero_heat + self.hparams.l1_weight * 0.5 * (
                l1_emo + l1_heat) + aux_loss
            total_loss = self.voxel_loss_weight * voxel_loss + loss_tok

        return total_loss, loss_emo, loss_heat, loss_tok, l1_emo, l1_heat, {}, {}, nonzero_emo, nonzero_heat, aux_loss

    def training_step(self, batch, batch_idx):
        loss, l_emo, l_heat, l_tok, l1_emo, l1_heat, _, _, nz_emo, nz_heat, aux_loss = self._shared_step(batch, is_training=True)
        self.log_dict(
            {
                'train_loss': loss,
                'train_emo': l_emo,
                'train_heat': l_heat,
                'train_tok': l_tok,
                'train_l1_emo': l1_emo,
                'train_l1_heat': l1_heat,
                'train_nonzero_emo': nz_emo,
                'train_nonzero_heat': nz_heat,
                'train_aux_loss': aux_loss,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch[0].size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        loss, l_emo, l_heat, l_tok, l1_emo, l1_heat, _, _, nz_emo, nz_heat, aux_loss = self._shared_step(batch, is_training=False)
        inputs, (emotion_labels, heatmaps, tokens) = batch
        emotion_labels = emotion_labels.float()
        heatmaps = heatmaps.float()
        emotion_preds, heatmap_preds, _ = self.forward(inputs, None)
        emotion_pred_binary = (torch.sigmoid(emotion_preds) > 0.5).float()
        emo_iou = (emotion_pred_binary * emotion_labels).sum() / (
            (emotion_pred_binary + emotion_labels).clamp(0,
                                                         1).sum() + 1e-6)
        heat_pred_binary = (torch.sigmoid(heatmap_preds) > 0.5).float()
        heat_target_binary = (heatmaps > 0.1).float()
        heat_iou = (heat_pred_binary * heat_target_binary).sum() / (
            (heat_pred_binary + heat_target_binary).clamp(0,
                                                          1).sum() + 1e-6)
        self.log_dict(
            {
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
                'val_emo_pred_max': torch.sigmoid(emotion_preds).max(),
                'val_heat_pred_max': torch.sigmoid(heatmap_preds).max(),
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch[0].size(0))

        if batch_idx == 0:
            emo_mean = torch.sigmoid(emotion_preds).mean().item()
            heat_mean = torch.sigmoid(heatmap_preds).mean().item()
            if emo_mean < self.emo_sparsity:
                print(
                    f"\nWARNING: Emotion pred mean = {emo_mean:.6f}. Lower than data sparsity"
                )
            if heat_mean < self.heat_sparsity:
                print(
                    f"WARNING: Heatmap pred mean = {heat_mean:.6f}. Lower than data sparsity"
                )

    def configure_optimizers(self):
        params = [
            {
                'params': self.model.encoder_blocks.parameters(),
                'lr': self.hparams.learning_rate
            },
            {
                'params': self.model.decoder_blocks.parameters(),
                'lr': self.hparams.learning_rate
            },
            {
                'params': self.model.skip_conv_blocks.parameters(),
                'lr': self.hparams.learning_rate
            },
            {
                'params': self.model.emotion_head.parameters(),
                'lr': self.hparams.learning_rate * 1.5
            },
            {
                'params': self.model.heatmap_head.parameters(),
                'lr': self.hparams.learning_rate * 1.5
            },
            {
                'params': self.model.token_embedding.parameters(),
                'lr': self.hparams.learning_rate * 0.5
            },
            {
                'params': self.model.transformer_decoder.parameters(),
                'lr': self.hparams.learning_rate
            },
            {
                'params': self.model.output_projection.parameters(),
                'lr': self.hparams.learning_rate
            },
            {
                'params': self.model.context_projection.parameters(),
                'lr': self.hparams.learning_rate
            },
            {
                'params': self.model.context_to_embed.parameters(),
                'lr': self.hparams.learning_rate
            },
            {
                'params': self.model.aux_emo_classifier.parameters(),
                'lr': self.hparams.learning_rate * 0.8
            },
            {
                'params': self.model.aux_heat_regressor.parameters(),
                'lr': self.hparams.learning_rate * 0.8
            },
            {
                'params': self.model.experts.parameters(),
                'lr': self.hparams.learning_rate * 1.5
            },
        ]
        optimizer = torch.optim.AdamW(params, weight_decay=0.01)
        linear_warmup = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                          start_factor=0.01,
                                                          total_iters=50)
        cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [linear_warmup,
             cosine],
            milestones=[50])
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }

    def on_validation_start(self):
        if hasattr(self.trainer.datamodule, 'val_paths'):
            self.val_data_paths = self.trainer.datamodule.val_paths
        else:
            self.val_data_paths = []


# ======================
# MAIN (UNCHANGED)
# ======================

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
        print(
            "\nERROR: No data files found after filtering. Please check paths and data."
        )
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
                EarlyStopping(monitor='val_loss',
                              patience=EARLY_STOPPING_PATIENCE,
                              mode='min',
                              verbose=True),
                ModelCheckpoint(monitor='val_loss',
                                save_top_k=3,
                                every_n_epochs=20,
                                save_last=True,
                                mode='min',
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
            torchinfo.summary(model,
                              input_size=(BATCH_SIZE,
                                          3,
                                          VOXEL_RESOLUTION,
                                          VOXEL_RESOLUTION,
                                          VOXEL_RESOLUTION))
        except Exception as e:
            print(f"torchinfo summary failed: {e}")

        if VISUALIZE_SAMPLES:
            show_n_samples(datamodule, NUM_SAMPLES)

        print("\nChecking Data Sparsity")
        print(
            f"Emotion target sparsity: {actual_emo_sparsity:.6f} ({actual_emo_sparsity*100:.4f}%)"
        )
        print(
            f"Heatmap target sparsity: {actual_heat_sparsity:.6f} ({actual_heat_sparsity*100:.4f}%)"
        )
        if emo_labels.sum() == 0:
            print("CRITICAL: Emotion labels in batch are ALL ZERO.")
        if heat_labels.sum() == 0:
            print("CRITICAL: Heatmap labels in batch are ALL ZERO.")

        trainer.fit(lightning_module, datamodule=datamodule)
        print("\n--- Training Finished ---")
