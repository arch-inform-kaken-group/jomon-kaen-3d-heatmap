import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
import random

# Config (keep your existing)
EMOTION_ORDER = ["面白い・気になる形だ", "美しい・芸術的だ", "不思議・意味不明", "不気味・不安・怖い", "何も感じない"]
RAW_DATA_DIR = r"D:\storage\jomon_kaen\jomon_kaen_dataset\japan"
MESH_DIR = r"D:\storage\jomon_kaen\pottery"
TEST_GROUPS = ['G9']
BATCH_SIZE = 8
VOXEL_RESOLUTION = 80
MAX_EPOCHS = 1000
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
L1_WEIGHT = 0.001
NONZERO_EMO_TARGET = 0.005
NONZERO_GAZE_TARGET = 0.01
SAVE_DIR = r"D:\storage\jomon_kaen\dsvt_full"
EARLY_STOPPING_PATIENCE = 1000
MAX_COMMENT_LEN = 150
NUM_EMOTIONS = len(EMOTION_ORDER)

class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class DSVTLayer(nn.Module):
    def __init__(self, embed_dim, axis='x', tau=36):
        super().__init__()
        self.embed_dim = embed_dim
        self.axis = axis
        self.tau = tau
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        if tau == 216:
            self.local_shape = (6, 6, 6)
        elif tau == 36:
            self.local_shape = (3, 3, 4)
        else:
            self.local_shape = (1, 1, tau)
        d_l, h_l, w_l = self.local_shape
        assert d_l * h_l * w_l == tau
        self.local_cnn = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim),
            nn.BatchNorm3d(embed_dim),
            nn.Conv3d(embed_dim, embed_dim, kernel_size=1),
            nn.GELU()
        )
    def forward(self, tokens, coords, batch_idx, D, H, W):
        if tokens.shape[0] == 0:
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
        N = tokens.shape[0]
        S = (N + self.tau - 1) // self.tau
        padded_N = S * self.tau
        pad_len = padded_N - N
        if pad_len > 0:
            tokens_pad = torch.zeros(pad_len, self.embed_dim, device=device)
            tokens_sorted = torch.cat([tokens_sorted, tokens_pad], dim=0)
        tokens_sets = tokens_sorted.view(S, self.tau, self.embed_dim)
        d_l, h_l, w_l = self.local_shape
        tokens_3d = tokens_sets.transpose(1, 2).reshape(S, self.embed_dim, d_l, h_l, w_l)
        tokens_cnn = self.local_cnn(tokens_3d)
        tokens_cnn = tokens_cnn.reshape(S, self.embed_dim, self.tau).transpose(1, 2)
        tokens_sets = tokens_sets + tokens_cnn
        tokens_norm = self.norm1(tokens_sets)
        attn_out, _ = self.attn(tokens_norm, tokens_norm, tokens_norm)
        tokens_sets = tokens_sets + attn_out
        tokens_sets = tokens_sets + self.mlp(self.norm2(tokens_sets))
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
        logits_dense = torch.zeros(B * D * H * W, self.out_channels, device=logits_sparse.device)
        logits_dense[coords] = logits_sparse
        logits_dense = logits_dense.view(B, D, H, W, self.out_channels).permute(0, 4, 1, 2, 3)
        return logits_dense, tokens_dec

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_weight = (1 - pt) ** self.gamma
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

from transformers import BertConfig, BertForMaskedLM
from model.helper import SimpleTokenizer

class DSVTWithBERTMLM(pl.LightningModule):
    def __init__(self, vocab_size, max_comment_len=50):
        super().__init__()
        self.save_hyperparameters()
        
        # DSVT Backbone
        self.backbone = DSVTBackbone(
            in_channels=3,
            embed_dim=32,
            resolution=VOXEL_RESOLUTION,
            tau=216,
            num_layers=6
        )
        self.decoder = DSVTDecoder(
            embed_dim=32,
            out_channels=6,
            resolution=VOXEL_RESOLUTION,
            tau=216,
            num_layers=6
        )
        
        # Global feature - BERT context
        self.feature_to_bert = nn.Linear(32, 256)  # Match BERT hidden size
        
        # Mini BERT for MLM (trained from scratch)
        bert_config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=512,
            max_position_embeddings=max_comment_len,
            pad_token_id=0,  # <pad>
            bos_token_id=1,  # <sos>
            eos_token_id=2,  # <eos>
            cls_token_id=1,
            sep_token_id=2,
            mask_token_id=3, # <unk> or add <mask>
        )
        self.bert_mlm = BertForMaskedLM(config=bert_config)
        
        # Losses
        self.focal_loss = FocalLoss(alpha=0.2, gamma=2.0)
        self.dice_loss = DiceLoss()
        self.nonzero_emo = NonZeroRegularization(weight=0.1, target_sparsity=NONZERO_EMO_TARGET)
        self.nonzero_gaze = NonZeroRegularization(weight=0.05, target_sparsity=NONZERO_GAZE_TARGET)
        self.mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.val_data_paths = []

    def forward(self, x, input_ids=None, attention_mask=None, labels=None):
        B = x.shape[0]
        enc_out = self.backbone(x)
        if enc_out[0] is None:
            emo = torch.zeros(B, 5, VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION, device=x.device)
            gaze = torch.zeros(B, 1, VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION, device=x.device)
            mlm_loss = torch.tensor(0.0, device=x.device)
            return emo, gaze, mlm_loss

        # Emotion/Gaze decoding
        tokens_enc, coords, batch_idx, xyz, dims, _, early_features = enc_out
        dec_logits, tokens_dec = self.decoder(tokens_enc, coords, batch_idx, xyz, dims, B, early_features)
        emo = dec_logits[:, :5]
        gaze = dec_logits[:, 5:6]

        # Global feature
        global_feats = []
        for b in range(B):
            mask_b = (batch_idx == b)
            if not mask_b.any():
                feat_b = torch.zeros(32, device=x.device)
            else:
                tokens_b = tokens_dec[mask_b].mean(dim=0)  # Simple mean pooling
                feat_b = tokens_b
            global_feats.append(feat_b)
        global_feats = torch.stack(global_feats)  # [B, 32]
        bert_context = self.feature_to_bert(global_feats)  # [B, 256]

        # BERT MLM forward
        mlm_loss = torch.tensor(0.0, device=x.device)
        if input_ids is not None and self.training:
            # Inject visual context as additional embedding
            inputs_embeds = self.bert_mlm.bert.embeddings.word_embeddings(input_ids)
            # Add context to [CLS] or all tokens
            context_embeds = bert_context.unsqueeze(1).expand(-1, inputs_embeds.size(1), -1)
            inputs_embeds = inputs_embeds + context_embeds
            outputs = self.bert_mlm(inputs_embeds=inputs_embeds, labels=labels)
            mlm_loss = outputs.loss

        return emo, gaze, mlm_loss

    def _shared_step(self, batch, stage):
        inputs, (emo_labels, gaze_labels, tokens) = batch
        emo_labels = emo_labels.float()
        gaze_labels = gaze_labels.float()
        
        # Prepare MLM inputs (dynamic masking)
        if self.training:
            input_ids, labels = self._mask_tokens(tokens)
        else:
            input_ids, labels = tokens, tokens  # No masking at val

        emo_pred, gaze_pred, mlm_loss = self(inputs, input_ids=input_ids, labels=labels)

        # Emotion/Gaze losses
        loss_emo_focal = self.focal_loss(emo_pred, emo_labels)
        loss_emo_dice = self.dice_loss(emo_pred, emo_labels)
        loss_emo_l1 = F.l1_loss(torch.sigmoid(emo_pred), emo_labels)
        loss_emo_reg = self.nonzero_emo(emo_pred)

        loss_gaze_focal = self.focal_loss(gaze_pred, gaze_labels)
        loss_gaze_dice = self.dice_loss(gaze_pred, gaze_labels)
        loss_gaze_l1 = F.l1_loss(torch.sigmoid(gaze_pred), gaze_labels)
        loss_gaze_reg = self.nonzero_gaze(gaze_pred)

        total = (
            0.5 * loss_emo_focal + 1.0 * loss_emo_dice + L1_WEIGHT * loss_emo_l1 + loss_emo_reg +
            1.0 * loss_gaze_focal + 0.5 * loss_gaze_dice + L1_WEIGHT * loss_gaze_l1 + loss_gaze_reg +
            mlm_loss  # Add MLM loss
        )

        self.log_dict({
            f"{stage}_loss": total,
            f"{stage}_emo_focal": loss_emo_focal,
            f"{stage}_gaze_focal": loss_gaze_focal,
            f"{stage}_mlm": mlm_loss,
        }, on_epoch=True, prog_bar=(stage == 'val'))
        return total

    def _mask_tokens(self, inputs, mask_prob=0.15):
        """Prepare masked tokens for MLM."""
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, mask_prob)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Ignore non-masked

        # 80% mask, 10% random, 10% unchanged
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = 3  # <mask> or <unk> (your mask token ID)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(4, self.hparams.vocab_size, labels.shape, device=inputs.device)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, 'val')
        # Optional: run iterative refinement on first sample for logging
        if batch_idx == 0 and self.global_rank == 0:
            self._log_iterative_caption(batch)
        return loss

    @torch.no_grad()
    def _log_iterative_caption(self, batch):
        """Run iterative MLM on first validation sample."""
        inputs, (_, _, tokens) = batch
        x = inputs[:1].to(self.device)
        ref_text = self.trainer.datamodule.tokenizer.decode(tokens[0].cpu().numpy())
        print(f"\n[Iterative Captioning] Reference: {ref_text}")

        # Get global feature
        enc_out = self.backbone(x)
        if enc_out[0] is None: return
        tokens_enc, coords, batch_idx, xyz, dims, _, early_features = enc_out
        _, tokens_dec = self.decoder(tokens_enc, coords, batch_idx, xyz, dims, 1, early_features)
        feat = tokens_dec.mean(dim=0, keepdim=True)  # [1, 32]
        bert_context = self.feature_to_bert(feat)  # [1, 256]

        # Iterative refinement
        max_len = self.hparams.max_comment_len
        unk_id = 3
        candidates = [torch.full((1, max_len), unk_id, device=self.device)]
        tokenizer = self.trainer.datamodule.tokenizer

        for step in range(20):
            new_candidates = []
            for cand in candidates:
                # Mask random positions
                mask = torch.rand_like(cand.float()) < 0.3
                masked_cand = cand.clone()
                masked_cand[mask] = 3  # <mask>
                # Forward through BERT
                inputs_embeds = self.bert_mlm.bert.embeddings.word_embeddings(masked_cand)
                context_embeds = bert_context.unsqueeze(1).expand(-1, max_len, -1)
                outputs = self.bert_mlm.bert(inputs_embeds=inputs_embeds + context_embeds)
                logits = self.bert_mlm.cls(outputs.last_hidden_state)
                # Sample top-k
                preds = torch.topk(logits, 3, dim=-1).indices
                new_cand = cand.clone()
                new_cand[mask] = preds[mask][:, torch.randint(0, 3, (mask.sum(),))]
                new_candidates.append(new_cand)
            candidates = new_candidates

        pred_ids = candidates[0].squeeze().cpu().numpy()
        pred_text = tokenizer.decode([idx for idx in pred_ids if idx not in [0, 1, 2]])
        print(f"[Iterative Captioning] Prediction: {pred_text}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {'params': self.backbone.parameters(), 'lr': LEARNING_RATE},
            {'params': self.decoder.parameters(), 'lr': LEARNING_RATE},
            {'params': self.feature_to_bert.parameters(), 'lr': LEARNING_RATE},
            {'params': self.bert_mlm.parameters(), 'lr': LEARNING_RATE * 0.5},
        ], weight_decay=0.01)
        return optimizer

from dataset.voxel_dataset import ExtendedVoxelDataset
from model.helper import SimpleTokenizer, SavePredictionCallback

class DSVTDataModule(pl.LightningDataModule):
    def __init__(self, all_data_paths, batch_size, num_workers, voxel_resolution, test_groups):
        super().__init__()
        self.all_data_paths = all_data_paths
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.voxel_resolution = voxel_resolution
        self.test_groups = test_groups
        self.tokenizer = SimpleTokenizer(max_len=MAX_COMMENT_LEN)  # Needed for dataset, but unused

    def setup(self, stage=None):
        np.random.shuffle(self.all_data_paths)
        train_paths = [p for p in self.all_data_paths if p['GROUP'] not in self.test_groups]
        val_paths = [p for p in self.all_data_paths if p['GROUP'] in self.test_groups]
        # Build vocab (required by dataset, even if unused)
        comments = []
        for p in self.all_data_paths:
            if os.path.exists(p.get('TRANSCRIPT', '')):
                with open(p['TRANSCRIPT'], 'r', encoding='utf-8') as f:
                    comments.append(f.read().strip())
        self.tokenizer.build_vocab(comments)
        common_args = {'voxel_resolution': self.voxel_resolution, 'tokenizer': self.tokenizer}
        self.train_dataset = ExtendedVoxelDataset(train_paths, augment_color_p=0.5, **common_args)
        self.val_dataset = ExtendedVoxelDataset(val_paths, augment_color_p=0.0, **common_args)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)


if __name__ == "__main__":
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('high')

    # Load data
    from dataset.utils import filter_data_on_condition
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

    model = DSVTWithBERTMLM(
        vocab_size=datamodule.tokenizer.vocab_size,
        max_comment_len=MAX_COMMENT_LEN
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=2, every_n_epochs=20, save_last=True, mode='min'),
            pl.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, mode='min'),
            SavePredictionCallback(save_dir=SAVE_DIR, emotion_order=EMOTION_ORDER, save_every_n_epochs=20, max_samples_to_save=100)
        ],
        log_every_n_steps=10,
        precision='16-mixed' if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0
    )

    trainer.fit(model, datamodule=datamodule)

    print("\n=== Inference Example ===")
    model.eval()
    sample_input, _ = next(iter(datamodule.val_dataloader()))
    sample_input = sample_input[:1].to(model.device)
    with torch.no_grad():
        emo, gaze, feat = model(sample_input)
        caption = model.generate_caption_iterative(feat[0])
    print("Generated caption:", caption)