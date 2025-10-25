import os
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import open3d as o3d

import neologdn
from sudachipy import dictionary as sudachi_dictionary
from sudachipy import tokenizer as sudachi_tokenizer

from dataset.utils import DEFAULT_QNA_ANSWER_COLOR_MAP
from model.config import VOXEL_RESOLUTION


# yapf: disable
# Module-level tokenizer (initialized once, not tied to instance)
_sudachi_tokenizer_instance = None
_sudachi_mode = None

def _get_sudachi_tokenizer():
    global _sudachi_tokenizer_instance, _sudachi_mode
    if _sudachi_tokenizer_instance is None:
        _sudachi_tokenizer_instance = sudachi_dictionary.Dictionary().create()
        _sudachi_mode = sudachi_tokenizer.Tokenizer.SplitMode.A
    return _sudachi_tokenizer_instance, _sudachi_mode

class SimpleTokenizer:
    """Picklable Japanese tokenizer using SudachiPy — keeps ALL tokens"""

    def __init__(self, max_len=50):
        self.word_to_idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)
        self.max_len = max_len

    def _tokenize_ja(self, text):
        normalized = neologdn.normalize(text)
        tokenizer, mode = _get_sudachi_tokenizer()
        tokens = [
            m.normalized_form()
            for m in tokenizer.tokenize(normalized, mode)
        ]
        return tokens

    def build_vocab(self, sentences):
        longest = 0
        total_length = 0
        for sentence in sentences:
            words = self._tokenize_ja(sentence)
            longest = max(longest, len(words))
            total_length += len(words)
            for word in words:
                if word and word not in self.word_to_idx:
                    idx = len(self.word_to_idx)
                    self.word_to_idx[word] = idx
                    self.idx_to_word[idx] = word

        self.vocab_size = len(self.word_to_idx)
        print(f"Tokens: {self.word_to_idx}")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Longest sentence (in tokens): {longest}")
        print(f"Average length: {total_length/len(sentences)}")

    def tokenize(self, sentence):
        words = self._tokenize_ja(sentence)
        tokens = [
            self.word_to_idx.get(word, self.word_to_idx['<unk>'])
            for word in words
        ]
        tokens = [self.word_to_idx['<sos>']] + tokens + [self.word_to_idx['<eos>']]

        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens += [self.word_to_idx['<pad>']] * (self.max_len - len(tokens))

        return torch.tensor(tokens, dtype=torch.long)

# class SimpleTokenizer:
#     """Maybe replace with uni-dict or other in the future"""

#     def __init__(self, max_len=50):
#         # <pad>: Padding for shorter sentences
#         # <sos>: Start of sentence
#         # <eos>: End of sentence
#         # <unk>: Unknown word placeholder
#         self.word_to_idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
#         self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
#         self.vocab_size = len(self.word_to_idx)
#         self.max_len = max_len

#     def build_vocab(self, sentences):
#         """Builds vocabulary from sentences, to include all words from training and validation"""
#         longest = 0

#         for sentence in sentences:
#             words = sentence.lower().split()

#             longest = max(longest, len(words))

#             for word in words:
#                 if word not in self.word_to_idx:
#                     idx = len(self.word_to_idx)
#                     self.word_to_idx[word] = idx
#                     self.idx_to_word[idx] = word
#         self.vocab_size = len(self.word_to_idx)
#         print(self.word_to_idx)
#         print(f"Longest sentence: {longest}")

#     def tokenize(self, sentence):
#         tokens = [
#             self.word_to_idx.get(word, self.word_to_idx['<unk>'])
#             for word in sentence.lower().split()
#         ]
#         tokens = [
#             [self.word_to_idx['<sos>']] +
#             tokens +
#             [self.word_to_idx['<eos>']]
#         ]

#         padded_tokens = tokens[:self.max_len]
#         padded_tokens += [self.word_to_idx['<pad>']] * (self.max_len - len(padded_tokens))

#         return torch.tensor(padded_tokens, dtype=torch.long)
# yapf: enable


def voxel_grid_to_point_cloud(voxel_grid_tensor,
                              intensity_threshold=0.001,
                              reference_pcd_path=None,
                              mask_tensor=None,
                              fixed_color_rgb=None,
                              colormap_name=None):
    """
    Convert a voxel grid tensor to an Open3D point cloud.
    
    Args:
        voxel_grid_tensor: Input voxel grid (C, D, H, W) or (D, H, W)
        intensity_threshold: Minimum intensity to include voxel
        reference_pcd_path: Optional path to reference point cloud for scaling
        mask_tensor: Optional mask to filter voxels
        fixed_color_rgb: Optional fixed RGB color [0-1 or 0-255]
        colormap_name: Optional colormap name (e.g., 'jet')
    """
    if voxel_grid_tensor.is_cuda:
        voxel_grid_tensor = voxel_grid_tensor.cpu()

    # Handle different input shapes
    if len(voxel_grid_tensor.shape) == 4:  # (C, D, H, W)
        # For RGB input, convert to (D, H, W, C) for easier processing
        voxel_grid = voxel_grid_tensor.permute(1, 2, 3, 0).numpy()
        has_channels = True
    else:  # (D, H, W)
        voxel_grid = voxel_grid_tensor.numpy()
        has_channels = False

    # Handle pottery mask
    pottery_mask = None
    if mask_tensor is not None:
        if mask_tensor.is_cuda:
            mask_tensor = mask_tensor.cpu()
        mask = mask_tensor.squeeze().numpy()
        if len(mask.shape) == 4:
            mask = mask.max(axis=0)
        if len(mask.shape) == 3:
            pottery_mask = mask > 0.01

    # Determine valid voxels based on intensity
    if has_channels:
        # For multi-channel, use sum across channels
        intensity_map = voxel_grid.sum(axis=-1)
    else:
        intensity_map = voxel_grid

    if intensity_threshold < 0:
        # When threshold is negative, show all voxels in the mask
        if pottery_mask is not None:
            points_indices = np.argwhere(pottery_mask)
        else:
            # Show all voxels with any value
            points_indices = np.argwhere(intensity_map >= 0)
    else:
        if pottery_mask is not None:
            valid_voxels = (intensity_map > intensity_threshold) & pottery_mask
        else:
            valid_voxels = intensity_map > intensity_threshold
        points_indices = np.argwhere(valid_voxels)

    if points_indices.shape[0] == 0:
        return o3d.geometry.PointCloud()

    resolution = voxel_grid.shape[0]

    # Scale and position points - convert to float64 BEFORE arithmetic
    points_indices_float = points_indices.astype(np.float64)

    if reference_pcd_path and os.path.exists(reference_pcd_path):
        try:
            ref_pcd = o3d.io.read_point_cloud(reference_pcd_path)
            if ref_pcd.has_points():
                min_bound = ref_pcd.get_min_bound()
                max_bound = ref_pcd.get_max_bound()
                scale = np.max(max_bound - min_bound)
                normalized_points = points_indices_float / (resolution - 1)
                points = normalized_points * scale + min_bound
            else:
                points = points_indices_float / (resolution - 1.0) - 0.5
        except Exception as e:
            print(f"Warning: Failed to use reference PCD: {e}")
            points = points_indices_float / (resolution - 1.0) - 0.5
    else:
        points = points_indices_float / (resolution - 1.0) - 0.5

    # Ensure points are contiguous and float64
    points = np.ascontiguousarray(points, dtype=np.float64)

    # Prepare colors
    if has_channels and voxel_grid.shape[
            -1] == 3 and fixed_color_rgb is None and colormap_name is None:
        # Use the actual RGB values from the voxel grid
        colors = voxel_grid[points_indices[:,
                                           0],
                            points_indices[:,
                                           1],
                            points_indices[:,
                                           2]]
        colors = np.clip(colors, 0, 1)
    else:
        # Extract intensities for coloring
        if has_channels:
            intensities = intensity_map[points_indices[:,
                                                       0],
                                        points_indices[:,
                                                       1],
                                        points_indices[:,
                                                       2]]
        else:
            intensities = voxel_grid[points_indices[:,
                                                    0],
                                     points_indices[:,
                                                    1],
                                     points_indices[:,
                                                    2]]

        if fixed_color_rgb is not None:
            norm_intensities = np.clip(intensities, 0, 1)

            # Simple binary coloring: if intensity > 0, use target color; else use black
            color_normalized = np.array(fixed_color_rgb, dtype=np.float64)
            if color_normalized.max() > 1.0:
                color_normalized = color_normalized / 255.0

            black_color = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            colors = np.zeros((points.shape[0], 3), dtype=np.float64)
            for idx in range(points.shape[0]):
                if norm_intensities[idx] > 0.5:
                    colors[idx] = color_normalized
                else:
                    colors[idx] = black_color

        elif colormap_name == 'jet':
            try:
                import matplotlib.cm as cm
                jet_map = cm.get_cmap('jet')
                max_val = np.max(intensities)
                norm_intensities = np.array(intensities) / max_val
                colors = jet_map(norm_intensities)[:, :3]
            except ImportError:
                colors = np.tile(norm_intensities[:, None], (1, 3))
        else:
            colors = np.tile(norm_intensities[:, None], (1, 3))

    # Ensure colors are contiguous and float64
    colors = np.ascontiguousarray(colors, dtype=np.float64)

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


class SavePredictionCallback(pl.Callback):

    def __init__(self,
                 save_dir="",
                 save_every_n_epochs=10,
                 max_samples_to_save=20,
                 emotion_order=[
                     "面白い・気になる形だ",
                     "美しい・芸術的だ",
                     "不思議・意味不明",
                     "不気味・不安・怖い",
                     "何も感じない"
                 ]):
        super().__init__()
        self.save_dir = save_dir
        self.save_every_n_epochs = save_every_n_epochs
        self.max_samples_to_save = max_samples_to_save
        self.emotion_order = emotion_order
        os.makedirs(self.save_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1
            ) % self.save_every_n_epochs != 0 or not trainer.is_global_zero:
            return
        print(f"\nSaving predictions for epoch {epoch + 1}...")
        epoch_dir = os.path.join(self.save_dir, f"epoch_{epoch + 1}")
        os.makedirs(epoch_dir, exist_ok=True)
        val_loader = trainer.datamodule.val_dataloader()
        val_data_paths = pl_module.val_data_paths
        if not val_data_paths:
            print(
                "Error: Could not retrieve validation data paths. Cannot save predictions."
            )
            return
        pl_module.eval()
        sample_count = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if sample_count >= self.max_samples_to_save:
                    break
                inputs, (emotion_labels, heatmaps, tokens) = batch
                inputs = inputs.to(pl_module.device)
                emotion_labels = emotion_labels.to(pl_module.device)
                heatmaps = heatmaps.to(pl_module.device)
                tokens = tokens.to(pl_module.device)

                # Get all expert outputs
                all_outputs = pl_module.model(inputs,
                                              target_tokens=None,
                                              return_all_experts=True)

                # Compute total loss per expert to pick best
                branch_losses = []
                current_epoch = trainer.current_epoch
                text_scale = 0.1 + 0.9 * min(current_epoch / 50.0, 1.0)

                for emo_pred, heat_pred, tok_pred in all_outputs:
                    loss_emo, _ = pl_module.emotion_criterion(emo_pred, emotion_labels)
                    loss_heat, _ = pl_module.heatmap_criterion(heat_pred, heatmaps)

                    pred_seq_len = tok_pred.size(1)
                    target_seq_len = pl_module.model.max_comment_len
                    if pred_seq_len < target_seq_len:
                        padding = torch.zeros(tok_pred.size(0),
                                              target_seq_len - pred_seq_len,
                                              tok_pred.size(2),
                                              device=tok_pred.device)
                        tok_pred = torch.cat([tok_pred, padding], dim=1)
                    elif pred_seq_len > target_seq_len:
                        tok_pred = tok_pred[:, :target_seq_len, :]
                    tokens_trimmed = tokens[:, :target_seq_len]
                    loss_tok = pl_module.token_criterion(
                        tok_pred.reshape(-1,
                                         pl_module.model.vocab_size),
                        tokens_trimmed.reshape(-1)) * text_scale

                    nonzero_emo = pl_module.nonzero_reg_emo(emo_pred)
                    nonzero_heat = pl_module.nonzero_reg_heat(heat_pred)
                    l1_emo = F.l1_loss(torch.sigmoid(emo_pred), emotion_labels)
                    l1_heat = F.l1_loss(torch.sigmoid(heat_pred), heatmaps)

                    with torch.no_grad():
                        flat_features = pl_module.model(inputs,
                                                        return_bottleneck=True)
                    aux_emo_logits = pl_module.model.aux_emo_classifier(
                        flat_features)
                    emo_any = (emotion_labels > 0.5).float().amax(dim=(2,
                                                                       3,
                                                                       4))
                    aux_emo_loss = F.binary_cross_entropy_with_logits(
                        aux_emo_logits,
                        emo_any)
                    heat_max = heatmaps.amax(dim=(2, 3, 4))
                    aux_heat_pred_raw = pl_module.model.aux_heat_regressor(
                        flat_features)
                    aux_heat_loss = F.mse_loss(
                        torch.sigmoid(aux_heat_pred_raw),
                        heat_max)
                    aux_loss = aux_emo_loss + aux_heat_loss

                    voxel_loss = loss_emo + loss_heat + nonzero_emo + nonzero_heat + pl_module.hparams.l1_weight * 0.5 * (
                        l1_emo + l1_heat) + aux_loss
                    total_loss = pl_module.voxel_loss_weight * voxel_loss + loss_tok
                    branch_losses.append(total_loss)

                branch_losses = torch.stack(branch_losses)
                best_idx = torch.argmin(branch_losses).item()
                emotion_preds, heatmap_preds, token_preds = all_outputs[best_idx]
                batch_size = inputs.size(0)
                for i in range(batch_size):
                    if sample_count >= self.max_samples_to_save:
                        break
                    dataset_idx = batch_idx * trainer.datamodule.hparams.batch_size + i
                    if dataset_idx < len(val_data_paths):
                        pottery_path = val_data_paths[dataset_idx][
                            'processed_pottery_path']
                    else:
                        pottery_path = None
                    self.save_input_pottery(inputs[i],
                                            epoch_dir,
                                            sample_count,
                                            pottery_path)
                    self.save_emotions(emotion_preds[i],
                                       emotion_labels[i],
                                       epoch_dir,
                                       sample_count,
                                       pottery_path,
                                       input_mask=inputs[i])
                    self.save_heatmap(heatmap_preds[i],
                                      heatmaps[i],
                                      epoch_dir,
                                      sample_count,
                                      pottery_path,
                                      input_mask=inputs[i])
                    self.save_caption(token_preds[i],
                                      tokens[i],
                                      trainer.datamodule.tokenizer,
                                      epoch_dir,
                                      sample_count)
                    sample_count += 1
        pl_module.train()
        print(f"Saved {sample_count} prediction samples to {epoch_dir}")

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

    def save_caption(self,
                     token_preds,
                     ground_truth_tokens,
                     tokenizer,
                     epoch_dir,
                     sample_idx):
        pred_ids_1d = torch.argmax(token_preds, dim=-1).cpu().numpy()
        gt_ids_1d = ground_truth_tokens.cpu().numpy()
        pred_words = []
        for idx in pred_ids_1d:
            if idx == 2: break
            if idx not in [0, 1]:
                word = tokenizer.idx_to_word.get(idx)
                if word is not None:
                    pred_words.append(word)
        gt_words = []
        for idx in gt_ids_1d:
            if idx == 2: break
            if idx not in [0, 1]:
                word = tokenizer.idx_to_word.get(idx)
                if word is not None:
                    gt_words.append(word)
        with open(os.path.join(epoch_dir,
                               f"sample_{sample_idx}_captions.txt"),
                  "w",
                  encoding="utf-8") as f:
            f.write(f"Ground Truth: {' '.join(gt_words)}\n")
            f.write(f"Prediction:   {' '.join(pred_words)}\n")


def show_n_samples(datamodule, num_samples_to_show):
    if len(datamodule.train_dataset) > 0:
        for i in range(min(num_samples_to_show,
                           len(datamodule.train_dataset))):
            dense_voxel_tensor, (emotion_labels, _, _) = datamodule.train_dataset[i]
            if dense_voxel_tensor.sum() == 0:
                print(
                    f"Sample {i+1}/{num_samples_to_show} is empty. Skipping.")
                continue
            voxel_grid = dense_voxel_tensor.permute(1, 2, 3, 0).numpy()
            points_indices = np.argwhere(voxel_grid.sum(axis=-1) > 0.01)
            if points_indices.shape[0] == 0:
                print(
                    f"Sample {i+1}/{num_samples_to_show} has no points > 0.01. Skipping."
                )
                continue
            colors = voxel_grid[points_indices[:,
                                               0],
                                points_indices[:,
                                               1],
                                points_indices[:,
                                               2]]
            points = points_indices / (VOXEL_RESOLUTION - 1.0) - 0.5
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            print(f"Showing input sample {i+1}/{num_samples_to_show}")
            o3d.visualization.draw_geometries(
                [pcd],
                window_name=f"Input Sample {i+1}")
