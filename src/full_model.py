import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import pytorch_lightning as pl
import open3d as o3d
import neologdn
from sudachipy import dictionary as sudachi_dictionary
from sudachipy import tokenizer as sudachi_tokenizer
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import FSDPStrategy
from functools import partial
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torchinfo
import random

from dataset.utils import DEFAULT_QNA_ANSWER_COLOR_MAP, filter_data_on_condition
from dataset.voxel_dataset import ExtendedVoxelDataset

EMOTION_ORDER = ["面白い・気になる形だ", "美しい・芸術的だ", "不思議・意味不明", "不気味・不安・怖い", "何も感じない"]

RAW_DATA_DIR = "./src/jomon_kaen_dataset/japan"
MESH_DIR = "./src/pottery"
TEST_GROUPS = ['G9']
AUGMENT_COLOR_P = 0.5
COLOR_JITTER_STD = 0.05
JITTER_VOXEL_P = 0.2

BATCH_SIZE = 1
VOXEL_RESOLUTION = 512
MAX_EPOCHS = 1000
NUM_WORKERS = 2
LEARNING_RATE = 1e-4
MAX_COMMENT_LEN = 150
L1_WEIGHT = 0.001
VOXEL_LOSS_WEIGHT = 1.5

CONV_DIMS = [3, 4, 4, 8, 16, 16, 32, 32]
TEACHER_FORCING_RATIO = 0.3

SAVE_EVERY_N_EPOCHS = 20
MAX_SAMPLES_TO_SAVE = 100
SAVE_DIR = "./predictions"

EARLY_STOPPING_PATIENCE = 1000

VISUALIZE_SAMPLES = False
NUM_SAMPLES = 3

# Module-level tokenizer (initialized once, not tied to instance)
_sudachi_tokenizer_instance = None
_sudachi_mode = None


def _get_sudachi_tokenizer():
    global _sudachi_tokenizer_instance, _sudachi_mode
    if _sudachi_tokenizer_instance is None:
        try:
            _sudachi_tokenizer_instance = sudachi_dictionary.Dictionary(
            ).create()
            _sudachi_mode = sudachi_tokenizer.Tokenizer.SplitMode.A
        except Exception as e:
            print(f"Error initializing Sudachi tokenizer: {e}")
            print(
                "Please ensure SudachiPy and its dictionary are installed correctly."
            )
            raise
    return _sudachi_tokenizer_instance, _sudachi_mode


class SimpleTokenizer:
    """Picklable Japanese tokenizer using SudachiPy - keeps ALL tokens"""

    def __init__(self, max_len=50):
        self.word_to_idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)
        self.max_len = max_len

    def _tokenize_ja(self, text):
        normalized = neologdn.normalize(text)
        tokenizer, mode = _get_sudachi_tokenizer()
        tokens = [m.surface() for m in tokenizer.tokenize(normalized, mode)]
        return tokens

    def build_vocab(self, sentences):
        longest = 0
        total_length = 0
        if not sentences:
            print("Warning: No sentences provided to build vocab.")
            return

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
        print(
            f"Average length: {total_length/len(sentences) if sentences else 0}"
        )

    def tokenize(self, sentence):
        words = self._tokenize_ja(sentence)
        tokens = [
            self.word_to_idx.get(word, self.word_to_idx['<unk>'])
            for word in words
        ]
        tokens = [self.word_to_idx['<sos>']
                  ] + tokens + [self.word_to_idx['<eos>']]

        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens += [self.word_to_idx['<pad>']
                       ] * (self.max_len - len(tokens))

        return torch.tensor(tokens, dtype=torch.long)


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
        colors = voxel_grid[points_indices[:, 0], points_indices[:, 1],
                            points_indices[:, 2]]
        colors = np.clip(colors, 0, 1)
    else:
        # Extract intensities for coloring
        if has_channels:
            intensities = intensity_map[points_indices[:, 0],
                                        points_indices[:, 1],
                                        points_indices[:, 2]]
        else:
            intensities = voxel_grid[points_indices[:, 0],
                                     points_indices[:, 1], points_indices[:,
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
                max_val = np.max(intensities) if np.max(
                    intensities) > 0 else 1.0
                norm_intensities = np.array(intensities) / max_val
                colors = jet_map(norm_intensities)[:, :3]
            except ImportError:
                print("Matplotlib not found, using grayscale for heatmap.")
                norm_intensities = np.clip(intensities, 0, 1)
                colors = np.tile(norm_intensities[:, None], (1, 3))
        else:
            norm_intensities = np.clip(intensities, 0, 1)
            colors = np.tile(norm_intensities[:, None], (1, 3))

    # Ensure colors are contiguous and float64
    colors = np.ascontiguousarray(colors, dtype=np.float64)

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


class FocalLoss(nn.Module):
    """
    A simplified implementation of https://github.com/itakurah/Focal-loss-PyTorch/blob/main/focal_loss.py
    Focal Loss, a modification of cross-entropy loss designed to 
    address class imbalance by focusing on hard-to-classify examples.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs,
                                                      targets,
                                                      reduction='none')

        pt = torch.exp(-bce_loss)
        focal_weight = (1 - pt)**self.gamma

        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = focal_weight * alpha_t * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    A implementation absed on :https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
    Dice loss is used primarily for image segmentation tasks, 
    especially when dealing with imbalanced datasets where one class 
    (like the background or empty voxels in emotion maps in this case) 
    is significantly larger than others.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)

        # Frames the 3D predictions into a 2D task
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
        targets = targets.view(targets.size(0), targets.size(1), -1)

        # Calculate at the prediction values, (B, C, DHW)
        intersection = (inputs * targets).sum(dim=2)
        union = inputs.sum(dim=2) + targets.sum(dim=2)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice.mean()


class IoULoss(nn.Module):
    """
    A implementation based on: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
    Commonly used loss for segmentation tasks
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
        targets = targets.view(targets.size(0), targets.size(1), -1)

        intersection = (inputs * targets).sum(dim=2)
        total = inputs.sum(dim=2) + targets.sum(dim=2)
        union = total - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)

        return 1.0 - iou.mean()


class CombinedSparseLoss(nn.Module):

    def __init__(
        self,
        use_focal=True,
        use_dice=True,
        use_iou=True,
        focal_weight=1.0,
        dice_weight=1.0,
        iou_weight=1.0,
    ):
        super().__init__()
        self.use_focal = use_focal
        self.use_dice = use_dice
        self.use_iou = use_iou
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight

        if use_focal:
            self.focal_loss = FocalLoss(alpha=0.1, gamma=2.0)

        if use_dice:
            self.dice_loss = DiceLoss(smooth=1.0)

        if use_iou:
            self.iou_loss = IoULoss(smooth=1.0)

    def forward(self, inputs, targets):
        total_loss = 0.0
        losses = {}

        targets = targets.float()

        if self.use_focal:
            focal = self.focal_loss(inputs, targets)
            total_loss += self.focal_weight * focal
            losses['focal'] = focal

        if self.use_dice:
            dice = self.dice_loss(inputs, targets)
            total_loss += self.dice_weight * dice
            losses['dice'] = dice

        if self.use_iou:
            iou = self.iou_loss(inputs, targets)
            total_loss += self.iou_weight * iou
            losses['iou'] = iou

        losses['combined'] = total_loss

        return total_loss, losses


class NonZeroRegularization(nn.Module):

    def __init__(self, weight=1.0, target_sparsity=0.05):
        super().__init__()
        self.weight = weight
        self.target_sparsity = target_sparsity

    def forward(self, predictions):
        probs = torch.sigmoid(predictions)

        # Get the mean of each prediction in the batch
        mean_activation = probs.view(probs.size(0), -1).mean(dim=1)
        target = torch.full_like(mean_activation, self.target_sparsity)
        loss = F.mse_loss(mean_activation, target)

        return self.weight * loss


def initialize_sparse_heads(model):
    print("Initializing sparse prediction heads")
    if hasattr(model, 'emotion_head'):
        for module in model.emotion_head.modules():
            if isinstance(module, (nn.ConvTranspose3d, nn.Conv3d)):
                nn.init.xavier_normal_(module.weight, gain=0.02)
                print(
                    f"Xavier normal initialization for emotion: {module._get_name}"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.5)
                    print("Emotion head bias initialized to 0.5")

    if hasattr(model, 'heatmap_head'):
        for module in model.heatmap_head.modules():
            if isinstance(module, (nn.ConvTranspose3d, nn.Conv3d)):
                nn.init.xavier_normal_(module.weight, gain=0.2)
                print(
                    f"Xavier normal initialization for heatmap: {module._get_name}"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.5)
                    print("Heatmap haad bias initialized to 0.5")


class ConvConvEncoder(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv3d(in_dim, out_dim, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_dim), nn.ReLU(inplace=True),
            nn.Conv3d(out_dim, out_dim, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.enc(x)


class UpSampleDecoder(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_dim,
                               out_dim,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False), nn.BatchNorm3d(out_dim), nn.ReLU())

    def forward(self, x):
        return self.up(x)


class SkipBlock(nn.Module):

    def __init__(self, upsample_out_dim, skip_channels_dim):
        super().__init__()
        self.skip_block = nn.Sequential(
            nn.Conv3d(upsample_out_dim + skip_channels_dim,
                      upsample_out_dim,
                      kernel_size=3,
                      padding=1,
                      bias=False), nn.BatchNorm3d(upsample_out_dim), nn.ReLU())

    def forward(self, x):
        return self.skip_block(x)


class SavePredictionCallback(pl.Callback):

    def __init__(self,
                 save_dir="",
                 save_every_n_epochs=10,
                 max_samples_to_save=20,
                 emotion_order=None):
        super().__init__()
        self.save_dir = save_dir
        self.save_every_n_epochs = save_every_n_epochs
        self.max_samples_to_save = max_samples_to_save
        self.emotion_order = emotion_order if emotion_order is not None else []
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

                # Simplified Model Call (No Expert Logic)
                emotion_preds, heatmap_preds, token_preds, _ = pl_module.model(
                    inputs,
                    target_tokens=None,  # Inference mode
                )
                # End Simplification

                batch_size = inputs.size(0)
                for i in range(batch_size):
                    if sample_count >= self.max_samples_to_save:
                        break
                    dataset_idx = batch_idx * trainer.datamodule.hparams.batch_size + i
                    if dataset_idx < len(val_data_paths):
                        # Try to get the *original* mesh path if available, otherwise fallback to voxel path
                        pottery_path = val_data_paths[dataset_idx].get(
                            'original_mesh_path',
                            val_data_paths[dataset_idx].get(
                                'processed_pottery_path'))
                    else:
                        pottery_path = None

                    self.save_input_pottery(inputs[i], epoch_dir, sample_count,
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
                    self.save_caption(token_preds[i], tokens[i],
                                      trainer.datamodule.tokenizer, epoch_dir,
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

        # Use the utility function
        pcd = voxel_grid_to_point_cloud(input_tensor,
                                        intensity_threshold=0.01,
                                        reference_pcd_path=reference_path,
                                        mask_tensor=None,
                                        fixed_color_rgb=None,
                                        colormap_name=None)

        o3d.io.write_point_cloud(
            os.path.join(epoch_dir, f"sample_{sample_idx}_input.ply"), pcd)

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
                fixed_color_rgb=target_color_rgb)
            o3d.io.write_point_cloud(
                os.path.join(
                    epoch_dir,
                    f"sample_{sample_idx}_{safe_emotion_name}_GT.ply"), gt_pcd)

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
            os.path.join(epoch_dir, f"sample_{sample_idx}_heatmap_PRED.ply"),
            pred_pcd)

        gt_pcd = voxel_grid_to_point_cloud(
            heatmap_gt_tensor.float(),
            intensity_threshold=-1.0,  # <-- disable thresholding
            reference_pcd_path=reference_path,
            mask_tensor=input_mask,
            colormap_name='jet')
        o3d.io.write_point_cloud(
            os.path.join(epoch_dir, f"sample_{sample_idx}_heatmap_GT.ply"),
            gt_pcd)

    def save_caption(self, token_preds, ground_truth_tokens, tokenizer,
                     epoch_dir, sample_idx):
        # token_preds for inference is (Seq, Vocab), for training (Batch, Seq, Vocab)
        # The callback runs in val mode, so token_preds is (Seq, Vocab)

        # We get (Seq, Vocab) from _forward_inference which is one-hot-like
        pred_ids_1d = torch.argmax(token_preds, dim=-1).cpu().numpy()
        gt_ids_1d = ground_truth_tokens.cpu().numpy()

        pred_words = []
        for idx in pred_ids_1d:
            if idx == 2: break  # <eos>
            if idx not in [0, 1]:  # <pad>, <sos>
                word = tokenizer.idx_to_word.get(idx)
                if word is not None:
                    pred_words.append(word)

        gt_words = []
        for idx in gt_ids_1d:
            if idx == 2: break  # <eos>
            if idx not in [0, 1]:  # <pad>, <sos>
                word = tokenizer.idx_to_word.get(idx)
                if word is not None:
                    gt_words.append(word)

        with open(os.path.join(epoch_dir, f"sample_{sample_idx}_captions.txt"),
                  "w",
                  encoding="utf-8") as f:
            f.write(f"Ground Truth: {' '.join(gt_words)}\n")
            f.write(f"Prediction:   {' '.join(pred_words)}\n")


def show_n_samples(datamodule, num_samples_to_show):
    if len(datamodule.train_dataset) > 0:
        for i in range(min(num_samples_to_show,
                           len(datamodule.train_dataset))):
            dense_voxel_tensor, (emotion_labels, _,
                                 _) = datamodule.train_dataset[i]
            if dense_voxel_tensor.sum() == 0:
                print(
                    f"Sample {i+1}/{num_samples_to_show} is empty. Skipping.")
                continue

            pcd = voxel_grid_to_point_cloud(dense_voxel_tensor,
                                            intensity_threshold=0.01,
                                            reference_pcd_path=None)
            if not pcd.has_points():
                print(
                    f"Sample {i+1}/{num_samples_to_show} has no points > 0.01. Skipping."
                )
                continue

            print(f"Showing input sample {i+1}/{num_samples_to_show}")
            o3d.visualization.draw_geometries(
                [pcd], window_name=f"Input Sample {i+1}")


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
                        print(f"WARNING: Failed to read {comment_path}: {e}")

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
                 hidden_dim=64):
        super().__init__()
        self.depth = len(conv_dims) - 1
        final_size = resolution // (2**self.depth)
        flat_feature_size = conv_dims[-1] * (final_size**3)
        self.encoder_blocks = nn.ModuleList()
        for i, (in_dim,
                out_dim) in enumerate(zip(conv_dims[:-1], conv_dims[1:])):
            self.encoder_blocks.append(ConvConvEncoder(in_dim, out_dim))
        self.pool = nn.MaxPool3d(2)

        # Define full decoder paths for both streams
        up_dims = [conv_dims[-1]
                   ] + conv_dims[1:][::-1]  # e.g., [128, 64, 32, 16]
        self.fusion_layer_idx = 2  # fuse at the output of the 2nd decoder layer (0-indexed: after 2 upsamples)

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
                        nn.Conv3d(out_dim,
                                  out_dim,
                                  kernel_size=3,
                                  padding=1,
                                  bias=False), nn.BatchNorm3d(out_dim),
                        nn.ReLU()))

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
                        nn.Conv3d(out_dim,
                                  out_dim,
                                  kernel_size=3,
                                  padding=1,
                                  bias=False), nn.BatchNorm3d(out_dim),
                        nn.ReLU()))

        # Fusion at layer 2
        # After processing layer 2 (i.e., after 2 decoder blocks), we fuse
        fused_dim = up_dims[
            self.fusion_layer_idx] * 2  # concat, double channels
        self.fusion_conv_emo = nn.Sequential(
            nn.Conv3d(fused_dim,
                      up_dims[self.fusion_layer_idx],
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm3d(up_dims[self.fusion_layer_idx]), nn.ReLU())
        self.fusion_conv_heat = nn.Sequential(
            nn.Conv3d(fused_dim,
                      up_dims[self.fusion_layer_idx],
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm3d(up_dims[self.fusion_layer_idx]), nn.ReLU())

        # Heads
        self.heatmap_head = nn.ConvTranspose3d(up_dims[-1], 1, kernel_size=1)
        self.emotion_head = nn.ConvTranspose3d(up_dims[-1],
                                               num_emotions,
                                               kernel_size=1)

        # Pooling
        self.emotion_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.heatmap_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))

        visual_context_size = flat_feature_size + num_emotions + 1
        self.context_projection = nn.Sequential(
            nn.Linear(visual_context_size, hidden_dim), nn.ReLU(),
            nn.Dropout(0.3))

        # Transformer
        self.vocab_size = vocab_size
        self.max_comment_len = max_comment_len
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size,
                                            embed_dim,
                                            padding_idx=0)
        self.pos_embedding = nn.Embedding(max_comment_len, embed_dim)
        self.context_to_embed = nn.Linear(hidden_dim, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim,
                                                   nhead=8,
                                                   dim_feedforward=256,
                                                   dropout=0.3,
                                                   batch_first=True,
                                                   activation='gelu')
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer,
                                                         num_layers=2)
        self.output_projection = nn.Linear(embed_dim, vocab_size)

        # Auxiliary
        self.aux_emo_classifier = nn.Linear(flat_feature_size, num_emotions)
        self.aux_heat_regressor = nn.Linear(flat_feature_size, 1)

        # Expert Logic Removed

    def forward(self,
                x,
                target_tokens=None,
                teacher_forcing_ratio=0.5,
                return_bottleneck=False):
        batch_size = x.size(0)
        skip_features = []
        for i, block in enumerate(self.encoder_blocks):
            # APPLIED CHECKPOINTING
            x = checkpoint.checkpoint(block, x, use_reentrant=False)

            skip_features.append(x)
            x = self.pool(x)
        bottleneck_features = x
        flat_features = bottleneck_features.view(batch_size, -1)

        if return_bottleneck:
            return flat_features  # Keep this for aux_loss calculation path

        # Expert Logic Removed
        # Directly call decode_and_predict with the bottleneck

        emo, heat, tok = self._decode_and_predict(
            bottleneck_features,  # Use bottleneck directly
            skip_features,
            target_tokens)

        # Return predictions AND flat_features for aux_loss
        return emo, heat, tok, flat_features

    def _decode_and_predict(self, bottleneck, skip_features, target_tokens):
        batch_size = bottleneck.size(0)

        # Initialize both streams from bottleneck
        emo_features = bottleneck
        heat_features = bottleneck

        # Process each layer
        for layer_idx in range(len(self.emo_decoder_blocks)):
            # Upsample (with checkpointing)
            # APPLIED CHECKPOINTING
            emo_features = checkpoint.checkpoint(
                self.emo_decoder_blocks[layer_idx],
                emo_features,
                use_reentrant=False)
            heat_features = checkpoint.checkpoint(
                self.heat_decoder_blocks[layer_idx],
                heat_features,
                use_reentrant=False)

            # Apply skip connections
            skip_idx = self.depth - 1 - layer_idx
            if skip_idx >= 0 and skip_idx < len(skip_features):
                skip_emo = skip_features[skip_idx]
                skip_heat = skip_features[skip_idx]

                # Interpolate if needed
                if emo_features.shape[2:] != skip_emo.shape[2:]:
                    skip_emo = F.interpolate(skip_emo,
                                             size=emo_features.shape[2:],
                                             mode='trilinear',
                                             align_corners=True)
                if heat_features.shape[2:] != skip_heat.shape[2:]:
                    skip_heat = F.interpolate(skip_heat,
                                              size=heat_features.shape[2:],
                                              mode='trilinear',
                                              align_corners=True)

                emo_features = torch.cat([emo_features, skip_emo], dim=1)
                heat_features = torch.cat([heat_features, skip_heat], dim=1)

            # Apply skip conv blocks (with checkpointing)
            # APPLIED CHECKPOINTING
            emo_features = checkpoint.checkpoint(
                self.emo_skip_blocks[layer_idx],
                emo_features,
                use_reentrant=False)
            heat_features = checkpoint.checkpoint(
                self.heat_skip_blocks[layer_idx],
                heat_features,
                use_reentrant=False)

            # FUSION AT LAYER 2 (after processing layer_idx == 1, i.e., at start of layer 2 output)
            if layer_idx == self.fusion_layer_idx - 1:  # because we just finished layer 1, now at layer 2 output
                # Concatenate cross-stream features
                fused_emo = torch.cat([emo_features, heat_features], dim=1)
                fused_heat = torch.cat([heat_features, emo_features],
                                       dim=1)  # same content

                # Project back to original channel size
                emo_features = self.fusion_conv_emo(fused_emo)
                heat_features = self.fusion_conv_heat(fused_heat)

        # Final predictions
        emotion_preds = self.emotion_head(emo_features)
        heatmap_preds = self.heatmap_head(heat_features)

        # Context for language (unchanged)
        emotion_context = self.emotion_pooling(emotion_preds.detach()).view(
            batch_size, -1)
        heatmap_context = self.heatmap_pooling(heatmap_preds.detach()).view(
            batch_size, -1)
        flat_features = bottleneck.view(batch_size, -1)
        visual_features = torch.cat(
            [flat_features, emotion_context, heatmap_context], dim=1)
        visual_context = self.context_projection(visual_features)
        context_embed = self.context_to_embed(visual_context)

        if self.training and target_tokens is not None:
            token_preds = self._forward_train(context_embed, target_tokens)
        else:
            token_preds = self._forward_inference(context_embed)
        return emotion_preds, heatmap_preds, token_preds

    # Keep _forward_train and _forward_inference exactly as before
    def _forward_train(self, context_embed, target_tokens):
        B, T = target_tokens.shape
        device = target_tokens.device
        input_tokens = torch.cat(
            [
                torch.ones(B, 1, dtype=torch.long, device=device) *
                1,  # <SOS> token
                target_tokens[:, :-1]
            ],
            dim=1)
        token_emb = self.token_embedding(input_tokens)
        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embedding(pos_ids)
        tgt = token_emb + pos_emb
        memory = context_embed.unsqueeze(1).expand(-1, T, -1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=device)
        decoded = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        logits = self.output_projection(decoded)
        return logits

    def _forward_inference(self, context_embed):
        B = context_embed.size(0)
        device = context_embed.device
        max_len = self.max_comment_len
        generated = torch.ones(B, 1, dtype=torch.long,
                               device=device) * 1  # <SOS>
        for t in range(1, max_len):
            T_curr = generated.size(1)
            token_emb = self.token_embedding(generated)
            pos_ids = torch.arange(T_curr,
                                   device=device).unsqueeze(0).expand(B, -1)
            pos_emb = self.pos_embedding(pos_ids)
            tgt = token_emb + pos_emb
            memory = context_embed.unsqueeze(1).expand(-1, T_curr, -1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                T_curr, device=device)
            decoded = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
            logits = self.output_projection(decoded)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == 2).all():  # <EOS> token
                break

        # Pad to max_len
        if generated.size(1) < max_len:
            pad = torch.zeros(B,
                              max_len - generated.size(1),
                              dtype=torch.long,
                              device=device)  # <PAD> token is 0
            generated = torch.cat([generated, pad], dim=1)

        # Create "logits" output for validation loss calculation
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
                 voxel_loss_weight=1.0,
                 emo_loss_weight=1.0,
                 heat_loss_weight=1.0,
                 global_loss_weight=0.2):
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
        self.emo_sparsity = self.hparams.nonzero_emo_target
        self.heat_sparsity = self.hparams.nonzero_heat_target
        self.nonzero_reg_emo = NonZeroRegularization(
            weight=0.5, target_sparsity=self.hparams.nonzero_emo_target)
        self.nonzero_reg_heat = NonZeroRegularization(
            weight=0.1, target_sparsity=self.hparams.nonzero_heat_target)
        self.token_criterion = nn.CrossEntropyLoss(ignore_index=0,
                                                   label_smoothing=0.1)
        self.voxel_loss_weight = self.hparams.voxel_loss_weight
        self.val_data_paths = []

    def forward(self, x, target_tokens=None):
        return self.model(x, target_tokens)

    def _shared_step(self, batch, is_training=True):
        inputs, (emotion_labels, heatmaps, tokens) = batch
        emotion_labels = emotion_labels.float()
        heatmaps = heatmaps.float()
        if is_training:
            # Label smoothing for voxel targets
            emotion_labels = emotion_labels * 0.98 + 0.01
            heatmaps = heatmaps * 0.98 + 0.01

        current_epoch = self.current_epoch if self.trainer else 0
        text_scale = 0.1 + 0.9 * min(current_epoch / 50.0, 1.0)

        # 1. Single forward pass to get all outputs and features
        #    Expert logic removed.
        if is_training:
            emo_pred, heat_pred, tok_pred, flat_features = self.model(
                inputs,
                target_tokens=tokens,
            )
        else:
            emo_pred, heat_pred, tok_pred, flat_features = self.model(
                inputs,
                target_tokens=None,  # Inference mode
            )

        # 2. Calculate Aux Loss (for encoder) using flat_features
        aux_emo_logits = self.model.aux_emo_classifier(flat_features)

        # Use 0.1 for training label smoothing, 0.5 for val
        emo_any_threshold = 0.1 if is_training else 0.5
        emo_any = (emotion_labels
                   > emo_any_threshold).float().amax(dim=(2, 3, 4))

        aux_emo_loss = F.binary_cross_entropy_with_logits(
            aux_emo_logits, emo_any)
        heat_max = heatmaps.amax(dim=(2, 3, 4))
        aux_heat_pred_raw = self.model.aux_heat_regressor(flat_features)
        aux_heat_loss = F.mse_loss(torch.sigmoid(aux_heat_pred_raw), heat_max)
        aux_loss = aux_emo_loss + aux_heat_loss

        # 3. Calculate Losses from the single prediction
        #    (No branch_losses or best_idx)
        loss_emo, _ = self.emotion_criterion(emo_pred, emotion_labels)
        loss_heat, _ = self.heatmap_criterion(heat_pred, heatmaps)

        # Ensure token preds and targets match length
        pred_seq_len = tok_pred.size(1)
        target_seq_len = self.model.max_comment_len
        tok_target = tokens  # Use full token target

        if pred_seq_len < target_seq_len:
            padding = torch.zeros(tok_pred.size(0),
                                  target_seq_len - pred_seq_len,
                                  tok_pred.size(2),
                                  device=tok_pred.device)
            tok_pred = torch.cat([tok_pred, padding], dim=1)
        elif pred_seq_len > target_seq_len:
            tok_pred = tok_pred[:, :target_seq_len, :]

        if tok_target.size(1) > target_seq_len:
            tok_target = tok_target[:, :target_seq_len]
        elif tok_target.size(1) < target_seq_len:
            pad = torch.zeros(tok_target.size(0),
                              target_seq_len - tok_target.size(1),
                              dtype=torch.long,
                              device=tok_target.device)
            tok_target = torch.cat([tok_target, pad], dim=1)

        loss_tok = self.token_criterion(
            tok_pred.reshape(-1, self.model.vocab_size),
            tok_target.reshape(-1)) * text_scale

        nonzero_emo = self.nonzero_reg_emo(emo_pred)
        nonzero_heat = self.nonzero_reg_heat(heat_pred)
        l1_emo = F.l1_loss(torch.sigmoid(emo_pred), emotion_labels)
        l1_heat = F.l1_loss(torch.sigmoid(heat_pred), heatmaps)

        # 4. Final loss calculation (simplified)

        # INDIVIDUAL LOSSES
        weighted_emo = self.hparams.emo_loss_weight * loss_emo
        weighted_heat = self.hparams.heat_loss_weight * loss_heat

        # GLOBAL LOSS: includes token loss
        global_loss = self.hparams.global_loss_weight * (loss_emo + loss_heat +
                                                         loss_tok)

        voxel_loss = (weighted_emo + weighted_heat + global_loss +
                      nonzero_emo + nonzero_heat +
                      self.hparams.l1_weight * 0.5 * (l1_emo + l1_heat))

        # Total loss combines voxel, token, and auxiliary losses
        total_loss = self.hparams.voxel_loss_weight * voxel_loss + loss_tok + aux_loss

        # 5. Return all computed values for logging
        return total_loss, loss_emo, loss_heat, loss_tok, l1_emo, l1_heat, emo_pred, heat_pred, nonzero_emo, nonzero_heat, aux_loss

    def training_step(self, batch, batch_idx):
        loss, l_emo, l_heat, l_tok, l1_emo, l1_heat, _, _, nz_emo, nz_heat, aux_loss = self._shared_step(
            batch, is_training=True)
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
        loss, l_emo, l_heat, l_tok, l1_emo, l1_heat, emotion_preds, heatmap_preds, nz_emo, nz_heat, aux_loss = self._shared_step(
            batch, is_training=False)

        inputs, (emotion_labels, heatmaps, tokens) = batch
        emotion_labels = emotion_labels.float()
        heatmaps = heatmaps.float()

        # Calculate IoU metrics
        emotion_pred_binary = (torch.sigmoid(emotion_preds) > 0.5).float()
        emo_target_binary = (
            emotion_labels > 0.5).float()  # Use same threshold for consistency
        emo_iou = (emotion_pred_binary * emo_target_binary).sum() / (
            (emotion_pred_binary + emo_target_binary).clamp(0, 1).sum() + 1e-6)

        heat_pred_binary = (torch.sigmoid(heatmap_preds) > 0.5).float()
        heat_target_binary = (
            heatmaps > 0.1).float()  # Target is sparse, so 0.1 is fine
        heat_iou = (heat_pred_binary * heat_target_binary).sum() / (
            (heat_pred_binary + heat_target_binary).clamp(0, 1).sum() + 1e-6)

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

        if batch_idx == 0 and self.global_rank == 0:  # Only print on rank 0
            emo_mean = torch.sigmoid(emotion_preds).mean().item()
            heat_mean = torch.sigmoid(heatmap_preds).mean().item()
            if emo_mean < self.emo_sparsity:
                print(
                    f"\n[Rank {self.global_rank}] WARNING: Emotion pred mean = {emo_mean:.6f}. Lower than data sparsity {self.emo_sparsity:.6f}"
                )
            if heat_mean < self.heat_sparsity:
                print(
                    f"[Rank {self.global_rank}] WARNING: Heatmap pred mean = {heat_mean:.6f}. Lower than data sparsity {self.heat_sparsity:.6f}"
                )

    def configure_optimizers(self):
        params = [
            {
                'params': self.model.encoder_blocks.parameters(),
                'lr': self.hparams.learning_rate
            },
            {
                'params': self.model.emo_decoder_blocks.parameters(),
                'lr': self.hparams.learning_rate
            },
            {
                'params': self.model.heat_decoder_blocks.parameters(),
                'lr': self.hparams.learning_rate
            },
            {
                'params': self.model.emo_skip_blocks.parameters(),
                'lr': self.hparams.learning_rate
            },
            {
                'params': self.model.heat_skip_blocks.parameters(),
                'lr': self.hparams.learning_rate
            },
            {
                'params': self.model.fusion_conv_emo.parameters(),
                'lr': self.hparams.learning_rate
            },
            {
                'params': self.model.fusion_conv_heat.parameters(),
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
            # Expert Params Removed
        ]
        optimizer = torch.optim.AdamW(params, weight_decay=0.01)

        linear_warmup_epochs = 20  # Warmup for 5 epochs

        linear_warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=linear_warmup_epochs)

        # Start cosine decay *after* warmup
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS - linear_warmup_epochs, eta_min=1e-6)

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [linear_warmup, cosine_scheduler],
            milestones=[linear_warmup_epochs
                        ]  # Switch schedulers after 5 epochs
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'  # Step scheduler every epoch
            }
        }

    def on_validation_start(self):
        if hasattr(self.trainer.datamodule, 'val_paths'):
            self.val_data_paths = self.trainer.datamodule.val_paths
        else:
            self.val_data_paths = []


if __name__ == "__main__":

    # Set environment variables for CUDA debugging if needed
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Check for placeholder data paths
    if "D:\\storage" in RAW_DATA_DIR:
        print("Warning: Using placeholder data paths.")
        print("This script will create dummy data for testing.")

    print("Starting Pottery Analysis Training (Efficient Version)")

    # Data Loading
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
            "\nERROR: No data files found. Please check paths and data filtering logic."
        )
    else:
        pl.seed_everything(42)
        torch.set_float32_matmul_precision('high')

        # Data Module
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

        # Sparsity Check
        try:
            sample_batch = next(iter(datamodule.train_dataloader()))
            _, (emo_labels, heat_labels, _) = sample_batch
            actual_emo_sparsity = emo_labels.float().mean().item()
            actual_heat_sparsity = heat_labels.float().mean().item()
            print(f"Actual emotion sparsity: {actual_emo_sparsity:.6f}")
            print(f"Actual heatmap sparsity: {actual_heat_sparsity:.6f}")
            target_emo = min(0.02, max(0.005, actual_emo_sparsity * 2))
            target_heat = min(0.02, max(0.005, actual_heat_sparsity * 2))
        except StopIteration:
            print(
                "Error: Training dataloader is empty. Cannot check sparsity.")
            actual_emo_sparsity = 0.01
            actual_heat_sparsity = 0.01
            target_emo = 0.01
            target_heat = 0.01

        # Model Initialization (Expert logic removed)
        model = MeaningMakingModel(num_emotions=len(EMOTION_ORDER),
                                   vocab_size=datamodule.tokenizer.vocab_size,
                                   max_comment_len=MAX_COMMENT_LEN,
                                   conv_dims=CONV_DIMS,
                                   resolution=VOXEL_RESOLUTION)

        lightning_module = MeaningMakingLightningModule(
            model,
            learning_rate=LEARNING_RATE,
            l1_weight=L1_WEIGHT,
            teacher_forcing_ratio=TEACHER_FORCING_RATIO,
            nonzero_emo_target=target_emo,
            nonzero_heat_target=target_heat,
            voxel_loss_weight=VOXEL_LOSS_WEIGHT)

        # Callbacks
        prediction_saver = SavePredictionCallback(
            save_dir=SAVE_DIR,
            save_every_n_epochs=SAVE_EVERY_N_EPOCHS,
            max_samples_to_save=MAX_SAMPLES_TO_SAVE,
            emotion_order=EMOTION_ORDER)

        # FSDP Strategy (Expert logic removed from policy)
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                ConvConvEncoder,
                UpSampleDecoder,
                SkipBlock,
                nn.
                TransformerDecoderLayer,  # Wraps sub-layers of nn.TransformerDecoder
            })

        # Trainer
        # Determine accelerator and devices
        if torch.cuda.is_available():
            accelerator = "gpu"
            devices = 4
            precision = '16-mixed'
            strategy = FSDPStrategy(
                auto_wrap_policy=auto_wrap_policy,
                # Use 'sharded' instead of 'full' for potentially better memory
                sharding_strategy='SHARD_GRAD_OP',
                cpu_offload=False  # Offload params to CPU to save GPU RAM
            )
            print(f"Using {devices} GPUs with FSDP (CPU Offload).")
        else:
            accelerator = "cpu"
            devices = 1
            precision = 32
            strategy = "auto"
            print("No GPU found. Training on CPU.")

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
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            accumulate_grad_batches=
            4,  # Accumulate gradients to simulate larger batch
            precision=precision,
            gradient_clip_val=None)

        # Model Summary
        # print("Running torchinfo summary")
        # try:
        #     torchinfo.summary(model,
        #                       input_size=(BATCH_SIZE, 3, VOXEL_RESOLUTION,
        #                                   VOXEL_RESOLUTION, VOXEL_RESOLUTION))
        # except Exception as e:
        #     print(f"torchinfo summary failed: {e}")

        # Visualize Samples
        if VISUALIZE_SAMPLES:
            show_n_samples(datamodule, NUM_SAMPLES)

        # Final Data Check
        print("\nChecking Data Sparsity")
        print(
            f"Emotion target sparsity: {actual_emo_sparsity:.6f} ({actual_emo_sparsity*100:.4f}%)"
        )
        print(
            f"Heatmap target sparsity: {actual_heat_sparsity:.6f} ({actual_heat_sparsity*100:.4f}%)"
        )
        if 'emo_labels' in locals() and emo_labels.sum() == 0:
            print("CRITICAL: Emotion labels in first batch are ALL ZERO.")
        if 'heat_labels' in locals() and heat_labels.sum() == 0:
            print("CRITICAL: Heatmap labels in first batch are ALL ZERO.")

        # Start Training
        print("\n--- Starting Training")
        trainer.fit(lightning_module, datamodule=datamodule)
        print("\n--- Training Finished")
