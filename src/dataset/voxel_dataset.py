import os
import numpy as np
import torch
from torch.utils.data import Dataset
import open3d as o3d

from dataset.utils import DEFAULT_QNA_ANSWER_COLOR_MAP


# yapf: disable
class ExtendedVoxelDataset(Dataset):
    """Upgraded version of the point cloud dataset
    Loads eye gaze heatmap, emotion channels, transcript
    Has data augmentation
    """

    def __init__(
        self,
        data_paths,
        voxel_resolution,
        tokenizer,
        augment_color_p=0.5,
        color_jitter_std=0.05,
        jitter_voxel_p=0.1,
        emotion_order=["面白い・気になる形だ", "美しい・芸術的だ", "不思議・意味不明", "不気味・不安・怖い", "何も感じない"]
    ):
        super().__init__()
        self.data_paths = data_paths
        self.voxel_resolution = voxel_resolution
        self.tokenizer = tokenizer

        self.augment_color_p = augment_color_p
        self.color_jitter_std = color_jitter_std
        self.jitter_voxel_p = jitter_voxel_p

        self.emotion_order = emotion_order

    def __len__(self):
        return len(self.data_paths)

    def color_jitter_pcd(self, voxel_pcd):
        if np.random.rand() < self.augment_color_p and voxel_pcd.has_colors():
            colors = np.asarray(voxel_pcd.colors).astype(np.float32)
            num_points = colors.shape[0]
            jitter_mask = np.random.rand(num_points) < self.jitter_voxel_p
            if np.any(jitter_mask):
                noise = np.random.normal(0.0,
                                         self.color_jitter_std,
                                         colors[jitter_mask].shape).astype(
                                             np.float32)
                colors[jitter_mask] += noise
                colors = np.clip(colors, 0.0, 1.0)
                voxel_pcd.colors = o3d.utility.Vector3dVector(colors)
        return voxel_pcd

    def _load_voxel_to_tensor(self, path, num_channels=3, mode="target"):
        tensor = torch.zeros((num_channels,
                              self.voxel_resolution,
                              self.voxel_resolution,
                              self.voxel_resolution),
                             dtype=torch.float32)

        if not os.path.exists(path):
            raise(FileNotFoundError(f"No file at: {path}"))

        try:
            pcd = o3d.io.read_point_cloud(path)
            if not pcd.has_points():
                raise(ValueError("Empty points"))
        except Exception as e:
            raise(ValueError(f"Fail to load points: {path}. Error: {e}"))

        if (mode == "input"):
            pcd = self.color_jitter_pcd(pcd)

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors).astype(np.float32)

        min_bound = pcd.get_min_bound()
        max_bound = pcd.get_max_bound()

        voxel_size = np.max(max_bound - min_bound) / (self.voxel_resolution - 1)

        indices = np.floor((points - min_bound) / voxel_size).astype(int)
        indices = np.clip(indices, 0, self.voxel_resolution - 1)

        if num_channels == 3:
            tensor[:, indices[:, 0], indices[:, 1], indices[:, 2]] = torch.from_numpy(colors.T)
        else:
            if colors.shape[1] == 3:
                intensity = torch.from_numpy(colors.mean(axis=1))
            elif colors.shape[1] == 1:
                intensity = torch.from_numpy(colors.squeeze())
            else:
                raise(ValueError("Intensities cannot be computed."))

            tensor[0, indices[:, 0], indices[:, 1], indices[:, 2]] = intensity

        return tensor

    def __getitem__(self, index):
        item_info = self.data_paths[index]

        pottery_voxel_path = item_info['processed_pottery_path']
        heatmap_voxel_path = item_info['eye_gaze_voxel']
        comment_path = item_info['TRANSCRIPT']
        qna_dir = item_info['qa_segmented_mesh']

        dense_voxel_tensor = self._load_voxel_to_tensor(
            pottery_voxel_path,
            num_channels=3,
            mode="input"
        )

        heatmap_tensor = self._load_voxel_to_tensor(
            heatmap_voxel_path,
            num_channels=1
        )

        emotion_tensor = torch.zeros(
            (len(self.emotion_order),
            self.voxel_resolution,
            self.voxel_resolution,
            self.voxel_resolution),
            dtype=torch.float32
        )

        for i, emotion_name in enumerate(self.emotion_order):
            emotion_file = os.path.join(
                qna_dir,
                f"{DEFAULT_QNA_ANSWER_COLOR_MAP[emotion_name]['name']}_voxel.ply"
            )
            if os.path.exists(emotion_file):
                emotion_channel = self._load_voxel_to_tensor(emotion_file, num_channels=1)
                emotion_tensor[i,:,:,:] = (emotion_channel.squeeze() > 0).float()
            
        comment = ""
        if os.path.exists(comment_path):
            try:
                with open(comment_path, 'r', encoding='utf-8') as f:
                    comment = f.read().strip()
            except Exception as e:
                raise(ValueError(f"Could not read transcript: {comment_path}. Error: {e}"))

        comment_token = self.tokenizer.tokenize(comment)

        return dense_voxel_tensor, (emotion_tensor, heatmap_tensor, comment_token)
# yapf: enable
