import os
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from dataset.utils import filter_data_on_condition
from dataset.metadata import CANONICAL_EMOTIONS, filter_paths_by_artifact_type


def split_into_concepts(text: str, max_concepts: int = 6) -> List[str]:
    """Splits Japanese or English transcripts into sentence-level concepts."""
    text = text.strip()
    if not text:
        return ["反応なし"]
    # Split on Japanese/English sentence delimiters
    sentences = [s.strip() for s in re.split(r"[。！？.!?\n]+", text) if s.strip()]
    if not sentences:
        sentences = [text]
    return sentences[:max_concepts]


class JomonPointJEPADataset(Dataset):
    """
    Point-native multimodal dataset:
      - coords:       [N, 3] normalized 3D surface coordinates
      - colors:       [N, 3] RGB in [0, 1]
      - normals:      [N, 3] unit surface normals
      - gaze:         [N, 1] continuous gaze heatmap intensity in [0, 1]
      - emotions:     [N, 5] multi-label emotion mask in {0, 1}
      - text:         raw transcript string
      - concepts:     list[str] of sentence-level concepts
      - country:      'japan' or 'malaysia'
      - artifact_type:'pottery' or 'dogu'
    """

    def __init__(
        self,
        data_paths: List[dict],
        num_points: int = 4096,
        max_concepts: int = 6,
        augment: bool = False,
    ):
        super().__init__()
        self.data_paths = data_paths
        self.num_points = num_points
        self.max_concepts = max_concepts
        self.augment = augment

    def __len__(self) -> int:
        return len(self.data_paths)

    def _read_pcd_arrays(self, path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points, dtype=np.float32)
        if len(pts) == 0:
            raise ValueError(f"Empty point cloud: {path}")

        if pcd.has_colors():
            cols = np.asarray(pcd.colors, dtype=np.float32)
        else:
            cols = np.full_like(pts, 0.5, dtype=np.float32)

        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
            )
        norms = np.asarray(pcd.normals, dtype=np.float32)
        return pts, cols, norms

    def _align_target_to_base(
        self,
        target_path: str,
        base_pts: np.ndarray,
        base_kdtree: o3d.geometry.KDTreeFlann,
    ) -> np.ndarray:
        """
        Loads a target .ply (gaze or emotion). If point counts match 1-to-1
        (standard in your aggregation.py when base_pottery_pcd is passed), uses direct indexing;
        otherwise maps via nearest-neighbor KDTree lookup.
        """
        if not os.path.exists(target_path):
            return np.zeros((len(base_pts),), dtype=np.float32)

        t_pcd = o3d.io.read_point_cloud(target_path)
        t_pts = np.asarray(t_pcd.points, dtype=np.float32)
        if len(t_pts) == 0 or not t_pcd.has_colors():
            return np.zeros((len(base_pts),), dtype=np.float32)

        t_cols = np.asarray(t_pcd.colors, dtype=np.float32)
        intensities = t_cols.max(axis=1)  # Works for both grayscale gaze & RGB emotion channels

        if len(t_pts) == len(base_pts) and np.allclose(t_pts[:10], base_pts[:10], atol=1e-4):
            return intensities.astype(np.float32)

        aligned = np.zeros((len(base_pts),), dtype=np.float32)
        for idx, pt in enumerate(t_pts):
            if intensities[idx] > 1e-4:
                k, nn_idx, _ = base_kdtree.search_knn_vector_3d(pt, 1)
                if k > 0:
                    aligned[nn_idx[0]] = max(aligned[nn_idx[0]], intensities[idx])
        return aligned

    def __getitem__(self, index: int) -> Dict:
        item = self.data_paths[index]
        pts, cols, norms = self._read_pcd_arrays(item["processed_pottery_path"])

        # Normalize coordinates to unit sphere centered at origin
        centroid = pts.mean(axis=0, keepdims=True)
        pts_norm = pts - centroid
        scale = np.max(np.linalg.norm(pts_norm, axis=1)) + 1e-6
        pts_norm = pts_norm / scale

        base_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        base_kdtree = o3d.geometry.KDTreeFlann(base_pcd)

        # 1. Load Gaze Intensity [M]
        gaze = self._align_target_to_base(item["eye_gaze_voxel"], pts, base_kdtree)
        if gaze.max() > 1e-6:
            gaze = gaze / gaze.max()

        # 2. Load 5 Canonical Emotions [M, 5]
        qna_dir = item["qa_segmented_mesh"]
        emotions = np.zeros((len(pts), 5), dtype=np.float32)
        for c_idx, emo_meta in enumerate(CANONICAL_EMOTIONS):
            emo_file = os.path.join(qna_dir, f"{emo_meta['color']}_voxel.ply")
            emo_vals = self._align_target_to_base(emo_file, pts, base_kdtree)
            emotions[:, c_idx] = (emo_vals > 0.1).astype(np.float32)

        # 3. Deterministic or Random Subsampling to Fixed N Points
        M = len(pts_norm)
        replace = M < self.num_points
        choice = np.random.choice(M, self.num_points, replace=replace)

        pts_sub = pts_norm[choice]
        cols_sub = cols[choice]
        norms_sub = norms[choice]
        gaze_sub = gaze[choice, None]
        emo_sub = emotions[choice]

        # 4. 3D Data Augmentation (Y-axis rotation + slight scale + color jitter)
        if self.augment:
            theta = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(theta), np.sin(theta)
            rot_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
            pts_sub = pts_sub @ rot_y.T
            norms_sub = norms_sub @ rot_y.T
            pts_sub *= np.random.uniform(0.92, 1.08)
            if np.random.rand() < 0.5:
                cols_sub = np.clip(
                    cols_sub + np.random.normal(0, 0.04, cols_sub.shape).astype(np.float32),
                    0.0,
                    1.0,
                )

        # 5. Transcript & Concepts
        transcript = ""
        t_path = item.get("TRANSCRIPT", "")
        if t_path and os.path.exists(t_path):
            with open(t_path, "r", encoding="utf-8") as f:
                transcript = f.read().strip()

        concepts = split_into_concepts(transcript, max_concepts=self.max_concepts)

        return {
            "coord": torch.from_numpy(pts_sub),
            "color": torch.from_numpy(cols_sub),
            "normal": torch.from_numpy(norms_sub),
            "gaze": torch.from_numpy(gaze_sub),
            "emotion": torch.from_numpy(emo_sub),
            "text": transcript,
            "concepts": concepts,
            "country": item.get("COUNTRY", "japan"),
            "artifact_type": item.get("ARTIFACT_TYPE", "pottery"),
            "vessel_id": item.get("ID", ""),
            "group": item.get("GROUP", ""),
            "session_id": item.get("SESSION_ID", ""),
            "ref_pcd_path": item["processed_pottery_path"],
        }


def jepa_collate_fn(batch: List[Dict]) -> Dict:
    return {
        "coord": torch.stack([b["coord"] for b in batch]),
        "color": torch.stack([b["color"] for b in batch]),
        "normal": torch.stack([b["normal"] for b in batch]),
        "gaze": torch.stack([b["gaze"] for b in batch]),
        "emotion": torch.stack([b["emotion"] for b in batch]),
        "text": [b["text"] for b in batch],
        "concepts": [b["concepts"] for b in batch],
        "country": [b["country"] for b in batch],
        "artifact_type": [b["artifact_type"] for b in batch],
        "vessel_id": [b["vessel_id"] for b in batch],
        "group": [b["group"] for b in batch],
        "session_id": [b["session_id"] for b in batch],
        "ref_pcd_path": [b["ref_pcd_path"] for b in batch],
    }


class JomonJEPADataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_paths = []
        self.val_paths = []

    def setup(self, stage: Optional[str] = None):
        if getattr(self, "train_dataset", None) is not None and getattr(self, "val_dataset", None) is not None:
            return

        countries = ["japan", "malaysia"] if self.cfg.country in ("all", "both") else [self.cfg.country]
        combined_paths = []

        pottery_dir = os.path.expanduser(self.cfg.pottery_mesh_dir)
        for c in countries:
            root_dir = os.path.expanduser(self.cfg.data_roots[c])
            if not os.path.exists(root_dir):
                print(f"[Warning] Country data root not found, skipping: {root_dir}")
                continue
            paths, _ = filter_data_on_condition(
                root=root_dir,
                pottery_path=pottery_dir,
                preprocess=True,
                use_cache=True,
                mode=0,
                target_voxel_resolution=self.cfg.voxel_resolution_preprocess,
                min_emotion_count=1,
                min_qa_size=1,
                limit=10000,
                generate_report=False,
                generate_sanity_check=False,
                generate_fixation=False,
            )
            for p in paths:
                p["COUNTRY"] = c
            combined_paths.extend(list(paths))

        # Filter by Pottery vs. Dogu
        combined_paths = filter_paths_by_artifact_type(
            combined_paths,
            artifact_type=self.cfg.artifact_type,
            custom_json_path=self.cfg.custom_artifact_type_json,
        )

        if len(combined_paths) == 0:
            raise ValueError("No valid data paths found after country and artifact_type filtering!")

        # 1. Split held-out TEST_GROUPS FIRST (before any limit_samples truncation)
        val_set = set(self.cfg.val_vessel_ids)
        val_base_set = {vid.split("(")[0].strip() for vid in val_set}
        train_paths = [
            p for p in combined_paths
            if p["ID"] not in val_set and p["ID"].split("(")[0].strip() not in val_base_set
        ]
        val_paths = [
            p for p in combined_paths
            if p["ID"] in val_set or p["ID"].split("(")[0].strip() in val_base_set
        ]

        # Fallback only if filtering for 'dogu' where none of the 8 pottery TEST_GROUPS exist
        if len(val_paths) == 0 or len(train_paths) == 0:
            unique_ids = sorted(list({p["ID"] for p in combined_paths}))
            n_val_ids = max(1, int(len(unique_ids) * self.cfg.val_split_ratio))
            fallback_val_ids = set(unique_ids[-n_val_ids:])
            train_paths = [p for p in combined_paths if p["ID"] not in fallback_val_ids]
            val_paths = [p for p in combined_paths if p["ID"] in fallback_val_ids]

        # 2. Apply limit_samples AFTER splitting so val_paths always comes from TEST_GROUPS
        if self.cfg.limit_samples is not None:
            train_paths = train_paths[: self.cfg.limit_samples]
            val_paths = val_paths[: max(4, self.cfg.limit_samples // 2)]
        self.train_paths = train_paths
        self.val_paths = val_paths
        print(
            f"[DataModule] Country={self.cfg.country} | Artifact={self.cfg.artifact_type} | "
            f"Train={len(self.train_paths)} | Val={len(self.val_paths)}"
        )

        self.train_dataset = JomonPointJEPADataset(
            self.train_paths,
            num_points=self.cfg.num_points,
            max_concepts=self.cfg.max_concepts_per_transcript,
            augment=True,
        )
        self.val_dataset = JomonPointJEPADataset(
            self.val_paths,
            num_points=self.cfg.num_points,
            max_concepts=self.cfg.max_concepts_per_transcript,
            augment=False,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=jepa_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=jepa_collate_fn,
        )