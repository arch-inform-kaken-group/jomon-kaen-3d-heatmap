"""
Author: Lu Hou Yang
Last updated: 17 July 2025

Contains Jomon Kaen Datasets
- Preprocessed
    - first load will be longer as all data will be 
      processed and stored
    - will take up more storage

- In-Time
    - load and processed as training happens
    - may cause large overhead and bottleneck

Tracking sheet can be downloaded from the Google Sheet.
Apply filters to the METADATA Sheet and export as CSV.
"""

import os
from typing import List

import numpy as np

import torch
from torch.utils.data.dataset import Dataset

import matplotlib.pyplot as plt
import open3d as o3d

from pathlib import Path
from tqdm import tqdm
from utils import *


def get_jomon_kaen_dataset(
    root: str = "",
    pottery_path: str = "",
    split: float = 0.1,
    seed: int = 42,
    mode:
    int = 0,  # 'HEATMAP(VOXEL), QNA, VOICE': 0 | 'HEATMAP(VOXEL), QNA': 1 | 'HEATMAP(VOXEL), VOICE': 2 | 'HEATMAP(VOXEL)': 3
    hololens_2_spatial_error: float = DEFAULT_HOLOLENS_2_SPATIAL_ERROR,
    target_voxel_resolution: float = DEFAULT_TARGET_VOXEL_RESOLUTION,
    qna_answer_color_map: dict = DEFAULT_QNA_ANSWER_COLOR_MAP,
    base_color: List = DEFAULT_BASE_COLOR,
    cmap=DEFAULT_CMAP,
    groups: List = [],
    session_ids: List = [],
    pottery_ids: List = [],
    min_pointcloud_size: float = 0.0,
    min_qa_size: float = 0.0,
    min_voice_quality: float = 0.1,
    min_emotion_count: int = 0,
    preprocess: bool = True,
    use_cache: bool = True,
    from_tracking_sheet: bool = False,
    tracking_sheet_path: str = "",
    generate_report: bool = True,
    generate_pc_hm_voxel: bool = True,
    generate_qna: bool = True,
    generate_voice: bool = False,
    generate_pottery_dogu_voxel: bool = True,
    generate_sanity_check: bool = False,
):
    """
    Checks all paths from the root directory -> group -> session -> pottery/dogu -> raw data.
    Apply filters from (tracking sheet, arguments).
    Based on preprocess, use_cache the function will generate the training data in processed folder.
    Finally returns a list of dictionaries that provide the path to all training data.

    Args:
        root (str): Root directory that contains all groups 
        pottery_path (str): Path to pottery files
        preprocess (bool): Weather to preprocess and save the data to processed folder. Default: True
        split (float): Fraction of test dataset. Default: 0.1,
        seed (int): np.random.seed(42),
        mode (int): 'HEATMAP(VOXEL), QNA, VOICE': 0 | 'HEATMAP(VOXEL), QNA': 1 | 'HEATMAP(VOXEL), VOICE': 2 | 'HEATMAP(VOXEL)': 3
        hololens_2_spatial_error (float): Eye tracker spatial error of HoloLens 2. Default: DEFAULT_HOLOLENS_2_SPATIAL_ERROR
        target_voxel_resolution (int): Target heatmap voxel resolution. Default: DEFAULT_TARGET_VOXEL_RESOLUTION
        qna_answer_color_map (dict): The dictionary containing QNA answers with the rbg & name (color name). Default: DEFAULT_QNA_ANSWER_COLOR_MAP
        base_color (list): Background color of all generated data. Default: DEFAULT_BASE_COLOR
        cmap (plt.Colormap): Color scheme for intensities. Default: DEFAULT_CMAP
        groups (list): The list of groups to include, leave empty for all groups. Default: []
        session_ids (list): The list of sessions to include, leave empty for all sessions. Default: []
        pottery_ids (list): The list of potteries to include, leave empty for all potteries. Default: []
        min_pointcloud_size (float): Minimum pointcoud data size. Default: 0.0
        min_qa_size (float): Minimum qa data size. Default: 0.0
        min_voice_quality (float): Minimum voice quality 1-5. Requires a tracking sheet to filter. Default: 0.1
        min_emotion_count (int): Minimum emotion count. Unique QNA answers. Default: 0
        use_cache (bool): Use previous preprocessed data. Default: True
        from_tracking_sheet (bool): Use a tracking sheet .csv, downloaded from Google Sheets (You can filter the data at Google Sheets and export the subset). Default: False
        tracking_sheet_path (str): Path to the tracking sheet. Default: ""
        generate_report (bool): Generate a data report. Default: True
        generate_pc_hm_voxel (bool): Generate pointcloud, heatmap & voxel. Default: True
        generate_qna (bool): Generate QNA combined meah, segmented mesh, pointcloud. Default: True
        generate_voice (bool): Generate voice. Default: False
        generate_pottery_dogu_voxel (bool): Generate the input pottery and dogu voxel. Default: True
        generate_sanity_check (bool): Generate sanity check png. Default: False

    Returns:
        dataset
    """

    np.random.seed(seed)

    if (preprocess):
        data, errors = filter_data_on_condition(
            root=root,
            pottery_path=pottery_path,
            mode=mode,
            preprocess=True,
            hololens_2_spatial_error=hololens_2_spatial_error,
            target_voxel_resolution=target_voxel_resolution,
            qna_answer_color_map=qna_answer_color_map,
            base_color=base_color,
            cmap=cmap,
            groups=groups,
            session_ids=session_ids,
            pottery_ids=pottery_ids,
            min_pointcloud_size=min_pointcloud_size,
            min_qa_size=min_qa_size,
            min_voice_quality=min_voice_quality,
            min_emotion_count=min_emotion_count,
            use_cache=use_cache,
            from_tracking_sheet=from_tracking_sheet,
            tracking_sheet_path=tracking_sheet_path,
            generate_report=generate_report,
            generate_pc_hm_voxel=generate_pc_hm_voxel,
            generate_qna=generate_qna,
            generate_voice=generate_voice,
            generate_pottery_dogu_voxel=generate_pottery_dogu_voxel,
            generate_sanity_check=generate_sanity_check,
        )

        random_indicies = np.random.choice(len(data), len(data), replace=False)

        data = data[random_indicies]

        train_data = data[int(split * len(data)):]
        test_data = data[:int(split * len(data))]

        train_dataset = PreprocessJomonKaenDataset(
            data=train_data,
            mode=mode,
        )
        test_dataset = PreprocessJomonKaenDataset(
            data=test_data,
            mode=mode,
        )

        # # More sanity checks
        # print(train_dataset.__len__(), test_dataset.__len__())
        # all_data = [*train_dataset.data, *test_dataset]
        # idx=[]
        # data_temp = data
        # for d in all_data:
        #     if d in data_temp:
        #         i = np.where(data == d)
        #         idx.append(i)
        #         data_temp = np.delete(data_temp, np.where(data_temp == d))
        # print(len(np.unique(idx)), len(data_temp))
    else:
        data, errors = filter_data_on_condition(
            root=root,
            pottery_path=pottery_path,
            mode=mode,
            preprocess=False,
            hololens_2_spatial_error=hololens_2_spatial_error,
            base_color=base_color,
            cmap=cmap,
            groups=groups,
            session_ids=session_ids,
            pottery_ids=pottery_ids,
            min_pointcloud_size=min_pointcloud_size,
            min_qa_size=min_qa_size,
            min_voice_quality=min_voice_quality,
        )

        random_indicies = np.random.choice(len(data), len(data), replace=False)

        data = data[random_indicies]

        train_data = data[int(split * len(data)):]
        test_data = data[:int(split * len(data))]

        train_dataset = InTimeJomonKaenDataset(
            data=train_data,
            mode=mode,
        )
        test_dataset = InTimeJomonKaenDataset(
            data=test_data,
            mode=mode,
        )

    return train_dataset, test_dataset


class PreprocessJomonKaenDataset(Dataset):

    def __init__(
        self,
        data,
        mode=0,  # 'HEATMAP(VOXEL), QNA, VOICE': 0 | 'HEATMAP(VOXEL), QNA': 1 | 'HEATMAP(VOXEL), VOICE': 2 | 'HEATMAP(VOXEL)': 3
    ):
        super(PreprocessJomonKaenDataset, self).__init__()

        self.data = data
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode == 0:
            print("NOT YET IMPLEMENT MODE=0")
        elif self.mode == 1:
            print("NOT YET IMPLEMENT MODE=1")
        elif self.mode == 2:
            print("NOT YET IMPLEMENT MODE=2")
        else:
            data_paths = self.data[index]
            voxel_file = str(data_paths[voxel_filename])
            voxel_data = o3d.io.read_point_cloud(voxel_file)

            pottery_file = str(data_paths['processed_pottery_path'])
            pottery_data = o3d.io.read_point_cloud(pottery_file)

            return pottery_data, voxel_data

    # # Sanity check __getitem__
    # def __getitem__(self, index):
    #     return self.data[index]


class InTimeJomonKaenDataset(Dataset):

    def __init__(
        self,
        data,
        mode=0,  # 'HEATMAP(VOXEL), QNA, VOICE': 0 | 'HEATMAP(VOXEL), QNA': 1 | 'HEATMAP(VOXEL), VOICE': 2 | 'HEATMAP(VOXEL)': 3
    ):
        super(InTimeJomonKaenDataset, self).__init__()

        self.data = data
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return super().__getitem__(index)


def main():
    # data, errors = filter_data_on_condition(
    #     root="./src/data",
    #     pottery_path="./src/pottery",
    #     preprocess=False,
    #     use_cache=True,
    #     # 'HEATMAP(VOXEL), QNA, VOICE': 0 | 'HEATMAP(VOXEL), QNA': 1 | 'HEATMAP(VOXEL), VOICE': 2 | 'HEATMAP(VOXEL)': 3
    #     mode=0,
    #     # generate_pc_hm_voxel=False,
    #     # generate_voice=False,
    # )

    train_dataset, test_dataset = get_jomon_kaen_dataset(
        root="./src/data",
        pottery_path="./src/pottery",
        preprocess=True,
        use_cache=False,
        pottery_ids=["IN0017"],
        # 'HEATMAP(VOXEL), QNA, VOICE': 0 | 'HEATMAP(VOXEL), QNA': 1 | 'HEATMAP(VOXEL), VOICE': 2 | 'HEATMAP(VOXEL)': 3
        mode=3,
        # generate_pc_hm_voxel=False,
        generate_qna=False,
        generate_voice=False,
        generate_pottery_dogu_voxel=False,
        generate_sanity_check=False
    )

    pottery, voxel = train_dataset.__getitem__(0)

    visualize_geometry(pottery, point_size=2)
    visualize_geometry(voxel, point_size=2)

    voxel_point_check(pottery, voxel)

def voxel_point_check(pottery_data, voxel_data):
    """
    Compares two point clouds to find matching, extra, and uncovered points.

    - Matching (Green): Points present in both clouds.
    - Extra Heatmap (Red): Points in the heatmap but not in the pottery.
    - Uncovered Pottery (Blue): Points in the pottery but not in the heatmap.
    """
    pottery_coords = np.asarray(pottery_data.points)
    voxel_coords = np.asarray(voxel_data.points)

    # Create sets for very fast lookups in both directions
    pottery_coords_set = {tuple(point) for point in pottery_coords}
    voxel_coords_set = {tuple(point) for point in voxel_coords}

    # --- Find matching points and extra heatmap points ---
    matching_points = []
    extra_heatmap_points = []
    for point in voxel_coords:
        if tuple(point) in pottery_coords_set:
            matching_points.append(point)
        else:
            extra_heatmap_points.append(point)

    # --- NEW: Find uncovered pottery points ---
    uncovered_pottery_points = []
    for point in pottery_coords:
        if tuple(point) not in voxel_coords_set:
            uncovered_pottery_points.append(point)

    # --- Report the findings ---
    print(f"🏺 Pottery point cloud has {len(pottery_coords)} points.")
    print(f"🔥 Voxel (heatmap) point cloud has {len(voxel_coords)} points.")
    print(f"✅ Matching points: {len(matching_points)}")
    print("-" * 30)
    print(f"❌ Extra heatmap points (in 🔥 but not 🏺): {len(extra_heatmap_points)}")
    print(f"🔵 Uncovered pottery points (in 🏺 but not 🔥): {len(uncovered_pottery_points)}")

    # --- Visualize all components ---
    print("\nVisualizing the correspondence check...")

    # 1. Matching points (Green)
    matching_pcd = o3d.geometry.PointCloud()
    matching_pcd.points = o3d.utility.Vector3dVector(matching_points)
    matching_pcd.paint_uniform_color([0.0, 0.8, 0.0])  # Green

    # 2. Extra heatmap points (Red)
    extra_heatmap_pcd = o3d.geometry.PointCloud()
    extra_heatmap_pcd.points = o3d.utility.Vector3dVector(extra_heatmap_points)
    extra_heatmap_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red

    # 3. Uncovered pottery points (Blue)
    uncovered_pottery_pcd = o3d.geometry.PointCloud()
    uncovered_pottery_pcd.points = o3d.utility.Vector3dVector(uncovered_pottery_points)
    uncovered_pottery_pcd.paint_uniform_color([0.0, 0.0, 1.0])  # Blue

    # Visualize all three point clouds together
    o3d.visualization.draw_geometries(
        [matching_pcd, extra_heatmap_pcd, uncovered_pottery_pcd],
        point_show_normal=False,
        window_name="Correspondence Check"
    )


if "__main__" == __name__:
    main()
