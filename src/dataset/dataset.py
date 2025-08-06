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
from dataset.utils import *


def get_jomon_kaen_dataset(
    root: str = "",
    pottery_path: str = "",
    split: float = 0.1,
    test_groups: List = [],
    seed: int = 42,
    mode:
    int = 0,  # 'HEATMAP(VOXEL), QNA, VOICE': 0 | 'HEATMAP(VOXEL), QNA': 1 | 'HEATMAP(VOXEL), VOICE': 2 | 'HEATMAP(VOXEL)': 3
    hololens_2_spatial_error: float = DEFAULT_HOLOLENS_2_SPATIAL_ERROR,
    target_voxel_resolution: float = DEFAULT_TARGET_VOXEL_RESOLUTION,
    num_points: int = 4096,
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
    generate_fixation: bool = False,
    voxel_color: str = 'gray', # 'gray' | 'rgb'
    qna_marker: bool = False,
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
        generate_fixation (bool): Generate gaze fixation point cloud and heatmap, with a duration aggregated point cloud, heatmap and legend. Default: False
        voxel_color (str): 'gray' or 'rgb'. NOT YET IMPLEMENTED. Default: 'gray'
        qna_marker (bool): Generate QNA point cloud as shaped markers. Default: False

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
            generate_fixation=generate_fixation,
            voxel_color=voxel_color,
            qna_marker=qna_marker,
        )

        random_indicies = np.random.choice(len(data), len(data), replace=False)

        data = data[random_indicies]

        train_data = []
        test_data = []
        if len(test_groups) > 0:
            for data_paths in data:
                if (data_paths['GROUP'] in test_groups):
                    test_data.append(data_paths)
                else:
                    train_data.append(data_paths)
        else:
            train_data = data[int(split * len(data)):]
            test_data = data[:int(split * len(data))]

        train_dataset = PreprocessJomonKaenDataset(
            data=train_data,
            mode=mode,
            num_points=num_points,
        )
        test_dataset = PreprocessJomonKaenDataset(
            data=test_data,
            mode=mode,
            num_points=num_points,
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
            target_voxel_resolution=target_voxel_resolution,
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
        target_voxel_resolution=512,
        num_points=4096,
    ):
        super(PreprocessJomonKaenDataset, self).__init__()

        self.data = data
        self.mode = mode
        self.target_voxel_resolution = target_voxel_resolution
        self.num_points=num_points

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
            # # 1. Load the Open3D point cloud objects
            # data_paths = self.data[index]
            # voxel_file = str(data_paths[voxel_filename])
            # pottery_file = str(data_paths['processed_pottery_path'])

            # voxel_data = o3d.io.read_point_cloud(voxel_file)
            # pottery_data = o3d.io.read_point_cloud(pottery_file)

            # # 2. Define a Shared Voxel Grid Coordinate System
            # min_bound = pottery_data.get_min_bound()
            # max_bound = pottery_data.get_max_bound()
            # grid_size_world = np.max(max_bound - min_bound)
            # target_resolution = self.target_voxel_resolution

            # if grid_size_world < 1e-9:
            #     grid_size_world = target_resolution
            # voxel_size = grid_size_world / (target_resolution - 1)

            # # 3. Create the Pottery RGB Voxel Grid
            # pottery_points = np.asarray(pottery_data.points, dtype=np.float32)
            # if pottery_data.has_colors():
            #     pottery_colors = np.asarray(pottery_data.colors,
            #                                 dtype=np.float32)
            # else:
            #     pottery_colors = np.full_like(pottery_points,
            #                                   0.5,
            #                                   dtype=np.float32)

            # pottery_indices = np.floor(
            #     (pottery_points - min_bound) / voxel_size).astype(int)
            # pottery_valid_mask = np.all(
            #     (pottery_indices >= 0) & (pottery_indices < target_resolution),
            #     axis=1)
            # pottery_valid_indices = pottery_indices[pottery_valid_mask]
            # pottery_valid_colors = pottery_colors[pottery_valid_mask]

            # pottery_rgb_grid = np.zeros(
            #     (target_resolution, target_resolution, target_resolution, 3),
            #     dtype=np.float32)
            # pottery_rgb_grid[pottery_valid_indices[:, 0],
            #                  pottery_valid_indices[:, 1],
            #                  pottery_valid_indices[:,
            #                                        2]] = pottery_valid_colors

            # # 4. Create the Heatmap Intensity Voxel Grid
            # voxel_points = np.asarray(voxel_data.points, dtype=np.float32)
            # voxel_colors = np.asarray(
            #     voxel_data.colors, dtype=np.float32) if voxel_data.has_colors(
            #     ) else np.zeros_like(voxel_points)

            # # Convert RGB to a single intensity value by averaging the channels.
            # voxel_intensities = np.mean(voxel_colors, axis=1)

            # voxel_indices = np.floor(
            #     (voxel_points - min_bound) / voxel_size).astype(int)
            # voxel_valid_mask = np.all(
            #     (voxel_indices >= 0) & (voxel_indices < target_resolution),
            #     axis=1)
            # voxel_valid_indices = voxel_indices[voxel_valid_mask]

            # # Filter the single-channel intensities.
            # voxel_valid_intensities = voxel_intensities[voxel_valid_mask]

            # # Initialize an empty grid for 1-channel intensity data.
            # voxel_intensity_grid = np.zeros(
            #     (target_resolution, target_resolution, target_resolution, 1),
            #     dtype=np.float32)

            # # Place the intensity value into the single channel of the grid.
            # # We add a new axis to match the grid's shape: (N,) -> (N, 1)
            # voxel_intensity_grid[
            #     voxel_valid_indices[:, 0], voxel_valid_indices[:, 1],
            #     voxel_valid_indices[:, 2]] = voxel_valid_intensities[:, np.newaxis]

            # return pottery_rgb_grid, voxel_intensity_grid

            # 1. Load the Open3D point cloud objects from their paths
            data_paths = self.data[index]
            # Assuming 'voxel_filename' is a variable holding the key for the heatmap file
            # and that self.data is a list of dictionaries.
            voxel_file = str(data_paths[voxel_filename]) 
            pottery_file = str(data_paths['processed_pottery_path'])

            pottery_pcd = o3d.io.read_point_cloud(pottery_file)
            voxel_pcd = o3d.io.read_point_cloud(voxel_file)

            # --- Process Pottery Point Cloud (Input) ---
            pottery_points = np.asarray(pottery_pcd.points, dtype=np.float32)
            if pottery_pcd.has_colors():
                pottery_colors = np.asarray(pottery_pcd.colors, dtype=np.float32)
            else:
                # If no colors exist, use a neutral gray (0.5)
                pottery_colors = np.full_like(pottery_points, 0.5, dtype=np.float32)

            # --- Process Voxel/Heatmap Point Cloud (for Target Intensities) ---
            if voxel_pcd.has_colors():
                # Convert heatmap RGB colors to a single intensity value (average)
                voxel_intensities = np.mean(np.asarray(voxel_pcd.colors, dtype=np.float32), axis=1, keepdims=True)
            else:
                # If the heatmap has no colors, default to zero intensity
                voxel_intensities = np.zeros((len(pottery_points), 1), dtype=np.float32)

            # --- Sample a Fixed Number of Points ---
            # Since both clouds share the same structure, we can sample indices once.
            num_available_points = len(pottery_points)

            # Handle cases where one of the point clouds might be empty
            if num_available_points == 0 or len(voxel_intensities) == 0:
                # Return zero tensors to prevent errors.
                pottery_xyz_rgb = torch.zeros((self.num_points, 6), dtype=torch.float32)
                target_intensities = torch.zeros((self.num_points, 1), dtype=torch.float32)
                return pottery_xyz_rgb, target_intensities
                
            # To be safe, ensure both arrays have the same length before sampling
            min_points = min(num_available_points, len(voxel_intensities))

            # Choose indices to sample. Use replacement if not enough points are available.
            replace = min_points < self.num_points
            sample_indices = np.random.choice(min_points, self.num_points, replace=replace)

            # Use the same indices to sample from both the pottery and heatmap data
            sampled_pottery_points = pottery_points[sample_indices]
            sampled_pottery_colors = pottery_colors[sample_indices]
            target_intensities = voxel_intensities[sample_indices]

            # --- Combine and Convert to Tensors ---
            # Input tensor: concatenate XYZ and RGB features
            pottery_xyz_rgb = np.hstack((sampled_pottery_points, sampled_pottery_colors))

            # Convert final numpy arrays to PyTorch tensors
            pottery_tensor = torch.from_numpy(pottery_xyz_rgb)
            target_tensor = torch.from_numpy(target_intensities)

            return pottery_tensor, target_tensor


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

    # train_dataset, test_dataset = get_jomon_kaen_dataset(
    #     root="./src/data",
    #     pottery_path="./src/pottery",
    #     preprocess=True,
    #     use_cache=False,
    #     pottery_ids=["IN0017"],
    #     # 'HEATMAP(VOXEL), QNA, VOICE': 0 | 'HEATMAP(VOXEL), QNA': 1 | 'HEATMAP(VOXEL), VOICE': 2 | 'HEATMAP(VOXEL)': 3
    #     mode=3,
    #     # generate_pc_hm_voxel=False,
    #     generate_qna=False,
    #     generate_voice=False,
    #     generate_pottery_dogu_voxel=False,
    #     generate_sanity_check=False)

    # train_dataset, test_dataset = get_jomon_kaen_dataset(
    #     root=r"D:\storage\jomon_kaen\data",
    #     pottery_path=r"D:\storage\jomon_kaen\pottery",
    #     preprocess=True,
    #     use_cache=False,
    #     # pottery_ids=["IN0017"],
    #     # 'HEATMAP(VOXEL), QNA, VOICE': 0 | 'HEATMAP(VOXEL), QNA': 1 | 'HEATMAP(VOXEL), VOICE': 2 | 'HEATMAP(VOXEL)': 3
    #     mode=3,
    #     # generate_pc_hm_voxel=False,
    #     generate_qna=False,
    #     generate_voice=False,
    #     generate_pottery_dogu_voxel=False,
    #     generate_sanity_check=False,
    # )

    # pottery, voxel = train_dataset.__getitem__(0)

    # visualize_geometry(grid_to_pointcloud(pottery), point_size=2)
    # visualize_geometry(grid_to_pointcloud(voxel), point_size=2)

    pass

def grid_to_pointcloud(voxel_grid):
    """
    Converts a NumPy voxel grid into an Open3D PointCloud.

    Args:
        voxel_grid (np.ndarray): A grid with shape (D, H, W, C),
                                 where C is the number of channels (1 or 3).

    Returns:
        o3d.geometry.PointCloud: A point cloud ready for visualization.
    """
    # Find the indices (i, j, k) of all non-empty voxels.
    # We check if the sum of channels is greater than a small number.
    indices = np.argwhere(voxel_grid.sum(axis=-1) > 1e-9)
    if indices.size == 0:
        print("Warning: Voxel grid is empty.")
        return o3d.geometry.PointCloud()

    # Get the color or intensity data from those non-empty voxels.
    data = voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]]

    # Create the PointCloud object.
    pcd = o3d.geometry.PointCloud()
    
    # Use the grid indices as the 3D points for a direct visualization.
    pcd.points = o3d.utility.Vector3dVector(indices.astype(np.float32))

    # Assign the colors. If the data is single-channel (intensity),
    # repeat it to create a grayscale color.
    if data.shape[1] == 1:
        # It's an intensity grid, convert to grayscale RGB.
        pcd.colors = o3d.utility.Vector3dVector(np.repeat(data, 3, axis=1))
    else:
        # It's already an RGB grid.
        pcd.colors = o3d.utility.Vector3dVector(data)

    return pcd

if "__main__" == __name__:
    main()
