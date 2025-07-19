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
    mode: int = 0,  # 'STRICT': 0 | 'LINIENT': 1
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
        mode (int): 'HEATMAP, QNA, VOICE': 0 | 'HEATMAP, QNA': 1 | 'HEATMAP, VOICE': 2 | 'HEATMAP': 3
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
        generate_voice (bool): Generate voice. Default: True
        generate_pottery_dogu_voxel (bool): Generate the input pottery and dogu voxel. Default: True

    Returns:
        dataset
    """
    if (preprocess):
        return PreprocessJomonKaenDataset(
            root=root,
            pottery_path=pottery_path,
            mode=mode,
            hololens_2_spatial_error=hololens_2_spatial_error,
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
        )
    else:
        return InTimeJomonKaenDataset(
            root=root,
            pottery_path=pottery_path,
            mode=mode,
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
        )


class PreprocessJomonKaenDataset(Dataset):

    def __init__(
        self,
        root: str = "",
        pottery_path: str = "",
        mode: int = 0,  # 'STRICT': 0 | 'LINIENT': 1
        hololens_2_spatial_error: float = DEFAULT_HOLOLENS_2_SPATIAL_ERROR,
        target_voxel_resolution: int = DEFAULT_TARGET_VOXEL_RESOLUTION,
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
        use_cache: bool = True,
        from_tracking_sheet: bool = False,
        tracking_sheet_path: str = "",
        generate_report: bool = True,
        generate_pc_hm_voxel: bool = True,
        generate_qna: bool = True,
        generate_voice: bool = False,
        generate_pottery_dogu_voxel: bool = True,
    ):
        super(PreprocessJomonKaenDataset, self).__init__()

        self.data, errors = filter_data_on_condition(
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
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return super().__getitem__(index)


class InTimeJomonKaenDataset(Dataset):

    def __init__(
        self,
        root: str = "",
        pottery_path: str = "",
        mode: int = 0,  # 'STRICT': 0 | 'LINIENT': 1
        hololens_2_spatial_error: float = DEFAULT_HOLOLENS_2_SPATIAL_ERROR,
        base_color: List = DEFAULT_BASE_COLOR,
        cmap=DEFAULT_CMAP,
        groups: List = [],
        session_ids: List = [],
        pottery_ids: List = [],
        min_pointcloud_size: float = 0.0,
        min_qa_size: float = 0.0,
        min_voice_quality: float = 0.1,
    ):
        super(InTimeJomonKaenDataset, self).__init__()

        self.data, errors = filter_data_on_condition(
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return super().__getitem__(index)


def main():
    data, errors = filter_data_on_condition(
        root="./src/data",
        pottery_path="./src/pottery",
        preprocess=True,
        use_cache=False,
        # 'HEATMAP, QNA, VOICE': 0 | 'HEATMAP, QNA': 1 | 'HEATMAP, VOICE': 2 | 'HEATMAP': 3
        mode=0,
        # generate_pc_hm_voxel=False,
        # generate_voice=False,
    )


if "__main__" == __name__:
    main()
