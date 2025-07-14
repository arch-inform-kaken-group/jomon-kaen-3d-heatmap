"""
Author: Lu Hou Yang
Last updated: 8 July 2025

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
    preprocess: bool = True,
    use_cache: bool = True,
    from_tracking_sheet: bool = False,
    tracking_sheet_path: str = "",
):
    if (preprocess):
        return PreprocessJomonKaenDataset(
            root=root,
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
            use_cache=use_cache,
            from_tracking_sheet=from_tracking_sheet,
            tracking_sheet_path=tracking_sheet_path,
        )
    else:
        return InTimeJomonKaenDataset(
            root=root,
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
        )


class PreprocessJomonKaenDataset(Dataset):

    def __init__(
        self,
        root: str = "",
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
        use_cache: bool = True,
        from_tracking_sheet: bool = False,
        tracking_sheet_path: str = "",
    ):
        super(PreprocessJomonKaenDataset, self).__init__()

        self.data, errors = filter_data_on_condition(
            root=root,
            mode=mode,
            preprocess=True,
            hololens_2_spatial_error=hololens_2_spatial_error,
            base_color=base_color,
            cmap=cmap,
            groups=groups,
            session_ids=session_ids,
            pottery_ids=pottery_ids,
            min_pointcloud_size=min_pointcloud_size,
            min_qa_size=min_qa_size,
            min_voice_quality=min_voice_quality,
            use_cache=use_cache,
            from_tracking_sheet=from_tracking_sheet,
            tracking_sheet_path=tracking_sheet_path,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return super().__getitem__(index)


class InTimeJomonKaenDataset(Dataset):

    def __init__(
        self,
        root: str = "",
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
        root=r"C:\Users\luhou\Desktop\python\jomon-kaen-3d-heatmap\archive\data",
        preprocess=False,
        use_cache=False,
        from_tracking_sheet=True,
        tracking_sheet_path=r"C:\Users\luhou\Desktop\python\jomon-kaen-3d-heatmap\archive\dataset\Tracking_Sheet_METADATA.csv",
        mode=0)
    print(errors, len(data))


if "__main__" == __name__:
    main()
