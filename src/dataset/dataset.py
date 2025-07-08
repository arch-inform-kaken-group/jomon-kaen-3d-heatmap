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

### VARIABLES ###

# Pottery parameters
# Coordinate range of xyz can be between [-400, 400]
ball_radius = 25
hololens_2_spatial_error = 2.66

# Dogu parameters
# Coordinate range of xyz are between [-100, 100]
dogu_parameters_dict = {
    "IN0295(86)": [5, 5],
    "IN0306(87)": [3, 1.5],
    "NZ0001(90)": [3, 1.5],
    "SK0035(91)": [7, 5],
    "MH0037(88)": [3, 1.5],
    "NM0239(89)": [3, 1.5],
    "TK0020(92)": [3, 1.25],
    "UD0028(93)": [6, 2],
}


def get_jomon_kaen_dataset(
    root: str = "",
    groups: List = [],
    session_ids: List = [],
    pottery_ids: List = [],
    min_pointcloud_size: float = 0.0,
    min_qa_size: float = 0.0,
    min_voice_quality: float = 0.0,
    preprocess: bool = True,
    use_cache: bool = True,
    from_tracking_sheet: bool = False,
    tracking_sheet_path: str = "",
):
    if (preprocess):
        return PreprocessJomonKaenDataset(
            root=root,
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
        groups: List = [],
        session_ids: List = [],
        pottery_ids: List = [],
        min_pointcloud_size: float = 0.0,
        min_qa_size: float = 0.0,
        min_voice_quality: float = 0.0,
        use_cache: bool = True,
        from_tracking_sheet: bool = False,
        tracking_sheet_path: str = "",
    ):
        super(PreprocessJomonKaenDataset, self).__init__()

        self.processed_dir = "processed"  # Create a folder at the same level as 'raw'
        self.data = filter_data_on_condition(preprocess=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return super().__getitem__(index)


class InTimeJomonKaenDataset(Dataset):

    def __init__(
        self,
        root: str = "",
        groups: List = [],
        session_ids: List = [],
        pottery_ids: List = [],
        min_pointcloud_size: float = 0.0,
        min_qa_size: float = 0.0,
        min_voice_quality: float = 0.0,
    ):
        super(InTimeJomonKaenDataset, self).__init__()

        self.data = filter_data_on_condition(preprocess=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return super().__getitem__(index)


def main():
    print(torch.__version__)


if "__main__" == __name__:
    main()
