"""
Author: Lu Hou Yang
Last updated: 7 July 2025

Contains Jomon Kaen Datasets
- Preprocessed
    - first load will be longer as all data will be 
      processed and stored
    - will take up more storage

- In-Time
    - load and processed as training happens
    - may cause large overhead and bottleneck
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

### VARIABLES ###

# Pottery parameters
# Coordinate range of xyz can be between [-400, 400]
point_cloud_ball_radius = 25
mesh_interpolation_radius = 10
ball_radius = 25

# Dogu parameters
# Coordinate range of xyz are between [-100, 100]
dogu_parameters_dict = {
    "IN0295(86)": [5, 5, 5],
    "IN0306(87)": [3, 1.5, 3],
    "NZ0001(90)": [3, 1.5, 3],
    "SK0035(91)": [7, 5, 7],
    "MH0037(88)": [3, 1.5, 3],
    "NM0239(89)": [3, 1.5, 3],
    "TK0020(92)": [3, 1.25, 3],
    "UD0028(93)": [6, 2, 6],
}


def get_jomon_kaen_dataset(
    root: str = "",
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
        return PreprocessJomonKaenDataset()
    else:
        return InTimeJomonKaenDataset()


class PreprocessJomonKaenDataset(Dataset):

    def __init__(
        self,
        root: str = "",
        groups: List = [],
        session_ids: List = [],
        pottery_ids: List = [],
        min_pointcloud_size: float = 0.0,
        min_qa_size: float = 0.0,
        min_voice_quality: float = 0.1,
    ):
        super(PreprocessJomonKaenDataset, self).__init__()

        self.processed_dir = "processed"  # Create a folder at the same level as 'raw'

    def __len__():
        pass

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
        min_voice_quality: float = 0.1,
    ):
        super(InTimeJomonKaenDataset, self).__init__()

    def __len__():
        pass

    def __getitem__(self, index):
        return super().__getitem__(index)


def main():
    print(torch.__version__)


if "__main__" == __name__:
    main()
