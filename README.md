# 3D Heatmap Generation

**Process experiment data from**

- PointCloud (.csv)
- QNA (.csv)
- model (.obj) 

into

- Segmented QNA (.ply)
- PointCloud (.ply)
- Heatmap (.ply)

**PyTorch Dataset & DataLoader**

Template to load the processed data into PyTorch for model training.

Functions

- Filter data based on

    - Group

    - Session ID

    - Pottery / Dogu ID

    - Point cloud data size

    - QNA data size

    - Voice quality, 1 - 5

    - Language, JP | EN

- Generate filtered data statistics

- Pre-process OR In-time process data

- TO BE ADDED: voice quality enhancement (normalization, background noise removal, AI to isolate comments)

## Clone the latest version

```
git clone --depth 1 https://github.com/luhouyang/3d-heatmap-generation.git
```

## PyTorch Dataset & DataLoader

[**SCRIPT**](src/dataset/dataset.py)

## Processing & Visualization Scripts

[**SCRIPT**](src/testing_use.py)

1. Create a folder in the `src` directory called `data`

1. Download the raw data [Google Drive](https://drive.google.com/drive/folders/1vzZZfysW4tDlc2oXPpJ2WI2H3DBBufG6?usp=drive_link)

1. Paste all group folders (*i.e. G1, G2*) into the `data` directory

1. Create a folder in the `src` directory called `pottery`

1. Download the downsized pottery and dogu [Google Drive](https://drive.google.com/drive/folders/17zaoAvf2vPFnV8Yj6pCrLF8rSl9DqJyM?usp=drive_link)

1. Paste all pottery and dogu into the `pottery` directory

1. Filter the data by passing in the arguments or by using a [tracking sheet](https://docs.google.com/spreadsheets/d/1FLe6tAEtF5eAC3YXU8YLfOeI-VT83V1C/edit?usp=sharing&ouid=100175822335349725367&rtpof=true&sd=true) to the `filter_data_on_condition` function

1. Run the script, visualizations will be created inside each model folder

Modify the parameters for different results in the `filter_data_on_condition` function

```python
root (str): Root directory that contains all groups
pottery_path (str): Path to pottery files
preprocess (bool): Weather to preprocess and save the data to processed folder. Default: True
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
```
