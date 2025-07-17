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

    - Voice quallity, 1 - 5

    - Languange, JP | EN

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

[**SCRIPT**](src/dataset/testing_use.py)

1. Create a folder in the `src` directory called `data`

1. Paste all group folders (*i.e. G1, G2*) into the `data` directory

1. Filter the data by passing in the arguments or by using a [tracking sheet](https://docs.google.com/spreadsheets/d/1FLe6tAEtF5eAC3YXU8YLfOeI-VT83V1C/edit?usp=sharing&ouid=100175822335349725367&rtpof=true&sd=true) to the `filter_data_on_condition` function

1. Run the script, visualizations will be created inside each model folder

Modify the parameters for different results in the `filter_data_on_condition` function

```python
root (str): Root directory that contains all groups 
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
```
