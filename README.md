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

[**SCRIPT**](src/modified_processing_pc_hm_qa.py)

1. Create a folder in the same directory as `modified_processing_pc_hm_qa.py` called `data`

1. Paste all session folders (*i.e. 2025_05_14_23_18_33*) into the `data` directory

1. Run the script, visualizations will be created inside each model folder

Modify the parameters for different results

```pythonm
# Parameters
point_cloud_ball_radius = 25  # 25 | 0.05 for points in range ~[-200, 400] | ~[-1, 1]
mesh_interpolation_radius = 10  # 10 | 0.05 for points in range ~[-200, 400] | ~[-1, 1]
ball_radius = 25  # 25 | 0.05 for points in range ~[-200, 400] | ~[-1, 1]

viz = False

# Choose what to generate
generate_point_cloud = True
generate_mesh = True
generate_voxel_answers = True
generate_segmented_meshes = True
generate_combined_mesh = True

parameters_dict = {
    "rembak7": [0.05, 0.05, 0.05],
    "J-7606": [0.02, 0.005, 0.02],
    "IN0295(86)": [5, 5, 5],
    "IN0306(87)": [3, 1.5, 3],
    "NZ0001(90)": [3, 1.5, 3],
    "SK0035(91)": [7, 5, 7],
    "MH0037(88)": [3, 1.5, 3],
    "NM0239(89)": [3, 1.5, 3],
    "TK0020(92)": [3, 1.25, 3],
    "UD0028(93)": [6, 2, 6],
}
```
