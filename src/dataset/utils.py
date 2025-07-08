"""
Author: Lu Hou Yang
Last updated: 8 July 2025

Contains utility functions for 
- 3D eye gaze data and voice recording
- Data filtering
- Data statistics

Notes
- yapf was used to format code, to preserve manual formatting at some sections
  # yapf: disable
  # yapf: enable
  was used to control formatting behaviour
"""

import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import open3d as o3d

import matplotlib.pyplot as plt
from tqdm import tqdm

import torchaudio

# Pottery parameters
# Coordinate range of xyz can be between [-400, 400]
DEFAULT_BALL_RADIUS = 25
# https://arxiv.org/abs/2111.07209 [An Assessment of the Eye Tracking Signal Quality Captured in the HoloLens 2]
# Official: 1.5 | Paper original: 6.45 | Paper recalibrated: 2.66
DEFAULT_HOLOLENS_2_SPATIAL_ERROR = 2.66

# Dogu parameters
# Coordinate range of xyz are between [-100, 100]
DEFAULT_DOGU_PARAMETERS_DICT = {
    "IN0295(86)": [5, 5],
    "IN0306(87)": [3, 1.5],
    "NZ0001(90)": [3, 1.5],
    "SK0035(91)": [7, 5],
    "MH0037(88)": [3, 1.5],
    "NM0239(89)": [3, 1.5],
    "TK0020(92)": [3, 1.25],
    "UD0028(93)": [6, 2],
}

# Colors
DEFAULT_CMAP = plt.get_cmap('jet')
DEFAULT_BASE_COLOR = [0.0, 0.0, 0.0]

# Pottery & Dogu assigned numbers
ASSIGNED_NUMBERS_DICT = {
    'AS0001': '1',
    'FH0008': '2',
    'IN0003': '3',
    'IN0008': '4',
    'IN0009': '5',
    'IN0017': '6',
    'IN0081': '7',
    'IN0104': '8',
    'IN0135': '9',
    'IN0148': '10',
    'IN0220': '11',
    'IN0228': '12',
    'IN0232': '13',
    'IN0239': '14',
    'IN0277': '15',
    'MY0001': '16',
    'MY0002': '17',
    'MY0004': '18',
    'MY0006': '19',
    'MY0007': '20',
    'ND0001': '21',
    'NM0001': '22',
    'NM0002': '23',
    'NM0009': '24',
    'NM0010': '25',
    'NM0014': '26',
    'NM0015': '27',
    'NM0017': '28',
    'NM0041': '29',
    'NM0049': '30',
    'NM0066': '31',
    'NM0070': '32',
    'NM0072': '33',
    'NM0073': '34',
    'NM0079': '35',
    'NM0080': '36',
    'NM0099': '37',
    'NM0106': '38',
    'NM0133': '39',
    'NM0135': '40',
    'NM0144': '41',
    'NM0154': '42',
    'NM0156': '43',
    'NM0159': '44',
    'NM0168': '45',
    'NM0173': '46',
    'NM0175': '47',
    'NM0189': '48',
    'NM0191': '49',
    'NM0206': '50',
    'SB0002': '51',
    'SB0004': '52',
    'SI0001': '53',
    'SJ0503': '54',
    'SJ0504': '55',
    'SK0001': '56',
    'SK0002': '57',
    'SK0003': '58',
    'SK0004': '59',
    'SK0005': '60',
    'SK0013': '61',
    'SS0001': '62',
    'TJ0004': '63',
    'TJ0005': '64',
    'TJ0010': '65',
    'TK0002': '66',
    'TK0048': '67',
    'TK0057': '68',
    'UD0001': '69',
    'UD0003': '70',
    'UD0005': '71',
    'UD0006': '72',
    'UD0011': '73',
    'UD0013': '74',
    'UD0014': '75',
    'UD0016': '76',
    'UD0023': '77',
    'UD0302': '78',
    'UD0304': '79',
    'UD0308': '80',
    'UD0318': '81',
    'UD0322': '82',
    'UD0411': '83',
    'UD0412': '84',
    'UK0001': '85',
    'IN0295': '86',
    'IN0306': '87',
    'MH0037': '88',
    'NM0239': '89',
    'NZ0001': '90',
    'SK0035': '91',
    'TK0020': '92',
    'UD0028': '93'
}

# QNA Answer Color
DEFAULT_QNA_ANSEWR_COLOR_MAP = {
    "面白い・気になる形だ": {
        "rgb": [255, 165, 0],
        "name": "Orange"
    },
    "美しい・芸術的だ": {
        "rgb": [0, 128, 0],
        "name": "Green"
    },
    "不思議・意味不明": {
        "rgb": [128, 0, 128],
        "name": "Purple"
    },
    "不気味・不安・怖い": {
        "rgb": [255, 0, 0],
        "name": "Red"
    },
    "何も感じない": {
        "rgb": [255, 255, 0],
        "name": "Yellow"
    },
}

### CALCULATION ###


# yapf: disable
# Read about KD-Tree (Medium | EN): https://medium.com/@isurangawarnasooriya/exploring-kd-trees-a-comprehensive-guide-to-implementation-and-applications-in-python-3385fd56a246
# Read about KD-Tree (Qiita | JP): https://qiita.com/RAD0N/items/7a192a4a5351f481c99f
def calculate_normalized_point_intensity(pcd, ball_radius):
    """
    Calculate point intensity using Open3D KDTree for ball radius search.

    Args:
        pcd: Open3D point cloud object
        ball_radius (float): Radius for neighborhood search

    Returns:
        normalized_intensity: Array of normalized intensity values
    """
    # Build a KD-Tree with FLANN for efficient radius search
    # Open3D docs: https://www.open3d.org/docs/release/tutorial/geometry/kdtree.html
    # FLANN docs: https://www.cs.ubc.ca/research/flann/
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # Calculate intensity | Find the number of points in a radius of each point
    intensity = []
    for i, point in enumerate(pcd.points):
        # Use KD-Tree to find points within the radius
        [k, idx, _] = kdtree.search_radius_vector_3d(point, ball_radius)
        # We are interested in the length of list of indexes (points) in the radius
        # i.e. the intensity of the point as measured by the density of points within the radius
        # len(idx) - 1 to exclude the point itself
        intensity.append(len(idx) - 1)
    intensity = np.array(intensity)

    # Normalize intensity to the range [0, 1] for color mapping
    # NORMALIZED = (VALUE - MIN) / (MAX - MIN)
    min_intensity = np.min(intensity)
    max_intensity = np.max(intensity)
    normalized_intensity = (intensity - min_intensity) / (max_intensity - min_intensity) \
                            if max_intensity > min_intensity else np.zeros_like(intensity)

    return normalized_intensity
# yapf: enable

### FILTERING ###


def generate_filtered_dataset_report(
    errors: dict = {},
    groups: List = [],
    session_ids: List = [],
    pottery_ids: List = [],
    min_pointcloud_size: float = 0.0,
    min_qa_size: float = 0.0,
    min_voice_quality: float = 0.0,
    from_tracking_sheet: bool = False,
    tracking_sheet_path: str = "",
):
    pass


def filter_data_on_condition(
    preprocess: bool = True,
    ball_radius: float = DEFAULT_BALL_RADIUS,
    hololens_2_spatial_error: float = DEFAULT_HOLOLENS_2_SPATIAL_ERROR,
    base_color: List = DEFAULT_BASE_COLOR,
    dogu_parameters_dict: dict = DEFAULT_DOGU_PARAMETERS_DICT,
    groups: List = [],
    session_ids: List = [],
    pottery_ids: List = [],
    min_pointcloud_size: float = 0.0,
    min_qa_size: float = 0.0,
    min_voice_quality: float = 0.0,
    from_tracking_sheet: bool = False,
    tracking_sheet_path: str = "",
    generate_report: bool = True,
):
    # Store {ERROR : Number of instance} pairs
    errors = {}

    # Check if each data instance / file path exists
    data = []

    if (preprocess):
        pass

    if (generate_report):
        generate_report(
            errors=errors,
            groups=groups,
            session_ids=session_ids,
            pottery_ids=pottery_ids,
            min_pointcloud_size=min_pointcloud_size,
            min_qa_size=min_qa_size,
            min_voice_quality=min_voice_quality,
            from_tracking_sheet=from_tracking_sheet,
            tracking_sheet_path=tracking_sheet_path,
        )

    return data


### PROCESS DATA ###


def create_gaze_intensity_point_cloud(
    input_file,
    ball_radius=DEFAULT_BALL_RADIUS,
    CMAP=DEFAULT_CMAP,
):
    """
    Creates a point cloud from xyz data in CSV file,
    coloring points based on their normalized intensity using a KD-tree for radius search.

    Args:
        input_file (str): Path to the input CSV file containing point cloud data
        ball_radius (float): Radius of the sphere used for neighborhood search

    Returns:
        pcd: The created Open3D point cloud object
    """
    data = pd.read_csv(input_file, header=None, skiprows=1).to_numpy()

    positions = data[:, :3]  # Only xyz

    # Create an Open3D point cloud object
    # Open3D docs: https://www.open3d.org/docs/latest/python_api/open3d.geometry.PointCloud.html
    pcd = o3d.geometry.PointCloud()

    # PointCloud.points accepts float64 of c=shape (num_points, 3)
    # Vector3dVector converts float64 numpy array of shape (n, 3) to Open3D format
    # Vector3dVector docs: https://www.open3d.org/docs/release/python_api/open3d.utility.Vector3dVector.html
    pcd.points = o3d.utility.Vector3dVector(positions)

    # Calculate intensity
    normalized_intensity = calculate_normalized_point_intensity(
        pcd, ball_radius)

    # Get color based on normalized intensity
    colors = CMAP(normalized_intensity)[:, :3]
    pcd.colors = o3d.utility.Vector3dVecctor(colors)

    return pcd


# yapf: disable
def create_gaze_intensity_heatmap_mesh(
    input_file,
    model_file,
    ball_radius = DEFAULT_BALL_RADIUS,
    # https://arxiv.org/abs/2111.07209 [An Assessment of the Eye Tracking Signal Quality Captured in the HoloLens 2]
    # Official: 1.5 | Paper original: 6.45 | Paper recalibrated: 2.66
    HOLOLENS_2_SPATIAL_ERROR = DEFAULT_HOLOLENS_2_SPATIAL_ERROR,
    BASE_COLOR = DEFAULT_BASE_COLOR,
    CMAP = DEFAULT_CMAP,
):
    """
    Create a heatmap on a provided 3D mesh model based on the intensity
    of the input points (eye gaze point cloud), using a gaussian adjusted intensity
    for region coloring.

    Args:
        input_file (str): Path to the input CSV file containing point cloud data
        model_file (str): Path to the input OBJ/PLY model file
        ball_radius (float): Radius of the sphere used for neighborhood search
        HOLOLENS_2_SPATIAL_ERROR (float): The spatial accuracy / error of HoloLens 2. Reference: https://arxiv.org/abs/2111.07209 [An Assessment of the Eye Tracking Signal Quality Captured in the HoloLens 2]. Official: 1.5 | Paper original: 6.45 | Paper recalibrated: 2.66
        BASE_COLOR (tuple): Background color of the heatmap mesh for vertex that do not have gaze points
        CMAP (plt.Colormap): Heatmap color scheme. DEFAULT_CMAP = plt.get_cmap('jet')
    
    Returns:
        mesh: The created Open3D mesh object
    """
    SD_2_SQUARED_SPATIAL_ACCURACY = (2 * HOLOLENS_2_SPATIAL_ERROR)**2

    data = pd.read_csv(input_file, header=None, skiprows=1).to_numpy()

    positions = data[:, :3]  # Only xyz

    # Create an Open3D point cloud object
    # Open3D docs: https://www.open3d.org/docs/latest/python_api/open3d.geometry.PointCloud.html
    pcd = o3d.geometry.PointCloud()

    # PointCloud.points accepts float64 of c=shape (num_points, 3)
    # Vector3dVector converts float64 numpy array of shape (n, 3) to Open3D format
    # Vector3dVector docs: https://www.open3d.org/docs/release/python_api/open3d.utility.Vector3dVector.html
    pcd.points = o3d.utility.Vector3dVector(positions)

    # Calculate intensity
    normalized_intensity = calculate_normalized_point_intensity(pcd, ball_radius)

    # Load the mesh (OBJ or PLY) to apply the normalized intensity to
    mesh = o3d.io_read_triangle_mesh(model_file)
    vertices = np.asarray(mesh.vertices)
    n_vertices = vertices.shape[0]

    # Build KD-Tree for efficient radius search
    # Color the mesh based on normalized intensity with a circular interpolation
    # Modify radius later based on the eye gaze error / range
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    colors_intensity = np.zeros(n_vertices)
    colors = np.zeros((n_vertices, 3))

    # Find points on model mesh that are close to the eye gaze intensity point cloud
    for i, vertex in enumerate(vertices):
        [k, idx, _] = kdtree.search_radius_vector_3d(vertex, HOLOLENS_2_SPATIAL_ERROR)

        if idx:
            # Calculate the gaussian adjusted intensity of each vertex based on nearby points within radius
            #
            #                               n(points in radius)                         squared_euclidean_distance
            # gaussian_adjusted_intensity =        SUM          weight_of_point * e ^ - _____________________________
            #                                     i = 1                                 SD_2_SQUARED_SPATIAL_ACCURACY
            #
            # SD_2_SQUARED_SPATIAL_ACCURACY = (2 * HOLOLENS_2_SPATIAL_ERROR) ^ 2
            #
            # Then, get the average intensity and assign to the colors_intensity list
            # After all aggregation are done, normalize the colors_intensity
            # Map to the CMAP color table
            gaussian_adjusted_intensities = []
            for point_index in idx:
                # Get the squared euclidean distance
                # Since square root will be canceled by the square operation of gaussian adjusted weight calculation
                difference_vector = vertex - positions[point_index]
                squared_euclidean_distance = np.sum(difference_vector**2)
                gaussian_adjusted_intensity = normalized_intensity[point_index] * np.exp(-squared_euclidean_distance / SD_2_SQUARED_SPATIAL_ACCURACY)
                gaussian_adjusted_intensities.append(gaussian_adjusted_intensity)
            gaussian_adjusted_intensities = np.array(gaussian_adjusted_intensities)
            colors_intensity[i] = np.average(gaussian_adjusted_intensities)

    # Normalize the gaussian adjusted intensities
    colors_max = np.max(colors_intensity)
    colors_min = np.min(colors_intensity)
    normalized_colors_weights = (colors_intensity - colors_min) / (colors_max - colors_min)

    # Map to the CMAP color table
    for i in range(n_vertices):
        if (normalized_colors_weights[i] == 0):
            colors[i, :] = BASE_COLOR
        else:
            colors[i, :] = CMAP(normalized_colors_weights[i])[:3]

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    return mesh
    # yapf: enable


#yapf: disable
def create_qna_segmentation_mesh(
    input_file,
    model_file,
    association_radius = DEFAULT_BALL_RADIUS,
    QNA_ANSEWR_COLOR_MAP = DEFAULT_QNA_ANSEWR_COLOR_MAP,
    BASE_COLOR = DEFAULT_BASE_COLOR,
):
    """
    Assigns specific colors and color names based on predefined answer choices,
    segmented meshes for each answer category by answers. For combined mesh reference
    the original processing script. However, some data is lost during combination into 3 channels,
    it is prefered to combine the different answers into more channels later.

    Args:
        input_file (str): Path to the input CSV file containing point cloud data
        model_file (str): Path to the input OBJ/PLY model file
        association_radius (float): Radius of the sphere used to assign region and search for nearby QNA answers
        BASE_COLOR (tuple): Background color of the heatmap mesh for vertex that do not have gaze points
        QNA_ANSEWR_COLOR_MAP (dict): The QNA asnwers, rbg color, color name. DEFAULT_QNA_ANSEWR_COLOR_MAP = {
            "面白い・気になる形だ": {
                "rgb": [255, 165, 0],
                "name": "Orange"
            },
            "美しい・芸術的だ": {
                "rgb": [0, 128, 0],
                "name": "Green"
            },
            "不思議・意味不明": {
                "rgb": [128, 0, 128],
                "name": "Purple"
            },
            "不気味・不安・怖い": {
                "rgb": [255, 0, 0],
                "name": "Red"
            },
            "何も感じない": {
                "rgb": [255, 255, 0],
                "name": "Yellow"
            },
        }

    Returns:
        pcd: The created Open3D point cloud object
        segmented_mesh_dict: {answer: Open3D mesh object} pairs
    """
    if not os.path.exists(input_file):
        print(
            f"Error: QNA file '{input_file}' not found. Cannot perform operations."
        )
        return None
    df = pd.read_csv(input_file, sep=',', header=0)

    df['estX'] = pd.to_numeric(df['estX'], errors='coerce')
    df['estY'] = pd.to_numeric(df['estY'], errors='coerce')
    df['estZ'] = pd.to_numeric(df['estZ'], errors='coerce')
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')

    df['answer'] = df['answer'].astype(str).str.strip()

    df = df.dropna(subset=['estX', 'estY', 'estZ', 'answer', 'timestamp'])

    assigned_colors_01 = []  # Store as 0-1 for Open3D PLY
    for answer_text in df['answer']:
        if answer_text in QNA_ANSEWR_COLOR_MAP:
            assigned_colors_01.append(np.array(QNA_ANSEWR_COLOR_MAP[answer_text]["rgb"]) / 255.0)
        else:
            assigned_colors_01.append(BASE_COLOR / 255.0)

    positions = df[['estX', 'estY', 'estZ']].values.astype(np.float64)

    # Create QNA Point Cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    pcd.colors = o3d.utility.Vector3dVector(np.array(assigned_colors_01))

    # Create QNA Combined Mesh
    if not os.path.exists(model_file):
        print(
            f"Error: Base model file '{model_file}' not found. Cannot perform mesh operations."
        )
        return None

    mesh = o3d.io.read_triangle_mesh(model_file)
    vertices = np.asarray(mesh.vertices)

    segmented_mesh_dict = {}
    unique_answers = df['answer'].unique()

    for answer_category in unique_answers:
        category_df = df[df['answer'] == answer_category]
        if category_df.empty:
            continue

        category_gaze_positions = category_df[['estX', 'estY', 'estZ']].values.astype(np.float64)
        if len(category_gaze_positions) == 0:
            continue

        pcd_category_gaze = o3d.geometry.PointCloud()
        pcd_category_gaze.points = o3d.utility.Vector3dVector(category_gaze_positions)

        # Create a KDTree for the gaze points of this category
        category_tree = o3d.geometry.KDTreeFlann(pcd_category_gaze)

        category_rgb_01 = np.array(QNA_ANSEWR_COLOR_MAP[answer_category]["rgb"]) / 255.0

        # Create a copy of the base mesh for this segment
        segmented_mesh = o3d.geometry.TriangleMesh(mesh)
        segmented_mesh.paint_uniform_color(BASE_COLOR)

        mesh_vertex_colors = np.asarray(segmented_mesh.vertex_colors)
        for i, vertex in enumerate(vertices):
            [k, idx, _] = category_tree.search_radius_vector_3d(vertex, association_radius)

            if idx:
                mesh_vertex_colors[i] = category_rgb_01

        segmented_mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_vertex_colors)
        segmented_mesh_dict[answer_category] = segmented_mesh

    return pcd, segmented_mesh_dict
    # yapf: enable


def process_voice_data(input_file):
    if not Path(input_file).exists():
        return

    # 2646000 / 60 * 45 = 1992000
    waveform, sample_rate = torchaudio.load(input_file, num_frames=1992000)

    return waveform, sample_rate


### VISUALIZATIONS ###


def visualize_geometry(geometry, point_size=1.0):
    """
    Visualize an Open3D geometry (point cloud or mesh).
    
    Args:
        geometry: Open3D geometry object (point cloud or mesh)
        point_size: Size of points if geometry is a point cloud
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(geometry)
    render_options = vis.get_render_options()

    if isinstance(geometry, o3d.geometry.PointCloud):
        render_options.point_size = point_size
    elif isinstance(geometry, o3d.geometry.TriangleMesh):
        render_options.mesh_show_back_face = True

    render_options.background_color = np.asarray([0.1, 0.1, 0.1])
    vis.run()
    vis.destroy_window()
