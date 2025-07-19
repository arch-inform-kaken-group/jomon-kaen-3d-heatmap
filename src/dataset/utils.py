"""
Author: Lu Hou Yang
Last updated: 17 July 2025

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

from collections import deque
import os
from pathlib import Path
import sys
import threading
import queue

import numpy as np
import pandas as pd
import open3d as o3d
import trimesh  # For voxelizing pottery and dogu with color

import matplotlib.pyplot as plt
from tqdm import tqdm

import torchaudio

import io
from reportlab.graphics.shapes import Rect, Drawing
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from PIL import Image as PILImage

# https://arxiv.org/abs/2111.07209 [An Assessment of the Eye Tracking Signal Quality Captured in the HoloLens 2]
# Official: 1.5 | Paper original: 6.45 | Paper recalibrated: 2.66
DEFAULT_HOLOLENS_2_SPATIAL_ERROR = 1.5
DEFAULT_GAUSSIAN_DENOMINATOR = 2 * (DEFAULT_HOLOLENS_2_SPATIAL_ERROR**2)
DEFAULT_TARGET_VOXEL_RESOLUTION = 512

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
DEFAULT_QNA_ANSWER_COLOR_MAP = {
    "面白い・気になる形だ": {
        "rgb": [255, 165, 0],
        "name": "orange"
    },
    "美しい・芸術的だ": {
        "rgb": [0, 128, 0],
        "name": "green"
    },
    "不思議・意味不明": {
        "rgb": [128, 0, 128],
        "name": "purple"
    },
    "不気味・不安・怖い": {
        "rgb": [255, 0, 0],
        "name": "red"
    },
    "何も感じない": {
        "rgb": [128, 128, 128],
        "name": "grey"
    },
}

# Threading
data_lock = threading.Lock()

# Data paths
sanity_plot_filename = "pointcloud_occurrence_plot"
eg_pointcloud_filename = "eye_gaze_intensity_pc"
eg_heatmap_filename = "eye_gaze_intensity_hm"
voxel_filename = "eye_gaze_voxel"
qa_pc_filename = "qa_pc"
segmented_meshes_dirname = "qa_segmented_mesh"
processed_voice_filename = "processed_voice"
combined_mesh_filename = "combined_qa_mesh"
pottery_dirname = "voxel_pottery"

MESH_PC_VOXEL_EXTENSION = ".ply"
VOICE_EXTENSION = ".wav"

### CALCULATION ###


# yapf: disable
# Read about KD-Tree (Medium | EN): https://medium.com/@isurangawarnasooriya/exploring-kd-trees-a-comprehensive-guide-to-implementation-and-applications-in-python-3385fd56a246
# Read about KD-Tree (Qiita | JP): https://qiita.com/RAD0N/items/7a192a4a5351f481c99f
def _calculate_smoothed_vertex_intensities(
    gaze_points_np,
    mesh,
    # https://arxiv.org/abs/2111.07209 [An Assessment of the Eye Tracking Signal Quality Captured in the HoloLens 2]
    # Official: 1.5 | Paper original: 6.45 | Paper recalibrated: 2.66
    hololens_2_spatial_error=DEFAULT_HOLOLENS_2_SPATIAL_ERROR,
    # gaussian_denominator = 2 * (hololens_2_spatial_error ^ 2)
    gaussian_denominator=DEFAULT_GAUSSIAN_DENOMINATOR,
):
    # Get the vertices, triangles of the mesh
    mesh_vertices_np = np.asarray(mesh.vertices)
    mesh_triangles_np = np.asarray(mesh.triangles)
    n_vertices = mesh_vertices_np.shape[0]

    # Mapping gaze points to nearest mesh faces (vertices)
    # Using a raycasting scene from Open3D
    # RaycastingScene docs: https://www.open3d.org/docs/release/python_api/open3d.t.geometry.RaycastingScene.html
    mesh_scene = o3d.t.geometry.RaycastingScene()
    mesh_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    # Add gaze points as the query points, to find closest triangle (primitive_ids)
    query_points = o3d.core.Tensor(gaze_points_np, dtype=o3d.core.Dtype.Float32)
    closest_geometry = mesh_scene.compute_closest_points(query_points)
    # A tensor with the primitive IDs, which corresponds to the triangle index.
    closest_face_indices = closest_geometry['primitive_ids'].numpy()

    # Calculate the raw hit counts on the mesh vertices
    # & assign each gaze point to its closest triangle index
    # triangle (face) = [vertex_1, vertex_2, vertex_3]
    #
    # Mesh vertices intensity   :   Loop over all vertices in each triangle (closest_face_indices)
    #                               aggregate to each vertex according to index
    # Point cloud intensity     :   Store the index of closest triangle (face), after the intensity
    #                               on each mesh vertex is calculated, the stored triangle index
    #                               can be used to find the intensity of point cloud
    raw_hit_counts = np.zeros(n_vertices, dtype=np.float64)
    point_to_face_map = np.empty(gaze_points_np.shape[0], dtype=int)

    for i, closest_face_idx in tqdm(enumerate(closest_face_indices), desc="Raycasting Gaze Points", leave=False):
        if closest_face_idx != o3d.t.geometry.RaycastingScene.INVALID_ID:
            point_to_face_map[i] = closest_face_idx
            for v_idx in mesh_triangles_np[closest_face_idx]:
                raw_hit_counts[v_idx] += 1

    # Log scaling improves visual detail, large numbers do not dominate the heatmap
    # causing the difference (comparison) to be lost. i.e. difference between 1 & 10 hits
    # and 100 & 1000 hits both are shown on the heatmap.
    #
    # Log scaling aligns with human perception (logarithmic)
    # more sensitive to change at lower levels of stimulus compared to high levels.
    #
    # Enables the handling of wide dynamic ranges. If gaze is recorded for 5 mins, etc.
    #
    # np.log1p() = np.log(1 + x) | Prevents log(0) = inf
    raw_hit_counts = np.log1p(raw_hit_counts)

    # Applying Gaussian spread (parameters tuned to the eye tracker error of HoloLens 2)
    #
    # This method of spreading will cause leakage for high values of error
    # However, it will ensure nearby vertices recieve color, even if the
    # original mesh is malformed (not fully connected, missing edges)
    #
    # Build a KD-tree with FLANN for efficient radius search
    # Open3D docs: https://www.open3d.org/docs/release/tutorial/geometry/kdtree.html
    # FLANN docs: https://www.cs.ubc.ca/research/flann/
    kdtree = o3d.geometry.KDTreeFlann(mesh)
    interpolated_heatmap_values = np.copy(raw_hit_counts)
    hit_vertices_indices = np.where(raw_hit_counts > 0)[0]

    for start_node_idx in tqdm(hit_vertices_indices, desc="Applying Gaussian Spread", leave=False):
        hit_value = raw_hit_counts[start_node_idx]
        # Use KD-Tree to find points within the radius
        # [num_points, point_indices, euclidean_distance]
        [k, indices, euclidean_dist] = kdtree.search_radius_vector_3d(mesh_vertices_np[start_node_idx], hololens_2_spatial_error)
        if k > 1:
            # Calculate the gaussian adjusted intensity of each vertex based on nearby points within radius
            #
            #       n(points in radius)                         squared_euclidean_distance
            # GAI =        SUM          weight_of_point * e ^ - _____________________________
            #             i = 1                                 gaussian_denominator
            #
            # gaussian_denominator = 2 * (hololens_2_spatial_error ^ 2)
            gaussian_weights = np.exp(-np.asarray(euclidean_dist)**2 / gaussian_denominator)
            for i, neighbor_idx in enumerate(indices):
                if neighbor_idx != start_node_idx:
                    interpolated_heatmap_values[neighbor_idx] += hit_value * gaussian_weights[i]

    # # ALTERNATIVE GAUSSIAN METHOD | NO LEAKAGE | BUT CAUSES UNCOLORED / DISCONNECTED MESH
    # # This method ensures that there is no leakage
    # # However, some vertices will not recieve color if they are not connected properly
    # # caused by errors during model downsizing or scanning
    # vertex_adjacency = {i: set() for i in range(n_vertices)}
    # for v0, v1, v2 in mesh_triangles_np:
    #     vertex_adjacency[v0].update([v1, v2])
    #     vertex_adjacency[v1].update([v0, v2])
    #     vertex_adjacency[v2].update([v0, v1])
    # interpolated_heatmap_values = np.zeros(n_vertices, dtype=np.float64)
    # hit_vertices_indices = np.where(raw_hit_counts > 0)[0]
    # for start_node_idx in tqdm(
    #     hit_vertices_indices, desc="Spreading heatmap via BFS"
    # ):
    #     hit_value = raw_hit_counts[start_node_idx]
    #     start_pos = mesh_vertices_np[start_node_idx]
    #     q = deque([start_node_idx])
    #     visited = {start_node_idx}
    #     interpolated_heatmap_values[start_node_idx] += hit_value
    #     while q:
    #         current_idx = q.popleft()
    #         for neighbor_idx in vertex_adjacency[current_idx]:
    #             if neighbor_idx not in visited:
    #                 dist_from_start = np.linalg.norm(mesh_vertices_np[neighbor_idx] - start_pos)
    #                 if dist_from_start <= hololens_2_spatial_error:
    #                     visited.add(neighbor_idx)
    #                     q.append(neighbor_idx)
    #                     distance_sq = dist_from_start**2
    #                     gaussian_weight = np.exp(-distance_sq / gaussian_denominator)
    #                     interpolated_heatmap_values[neighbor_idx] += hit_value * gaussian_weight

    return interpolated_heatmap_values, point_to_face_map
# yapf: enable

### UTILS ###


def increment_error(key, path, errors: dict):
    if errors.get(key) == None:
        errors[key] = {'count': 1, 'paths': set([path])}
    else:
        errors[key]['count'] += 1
        errors[key]['paths'].add(path)

    return errors


def save_geometry_threaded(save_path, geometry, error_queue):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def _save_geometry(path, geom, errq):
        original_verbosity = o3d.utility.VerbosityLevel.Warning
        try:
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

            if isinstance(geom, o3d.geometry.PointCloud):
                o3d.io.write_point_cloud(path, geom, write_ascii=True)
            elif isinstance(geom, o3d.geometry.TriangleMesh):
                o3d.io.write_triangle_mesh(path, geom, write_ascii=True)
            else:
                print(f"Unsupported geometry type for saving: {type(geom)}",
                      file=sys.stderr)
        except Exception as e:
            print(f"An error occurred while saving geometry to {path}: {e}",
                  file=sys.stderr)
            errq.put({'Save error': str(path)})
        finally:
            o3d.utility.set_verbosity_level(original_verbosity)

    save_thread = threading.Thread(target=_save_geometry,
                                   args=(save_path, geometry, error_queue))
    save_thread.daemon = True
    save_thread.start()
    return save_thread


def save_plot_threaded(fig, output_plot_path, error_queue):
    """
    Saves a matplotlib plot in a separate thread.
    """

    def _save_plot(errq):
        try:
            os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
            fig.savefig(output_plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"\nError saving plot to {output_plot_path}: {e}")
            errq.put({'Save plot error': output_plot_path})

    plot_thread = threading.Thread(target=_save_plot, args=(error_queue))
    plot_thread.daemon = True
    plot_thread.start()
    return plot_thread


### FILTERING ###


# yapf: disable
def filter_data_on_condition(
    root: str = "",
    pottery_path: str = "",
    preprocess: bool = True,
    mode: int = 0, # 'HEATMAP, QNA, VOICE': 0 | 'HEATMAP, QNA': 1 | 'HEATMAP, VOICE': 2 | 'HEATMAP': 3
    hololens_2_spatial_error: float = DEFAULT_HOLOLENS_2_SPATIAL_ERROR,
    target_voxel_resolution: int = DEFAULT_TARGET_VOXEL_RESOLUTION,
    qna_answer_color_map: dict = DEFAULT_QNA_ANSWER_COLOR_MAP,
    base_color: list = DEFAULT_BASE_COLOR,
    cmap = DEFAULT_CMAP,
    groups: list = [],
    session_ids: list = [],
    pottery_ids: list = [],
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
    """
    Checks all paths from the root directory -> group -> session -> pottery/dogu -> raw data.
    Apply filters from (tracking sheet, arguments).
    Based on preprocess, use_cache the function will generate the training data in processed folder.
    Finally returns a list of dictionaries that provide the path to all training data.

    TODO: Implement a system to filter according to text language before group

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
        data (list[dict]): A list of dictionaries containing the path to processed data and raw data
        errors (list[dict]): A list of dictionaries containing all errors that occured, each dictionary has the count and list of paths that has errors
    """
    active_threads = []

    GAUSSIAN_DENOMINATOR = 2 * (hololens_2_spatial_error**2)

    # Store {ERROR : Number of instance} pairs
    # ERROR LIST:
    # Eye gaze point cloud generation
    # Eye gaze heatmap generation
    # QNA Point cloud generation
    # QNA Segmented heatmap generation
    # Voice processing
    # Model path does not exist
    # Point cloud path does not exist
    # QNA path does not exist
    # Voice path does not exist
    # Save error
    # Tracking sheet, data mismatch
    # Missing pottery / dogu regardless of file extension
    #
    # FORMAT:
    # "ERROR": {
    #   count: int
    #   paths: list
    # }
    errors = {}
    error_queue = queue.Queue()

    # Check if each data instance / file path exists
    data = []
    pottery_id_to_path = {}
    if not Path(root).exists():
        raise (ValueError(f"Root directory not found: {root}"))
    if not Path(pottery_path).exists():
        raise (ValueError(f"Pottery directory not found: {pottery_path}"))
    processed_dir = "\\".join(Path(root).parts[:-1]) / Path('processed')
    processed_pottery_dir = processed_dir / Path(pottery_dirname)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(processed_pottery_dir, exist_ok=True)

    pottery_ids = [f"{pid}({ASSIGNED_NUMBERS_DICT[pid]})" for pid in pottery_ids]

    print(f"\nCHECKING POTTERY PATHS")
    unique_pottery_dogu_voxel = set([])
    pottery_id_all = [f"{pid}({num})" for pid, num in ASSIGNED_NUMBERS_DICT.items()]
    pottery_available = os.listdir(pottery_path)
    pottery_id_available = [p.split(".")[0] for p in pottery_available]
    for p in tqdm(pottery_id_all, desc="POTTERY & DOGU"):
        if p in pottery_id_available:
            pottery_id_to_path[p] = pottery_path / Path(pottery_available[pottery_id_available.index(p)])
        else:
            pottery_id_to_path[p] = ""
            errors = increment_error('Missing pottery / dogu regardless of file extension', str(pottery_path / Path(f"{p}.*")), errors)

    # Filter based on group, session, model
    print(f"\nCHECKING RAW DATA PATHS")
    unique_group_keys = set([])
    unique_session_keys = set([])
    unique_pottery_keys = set([])

    group_keys = os.listdir(root)
    unique_group_keys.update(group_keys)
    for g in group_keys:
        group_path = root / Path(g)
        processed_group_path = processed_dir / Path(g)

        session_keys = os.listdir(group_path)
        unique_session_keys.update(session_keys)
        for s in tqdm(session_keys, desc=g):
            session_path = group_path / Path(s)
            processed_session_path = processed_group_path / Path(s)

            pottery_keys = os.listdir(session_path)
            unique_pottery_keys.update(pottery_keys)
            for p in pottery_keys:
                hm_error = False
                qna_error = False
                voice_error = False
                data_paths = {}
                pottery_path = session_path / Path(p)
                processed_pottery_path = processed_session_path / Path(p)

                pointcloud_path = pottery_path / Path("pointcloud.csv")
                qa_path = pottery_path / Path("qa.csv")
                model_path = pottery_path / Path("model.obj")
                voice_path = pottery_path / Path("session_audio_0.wav")

                output_sanity_plot = processed_pottery_path / f"{sanity_plot_filename}.png"
                output_point_cloud = processed_pottery_path / f"{eg_pointcloud_filename}{MESH_PC_VOXEL_EXTENSION}"
                output_heatmap = processed_pottery_path / f"{eg_heatmap_filename}{MESH_PC_VOXEL_EXTENSION}"
                output_voxel = processed_pottery_path / f"{voxel_filename}{MESH_PC_VOXEL_EXTENSION}"
                output_qa_pc = processed_pottery_path / f"{qa_pc_filename}{MESH_PC_VOXEL_EXTENSION}"
                output_segmented_meshes_dir = processed_pottery_path / segmented_meshes_dirname
                output_combined_mesh_file = processed_pottery_path / f"{combined_mesh_filename}{MESH_PC_VOXEL_EXTENSION}"
                output_voice = processed_pottery_path / f"{processed_voice_filename}{VOICE_EXTENSION}"

                # Check if paths exist and increment error
                if Path(model_path).exists():
                    data_paths['model'] = str(model_path)
                    if Path(pointcloud_path).exists():
                        data_paths['pointcloud'] = str(pointcloud_path)
                        data_paths['POINTCLOUD_SIZE_KB'] = os.path.getsize(pointcloud_path)/1024
                        data_paths[sanity_plot_filename] = str(output_sanity_plot)
                        data_paths[eg_pointcloud_filename] = str(output_point_cloud)
                        data_paths[eg_heatmap_filename] = str(output_heatmap)
                        data_paths[voxel_filename] = str(output_voxel)
                    else:
                        hm_error = True
                        errors = increment_error('Point cloud path does not exist', str(pointcloud_path), errors)

                    if Path(qa_path).exists():
                        data_paths['qa'] = str(qa_path)
                        data_paths['QA_SIZE_KB'] = os.path.getsize(qa_path)/1024
                        data_paths[qa_pc_filename] = str(output_qa_pc)
                        data_paths[segmented_meshes_dirname] = str(output_segmented_meshes_dir)
                        data_paths[combined_mesh_filename] = str(output_combined_mesh_file)
                    else:
                        data_paths['QA_SIZE_KB'] = 0
                        qna_error = True
                        errors = increment_error('QNA path does not exist', str(qa_path), errors)
                else:
                    hm_error = True
                    qna_error = True
                    errors = increment_error('Model path does not exist', str(model_path), errors)

                if Path(voice_path).exists():
                    data_paths['voice'] = str(voice_path)
                    data_paths[processed_voice_filename] = str(output_voice)
                else:
                    voice_error = True
                    errors = increment_error('Voice path does not exist', str(voice_path), errors)

                pottery_dogu_path = pottery_id_to_path[p]
                if (hm_error):
                    continue
                elif (qna_error and (mode==0 or mode==1)):
                    continue
                elif (voice_error and (mode==0 or mode==2)):
                    continue
                elif pottery_dogu_path == "":
                    continue
                else:
                    data_paths['GROUP'] = g
                    data_paths['SESSION_ID'] = s
                    data_paths['ID'] = p
                    data_paths['processed_pottery_path'] = str(processed_pottery_path)
                    unique_pottery_dogu_voxel.add(str(pottery_dogu_path))
                    data.append(data_paths)

    n_valid_data = len(data)

    # Filtering based on pointcloud, qa size and voice quality
    print("\nFILTERING")
    if from_tracking_sheet:
        if not Path(tracking_sheet_path).exists():
            from_tracking_sheet = False
            raise(ValueError(f"Tracking sheet at {tracking_sheet_path} does not exist."))

    if min_voice_quality >= 1.0 and not (from_tracking_sheet and Path(tracking_sheet_path).exists()):
        raise(ValueError(f"To filter with voice quality, a tracking sheet is needed"))

    # Filter from tracking sheet
    n_filtered_from_tracking_sheet = 0
    if from_tracking_sheet:
        tracking_sheet = pd.read_csv(tracking_sheet_path, header=0)
        tracking_sheet = tracking_sheet[tracking_sheet.get('VOICE_QUALITY_0_TO_5') > min_voice_quality]
        unique_groups = np.unique(tracking_sheet.get('GROUP'))
        unique_session = np.unique(tracking_sheet.get('SESSION_ID'))
        unique_pottery_id = [f"{pid}({ASSIGNED_NUMBERS_DICT[pid]})" for pid in np.unique(tracking_sheet['ID'])]
        keep_data_index = np.array(np.zeros(len(data)), dtype=bool)

        for i, data_paths in enumerate(tqdm(data, desc="TRACKING SHEET")):
            if data_paths['GROUP'] in unique_groups \
                and data_paths['SESSION_ID'] in unique_session \
                and data_paths['ID'] in unique_pottery_id:
                keep_data_index[i] = True

        data = np.array(data)[keep_data_index]
        n_filtered_from_tracking_sheet = n_valid_data - len(data)
        print(f"{n_filtered_from_tracking_sheet} instances filtered from tracking sheet. Check final PDF report for more details.")

    # Filter from arguments
    n_filtered_from_arguments = 0
    if (mode==0 or mode==1):
        data = filter_qna_by_emotion_count(data, min_emotion_count=min_emotion_count)
    if len(groups) > 0: unique_group_keys = list(unique_group_keys & set(groups))
    unique_group_keys = list(unique_group_keys)
    if len(session_ids) > 0: unique_session_keys = list(unique_session_keys & set(session_ids))
    unique_session_keys = list(unique_session_keys)
    if len(pottery_ids) > 0: unique_pottery_keys = list(unique_pottery_keys & set(pottery_ids))
    unique_pottery_keys = list(unique_pottery_keys)
    keep_data_index = np.array(np.zeros(len(data)), dtype=bool)

    for i, data_paths in enumerate(tqdm(data, desc="FUNCTION ARGUMENTS")):
        if data_paths['GROUP'] in unique_group_keys \
            and data_paths['SESSION_ID'] in unique_session_keys \
            and data_paths['ID'] in unique_pottery_keys \
            and data_paths['POINTCLOUD_SIZE_KB'] >= min_pointcloud_size \
            and data_paths['QA_SIZE_KB'] >= min_qa_size:
            keep_data_index[i] = True

    data = np.array(data)[keep_data_index]
    n_filtered_from_arguments = n_valid_data - n_filtered_from_tracking_sheet - len(data)
    print(f"{n_filtered_from_arguments} instances filtered from provided function arguments. Check final PDF report for more details.")

    # Finalizing the data paths
    # Generate Pottery & Dogu Voxels
    print(f"\nGENERATING POTTERY & DOGU VOXELS")
    for data_path in tqdm(list(unique_pottery_dogu_voxel)):
        if (generate_pc_hm_voxel or generate_qna) and generate_pottery_dogu_voxel:
            save_path = processed_pottery_dir / f"{str(data_path).split("\\")[-1].split(".")[0]}{MESH_PC_VOXEL_EXTENSION}"
            if use_cache and Path(save_path).exists():
                pass
            else:
                pottery_dogu_voxel = voxelize_pottery_dogu(
                    input_file=data_path,
                    target_voxel_resolution=target_voxel_resolution,
                )    
                active_threads.append(save_geometry_threaded(save_path, pottery_dogu_voxel, error_queue))

    # Preprocessed
    if (preprocess):
        print(f"\nPREPROCESSING")
        for data_paths in tqdm(data):
            os.makedirs(data_paths['processed_pottery_path'], exist_ok=True)

            if generate_pc_hm_voxel:
                # Eye gaze intensity point cloud & heatmap
                # Eye gaze voxel
                if use_cache and Path(data_paths[eg_pointcloud_filename]).exists() \
                    and Path(data_paths[eg_heatmap_filename]).exists() \
                    and Path(data_paths[voxel_filename]).exists():
                    pass
                else:
                    eye_gaze_pointcloud, eye_gaze_heatmap_mesh, final_vertex_intensities = generate_gaze_pointcloud_heatmap(
                        input_file=data_paths['pointcloud'],
                        model_file=data_paths['model'],
                        cmap=cmap,
                        base_color=base_color,
                        hololens_2_spatial_error=hololens_2_spatial_error,
                        gaussian_denominator=GAUSSIAN_DENOMINATOR
                    )
                    active_threads.append(save_geometry_threaded(data_paths[eg_pointcloud_filename], eye_gaze_pointcloud, error_queue))
                    active_threads.append(save_geometry_threaded(data_paths[eg_heatmap_filename], eye_gaze_heatmap_mesh, error_queue))

                    eye_gaze_voxel = generate_voxel_from_mesh(
                        mesh=eye_gaze_heatmap_mesh,
                        vertex_intensities=final_vertex_intensities,
                        target_voxel_resolution=target_voxel_resolution,
                        cmap=cmap,
                        base_color=base_color,
                    )
                    active_threads.append(save_geometry_threaded(data_paths[voxel_filename], eye_gaze_voxel, error_queue))

            if generate_qna and (mode==0 or mode==1):
                # QNA combined point cloud
                # QNA segmented mesh
                if use_cache and Path(data_paths[qa_pc_filename]).exists() \
                    and Path(data_paths[segmented_meshes_dirname]).exists() \
                    and Path(data_paths[combined_mesh_filename]).exists():
                    pass
                else:
                    qa_pointcloud, qa_segmented_mesh, combined_mesh = process_questionnaire_answeres(
                        input_file=data_paths['qa'],
                        model_file=data_paths['model'],
                        base_color=base_color,
                        qna_answer_color_map=qna_answer_color_map,
                        hololens_2_spatial_error=hololens_2_spatial_error,
                        gaussian_denominator=GAUSSIAN_DENOMINATOR
                    )
                    active_threads.append(save_geometry_threaded(data_paths[qa_pc_filename], qa_pointcloud, error_queue))
                    active_threads.append(save_geometry_threaded(data_paths[combined_mesh_filename], combined_mesh, error_queue))

                    os.makedirs(data_paths[segmented_meshes_dirname], exist_ok=True)
                    for k in qa_segmented_mesh.keys():
                        segmented_mesh = qa_segmented_mesh[k]
                        individual_segment = data_paths[segmented_meshes_dirname] / Path(f"{k}.ply")
                        active_threads.append(save_geometry_threaded(individual_segment, segmented_mesh, error_queue))

            if generate_voice and (mode==0 or mode==2):
                # Voice
                if use_cache and Path(data_paths[processed_voice_filename]).exists():
                    pass
                else:
                    waveform, sample_rate = process_voice_data(data_paths['voice'])
                    torchaudio.save(data_paths[processed_voice_filename], waveform, sample_rate)

    # In-time processing, only return path to raw data
    else:
        pass

    for t in active_threads:
        t.join()

    try:
        # Pull all available errors from the queue without blocking
        while True:
            error_list = error_queue.get_nowait()
            for dictionary in error_list:
                for key, val in dictionary.items():
                    increment_error(key, val, errors)
    except queue.Empty:
        pass

    if (generate_report):
        generate_filtered_dataset_report(
            errors=errors,
            mode=mode,
            hololens_2_spatial_error=hololens_2_spatial_error,
            base_color=base_color,
            groups=groups,
            session_ids=session_ids,
            pottery_ids=pottery_ids,
            min_pointcloud_size=min_pointcloud_size,
            min_qa_size=min_qa_size,
            min_voice_quality=min_voice_quality,
            min_emotion_count=min_emotion_count,
            from_tracking_sheet=from_tracking_sheet,
            tracking_sheet_path=tracking_sheet_path,
            n_filtered_from_tracking_sheet=n_filtered_from_tracking_sheet,
            n_filtered_from_arguments=n_filtered_from_arguments,
            n_valid_data=n_valid_data,
            filtered_data=data,
        )

    return data, errors
# yapf: enable


def filter_qna_by_emotion_count(data: list, min_emotion_count: int = 1):
    """
    Filters a list of data dictionaries based on the number of unique emotions
    (answers) in their associated QNA file.

    This function iterates through each data entry, reads its corresponding 
    questionnaire CSV file, and counts the number of distinct answers provided.
    It returns a new list containing only the data entries that meet or exceed
    the specified minimum count of unique emotions.

    Args:
        data (list): A list of dictionaries, where each dictionary
                           represents a data point and must contain a 'qa' key
                           with the path to the questionnaire CSV file.
        min_emotion_count (int, optional): The minimum number of unique emotions
                                           required for a data point to be
                                           kept. Defaults to 1.

    Returns:
        filtered_data: A new list containing only the data dictionaries that
                    meet the minimum emotion count criterion.
    """
    filtered_data = []
    for data_item in tqdm(data, desc="QNA UNIQUE COUNT"):
        qna_path = data_item.get('qa')

        # Skip this item if the 'qa' key is missing or the file doesn't exist
        if not qna_path or not os.path.exists(qna_path):
            continue

        try:
            df = pd.read_csv(qna_path)

            if 'answer' not in df.columns:
                continue

            unique_emotion_count = df['answer'].nunique()

            if unique_emotion_count >= min_emotion_count:
                filtered_data.append(data_item)

        except pd.errors.EmptyDataError:
            continue
        except Exception as e:
            continue

    return filtered_data


### PROCESS DATA ###


# yapf: disable
def generate_gaze_pointcloud_heatmap(
    input_file,
    model_file,
    cmap,
    base_color,
    hololens_2_spatial_error,
    gaussian_denominator,
):
    data = pd.read_csv(input_file, header=0).to_numpy()
    gaze_points_np = data[:, :3]
    mesh = o3d.io.read_triangle_mesh(model_file)
    if not mesh.has_vertices():
        raise ValueError(f"Mesh file '{model_file}' contains no vertices.")
    mesh.compute_vertex_normals()

    final_vertex_intensities, point_to_face_map = _calculate_smoothed_vertex_intensities(
        gaze_points_np=gaze_points_np,
        mesh=mesh,
        hololens_2_spatial_error=hololens_2_spatial_error,
        gaussian_denominator=gaussian_denominator,
    )

    # Generate Heatmap Mesh
    max_mesh = np.max(final_vertex_intensities)
    normalized_vertex_intensities = final_vertex_intensities / max_mesh

    mesh_vertex_colors = cmap(normalized_vertex_intensities)[:, :3]
    mesh_vertex_colors[final_vertex_intensities < 1e-9] = base_color
    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_vertex_colors)

    # Generate Intensity Point Cloud
    mesh_triangles_np = np.asarray(mesh.triangles)
    final_point_intensities = []
    for face_idx in tqdm(point_to_face_map, desc="Making Intensity Point Cloud", leave=False):
        final_point_intensities.append(np.mean(final_vertex_intensities[mesh_triangles_np[face_idx]]))
    final_point_intensities = np.array(final_point_intensities)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gaze_points_np))
    max_pc = np.max(final_point_intensities)
    normalized_point_intensities = final_point_intensities / max_pc

    pc_colors = cmap(normalized_point_intensities)[:, :3]
    pc_colors[final_point_intensities < 1e-9] = base_color
    pcd.colors = o3d.utility.Vector3dVector(pc_colors)

    return pcd, mesh, final_vertex_intensities
# yapf: enable


# yapf: disable
def generate_voxel_from_mesh(
    mesh,
    vertex_intensities,
    target_voxel_resolution,
    cmap,
    base_color,
):
    if mesh is None or vertex_intensities is None:
        raise ValueError("Skipping voxel heatmap: Missing mesh or intensity data.")

    if not mesh.has_triangles() or not mesh.has_vertices():
        raise ValueError("Skipping voxel heatmap: Mesh has no triangles or vertices.")

    # Initial Setup (Vectorized)
    mesh_vertices_np = np.asarray(mesh.vertices)
    mesh_triangles_np = np.asarray(mesh.triangles)
    min_bound = mesh.get_min_bound()
    max_bound = mesh.get_max_bound()
    max_range = np.max(max_bound - min_bound)
    voxel_size = max_range / (target_voxel_resolution - 1)
    voxel_size_sq = voxel_size**2

    # Calculate Adaptive Sample Counts for ALL Triangles at Once
    # Get vertices and intensities for all triangles: shape (num_triangles, 3, 3) and (num_triangles, 3)
    tri_vertices = mesh_vertices_np[mesh_triangles_np]
    tri_intensities = vertex_intensities[mesh_triangles_np]

    # Calculate areas for all triangles to determine sample density
    # Get the 3D coordinates of each vertex
    v0, v1, v2 = tri_vertices[:, 0], tri_vertices[:, 1], tri_vertices[:, 2]
    edge1, edge2 = v1 - v0, v2 - v0
    # Adaptive Sampling Density
    # To ensure larger triangles are adequately filled with voxels, we sample them more densely
    # The number of samples is proportional to the triangle's area
    # AREA = 1/2 * ||E1 x E2||
    triangle_areas = 0.5 * np.linalg.norm(np.cross(edge1, edge2), axis=1)
    
    # Number of sample, with minimum of 10, to ensure small triangles get sample as well
    num_samples_per_triangle = np.ceil(triangle_areas / voxel_size_sq).astype(int) + 10
    total_samples = np.sum(num_samples_per_triangle)

    # Generate and Interpolate ALL Sample Points at Once
    # Create an index to map each sample back to its original triangle
    triangle_indices = np.repeat(np.arange(len(mesh_triangles_np)), num_samples_per_triangle)

    # Barycentric Coordinate Generation for Linear Interpolation
    # Reading a bit about barycentric coordinate system: https://www.sciencedirect.com/topics/computer-science/barycentric-coordinate#:~:text=3.5%20Barycentric%20Coordinates%20in%20the%20Plane
    #
    # Since we do not know the exact positions of points, we just have to make sure that
    # the generated barycentric coordinates satisfy the condition where u + v = w = 1
    #
    # Generate random points within a square, then fold them into a triangle
    # This is an efficient way to get uniformly distributed points
    # within a right angle unit triangle [(0, 0), (1, 0), (0, 1)]
    # Hence, the bounding linear equations of the triangle are
    # y = 0, x = 0, y = -x + 1
    # To satisfy the condition of points being in the triangle,
    # y > 0, x > 0 and y + x < 1
    #
    # Generate random points i.e. [(0.2, 0.4), (0.7, 0.6)]
    rand_points = np.random.rand(total_samples, 2)
    # Sum i.e. [0.6, 1.3]
    rand_points_sum = np.sum(rand_points, axis=1)
    # Fold points from outside the triangles= back inside
    # i.e. [(0.2, 0.4), (0.3, 0.4)]
    rand_points[rand_points_sum > 1] = 1 - rand_points[rand_points_sum > 1]
    
    # Convert the random points into barycentric coordinates (u, w, v) | (w0, w1, w2)
    # Barycentric coordinates are weights for each vertex, summing to 1
    # u + v + w = 1
    #
    # i.e. (0.2, 0.4)
    # u = 1 - 0.2 - 0.4 = 0.4
    # v = 0.2
    # w = 0.4
    # u + v + w = 0.4 + 0.2 + 0.4 = 1
    bary_coords = np.zeros((total_samples, 3))
    bary_coords[:, 0] = 1 - rand_points[:, 0] - rand_points[:, 1] # u | w0
    bary_coords[:, 1] = rand_points[:, 0]  # v | w1
    bary_coords[:, 2] = rand_points[:, 1]  # w | w2

    # Interpolate positions and intensities for all points using the barycentric coordinates
    # np.einsum is a fast way to do batched weighted averages
    all_sample_points = np.einsum('ij,ijk->ik', bary_coords, tri_vertices[triangle_indices])
    # Creates a smooth gradient of intensity across the triangle
    # Since all three vertex may have a different intensity
    all_interpolated_intensities = np.einsum('ij,ij->i', bary_coords, tri_intensities[triangle_indices])

    # Voxel Assignment using Pandas
    # Calculate voxel coordinates for all points in a single operation
    voxel_coords_all = np.floor((all_sample_points - min_bound) / voxel_size).astype(int)

    # Create a DataFrame to hold voxel coordinates and their intensities
    df = pd.DataFrame(voxel_coords_all, columns=['x', 'y', 'z'])
    df['intensity'] = all_interpolated_intensities
    
    # Use groupby().max() to find the maximum intensity for each unique voxel
    voxel_data_df = df.groupby(['x', 'y', 'z'])['intensity'].max()

    # Final Processing (Vectorized)
    final_coords_np = np.array(voxel_data_df.index.to_list())
    final_intensities_np = voxel_data_df.to_numpy()

    # Calculate world coordinates of voxel centers
    voxel_points = min_bound + (final_coords_np + 0.5) * voxel_size
    
    # Normalize and apply colormap
    max_val = np.max(final_intensities_np)
    if max_val > 1e-9:
        normalized_intensities = final_intensities_np / max_val
    else:
        normalized_intensities = np.zeros_like(final_intensities_np)
        
    colors = cmap(normalized_intensities)[:, :3]
    colors[normalized_intensities < 1e-9] = base_color

    # Create Final Point Cloud
    voxel_pcd = o3d.geometry.PointCloud()
    voxel_pcd.points = o3d.utility.Vector3dVector(voxel_points)
    voxel_pcd.colors = o3d.utility.Vector3dVector(colors)

    return voxel_pcd
# yapf: enable


# yapf: disable
def process_questionnaire_answeres(
    input_file,
    model_file,
    base_color,
    qna_answer_color_map,
    hololens_2_spatial_error,
    gaussian_denominator,
):
    qa_segmented_meshes = {}

    df = pd.read_csv(input_file, header=0, sep=",")
    df["estX"] = pd.to_numeric(df["estX"], errors="coerce")
    df["estY"] = pd.to_numeric(df["estY"], errors="coerce")
    df["estZ"] = pd.to_numeric(df["estZ"], errors="coerce")
    df["answer"] = df["answer"].astype(str).str.strip()
    df.dropna(subset=["estX", "estY", "estZ", "answer"], inplace=True)
    mesh = o3d.io.read_triangle_mesh(model_file)
    if not mesh.has_vertices():
        raise ValueError(f"Mesh file '{model_file}' contains no vertices.")

    # Segmented QNA mesh
    mesh_vertices_np = np.asarray(mesh.vertices)
    mesh_triangles_np = np.asarray(mesh.triangles)
    n_vertices = mesh_vertices_np.shape[0]

    qa_points = df[["estX", "estY", "estZ"]].values
    qa_colors_01 = []
    for answer in df["answer"]:
        qa_colors_01.append(qna_answer_color_map.get(answer)["rgb"])
    qa_colors_01 = np.array(qa_colors_01) / 255.0
    qa_pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(qa_points))
    qa_pcd.colors = o3d.utility.Vector3dVector(np.array(qa_colors_01))

    mesh_kdtree = o3d.geometry.KDTreeFlann(mesh)
    mesh_scene = o3d.t.geometry.RaycastingScene()
    mesh_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    final_colors = np.zeros((n_vertices, 3), dtype=float)
    total_weights = np.zeros(n_vertices, dtype=float)
    unique_answers = df["answer"].unique()
    all_hit_vertices_by_answer = {}

    for answer in unique_answers:
        category_df = df[df["answer"] == answer]
        category_points = category_df[["estX", "estY", "estZ"]].values
        color_vector = np.array(qna_answer_color_map.get(answer)["rgb"]) / 255.0
        hit_vertices = set()

        query_points = o3d.core.Tensor(category_points, dtype=o3d.core.Dtype.Float32)
        closest_geometry = mesh_scene.compute_closest_points(query_points)
        closest_face_indices = closest_geometry["primitive_ids"].numpy()

        for closest_face_idx in tqdm(closest_face_indices, desc=f"{answer} | Mapping Hits", leave=False):
            if closest_face_idx != o3d.t.geometry.RaycastingScene.INVALID_ID:
                for v_idx in mesh_triangles_np[closest_face_idx]:
                    hit_vertices.add(v_idx)
        all_hit_vertices_by_answer[answer] = hit_vertices

        for start_node_idx in tqdm(list(hit_vertices), desc="Applying Gaussian Spread", leave=False):
            # Find all vertices within the specified radius
            [k, indices, euclidean_distance] = mesh_kdtree.search_radius_vector_3d(mesh_vertices_np[start_node_idx], hololens_2_spatial_error)

            if k > 0:
                # Calculate Gaussian weights based on squared Euclidean distance
                gaussian_weights = np.exp(-np.asarray(euclidean_distance)**2 / gaussian_denominator)

                # Apply weighted colors to all neighbors found in the radius
                final_colors[indices] += color_vector * gaussian_weights[:, np.newaxis]
                total_weights[indices] += gaussian_weights

    valid_weights_mask = total_weights > 1e-9
    final_colors[valid_weights_mask] /= total_weights[valid_weights_mask, np.newaxis]
    final_colors[~valid_weights_mask] = base_color # Negation of bool mask

    combined_mesh = o3d.geometry.TriangleMesh(mesh)
    combined_mesh.vertex_colors = o3d.utility.Vector3dVector(final_colors)

    for answer, hit_vertices in tqdm(all_hit_vertices_by_answer.items(), desc="Making Segmented Maps", leave=False):
        color_info = qna_answer_color_map.get(answer)
        color_vec = np.array(color_info["rgb"]) / 255.0
        segmented_colors = np.zeros((n_vertices, 3), dtype=float)

        for start_node_idx in list(hit_vertices):
            [k, indices, _] = mesh_kdtree.search_radius_vector_3d(mesh_vertices_np[start_node_idx], hololens_2_spatial_error)
            if k > 0:
                # Assign the solid color to all vertices found within the radius
                segmented_colors[indices] = color_vec

        segmented_mesh = o3d.geometry.TriangleMesh(mesh)
        segmented_mesh.vertex_colors = o3d.utility.Vector3dVector(segmented_colors)
        qa_segmented_meshes[qna_answer_color_map[answer]["name"]] = segmented_mesh

    return qa_pcd, qa_segmented_meshes, combined_mesh
# yapf: enable


def process_voice_data(input_file):
    if not Path(input_file).exists():
        return

    # 2646000 / 60 * 45 = 1992000
    waveform, sample_rate = torchaudio.load(input_file, num_frames=1992000)

    return waveform, sample_rate


# yapf: disable
def voxelize_pottery_dogu(input_file, target_voxel_resolution):
    # Load Mesh and Prepare Data
    scene = trimesh.load(str(input_file), force="scene")
    mesh_trimesh = trimesh.util.concatenate(scene.geometry.values())
    vertex_color_trimesh = mesh_trimesh.visual.to_color().vertex_colors

    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh_trimesh.vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh_trimesh.faces)
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_color_trimesh[:, :3] / 255.0)

    mesh_vertices_np = np.asarray(mesh_o3d.vertices)
    mesh_vertex_colors_np = np.asarray(mesh_o3d.vertex_colors)
    mesh_triangles_np = np.asarray(mesh_o3d.triangles)

    min_bound = mesh_o3d.get_min_bound()
    max_bound = mesh_o3d.get_max_bound()
    max_range = np.max(max_bound - min_bound)
    voxel_size = max_range / (target_voxel_resolution - 1)
    voxel_size_sq = voxel_size**2

    # Vectorized Rasterization
    # Get vertices and colors for all triangles at once
    tri_vertices = mesh_vertices_np[mesh_triangles_np]
    tri_colors = mesh_vertex_colors_np[mesh_triangles_np]

    # Calculate areas for all triangles to determine sample density
    v0, v1, v2 = tri_vertices[:, 0], tri_vertices[:, 1], tri_vertices[:, 2]
    triangle_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    num_samples_per_triangle = np.ceil(triangle_areas / voxel_size_sq).astype(int) + 10
    total_samples = np.sum(num_samples_per_triangle)

    # Vectorized Barycentric Sampling and Interpolation
    # Create an index to map each sample back to its original triangle
    triangle_indices = np.repeat(np.arange(len(mesh_triangles_np)), num_samples_per_triangle)

    # Generate barycentric coordinates for all sample points at once
    r = np.random.rand(total_samples, 2)
    r_sum = np.sum(r, axis=1)
    r[r_sum > 1] = 1 - r[r_sum > 1]
    bary_coords = np.zeros((total_samples, 3))
    bary_coords[:, [1, 2]] = r
    bary_coords[:, 0] = 1 - np.sum(r, axis=1)

    # Interpolate positions and colors for all points using the barycentric coordinates
    # np.einsum is a fast way to do this batched weighted average
    all_sample_points = np.einsum('ij,ijk->ik', bary_coords, tri_vertices[triangle_indices])
    all_interp_colors = np.einsum('ij,ijk->ik', bary_coords, tri_colors[triangle_indices])

    # Voxel Assignment using Pandas
    # Calculate voxel grid coordinates for all points at once
    voxel_coords_all = np.floor((all_sample_points - min_bound) / voxel_size).astype(int)

    # Create a DataFrame to hold voxel coordinates and their interpolated colors
    df = pd.DataFrame(voxel_coords_all, columns=['x', 'y', 'z'])
    df[['r', 'g', 'b']] = all_interp_colors

    # Use groupby().mean() to find the average color for each unique voxel
    # This replaces the inefficient list-building and vstacking
    voxel_data_df = df.groupby(['x', 'y', 'z'])[['r', 'g', 'b']].mean()

    # Final Point Cloud Generation
    # Extract the unique voxel coordinates and their final averaged colors
    final_coords = voxel_data_df.index.to_numpy()
    final_colors_np = voxel_data_df.to_numpy()

    # Calculate world coordinates of voxel centers
    final_coords_np = np.stack(final_coords)
    voxel_points = min_bound + (final_coords_np + 0.5) * voxel_size

    voxel_pcd = o3d.geometry.PointCloud()
    voxel_pcd.points = o3d.utility.Vector3dVector(voxel_points)
    voxel_pcd.colors = o3d.utility.Vector3dVector(final_colors_np)

    return voxel_pcd
# yapf: enable

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


def analyze_and_plot_point_cloud(csv_file_path, output_plot_path, error_queue):
    """
    Reads a CSV file with 3D point data, counts occurrences, and saves a 3D scatter plot.
    """
    df = pd.read_csv(csv_file_path, header=0, usecols=[0, 1, 2])
    df.columns = ['x', 'y', 'z']

    if df.empty:
        return

    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df['z'] = pd.to_numeric(df['z'], errors='coerce')
    df.dropna(inplace=True)

    if df.empty:
        return

    point_counts = df.groupby(['x', 'y', 'z']).size().reset_index(name='count')

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Swap Z and Y to retain upright orientation
    scatter = ax.scatter(
        point_counts['x'],
        point_counts['z'],  # Z is now plotted on the Y-axis of the plot
        point_counts['y'],  # Y is now plotted on the Z-axis of the plot
        c=point_counts['count'],
        s=point_counts['count'] * 5 + 10,
        cmap='viridis',
        alpha=0.8,
        edgecolors='k',
        linewidth=0.5,
    )

    ax.set_title(
        '3D Distribution of Gaze Points (Colored by Occurrence Count)')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Z Coordinate')
    ax.set_zlabel('Y Coordinate')
    cbar = fig.colorbar(scatter, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Occurrence Count')

    ax.view_init(elev=30, azim=-45)

    try:
        max_range = np.array([
            df['x'].max() - df['x'].min(),
            df['y'].max() - df['y'].min(),
            df['z'].max() - df['z'].min(),
        ]).max()
        mid_x = (df['x'].max() + df['x'].min()) * 0.5
        mid_y = (df['y'].max() + df['y'].min()) * 0.5
        mid_z = (df['z'].max() + df['z'].min()) * 0.5
        ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
        ax.set_ylim(mid_z - max_range * 0.5, mid_z +
                    max_range * 0.5)  # Y-axis of plot gets Z-data limits
        ax.set_zlim(mid_y - max_range * 0.5, mid_y +
                    max_range * 0.5)  # Z-axis of plot gets Y-data limits
    except ValueError:
        print("Could not determine axis range automatically.")
    ax.grid(True)

    # Move the plot saving to a separate thread
    plot_thread = threading.Thread(target=save_plot_threaded,
                                   args=(fig, output_plot_path, error_queue))
    plot_thread.daemon = True


### PDF Report | AI Generated Visualization Functions ###


def _create_cmap_image(cmap, width=1.5 * inch, height=0.2 * inch):
    """Generates an image of a matplotlib colormap in an in-memory buffer."""
    fig, ax = plt.subplots(figsize=(width / inch, height / inch))
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    ax.set_axis_off()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf


def _generate_analysis_plots(
    data: np.ndarray,
    tracking_sheet_path: str = None,
):
    """
    Uses pandas and matplotlib to generate quantitative analysis plots.
    Returns a list of in-memory image buffers.
    """
    if data.size == 0:
        return []

    try:
        df = pd.DataFrame(list(data))
    except Exception:
        return []

    plots = []

    # --- Plot 1: Count of Pottery ---
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(11, 4))
        id_counts = df['ID'].value_counts().sort_index()
        id_counts.plot(kind='bar', ax=ax, color='steelblue', zorder=2)
        ax.set_title('Count of Data Instances per Pottery ID', fontsize=14)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_xlabel('Pottery ID', fontsize=10)
        plt.xticks(rotation=90, fontsize=8)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plots.append({'title': 'Instance Count Distribution', 'buffer': buf})
        plt.close(fig)
    except Exception:
        pass

    # --- Plot 2: Average Data Sizes ---
    try:
        fig, ax = plt.subplots(figsize=(11, 4))
        avg_sizes = df.groupby('ID')[['POINTCLOUD_SIZE_KB',
                                      'QA_SIZE_KB']].mean().round(2)
        avg_sizes.plot(kind='bar', ax=ax, zorder=2)
        ax.set_title('Average Data Size per Pottery ID', fontsize=14)
        ax.set_ylabel('Average Size (KB)', fontsize=10)
        ax.set_xlabel('Pottery ID', fontsize=10)
        ax.legend(['PointCloud', 'Q&A'])
        plt.xticks(rotation=90, fontsize=8)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plots.append({
            'title': 'Average Data Size Distribution',
            'buffer': buf
        })
        plt.close(fig)
    except Exception:
        pass

    # --- Plot 3: Average Voice Quality ---
    if tracking_sheet_path and Path(tracking_sheet_path).exists():
        try:
            tracking_df = pd.read_csv(tracking_sheet_path)
            df['SESSION_KEY'] = df['GROUP'] + '_' + df[
                'SESSION_ID'] + '_' + df['ID']
            tracking_df[
                'SESSION_KEY'] = tracking_df['GROUP'] + '_' + tracking_df[
                    'SESSION_ID'] + '_' + tracking_df['ID']
            merged_df = pd.merge(
                df,
                tracking_df[['SESSION_KEY', 'VOICE_QUALITY_0_TO_5']],
                on='SESSION_KEY',
                how='left')

            if not merged_df['VOICE_QUALITY_0_TO_5'].dropna().empty:
                fig, ax = plt.subplots(figsize=(11, 4))
                voice_quality = merged_df.groupby(
                    'ID')['VOICE_QUALITY_0_TO_5'].mean().round(2)
                voice_quality.plot(kind='bar', ax=ax, color='teal', zorder=2)
                ax.set_title('Average Voice Quality per Pottery ID',
                             fontsize=14)
                ax.set_ylabel('Average Quality (0-5)', fontsize=10)
                ax.set_xlabel('Pottery ID', fontsize=10)
                plt.xticks(rotation=90, fontsize=8)
                plt.tight_layout()

                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150)
                buf.seek(0)
                plots.append({
                    'title': 'Average Voice Quality Distribution',
                    'buffer': buf
                })
                plt.close(fig)
        except Exception as e:
            print(f"Could not generate 'Voice Quality' plot: {e}",
                  file=sys.stderr)

    return plots


def generate_filtered_dataset_report(
    errors: dict = {},
    mode: int = 0,
    hololens_2_spatial_error: float = DEFAULT_HOLOLENS_2_SPATIAL_ERROR,
    base_color: list = DEFAULT_BASE_COLOR,
    cmap=DEFAULT_CMAP,
    groups: list = [],
    session_ids: list = [],
    pottery_ids: list = [],
    min_pointcloud_size: float = 0.0,
    min_qa_size: float = 0.0,
    min_voice_quality: float = 0.0,
    min_emotion_count: int = 1,
    from_tracking_sheet: bool = False,
    tracking_sheet_path: str = "",
    n_filtered_from_tracking_sheet: int = 0,
    n_filtered_from_arguments: int = 0,
    n_valid_data: int = 0,
    filtered_data: list[dict] = [],
    output_dir: str = ".",
):
    """
    Generates a PDF report summarizing the data filtering process,
    including quantitative analysis plots.
    """
    report_path = os.path.join(output_dir,
                               "filtering_and_processing_report.pdf")
    doc = SimpleDocTemplate(report_path)
    story = []
    styles = getSampleStyleSheet()

    # --- Define Styles ---
    title_style = ParagraphStyle('Title',
                                 parent=styles['h1'],
                                 fontSize=18,
                                 alignment=TA_CENTER,
                                 spaceAfter=18)
    heading_style = ParagraphStyle('Heading2',
                                   parent=styles['h2'],
                                   fontSize=14,
                                   spaceAfter=12)
    body_style = styles['Normal']

    # --- 1. Title ---
    story.append(Paragraph("Data Filtering and Processing Report",
                           title_style))
    story.append(Spacer(1, 0.2 * inch))

    # --- 2. Summary & Parameters (Combined for brevity) ---
    # --- Summary Section ---
    story.append(Paragraph("1. Summary", heading_style))
    final_count = n_valid_data - n_filtered_from_tracking_sheet - n_filtered_from_arguments
    summary_data = [
        ['Initial Datasets Found:',
         str(n_valid_data)],
        ['Filtered by Tracking Sheet:',
         str(n_filtered_from_tracking_sheet)],
        ['Filtered by Arguments:',
         str(n_filtered_from_arguments)],
        ['Final Datasets for Processing:',
         str(final_count)],
    ]
    summary_table = Table(summary_data, colWidths=[2.5 * inch, 2.5 * inch])
    summary_table.setStyle(
        TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 3), (0, 3), 'Helvetica-Bold'),
            ('FONTNAME', (1, 3), (1, 3), 'Helvetica-Bold'),
        ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.25 * inch))

    # --- Filtering Parameters Section ---
    story.append(Paragraph("2. Filtering Parameters", heading_style))

    def format_list(items):
        return ", ".join(map(str, items)) if items else "Not specified"

    # Create a small colored box for the base_color
    base_color_box = Drawing(20, 10)
    base_color_box.add(
        Rect(0,
             0,
             40,
             10,
             fillColor=colors.Color(*base_color),
             strokeColor=None))
    base_color_display = Table(
        [[Paragraph(str(base_color), body_style), base_color_box]],
        colWidths=[1.0 * inch, None])
    base_color_display.setStyle(
        TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))

    # Create an image of the colormap
    cmap_image_buffer = _create_cmap_image(cmap)
    cmap_display = Table([[
        Paragraph(f"'{cmap.name}'", body_style),
        Image(cmap_image_buffer, width=1.5 * inch, height=0.2 * inch)
    ]],
                         colWidths=[0.7 * inch, 1.6 * inch])
    cmap_display.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))

    params_data = [
        ['Mode:', 'Linient' if mode == 1 else 'Strict'],
        ['HoloLens 2 Spatial Error:', f"{hololens_2_spatial_error}°"],
        ['Base Color:', base_color_display],  # <-- Visual element
        ['Colormap:', cmap_display],  # <-- Visual element
        ['Filter by Tracking Sheet:', 'Yes' if from_tracking_sheet else 'No'],
        [
            'Tracking Sheet Path:',
            Paragraph(tracking_sheet_path, body_style)
            if from_tracking_sheet else 'N/A'
        ],
        [
            'Min Voice Quality (0-5):',
            str(min_voice_quality) if from_tracking_sheet else 'N/A'
        ],
        ['Min Point Cloud Size (KB):',
         str(min_pointcloud_size)],
        ['Min Q&A Size (KB):', str(min_qa_size)],
        ['Min Emotion Count:', str(min_emotion_count)],
        ['Groups:', Paragraph(format_list(groups), body_style)],
        ['Session IDs:',
         Paragraph(format_list(session_ids), body_style)],
        ['Pottery IDs:',
         Paragraph(format_list(pottery_ids), body_style)],
    ]

    params_table = Table(params_data, colWidths=[2.0 * inch, 3.0 * inch])
    params_table.setStyle(
        TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
    story.append(params_table)
    story.append(Spacer(1, 0.25 * inch))
    story.append(PageBreak())

    # --- 3. Quantitative Analysis Plots ---
    story.append(Paragraph("2. Quantitative Analysis", heading_style))

    analysis_plots = _generate_analysis_plots(filtered_data,
                                              tracking_sheet_path)

    if not analysis_plots:
        story.append(
            Paragraph("No analysis plots could be generated.", body_style))
        story.append(PageBreak())
    else:
        plot_title_style = ParagraphStyle('PlotTitle',
                                          parent=styles['h3'],
                                          spaceBefore=12,
                                          spaceAfter=4)
        for plot_info in analysis_plots:
            story.append(Paragraph(plot_info['title'], plot_title_style))

            # 1. Get the original image buffer
            original_buffer = plot_info['buffer']

            # 2. Open the image with Pillow
            pil_image = PILImage.open(original_buffer)

            # 3. Rotate it 90 degrees. `expand=True` ensures the canvas resizes.
            # Use COUNTERCLOCKWISE to make the y-axis appear on the left.
            rotated_image = pil_image.rotate(90,
                                             expand=True,
                                             resample=PILImage.BICUBIC)

            # 4. Save the new, rotated image into a new buffer
            rotated_buffer = io.BytesIO()
            rotated_image.save(rotated_buffer, format='PNG')
            rotated_buffer.seek(0)

            # 5. Add the rotated image to the PDF, adjusting the width to fit.
            # Constrain the image to a bounding box of 5x8 inches.
            # Reportlab will scale the image to fit inside this box while
            # preserving its aspect ratio. Since the image is tall and narrow,
            # the height=8*inch will be the limiting factor, guaranteeing it fits.
            img = Image(rotated_buffer, width=5 * inch, height=8 * inch)
            img.hAlign = 'CENTER'
            story.append(img)
            story.append(Spacer(1, 0.2 * inch))
            story.append(PageBreak())

    # --- 4. Error Report ---
    story.append(Paragraph("3. Error Report", heading_style))

    if not errors:
        story.append(
            Paragraph("No errors were encountered during the process.",
                      body_style))
    else:
        # Define a new style for the error titles
        error_title_style = ParagraphStyle(
            'ErrorTitle',
            parent=body_style,
            fontName='Helvetica-Bold',
            fontSize=11,
            spaceAfter=6  # Space between the title and the table
        )

        for error_type, details in errors.items():
            count = details['count']
            paths = sorted(list(details['paths']))

            # 1. Create the bold title with the count
            title_text = f"{error_type} (Occurrences: {count})"
            error_title = Paragraph(title_text, error_title_style)
            story.append(error_title)

            # 2. Create a simple, single-column table for the paths
            # Each path is wrapped in a Paragraph to allow for automatic line wrapping
            path_data = [[Paragraph(p, body_style)] for p in paths]

            # Use the full available width of the document frame
            # (5.5 inches is a safe width for a standard Letter/A4 page with margins)
            path_table = Table(path_data, colWidths=[5.5 * inch])

            path_table.setStyle(
                TableStyle([
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    # Add some padding inside the cells
                    ('LEFTPADDING', (0, 0), (-1, -1), 6),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))

            story.append(path_table)
            story.append(PageBreak())

    # --- Build PDF ---
    try:
        doc.build(story)
        print(f"\nSuccessfully generated report at: {report_path}")
    except Exception as e:
        print(f"\nFailed to generate PDF report: {e}", file=sys.stderr)
