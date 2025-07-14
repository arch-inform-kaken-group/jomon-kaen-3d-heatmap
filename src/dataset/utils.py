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
import sys
import threading
import queue
from typing import List

import numpy as np
import pandas as pd
import open3d as o3d

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
DEFAULT_HOLOLENS_2_SPATIAL_ERROR = 2.66

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
        "rgb": [255, 255, 0],
        "name": "yellow"
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
combined_qa_mesh_filename = "combined_qa_mesh"
processed_voice_filename = "processed_voice"

### CALCULATION ###


def _calculate_smoothed_vertex_intensities(
    gaze_points_np,
    mesh,
    hololens_2_spatial_error,
    gaussian_denominator,
):
    # Get the vertices, triangles of the mesh
    mesh_vertices_np = np.asarray(mesh.vertices)
    mesh_triangles_np = np.asarray(mesh.triangles)
    n_vertices = mesh_vertices_np.shape[0]

    # Mapping gaze points to nearest mesh faces (vertices)
    # Using a raycasting scene from Open3D
    mesh_scene = o3d.t.geometry.RaycastingScene()
    mesh_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    query_points = o3d.core.Tensor(gaze_points_np, dtype=o3d.core.Dtype.Float32)
    closest_geometry = mesh_scene.compute_closest_points(query_points)
    closest_face_indices = closest_geometry['primitive_ids'].numpy()


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
    preprocess: bool = True,
    mode: int = 0, # 'STRICT': 0 | 'LINIENT': 1
    hololens_2_spatial_error: float = DEFAULT_HOLOLENS_2_SPATIAL_ERROR,
    base_color: List = DEFAULT_BASE_COLOR,
    cmap = DEFAULT_CMAP,
    groups: List = [],
    session_ids: List = [],
    pottery_ids: List = [],
    min_pointcloud_size: float = 0.0,
    min_qa_size: float = 0.0,
    min_voice_quality: float = 0.1,
    use_cache: bool = True,
    from_tracking_sheet: bool = False,
    tracking_sheet_path: str = "",
    generate_report: bool = True,
):
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
    #
    # FORMAT:
    # "ERROR": {
    #   count: int
    #   paths: List
    # }
    errors = {}
    error_queue = queue.Queue()

    # Check if each data instance / file path exists
    data = []
    if not Path(root).exists():
        raise (ValueError(f"Root directory not found: {root}"))
    processed_path = "\\".join(Path(root).parts[:-1]) / Path('processed')
    os.makedirs(processed_path, exist_ok=True)

    pottery_ids = [f"{pid}({ASSIGNED_NUMBERS_DICT[pid]})" for pid in pottery_ids]

    # Filter based on group, session, model
    print(f"\nCHECKING RAW DATA PATHS")
    unique_group_keys = set([])
    unique_session_keys = set([])
    unique_pottery_keys = set([])

    group_keys = os.listdir(root)
    unique_group_keys.update(group_keys)
    for g in group_keys:
        group_path = root / Path(g)
        processed_group_path = processed_path / Path(g)

        session_keys = os.listdir(group_path)
        unique_session_keys.update(session_keys)
        for s in tqdm(session_keys, desc=g):
            session_path = group_path / Path(s)
            processed_session_path = processed_group_path / Path(s)

            pottery_keys = os.listdir(session_path)
            unique_pottery_keys.update(pottery_keys)
            for p in pottery_keys:
                has_error = False
                data_paths = {}
                pottery_path = session_path / Path(p)
                processed_pottery_path = processed_session_path / Path(p)

                pointcloud_path = pottery_path / Path("pointcloud.csv")
                qa_path = pottery_path / Path("qa.csv")
                model_path = pottery_path / Path("model.obj")
                voice_path = pottery_path / Path("session_audio_0.wav")

                output_sanity_plot = processed_pottery_path / f"{sanity_plot_filename}.png"
                output_point_cloud = processed_pottery_path / f"{eg_pointcloud_filename}.ply"
                output_heatmap = processed_pottery_path / f"{eg_heatmap_filename}.ply"
                output_voxel = processed_pottery_path / f"{voxel_filename}.ply"
                output_qa_pc = processed_pottery_path / f"{qa_pc_filename}.ply"
                output_segmented_meshes_dir = processed_pottery_path / segmented_meshes_dirname
                output_combined_qa_mesh_file = processed_pottery_path / f"{combined_qa_mesh_filename}.ply"
                output_voice = processed_pottery_path / f"{processed_voice_filename}.wav"

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
                        has_error = True
                        errors = increment_error('Point cloud path does not exist', str(pointcloud_path), errors)

                    if Path(qa_path).exists():
                        data_paths['qa'] = str(qa_path)
                        data_paths['QA_SIZE_KB'] = os.path.getsize(qa_path)/1024
                        data_paths[qa_pc_filename] = str(output_qa_pc)
                        data_paths[segmented_meshes_dirname] = str(output_segmented_meshes_dir)
                        data_paths[combined_qa_mesh_filename] = str(output_combined_qa_mesh_file)
                    else:
                        has_error = True
                        errors = increment_error('QNA path does not exist', str(qa_path), errors)
                else:
                    has_error = True
                    errors = increment_error('Model path does not exist', str(model_path), errors)

                if Path(voice_path).exists():
                    data_paths['voice'] = str(voice_path)
                    data_paths[processed_voice_filename] = str(output_voice)
                else:
                    has_error = True
                    errors = increment_error('Voice path does not exist', str(voice_path), errors)

                if (not has_error or mode==1):
                    data_paths['GROUP'] = g
                    data_paths['SESSION_ID'] = s
                    data_paths['ID'] = p
                    data_paths['processed_pottery_path'] = str(processed_pottery_path)
                    data.append(data_paths)

    n_valid_data = len(data)

    # Filtering based on pointcloud, qa size and voice quality
    print("\nFILTERING")
    if from_tracking_sheet:
        if not Path(tracking_sheet_path).exists():
            from_tracking_sheet = False
            raise(ValueError(f"Tracking sheet at {tracking_sheet_path} does not exist."))

    if min_voice_quality > 0.0 and not (from_tracking_sheet and Path(tracking_sheet_path).exists()):
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

        for i, data_paths in enumerate(tqdm(data)):
            if data_paths['GROUP'] in unique_groups \
                and data_paths['SESSION_ID'] in unique_session \
                and data_paths['ID'] in unique_pottery_id:
                keep_data_index[i] = True

        data = np.array(data)[keep_data_index]
        n_filtered_from_tracking_sheet = n_valid_data - len(data)

    # Filter from parameters
    n_filtered_from_parameters = 0
    if len(groups) > 0: unique_group_keys = list(unique_group_keys & set(groups))
    unique_group_keys = list(unique_group_keys)
    if len(session_ids) > 0: unique_session_keys = list(unique_session_keys & set(session_ids))
    unique_session_keys = list(unique_session_keys)
    if len(pottery_ids) > 0: unique_pottery_keys = list(unique_pottery_keys & set(pottery_ids))
    unique_pottery_keys = list(unique_pottery_keys)
    keep_data_index = np.array(np.zeros(len(data)), dtype=bool)

    for i, data_paths in enumerate(tqdm(data)):
        if data_paths['GROUP'] in unique_group_keys \
            and data_paths['SESSION_ID'] in unique_session_keys \
            and data_paths['ID'] in unique_pottery_keys \
            and data_paths['POINTCLOUD_SIZE_KB'] >= min_pointcloud_size \
            and data_paths['QA_SIZE_KB'] >= min_qa_size:
            keep_data_index[i] = True

    data = np.array(data)[keep_data_index]
    n_filtered_from_parameters = n_valid_data - n_filtered_from_tracking_sheet - len(data)

    # Finalizing the data paths
    # Preprocessed
    if (preprocess):
        for data_paths in tqdm(data):
            os.makedirs(data_paths['processed_pottery_path'], exist_ok=True)

            # Eye gaze intensity point cloud & heatmap
            if use_cache and Path(data_paths[eg_pointcloud_filename]).exists() and Path(data_paths[eg_heatmap_filename]).exists():
                pass
            else:
                eye_gaze_pointcloud, eye_gaze_heatmap_mesh = generate_gaze_pointcloud_heatmap(
                    input_file=data_paths['pointcloud'],
                    model_file=data_paths['model'],
                    cmap=cmap,
                    base_color=base_color,
                    hololens_2_spatial_error=hololens_2_spatial_error,
                    gaussian_denominator=GAUSSIAN_DENOMINATOR
                )
                active_threads.append(save_geometry_threaded(data_paths[eg_pointcloud_filename], eye_gaze_pointcloud, error_queue))
                active_threads.append(save_geometry_threaded(data_paths[eg_heatmap_filename], eye_gaze_heatmap_mesh, error_queue))

            # QNA combined point cloud
            # QNA segmented mesh
            if use_cache and Path(data_paths[qa_pc_filename]).exists() and Path(data_paths[segmented_meshes_dirname]).exists():
                pass
            else:
                qa_pointcloud, qa_segmented_mesh = process_questionnaire_answeres(
                    input_file=data_paths['pointcloud'],
                    model_file=data_paths['model'],
                    base_color=base_color,
                    hololens_2_spatial_error=hololens_2_spatial_error,
                    gaussian_denominator=GAUSSIAN_DENOMINATOR
                )
                active_threads.append(save_geometry_threaded(data_paths[qa_pc_filename], qa_pointcloud, error_queue))

                os.makedirs(data_paths[segmented_meshes_dirname], exist_ok=True)
                for k in qa_segmented_mesh.keys():
                    segmented_mesh = qa_segmented_mesh[k]
                    individual_segment = data_paths[segmented_meshes_dirname] / Path(f"{k}.ply")
                    active_threads.append(save_geometry_threaded(individual_segment, segmented_mesh, error_queue))

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
            from_tracking_sheet=from_tracking_sheet,
            tracking_sheet_path=tracking_sheet_path,
            n_filtered_from_tracking_sheet=n_filtered_from_tracking_sheet,
            n_filtered_from_parameters=n_filtered_from_parameters,
            n_valid_data=n_valid_data,
            filtered_data=data,
        )

    return data, errors
# yapf: enable

### PROCESS DATA ###


def generate_gaze_pointcloud_heatmap(
    input_file,
    model_file,
    cmap,
    base_color,
    hololens_2_spatial_error,
    gaussian_denominator,
):
    pass


def generate_voxel_from_mesh(
    mesh,
    vertex_intensities,
    cmap,
    base_color,
):
    pass


def process_questionnaire_answeres(
    input_file,
    model_file,
    base_color,
    hololens_2_spatial_error,
    gaussian_denominator,
):
    pass


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


def _generate_analysis_plots(data: np.ndarray,
                             tracking_sheet_path: str = None):
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
    base_color: List = DEFAULT_BASE_COLOR,
    cmap=DEFAULT_CMAP,
    groups: List = [],
    session_ids: List = [],
    pottery_ids: List = [],
    min_pointcloud_size: float = 0.0,
    min_qa_size: float = 0.0,
    min_voice_quality: float = 0.0,
    from_tracking_sheet: bool = False,
    tracking_sheet_path: str = "",
    n_filtered_from_tracking_sheet: int = 0,
    n_filtered_from_parameters: int = 0,
    n_valid_data: int = 0,
    filtered_data: List[dict] = [],
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
    final_count = n_valid_data - n_filtered_from_tracking_sheet - n_filtered_from_parameters
    summary_data = [
        ['Initial Datasets Found:',
         str(n_valid_data)],
        ['Filtered by Tracking Sheet:',
         str(n_filtered_from_tracking_sheet)],
        ['Filtered by Parameters:',
         str(n_filtered_from_parameters)],
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
