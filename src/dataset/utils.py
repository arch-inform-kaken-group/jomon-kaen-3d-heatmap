"""
Author: Lu Hou Yang
Last updated: 06 August 2025

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

import numpy as np #
import pandas as pd #
import open3d as o3d #

import matplotlib
import japanize_matplotlib

from dataset.processing.affective_state import process_questionnaire_answers_fast, process_questionnaire_answers_markers
from dataset.processing.aggregation import generate_gaze_pointcloud_heatmap, generate_voxel_from_mesh
from dataset.processing.fixation import generate_gaze_visualizations_from_files, generate_fixation_pointcloud_heatmap #
from dataset.processing.pottery import voxelize_pottery_dogu #
from dataset.processing.report import generate_filtered_dataset_report #
from dataset.processing.sanity_check import analyze_and_plot_point_cloud, generate_original_pointcloud #
from dataset.processing.voice import process_voice_data #

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm #


# https://arxiv.org/abs/2111.07209 [An Assessment of the Eye Tracking Signal Quality Captured in the HoloLens 2]
# Official: 1.5 | Paper original: 6.45 | Paper recalibrated: 2.66
DEFAULT_HOLOLENS_2_SPATIAL_ERROR = 1.5
DEFAULT_GAUSSIAN_DENOMINATOR = 2 * (DEFAULT_HOLOLENS_2_SPATIAL_ERROR**2)
DEFAULT_TARGET_VOXEL_RESOLUTION = 512

# Colors
# Gaze duration gradient from bright Cyan to dark Red
# cyan_to_dark_red_colors = [(0.0, 1.0, 1.0), (0.5, 0.0, 0.0)]  # Bright Cyan to Dark Red
# DEFAULT_CMAP = LinearSegmentedColormap.from_list("cyan_to_dark_red", cyan_to_dark_red_colors)

cyan_to_black_colors = [(0.0, 0.0, 0.0), (0.0, 1.0, 1.0)]
DEFAULT_CMAP = LinearSegmentedColormap.from_list("cyan_to_black", cyan_to_black_colors)

# DEFAULT_CMAP = plt.get_cmap('jet')

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
    'UD0028': '93',
    'rembak7': 'A',
}

# QNA Answer Color
DEFAULT_QNA_ANSWER_COLOR_MAP = {
    "面白い・気になる形だ": {
        "rgb": [0, 255, 255],
        "name": "cyan"
    },
    "美しい・芸術的だ": {
        "rgb": [0, 255, 0],
        "name": "green"
    },
    "不思議・意味不明": {
        "rgb": [255, 255, 0],
        "name": "yellow"
    },
    "不気味・不安・怖い": {
        "rgb": [255, 0, 0],
        "name": "red"
    },
    "何も感じない": {
        "rgb": [128, 128, 128],
        "name": "grey"
    },
    "Interesting and attentional shape": {
          "rgb": [0, 255, 255],
          "name": "cyan"
     },
     "Beautiful and artistic": {
          "rgb": [0, 255, 0],
        # "rgb": [0, 255, 255],
          "name": "green"
     },
     "Strange and incomprehensible": {
          "rgb": [255, 255, 0],
        # "rgb": [0, 255, 255],
          "name": "yellow"
     },
     "Creepy / unsettling / scary": {
          "rgb": [255, 0, 0],
        # "rgb": [0, 255, 255],
          "name": "red"
     },
     "Feel nothing": {
          "rgb": [128, 128, 128],
        # "rgb": [0, 255, 255],
          "name": "grey"
     },
}

# Threading
data_lock = threading.Lock()

# Data paths
original_pointcloud_filename = "original_pointcloud"
sanity_plot_filename = "pointcloud_occurrence_plot"
eg_pointcloud_filename = "eye_gaze_intensity_pc"
eg_heatmap_rgb_filename = "eye_gaze_intensity_hm_rgb"
voxel_filename = "eye_gaze_voxel"
qa_pc_filename = "qa_pc"
segmented_meshes_dirname = "qa_segmented_mesh"
processed_voice_filename = "processed_voice"
combined_mesh_filename = "combined_qa_mesh"
pottery_dirname = "voxel_pottery"
fixation_pointcloud_filename = "fixation_pointcloud"
fixation_heatmap_filename = "fixation_heatmap"

MESH_PC_VOXEL_EXTENSION = ".ply"
VOICE_EXTENSION = ".mp3"
SANITY_CHECK_EXTENSION = ".png"


### UTILS ###


def get_pottery_id_list():
    return [f"{pid}({num})" for pid, num in ASSIGNED_NUMBERS_DICT.items()]


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


def save_plot_threaded(output_plot_path, fig, error_queue):
    """
    Saves a matplotlib plot in a separate thread.
    """

    def _save_plot(save_path, f, errq):
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            f.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(f)
        except Exception as e:
            print(f"\nError saving plot to {save_path}: {e}")
            errq.put({'Save plot error': save_path})

    plot_thread = threading.Thread(target=_save_plot,
                                   args=(output_plot_path, fig, error_queue))
    plot_thread.daemon = True
    plot_thread.start()
    return plot_thread


# yapf: disable
def filter_data_on_condition(
    root: str = "",
    pottery_path: str = "",
    preprocess: bool = True,
    mode: int = 0, # 'HEATMAP(VOXEL), QNA, VOICE': 0 | 'HEATMAP(VOXEL), QNA': 1 | 'HEATMAP(VOXEL), VOICE': 2 | 'HEATMAP(VOXEL)': 3
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
    max_emotion_count: int = 5,
    emotion_type: list = [],
    use_cache: bool = True,
    from_tracking_sheet: bool = False,
    tracking_sheet_path: str = "",
    generate_report: bool = True,
    generate_pc_hm_voxel: bool = True,
    generate_qna: bool = True,
    generate_voice: bool = False,
    generate_pottery_dogu_voxel: bool = True,
    generate_sanity_check: bool = False,
    limit: int = 9,
    generate_fixation: bool = False,
    voxel_color: str = 'gray', # 'gray' | 'rgb'
    qna_marker: bool = False,
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
    processed_dir = "/".join(Path(root).parts[:-1]) / Path('processed')
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
    limit_track = {key: 0 for key in pottery_id_all}
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
                if (p == 'language.txt'):
                    continue

                hm_error = False
                qna_error = False
                voice_error = False
                data_paths = {}
                pottery_path = session_path / Path(p)
                processed_pottery_path = processed_session_path / Path(p)

                pointcloud_path = pottery_path / Path("pointcloud.csv")
                qa_path = pottery_path / Path("qa_corrected.csv")
                model_path = pottery_path / Path("model.obj")
                voice_path = pottery_path / Path("session_audio_0.wav")

                output_sanity_plot = processed_pottery_path / f"{sanity_plot_filename}{SANITY_CHECK_EXTENSION}"
                output_point_cloud = processed_pottery_path / f"{eg_pointcloud_filename}{MESH_PC_VOXEL_EXTENSION}"
                output_heatmap_rgb = processed_pottery_path / f"{eg_heatmap_rgb_filename}{MESH_PC_VOXEL_EXTENSION}"
                output_voxel = processed_pottery_path / f"{voxel_filename}{MESH_PC_VOXEL_EXTENSION}"
                output_qa_pc = processed_pottery_path / f"{qa_pc_filename}{MESH_PC_VOXEL_EXTENSION}"
                output_segmented_meshes_dir = processed_pottery_path / segmented_meshes_dirname
                output_combined_mesh_file = processed_pottery_path / f"{combined_mesh_filename}{MESH_PC_VOXEL_EXTENSION}"
                output_voice = processed_pottery_path / f"{processed_voice_filename}{VOICE_EXTENSION}"
                output_fixation_point_cloud = processed_pottery_path / f"{fixation_pointcloud_filename}{MESH_PC_VOXEL_EXTENSION}"
                output_fixation_heatmap = processed_pottery_path / f"{fixation_heatmap_filename}{MESH_PC_VOXEL_EXTENSION}"
                output_original_point_cloud = processed_pottery_path / f"{original_pointcloud_filename}{MESH_PC_VOXEL_EXTENSION}"

                # Check if paths exist and increment error
                if Path(model_path).exists():
                    data_paths['model'] = str(model_path)
                    if Path(pointcloud_path).exists():
                        data_paths['pointcloud'] = str(pointcloud_path)
                        data_paths['POINTCLOUD_SIZE_KB'] = os.path.getsize(pointcloud_path)/1024
                        data_paths[sanity_plot_filename] = str(output_sanity_plot)
                        data_paths[eg_pointcloud_filename] = str(output_point_cloud)
                        data_paths[eg_heatmap_rgb_filename] = str(output_heatmap_rgb)
                        data_paths[voxel_filename] = str(output_voxel)
                        data_paths[fixation_pointcloud_filename] = str(output_fixation_point_cloud)
                        data_paths[fixation_heatmap_filename] = str(output_fixation_heatmap)
                        data_paths[original_pointcloud_filename] = str(output_original_point_cloud)
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
                    data_paths['PROCESSED_DATA'] = str(processed_pottery_path)
                    data_paths['GROUP'] = g
                    data_paths['SESSION_ID'] = s
                    data_paths['ID'] = p
                    save_path = processed_pottery_dir / f"{str(pottery_dogu_path).split('\\')[-1].split('.')[0]}{MESH_PC_VOXEL_EXTENSION}"
                    data_paths['processed_pottery_path'] = str(save_path)
                    unique_pottery_dogu_voxel.add(str(pottery_dogu_path))
                    
                    if (limit_track[p] < limit):
                        limit_track[p] += 1
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
        data = filter_qna_by_emotion_count_and_type(data, min_emotion_count=min_emotion_count, max_emotion_count=max_emotion_count, emotion_type=emotion_type)
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
            save_path = processed_pottery_dir / f"{str(data_path).split('\\')[-1].split('.')[0]}{MESH_PC_VOXEL_EXTENSION}"
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
            if generate_sanity_check:
                sanity_plot = analyze_and_plot_point_cloud(csv_file_path=data_paths['pointcloud'])
                active_threads.append(save_plot_threaded(data_paths[sanity_plot_filename], sanity_plot, error_queue))
        
                original_pointcloud = generate_original_pointcloud(input_file=data_paths['pointcloud'])
                active_threads.append(save_geometry_threaded(data_paths[original_pointcloud_filename], original_pointcloud, error_queue))

            if generate_pc_hm_voxel:
                # Eye gaze intensity point cloud & heatmap
                # Eye gaze voxel
                if use_cache and Path(data_paths[eg_pointcloud_filename]).exists() \
                    and Path(data_paths[eg_heatmap_rgb_filename]).exists() \
                    and Path(data_paths[voxel_filename]).exists():
                    pass
                else:
                    eye_gaze_pointcloud, eye_gaze_heatmap_rgb_mesh, final_vertex_intensities, mesh_greyscale = generate_gaze_pointcloud_heatmap(
                        input_file=data_paths['pointcloud'],
                        model_file=data_paths['model'],
                        cmap=cmap,
                        base_color=base_color,
                        hololens_2_spatial_error=hololens_2_spatial_error,
                        gaussian_denominator=GAUSSIAN_DENOMINATOR
                    )
                    active_threads.append(save_geometry_threaded(data_paths[eg_pointcloud_filename], eye_gaze_pointcloud, error_queue))
                    active_threads.append(save_geometry_threaded(data_paths[eg_heatmap_rgb_filename], eye_gaze_heatmap_rgb_mesh, error_queue))

                    eye_gaze_voxel = generate_voxel_from_mesh(
                        mesh=mesh_greyscale,
                        vertex_intensities=final_vertex_intensities,
                        target_voxel_resolution=target_voxel_resolution,
                        cmap=cmap,
                        base_color=base_color,
                        base_pottery_pcd=data_paths['processed_pottery_path'],
                    )
                    active_threads.append(save_geometry_threaded(data_paths[voxel_filename], eye_gaze_voxel, error_queue))

            if generate_fixation:
                generate_gaze_visualizations_from_files(
                    gaze_csv_path=data_paths['pointcloud'],
                    model_path=data_paths['model'],
                    output_colorbar_path=data_paths['PROCESSED_DATA']+"/cb.png",
                    output_pointcloud_path=data_paths['PROCESSED_DATA']+"/gaze_duration_pc.ply",
                    output_heatmap_path=data_paths['PROCESSED_DATA']+"/gaze_duration_hm.ply",
                    hololens_2_spatial_error=hololens_2_spatial_error,
                    base_color=base_color,
                    cmap=cmap,
                )

                fixation_pointcloud, fixation_heatmap = generate_fixation_pointcloud_heatmap(
                    input_file=data_paths['pointcloud'],
                    model_file=data_paths['model'],
                    cmap=cmap,
                    base_color=base_color,
                    dispersion_threshold=3,
                    min_fixation_duration=0.1,
                )
                active_threads.append(save_geometry_threaded(data_paths[fixation_pointcloud_filename], fixation_pointcloud, error_queue))
                active_threads.append(save_geometry_threaded(data_paths[fixation_heatmap_filename], fixation_heatmap, error_queue))

            if generate_qna and (mode==0 or mode==1):
                # QNA combined point cloud
                # QNA segmented mesh
                if use_cache and Path(data_paths[qa_pc_filename]).exists() \
                    and Path(data_paths[segmented_meshes_dirname]).exists() \
                    and Path(data_paths[combined_mesh_filename]).exists():
                    pass
                else:
                    if qna_marker:
                            qa_pointcloud, qa_segmented_mesh, combined_mesh = process_questionnaire_answers_markers(
                            input_file=data_paths['qa'],
                            model_file=data_paths['model'],
                            base_color=base_color,
                            qna_answer_color_map=qna_answer_color_map,
                            hololens_2_spatial_error=hololens_2_spatial_error,
                            gaussian_denominator=GAUSSIAN_DENOMINATOR
                        )
                    else:
                        qa_pointcloud, qa_segmented_mesh, combined_mesh, timeline_fig = process_questionnaire_answers_fast(
                            input_file=data_paths['qa'],
                            model_file=data_paths['model'],
                            base_color=base_color,
                            qna_answer_color_map=qna_answer_color_map,
                            hololens_2_spatial_error=hololens_2_spatial_error,
                            gaussian_denominator=GAUSSIAN_DENOMINATOR
                        )
                    active_threads.append(save_geometry_threaded(data_paths[qa_pc_filename], qa_pointcloud, error_queue))
                    active_threads.append(save_geometry_threaded(data_paths[combined_mesh_filename], combined_mesh, error_queue))
                    active_threads.append(save_plot_threaded(str(Path(data_paths['PROCESSED_DATA']) / 'qa_timeline.png'), timeline_fig, error_queue))

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
                    audio_segment = process_voice_data(data_paths['voice'])
                    
                    audio_segment.export(
                        data_paths[processed_voice_filename],
                        format="mp3",
                        bitrate="192k"
                    )

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
            cmap=cmap,
            groups=groups,
            session_ids=session_ids,
            pottery_ids=pottery_ids,
            min_pointcloud_size=min_pointcloud_size,
            min_qa_size=min_qa_size,
            min_voice_quality=min_voice_quality,
            min_emotion_count=min_emotion_count,
            max_emotion_count=max_emotion_count,
            emotion_type=emotion_type,
            limit=limit,
            from_tracking_sheet=from_tracking_sheet,
            tracking_sheet_path=tracking_sheet_path,
            n_filtered_from_tracking_sheet=n_filtered_from_tracking_sheet,
            n_filtered_from_arguments=n_filtered_from_arguments,
            n_valid_data=n_valid_data,
            filtered_data=data,
        )

    return data, errors
# yapf: enable


def filter_qna_by_emotion_count_and_type(data: list, min_emotion_count: int = 1, max_emotion_count: int = 5, emotion_type: list = []):
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
    unique_emotion_type = [emotion for emotion, _ in DEFAULT_QNA_ANSWER_COLOR_MAP.items()]
    if len(emotion_type) > 0: unique_emotion_type = list(set(unique_emotion_type) & set(emotion_type))
    unique_emotion_type = list(unique_emotion_type)

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

            if unique_emotion_count >= min_emotion_count \
                and unique_emotion_count <= max_emotion_count:
                filtered_data.append(data_item)

        except pd.errors.EmptyDataError:
            continue
        except Exception as e:
            continue

    return filtered_data


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
    render_options = vis.get_render_option()

    if isinstance(geometry, o3d.geometry.PointCloud):
        render_options.point_size = point_size
    elif isinstance(geometry, o3d.geometry.TriangleMesh):
        render_options.mesh_show_back_face = True

    render_options.background_color = np.asarray([0.1, 0.1, 0.1])
    vis.run()
    vis.destroy_window()
