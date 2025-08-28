from copy import deepcopy
from pathlib import Path

import numpy as np #
import pandas as pd #
import open3d as o3d #
from tqdm import tqdm #
import matplotlib  #
import japanize_matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt #
import matplotlib.patches as mpatches #

# yapf: disable
# def process_questionnaire_answers_fast(
#     input_file,
#     model_file,
#     base_color,
#     qna_answer_color_map,
#     hololens_2_spatial_error,
#     gaussian_denominator,
# ):
#     qa_segmented_meshes = {}

#     df = pd.read_csv(input_file, header=0, sep=",")
#     df["estX"] = pd.to_numeric(df["estX"], errors="coerce")
#     df["estY"] = pd.to_numeric(df["estY"], errors="coerce")
#     df["estZ"] = pd.to_numeric(df["estZ"], errors="coerce")
#     df["answer"] = df["answer"].astype(str).str.strip()
#     df.dropna(subset=["estX", "estY", "estZ", "answer"], inplace=True)
#     mesh = o3d.io.read_triangle_mesh(model_file)
#     if not mesh.has_vertices():
#         raise ValueError(f"Mesh file '{model_file}' contains no vertices.")

#     # Segmented QNA mesh
#     mesh_vertices_np = np.asarray(mesh.vertices)
#     mesh_triangles_np = np.asarray(mesh.triangles)
#     n_vertices = mesh_vertices_np.shape[0]

#     qa_points = df[["estX", "estY", "estZ"]].values
#     qa_colors_01 = []
#     for answer in df["answer"]:
#         qa_colors_01.append(qna_answer_color_map.get(answer)["rgb"])
#     qa_colors_01 = np.array(qa_colors_01) / 255.0
#     qa_pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(qa_points))
#     qa_pcd.colors = o3d.utility.Vector3dVector(np.array(qa_colors_01))

#     mesh_kdtree = o3d.geometry.KDTreeFlann(mesh)
#     mesh_scene = o3d.t.geometry.RaycastingScene()
#     mesh_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

#     final_colors = np.zeros((n_vertices, 3), dtype=float)
#     total_weights = np.zeros(n_vertices, dtype=float)
#     unique_answers = df["answer"].unique()
#     all_hit_vertices_by_answer = {}

#     for answer in unique_answers:
#         category_df = df[df["answer"] == answer]
#         category_points = category_df[["estX", "estY", "estZ"]].values
#         color_vector = np.array(qna_answer_color_map.get(answer)["rgb"]) / 255.0
#         hit_vertices = set()

#         query_points = o3d.core.Tensor(category_points, dtype=o3d.core.Dtype.Float32)
#         closest_geometry = mesh_scene.compute_closest_points(query_points)
#         closest_face_indices = closest_geometry["primitive_ids"].numpy()

#         for closest_face_idx in tqdm(closest_face_indices, desc=f"{answer} | Mapping Hits", leave=False):
#             if closest_face_idx != o3d.t.geometry.RaycastingScene.INVALID_ID:
#                 for v_idx in mesh_triangles_np[closest_face_idx]:
#                     hit_vertices.add(v_idx)
#         all_hit_vertices_by_answer[answer] = hit_vertices

#         for start_node_idx in tqdm(list(hit_vertices), desc="Applying Gaussian Spread", leave=False):
#             # Find all vertices within the specified radius
#             [k, indices, euclidean_distance] = mesh_kdtree.search_radius_vector_3d(mesh_vertices_np[start_node_idx], hololens_2_spatial_error)

#             if k > 0:
#                 # Calculate Gaussian weights based on squared Euclidean distance
#                 gaussian_weights = np.exp(-np.asarray(euclidean_distance)**2 / gaussian_denominator)

#                 # Apply weighted colors to all neighbors found in the radius
#                 final_colors[indices] += color_vector * gaussian_weights[:, np.newaxis]
#                 total_weights[indices] += gaussian_weights

#     valid_weights_mask = total_weights > 1e-9
#     final_colors[valid_weights_mask] /= total_weights[valid_weights_mask, np.newaxis]
#     final_colors[~valid_weights_mask] = base_color # Negation of bool mask

#     combined_mesh = o3d.geometry.TriangleMesh(mesh)
#     combined_mesh.vertex_colors = o3d.utility.Vector3dVector(final_colors)

#     for answer, hit_vertices in tqdm(all_hit_vertices_by_answer.items(), desc="Making Segmented Maps", leave=False):
#         color_info = qna_answer_color_map.get(answer)
#         color_vec = np.array(color_info["rgb"]) / 255.0
#         segmented_colors = np.zeros((n_vertices, 3), dtype=float)

#         for start_node_idx in list(hit_vertices):
#             [k, indices, _] = mesh_kdtree.search_radius_vector_3d(mesh_vertices_np[start_node_idx], hololens_2_spatial_error)
#             if k > 0:
#                 # Assign the solid color to all vertices found within the radius
#                 segmented_colors[indices] = color_vec

#         segmented_mesh = o3d.geometry.TriangleMesh(mesh)
#         segmented_mesh.vertex_colors = o3d.utility.Vector3dVector(segmented_colors)
#         qa_segmented_meshes[qna_answer_color_map[answer]["name"]] = segmented_mesh

#     return qa_pcd, qa_segmented_meshes, combined_mesh

def process_questionnaire_answers_fast(
    input_file,
    model_file,
    base_color,
    qna_answer_color_map,
    hololens_2_spatial_error,
    gaussian_denominator,
):
    """
    Processes questionnaire data to map gaze points onto a 3D model and
    generates a timeline visualization of the emotional responses.

    Returns:
        A tuple containing:
        - qa_pcd (open3d.geometry.PointCloud): Point cloud of gaze data.
        - qa_segmented_meshes (dict): Dictionary of meshes segmented by emotion.
        - combined_mesh (open3d.geometry.TriangleMesh): Mesh with blended colors.
        - timeline_fig (matplotlib.figure.Figure): Figure object for the emotion timeline.
    """
    qa_segmented_meshes = {}

    df = pd.read_csv(input_file, header=0, sep=",")
    df["estX"] = pd.to_numeric(df["estX"], errors="coerce")
    df["estY"] = pd.to_numeric(df["estY"], errors="coerce")
    df["estZ"] = pd.to_numeric(df["estZ"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce") # Ensure timestamp is numeric
    df["answer"] = df["answer"].astype(str).str.strip()
    df.dropna(subset=["estX", "estY", "estZ", "answer", "timestamp"], inplace=True)
    
    # Sort by timestamp for correct timeline plotting
    df = df.sort_values('timestamp').reset_index(drop=True)

    mesh = o3d.io.read_triangle_mesh(model_file)
    if not mesh.has_vertices():
        raise ValueError(f"Mesh file '{model_file}' contains no vertices.")

    mesh_vertices_np = np.asarray(mesh.vertices)
    mesh_triangles_np = np.asarray(mesh.triangles)
    n_vertices = mesh_vertices_np.shape[0]

    qa_points = df[["estX", "estY", "estZ"]].values
    qa_colors_01 = []
    for answer in df["answer"]:
        # Use a default color if an answer is not in the map
        color_info = qna_answer_color_map.get(answer, {"rgb": [128, 128, 128]})
        qa_colors_01.append(color_info["rgb"])
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
            [k, indices, euclidean_distance] = mesh_kdtree.search_radius_vector_3d(mesh_vertices_np[start_node_idx], hololens_2_spatial_error)
            if k > 0:
                gaussian_weights = np.exp(-np.asarray(euclidean_distance)**2 / gaussian_denominator)
                final_colors[indices] += color_vector * gaussian_weights[:, np.newaxis]
                total_weights[indices] += gaussian_weights

    valid_weights_mask = total_weights > 1e-9
    final_colors[valid_weights_mask] /= total_weights[valid_weights_mask, np.newaxis]
    final_colors[~valid_weights_mask] = base_color

    combined_mesh = o3d.geometry.TriangleMesh(mesh)
    combined_mesh.vertex_colors = o3d.utility.Vector3dVector(final_colors)

    for answer, hit_vertices in tqdm(all_hit_vertices_by_answer.items(), desc="Making Segmented Maps", leave=False):
        color_info = qna_answer_color_map.get(answer)
        color_vec = np.array(color_info["rgb"]) / 255.0
        segmented_colors = np.zeros((n_vertices, 3), dtype=float)
        for start_node_idx in list(hit_vertices):
            [k, indices, _] = mesh_kdtree.search_radius_vector_3d(mesh_vertices_np[start_node_idx], hololens_2_spatial_error)
            if k > 0:
                segmented_colors[indices] = color_vec
        segmented_mesh = o3d.geometry.TriangleMesh(mesh)
        segmented_mesh.vertex_colors = o3d.utility.Vector3dVector(segmented_colors)
        qa_segmented_meshes[qna_answer_color_map[answer]["name"]] = segmented_mesh

    ### Timeline figure

    # A new block starts if the emotion changes OR if the time gap is > 50ms.
    df['time_diff'] = df['timestamp'].diff()
    emotion_changed = df['answer'] != df['answer'].shift()
    time_gap_exceeded = df['time_diff'] > 0.05
    df['block_id'] = (emotion_changed | time_gap_exceeded).cumsum()

    # Group by blocks to get start, end, and emotion for each continuous block.
    block_df = df.groupby('block_id').agg(
        start_time=('timestamp', 'min'),
        end_time=('timestamp', 'max'),
        answer=('answer', 'first')
    ).reset_index()

    # Duration of a block is its end time minus its start time.
    block_df['duration'] = block_df['end_time'] - block_df['start_time']
    
    # Map answers to colors, normalizing RGB from [0, 255] to [0, 1]
    colors = [
        np.array(qna_answer_color_map.get(ans, {"rgb": [128, 128, 128]})["rgb"]) / 255.0
        for ans in block_df["answer"]
    ]

    timeline_fig, ax = plt.subplots(figsize=(15, 2))
    
    # Plot each block as a bar. Gaps will appear as white space.
    ax.barh(y=[0] * len(block_df), width=block_df['duration'], left=block_df['start_time'], color=colors, height=1)
    
    # Formatting the plot
    ax.set_yticks([])
    ax.set_xlabel("Time (seconds)")
    ax.set_xlim(left=0)
    
    # Extract pottery ID from the input file path for the title
    pottery_id_title = Path(input_file).parent.name
    ax.set_title(f"Emotion Timeline for {pottery_id_title}")
    
    # Create custom legend
    legend_patches = [
        mpatches.Patch(color=np.array(info["rgb"]) / 255.0, label=name)
        for name, info in qna_answer_color_map.items()
    ]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.02, 1), loc='upper left')
    
    timeline_fig.tight_layout(rect=[0, 0, 0.85, 1])

    return qa_pcd, qa_segmented_meshes, combined_mesh, timeline_fig


# MARKERS
def _create_base_dot_geometry(size):
  """Creates the base dot geometry (a sphere) at the origin."""
  marker = o3d.geometry.TriangleMesh.create_sphere(radius=size/1.5)
  marker.compute_vertex_normals()
  return marker

def _create_base_square_geometry(size):
    """Creates the base cube geometry at the origin."""
    marker = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
    marker.compute_vertex_normals()
    return marker

def _create_base_triangle_geometry(size):
    """Creates the base tetrahedron geometry at the origin."""
    marker = o3d.geometry.TriangleMesh.create_tetrahedron(radius=size)
    marker.compute_vertex_normals()
    return marker

# def _create_base_star_geometry(size):
#     """Creates the base star (bipyramid) geometry at the origin."""
#     pyramid = o3d.geometry.TriangleMesh.create_cone(radius=size, height=size * 0.75, resolution=5, split=1)
#     inverted_pyramid = deepcopy(pyramid)
#     flip_rotation = o3d.geometry.get_rotation_matrix_from_xyz((np.pi, 0, 0))
#     inverted_pyramid.rotate(flip_rotation, center=(0, 0, 0))
#     marker = pyramid + inverted_pyramid
#     marker.compute_vertex_normals()
#     return marker

def _create_diamond_geometry(size):
    """Creates a diamond (octahedron) geometry at the origin."""
    diamond = o3d.geometry.TriangleMesh.create_octahedron(radius=size)

    diamond.compute_vertex_normals()

    return diamond

def _create_base_x_geometry(size):
    """Creates the base 3D 'X' geometry at the origin."""
    thickness = size / 4.0
    arm1 = o3d.geometry.TriangleMesh.create_box(width=size, height=thickness, depth=thickness)
    arm2 = deepcopy(arm1)
    rotation_z_45_pos = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.pi / 4))
    rotation_z_45_neg = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, -np.pi / 4))
    arm1.rotate(rotation_z_45_pos, center=(0, 0, 0))
    arm2.rotate(rotation_z_45_neg, center=(0, 0, 0))
    marker_bottom = arm1 + arm2
    marker_top = deepcopy(marker_bottom)
    marker_top.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.pi)), center=(0, 0, 0))
    marker_top.translate((0, thickness, 0))
    marker = marker_bottom + marker_top
    rotation_y_90 = o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi / 2, 0))
    marker.rotate(rotation_y_90, center=(0, 0, 0))
    marker.compute_vertex_normals()
    return marker

def process_questionnaire_answers_markers(
    input_file: str,
    model_file: str,
    base_color: list,
    qna_answer_color_map: dict,
    hololens_2_spatial_error: float,
    gaussian_denominator: float,
):
    df = pd.read_csv(input_file, header=0, sep=",")
    df.dropna(subset=["estX", "estY", "estZ", "answer"], inplace=True)

    mesh = o3d.io.read_triangle_mesh(model_file)
    if not mesh.has_vertices():
        raise ValueError(f"Mesh file '{model_file}' contains no vertices.")
    mesh_vertices_np = np.asarray(mesh.vertices)
    mesh_triangles_np = np.asarray(mesh.triangles)
    min_bound = mesh.get_min_bound()
    max_bound = mesh.get_max_bound()
    max_range = np.max(max_bound - min_bound)

    # marker_size = max_range / 110
    marker_size = max_range / 55

    # 1. Create Base Geometries (Templates)
    # This is done only ONCE per shape type for maximum efficiency.
    base_geometry_cache = {
        'dot': _create_base_dot_geometry(marker_size),
        'square': _create_base_square_geometry(marker_size),
        'triangle': _create_base_triangle_geometry(marker_size),
        'diamond': _create_diamond_geometry(marker_size),
        'x': _create_base_x_geometry(marker_size),
    }

    # Map answers to the string key of the geometry cache
    QNA_SHAPE_KEY_MAP = {
        "面白い・気になる形だ": 'diamond',
        "美しい・芸術的だ": 'square',
        "不思議・意味不明": 'triangle',
        "不気味・不安・怖い": 'x',
        "何も感じない": 'dot',
        "Interesting and attentional shape": 'diamond',
        "Beautiful and artistic": 'square',
        "Strange and incomprehensible": 'triangle',
        "Creepy / unsettling / scary": 'x',
        "Feel nothing": 'dot',
    }

    # 2. Place Shape Instances
    all_marker_meshes = []
    # Group by answer to handle one color/shape combo at a time
    for answer, group_df in tqdm(df.groupby("answer"), desc="Placing Shape Instances", leave=False):
        if answer not in qna_answer_color_map: continue

        # Get the template geometry for this category
        shape_key = QNA_SHAPE_KEY_MAP.get(answer, 'dot')
        template_mesh = base_geometry_cache[shape_key]

        # Get the color for this category
        color_vec = np.array(qna_answer_color_map[answer]["rgb"]) / 255.0

        # For each point, copy the template, color it, and move it
        for i, point in enumerate(group_df[["estX", "estY", "estZ"]].values):
            if (i%4==0):
                # Create a fresh copy to avoid moving the original template
                marker_instance = deepcopy(template_mesh)
                marker_instance.paint_uniform_color(color_vec)
                marker_instance.translate(point, relative=False)
                all_marker_meshes.append(marker_instance)

    # Combine all the placed instances into a single mesh
    shaped_qna_mesh = o3d.geometry.TriangleMesh()
    for m in all_marker_meshes:
        shaped_qna_mesh += m

    # 3. Create Colored and Segmented Meshes (Gaussian Spread)
    mesh_vertices_np = np.asarray(mesh.vertices)
    mesh_triangles_np = np.asarray(mesh.triangles)
    n_vertices = mesh_vertices_np.shape[0]
    mesh_kdtree = o3d.geometry.KDTreeFlann(mesh)
    mesh_scene = o3d.t.geometry.RaycastingScene()
    mesh_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    final_colors = np.zeros((n_vertices, 3), dtype=float)
    total_weights = np.zeros(n_vertices, dtype=float)
    unique_answers = df["answer"].unique()
    all_hit_vertices_by_answer = {}
    qa_segmented_meshes = {}

    for answer in unique_answers:
        if answer not in qna_answer_color_map: continue
        category_df = df[df["answer"] == answer]
        category_points = category_df[["estX", "estY", "estZ"]].values
        color_vector = np.array(qna_answer_color_map[answer]["rgb"]) / 255.0
        query_points = o3d.core.Tensor(category_points, dtype=o3d.core.Dtype.Float32)
        closest_geometry = mesh_scene.compute_closest_points(query_points)
        closest_face_indices = closest_geometry["primitive_ids"].numpy()
        hit_vertices = set()
        for face_idx in closest_face_indices:
            if face_idx != o3d.t.geometry.RaycastingScene.INVALID_ID:
                hit_vertices.update(mesh_triangles_np[face_idx])
        all_hit_vertices_by_answer[answer] = list(hit_vertices)

        for start_node_idx in tqdm(list(hit_vertices), desc=f"Applying Gaussian for {answer}", leave=False):
            [k, indices, sq_dist] = mesh_kdtree.search_radius_vector_3d(mesh_vertices_np[start_node_idx], hololens_2_spatial_error)
            if k > 0:
                gaussian_weights = np.exp(-np.asarray(sq_dist) / gaussian_denominator)
                final_colors[indices] += color_vector * gaussian_weights[:, np.newaxis]
                total_weights[indices] += gaussian_weights

    valid_weights_mask = total_weights > 1e-9
    final_colors[valid_weights_mask] /= total_weights[valid_weights_mask, np.newaxis]
    final_colors[~valid_weights_mask] = base_color

    combined_gaussian_mesh = o3d.geometry.TriangleMesh(mesh)
    combined_gaussian_mesh.vertex_colors = o3d.utility.Vector3dVector(final_colors)

    for answer, hit_vertices in tqdm(all_hit_vertices_by_answer.items(), desc="Making Segmented Maps", leave=False):
        color_info = qna_answer_color_map[answer]
        color_vec = np.array(color_info["rgb"]) / 255.0
        segmented_colors = np.full((n_vertices, 3), base_color, dtype=float)
        all_affected_indices = set()
        for start_node_idx in hit_vertices:
            [k, indices, _] = mesh_kdtree.search_radius_vector_3d(mesh_vertices_np[start_node_idx], hololens_2_spatial_error)
            if k > 0: all_affected_indices.update(indices)
        segmented_colors[list(all_affected_indices)] = color_vec
        segmented_mesh = o3d.geometry.TriangleMesh(mesh)
        segmented_mesh.vertex_colors = o3d.utility.Vector3dVector(segmented_colors)
        qa_segmented_meshes[color_info["name"]] = segmented_mesh

    return shaped_qna_mesh, qa_segmented_meshes, combined_gaussian_mesh
# yapf: enable