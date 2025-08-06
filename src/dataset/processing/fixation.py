from copy import deepcopy
import sys
import os

import numpy as np #
import pandas as pd #
import open3d as o3d #
from tqdm import tqdm #

import matplotlib #

matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_gaze_visualizations_from_files(
    gaze_csv_path,
    model_path,
    output_pointcloud_path,
    output_heatmap_path,
    output_colorbar_path,
    hololens_2_spatial_error,
    base_color,
    cmap,
):
    """
    Processes gaze data to generate and save a duration-based heatmap, a colored
    point cloud, and a color bar legend.

    This function encapsulates the entire workflow:
    1. Reads gaze data (with timestamps) and a 3D model from file paths.
    2. Calculates gaze duration and aggregates it onto the model's vertices.
    3. Applies a Gaussian spread to create a smooth heatmap.
    4. Generates and saves a colored point cloud.
    5. Generates and saves a colored heatmap mesh.
    6. Generates and saves a color bar image corresponding to the heatmap.

    Args:
        gaze_csv_path (str): Path to the input CSV file with gaze data.
                             Must have columns [x, y, z, timestamp].
        model_path (str): Path to the input 3D model file (e.g., .obj, .ply).
        output_pointcloud_path (str): Path to save the output colored point cloud (.ply).
        output_heatmap_path (str): Path to save the output heatmap mesh (.ply).
        output_colorbar_path (str): Path to save the output color bar image (.png).
        hololens_2_spatial_error (float): Spatial error of the eye tracker for Gaussian spread.
    """
    gaussian_denominator = 2 * (hololens_2_spatial_error**2)

    # 1. Load Data
    try:
        data = pd.read_csv(gaze_csv_path, header=0).to_numpy()
        if data.shape[1] < 4:
            raise ValueError("Input CSV must have at least 4 columns: x, y, z, timestamp.")
    except FileNotFoundError:
        print(f"Error: Gaze data file not found at {gaze_csv_path}", file=sys.stderr)
        return

    try:
        mesh = o3d.io.read_triangle_mesh(model_path)
        if not mesh.has_vertices():
            raise ValueError(f"Mesh file '{model_path}' contains no vertices.")
        mesh.compute_vertex_normals()
    except Exception as e:
        print(f"Error: Could not read or process model file at {model_path}: {e}", file=sys.stderr)
        return

    # 2. Calculate Gaze Durations and Aggregate on Mesh
    gaze_points_np = data[:, :3]
    timestamps = data[:, 3]
    durations = np.diff(timestamps)
    durations = np.append(durations, durations[-1])

    mesh_vertices_np = np.asarray(mesh.vertices)
    mesh_triangles_np = np.asarray(mesh.triangles)
    n_vertices = mesh_vertices_np.shape[0]

    mesh_scene = o3d.t.geometry.RaycastingScene()
    mesh_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    query_points = o3d.core.Tensor(gaze_points_np, dtype=o3d.core.Dtype.Float32)
    closest_geometry = mesh_scene.compute_closest_points(query_points)
    closest_face_indices = closest_geometry['primitive_ids'].numpy()

    raw_gaze_duration = np.zeros(n_vertices, dtype=np.float64)
    point_to_face_map = np.empty(gaze_points_np.shape[0], dtype=int)

    for i, closest_face_idx in enumerate(closest_face_indices):
        if closest_face_idx != o3d.t.geometry.RaycastingScene.INVALID_ID:
            point_to_face_map[i] = closest_face_idx
            for v_idx in mesh_triangles_np[closest_face_idx]:
                raw_gaze_duration[v_idx] += 0.03 # Only 30ms for each point since the person may not always be looking at the pottery

    # 3. Apply Smoothing and Spreading
    log_scaled_duration = np.log1p(raw_gaze_duration)
    kdtree = o3d.geometry.KDTreeFlann(mesh)
    final_vertex_intensities = np.copy(log_scaled_duration)
    hit_vertices_indices = np.where(log_scaled_duration > 0)[0]

    for start_node_idx in tqdm(hit_vertices_indices, desc="Applying Gaussian Spread", leave=False):
        hit_value = log_scaled_duration[start_node_idx]
        [k, indices, euclidean_dist] = kdtree.search_radius_vector_3d(
            mesh_vertices_np[start_node_idx], hololens_2_spatial_error)
        if k > 1:
            gaussian_weights = np.exp(-np.asarray(euclidean_dist)**2 / gaussian_denominator)
            for i, neighbor_idx in enumerate(indices):
                if neighbor_idx != start_node_idx:
                    final_vertex_intensities[neighbor_idx] += hit_value * gaussian_weights[i]

    # 4. Generate and Save Color Bar
    min_gaze_duration = np.min(raw_gaze_duration[raw_gaze_duration > 0]) if np.any(raw_gaze_duration > 0) else 0
    max_gaze_duration = np.max(raw_gaze_duration)

    fig, ax = plt.subplots(figsize=(2, 8))
    norm = matplotlib.colors.Normalize(vmin=min_gaze_duration, vmax=max_gaze_duration)
    cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label('Total Gaze Duration (s)', size=14, weight='bold', labelpad=15)
    tick_values = np.linspace(min_gaze_duration, max_gaze_duration, num=5)
    cb.set_ticks(tick_values)
    cb.set_ticklabels([f'{val:.2f} s' for val in tick_values])
    cb.ax.tick_params(labelsize=12, length=8, width=1.5)

    try:
        os.makedirs(os.path.dirname(output_colorbar_path), exist_ok=True)
        fig.savefig(output_colorbar_path, bbox_inches='tight', dpi=100, transparent=False)
        plt.close(fig)
    except Exception as e:
        print(f"Error saving color bar image: {e}", file=sys.stderr)


    # 5. Generate and Save Heatmap Mesh
    max_intensity = np.max(final_vertex_intensities)
    normalized_intensities = final_vertex_intensities / max_intensity if max_intensity > 1e-9 else np.zeros_like(final_vertex_intensities)
    mesh_vertex_colors = cmap(normalized_intensities)[:, :3]
    mesh_vertex_colors[final_vertex_intensities < 1e-9] = base_color
    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_vertex_colors)

    try:
        os.makedirs(os.path.dirname(output_heatmap_path), exist_ok=True)
        o3d.io.write_triangle_mesh(output_heatmap_path, mesh, write_ascii=True)
    except Exception as e:
        print(f"Error saving heatmap mesh: {e}", file=sys.stderr)


    # 6. Generate and Save Point Cloud
    final_point_intensities = []
    for face_idx in point_to_face_map:
        final_point_intensities.append(np.mean(final_vertex_intensities[mesh_triangles_np[face_idx]]))
    final_point_intensities = np.array(final_point_intensities)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gaze_points_np))
    max_point_intensity = np.max(final_point_intensities)
    normalized_point_intensities = final_point_intensities / max_point_intensity if max_point_intensity > 1e-9 else np.zeros_like(final_point_intensities)

    pc_colors = cmap(normalized_point_intensities)[:, :3]
    pc_colors[final_point_intensities < 1e-9] = base_color
    pcd.colors = o3d.utility.Vector3dVector(pc_colors)

    try:
        os.makedirs(os.path.dirname(output_pointcloud_path), exist_ok=True)
        o3d.io.write_point_cloud(output_pointcloud_path, pcd, write_ascii=True)
    except Exception as e:
        print(f"Error saving point cloud: {e}", file=sys.stderr)


# yapf: disable
def generate_fixation_pointcloud_heatmap(
    input_file: str,
    model_file: str,
    cmap: matplotlib.colors.Colormap,
    base_color: list,
    dispersion_threshold: float,
    min_fixation_duration: float,
):
    """
    Analyzes gaze data using a dispersion-based algorithm (I-DT) to identify fixations.
    It generates a heatmap on a 3D mesh and a point cloud of fixation centroids.
    The color mapping for both is fixed to a scale of 0ms (minimum) to 1500ms (maximum).

    Args:
        input_file (str): Path to the CSV file with gaze data.
                          Expected format: [gaze_x, gaze_y, gaze_z, timestamp, ...].
        model_file (str): Path to the 3D model file (e.g., .obj, .ply).
        cmap (matplotlib.colors.Colormap): Matplotlib colormap object for the heatmap.
        base_color (list): RGB color (range [0,1]) for the mesh.
        dispersion_threshold (float): The maximum spatial dispersion (in meters) for a
                                      group of points to be considered a fixation.
        min_fixation_duration (float): The minimum duration (in seconds) for a group of gaze
                                       points to be considered a valid fixation.

    Returns:
        tuple[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]:
            - A point cloud where each point represents a fixation centroid, colored by duration.
            - The colored 3D mesh with the fixation duration heatmap.
    """
    # 1. Load Data and Mesh
    try:
        data = pd.read_csv(input_file, header=0).to_numpy()
        if data.shape[1] < 4:
            print(f"Warning: Not enough columns in {input_file} for fixation analysis. Skipping.")
            return o3d.geometry.PointCloud(), o3d.geometry.TriangleMesh()
        gaze_points_np = data[:, :4]
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        print(f"Warning: Could not read or parse {input_file}: {e}. Skipping fixation analysis.")
        return o3d.geometry.PointCloud(), o3d.geometry.TriangleMesh()

    mesh = o3d.io.read_triangle_mesh(model_file)
    if not mesh.has_vertices():
        return o3d.geometry.PointCloud(), o3d.geometry.TriangleMesh()
    mesh.compute_vertex_normals()

    # 2. Group Gaze Points into Fixations using I-DT Algorithm
    gaze_groups = []
    i = 0
    while i < len(gaze_points_np):
        window_start_index = i
        for j in range(window_start_index, len(gaze_points_np)):
            window_coords = gaze_points_np[window_start_index : j + 1, :3]
            if window_coords.shape[0] < 2:
                continue

            max_coords = np.max(window_coords, axis=0)
            min_coords = np.min(window_coords, axis=0)
            dispersion = np.sum(max_coords - min_coords)

            if dispersion > dispersion_threshold:
                fixation_end_index = j - 1
                if fixation_end_index >= window_start_index:
                    gaze_groups.append(gaze_points_np[window_start_index : fixation_end_index + 1])
                i = j
                break
        else:
            gaze_groups.append(gaze_points_np[window_start_index:, :])
            i = len(gaze_points_np)

    # 3. Process Groups to Get Valid Fixations (Centroid and Duration)
    fixations = []
    for group in gaze_groups:
        if len(group) > 1:
            duration = group[-1, 3] - group[0, 3]  # timestamp_end - timestamp_start
            if duration >= min_fixation_duration:
                centroid = np.mean(group[:, :3], axis=0) # Average X, Y, Z
                fixations.append({'centroid': centroid, 'duration': duration})

    if not fixations:
        mesh.paint_uniform_color(base_color)
        return o3d.geometry.PointCloud(), mesh

    # 4. Map Fixations to the Closest Vertex on the Mesh for Heatmap
    mesh_vertices = np.asarray(mesh.vertices)
    vertex_durations = np.zeros(len(mesh_vertices))
    pcd_tree = o3d.geometry.KDTreeFlann(mesh)

    for fix in fixations:
        [k, idx, _] = pcd_tree.search_knn_vector_3d(fix['centroid'], 1)
        if k > 0:
            vertex_durations[idx[0]] += fix['duration']

    # 5. Generate and Apply Heatmap Colors to Mesh with FIXED 0-1500ms Scale
    fixated_indices = np.where(vertex_durations > 0)[0]
    heatmap_mesh = deepcopy(mesh)

    if len(fixated_indices) > 0:
        # Clip durations at 1.5s for normalization
        clipped_vertex_durations = np.clip(vertex_durations, 0.0, 1.0)

        # Use a fixed normalization scale from 0 to 1.5 seconds (1500ms)
        norm = plt.Normalize(vmin=0.0, vmax=1.0)

        # Normalize only the vertices that were actually hit
        heatmap_colors = cmap(norm(clipped_vertex_durations[fixated_indices]))[:, :3]

        vertex_colors = np.tile(np.array(base_color), (len(mesh_vertices), 1))
        vertex_colors[fixated_indices] = heatmap_colors
        heatmap_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    else:
        heatmap_mesh.paint_uniform_color(base_color)

    # 6. Create Fixation Centroid Point Cloud with FIXED 0-1500ms Scale
    fixation_centroids_np = np.array([f['centroid'] for f in fixations])
    fixation_durations_np = np.array([f['duration'] for f in fixations])

    # Clip individual fixation durations at 1.5s for normalization
    clipped_point_durations = np.clip(fixation_durations_np, 0.0, 1.0)

    # Use the same fixed normalization scale for the points
    norm_points = plt.Normalize(vmin=0.0, vmax=1.0)
    point_colors = cmap(norm_points(clipped_point_durations))[:, :3]

    fixation_pcd = o3d.geometry.PointCloud()
    fixation_pcd.points = o3d.utility.Vector3dVector(fixation_centroids_np)
    fixation_pcd.colors = o3d.utility.Vector3dVector(point_colors)

    return fixation_pcd, heatmap_mesh
# yapf: enable