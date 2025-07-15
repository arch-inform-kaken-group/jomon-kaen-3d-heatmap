import os
import pathlib
import time
from collections import deque
import sys
import threading

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
from tqdm import tqdm

# --- Constants ---
HOLOLENS_2_SD_ERROR = 2.66
GAUSSIAN_SPREAD_DENOMINATOR = 2 * (HOLOLENS_2_SD_ERROR**2)
CMAP = plt.get_cmap('jet')

ANSWER_COLOR_MAP = {
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
DEFAULT_COLOR_RGB = [0, 0, 0]
DEFAULT_COLOR_NAME = "default"

# Define the target resolution for voxelization
TARGET_RESOLUTION = 512

# --- Configuration Flags ---
VISUALIZE = False
GENERATE_SANITY_CHECK = False
GENERATE_GAZE_VISUALIZATIONS = True
GENERATE_VOXEL = False
GENERATE_IMPRESSION = False

# --- Utility Functions ---


def visualize_geometry(geometry, point_size=1.0):
    """
    Visualizes a given Open3D geometry object.
    Supports setting point_size for point clouds.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(geometry)
    render_options = vis.get_render_option()
    if isinstance(geometry, o3d.geometry.PointCloud):
        render_options.point_size = point_size
    render_options.background_color = np.asarray([0.1, 0.1, 0.1])
    vis.run()
    vis.destroy_window()


# --- Threaded Save Functions ---


def save_geometry_threaded(save_path, geometry):
    """
    Initiates saving of an Open3D geometry object to a file in a separate thread.
    Returns the thread object for later joining.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def _save_task(path, geom):
        original_verbosity = o3d.utility.VerbosityLevel.Warning
        try:
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
            if isinstance(geom, o3d.geometry.PointCloud):
                o3d.io.write_point_cloud(path, geom, write_ascii=True)
            elif isinstance(geom, o3d.geometry.TriangleMesh):
                o3d.io.write_triangle_mesh(path, geom, write_ascii=True)
            else:
                print(
                    f"\nUnsupported geometry type for saving: {type(geom)} to {path}",
                    file=sys.stderr)
        except Exception as e:
            print(f"\nAn error occurred while saving geometry to {path}: {e}",
                  file=sys.stderr)
        finally:
            o3d.utility.set_verbosity_level(original_verbosity)
        print(
            f"\nSuccessfully saved {type(geom).__name__} to {path} in a separate thread."
        )

    save_thread = threading.Thread(target=_save_task,
                                   args=(save_path, geometry))
    save_thread.daemon = True
    save_thread.start()
    print(f"\nSaving of {type(geometry).__name__} to {save_path} initiated.")
    return save_thread


def save_plot_threaded(fig, output_plot_path):
    """
    Saves a matplotlib plot in a separate thread.
    """

    def _save_task():
        try:
            os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
            fig.savefig(output_plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(
                f"\nPlot saved successfully to {output_plot_path} in a separate thread."
            )
        except Exception as e:
            print(f"\nError saving plot to {output_plot_path}: {e}")

    plot_thread = threading.Thread(target=_save_task)
    plot_thread.daemon = True
    plot_thread.start()
    print(f"\nPlot saving initiated for {output_plot_path}.")
    return plot_thread


# --- Core Calculation and Processing Functions ---


def _calculate_smoothed_vertex_intensities(pcd_points_np, mesh):
    """
    Calculates final, smoothed intensity values on mesh vertices using a
    geometry-aware Gaussian spread to prevent "leaking" through mesh walls.
    """
    mesh_vertices_np = np.asarray(mesh.vertices)
    mesh_triangles_np = np.asarray(mesh.triangles)
    n_vertices = mesh_vertices_np.shape[0]

    print("Step 1/2: Mapping gaze points to nearest mesh faces...")
    mesh_scene = o3d.t.geometry.RaycastingScene()
    mesh_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    query_points = o3d.core.Tensor(pcd_points_np, dtype=o3d.core.Dtype.Float32)
    closest_geometry = mesh_scene.compute_closest_points(query_points)
    closest_face_indices = closest_geometry['primitive_ids'].numpy()

    raw_hit_counts = np.zeros(n_vertices, dtype=np.float64)
    point_to_face_map = np.empty(pcd_points_np.shape[0], dtype=int)

    for i, face_idx in enumerate(
            tqdm(closest_face_indices, desc="Accumulating hits on vertices")):
        if face_idx != o3d.t.geometry.RaycastingScene.INVALID_ID:
            point_to_face_map[i] = face_idx
            for v_idx in mesh_triangles_np[face_idx]:
                raw_hit_counts[v_idx] += 1

    # Log scaling improves visual detail, large numbers do not dominate the heatmap
    # causing the difference (comparison) to be lost. i.e. difference between 1 & 10 hits
    # and 100 & 1000 hits both are shown on the heatmap.
    #
    # Log scaling aligns with human perception (logarithmic)
    # more sensitive to change at lower levels of stimulus compared to high levels.
    #
    # Enables the handling of wide dynamic ranges. If gaze is recorded for 5 mins, etc.
    raw_hit_counts = np.log1p(raw_hit_counts)

    print("Step 2/2: Applying Gaussian spread along mesh surface...")
    if (HOLOLENS_2_SD_ERROR <= 8):
        # This method of spreading will cause leakage for high values of error
        # However, it will ensure nearby vertices recieve color, even if the
        # original mesh is malformed (not fully connected, missing edges)
        kdtree = o3d.geometry.KDTreeFlann(mesh)
        interpolated_heatmap_values = np.copy(raw_hit_counts)
        hit_vertices_indices = np.where(raw_hit_counts > 0)[0]

        for start_node_idx in tqdm(
                hit_vertices_indices,
                desc="Spreading heatmap within HoloLens 2 Error Range"):
            hit_value = raw_hit_counts[start_node_idx]
            [k, indices, euclidean_dist] = kdtree.search_radius_vector_3d(
                mesh_vertices_np[start_node_idx], HOLOLENS_2_SD_ERROR)
            if k > 1:
                gaussian_weights = np.exp(-np.asarray(euclidean_dist)**2 /
                                          GAUSSIAN_SPREAD_DENOMINATOR)
                for i, neighbor_idx in enumerate(indices):
                    if neighbor_idx != start_node_idx:
                        interpolated_heatmap_values[
                            neighbor_idx] += hit_value * gaussian_weights[i]

    else:
        # This method ensures that there is no leakage
        # However, some vertices will not recieve color if they are not connected properly
        # caused by errors during model downsizing or scanning
        vertex_adjacency = {i: set() for i in range(n_vertices)}
        for v0, v1, v2 in mesh_triangles_np:
            vertex_adjacency[v0].update([v1, v2])
            vertex_adjacency[v1].update([v0, v2])
            vertex_adjacency[v2].update([v0, v1])

        interpolated_heatmap_values = np.zeros(n_vertices, dtype=np.float64)
        hit_vertices_indices = np.where(raw_hit_counts > 0)[0]

        for start_node_idx in tqdm(hit_vertices_indices,
                                   desc="Spreading heatmap via BFS"):
            hit_value = raw_hit_counts[start_node_idx]
            start_pos = mesh_vertices_np[start_node_idx]

            q = deque([start_node_idx])
            visited = {start_node_idx}
            interpolated_heatmap_values[start_node_idx] += hit_value

            while q:
                current_idx = q.popleft()

                for neighbor_idx in vertex_adjacency[current_idx]:
                    if neighbor_idx not in visited:
                        dist_from_start = np.linalg.norm(
                            mesh_vertices_np[neighbor_idx] - start_pos)
                        if dist_from_start <= HOLOLENS_2_SD_ERROR:
                            visited.add(neighbor_idx)
                            q.append(neighbor_idx)
                            distance_sq = dist_from_start**2
                            gaussian_weight = np.exp(
                                -distance_sq / GAUSSIAN_SPREAD_DENOMINATOR)
                            interpolated_heatmap_values[
                                neighbor_idx] += hit_value * gaussian_weight

    return interpolated_heatmap_values, point_to_face_map


def generate_gaze_visualizations(input_file,
                                 model_file,
                                 output_mesh_file,
                                 output_point_cloud_file,
                                 visualize=False):
    """
    Generates a gaze density heatmap on a 3D mesh and an intensity-colored point cloud.
    Returns the processing threads, the final mesh, and the calculated vertex intensities.
    """
    print(
        "\n--- Generating Gaze Visualizations (Heatmap Mesh & Point Cloud) ---"
    )
    active_threads = []
    try:
        data = pd.read_csv(input_file, header=0).to_numpy()
        pcd_points_np = data[:, :3]
        mesh = o3d.io.read_triangle_mesh(model_file)
        if not mesh.has_vertices():
            raise ValueError(f"Mesh file '{model_file}' contains no vertices.")
        mesh.compute_vertex_normals()
    except Exception as e:
        print(f"Error during setup: {e}")
        return [], None, None

    final_vertex_intensities, point_to_face_map = _calculate_smoothed_vertex_intensities(
        pcd_points_np, mesh)

    # --- Generate Heatmap Mesh ---
    max_val_mesh = np.max(final_vertex_intensities)
    normalized_values_mesh = final_vertex_intensities / max_val_mesh if max_val_mesh > 0 else np.zeros_like(
        final_vertex_intensities)

    final_mesh_vertex_colors = CMAP(normalized_values_mesh)[:, :3]
    final_mesh_vertex_colors[final_vertex_intensities <
                             1e-9] = DEFAULT_COLOR_RGB
    mesh.vertex_colors = o3d.utility.Vector3dVector(final_mesh_vertex_colors)
    active_threads.append(save_geometry_threaded(output_mesh_file, mesh))

    # --- Generate Intensity-Colored Point Cloud ---
    mesh_triangles_np = np.asarray(mesh.triangles)
    # final_point_intensities = np.array([
    #     np.mean(final_vertex_intensities[mesh_triangles_np[face_idx]])
    #     for face_idx in point_to_face_map
    # ])
    final_point_intensities = []
    for face_idx in point_to_face_map:
        final_point_intensities.append(
            np.mean(final_vertex_intensities[mesh_triangles_np[face_idx]]))
    final_point_intensities = np.array(final_point_intensities)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_points_np))
    max_val_pc = np.max(final_point_intensities)
    normalized_values_pc = final_point_intensities / max_val_pc if max_val_pc > 0 else np.zeros_like(
        final_point_intensities)

    colors_pc = CMAP(normalized_values_pc)[:, :3]
    colors_pc[final_point_intensities < 1e-9] = DEFAULT_COLOR_RGB
    pcd.colors = o3d.utility.Vector3dVector(colors_pc)
    active_threads.append(save_geometry_threaded(output_point_cloud_file, pcd))

    if visualize:
        o3d.visualization.draw_geometries([mesh, pcd])

    return active_threads, mesh, final_vertex_intensities


def generate_voxel_from_mesh(
    mesh,
    vertex_intensities,
    output_voxel_ply_file,
    visualize=False,
):
    """
    Generates a voxel representation by rasterizing mesh triangles, creating voxels
    for the entire surface with smoothly interpolated intensities.
    """
    print(
        f"\n--- Generating Voxel Heatmap ({TARGET_RESOLUTION}^3 Resolution) ---"
    )
    if mesh is None or vertex_intensities is None:
        print("Skipping voxel heatmap: Missing mesh or intensity data.")
        return None

    if not mesh.has_triangles() or not mesh.has_vertices():
        print("Skipping voxel heatmap: Mesh has no triangles or vertices.")
        return None

    mesh_vertices_np = np.asarray(mesh.vertices)
    mesh_triangles_np = np.asarray(mesh.triangles)

    min_bound = mesh.get_min_bound()
    max_bound = mesh.get_max_bound()
    max_range = np.max(max_bound - min_bound)
    if max_range < 1e-9: max_range = 1.0

    voxel_size = max_range / (TARGET_RESOLUTION - 1)

    print(f"Calculated uniform voxel size: {voxel_size:.4f}")

    # This dictionary will store the maximum intensity found for each voxel
    voxel_data = {}

    # Iterate over each triangle in the mesh to rasterize it
    for triangle in tqdm(mesh_triangles_np,
                         desc="Rasterizing triangles into voxels"):
        v_idx0, v_idx1, v_idx2 = triangle
        v0, v1, v2 = mesh_vertices_np[v_idx0], mesh_vertices_np[
            v_idx1], mesh_vertices_np[v_idx2]
        i0, i1, i2 = vertex_intensities[v_idx0], vertex_intensities[
            v_idx1], vertex_intensities[v_idx2]

        # --- Determine the number of samples for this triangle ---
        # We sample more densely on larger triangles to ensure they are filled
        edge1 = v1 - v0
        edge2 = v2 - v0
        triangle_area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))

        # Number of samples is proportional to the triangle's area relative to a voxel's area
        # A base number ensures even tiny triangles get sampled
        num_samples = int(np.ceil(triangle_area / (voxel_size**2))) + 10

        # --- Generate sample points within the triangle using barycentric coordinates ---
        r = np.random.rand(num_samples, 2)
        # If r1+r2 > 1, the point is outside the triangle. Reflect it back inside.
        r_sum = np.sum(r, axis=1)
        r[r_sum > 1] = 1 - r[r_sum > 1]

        # Convert to barycentric coordinates (w0, w1, w2)
        bary_coords = np.zeros((num_samples, 3))
        bary_coords[:, 0] = 1 - r[:, 0] - r[:, 1]
        bary_coords[:, 1] = r[:, 0]
        bary_coords[:, 2] = r[:, 1]

        # Calculate the 3D position and interpolated intensity for each sample
        sample_points = bary_coords @ np.array([v0, v1, v2])
        interp_intensities = bary_coords @ np.array([i0, i1, i2])

        # --- Assign each sample to a voxel ---
        for point, intensity in zip(sample_points, interp_intensities):
            # Calculate the voxel grid coordinates for the sample point
            voxel_coords = tuple(
                np.floor((point - min_bound) / voxel_size).astype(int))

            # Update the voxel's intensity only if the new sample is higher
            # This preserves the peaks of the heatmap
            current_max = voxel_data.get(voxel_coords, -1.0)
            voxel_data[voxel_coords] = max(current_max, intensity)

    print(
        f"Generated {len(voxel_data)} occupied voxels from triangle rasterization."
    )
    if not voxel_data:
        return None

    # --- Convert the final voxel data into a colored point cloud ---
    voxel_points = []
    final_intensities = []
    for coords, intensity in voxel_data.items():
        voxel_center = min_bound + (np.array(coords) + 0.5) * voxel_size
        voxel_points.append(voxel_center)
        final_intensities.append(intensity)

    final_intensities_np = np.array(final_intensities)
    max_val = np.max(final_intensities_np)

    # Normalize intensities for coloring
    if max_val > 1e-9:
        normalized_intensities = final_intensities_np / max_val
    else:
        normalized_intensities = np.zeros_like(final_intensities_np)

    # Apply the colormap
    colors = CMAP(normalized_intensities)[:, :3]
    # Use a neutral dark grey for areas with no recorded gaze for better visibility
    colors[normalized_intensities < 1e-9] = DEFAULT_COLOR_RGB

    voxel_pcd = o3d.geometry.PointCloud()
    voxel_pcd.points = o3d.utility.Vector3dVector(np.array(voxel_points))
    voxel_pcd.colors = o3d.utility.Vector3dVector(colors)

    thread = save_geometry_threaded(output_voxel_ply_file, voxel_pcd)

    if visualize:
        print("Visualizing rasterized voxel grid...")
        voxel_grid_vis = o3d.geometry.VoxelGrid.create_from_point_cloud(
            voxel_pcd, voxel_size=voxel_size)
        o3d.visualization.draw_geometries([voxel_grid_vis])

    return thread


def process_questionnaire_answers(
    qa_input_file,
    model_file,
    output_qa_pointcloud_file,
    output_combined_mesh_file,
    output_segmented_meshes_dir,
    feature_size,
    visualize=False,
):
    """
    Generates a raw answer-colored point cloud, a blended mesh, and individual
    segmented meshes, using a BFS constrained by squared Euclidean distance for speed.
    """
    print("\n--- Processing Questionnaire Answers ---")

    try:
        df = pd.read_csv(qa_input_file, sep=',', header=0)
        df['estX'] = pd.to_numeric(df['estX'], errors='coerce')
        df['estY'] = pd.to_numeric(df['estY'], errors='coerce')
        df['estZ'] = pd.to_numeric(df['estZ'], errors='coerce')
        df['answer'] = df['answer'].astype(str).str.strip()
        df.dropna(subset=['estX', 'estY', 'estZ', 'answer'], inplace=True)
        mesh = o3d.io.read_triangle_mesh(model_file)
        if not mesh.has_vertices():
            raise ValueError(f"Mesh file '{model_file}' contains no vertices.")
    except Exception as e:
        print(f"Error during setup for questionnaire processing: {e}")
        return None

    print("Generating raw answer-colored point cloud...")
    qa_points = df[['estX', 'estY', 'estZ']].values
    qa_colors_01 = [
        np.array(ANSWER_COLOR_MAP.get(ans, {"rgb": DEFAULT_COLOR_RGB})["rgb"])
        / 255.0 for ans in df['answer']
    ]
    qa_pcd = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(qa_points))
    qa_pcd.colors = o3d.utility.Vector3dVector(np.array(qa_colors_01))
    save_geometry_threaded(output_qa_pointcloud_file, qa_pcd)

    print("\nGenerating blended and segmented answer-colored meshes...")
    mesh_vertices_np = np.asarray(mesh.vertices)
    mesh_triangles_np = np.asarray(mesh.triangles)
    n_vertices = mesh_vertices_np.shape[0]

    mesh_triangle_centroids = mesh_vertices_np[mesh_triangles_np].mean(axis=1)
    mesh_centroid_kdtree = o3d.geometry.KDTreeFlann(
        o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(mesh_triangle_centroids)))

    vertex_adjacency = {i: set() for i in range(n_vertices)}
    for v0, v1, v2 in mesh_triangles_np:
        vertex_adjacency[v0].update([v1, v2])
        vertex_adjacency[v1].update([v0, v2])
        vertex_adjacency[v2].update([v0, v1])

    final_colors = np.zeros((n_vertices, 3), dtype=float)
    total_weights = np.zeros(n_vertices, dtype=float)

    # --- OPTIMIZATION APPLIED HERE ---
    # Pre-calculate squared values to avoid square roots in the loops
    feature_size_sq = feature_size**2
    gaussian_spread_denominator_qa = 2 * feature_size_sq
    # --------------------------------

    unique_answers = df['answer'].unique()
    all_hit_vertices_by_answer = {}

    for answer in unique_answers:
        category_df = df[df['answer'] == answer]
        category_points = category_df[['estX', 'estY', 'estZ']].values
        color_info = ANSWER_COLOR_MAP.get(answer, {"rgb": DEFAULT_COLOR_RGB})
        color_vec = np.array(color_info["rgb"]) / 255.0
        hit_vertices = set()

        for point in tqdm(category_points,
                          desc=f"Mapping '{answer}' points",
                          leave=False):
            [k, idx_list,
             _] = mesh_centroid_kdtree.search_knn_vector_3d(point, 1)
            if k > 0:
                for v_idx in mesh_triangles_np[idx_list[0]]:
                    hit_vertices.add(v_idx)
        all_hit_vertices_by_answer[answer] = hit_vertices

        for start_node_idx in tqdm(list(hit_vertices),
                                   desc=f"Spreading '{answer}' color",
                                   leave=False):
            start_pos = mesh_vertices_np[start_node_idx]
            q = deque([start_node_idx])
            visited = {start_node_idx}
            final_colors[start_node_idx] += color_vec
            total_weights[start_node_idx] += 1.0

            while q:
                current_idx = q.popleft()
                for neighbor_idx in vertex_adjacency[current_idx]:
                    if neighbor_idx not in visited:
                        # Use squared distance for comparison
                        dist_sq_from_start = np.sum(
                            (mesh_vertices_np[neighbor_idx] - start_pos)**2)
                        if dist_sq_from_start <= feature_size_sq:
                            visited.add(neighbor_idx)
                            q.append(neighbor_idx)
                            gaussian_weight = np.exp(
                                -dist_sq_from_start /
                                gaussian_spread_denominator_qa)
                            final_colors[
                                neighbor_idx] += color_vec * gaussian_weight
                            total_weights[neighbor_idx] += gaussian_weight

    print("Blending colors across the mesh...")
    valid_weights_mask = total_weights > 1e-9
    final_colors[valid_weights_mask] /= total_weights[valid_weights_mask,
                                                      np.newaxis]
    final_colors[~valid_weights_mask] = [0.0, 0.0, 0.0]
    combined_mesh = o3d.geometry.TriangleMesh(mesh)
    combined_mesh.vertex_colors = o3d.utility.Vector3dVector(final_colors)
    save_geometry_threaded(output_combined_mesh_file, combined_mesh)

    print("\nGenerating Individual Segmented Meshes...")
    os.makedirs(output_segmented_meshes_dir, exist_ok=True)
    for answer, hit_vertices in all_hit_vertices_by_answer.items():
        color_info = ANSWER_COLOR_MAP.get(answer, {
            "rgb": DEFAULT_COLOR_RGB,
            "name": DEFAULT_COLOR_NAME
        })
        color_vec = np.array(color_info["rgb"]) / 255.0
        segmented_colors = np.zeros((n_vertices, 3), dtype=float)

        for start_node_idx in tqdm(list(hit_vertices),
                                   desc=f"Coloring '{answer}' segment",
                                   leave=False):
            start_pos = mesh_vertices_np[start_node_idx]
            q = deque([start_node_idx])
            visited = {start_node_idx}
            segmented_colors[start_node_idx] = color_vec
            while q:
                current_idx = q.popleft()
                for neighbor_idx in vertex_adjacency[current_idx]:
                    if neighbor_idx not in visited:
                        # Use squared distance for comparison
                        dist_sq_from_start = np.sum(
                            (mesh_vertices_np[neighbor_idx] - start_pos)**2)
                        if dist_sq_from_start <= feature_size_sq:
                            visited.add(neighbor_idx)
                            q.append(neighbor_idx)
                            segmented_colors[neighbor_idx] = color_vec

        segmented_mesh = o3d.geometry.TriangleMesh(mesh)
        segmented_mesh.vertex_colors = o3d.utility.Vector3dVector(
            segmented_colors)
        output_path = os.path.join(output_segmented_meshes_dir,
                                   f"{color_info['name']}.ply")
        save_geometry_threaded(output_path, segmented_mesh)

    if visualize:
        print("Visualizing the combined blended mesh...")
        visualize_geometry(combined_mesh)
    return combined_mesh


def analyze_and_plot_point_cloud(csv_file_path, output_plot_path):
    """
    Reads a CSV file with 3D point data, counts occurrences, and saves a 3D scatter plot.
    """
    print("\n--- Sanity Check: Analyzing Raw Point Cloud ---")
    if not os.path.exists(csv_file_path):
        print(f"Error: The file was not found at '{csv_file_path}'")
        return

    print(f"Reading data from '{csv_file_path}'...")
    try:
        df = pd.read_csv(csv_file_path, header=0, usecols=[0, 1, 2])
        df.columns = ['x', 'y', 'z']
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return

    if df.empty:
        print("The CSV file is empty. Nothing to process.")
        return

    initial_points = len(df)
    print(f"Successfully loaded {initial_points} total points.")

    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df['z'] = pd.to_numeric(df['z'], errors='coerce')
    df.dropna(inplace=True)
    cleaned_points = len(df)
    if cleaned_points < initial_points:
        print(
            f"Warning: Dropped {initial_points - cleaned_points} non-numeric row(s)."
        )
    if df.empty:
        print("No valid numeric data remains after cleaning. Exiting.")
        return

    print("\n--- Point Occurrence Analysis ---")
    point_counts = df.groupby(['x', 'y', 'z']).size().reset_index(name='count')

    print(f"Generating occurrence plot and saving to {output_plot_path}...")
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
                                   args=(fig, output_plot_path))
    plot_thread.daemon = True  # Allows the main program to exit even if the thread is still running
    plot_thread.start()

    print(
        f"Plot saving initiated for {output_plot_path} in a separate thread.")


if __name__ == '__main__':
    curr_dir = pathlib.Path.cwd()

    start_time = time.time_ns()
    group_root_path = curr_dir / 'src' / 'data'

    if not group_root_path.exists():
        print(f"Error: Data directory not found at {group_root_path}")
    else:
        active_save_threads = []
        for g in os.listdir(group_root_path):
            session_root_path = group_root_path / str(g)
            for session_name in os.listdir(session_root_path):
                model_root_path = session_root_path / session_name
                if not model_root_path.is_dir(): continue

                for model_name in os.listdir(model_root_path):
                    datafile_paths = model_root_path / model_name
                    if not datafile_paths.is_dir(): continue

                    input_file = datafile_paths / "pointcloud.csv"
                    model_file = datafile_paths / "model.obj"
                    qa_input_file = datafile_paths / "qa.csv"

                    # Define output paths
                    output_sanity_plot = datafile_paths / "pointcloud_occurrence_plot.png"
                    output_point_cloud = datafile_paths / "eye_gaze_intensity_pc.ply"
                    output_mesh = datafile_paths / "eye_gaze_intensity_hm.ply"
                    output_voxel_heatmap_ply = datafile_paths / "eye_gaze_voxel.ply"
                    output_qa_ply = datafile_paths / "qa_pc.ply"
                    output_segmented_meshes_dir = datafile_paths / "qa_segmented_mesh"
                    output_combined_mesh_file = datafile_paths / "combined_qa_mesh.ply"

                    print(
                        f"\n----- Processing Session: {datafile_paths.relative_to(session_root_path)} -----"
                    )

                    processed_mesh = None
                    calculated_intensities = None

                    if GENERATE_SANITY_CHECK and os.path.exists(input_file):
                        analyze_and_plot_point_cloud(str(input_file),
                                                     output_sanity_plot)

                    if GENERATE_GAZE_VISUALIZATIONS and input_file.exists(
                    ) and model_file.exists():
                        threads, processed_mesh, calculated_intensities = generate_gaze_visualizations(
                            str(input_file),
                            str(model_file),
                            str(output_mesh),
                            str(output_point_cloud),
                            visualize=VISUALIZE,
                        )
                        active_save_threads.extend(threads)
                    elif GENERATE_GAZE_VISUALIZATIONS:
                        print(
                            f"Skipping gaze visualizations: Missing input or model file."
                        )

                    if GENERATE_IMPRESSION and os.path.exists(
                            qa_input_file) and os.path.exists(model_file):
                        print("\n=== Processing questionnaire answers ===")
                        process_questionnaire_answers(
                            str(qa_input_file),
                            str(model_file),
                            str(output_qa_ply),
                            str(output_combined_mesh_file),
                            str(output_segmented_meshes_dir),
                            HOLOLENS_2_SD_ERROR,
                            visualize=VISUALIZE,
                        )
                    elif GENERATE_IMPRESSION:
                        print(
                            f"Skipping questionnaire answer processing: Missing qa_input_file or model_file for {datafile_paths}"
                        )

                    if GENERATE_VOXEL:
                        thread = generate_voxel_from_mesh(
                            processed_mesh,
                            calculated_intensities,
                            str(output_voxel_heatmap_ply),
                            visualize=VISUALIZE,
                        )
                        if thread: active_save_threads.append(thread)

        print(
            "\n--- All processing tasks initiated. Waiting for file saves to complete... ---"
        )
        for t in active_save_threads:
            t.join()

    end_time = time.time_ns()
    print(f"\n TOTAL ELAPSED TIME: {(end_time - start_time)/1e9:.2f} seconds")
