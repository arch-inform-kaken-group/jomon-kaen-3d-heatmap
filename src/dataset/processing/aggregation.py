from copy import deepcopy

import numpy as np  #
import pandas as pd  #
import open3d as o3d  #
from tqdm import tqdm  #


# yapf: disable
# Read about KD-Tree (Medium | EN): https://medium.com/@isurangawarnasooriya/exploring-kd-trees-a-comprehensive-guide-to-implementation-and-applications-in-python-3385fd56a246
# Read about KD-Tree (Qiita | JP): https://qiita.com/RAD0N/items/7a192a4a5351f481c99f
def _calculate_smoothed_vertex_intensities(
    gaze_points_np,
    mesh,
    # https://arxiv.org/abs/2111.07209 [An Assessment of the Eye Tracking Signal Quality Captured in the HoloLens 2]
    # Official: 1.5 | Paper original: 6.45 | Paper recalibrated: 2.66
    hololens_2_spatial_error,
    # gaussian_denominator = 2 * (hololens_2_spatial_error ^ 2)
    gaussian_denominator,
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
            #           n(points in radius)                     squared_euclidean_distance
            # GAI =         SUM             weight_of_point * e ^ - _____________________________
            #               i = 1                                   gaussian_denominator
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

    # Generate Greyscale Mesh
    # 1. Create a deep copy to avoid modifying the colored mesh
    mesh_greyscale = deepcopy(mesh)

    # 2. Create grayscale colors by repeating the intensity value for R, G, and B channels
    greyscale_colors = np.repeat(normalized_vertex_intensities[:, np.newaxis], 3, axis=1)

    # 3. Assign the new grayscale colors to the copied mesh
    mesh_greyscale.vertex_colors = o3d.utility.Vector3dVector(greyscale_colors)

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

    return pcd, mesh, final_vertex_intensities, mesh_greyscale
# yapf: enable


# yapf: disable
def generate_voxel_from_mesh(
    mesh,
    vertex_intensities,
    target_voxel_resolution,
    cmap,
    base_color,
    base_pottery_pcd=None,
):
    """
    Generates a voxel heatmap from a mesh.

    Two modes of operation:
    1. If base_pottery_pcd is None (default):
        Generates a new voxel grid by sampling points within the mesh's triangles.
        The output may not align with other external voxel grids.

    2. If base_pottery_pcd is provided:
        Uses the exact points from the provided point cloud as the basis. It casts rays
        from these points to find their corresponding intensity on the mesh surface.
        This guarantees the output heatmap is perfectly aligned with the base pottery.
    """
    if mesh is None or vertex_intensities is None:
        raise ValueError("Skipping voxel heatmap: Missing mesh or intensity data.")

    if not mesh.has_triangles() or not mesh.has_vertices():
        raise ValueError("Skipping voxel heatmap: Mesh has no triangles or vertices.")

    if base_pottery_pcd is not None:

        base_pottery_pcd = o3d.io.read_point_cloud(base_pottery_pcd)

        num_pottery_points = len(base_pottery_pcd.points)
        if num_pottery_points == 0:
            print("WARNING: Input pottery point cloud is empty.")
            return o3d.geometry.PointCloud()

        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        scene = o3d.t.geometry.RaycastingScene()
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        _ = scene.add_triangles(mesh_t)

        query_points = o3d.core.Tensor(np.asarray(base_pottery_pcd.points), dtype=o3d.core.Dtype.Float32)
        closest_points_ans = scene.compute_closest_points(query_points)

        # 1. Get the essential data that IS available from the result.
        #    'primitive_ids' = The triangle that the closest point lies on.
        #    'points' = The 3D coordinate of that closest point on the surface.
        triangle_ids = closest_points_ans['primitive_ids'].numpy()
        closest_surface_points = closest_points_ans['points'].numpy()

        # 2. Get the three vertices (a, b, c) for each identified triangle.
        mesh_vertices = np.asarray(mesh.vertices)
        mesh_triangles = np.asarray(mesh.triangles)
        hit_tri_vertices = mesh_vertices[mesh_triangles[triangle_ids]]
        a, b, c = hit_tri_vertices[:, 0], hit_tri_vertices[:, 1], hit_tri_vertices[:, 2]

        # 3. Calculate barycentric coordinates (u, v, w) using the point and triangle vertices.
        v0, v1 = b - a, c - a
        v2 = closest_surface_points - a

        d00 = np.einsum('ij,ij->i', v0, v0)
        d01 = np.einsum('ij,ij->i', v0, v1)
        d11 = np.einsum('ij,ij->i', v1, v1)
        d20 = np.einsum('ij,ij->i', v2, v0)
        d21 = np.einsum('ij,ij->i', v2, v1)

        denom = d00 * d11 - d01 * d01
        # Add a small epsilon to the denominator to prevent division by zero for degenerate triangles
        denom[np.abs(denom) < 1e-9] = 1e-9

        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        bary_coords = np.vstack([u, v, w]).T

        # 4. Interpolate the intensity using our manually calculated coordinates.
        tri_intensities = vertex_intensities[mesh_triangles[triangle_ids]]
        final_intensities = np.einsum('ij,ij->i', bary_coords, tri_intensities)

        # 5. Normalize and apply colormap.
        max_val = np.max(final_intensities) if len(final_intensities) > 0 else 0
        if max_val > 1e-9:
            normalized_intensities = final_intensities / max_val
        else:
            normalized_intensities = np.zeros_like(final_intensities)

        # colors = cmap(normalized_intensities)[:, :3]
        # colors[normalized_intensities < 1e-9] = base_color
        colors = np.repeat(normalized_intensities[:, np.newaxis], 3, axis=1)
        colors[normalized_intensities < 1e-9] = base_color

        # 6. Create the final heatmap with guaranteed 1-to-1 correspondence.
        heatmap_pcd = o3d.geometry.PointCloud()
        heatmap_pcd.points = base_pottery_pcd.points
        heatmap_pcd.colors = o3d.utility.Vector3dVector(colors)

        return heatmap_pcd

    else:
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
