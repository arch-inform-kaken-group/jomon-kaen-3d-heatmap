import numpy as np #
import pandas as pd #
import open3d as o3d #
import trimesh  # For voxelizing pottery and dogu with color

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
    num_samples_per_triangle = np.ceil(triangle_areas / voxel_size_sq).astype(int) + 20
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
