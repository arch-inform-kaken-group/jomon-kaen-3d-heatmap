import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
# from scipy.spatial import KDTree
from tqdm import tqdm
import os
import pathlib
import pandas as pd
import polars as pl

cmap = plt.get_cmap('jet')
# SD_2_SQUARED_SPATIAL_ACCURACY = (2 * 6.45)**2

# def gaussian_blur_intensity(positions, sigma):
#     """Applies Gaussian blur to estimate point intensity."""
#     n_points = len(positions)
#     intensity = np.zeros(n_points)
#     tree = KDTree(positions)
#     radius = 3 * sigma

#     for i in tqdm(range(n_points), desc="Calculating Gaussian intensity"):
#         neighbors_idx = tree.query_ball_point(positions[i], radius)
#         # Subtract 1 to exclude the point itself
#         intensity[i] = len(neighbors_idx) - 1

#     # Normalize intensity values
#     min_intensity = np.min(intensity)
#     max_intensity = np.max(intensity)
#     normalized_intensity = (intensity - min_intensity) / (
#         max_intensity - min_intensity
#     ) if max_intensity > min_intensity else np.zeros_like(intensity)

#     return normalized_intensity


# Read about KD-Tree (Medium | EN): https://medium.com/@isurangawarnasooriya/exploring-kd-trees-a-comprehensive-guide-to-implementation-and-applications-in-python-3385fd56a246
# Read about KD-Tree (Qiita | JP): https://qiita.com/RAD0N/items/7a192a4a5351f481c99f
def calculate_point_intensity(pcd, ball_radius):
    """
    Calculate point intensity using Open3D's KDTree for radius search.
    
    Args:
        pcd: Open3D point cloud object
        ball_radius: Radius for neighborhood search
        
    Returns:
        normalized_intensity: Array of normalized intensity values
    """
    # Build a KD-tree with FLANN for efficient radius search
    # Open3D docs: https://www.open3d.org/docs/release/tutorial/geometry/kdtree.html
    # FLANN docs: https://www.cs.ubc.ca/research/flann/
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # Calculate intensity | Find the number of points in a radius of each point
    intensity = []
    for i, point in enumerate(
            tqdm(pcd.points, desc="Calculating point intensity")):
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
    normalized_intensity = (intensity - min_intensity) / (
        max_intensity - min_intensity
    ) if max_intensity > min_intensity else np.zeros_like(intensity)

    return normalized_intensity


def create_and_visualize_xyz_colored_point_cloud(input_file,
                                                 output_file,
                                                 ball_radius=10,
                                                 visualize=True):
    """
    Creates and visualizes a point cloud from XYZ data in a CSV file,
    coloring points based on their local intensity using a KD-tree for radius search.
    
    Args:
        input_file (str): Path to the input CSV file containing point data (x, y, z, ...).
        output_file (str): Path to save the resulting PLY file.
        ball_radius (float): Radius of the sphere used for neighborhood search.
        visualize (bool): Whether to visualize the point cloud after creation.
        
    Returns:
        pcd: The created Open3D point cloud object.
    """
    # Load data from the CSV file
    try:
        # data = np.genfromtxt(input_file, delimiter=',', skip_header=1)
        data = pd.read_csv(input_file, header=None, skiprows=1).to_numpy()
        # data = pl.read_csv(input_file, skip_lines=1, has_header=False).to_numpy()

    except Exception as e:
        print(f"Error loading data from {input_file}: {e}")
        return None

    if (data.ndim == 1):
        return

    positions = data[:, :3]  # Extract XYZ coordinates

    # Create an Open3d point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)

    # # Use the previously calculated average distance to set a default ball_radius
    # # Further investigation will be done to evaluate if this should be included in the final dataset code
    # distances = pcd.compute_nearest_neighbor_distance()
    # avg_distance = np.mean(distances)
    # if ball_radius <= 0:
    #     ball_radius = avg_distance * 2

    # Calculate intensity
    normalized_intensity = calculate_point_intensity(pcd, ball_radius)

    # Use the 'jet' colormap from matplotlib to map intensity to colors
    # colors = plt.get_cmap('jet')(normalized_intensity)[:, :3] # Get RGB values
    colors = cmap(normalized_intensity)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save the point cloud to a PLY file
    o3d.io.write_point_cloud(output_file, pcd, write_ascii=True)
    print(f"Point cloud saved to: {output_file}")

    # Visualize the point cloud
    if visualize:
        visualize_geometry(pcd, point_size=4.0)

    return pcd


def create_heatmap_mesh_from_intensity(
        input_file,
        model_file,
        output_file,
        # https://arxiv.org/abs/2111.07209 [An Assessment of the Eye Tracking Signal Quality Captured in the HoloLens 2]
        # Official: 1.5 | Paper original: 6.45 | Paper recalibrated: 2.66
        HOLOLENS_2_SPATIAL_ERROR=6.45,
        visualize=True,
        ball_radius=0.05):
    """
    Generates a heatmap on a provided 3D mesh model based on the intensity
    of the input points, using a gaussian adjusted intensity
    for region coloring.

    Args:
        input_file (str): Path to the input CSV file containing point data (x, y, z).
        model_file (str): Path to the input OBJ/PLY model file.
        output_file (str): Path to save the resulting OBJ file with the intensity heatmap on the mesh.
        interpolation_radius (float): Radius for interpolation around each mesh vertex.
        visualize (bool): Whether to visualize the mesh after creation.
        
    Returns:
        mesh: The created Open3D mesh object with intensity colors.
    """
    SD_2_SQUARED_SPATIAL_ACCURACY = (2 * HOLOLENS_2_SPATIAL_ERROR)**2

    # data = np.genfromtxt(input_file, delimiter=',', skip_header=1)
    data = pd.read_csv(input_file, header=None, skiprows=1).to_numpy()
    # data = pl.read_csv(input_file, skip_lines=1, has_header=False).to_numpy()

    if (data.ndim == 1):
        return

    positions = data[:, :3]  # Extract XYZ coordinates

    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)

    # # Use the previously calculated average distance to set a default ball_radius
    # distances = pcd.compute_nearest_neighbor_distance()
    # avg_distance = np.mean(distances)
    # if ball_radius <= 0:
    #     ball_radius = avg_distance * 10

    # Calculate intensity
    normalized_intensity = calculate_point_intensity(pcd, ball_radius)

    # Load the mesh (OBJ or PLY supported by Open3D)
    mesh = o3d.io.read_triangle_mesh(model_file)
    vertices = np.asarray(mesh.vertices)
    n_vertices = vertices.shape[0]

    # Build KDTree for efficient radius search
    # tree = KDTree(positions)
    tree = o3d.geometry.KDTreeFlann(pcd)
    colors_weights = np.zeros(n_vertices)
    colors = np.zeros((n_vertices, 3))
    # cmap = plt.get_cmap("jet")

    for i in tqdm(range(n_vertices), desc="Applying intensity to mesh"):
        vertex = vertices[i]
        # Find points within interpolation_radius of the vertex
        # nearby_point_indices = tree.query_ball_point(vertex, interpolation_radius)
        [k, nearby_point_indices,
         _] = tree.search_radius_vector_3d(vertex, HOLOLENS_2_SPATIAL_ERROR)

        # Weighted interpolation
        if nearby_point_indices:
            # weights = []
            # for point_index in nearby_point_indices:
            #     distance = np.linalg.norm(vertex - positions[point_index])
            #     weights.append(1.0 / (distance + 1e-6))
            # weights = np.array(weights)
            # weighted_densities = weights * normalized_intensity[nearby_point_indices]
            # interpolated_intensity = np.sum(weighted_densities) / np.sum(
            #     weights)
            # colors[i, :] = cmap(interpolated_intensity)[:3]
            
            # colors[i, :] = cmap(np.average(normalized_intensity[nearby_point_indices]))[:3]

            # Calculate the gaussian adjusted intensity of each vertex based on nearby points within radius
            #
            #       n(points in radius)                         squared_euclidean_distance
            # GAI =        SUM          weight_of_point * e ^ - _____________________________
            #             i = 1                                 SD_2_SQUARED_SPATIAL_ACCURACY
            #
            # SD_2_SQUARED_SPATIAL_ACCURACY = (2 * HOLOLENS_2_SPATIAL_ERROR) ^ 2
            #
            # Then, get the average intensity and assign to the colors_intensity list
            # After all aggregation are done, normalize the colors_intensity
            # Map to the CMAP color table
            gaussian_adjusted_weights = []
            for point_index in nearby_point_indices:
                # Get the squared euclidean distance,
                # since square root will be canceled by the square operation of gaussian adjusted weight calculation
                difference_vector = vertex - positions[point_index]
                squared_euclidean_distance = np.sum(difference_vector**2)
                gaussian_adjusted_weight = normalized_intensity[
                    point_index] * np.exp(-squared_euclidean_distance /
                                          SD_2_SQUARED_SPATIAL_ACCURACY)
                gaussian_adjusted_weights.append(gaussian_adjusted_weight)
            gaussian_adjusted_weights = np.array(gaussian_adjusted_weights)
            colors_weights[i] = np.average(gaussian_adjusted_weights)
        # else:
        #     colors[i, :] = [0.0, 0.0, 0.0]  # This adds a dark blue background

    colors_max = np.max(colors_weights)
    colors_min = np.min(colors_weights)
    normalized_colors_weights = (colors_weights - colors_min) / (colors_max -
                                                                 colors_min)
    # colors = cmap(normalized_colors_weights)[:, :3]
    for i in range(n_vertices):
        if (normalized_colors_weights[i] == 0):
            colors[i, :] = [0.0, 0.0, 0.0]
        else:
            colors[i, :] = cmap(normalized_colors_weights[i])[:3]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # Write output mesh in .obj format
    o3d.io.write_triangle_mesh(output_file, mesh, write_ascii=True)
    print(f"intensity heatmap mesh saved to: {output_file}")

    # Visualize the mesh
    if visualize:
        visualize_geometry(mesh)

    return mesh
    # yapf: enable


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


def process_questionnaire_answers(qa_input_file, model_file, output_ply_file,
                                  output_lookup_csv,
                                  output_segmented_meshes_dir,
                                  output_combined_mesh_file,
                                  association_radius, search_radius):
    """
    Processes the qa.csv file to extract gazed voxel XYZ and answers.
    Assigns specific colors and color names based on predefined answer choices,
    generates a PLY point cloud with these colors, a lookup CSV including the color name,
    segmented meshes for each answer category, and a single combined mesh colored by answers.

    Args:
        qa_input_file (str): Path to the qa.csv file.
        model_file (str): Path to the base mesh model file (e.g., model.obj).
        output_ply_file (str): Path to save the PLY file for gazed voxels.
        output_lookup_csv (str): Path to save the CSV lookup file.
        output_segmented_meshes_dir (str): Directory to save the segmented mesh files.
        output_combined_mesh_file (str): Path to save the combined mesh.
    """
    answer_color_map = {
        "面白い・気になる形だ": {
            "rgb": [255, 165, 0],
            "name": "Orange"
        },  # Attention to shape
        "美しい・芸術的だ": {
            "rgb": [0, 128, 0],
            "name": "Green"
        },  # Positive aesthetic
        "不思議・意味不明": {
            "rgb": [128, 0, 128],
            "name": "Purple"
        },  # Confusion/Thought
        "不気味・不安・怖い": {
            "rgb": [255, 0, 0],
            "name": "Red"
        },  # Negative feeling
        "何も感じない": {
            "rgb": [255, 255, 0],
            "name": "Yellow"
        },  # No specific reason
    }

    default_color_rgb = [200, 200, 200]
    default_color_name = "Light Grey (Other)"

    try:
        df = pd.read_csv(qa_input_file, sep=',', header=0)

        # Ensure numeric columns are properly typed, coercing errors
        df['estX'] = pd.to_numeric(df['estX'], errors='coerce')
        df['estY'] = pd.to_numeric(df['estY'], errors='coerce')
        df['estZ'] = pd.to_numeric(df['estZ'], errors='coerce')
        df['timestamp'] = pd.to_numeric(
            df['timestamp'], errors='coerce')  # Make sure timestamp is numeric

        df['answer'] = df['answer'].astype(str).str.strip()

        # Drop rows with NaN in essential columns before processing
        df = df.dropna(subset=['estX', 'estY', 'estZ', 'answer', 'timestamp'])

        # # Create a combined 'gazedVoxelID' from estX, estY, estZ and timestamp
        # # This creates a unique identifier for each gaze point that can be associated with an answer
        # df['gazedVoxelID'] = df['estX'].astype(str) + '_' + \
        #                      df['estY'].astype(str) + '_' + \
        #                      df['estZ'].astype(str) + '_' + \
        #                      df['timestamp'].astype(str)

        # Initialize lists for colors and color names
        assigned_colors_255 = []  # Store as 0-255 for Open3D PLY
        assigned_colors_01 = []  # Store as 0-1 for Open3D PLY
        assigned_color_names = []  # Store color names for lookup CSV

        # Assign colors and names based on the answer
        for index, row in df.iterrows():
            answer_text = row['answer']
            if answer_text in answer_color_map:
                color_info = answer_color_map[answer_text]
                assigned_colors_255.append(color_info["rgb"])
                assigned_colors_01.append(np.array(color_info["rgb"]) / 255.0)
                assigned_color_names.append(color_info["name"])
            else:
                # Assign default color if the answer doesn't match a predefined choice
                assigned_colors_255.append(default_color_rgb)
                assigned_colors_01.append(np.array(default_color_rgb) / 255.0)
                assigned_color_names.append(default_color_name)

        df['color_name'] = assigned_color_names  # Add color names to DataFrame for lookup CSV
        df['color_rgb_255'] = assigned_colors_255  # Also keep RGB for completeness in lookup if desired

        # Extract XYZ positions and ensure float64 type for Open3D
        positions = df[['estX', 'estY',
                              'estZ']].values.astype(np.float64)

        # Create an Open3D point cloud for these voxels
        pcd_voxels = o3d.geometry.PointCloud()
        pcd_voxels.points = o3d.utility.Vector3dVector(positions)

        # Assign the calculated colors (0-1 range) to the point cloud
        pcd_voxels.colors = o3d.utility.Vector3dVector(
            np.array(assigned_colors_01))

        # Save the voxel points to a PLY file
        o3d.io.write_point_cloud(output_ply_file, pcd_voxels, write_ascii=True)
        print(f"Gazed voxel point cloud saved to: {output_ply_file}")

        # Save the lookup table (voxel ID, XYZ, answer, and color name) to a CSV
        # df_lookup = df[[
        #     'gazedVoxelID', 'estX', 'estY', 'estZ', 'answer', 'color_name',
        #     'color_rgb_255'
        # ]]
        df_lookup = df[[
            'estX', 'estY', 'estZ', 'answer', 'color_name',
            'color_rgb_255'
        ]]
        df_lookup.to_csv(output_lookup_csv, index=False)
        print(f"Voxel answer lookup table saved to: {output_lookup_csv}")

        # Mesh Processing
        # Load the base mesh for segmentation and combined mesh
        if not os.path.exists(model_file):
            print(
                f"Error: Base model file '{model_file}' not found. Cannot perform mesh operations."
            )
            return pcd_voxels

        base_mesh = o3d.io.read_triangle_mesh(model_file)
        base_vertices = np.asarray(base_mesh.vertices)

        # Create a KDTree for the gaze points for efficient searching
        # gaze_kdtree = KDTree(voxel_positions)
        gaze_kdtree = o3d.geometry.KDTreeFlann(pcd_voxels)

        # Initialize colors for the combined mesh
        # Default to a neutral color (e.g., light gray or black) for parts of the mesh not gazed upon
        combined_mesh_vertex_colors = np.full_like(
            base_vertices, [0.0, 0.0, 0.0])  # Start with black

        # # Define a radius to associate gaze points with mesh vertices
        # # This radius needs to be tuned based on the scale of your model and gaze data
        # association_radius = 25.0

        print(
            "\n=== Generating combined mesh colored by questionnaire answers ==="
        )
        # Iterate through each vertex of the base mesh
        for i, vertex in enumerate(
                tqdm(base_vertices, desc="Coloring combined mesh")):
            # Find all gaze points within the association radius of the current mesh vertex
            [k, nearby_gaze_indices,
             _] = gaze_kdtree.search_radius_vector_3d(vertex,
                                                      association_radius)

            if nearby_gaze_indices:
                # If there are nearby gaze points, determine the dominant color/answer
                # For simplicity, we'll average the colors of nearby gaze points
                # A more complex approach might involve weighting by inverse distance or voting

                # Get the colors (0-1 range) of the nearby gaze points
                nearby_colors_01 = np.array(
                    [assigned_colors_01[j] for j in nearby_gaze_indices])

                # Calculate the average color
                averaged_color = np.mean(nearby_colors_01, axis=0)
                combined_mesh_vertex_colors[i] = averaged_color
            else:
                # If no gaze points are nearby, keep the vertex black (or default color)
                combined_mesh_vertex_colors[i] = [0.0, 0.0, 0.0
                                                  ]  # Black for un-gazed areas

        # Assign the calculated colors to the combined mesh
        base_mesh.vertex_colors = o3d.utility.Vector3dVector(
            combined_mesh_vertex_colors)

        # Save the combined mesh
        o3d.io.write_triangle_mesh(output_combined_mesh_file,
                                   base_mesh,
                                   write_ascii=True)
        print(f"Combined mesh saved to: {output_combined_mesh_file}")

        # Optional: Visualize the combined mesh
        # visualize_geometry(base_mesh) # Uncomment to visualize the combined mesh

        # Segmentation and Individual Segmented Mesh Generation
        # Create output directory for segmented meshes if it doesn't exist
        os.makedirs(output_segmented_meshes_dir, exist_ok=True)

        # Process each unique answer category for individual segmented meshes
        unique_answers = df['answer'].unique()
        for answer_category in unique_answers:
            print(f"Segmenting mesh for answer: '{answer_category}'")
            category_df = df[df['answer'] == answer_category]
            if category_df.empty:
                continue

            # Extract positions for this answer category
            category_gaze_positions = category_df[['estX', 'estY', 'estZ'
                                                   ]].values.astype(np.float64)
            if len(category_gaze_positions) == 0:
                continue

            pcd_category_gaze = o3d.geometry.PointCloud()
            pcd_category_gaze.points = o3d.utility.Vector3dVector(
                category_gaze_positions)

            # Create a KDTree for the gaze points of this category
            category_tree = o3d.geometry.KDTreeFlann(pcd_category_gaze)

            # Determine color for this category
            category_color_info = answer_color_map.get(
                answer_category, {
                    "rgb": default_color_rgb,
                    "name": default_color_name
                })
            category_rgb_01 = np.array(category_color_info["rgb"]) / 255.0

            # Create a copy of the base mesh for this segment
            segmented_mesh = o3d.geometry.TriangleMesh(
                base_mesh)  # Use base_mesh to copy its structure
            segmented_mesh.paint_uniform_color(
                [0.0, 0.0, 0.0])  # Start with black for transparent background

            # Iterate through base mesh vertices to color them based on proximity to gaze points
            mesh_vertex_colors = np.asarray(segmented_mesh.vertex_colors)
            for i, vertex in enumerate(
                    tqdm(base_vertices,
                         desc=f"Coloring mesh for {answer_category}")):
                # # Find if this mesh vertex is close to any gaze point for this answer category
                # search_radius = 25.0
                [k, nearby_gaze_indices,
                 _] = category_tree.search_radius_vector_3d(
                     vertex, search_radius)

                if nearby_gaze_indices:
                    # If nearby gaze points exist for this category, color the vertex with the category's color
                    mesh_vertex_colors[i] = category_rgb_01
                # Else, the vertex remains black (or transparent in visualization)

            segmented_mesh.vertex_colors = o3d.utility.Vector3dVector(
                mesh_vertex_colors)

            # Save the segmented mesh
            output_segmented_mesh_path = os.path.join(
                output_segmented_meshes_dir,
                f"{category_color_info['name'].replace(' ', '_')}_mesh.ply"
            )  # Changed extension to .ply for consistency
            o3d.io.write_triangle_mesh(output_segmented_mesh_path,
                                       segmented_mesh,
                                       write_ascii=True)
            print(
                f"Segmented mesh for '{answer_category}' saved to: {output_segmented_mesh_path}"
            )

        return pcd_voxels

    except FileNotFoundError:
        print(
            f"Error: QA file '{qa_input_file}' not found. Skipping voxel answer processing."
        )
        return None
    except Exception as e:
        print(f"Error processing QA file {qa_input_file}: {e}")
        return None


if __name__ == '__main__':
    curr_dir = pathlib.Path.cwd()

    # Parameters
    point_cloud_ball_radius = 25  # 25 | 0.05 for points in range ~[-200, 400] | ~[-1, 1]
    HOLOLENS_2_SPATIAL_ERROR = 2.66  # Original: 6.45 | Recalibrated: 2.66
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

    import time
    ts = time.time_ns()

    session_paths = curr_dir / pathlib.Path('src/data')
    for sessions in os.listdir(session_paths):
        model_paths = session_paths / pathlib.Path(sessions)

        for models in os.listdir(model_paths):
            datafile_paths = model_paths / pathlib.Path(models)

            input_file = os.path.join(datafile_paths, "pointcloud.csv")
            output_point_cloud = os.path.join(datafile_paths,
                                              "pointcloud_viz.ply")
            model_file = os.path.join(datafile_paths, "model.obj")
            output_mesh = os.path.join(datafile_paths, "heatmap_viz.ply")

            qa_input_file = os.path.join(datafile_paths, "qa.csv")
            output_qa_ply = os.path.join(datafile_paths,
                                         "gazed_voxels_answers.ply")
            output_qa_lookup_csv = os.path.join(
                datafile_paths, "gazed_voxels_answers_lookup.csv")
            output_segmented_meshes_dir = os.path.join(datafile_paths,
                                                       "segmented_meshes")
            output_combined_mesh_file = os.path.join(datafile_paths,
                                                     "combined_qa_mesh.ply")

            if (models not in parameters_dict.keys()):
                # Generate colored point cloud based on intensity
                if generate_point_cloud:
                    print("\n=== Generating intensity-colored point cloud ===")
                    pcd = create_and_visualize_xyz_colored_point_cloud(
                        input_file,
                        output_point_cloud,
                        visualize=viz,
                        ball_radius=point_cloud_ball_radius)

                # Generate heatmap mesh based on point intensity
                if generate_mesh and os.path.exists(model_file):
                    print("\n=== Generating intensity heatmap on mesh ===")
                    heatmap_mesh = create_heatmap_mesh_from_intensity(
                        input_file,
                        model_file,
                        output_mesh,
                        visualize=viz,
                        HOLOLENS_2_SPATIAL_ERROR=HOLOLENS_2_SPATIAL_ERROR,
                        ball_radius=ball_radius)
                elif generate_mesh and not os.path.exists(model_file):
                    print(
                        f"Error: Model file '{model_file}' not found. Skipping heatmap mesh generation."
                    )

                if generate_voxel_answers:
                    print(
                        "\n=== Processing questionnaire answers and generating voxel points ==="
                    )
                    process_questionnaire_answers(
                        qa_input_file, model_file, output_qa_ply,
                        output_qa_lookup_csv, output_segmented_meshes_dir,
                        output_combined_mesh_file, HOLOLENS_2_SPATIAL_ERROR,
                        HOLOLENS_2_SPATIAL_ERROR)

            else:
                # Generate colored point cloud based on intensity
                if generate_point_cloud:
                    print("\n=== Generating intensity-colored point cloud ===")
                    pcd = create_and_visualize_xyz_colored_point_cloud(
                        input_file,
                        output_point_cloud,
                        visualize=viz,
                        ball_radius=parameters_dict[models][0])

                # Generate heatmap mesh based on point intensity
                if generate_mesh and os.path.exists(model_file):
                    print("\n=== Generating intensity heatmap on mesh ===")
                    heatmap_mesh = create_heatmap_mesh_from_intensity(
                        input_file,
                        model_file,
                        output_mesh,
                        visualize=viz,
                        HOLOLENS_2_SPATIAL_ERROR=parameters_dict[models][1],
                        ball_radius=parameters_dict[models][2])
                elif generate_mesh and not os.path.exists(model_file):
                    print(
                        f"Error: Model file '{model_file}' not found. Skipping heatmap mesh generation."
                    )

                if generate_voxel_answers:
                    print(
                        "\n=== Processing questionnaire answers and generating voxel points ==="
                    )
                    process_questionnaire_answers(
                        qa_input_file, model_file, output_qa_ply,
                        output_qa_lookup_csv, output_segmented_meshes_dir,
                        output_combined_mesh_file, parameters_dict[models][1],
                        parameters_dict[models][1])

    te = time.time_ns()

    print(f"TOTAL TIME: {(te - ts)/10e8}")
