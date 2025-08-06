import numpy as np  #
import pandas as pd  #
import open3d as o3d  #

import matplotlib  #

matplotlib.use('Agg')
import matplotlib.pyplot as plt


# yapf: disable
def analyze_and_plot_point_cloud(csv_file_path):
    """
    Reads a CSV file with 3D point data, counts occurrences.
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

    ax.set_title('3D Distribution of Gaze Points (Colored by Occurrence Count)')
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
        ax.set_ylim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)  # Y-axis of plot gets Z-data limits
        ax.set_zlim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)  # Z-axis of plot gets Y-data limits
    except ValueError:
        print("Could not determine axis range automatically.")
    ax.grid(True)

    return fig


def generate_original_pointcloud(input_file: str,
                                 jitter_std_dev: float = 0.015):
    """
    Loads 3D gaze points from a CSV and adds a small amount of Gaussian jitter.

    This jitter helps visualize the density of points that are in the exact
    same location by slightly offsetting them, preventing them from perfectly
    overlapping.

    Args:
        input_file (str): Path to the CSV file with gaze data. Expected format: [gaze_x, gaze_y, gaze_z, ...].
        jitter_std_dev (float): The standard deviation (in meters) of the
                                Gaussian noise to add to each point's
                                coordinates. A larger value creates more spread.
                                Default: 0.015.

    Returns:
        o3d.geometry.PointCloud: An Open3D PointCloud object with the jittered points.
    """
    try:
        # Read the first 3 columns (X, Y, Z) from the CSV
        data = pd.read_csv(input_file, header=0, usecols=[0, 1, 2]).to_numpy()
    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as e:
        print(
            f"Warning: Could not read or parse {input_file}: {e}. Returning empty point cloud."
        )
        return o3d.geometry.PointCloud()

    gaze_points_np = data

    # If there's no data after loading, return an empty point cloud
    if gaze_points_np.shape[0] == 0:
        return o3d.geometry.PointCloud()

    # Generate Gaussian noise with the same shape as the data
    # loc=0.0 means the noise is centered around zero (no systematic drift)
    # scale=jitter_std_dev controls the amount of spread
    noise = np.random.normal(loc=0.0,
                             scale=jitter_std_dev,
                             size=gaze_points_np.shape)

    # Add the noise to the original points to create the jittered effect
    jittered_points_np = gaze_points_np + noise

    # Create and return the Open3D PointCloud from the new jittered data
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(jittered_points_np))
# yapf: enable
