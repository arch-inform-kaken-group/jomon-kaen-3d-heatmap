import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LinearSegmentedColormap
import trimesh

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
# Paths to your SMR model and corresponding data
SMR_GLB_PATH = r"archive\smr\SMR.glb"
SMR_CSV_PATH = r"archive\smr\fixation\fixations_agtzidis_holo5_nuc2.csv"

OUTPUT_BASE_DIR = "smr_eeg_plots_output"
EEG_COLUMN = "FAA"
COLORBAR_TICKS = np.arange(-1.0, 1.1, 0.2)

# Custom grayscale colormap starting at ~33% brightness (#555555)
GRAY_SHIFTED_CMAP = LinearSegmentedColormap.from_list('gray_shifted',
                                                      ['#555555',
                                                       '#FFFFFF'])


# ==========================================
# 🛠️ DATA LOADING UTILITY
# ==========================================
def load_artefact_data(csv_path, glb_path):
    """Loads CSV and GLB once to speed up multiple plotting calls."""
    df = pd.read_csv(csv_path)

    if EEG_COLUMN not in df.columns:
        df[EEG_COLUMN] = 0.0

    scene = trimesh.load(glb_path, force='mesh')
    if isinstance(scene, trimesh.Scene):
        meshes = [
            g for g in scene.geometry.values()
            if isinstance(g, trimesh.Trimesh)
        ]
        tm_mesh = meshes[0]
    else:
        tm_mesh = scene

    # Map Y-up to Z-up: Plot_X = Data_X, Plot_Y = Data_Z, Plot_Z = Data_Y
    plot_vertices = np.column_stack((tm_mesh.vertices[:, 0],
                                     tm_mesh.vertices[:, 2],
                                     tm_mesh.vertices[:, 1]))
    return df, tm_mesh, plot_vertices


# ==========================================
# 🛠️ CORE PLOTTING FUNCTION
# ==========================================
def draw_3d_artefact(ax,
                     df,
                     tm_mesh,
                     plot_vertices,
                     title,
                     elev,
                     azim,
                     color_mode='color'):
    """Draws the mesh and scatter points on a given matplotlib 3D axis."""

    # Select Color Palette & Colormap (Metallic/Steel look for Reactor)
    if color_mode == 'color':
        face_color = '#B0C4DE'  # Light steel blue
        edge_color = '#708090'  # Slate gray
        cmap = 'jet'
    else:
        face_color = '#D3D3D3'
        edge_color = '#999999'
        cmap = GRAY_SHIFTED_CMAP

    mesh_collection = Poly3DCollection(plot_vertices[tm_mesh.faces],
                                       alpha=0.15,
                                       facecolor=face_color,
                                       edgecolor=edge_color,
                                       linewidth=0.35,
                                       zorder=1)
    ax.add_collection3d(mesh_collection)

    df_clean = df.dropna(
        subset=['centroid_x',
                'centroid_y',
                'centroid_z',
                EEG_COLUMN])
    plot_cx = df_clean['centroid_x'].to_numpy()
    plot_cy = df_clean['centroid_z'].to_numpy()
    plot_cz = df_clean['centroid_y'].to_numpy()

    # White halo layer to prevent color mixing with the mesh transparency
    ax.scatter(plot_cx,
               plot_cy,
               plot_cz,
               c='white',
               s=110,
               edgecolors='none',
               depthshade=False,
               alpha=1.0,
               zorder=10)

    # Actual data scatter points
    scatter = ax.scatter(plot_cx,
                         plot_cy,
                         plot_cz,
                         c=df_clean[EEG_COLUMN],
                         cmap=cmap,
                         vmin=-1.0,
                         vmax=1.0,
                         s=65,
                         edgecolors='black',
                         linewidth=1.2,
                         depthshade=False,
                         alpha=1.0,
                         zorder=11)

    ax.set_title(title, pad=8, fontsize=16, fontweight='bold')

    # Remove axis grid, ticks, and panes
    ax.set_axis_off()

    # Maintain 1:1:1 aspect ratio bounding box based precisely on the data
    all_x = np.concatenate([plot_vertices[:, 0], plot_cx])
    all_y = np.concatenate([plot_vertices[:, 1], plot_cy])
    all_z = np.concatenate([plot_vertices[:, 2], plot_cz])

    x_range = np.nanmax(all_x) - np.nanmin(all_x)
    y_range = np.nanmax(all_y) - np.nanmin(all_y)
    z_range = np.nanmax(all_z) - np.nanmin(all_z)
    max_range = np.max([x_range, y_range, z_range]) / 2.0

    mid_x = (np.nanmax(all_x) + np.nanmin(all_x)) * 0.5
    mid_y = (np.nanmax(all_y) + np.nanmin(all_y)) * 0.5
    mid_z = (np.nanmax(all_z) + np.nanmin(all_z)) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Force max size/zoom to fill the whitespace left by the removed axes
    try:
        ax.set_box_aspect([1, 1, 1], zoom=1.35)
    except TypeError:
        # Fallback for older Matplotlib versions
        ax.set_box_aspect([1, 1, 1])
        ax.dist = 7

    ax.view_init(elev=elev, azim=azim)
    return scatter


def plot_single_view(df,
                     mesh,
                     verts,
                     title,
                     base_filepath,
                     elev,
                     azim,
                     color_mode='color'):
    """Generates and saves a single full-sized plot for a specific view."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    sc = draw_3d_artefact(ax,
                          df,
                          mesh,
                          verts,
                          title,
                          elev,
                          azim,
                          color_mode=color_mode)

    # Add a dedicated colorbar per image
    fig.colorbar(sc,
                 ax=ax,
                 shrink=0.6,
                 pad=0.02,
                 label='FAA',
                 ticks=COLORBAR_TICKS)

    # Use zero padding to further enlarge the model rendering inside the figure
    plt.tight_layout(pad=0)
    fig.savefig(base_filepath, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Saved '{base_filepath}'")


# ==========================================
# 🚀 MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("Setting up output directory...")
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    print("Loading SMR dataset and mesh (this may take a moment)...")
    smr_df, smr_mesh, smr_verts = load_artefact_data(SMR_CSV_PATH, SMR_GLB_PATH)
    print("Data loaded successfully.\n")

    # Defined Camera Angles for individual saving
    orbit_low = [("Front (Azim: -90°)", 15, -90, "front"),
                 ("Right (Azim: 0°)", 15, 0, "right"),
                 ("Back (Azim: 90°)", 15, 90, "back"),
                 ("Left (Azim: 180°)", 15, 180, "left")]

    orbit_high = [("Front High (-90° Azim)", 30, -90, "front"),
                  ("Right High (0° Azim)", 30, 0, "right"),
                  ("Back High (90° Azim)", 30, 90, "back"),
                  ("Left High (180° Azim)", 30, 180, "left")]

    elev_sweep = [("Eye-Level View (0° Elev)", 0, -60, "0deg"),
                  ("Slight High Angle (30° Elev)", 30, -60, "30deg"),
                  ("High Angle (60° Elev)", 60, -60, "60deg"),
                  ("Top-Down View (90° Elev)", 90, -60, "90deg")]

    categories = [
        ("orbit_low", orbit_low),
        ("orbit_high", orbit_high),
        ("elev_sweep", elev_sweep)
    ]

    # Generate plots in both Color and Grayscale modes
    for mode in ['color', 'gray']:
        mode_str = "Colored" if mode == 'color' else "Grayscale"
        print(f"--- Generating {mode_str} Single-View Plots ---")

        for category_name, views in categories:
            for view_title, elev, azim, suffix in views:
                title = f"FAA points on SMR\nSMR Model Credit: EnergyEncyclopedia.com"
                filename = f"smr_{category_name}_{suffix}_{mode}.png"
                filepath = os.path.join(OUTPUT_BASE_DIR, filename)

                plot_single_view(smr_df,
                                 smr_mesh,
                                 smr_verts,
                                 title,
                                 filepath,
                                 elev,
                                 azim,
                                 color_mode=mode)

    print("\nAll full-sized borderless plots generated successfully!")