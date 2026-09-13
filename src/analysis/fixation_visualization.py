# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import trimesh

# # ==========================================
# # ⚙️ CONFIGURATION
# # ==========================================
# # Paths to your 3D models
# POTTERY_GLB_PATH = r"D:\storage\jomon_kaen\pottery\IN0009(5).glb"
# FIGURINE_GLB_PATH = r"D:\storage\jomon_kaen\pottery\UD0028(93).glb"

# # First-time subject CSVs
# POTTERY_FIRST_CSV = r"C:\Users\User\Desktop\Grant\MREEG\Trail_Experiment\2026_08_07_09_43_48\fixations_agtzidis_pot.csv"
# FIGURINE_FIRST_CSV = r"C:\Users\User\Desktop\Grant\MREEG\Trail_Experiment\2026_08_07_09_43_48\fixations_agtzidis_fig.csv"

# # Familiar subject CSVs
# POTTERY_FAMILIAR_CSV = r"C:\Users\User\Desktop\Grant\MREEG\Trail_Experiment\2026_08_04_09_27_45\eeg_fixations_agtzidis_pottery.csv"
# FIGURINE_FAMILIAR_CSV = r"C:\Users\User\Desktop\Grant\MREEG\Trail_Experiment\2026_08_04_09_27_45\eeg_fixations_agtzidis_figurine.csv"

# # Output folder structure
# OUTPUT_BASE_DIR = "eeg_plots_output"

# EEG_COLUMN = "FAA"
# COLORBAR_TICKS = np.arange(-1.0, 1.1, 0.2)


# # ==========================================
# # 🛠️ DATA LOADING UTILITY
# # ==========================================
# def load_artefact_data(csv_path, glb_path):
#     """Loads CSV and GLB once to speed up multiple plotting calls."""
#     df = pd.read_csv(csv_path)
    
#     # Safeguard: Add default FAA column if missing (e.g., in the pottery CSV)
#     if EEG_COLUMN not in df.columns:
#         df[EEG_COLUMN] = 0.0

#     scene = trimesh.load(glb_path, force='mesh')
#     if isinstance(scene, trimesh.Scene):
#         meshes = [
#             g for g in scene.geometry.values()
#             if isinstance(g, trimesh.Trimesh)
#         ]
#         tm_mesh = meshes[0]
#     else:
#         tm_mesh = scene

#     # Map Y-up to Z-up: Plot_X = Data_X, Plot_Y = Data_Z, Plot_Z = Data_Y
#     plot_vertices = np.column_stack((tm_mesh.vertices[:, 0],
#                                      tm_mesh.vertices[:, 2],
#                                      tm_mesh.vertices[:, 1]))

#     return df, tm_mesh, plot_vertices


# # ==========================================
# # 🛠️ CORE PLOTTING FUNCTION
# # ==========================================
# def draw_3d_artefact(ax, df, tm_mesh, plot_vertices, title, elev, azim):
#     """Draws the mesh and scatter points on a given matplotlib 3D axis."""

#     # 1. Plot Mesh (Earthware Pottery Color)
#     # Matplotlib cannot render .glb textures natively. We use a realistic earthware/clay 
#     # hex color (#C19A6B) with subtle sienna edges (#8B4513) for a 3D ceramic feel.
#     mesh_collection = Poly3DCollection(plot_vertices[tm_mesh.faces],
#                                        alpha=0.65,
#                                        facecolor='#C19A6B', 
#                                        edgecolor='#8B4513', 
#                                        linewidth=0.15,
#                                        zorder=1)
#     ax.add_collection3d(mesh_collection)

#     # Clean data to avoid NaN/Inf errors during limit calculation
#     df_clean = df.dropna(
#         subset=['centroid_x',
#                 'centroid_y',
#                 'centroid_z',
#                 EEG_COLUMN])

#     # 2. Plot Gaze Points (Centroids only, NO paths)
#     plot_cx = df_clean['centroid_x'].to_numpy()
#     plot_cy = df_clean['centroid_z'].to_numpy()
#     plot_cz = df_clean['centroid_y'].to_numpy()

#     # --- FIX FOR COLOR MIXING ---
#     # Step A: Draw a solid white "halo" behind the points. 
#     # This creates an opaque barrier that stops the earthware clay color from bleeding 
#     # into the 'jet' colormap, ensuring your FAA data remains 100% pure and vibrant.
#     ax.scatter(plot_cx, plot_cy, plot_cz,
#                c='white', s=110, edgecolors='none',
#                depthshade=False, alpha=1.0, zorder=10)

#     # Step B: Draw the actual colored points on top of the white halo
#     scatter = ax.scatter(plot_cx, plot_cy, plot_cz,
#                          c=df_clean[EEG_COLUMN], cmap='jet', vmin=-1.0, vmax=1.0,
#                          s=65, edgecolors='black', linewidth=1.2,
#                          depthshade=False, alpha=1.0, zorder=11)

#     # 3. Formatting
#     ax.set_title(title, pad=10, fontsize=12, fontweight='bold')
#     ax.set_xlabel('Gaze-X')
#     ax.set_ylabel('Gaze-Z (Original)')
#     ax.set_zlabel('Gaze-Y (Original)')

#     # 4. Bounding Box (Force Equal Aspect Ratio to avoid stretching)
#     all_x = np.concatenate([plot_vertices[:, 0], plot_cx])
#     all_y = np.concatenate([plot_vertices[:, 1], plot_cy])
#     all_z = np.concatenate([plot_vertices[:, 2], plot_cz])

#     max_range = np.array([
#         np.nanmax(all_x) - np.nanmin(all_x),
#         np.nanmax(all_y) - np.nanmin(all_y),
#         np.nanmax(all_z) - np.nanmin(all_z)
#     ]).max() / 2.0

#     mid_x = (np.nanmax(all_x) + np.nanmin(all_x)) * 0.5
#     mid_y = (np.nanmax(all_y) + np.nanmin(all_y)) * 0.5
#     mid_z = (np.nanmax(all_z) + np.nanmin(all_z)) * 0.5

#     ax.set_xlim(mid_x - max_range, mid_x + max_range)
#     ax.set_ylim(mid_y - max_range, mid_y + max_range)
#     ax.set_zlim(mid_z - max_range, mid_z + max_range)

#     try:
#         ax.set_box_aspect([1, 1, 1])
#     except AttributeError:
#         pass

#     # 5. Set camera angle
#     ax.view_init(elev=elev, azim=azim)
#     return scatter


# def plot_4_views_grid(df, mesh, verts, main_title, filepath, view_configs):
#     """Helper function to generate a 2x2 grid based on a list of view configurations."""
#     fig = plt.figure(figsize=(12, 10))
#     fig.suptitle(main_title, fontsize=14, fontweight='bold')

#     for i, (title, elev, azim) in enumerate(view_configs):
#         ax = fig.add_subplot(2, 2, i + 1, projection='3d')
#         sc = draw_3d_artefact(ax, df, mesh, verts, title, elev, azim)
#         if i == 1:
#             fig.colorbar(sc,
#                          ax=ax,
#                          shrink=0.5,
#                          pad=0.1,
#                          label='FAA',
#                          ticks=COLORBAR_TICKS)

#     plt.tight_layout()
#     fig.savefig(filepath, dpi=300, bbox_inches='tight')
#     plt.close(fig)
#     print(f"Saved '{filepath}'")


# # ==========================================
# # 🚀 MAIN EXECUTION
# # ==========================================
# if __name__ == "__main__":
#     print("Setting up output directories...")
#     first_time_dir = os.path.join(OUTPUT_BASE_DIR, "first_time_subject")
#     familiar_dir = os.path.join(OUTPUT_BASE_DIR, "familiar_subject")

#     os.makedirs(first_time_dir, exist_ok=True)
#     os.makedirs(familiar_dir, exist_ok=True)

#     print("Loading datasets and meshes (this may take a moment)...")
#     pot_first_df, pot_mesh, pot_verts = load_artefact_data(POTTERY_FIRST_CSV, POTTERY_GLB_PATH)
#     pot_fam_df, _, _ = load_artefact_data(POTTERY_FAMILIAR_CSV, POTTERY_GLB_PATH)

#     fig_first_df, fig_mesh, fig_verts = load_artefact_data(FIGURINE_FIRST_CSV, FIGURINE_GLB_PATH)
#     fig_fam_df, _, _ = load_artefact_data(FIGURINE_FAMILIAR_CSV, FIGURINE_GLB_PATH)
#     print("Data loaded successfully.\n")

#     # ---------------------------------------------------------
#     # CAMERA VIEWS 
#     # ---------------------------------------------------------
#     orbit_low = [("Front (Azim: -90°)", 15, -90),
#                  ("Right (Azim: 0°)", 15, 0),
#                  ("Back (Azim: 90°)", 15, 90),
#                  ("Left (Azim: 180°)", 15, 180)]

#     orbit_high = [("Front High (30° Elev, -90° Azim)", 30, -90),
#                   ("Right High (30° Elev, 0° Azim)", 30, 0),
#                   ("Back High (30° Elev, 90° Azim)", 30, 90),
#                   ("Left High (30° Elev, 180° Azim)", 30, 180)]

#     elev_sweep = [("Eye-Level View (0° Elev)", 0, -60),
#                   ("Slight High Angle (30° Elev)", 30, -60),
#                   ("High Angle (60° Elev)", 60, -60),
#                   ("Top-Down View (90° Elev)", 90, -60)]

#     # ---------------------------------------------------------
#     # 1. SAVE FIRST-TIME SUBJECT PLOTS
#     # ---------------------------------------------------------
#     print("Generating First-Time Subject grids...")
#     plot_4_views_grid(pot_first_df, pot_mesh, pot_verts, "Pottery (First-Time) - Low Orbit (15°)", os.path.join(first_time_dir, "pottery_orbit_low.png"), orbit_low)
#     plot_4_views_grid(pot_first_df, pot_mesh, pot_verts, "Pottery (First-Time) - High Orbit (30°)", os.path.join(first_time_dir, "pottery_orbit_high.png"), orbit_high)
#     plot_4_views_grid(pot_first_df, pot_mesh, pot_verts, "Pottery (First-Time) - Elevation Sweep", os.path.join(first_time_dir, "pottery_elev_sweep.png"), elev_sweep)

#     plot_4_views_grid(fig_first_df, fig_mesh, fig_verts, "Figurine (First-Time) - Low Orbit (15°)", os.path.join(first_time_dir, "figurine_orbit_low.png"), orbit_low)
#     plot_4_views_grid(fig_first_df, fig_mesh, fig_verts, "Figurine (First-Time) - High Orbit (30°)", os.path.join(first_time_dir, "figurine_orbit_high.png"), orbit_high)
#     plot_4_views_grid(fig_first_df, fig_mesh, fig_verts, "Figurine (First-Time) - Elevation Sweep", os.path.join(first_time_dir, "figurine_elev_sweep.png"), elev_sweep)

#     # ---------------------------------------------------------
#     # 2. SAVE FAMILIAR SUBJECT PLOTS
#     # ---------------------------------------------------------
#     print("\nGenerating Familiar Subject grids...")
#     plot_4_views_grid(pot_fam_df, pot_mesh, pot_verts, "Pottery (Familiar) - Low Orbit (15°)", os.path.join(familiar_dir, "pottery_orbit_low.png"), orbit_low)
#     plot_4_views_grid(pot_fam_df, pot_mesh, pot_verts, "Pottery (Familiar) - High Orbit (30°)", os.path.join(familiar_dir, "pottery_orbit_high.png"), orbit_high)
#     plot_4_views_grid(pot_fam_df, pot_mesh, pot_verts, "Pottery (Familiar) - Elevation Sweep", os.path.join(familiar_dir, "pottery_elev_sweep.png"), elev_sweep)

#     plot_4_views_grid(fig_fam_df, fig_mesh, fig_verts, "Figurine (Familiar) - Low Orbit (15°)", os.path.join(familiar_dir, "figurine_orbit_low.png"), orbit_low)
#     plot_4_views_grid(fig_fam_df, fig_mesh, fig_verts, "Figurine (Familiar) - High Orbit (30°)", os.path.join(familiar_dir, "figurine_orbit_high.png"), orbit_high)
#     plot_4_views_grid(fig_fam_df, fig_mesh, fig_verts, "Figurine (Familiar) - Elevation Sweep", os.path.join(familiar_dir, "figurine_elev_sweep.png"), elev_sweep)

#     # ---------------------------------------------------------
#     # 3. RESTORE COMPARISON PLOT (2x2 Grid)
#     # ---------------------------------------------------------
#     print("\nGenerating 2x2 Subject Comparison Plot...")
#     fig4 = plt.figure(figsize=(16, 12))

#     ax_tl = fig4.add_subplot(2, 2, 1, projection='3d')
#     sc_tl = draw_3d_artefact(ax_tl, pot_first_df, pot_mesh, pot_verts, "Pottery: First-time subject", elev=15, azim=-60)
#     fig4.colorbar(sc_tl, ax=ax_tl, shrink=0.6, pad=0.1, label='FAA', ticks=COLORBAR_TICKS)

#     ax_tr = fig4.add_subplot(2, 2, 2, projection='3d')
#     sc_tr = draw_3d_artefact(ax_tr, pot_fam_df, pot_mesh, pot_verts, "Pottery: Familiar subject", elev=15, azim=-60)
#     fig4.colorbar(sc_tr, ax=ax_tr, shrink=0.6, pad=0.1, label='FAA', ticks=COLORBAR_TICKS)

#     ax_bl = fig4.add_subplot(2, 2, 3, projection='3d')
#     sc_bl = draw_3d_artefact(ax_bl, fig_first_df, fig_mesh, fig_verts, "Figurine: First-time subject", elev=15, azim=-60)
#     fig4.colorbar(sc_bl, ax=ax_bl, shrink=0.6, pad=0.1, label='FAA', ticks=COLORBAR_TICKS)

#     ax_br = fig4.add_subplot(2, 2, 4, projection='3d')
#     sc_br = draw_3d_artefact(ax_br, fig_fam_df, fig_mesh, fig_verts, "Figurine: Familiar subject", elev=15, azim=-60)
#     fig4.colorbar(sc_br, ax=ax_br, shrink=0.6, pad=0.1, label='FAA', ticks=COLORBAR_TICKS)

#     plt.subplots_adjust(wspace=0.05, hspace=0.1)

#     comparison_output = os.path.join(OUTPUT_BASE_DIR, "subject_comparison_2x2.png")
#     fig4.savefig(comparison_output, dpi=300, bbox_inches='tight')
#     plt.close(fig4)
#     print(f"Saved '{comparison_output}'")

#     print("\nAll plots generated successfully!")


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LinearSegmentedColormap
import trimesh

# ==========================================
# ️ CONFIGURATION
# ==========================================
POTTERY_GLB_PATH = r"D:\storage\jomon_kaen\pottery\IN0009(5).glb"
FIGURINE_GLB_PATH = r"D:\storage\jomon_kaen\pottery\UD0028(93).glb"

POTTERY_FIRST_CSV = r"C:\Users\User\Desktop\Grant\MREEG\Trail_Experiment\archive\2026_08_07_09_43_48\fixations_agtzidis_pot.csv"
FIGURINE_FIRST_CSV = r"C:\Users\User\Desktop\Grant\MREEG\Trail_Experiment\archive\2026_08_07_09_43_48\fixations_agtzidis_fig.csv"

POTTERY_FAMILIAR_CSV = r"C:\Users\User\Desktop\Grant\MREEG\Trail_Experiment\archive\2026_08_04_09_27_45\eeg_fixations_agtzidis_pottery.csv"
FIGURINE_FAMILIAR_CSV = r"C:\Users\User\Desktop\Grant\MREEG\Trail_Experiment\archive\2026_08_04_09_27_45\eeg_fixations_agtzidis_figurine.csv"

OUTPUT_BASE_DIR = "eeg_plots_output"
EEG_COLUMN = "FAA"
COLORBAR_TICKS = np.arange(-1.0, 1.1, 0.2)

# Create a custom grayscale colormap that starts at ~33% brightness (#555555) 
# instead of 0% (black). This ensures value 0.4 maps to ~80% brightness.
GRAY_SHIFTED_CMAP = LinearSegmentedColormap.from_list('gray_shifted', ['#555555', '#FFFFFF'])

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
        meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        tm_mesh = meshes[0]
    else:
        tm_mesh = scene
    plot_vertices = np.column_stack((tm_mesh.vertices[:, 0],
                                     tm_mesh.vertices[:, 2],
                                     tm_mesh.vertices[:, 1]))
    return df, tm_mesh, plot_vertices

# ==========================================
# ️ CORE PLOTTING FUNCTION
# ==========================================
def draw_3d_artefact(ax, df, tm_mesh, plot_vertices, title, elev, azim, color_mode='color'):
    """Draws the mesh and scatter points on a given matplotlib 3D axis."""
    # Select Color Palette & Colormap
    if color_mode == 'color':
        face_color = '#C19A6B'
        edge_color = '#A0826D'
        cmap = 'jet'
    else:
        face_color = '#D3D3D3'
        edge_color = '#999999'
        cmap = GRAY_SHIFTED_CMAP  # <-- Use the custom shifted gray colormap

    mesh_collection = Poly3DCollection(plot_vertices[tm_mesh.faces],
                                       alpha=0.12,
                                       facecolor=face_color,
                                       edgecolor=edge_color,
                                       linewidth=0.3,
                                       zorder=1)
    ax.add_collection3d(mesh_collection)

    df_clean = df.dropna(subset=['centroid_x', 'centroid_y', 'centroid_z', EEG_COLUMN])
    plot_cx = df_clean['centroid_x'].to_numpy()
    plot_cy = df_clean['centroid_z'].to_numpy()
    plot_cz = df_clean['centroid_y'].to_numpy()

    # White halo
    ax.scatter(plot_cx, plot_cy, plot_cz,
               c='white', s=110, edgecolors='none',
               depthshade=False, alpha=1.0, zorder=10)

    # Scatter with standard vmin/vmax (-1 to 1), but the custom colormap handles the brightness shift
    scatter = ax.scatter(plot_cx, plot_cy, plot_cz,
                         c=df_clean[EEG_COLUMN],
                         cmap=cmap,
                         vmin=-1.0, vmax=1.0,  # Keep standard range for even tick spacing
                         s=65, edgecolors='black', linewidth=1.2,
                         depthshade=False, alpha=1.0, zorder=11)

    ax.set_title(title, pad=10, fontsize=12, fontweight='bold')
    ax.set_xlabel('Gaze-X')
    ax.set_ylabel('Gaze-Z (Original)')
    ax.set_zlabel('Gaze-Y (Original)')

    all_x = np.concatenate([plot_vertices[:, 0], plot_cx])
    all_y = np.concatenate([plot_vertices[:, 1], plot_cy])
    all_z = np.concatenate([plot_vertices[:, 2], plot_cz])
    max_range = np.array([
        np.nanmax(all_x) - np.nanmin(all_x),
        np.nanmax(all_y) - np.nanmin(all_y),
        np.nanmax(all_z) - np.nanmin(all_z)
    ]).max() / 2.0
    mid_x = (np.nanmax(all_x) + np.nanmin(all_x)) * 0.5
    mid_y = (np.nanmax(all_y) + np.nanmin(all_y)) * 0.5
    mid_z = (np.nanmax(all_z) + np.nanmin(all_z)) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass
    ax.view_init(elev=elev, azim=azim)
    return scatter

def plot_4_views_grid(df, mesh, verts, main_title, base_filepath, view_configs, color_mode='color'):
    if color_mode == 'gray':
        name, ext = os.path.splitext(base_filepath)
        filepath = f"{name}_gray{ext}"
    else:
        filepath = base_filepath

    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(main_title, fontsize=14, fontweight='bold')
    for i, (title, elev, azim) in enumerate(view_configs):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        sc = draw_3d_artefact(ax, df, mesh, verts, title, elev, azim, color_mode=color_mode)
        if i == 1:
            fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.1, label='FAA', ticks=COLORBAR_TICKS)
    plt.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved '{filepath}'")

def plot_2x2_comparison(pot_first_df, pot_fam_df, fig_first_df, fig_fam_df,
                        pot_mesh, pot_verts, fig_mesh, fig_verts, base_filepath, color_mode='color'):
    if color_mode == 'gray':
        name, ext = os.path.splitext(base_filepath)
        filepath = f"{name}_gray{ext}"
    else:
        filepath = base_filepath

    fig4 = plt.figure(figsize=(16, 12))

    ax_tl = fig4.add_subplot(2, 2, 1, projection='3d')
    sc_tl = draw_3d_artefact(ax_tl, pot_first_df, pot_mesh, pot_verts,
                             "Pottery: First-time subject", elev=15, azim=-60, color_mode=color_mode)
    fig4.colorbar(sc_tl, ax=ax_tl, shrink=0.6, pad=0.1, label='FAA', ticks=COLORBAR_TICKS)

    ax_tr = fig4.add_subplot(2, 2, 2, projection='3d')
    sc_tr = draw_3d_artefact(ax_tr, pot_fam_df, pot_mesh, pot_verts,
                             "Pottery: Familiar subject", elev=15, azim=-60, color_mode=color_mode)
    fig4.colorbar(sc_tr, ax=ax_tr, shrink=0.6, pad=0.1, label='FAA', ticks=COLORBAR_TICKS)

    ax_bl = fig4.add_subplot(2, 2, 3, projection='3d')
    sc_bl = draw_3d_artefact(ax_bl, fig_first_df, fig_mesh, fig_verts,
                             "Figurine: First-time subject", elev=15, azim=-60, color_mode=color_mode)
    fig4.colorbar(sc_bl, ax=ax_bl, shrink=0.6, pad=0.1, label='FAA', ticks=COLORBAR_TICKS)

    ax_br = fig4.add_subplot(2, 2, 4, projection='3d')
    sc_br = draw_3d_artefact(ax_br, fig_fam_df, fig_mesh, fig_verts,
                             "Figurine: Familiar subject", elev=15, azim=-60, color_mode=color_mode)
    fig4.colorbar(sc_br, ax=ax_br, shrink=0.6, pad=0.1, label='FAA', ticks=COLORBAR_TICKS)

    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    fig4.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print(f"Saved '{filepath}'")

# ==========================================
# 🚀 MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("Setting up output directories...")
    first_time_dir = os.path.join(OUTPUT_BASE_DIR, "first_time_subject")
    familiar_dir = os.path.join(OUTPUT_BASE_DIR, "familiar_subject")
    os.makedirs(first_time_dir, exist_ok=True)
    os.makedirs(familiar_dir, exist_ok=True)

    print("Loading datasets and meshes (this may take a moment)...")
    pot_first_df, pot_mesh, pot_verts = load_artefact_data(POTTERY_FIRST_CSV, POTTERY_GLB_PATH)
    pot_fam_df, _, _ = load_artefact_data(POTTERY_FAMILIAR_CSV, POTTERY_GLB_PATH)
    fig_first_df, fig_mesh, fig_verts = load_artefact_data(FIGURINE_FIRST_CSV, FIGURINE_GLB_PATH)
    fig_fam_df, _, _ = load_artefact_data(FIGURINE_FAMILIAR_CSV, FIGURINE_GLB_PATH)
    print("Data loaded successfully.\n")

    orbit_low = [("Front (Azim: -90°)", 15, -90), ("Right (Azim: 0°)", 15, 0),
                 ("Back (Azim: 90°)", 15, 90), ("Left (Azim: 180°)", 15, 180)]
    orbit_high = [("Front High (30° Elev, -90° Azim)", 30, -90), ("Right High (30° Elev, 0° Azim)", 30, 0),
                  ("Back High (30° Elev, 90° Azim)", 30, 90), ("Left High (30° Elev, 180° Azim)", 30, 180)]
    elev_sweep = [("Eye-Level View (0° Elev)", 0, -60), ("Slight High Angle (30° Elev)", 30, -60),
                  ("High Angle (60° Elev)", 60, -60), ("Top-Down View (90° Elev)", 90, -60)]

    for mode in ['color', 'gray']:
        mode_str = "Colored (Earthware)" if mode == 'color' else "Grayscale"
        print(f"\n--- Generating {mode_str} Plots ---")

        # First-Time Subject
        plot_4_views_grid(pot_first_df, pot_mesh, pot_verts,
                          f"Pottery (First-Time) - Low Orbit (15°) [{mode_str}]",
                          os.path.join(first_time_dir, "pottery_orbit_low.png"), orbit_low, color_mode=mode)
        plot_4_views_grid(pot_first_df, pot_mesh, pot_verts,
                          f"Pottery (First-Time) - High Orbit (30°) [{mode_str}]",
                          os.path.join(first_time_dir, "pottery_orbit_high.png"), orbit_high, color_mode=mode)
        plot_4_views_grid(pot_first_df, pot_mesh, pot_verts,
                          f"Pottery (First-Time) - Elevation Sweep [{mode_str}]",
                          os.path.join(first_time_dir, "pottery_elev_sweep.png"), elev_sweep, color_mode=mode)
        plot_4_views_grid(fig_first_df, fig_mesh, fig_verts,
                          f"Figurine (First-Time) - Low Orbit (15°) [{mode_str}]",
                          os.path.join(first_time_dir, "figurine_orbit_low.png"), orbit_low, color_mode=mode)
        plot_4_views_grid(fig_first_df, fig_mesh, fig_verts,
                          f"Figurine (First-Time) - High Orbit (30°) [{mode_str}]",
                          os.path.join(first_time_dir, "figurine_orbit_high.png"), orbit_high, color_mode=mode)
        plot_4_views_grid(fig_first_df, fig_mesh, fig_verts,
                          f"Figurine (First-Time) - Elevation Sweep [{mode_str}]",
                          os.path.join(first_time_dir, "figurine_elev_sweep.png"), elev_sweep, color_mode=mode)

        # Familiar Subject
        plot_4_views_grid(pot_fam_df, pot_mesh, pot_verts,
                          f"Pottery (Familiar) - Low Orbit (15°) [{mode_str}]",
                          os.path.join(familiar_dir, "pottery_orbit_low.png"), orbit_low, color_mode=mode)
        plot_4_views_grid(pot_fam_df, pot_mesh, pot_verts,
                          f"Pottery (Familiar) - High Orbit (30°) [{mode_str}]",
                          os.path.join(familiar_dir, "pottery_orbit_high.png"), orbit_high, color_mode=mode)
        plot_4_views_grid(pot_fam_df, pot_mesh, pot_verts,
                          f"Pottery (Familiar) - Elevation Sweep [{mode_str}]",
                          os.path.join(familiar_dir, "pottery_elev_sweep.png"), elev_sweep, color_mode=mode)
        plot_4_views_grid(fig_fam_df, fig_mesh, fig_verts,
                          f"Figurine (Familiar) - Low Orbit (15°) [{mode_str}]",
                          os.path.join(familiar_dir, "figurine_orbit_low.png"), orbit_low, color_mode=mode)
        plot_4_views_grid(fig_fam_df, fig_mesh, fig_verts,
                          f"Figurine (Familiar) - High Orbit (30°) [{mode_str}]",
                          os.path.join(familiar_dir, "figurine_orbit_high.png"), orbit_high, color_mode=mode)
        plot_4_views_grid(fig_fam_df, fig_mesh, fig_verts,
                          f"Figurine (Familiar) - Elevation Sweep [{mode_str}]",
                          os.path.join(familiar_dir, "figurine_elev_sweep.png"), elev_sweep, color_mode=mode)

        # 2x2 Comparison
        plot_2x2_comparison(pot_first_df, pot_fam_df, fig_first_df, fig_fam_df,
                            pot_mesh, pot_verts, fig_mesh, fig_verts,
                            os.path.join(OUTPUT_BASE_DIR, "subject_comparison_2x2.png"), color_mode=mode)

    print("\nAll colored and grayscale plots generated successfully!")