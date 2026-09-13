import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
# Paths to your 3D models
POTTERY_GLB_PATH = r"D:\storage\jomon_kaen\pottery\IN0009(5).glb"
FIGURINE_GLB_PATH = r"D:\storage\jomon_kaen\pottery\UD0028(93).glb"

# First-time subject CSVs
POTTERY_FIRST_CSV = r"C:\Users\User\Desktop\Grant\MREEG\Trail_Experiment\2026_08_07_09_43_48\fixations_agtzidis_pot.csv"
FIGURINE_FIRST_CSV = r"C:\Users\User\Desktop\Grant\MREEG\Trail_Experiment\2026_08_07_09_43_48\fixations_agtzidis_fig.csv"

# Familiar subject CSVs
POTTERY_FAMILIAR_CSV = r"C:\Users\User\Desktop\Grant\MREEG\Trail_Experiment\2026_08_04_09_27_45\eeg_fixations_agtzidis_pottery.csv"
FIGURINE_FAMILIAR_CSV = r"C:\Users\User\Desktop\Grant\MREEG\Trail_Experiment\2026_08_04_09_27_45\eeg_fixations_agtzidis_figurine.csv"

# Output folder structure
OUTPUT_BASE_DIR = "eeg_plots_output"

EEG_COLUMN = "FAA"
COLORBAR_TICKS = np.arange(-1.0, 1.1, 0.2)

# --- Occlusion handling ---
# Front-facing points are always drawn fully opaque/crisp on top of the mesh.
# Back-facing (occluded) points are still shown, but drawn *underneath* the mesh in
# paint order so the mesh's own translucency blends naturally over them -- giving a
# clear "seen through the object" muted look instead of either hiding them or making
# them fully opaque like the front points.
HIDE_OCCLUDED_POINTS = False
# Fraction of the mesh's bounding-box diagonal used as a tolerance so points that sit
# essentially ON the surface aren't flagged as "occluded by themselves".
OCCLUSION_EPS_FRAC = 0.01


# ==========================================
# 🛠️ DATA LOADING UTILITY
# ==========================================
def load_artefact_data(csv_path, glb_path):
    """Loads CSV and GLB once to speed up multiple plotting calls."""
    df = pd.read_csv(csv_path)

    # Safeguard: Add default FAA column if missing (e.g., in the pottery CSV)
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

    # Build a mesh in PLOT space (same faces, permuted vertices) so ray-casting for
    # visibility uses the exact same coordinate system as the matplotlib axes/camera.
    plot_mesh = trimesh.Trimesh(vertices=plot_vertices, faces=tm_mesh.faces, process=False)

    # Scanned artifact meshes are frequently non-watertight / have inconsistent face
    # winding, which can make trimesh's ray-triangle intersector silently return zero
    # hits (i.e. every point looks "visible" even when it's clearly behind the object).
    # Repair before ray-casting so occlusion actually works.
    if not plot_mesh.is_winding_consistent:
        trimesh.repair.fix_winding(plot_mesh)
    if not plot_mesh.is_watertight:
        trimesh.repair.fill_holes(plot_mesh)
    print(f"  [mesh] {os.path.basename(glb_path)}: watertight={plot_mesh.is_watertight}, "
          f"winding_consistent={plot_mesh.is_winding_consistent}, faces={len(plot_mesh.faces)}")

    return df, tm_mesh, plot_vertices, plot_mesh


# ==========================================
# 🛠️ VISIBILITY / OCCLUSION UTILITY
# ==========================================
def compute_point_visibility(plot_mesh, points, elev, azim, eps_frac=OCCLUSION_EPS_FRAC):
    """
    For a given camera (elev, azim, matplotlib convention) determine which of `points`
    have an unobstructed line of sight to the camera (i.e. are on the near/front side
    of the mesh) vs. which are blocked by the mesh itself (back side / interior).

    Returns a boolean array, True = visible / front-facing.
    """
    if len(points) == 0:
        return np.array([], dtype=bool)

    elev_r, azim_r = np.radians(elev), np.radians(azim)
    # Matplotlib's camera direction convention
    view_dir = np.array([
        np.cos(elev_r) * np.cos(azim_r),
        np.cos(elev_r) * np.sin(azim_r),
        np.sin(elev_r)
    ])
    view_dir = view_dir / np.linalg.norm(view_dir)

    bounds = plot_mesh.bounds
    diag = np.linalg.norm(bounds[1] - bounds[0])
    center = plot_mesh.bounding_box.centroid

    # Push the "camera" well outside the mesh along the view direction
    cam_pos = center + view_dir * diag * 3.0
    eps = diag * eps_frac

    ray_origins = np.tile(cam_pos, (len(points), 1))
    ray_vecs = points - ray_origins
    dist_to_point = np.linalg.norm(ray_vecs, axis=1)
    ray_dirs = ray_vecs / dist_to_point[:, None]

    visible = np.ones(len(points), dtype=bool)

    try:
        locations, index_ray, _ = plot_mesh.ray.intersects_location(
            ray_origins=ray_origins, ray_directions=ray_dirs)
    except Exception as e:
        print(f"  [visibility] ray-cast failed ({e}); treating all points as visible")
        return visible

    if len(index_ray) == 0:
        # No intersections at all across every point's ray usually means the mesh
        # isn't valid for ray-casting (e.g. not watertight, inverted/degenerate faces,
        # or duplicate geometry) rather than every point genuinely having a clear line
        # of sight. Surface it loudly instead of silently marking everything visible.
        print(f"  [visibility] WARNING: 0 ray-mesh intersections for {len(points)} points "
              f"at elev={elev}, azim={azim} -- check plot_mesh.is_watertight "
              f"({plot_mesh.is_watertight}) / plot_mesh.is_winding_consistent "
              f"({plot_mesh.is_winding_consistent})")
        return visible

    hit_dist = np.linalg.norm(locations - ray_origins[index_ray], axis=1)
    for ray_idx in np.unique(index_ray):
        mask = index_ray == ray_idx
        nearest_hit = hit_dist[mask].min()
        # If the mesh is hit meaningfully BEFORE reaching the point, the point is
        # occluded (it's on the far side of the object from this camera angle).
        if nearest_hit < dist_to_point[ray_idx] - eps:
            visible[ray_idx] = False

    return visible


# ==========================================
# 🛠️ CORE PLOTTING FUNCTION
# ==========================================
def draw_3d_artefact(ax, df, tm_mesh, plot_vertices, plot_mesh, title, elev, azim):
    """Draws the mesh and scatter points on a given matplotlib 3D axis."""

    # By default Axes3D recomputes each artist's draw order from its *average* depth
    # (computed_zorder=True) and this OVERRIDES any zorder you set manually. Since the
    # mesh's faces span near and far, its average depth can rank "in front of" points
    # that are geometrically closer to the camera, so mpl paints the translucent mesh
    # back over those points -- this, not colormap blending, is what was causing the
    # muddy/occluded look even for front-facing points. Disabling it makes mpl respect
    # the explicit zorder values below (mesh=1, points=10/11) instead.
    ax.computed_zorder = False

    # 1. Plot Mesh (Earthware Pottery Color)
    # Matplotlib cannot render .glb textures natively. We use a realistic earthware/clay
    # hex color (#C19A6B) with subtle sienna edges (#8B4513) for a 3D ceramic feel.
    mesh_collection = Poly3DCollection(plot_vertices[tm_mesh.faces],
                                       alpha=0.35,
                                       facecolor='#C19A6B',
                                       edgecolor='#8B4513',
                                       linewidth=0.15,
                                       zorder=1)
    ax.add_collection3d(mesh_collection)

    # Clean data to avoid NaN/Inf errors during limit calculation
    df_clean = df.dropna(
        subset=['centroid_x',
                'centroid_y',
                'centroid_z',
                EEG_COLUMN])

    # 2. Plot Gaze Points (Centroids only, NO paths)
    plot_cx = df_clean['centroid_x'].to_numpy()
    plot_cy = df_clean['centroid_z'].to_numpy()
    plot_cz = df_clean['centroid_y'].to_numpy()
    plot_c = df_clean[EEG_COLUMN].to_numpy()

    points = np.column_stack((plot_cx, plot_cy, plot_cz))

    # --- Determine which points are actually on the near/front side of the mesh
    # for THIS camera angle, instead of relying on matplotlib's unreliable 3D
    # depth-sorting (which is what caused colors to bleed/mix before). ---
    visible_mask = compute_point_visibility(plot_mesh, points, elev, azim)

    scatter = None

    # --- Occluded (back-side) points: drawn BEFORE/BELOW the mesh (zorder lower than
    # the mesh's zorder=1) so the mesh's own alpha=0.35 blends naturally on top of
    # them -- this intentionally reproduces the "seen through the object" muted look,
    # as a deliberate visual cue that these points are on the far side, in contrast to
    # the fully opaque front points drawn on top of the mesh below.
    if np.any(~visible_mask) and not HIDE_OCCLUDED_POINTS:
        ax.scatter(plot_cx[~visible_mask], plot_cy[~visible_mask], plot_cz[~visible_mask],
                   c=plot_c[~visible_mask], cmap='jet', vmin=-1.0, vmax=1.0,
                   s=65, edgecolors='black', linewidth=1.0,
                   depthshade=False, alpha=1.0, zorder=0)

    # --- Visible (front-facing) points ---
    if np.any(visible_mask):
        vx, vy, vz = plot_cx[visible_mask], plot_cy[visible_mask], plot_cz[visible_mask]
        vc = plot_c[visible_mask]

        # Step A: Solid white "halo" behind the points. Still useful even with
        # occlusion filtering, since it keeps the marker edge crisp against the mesh.
        ax.scatter(vx, vy, vz,
                   c='white', s=110, edgecolors='none',
                   depthshade=False, alpha=1.0, zorder=10)

        # Step B: Draw the actual colored points on top of the white halo
        scatter = ax.scatter(vx, vy, vz,
                             c=vc, cmap='jet', vmin=-1.0, vmax=1.0,
                             s=65, edgecolors='black', linewidth=1.2,
                             depthshade=False, alpha=1.0, zorder=11)

    # Colorbar needs a mappable even if every point on this particular view happened
    # to be occluded (rare, but possible on a fully back-facing view) -- fall back to
    # an invisible scatter carrying the full colormap range.
    if scatter is None:
        scatter = ax.scatter([], [], [], c=[], cmap='jet', vmin=-1.0, vmax=1.0)

    # 3. Formatting
    ax.set_title(title, pad=10, fontsize=12, fontweight='bold')
    ax.set_xlabel('Gaze-X')
    ax.set_ylabel('Gaze-Z (Original)')
    ax.set_zlabel('Gaze-Y (Original)')

    # 4. Bounding Box (Force Equal Aspect Ratio to avoid stretching)
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

    # 5. Set camera angle
    ax.view_init(elev=elev, azim=azim)
    return scatter


def plot_4_views_grid(df, mesh, verts, plot_mesh, main_title, filepath, view_configs):
    """Helper function to generate a 2x2 grid based on a list of view configurations."""
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(main_title, fontsize=14, fontweight='bold')

    for i, (title, elev, azim) in enumerate(view_configs):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        sc = draw_3d_artefact(ax, df, mesh, verts, plot_mesh, title, elev, azim)
        if i == 1:
            fig.colorbar(sc,
                         ax=ax,
                         shrink=0.5,
                         pad=0.1,
                         label='FAA',
                         ticks=COLORBAR_TICKS)

    plt.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
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
    pot_first_df, pot_mesh, pot_verts, pot_plot_mesh = load_artefact_data(POTTERY_FIRST_CSV, POTTERY_GLB_PATH)
    pot_fam_df, _, _, _ = load_artefact_data(POTTERY_FAMILIAR_CSV, POTTERY_GLB_PATH)

    fig_first_df, fig_mesh, fig_verts, fig_plot_mesh = load_artefact_data(FIGURINE_FIRST_CSV, FIGURINE_GLB_PATH)
    fig_fam_df, _, _, _ = load_artefact_data(FIGURINE_FAMILIAR_CSV, FIGURINE_GLB_PATH)
    print("Data loaded successfully.\n")

    # ---------------------------------------------------------
    # CAMERA VIEWS
    # ---------------------------------------------------------
    orbit_low = [("Front (Azim: -90°)", 15, -90),
                 ("Right (Azim: 0°)", 15, 0),
                 ("Back (Azim: 90°)", 15, 90),
                 ("Left (Azim: 180°)", 15, 180)]

    orbit_high = [("Front High (30° Elev, -90° Azim)", 30, -90),
                  ("Right High (30° Elev, 0° Azim)", 30, 0),
                  ("Back High (30° Elev, 90° Azim)", 30, 90),
                  ("Left High (30° Elev, 180° Azim)", 30, 180)]

    elev_sweep = [("Eye-Level View (0° Elev)", 0, -60),
                  ("Slight High Angle (30° Elev)", 30, -60),
                  ("High Angle (60° Elev)", 60, -60),
                  ("Top-Down View (90° Elev)", 90, -60)]

    # ---------------------------------------------------------
    # 1. SAVE FIRST-TIME SUBJECT PLOTS
    # ---------------------------------------------------------
    print("Generating First-Time Subject grids...")
    plot_4_views_grid(pot_first_df, pot_mesh, pot_verts, pot_plot_mesh, "Pottery (First-Time) - Low Orbit (15°)", os.path.join(first_time_dir, "pottery_orbit_low.png"), orbit_low)
    plot_4_views_grid(pot_first_df, pot_mesh, pot_verts, pot_plot_mesh, "Pottery (First-Time) - High Orbit (30°)", os.path.join(first_time_dir, "pottery_orbit_high.png"), orbit_high)
    plot_4_views_grid(pot_first_df, pot_mesh, pot_verts, pot_plot_mesh, "Pottery (First-Time) - Elevation Sweep", os.path.join(first_time_dir, "pottery_elev_sweep.png"), elev_sweep)

    plot_4_views_grid(fig_first_df, fig_mesh, fig_verts, fig_plot_mesh, "Figurine (First-Time) - Low Orbit (15°)", os.path.join(first_time_dir, "figurine_orbit_low.png"), orbit_low)
    plot_4_views_grid(fig_first_df, fig_mesh, fig_verts, fig_plot_mesh, "Figurine (First-Time) - High Orbit (30°)", os.path.join(first_time_dir, "figurine_orbit_high.png"), orbit_high)
    plot_4_views_grid(fig_first_df, fig_mesh, fig_verts, fig_plot_mesh, "Figurine (First-Time) - Elevation Sweep", os.path.join(first_time_dir, "figurine_elev_sweep.png"), elev_sweep)

    # ---------------------------------------------------------
    # 2. SAVE FAMILIAR SUBJECT PLOTS
    # ---------------------------------------------------------
    print("\nGenerating Familiar Subject grids...")
    plot_4_views_grid(pot_fam_df, pot_mesh, pot_verts, pot_plot_mesh, "Pottery (Familiar) - Low Orbit (15°)", os.path.join(familiar_dir, "pottery_orbit_low.png"), orbit_low)
    plot_4_views_grid(pot_fam_df, pot_mesh, pot_verts, pot_plot_mesh, "Pottery (Familiar) - High Orbit (30°)", os.path.join(familiar_dir, "pottery_orbit_high.png"), orbit_high)
    plot_4_views_grid(pot_fam_df, pot_mesh, pot_verts, pot_plot_mesh, "Pottery (Familiar) - Elevation Sweep", os.path.join(familiar_dir, "pottery_elev_sweep.png"), elev_sweep)

    plot_4_views_grid(fig_fam_df, fig_mesh, fig_verts, fig_plot_mesh, "Figurine (Familiar) - Low Orbit (15°)", os.path.join(familiar_dir, "figurine_orbit_low.png"), orbit_low)
    plot_4_views_grid(fig_fam_df, fig_mesh, fig_verts, fig_plot_mesh, "Figurine (Familiar) - High Orbit (30°)", os.path.join(familiar_dir, "figurine_orbit_high.png"), orbit_high)
    plot_4_views_grid(fig_fam_df, fig_mesh, fig_verts, fig_plot_mesh, "Figurine (Familiar) - Elevation Sweep", os.path.join(familiar_dir, "figurine_elev_sweep.png"), elev_sweep)

    # ---------------------------------------------------------
    # 3. RESTORE COMPARISON PLOT (2x2 Grid)
    # ---------------------------------------------------------
    print("\nGenerating 2x2 Subject Comparison Plot...")
    fig4 = plt.figure(figsize=(16, 12))

    ax_tl = fig4.add_subplot(2, 2, 1, projection='3d')
    sc_tl = draw_3d_artefact(ax_tl, pot_first_df, pot_mesh, pot_verts, pot_plot_mesh, "Pottery: First-time subject", elev=15, azim=-60)
    fig4.colorbar(sc_tl, ax=ax_tl, shrink=0.6, pad=0.1, label='FAA', ticks=COLORBAR_TICKS)

    ax_tr = fig4.add_subplot(2, 2, 2, projection='3d')
    sc_tr = draw_3d_artefact(ax_tr, pot_fam_df, pot_mesh, pot_verts, pot_plot_mesh, "Pottery: Familiar subject", elev=15, azim=-60)
    fig4.colorbar(sc_tr, ax=ax_tr, shrink=0.6, pad=0.1, label='FAA', ticks=COLORBAR_TICKS)

    ax_bl = fig4.add_subplot(2, 2, 3, projection='3d')
    sc_bl = draw_3d_artefact(ax_bl, fig_first_df, fig_mesh, fig_verts, fig_plot_mesh, "Figurine: First-time subject", elev=15, azim=-60)
    fig4.colorbar(sc_bl, ax=ax_bl, shrink=0.6, pad=0.1, label='FAA', ticks=COLORBAR_TICKS)

    ax_br = fig4.add_subplot(2, 2, 4, projection='3d')
    sc_br = draw_3d_artefact(ax_br, fig_fam_df, fig_mesh, fig_verts, fig_plot_mesh, "Figurine: Familiar subject", elev=15, azim=-60)
    fig4.colorbar(sc_br, ax=ax_br, shrink=0.6, pad=0.1, label='FAA', ticks=COLORBAR_TICKS)

    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    comparison_output = os.path.join(OUTPUT_BASE_DIR, "subject_comparison_2x2.png")
    fig4.savefig(comparison_output, dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print(f"Saved '{comparison_output}'")

    print("\nAll plots generated successfully!")