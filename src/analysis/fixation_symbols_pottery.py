# import os
# from copy import deepcopy
# import numpy as np
# import pandas as pd
# import open3d as o3d
# import matplotlib.pyplot as plt
# import matplotlib.lines as mlines

# # ==========================================
# # ⚙️ CONFIGURATION
# # ==========================================
# OBJECTS = {
#     "pottery": {
#         "glb": r"D:\storage\jomon_kaen\pottery\IN0009(5).glb",
#         "first_time_csv": r"C:\Users\User\Desktop\Grant\MREEG\Trail_Experiment\archive\2026_08_07_09_43_48\fixations_agtzidis_pot.csv",
#         "familiar_csv": r"C:\Users\User\Desktop\Grant\MREEG\Trail_Experiment\archive\2026_08_04_09_27_45\eeg_fixations_agtzidis_pottery.csv",
#         "output_dir": r"C:\Users\User\Desktop\Grant\MREEG\Trail_Experiment\outputs_pottery"
#     },
#     "figurine": {
#         "glb": r"D:\storage\jomon_kaen\pottery\UD0028(93).glb",
#         "first_time_csv": r"C:\Users\User\Desktop\Grant\MREEG\Trail_Experiment\archive\2026_08_07_09_43_48\fixations_agtzidis_fig.csv",
#         "familiar_csv": r"C:\Users\User\Desktop\Grant\MREEG\Trail_Experiment\archive\2026_08_04_09_27_45\eeg_fixations_agtzidis_figurine.csv",
#         "output_dir": r"C:\Users\User\Desktop\Grant\MREEG\Trail_Experiment\outputs_figurine"
#     }
# }

# EEG_COLUMN = "FAA"
# BROWN      = (0.757, 0.604, 0.420) # Earthware fallback color if texture baking fails

# MARKER_DIVISOR  = 25.0
# THICKNESS_RATIO = 0.30
# SMOOTH_MESH     = True
# MODEL_ALPHA     = 0.55
# NORMAL_FLIP     = False

# # ---- ALL SYMBOLS WHITE ----
# SYMBOL_COLOR = (1.0, 1.0, 1.0)

# # ---- THREE CLASSES (thirds, logical 0.8-wide bins covering the data) ----
# BIN_EDGES = np.array([-1.2, -0.4, 0.4, 1.2])
# BIN_WIDTH = 0.8

# # bin order: [lower 1/3, middle 1/3, upper 1/3]
# SHAPE_ORDER = ['x', 'triangle', 'circle']          # X = lower, △ = middle, ○ = upper
# LEGEND_SYM  = {'x': 'X', 'triangle': '^', 'circle': 'o'}
# CLASS_NAME  = ['Lower 1/3', 'Middle 1/3', 'Upper 1/3']

# # ==========================================
# # 🧱 MODEL PREP (keep texture detail)
# # ==========================================
# def _bake_texture_to_vertex_colors(mesh):
#     tex = np.asarray(mesh.textures[0]).astype(np.float64)
#     if tex.max() > 1.5: tex /= 255.0
#     if tex.ndim == 2: tex = np.stack([tex]*3, -1)
#     if tex.shape[2] == 4: tex = tex[..., :3]
#     H, W, _ = tex.shape
#     uvs = np.asarray(mesh.triangle_uvs); tris = np.asarray(mesh.triangles)
#     u = np.clip(uvs[:,0],0,1); v = np.clip(uvs[:,1],0,1)
#     px = np.clip((u*W).astype(int),0,W-1); py = np.clip(((1-v)*H).astype(int),0,H-1)
#     cols = tex[py, px]; vidx = tris.reshape(-1)
#     acc = np.zeros((len(mesh.vertices),3)); cnt = np.zeros(len(mesh.vertices))
#     np.add.at(acc, vidx, cols); np.add.at(cnt, vidx, 1)
#     mesh.vertex_colors = o3d.utility.Vector3dVector(acc/np.maximum(cnt,1)[:,None])

# def prepare_model(path):
#     mesh = o3d.io.read_triangle_mesh(path, enable_post_processing=True)
#     if not mesh.has_vertices(): raise ValueError("Mesh has no vertices.")
#     try:
#         if not mesh.has_vertex_colors() and mesh.has_triangle_uvs() and len(mesh.textures) > 0:
#             _bake_texture_to_vertex_colors(mesh)
#     except Exception as e:
#         print(f"[warn] texture bake failed: {e}")
#     if not mesh.has_vertex_colors(): mesh.paint_uniform_color(BROWN)
#     if SMOOTH_MESH and len(mesh.triangles) < 200000:
#         mesh = mesh.subdivide_loop(number_of_iterations=1)
#     mesh.compute_vertex_normals()
#     return mesh

# # ==========================================
# # 🛠️ TEMPLATES: X, triangle, circle — ALL WHITE
# # ==========================================
# def _rz(deg): return o3d.geometry.get_rotation_matrix_from_xyz((0,0,np.deg2rad(deg)))
# def _cbox(w,h,d):
#     b = o3d.geometry.TriangleMesh.create_box(width=w,height=h,depth=d)
#     b.translate((-w/2,-h/2,-d/2)); return b

# def build_templates(size, thickness):
#     t = {}
#     # X : two diagonal arms
#     arm = size/3.0
#     xm = _cbox(size*1.15, arm, thickness) + _cbox(arm, size*1.15, thickness)
#     xm.rotate(_rz(45), center=(0,0,0))
#     t['x'] = xm
#     # △ : triangular prism, apex up
#     tri = o3d.geometry.TriangleMesh.create_cylinder(radius=size*0.65, height=thickness, resolution=3)
#     tri.rotate(_rz(90), center=(0,0,0))
#     t['triangle'] = tri
#     # ○ : disc
#     t['circle'] = o3d.geometry.TriangleMesh.create_cylinder(radius=size*0.5, height=thickness, resolution=32)
#     for key in t:
#         t[key].compute_vertex_normals()
#         t[key].paint_uniform_color(SYMBOL_COLOR)   # <-- WHITE everywhere
#     return t

# def rotation_from_normal(nrm, up=(0,0,1)):
#     nrm = nrm/(np.linalg.norm(nrm)+1e-12)
#     up = np.array(up,float)
#     if abs(np.dot(nrm,up)) > 0.98: up = np.array([0.,1.,0.])
#     x = np.cross(up,nrm); x /= (np.linalg.norm(x)+1e-12)
#     y = np.cross(nrm,x)
#     return np.column_stack([x,y,nrm])

# # ==========================================
# # 🎯 MARKERS (3 terciles, normal-oriented, white)
# # ==========================================
# def build_faa_markers(df, scene, tri_normals, templates, thickness):
#     dfc = df.dropna(subset=['centroid_x','centroid_y','centroid_z',EEG_COLUMN]).reset_index(drop=True)
#     if dfc.empty: return o3d.geometry.TriangleMesh()
#     pts  = dfc[['centroid_x','centroid_y','centroid_z']].to_numpy()
#     eyes = dfc[['eye_origin_x','eye_origin_y','eye_origin_z']].to_numpy()
#     faa  = dfc[EEG_COLUMN].to_numpy()

#     res = scene.compute_closest_points(o3d.core.Tensor(pts.astype(np.float32)))
#     closest = res['points'].numpy().astype(np.float64)
#     prims   = res['primitive_ids'].numpy()

#     offset = thickness*0.55
#     pieces = []
#     for p, eye, f, cp, pid in zip(pts, eyes, faa, closest, prims):
#         nrm = tri_normals[pid].copy() if 0 <= pid < len(tri_normals) else (p-cp)
#         if np.linalg.norm(nrm) < 1e-9: nrm = np.array([0.,0.,1.])
#         nrm /= np.linalg.norm(nrm)
#         if np.dot(nrm, eye-cp) < 0: nrm = -nrm
#         if NORMAL_FLIP: nrm = -nrm

#         idx = int(np.clip(np.floor((f - BIN_EDGES[0]) / BIN_WIDTH), 0, 2))
#         inst = deepcopy(templates[SHAPE_ORDER[idx]])
#         inst.rotate(rotation_from_normal(nrm), center=(0,0,0))

#         # FIX: Anchor to the closest surface point (cp) instead of the raw CSV point (p)
#         # This snaps any floating points perfectly onto the skin of the 3D mesh.
#         inst.translate(cp + nrm*offset)

#         pieces.append(inst)

#     out = o3d.geometry.TriangleMesh()
#     for m in pieces: out += m
#     return out

# # ==========================================
# # 💾 LEGEND PNG (separate file; white symbols with black edge)
# # ==========================================
# def save_legend_png(filepath, obj_name):
#     handles = []
#     for i in reversed(range(3)):          # upper 1/3 on top, lower 1/3 at bottom
#         lo, hi = BIN_EDGES[i], BIN_EDGES[i+1]
#         lab = ""
#         handles.append(mlines.Line2D([],[], color=SYMBOL_COLOR, marker=LEGEND_SYM[SHAPE_ORDER[i]],
#                            linestyle='None', markersize=15,
#                            markeredgecolor='black', markeredgewidth=1.5, label=lab))
#     fig = plt.figure(figsize=(5.2, 2.4))
#     ax = fig.add_subplot(111); ax.axis('off')
#     ax.legend(handles=handles, loc='center', fontsize=13, frameon=True,
#               edgecolor='black', title=f"{obj_name.capitalize()} FAA classes", title_fontsize=14)
#     fig.savefig(filepath, bbox_inches='tight', dpi=300); plt.close(fig)
#     print(f"Saved separate legend: {filepath}")

# # ==========================================
# # 👁️ VISUALIZATION (no legend inside the window)
# # ==========================================
# def visualize(model, markers, title):
#     try:
#         mat = o3d.visualization.rendering.MaterialRecord()
#         mat.shader = "defaultLit"
#         mat.base_color = [1.0,1.0,1.0,MODEL_ALPHA]
#         o3d.visualization.draw(
#             [{"name":"model","geometry":model,"material":mat},
#              {"name":"faa_shapes","geometry":markers}], title=title)
#     except Exception:
#         o3d.visualization.draw_geometries([model, markers], title=title, mesh_show_back_face=True)

# # ==========================================
# # 🚀 MAIN
# # ==========================================
# if __name__ == "__main__":

#     for obj_name, paths in OBJECTS.items():
#         print(f"\n{'='*20} Processing {obj_name.upper()} {'='*20}")
#         os.makedirs(paths["output_dir"], exist_ok=True)

#         dfs = {}
#         for name, path in [("first_time", paths["first_time_csv"]), ("familiar", paths["familiar_csv"])]:
#             if not os.path.exists(path):
#                 print(f"[warn] CSV not found: {path}. Skipping {name}.")
#                 continue

#             dfs[name] = pd.read_csv(path)

#             # Safeguard: Add default FAA column if missing
#             if EEG_COLUMN not in dfs[name].columns:
#                 print(f"[warn] {EEG_COLUMN} column not found in {name} CSV. Defaulting to 0.0.")
#                 dfs[name][EEG_COLUMN] = 0.0

#             v = dfs[name][EEG_COLUMN]
#             print(f"{name}: FAA min={v.min():+.4f}, max={v.max():+.4f}")

#         print("Class edges:", BIN_EDGES)

#         model_mesh = prepare_model(paths["glb"])
#         model_mesh.compute_triangle_normals()
#         tri_normals = np.asarray(model_mesh.triangle_normals)

#         marker_size = np.max(model_mesh.get_max_bound()-model_mesh.get_min_bound()) / MARKER_DIVISOR
#         thickness   = marker_size * THICKNESS_RATIO

#         scene = o3d.t.geometry.RaycastingScene()
#         scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(model_mesh))
#         templates = build_templates(marker_size, thickness)

#         results = []
#         for name, df in dfs.items():
#             markers = build_faa_markers(df, scene, tri_normals, templates, thickness)
#             ply_out = os.path.join(paths["output_dir"], f"{obj_name}_{name}_faa_shapes.ply")
#             o3d.io.write_triangle_mesh(ply_out, markers)
#             print(f"Saved {ply_out} ({len(markers.triangles)} tris)")
#             results.append((name, markers))

#         save_legend_png(os.path.join(paths["output_dir"], f"{obj_name}_faa_class_legend.png"), obj_name)

#         print(f"\nOpening interactive 3D windows for {obj_name} (close each to continue)...")
#         for name, markers in results:
#             visualize(deepcopy(model_mesh), markers, f"{obj_name.capitalize()} {name} — FAA terciles (drag to rotate)")

#     print("\nAll processing complete.")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import trimesh

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
POTTERY_GLB_PATH = r"D:\storage\jomon_kaen\pottery\IN0009(5).glb"
FIGURINE_GLB_PATH = r"D:\storage\jomon_kaen\pottery\UD0028(93).glb"

POTTERY_FIRST_CSV = r"C:\Users\User\Desktop\Grant\MREEG\Trail_Experiment\archive\2026_08_07_09_43_48\fixations_agtzidis_pot.csv"
FIGURINE_FIRST_CSV = r"C:\Users\User\Desktop\Grant\MREEG\Trail_Experiment\archive\2026_08_07_09_43_48\fixations_agtzidis_fig.csv"

POTTERY_FAMILIAR_CSV = r"C:\Users\User\Desktop\Grant\MREEG\Trail_Experiment\archive\2026_08_04_09_27_45\eeg_fixations_agtzidis_pottery.csv"
FIGURINE_FAMILIAR_CSV = r"C:\Users\User\Desktop\Grant\MREEG\Trail_Experiment\archive\2026_08_04_09_27_45\eeg_fixations_agtzidis_figurine.csv"

OUTPUT_BASE_DIR = "eeg_plots_output"
EEG_COLUMN = "FAA"


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
    # Map Y-up to Z-up
    plot_vertices = np.column_stack((tm_mesh.vertices[:,
                                                      0],
                                     tm_mesh.vertices[:,
                                                      2],
                                     tm_mesh.vertices[:,
                                                      1]))
    return df, tm_mesh, plot_vertices


# ==========================================
# 👁️ OCCLUSION TEST (ray-cast from camera)
# ==========================================
def classify_occlusion(mesh_plot, points, elev_deg, azim_deg, eps_frac=1e-3):
    """
    Returns a boolean array: True if the point is OCCLUDED by the mesh
    when viewed from the camera direction defined by (elev, azim).
    A ray is fired from far outside the object toward each point; if any
    triangle is intersected before reaching the point, the point is behind
    the surface from this viewpoint.
    """
    n = points.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)

    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)
    # Unit vector pointing from scene center toward the camera
    view = np.array([
        np.cos(elev) * np.cos(azim),
        np.cos(elev) * np.sin(azim),
        np.sin(elev)
    ])

    rng = np.max(mesh_plot.bounds[1] - mesh_plot.bounds[0])
    L = 5.0 * rng  # camera stand-off distance (well outside mesh)
    eps = eps_frac * rng  # tolerance so points ON the surface stay "visible"

    origins = points + view * L
    dirs = np.tile(-view, (n, 1))

    occ = np.zeros(n, dtype=bool)
    try:
        locs, idx_ray, _ = mesh_plot.ray.intersects_location(
            origins, dirs, multiple_hits=True)
        if len(locs) > 0:
            t = np.einsum('ij,ij->i', locs - origins[idx_ray], dirs[idx_ray])
            min_t = np.full(n, np.inf)
            np.minimum.at(min_t, idx_ray, t)
            occ = min_t < (L - eps)
    except Exception:
        occ = np.zeros(n, dtype=bool)  # safe fallback: everything visible
    return occ


# ==========================================
# 🛠️ CORE PLOTTING FUNCTION
#    (fine wireframe + low-opacity faces + occlusion-aware symbols)
# ==========================================
def draw_3d_artefact(ax, df, tm_mesh, plot_vertices, title, elev, azim):
    """Grayscale mesh as fine lines over faint faces; tercile symbols are
    depth-sorted against the mesh (occluded ones drawn faded behind it)."""

    # Mesh in plot coordinates, used for ray-casting
    mesh_plot = trimesh.Trimesh(vertices=plot_vertices,
                                faces=tm_mesh.faces,
                                process=False)

    # ---- 1) VERY low-opacity faces (a faint veil, no edges) ----
    face_collection = Poly3DCollection(
        plot_vertices[tm_mesh.faces],
        facecolors=(0.25, 0.25, 0.25, 0.05),   # ~6% opaque gray
        edgecolors='none',
        zorder=2)                               # above faded symbols, below lines
    ax.add_collection3d(face_collection)

    # ---- 2) FINE wireframe lines on top of the faces ----
    segments = plot_vertices[tm_mesh.edges_unique]  # (E, 2, 3)
    wire_collection = Line3DCollection(
        segments,
        colors=(0.35, 0.35, 0.35, 0.175),        # thin, semi-transparent lines
        linewidths=0.13,
        zorder=3)
    ax.add_collection3d(wire_collection)

    # ---- Fixation data ----
    df_clean = df.dropna(
        subset=['centroid_x',
                'centroid_y',
                'centroid_z',
                EEG_COLUMN])
    plot_cx = df_clean['centroid_x'].to_numpy()
    plot_cy = df_clean['centroid_z'].to_numpy()
    plot_cz = df_clean['centroid_y'].to_numpy()
    faa_vals = df_clean[EEG_COLUMN].to_numpy()
    pts = np.column_stack([plot_cx, plot_cy, plot_cz])

    # ---- Tercile masks ----
    mask_lower = faa_vals < -0.4
    mask_middle = (faa_vals >= -0.4) & (faa_vals < 0.4)
    mask_upper = faa_vals >= 0.4

    # ---- Per-viewpoint occlusion classification ----
    occ = classify_occlusion(mesh_plot, pts, elev, azim)

    # ---- Draw in depth layers: faded/occluded -> mesh -> crisp/visible ----
    # (sel_mask, alpha, zorder, size, linewidth)
    layers = [
        (occ,  0.55, 1, 90, 1.5),   # BEHIND the mesh: faded, drawn first
        (~occ, 1.00, 5, 90, 1.5),   # IN FRONT of the mesh: crisp, drawn last
    ]
    for sel, alpha, zo, sz, lw in layers:
        if np.any(mask_lower & sel):
            ax.scatter(plot_cx[mask_lower & sel],
                       plot_cy[mask_lower & sel],
                       plot_cz[mask_lower & sel],
                       color='black',
                       marker='x',
                       s=sz,
                       linewidths=lw * 1.1,
                       depthshade=False,
                       alpha=alpha,
                       zorder=zo)
        if np.any(mask_middle & sel):
            ax.scatter(plot_cx[mask_middle & sel],
                       plot_cy[mask_middle & sel],
                       plot_cz[mask_middle & sel],
                       facecolors='white',
                       edgecolors='black',
                       marker='^',
                       s=sz,
                       linewidths=lw * 0.8,
                       depthshade=False,
                       alpha=alpha,
                       zorder=zo)
        if np.any(mask_upper & sel):
            ax.scatter(plot_cx[mask_upper & sel],
                       plot_cy[mask_upper & sel],
                       plot_cz[mask_upper & sel],
                       facecolors='white',
                       edgecolors='black',
                       marker='o',
                       s=sz,
                       linewidths=lw * 0.95,
                       depthshade=False,
                       alpha=alpha,
                       zorder=zo)

    ax.set_title(title, pad=10, fontsize=12, fontweight='bold')
    ax.set_xlabel('Gaze-X')
    ax.set_ylabel('Gaze-Z (Original)')
    ax.set_zlabel('Gaze-Y (Original)')

    # Bounding Box
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


PAD_INCHES = 0.4  # <-- margin size in inches (0.4 in @ 300 dpi = 120 px)


def save_figure(fig, filepath, dpi=300):
    """Save figure with a uniform white margin around the entire image."""
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight',
                pad_inches=PAD_INCHES,      # <- the outer margin
                facecolor='white')          # <- make sure the margin is white
    plt.close(fig)
    print(f"Saved '{filepath}'")


# ==========================================
# 📊 GRID & COMPARISON PLOTS
# ==========================================
def plot_4_views_grid(df, mesh, verts, main_title, filepath, view_configs):
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(main_title, fontsize=14, fontweight='bold')
    for i, (title, elev, azim) in enumerate(view_configs):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        draw_3d_artefact(ax, df, mesh, verts, title, elev, azim)
    plt.tight_layout()
    save_figure(fig, filepath)
    # fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved '{filepath}'")


def plot_2x2_comparison(pot_first_df,
                        pot_fam_df,
                        fig_first_df,
                        fig_fam_df,
                        pot_mesh,
                        pot_verts,
                        fig_mesh,
                        fig_verts,
                        filepath):
    fig4 = plt.figure(figsize=(16, 12))

    ax_tl = fig4.add_subplot(2, 2, 1, projection='3d')
    draw_3d_artefact(ax_tl,
                     pot_first_df,
                     pot_mesh,
                     pot_verts,
                     "Pottery: First-time subject",
                     elev=15,
                     azim=-60)

    ax_tr = fig4.add_subplot(2, 2, 2, projection='3d')
    draw_3d_artefact(ax_tr,
                     pot_fam_df,
                     pot_mesh,
                     pot_verts,
                     "Pottery: Familiar subject",
                     elev=15,
                     azim=-60)

    ax_bl = fig4.add_subplot(2, 2, 3, projection='3d')
    draw_3d_artefact(ax_bl,
                     fig_first_df,
                     fig_mesh,
                     fig_verts,
                     "Figurine: First-time subject",
                     elev=15,
                     azim=-60)

    ax_br = fig4.add_subplot(2, 2, 4, projection='3d')
    draw_3d_artefact(ax_br,
                     fig_fam_df,
                     fig_mesh,
                     fig_verts,
                     "Figurine: Familiar subject",
                     elev=15,
                     azim=-60)

    plt.subplots_adjust(bottom=0.1)
    save_figure(fig4, filepath)
    # fig4.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print(f"Saved 2x2 Comparison: '{filepath}'")


# ==========================================
# 🏷️ SEPARATE LEGEND
# ==========================================
def save_legend(filepath):
    handles = [
        mlines.Line2D([],
                      [],
                      color='white',
                      marker='o',
                      linestyle='None',
                      markersize=15,
                      markeredgecolor='black',
                      markeredgewidth=1.5,
                      label=f"+0.4 ≤ FAA ≤ +1.2"),
        mlines.Line2D([],
                      [],
                      color='white',
                      marker='^',
                      linestyle='None',
                      markersize=15,
                      markeredgecolor='black',
                      markeredgewidth=1.5,
                      label=f"-0.4 ≤ FAA < +0.4"),
        mlines.Line2D([],
                      [],
                      color='black',
                      marker='x',
                      linestyle='None',
                      markersize=15,
                      markeredgewidth=1.5,
                      label=f"-1.2 ≤ FAA < -0.4"),
    ]

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.legend(handles=handles,
              loc='center',
              fontsize=14,
              frameon=True,
              edgecolor='black',
              title="FAA Classes",
              title_fontsize=14)
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved Legend: '{filepath}'")


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

    orbit_low = [("Front (Azim: -90°)",
                  15,
                  -90),
                 ("Right (Azim: 0°)",
                  15,
                  0),
                 ("Back (Azim: 90°)",
                  15,
                  90),
                 ("Left (Azim: 180°)",
                  15,
                  180)]
    orbit_high = [("Front High (30° Elev, -90° Azim)",
                   30,
                   -90),
                  ("Right High (30° Elev, 0° Azim)",
                   30,
                   0),
                  ("Back High (30° Elev, 90° Azim)",
                   30,
                   90),
                  ("Left High (30° Elev, 180° Azim)",
                   30,
                   180)]
    elev_sweep = [("Eye-Level View (0° Elev)",
                   0,
                   -60),
                  ("Slight High Angle (30° Elev)",
                   30,
                   -60),
                  ("High Angle (60° Elev)",
                   60,
                   -60),
                  ("Top-Down View (90° Elev)",
                   90,
                   -60)]

    print("--- Generating Wireframe + Occlusion-Aware Symbol Plots ---")

    # First-Time Subject
    # plot_4_views_grid(pot_first_df, pot_mesh, pot_verts, "Pottery (First-Time) - Low Orbit (15°)", os.path.join(first_time_dir, "pottery_orbit_low.png"), orbit_low)
    # plot_4_views_grid(pot_first_df, pot_mesh, pot_verts, "Pottery (First-Time) - High Orbit (30°)", os.path.join(first_time_dir, "pottery_orbit_high.png"), orbit_high)
    # plot_4_views_grid(pot_first_df, pot_mesh, pot_verts, "Pottery (First-Time) - Elevation Sweep", os.path.join(first_time_dir, "pottery_elev_sweep.png"), elev_sweep)

    # plot_4_views_grid(fig_first_df, fig_mesh, fig_verts, "Figurine (First-Time) - Low Orbit (15°)", os.path.join(first_time_dir, "figurine_orbit_low.png"), orbit_low)
    # plot_4_views_grid(fig_first_df, fig_mesh, fig_verts, "Figurine (First-Time) - High Orbit (30°)", os.path.join(first_time_dir, "figurine_orbit_high.png"), orbit_high)
    # plot_4_views_grid(fig_first_df, fig_mesh, fig_verts, "Figurine (First-Time) - Elevation Sweep", os.path.join(first_time_dir, "figurine_elev_sweep.png"), elev_sweep)

    # # Familiar Subject
    # plot_4_views_grid(pot_fam_df, pot_mesh, pot_verts, "Pottery (Familiar) - Low Orbit (15°)", os.path.join(familiar_dir, "pottery_orbit_low.png"), orbit_low)
    # plot_4_views_grid(pot_fam_df, pot_mesh, pot_verts, "Pottery (Familiar) - High Orbit (30°)", os.path.join(familiar_dir, "pottery_orbit_high.png"), orbit_high)
    # plot_4_views_grid(pot_fam_df, pot_mesh, pot_verts, "Pottery (Familiar) - Elevation Sweep", os.path.join(familiar_dir, "pottery_elev_sweep.png"), elev_sweep)

    # plot_4_views_grid(fig_fam_df, fig_mesh, fig_verts, "Figurine (Familiar) - Low Orbit (15°)", os.path.join(familiar_dir, "figurine_orbit_low.png"), orbit_low)
    # plot_4_views_grid(fig_fam_df, fig_mesh, fig_verts, "Figurine (Familiar) - High Orbit (30°)", os.path.join(familiar_dir, "figurine_orbit_high.png"), orbit_high)
    # plot_4_views_grid(fig_fam_df, fig_mesh, fig_verts, "Figurine (Familiar) - Elevation Sweep", os.path.join(familiar_dir, "figurine_elev_sweep.png"), elev_sweep)

    # # 2x2 Side-by-Side Comparison
    # plot_2x2_comparison(pot_first_df, pot_fam_df, fig_first_df, fig_fam_df,
    #                     pot_mesh, pot_verts, fig_mesh, fig_verts,
    #                     os.path.join(OUTPUT_BASE_DIR, "subject_comparison_2x2.png"))

    # Separate Legend
    save_legend(os.path.join(OUTPUT_BASE_DIR, "faa_tercile_legend.png"))

    print("\nAll wireframe + occlusion-aware plots generated successfully!")
