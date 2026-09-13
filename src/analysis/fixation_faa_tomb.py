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
# OBJECT_GLB_PATH = r"C:\Users\User\Desktop\Grant\MREEG\remmbak7(A).glb"
# HOLO3_CSV_PATH = r"C:\Users\User\Desktop\Grant\MREEG\Trail_Experiment\latest_HoloLens_csvraw\fixations_agtzidis_holo3_obj2.csv"
# HOLO4_CSV_PATH = r"C:\Users\User\Desktop\Grant\MREEG\Trail_Experiment\latest_HoloLens_csvraw\fixations_agtzidis_holo4_obj2.csv"

# OUTPUT_DIR = r"C:\Users\User\Desktop\Grant\MREEG\Trail_Experiment\latest_HoloLens_csvraw\outputs"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# EEG_COLUMN = "FAA"
# BROWN = (0.757, 0.604, 0.420)

# MARKER_DIVISOR = 75.0
# THICKNESS_RATIO = 0.30
# SMOOTH_MESH = True
# TOMB_ALPHA = 0.55
# NORMAL_FLIP = False

# # ---- LOGICAL BINS: 0.4 increments, zero-anchored, cover [-0.8412, 1.0711] ----
# BIN_EDGES = np.array([-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2])
# BIN_WIDTH = 0.4

# # Shapes ordered: least edges (min FAA) -> circle (max FAA)
# SHAPE_ORDER = ['triangle', 'square', 'pentagon', 'hexagon', 'star', 'circle']
# BIN_COLORS  = [(0.0, 0.2, 1.0),   # blue    [-1.2, -0.8)
#                (0.0, 1.0, 1.0),   # cyan    [-0.8, -0.4)
#                (0.2, 1.0, 0.2),   # lime    [-0.4,  0.0)
#                (1.0, 1.0, 0.0),   # yellow  [ 0.0,  0.4)
#                (1.0, 0.0, 1.0),   # magenta [ 0.4,  0.8)
#                (1.0, 1.0, 1.0)]   # white   [ 0.8,  1.2]
# LEGEND_SYM = {
#     'triangle': '^',
#     'square': 's',
#     'pentagon': 'p',
#     'hexagon': 'h',
#     'star': (6,
#              1,
#              0),
#     'circle': 'o'
# }


# # ==========================================
# # 🧱 TOMB PREP (keep texture detail)
# # ==========================================
# def _bake_texture_to_vertex_colors(mesh):
#     tex = np.asarray(mesh.textures[0]).astype(np.float64)
#     if tex.max() > 1.5: tex /= 255.0
#     if tex.ndim == 2: tex = np.stack([tex] * 3, -1)
#     if tex.shape[2] == 4: tex = tex[..., :3]
#     H, W, _ = tex.shape
#     uvs = np.asarray(mesh.triangle_uvs)
#     tris = np.asarray(mesh.triangles)
#     u = np.clip(uvs[:, 0], 0, 1)
#     v = np.clip(uvs[:, 1], 0, 1)
#     px = np.clip((u * W).astype(int), 0, W - 1)
#     py = np.clip(((1 - v) * H).astype(int),
#                  0,
#                  H - 1)
#     cols = tex[py, px]
#     vidx = tris.reshape(-1)
#     acc = np.zeros((len(mesh.vertices), 3))
#     cnt = np.zeros(len(mesh.vertices))
#     np.add.at(acc, vidx, cols)
#     np.add.at(cnt, vidx, 1)
#     mesh.vertex_colors = o3d.utility.Vector3dVector(acc / np.maximum(cnt,
#                                                                      1)[:,
#                                                                         None])


# def prepare_tomb(path):
#     mesh = o3d.io.read_triangle_mesh(path, enable_post_processing=True)
#     if not mesh.has_vertices(): raise ValueError("Mesh has no vertices.")
#     try:
#         if not mesh.has_vertex_colors() and mesh.has_triangle_uvs() and len(
#                 mesh.textures) > 0:
#             _bake_texture_to_vertex_colors(mesh)
#     except Exception as e:
#         print(f"[warn] texture bake failed: {e}")
#     if not mesh.has_vertex_colors(): mesh.paint_uniform_color(BROWN)
#     if SMOOTH_MESH and len(mesh.triangles) < 200000:
#         mesh = mesh.subdivide_loop(number_of_iterations=1)
#     mesh.compute_vertex_normals()
#     return mesh


# # ==========================================
# # 🛠️ TEMPLATES: 3 -> 4 -> 5 -> 6 -> 12 -> circle edges
# # ==========================================
# def _rz(deg):
#     return o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.deg2rad(deg)))


# def build_templates(size, thickness):
#     t = {}
#     tri = o3d.geometry.TriangleMesh.create_cylinder(radius=size * 0.62,
#                                                     height=thickness,
#                                                     resolution=3)
#     tri.rotate(_rz(90), center=(0, 0, 0))
#     t['triangle'] = tri
#     sq = o3d.geometry.TriangleMesh.create_cylinder(radius=size * 0.55,
#                                                    height=thickness,
#                                                    resolution=4)
#     sq.rotate(_rz(45), center=(0, 0, 0))
#     t['square'] = sq
#     pen = o3d.geometry.TriangleMesh.create_cylinder(radius=size * 0.55,
#                                                     height=thickness,
#                                                     resolution=5)
#     pen.rotate(_rz(90), center=(0, 0, 0))
#     t['pentagon'] = pen
#     hexa = o3d.geometry.TriangleMesh.create_cylinder(radius=size * 0.55,
#                                                      height=thickness,
#                                                      resolution=6)
#     hexa.rotate(_rz(90), center=(0, 0, 0))
#     t['hexagon'] = hexa
#     s1 = o3d.geometry.TriangleMesh.create_cylinder(radius=size * 0.62,
#                                                    height=thickness,
#                                                    resolution=3)
#     s1.rotate(_rz(90), center=(0, 0, 0))
#     s2 = deepcopy(s1)
#     s2.rotate(_rz(60), center=(0, 0, 0))
#     t['star'] = s1 + s2  # hexagram: clearly not a hexagon or circle
#     t['circle'] = o3d.geometry.TriangleMesh.create_cylinder(radius=size * 0.5,
#                                                             height=thickness,
#                                                             resolution=32)
#     for i, key in enumerate(SHAPE_ORDER):
#         t[key].compute_vertex_normals()
#         t[key].paint_uniform_color(BIN_COLORS[i])
#     return t


# def rotation_from_normal(nrm, up=(0, 0, 1)):
#     nrm = nrm / (np.linalg.norm(nrm) + 1e-12)
#     up = np.array(up, float)
#     if abs(np.dot(nrm, up)) > 0.98: up = np.array([0., 1., 0.])
#     x = np.cross(up, nrm)
#     x /= (np.linalg.norm(x) + 1e-12)
#     y = np.cross(nrm, x)
#     return np.column_stack([x, y, nrm])


# # ==========================================
# # 🎯 MARKERS (logical 0.4 bins)
# # ==========================================
# def build_faa_markers(df, scene, tri_normals, templates, thickness):
#     dfc = df.dropna(
#         subset=['centroid_x',
#                 'centroid_y',
#                 'centroid_z',
#                 EEG_COLUMN]).reset_index(drop=True)
#     if dfc.empty: return o3d.geometry.TriangleMesh()
#     pts = dfc[['centroid_x', 'centroid_y', 'centroid_z']].to_numpy()
#     eyes = dfc[['eye_origin_x', 'eye_origin_y', 'eye_origin_z']].to_numpy()
#     faa = dfc[EEG_COLUMN].to_numpy()

#     res = scene.compute_closest_points(o3d.core.Tensor(pts.astype(np.float32)))
#     closest = res['points'].numpy().astype(np.float64)
#     prims = res['primitive_ids'].numpy()

#     n_bins = len(SHAPE_ORDER)
#     offset = thickness * 0.55
#     pieces = []
#     for p, eye, f, cp, pid in zip(pts, eyes, faa, closest, prims):
#         nrm = tri_normals[pid].copy() if 0 <= pid < len(tri_normals) else (p -
#                                                                            cp)
#         if np.linalg.norm(nrm) < 1e-9: nrm = np.array([0., 0., 1.])
#         nrm /= np.linalg.norm(nrm)
#         if np.dot(nrm, eye - cp) < 0: nrm = -nrm
#         if NORMAL_FLIP: nrm = -nrm

#         idx = int(
#             np.clip(np.floor((f - BIN_EDGES[0]) / BIN_WIDTH),
#                     0,
#                     n_bins - 1))
#         inst = deepcopy(templates[SHAPE_ORDER[idx]])
#         inst.rotate(rotation_from_normal(nrm), center=(0, 0, 0))
#         inst.translate(p + nrm * offset)
#         pieces.append(inst)

#     out = o3d.geometry.TriangleMesh()
#     for m in pieces:
#         out += m
#     return out


# # ==========================================
# # 💾 LEGEND PNG
# # ==========================================
# def save_legend_png(filepath):
#     handles = []
#     for i, key in enumerate(SHAPE_ORDER):
#         lo, hi = BIN_EDGES[i], BIN_EDGES[i + 1]
#         lab = f"{lo:+.1f} ≤ FAA < {hi:+.1f}" if i < len(
#             SHAPE_ORDER) - 1 else f"{lo:+.1f} ≤ FAA ≤ {hi:+.1f} (Max)"
#         if i == 0: lab = f"{lo:+.1f} (Min) ≤ FAA < {hi:+.1f}"
#         handles.append(
#             mlines.Line2D([],
#                           [],
#                           color=BIN_COLORS[i],
#                           marker=LEGEND_SYM[key],
#                           linestyle='None',
#                           markersize=13,
#                           markeredgecolor='black',
#                           markeredgewidth=1.2,
#                           label=lab))

#     # 🔃 REVERSED ORDER: positive (Max) at top -> negative (Min) at bottom
#     handles = handles[::-1]

#     fig = plt.figure(figsize=(5.2, 4.0))
#     ax = fig.add_subplot(111)
#     ax.axis('off')
#     ax.legend(handles=handles,
#               loc='center',
#               fontsize=13,
#               frameon=True,
#             #   facecolor='#8a7355',
#               edgecolor='black',
#             #   title="FAA bins (0.4 steps): 3 edges -> circle",
#               title="FAA bins (0.4 steps)",
#               title_fontsize=13)
#     fig.savefig(filepath, bbox_inches='tight', dpi=300)
#     plt.close(fig)
#     print(f"Saved Legend to '{filepath}'")


# # ==========================================
# # 👁️ VISUALIZATION
# # ==========================================
# def visualize(tomb, markers, title):
#     try:
#         mat = o3d.visualization.rendering.MaterialRecord()
#         mat.shader = "defaultLit"
#         mat.base_color = [1.0, 1.0, 1.0, TOMB_ALPHA]
#         o3d.visualization.draw([{
#             "name": "tomb",
#             "geometry": tomb,
#             "material": mat
#         },
#                                 {
#                                     "name": "faa_shapes",
#                                     "geometry": markers
#                                 }],
#                                title=title)
#     except Exception:
#         o3d.visualization.draw_geometries([tomb,
#                                            markers],
#                                           title=title,
#                                           mesh_show_back_face=True)


# # ==========================================
# # 🚀 MAIN
# # ==========================================
# if __name__ == "__main__":
#     dfs = {}
#     for name, path in [("holo3", HOLO3_CSV_PATH), ("holo4", HOLO4_CSV_PATH)]:
#         dfs[name] = pd.read_csv(path)
#         v = dfs[name][EEG_COLUMN]
#         print(f"{name}: FAA min={v.min():+.4f}, max={v.max():+.4f}")
#     print("Logical bins (0.4 increments):", BIN_EDGES)

#     tomb = prepare_tomb(OBJECT_GLB_PATH)
#     tomb.compute_triangle_normals()
#     tri_normals = np.asarray(tomb.triangle_normals)

#     marker_size = np.max(tomb.get_max_bound() -
#                          tomb.get_min_bound()) / MARKER_DIVISOR
#     thickness = marker_size * THICKNESS_RATIO

#     scene = o3d.t.geometry.RaycastingScene()
#     scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(tomb))
#     templates = build_templates(marker_size, thickness)

#     results = []
#     for name in ["holo3", "holo4"]:
#         markers = build_faa_markers(dfs[name],
#                                     scene,
#                                     tri_normals,
#                                     templates,
#                                     thickness)
#         ply_out = os.path.join(OUTPUT_DIR, f"{name}_faa_shapes.ply")
#         o3d.io.write_triangle_mesh(ply_out, markers)
#         print(f"Saved {ply_out} ({len(markers.triangles)} tris)")
#         results.append((name, markers))

#     save_legend_png(os.path.join(OUTPUT_DIR, "faa_shape_legend.png"))

#     print("\nOpening interactive 3D windows (close each to continue)...")
#     for name, markers in results:
#         visualize(deepcopy(tomb),
#                   markers,
#                   f"{name} — FAA shapes (drag to rotate)")
#     print("Done.")




import os
from copy import deepcopy
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
OBJECT_GLB_PATH = r"C:\Users\User\Desktop\Grant\MREEG\remmbak7(A).glb"
HOLO3_CSV_PATH = r"C:\Users\User\Desktop\Grant\MREEG\Trail_Experiment\latest_HoloLens_csvraw\fixations_agtzidis_holo3_obj2.csv"
HOLO4_CSV_PATH = r"C:\Users\User\Desktop\Grant\MREEG\Trail_Experiment\latest_HoloLens_csvraw\fixations_agtzidis_holo4_obj2.csv"

OUTPUT_DIR = r"C:\Users\User\Desktop\Grant\MREEG\Trail_Experiment\latest_HoloLens_csvraw\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EEG_COLUMN = "FAA"
BROWN      = (0.757, 0.604, 0.420)

MARKER_DIVISOR  = 50.0
THICKNESS_RATIO = 0.30
SMOOTH_MESH     = True
TOMB_ALPHA      = 0.55
NORMAL_FLIP     = False

# ---- ALL SYMBOLS WHITE ----
SYMBOL_COLOR = (1.0, 1.0, 1.0)

# ---- THREE CLASSES (thirds, logical 0.8-wide bins covering the data) ----
BIN_EDGES = np.array([-1.2, -0.4, 0.4, 1.2])
BIN_WIDTH = 0.8

# bin order: [lower 1/3, middle 1/3, upper 1/3]
SHAPE_ORDER = ['x', 'triangle', 'circle']          # X = lower, △ = middle, ○ = upper
LEGEND_SYM  = {'x': 'X', 'triangle': '^', 'circle': 'o'}
# CLASS_NAME  = ['Lower 1/3', 'Middle 1/3', 'Upper 1/3']
CLASS_NAME  = ['Lower 1/3', 'Middle 1/3', 'Upper 1/3']

# ==========================================
# 🧱 TOMB PREP (keep texture detail)
# ==========================================
def _bake_texture_to_vertex_colors(mesh):
    tex = np.asarray(mesh.textures[0]).astype(np.float64)
    if tex.max() > 1.5: tex /= 255.0
    if tex.ndim == 2: tex = np.stack([tex]*3, -1)
    if tex.shape[2] == 4: tex = tex[..., :3]
    H, W, _ = tex.shape
    uvs = np.asarray(mesh.triangle_uvs); tris = np.asarray(mesh.triangles)
    u = np.clip(uvs[:,0],0,1); v = np.clip(uvs[:,1],0,1)
    px = np.clip((u*W).astype(int),0,W-1); py = np.clip(((1-v)*H).astype(int),0,H-1)
    cols = tex[py, px]; vidx = tris.reshape(-1)
    acc = np.zeros((len(mesh.vertices),3)); cnt = np.zeros(len(mesh.vertices))
    np.add.at(acc, vidx, cols); np.add.at(cnt, vidx, 1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(acc/np.maximum(cnt,1)[:,None])

def prepare_tomb(path):
    mesh = o3d.io.read_triangle_mesh(path, enable_post_processing=True)
    if not mesh.has_vertices(): raise ValueError("Mesh has no vertices.")
    try:
        if not mesh.has_vertex_colors() and mesh.has_triangle_uvs() and len(mesh.textures) > 0:
            _bake_texture_to_vertex_colors(mesh)
    except Exception as e:
        print(f"[warn] texture bake failed: {e}")
    if not mesh.has_vertex_colors(): mesh.paint_uniform_color(BROWN)
    if SMOOTH_MESH and len(mesh.triangles) < 200000:
        mesh = mesh.subdivide_loop(number_of_iterations=1)
    mesh.compute_vertex_normals()
    return mesh

# ==========================================
# 🛠️ TEMPLATES: X, triangle, circle — ALL WHITE
# ==========================================
def _rz(deg): return o3d.geometry.get_rotation_matrix_from_xyz((0,0,np.deg2rad(deg)))
def _cbox(w,h,d):
    b = o3d.geometry.TriangleMesh.create_box(width=w,height=h,depth=d)
    b.translate((-w/2,-h/2,-d/2)); return b

def build_templates(size, thickness):
    t = {}
    # X : two diagonal arms
    arm = size/3.0
    xm = _cbox(size*1.15, arm, thickness) + _cbox(arm, size*1.15, thickness)
    xm.rotate(_rz(45), center=(0,0,0))
    t['x'] = xm
    # △ : triangular prism, apex up
    tri = o3d.geometry.TriangleMesh.create_cylinder(radius=size*0.65, height=thickness, resolution=3)
    tri.rotate(_rz(90), center=(0,0,0))
    t['triangle'] = tri
    # ○ : disc
    t['circle'] = o3d.geometry.TriangleMesh.create_cylinder(radius=size*0.5, height=thickness, resolution=32)
    for key in t:
        t[key].compute_vertex_normals()
        t[key].paint_uniform_color(SYMBOL_COLOR)   # <-- WHITE everywhere
    return t

def rotation_from_normal(nrm, up=(0,0,1)):
    nrm = nrm/(np.linalg.norm(nrm)+1e-12)
    up = np.array(up,float)
    if abs(np.dot(nrm,up)) > 0.98: up = np.array([0.,1.,0.])
    x = np.cross(up,nrm); x /= (np.linalg.norm(x)+1e-12)
    y = np.cross(nrm,x)
    return np.column_stack([x,y,nrm])

# ==========================================
# 🎯 MARKERS (3 terciles, normal-oriented, white)
# ==========================================
def build_faa_markers(df, scene, tri_normals, templates, thickness):
    dfc = df.dropna(subset=['centroid_x','centroid_y','centroid_z',EEG_COLUMN]).reset_index(drop=True)
    if dfc.empty: return o3d.geometry.TriangleMesh()
    pts  = dfc[['centroid_x','centroid_y','centroid_z']].to_numpy()
    eyes = dfc[['eye_origin_x','eye_origin_y','eye_origin_z']].to_numpy()
    faa  = dfc[EEG_COLUMN].to_numpy()

    res = scene.compute_closest_points(o3d.core.Tensor(pts.astype(np.float32)))
    closest = res['points'].numpy().astype(np.float64)
    prims   = res['primitive_ids'].numpy()

    offset = thickness*0.55
    pieces = []
    for p, eye, f, cp, pid in zip(pts, eyes, faa, closest, prims):
        nrm = tri_normals[pid].copy() if 0 <= pid < len(tri_normals) else (p-cp)
        if np.linalg.norm(nrm) < 1e-9: nrm = np.array([0.,0.,1.])
        nrm /= np.linalg.norm(nrm)
        if np.dot(nrm, eye-cp) < 0: nrm = -nrm
        if NORMAL_FLIP: nrm = -nrm

        idx = int(np.clip(np.floor((f - BIN_EDGES[0]) / BIN_WIDTH), 0, 2))
        inst = deepcopy(templates[SHAPE_ORDER[idx]])
        inst.rotate(rotation_from_normal(nrm), center=(0,0,0))
        inst.translate(p + nrm*offset)
        pieces.append(inst)

    out = o3d.geometry.TriangleMesh()
    for m in pieces: out += m
    return out

# ==========================================
# 💾 LEGEND PNG (separate file; white symbols with black edge)
# ==========================================
def save_legend_png(filepath):
    handles = []
    for i in reversed(range(3)):          # upper 1/3 on top, lower 1/3 at bottom
        lo, hi = BIN_EDGES[i], BIN_EDGES[i+1]
        # lab = f"{CLASS_NAME[i]}:  {lo:+.1f} ≤ FAA < {hi:+.1f}"
        lab = f"{lo:+.1f} ≤ FAA < {hi:+.1f}"
        # lab = f"{CLASS_NAME[i]}"
        # if i == 2: lab = f"{CLASS_NAME[i]}:  {lo:+.1f} ≤ FAA ≤ {hi:+.1f}"
        if i == 2: lab = f"{lo:+.1f} ≤ FAA ≤ {hi:+.1f}"
        # if i == 2: lab = f"{CLASS_NAME[i]}"
        # lab = ""
        handles.append(mlines.Line2D([],[], color=SYMBOL_COLOR, marker=LEGEND_SYM[SHAPE_ORDER[i]],
                           linestyle='None', markersize=15,
                           markeredgecolor='black', markeredgewidth=1.5, label=lab))
    fig = plt.figure(figsize=(5.2, 2.4))
    ax = fig.add_subplot(111); ax.axis('off')
    ax.legend(handles=handles, loc='center', fontsize=13, frameon=True,
              edgecolor='black', title="FAA classes", title_fontsize=14)
    fig.savefig(filepath, bbox_inches='tight', dpi=300); plt.close(fig)
    print(f"Saved separate legend: {filepath}")

# ==========================================
# 👁️ VISUALIZATION (no legend inside the window)
# ==========================================
def visualize(tomb, markers, title):
    try:
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        mat.base_color = [1.0,1.0,1.0,TOMB_ALPHA]
        o3d.visualization.draw(
            [{"name":"tomb","geometry":tomb,"material":mat},
             {"name":"faa_shapes","geometry":markers}], title=title)
    except Exception:
        o3d.visualization.draw_geometries([tomb, markers], title=title, mesh_show_back_face=True)

# ==========================================
# 🚀 MAIN
# ==========================================
if __name__ == "__main__":
    dfs = {}
    for name, path in [("holo3", HOLO3_CSV_PATH), ("holo4", HOLO4_CSV_PATH)]:
        dfs[name] = pd.read_csv(path)
        v = dfs[name][EEG_COLUMN]
        print(f"{name}: FAA min={v.min():+.4f}, max={v.max():+.4f}")
    print("Class edges:", BIN_EDGES)

    tomb = prepare_tomb(OBJECT_GLB_PATH)
    tomb.compute_triangle_normals()
    tri_normals = np.asarray(tomb.triangle_normals)

    marker_size = np.max(tomb.get_max_bound()-tomb.get_min_bound()) / MARKER_DIVISOR
    thickness   = marker_size * THICKNESS_RATIO

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(tomb))
    templates = build_templates(marker_size, thickness)

    results = []
    for name in ["holo3", "holo4"]:
        markers = build_faa_markers(dfs[name], scene, tri_normals, templates, thickness)
        ply_out = os.path.join(OUTPUT_DIR, f"{name}_faa_shapes.ply")
        o3d.io.write_triangle_mesh(ply_out, markers)
        print(f"Saved {ply_out} ({len(markers.triangles)} tris)")
        results.append((name, markers))

    save_legend_png(os.path.join(OUTPUT_DIR, "faa_class_legend.png"))

    print("\nOpening interactive 3D windows (close each to continue)...")
    for name, markers in results:
        visualize(deepcopy(tomb), markers, f"{name} — FAA terciles (drag to rotate)")
    print("Done.")