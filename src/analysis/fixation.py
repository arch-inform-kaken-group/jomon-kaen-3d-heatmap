import os
import math
import numpy as np
import pandas as pd
import open3d as o3d
import trimesh
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# CONFIGURATION
# ==========================================

# CSV_FILE_PATH = r"C:\Users\User\Desktop\Grant\MREEG\Trail_Experiment\2026_08_04_09_27_45\UD0028(93)\pointcloud.csv"
# GLB_FILE_PATH = r"D:\storage\jomon_kaen\pottery\UD0028(93).glb"
CSV_FILE_PATH = r"C:\Users\User\Desktop\Grant\MREEG\Trail_Experiment\2026_08_04_09_27_45\IN0009(5)\pointcloud.csv"
GLB_FILE_PATH = r"D:\storage\jomon_kaen\pottery\IN0009(5).glb"
OUTPUT_DIRECTORY = "fixation_images_output"

CAMERA_UP_VECTOR = [0, 1, 0]  # Y-up, standard for Unity exports
MIN_DURATION = 0.25  # 250 ms minimum fixation duration

# I-DT
IDT_DISPERSION_THRESHOLD = 20.0

# I-VT
IVT_VELOCITY_THRESHOLD = 30.0

# Agtzidis I-S5T
AGTZIDIS_LOW_BASE_THRESHOLD = None  # Set to None to auto-detect from GLB bounds, or float to override
AGTZIDIS_DYNAMIC_THRESHOLD_MIN = 10.0
AGTZIDIS_DYNAMIC_THRESHOLD_MAX = 25.0
AGTZIDIS_SIZE_CALIBRATION = [
    (46.56, 10.0),    # Smallest object (IN0306)
    (185.02, 15.0),   # Small object (UD0028(93))
    (310.27, 25.0),   # Large object (IN0009(5))
]
AGTZIDIS_SACCADE_PEAK_THRESHOLD = 100.0
AGTZIDIS_SACCADE_SURROUND_THRESHOLD = 35.0
AGTZIDIS_HEAD_THRESHOLD = 7.0
AGTZIDIS_MAX_DYNAMIC_SCALE = 2.0
AGTZIDIS_MAX_TIME_GAP = 0.11  # seconds

# ==========================================
# 1. MATH & VECTOR UTILITIES
# ==========================================

def normalize_vectors(vectors):
    vectors = np.asarray(vectors, dtype=float)
    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)

def calculate_angular_velocities(vectors, delta_t):
    """
    Calculates angular velocity in degrees/second from successive direction vectors.
    No smoothing is applied.
    """
    vectors = np.asarray(vectors, dtype=float)
    delta_t = np.asarray(delta_t, dtype=float)
    velocities = np.zeros(len(vectors), dtype=float)
    if len(vectors) <= 1:
        return velocities
    dots = np.sum(vectors[:-1] * vectors[1:], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(dots))
    dt_next = delta_t[1:]
    velocities[1:] = np.divide(angles_deg, dt_next, out=np.zeros_like(angles_deg, dtype=float), where=dt_next > 0)
    return velocities

def build_head_relative_eye_dirs(world_eye_dirs, head_forward_world, up=(0, 1, 0)):
    """
    Approximates eye direction in a head-centered coordinate frame.
    """
    world_eye_dirs = normalize_vectors(world_eye_dirs)
    head_forward_world = normalize_vectors(head_forward_world)
    up = np.array(up, dtype=float)
    fallback_up = np.array([0, 0, 1], dtype=float)
    head_relative_dirs = np.zeros_like(world_eye_dirs)
    if len(world_eye_dirs) == 0:
        return head_relative_dirs
    for i in range(len(world_eye_dirs)):
        eye_dir = world_eye_dirs[i]
        head_forward = head_forward_world[i]
        eye_norm = np.linalg.norm(eye_dir)
        head_norm = np.linalg.norm(head_forward)
        if eye_norm < 1e-8 or head_norm < 1e-8:
            head_relative_dirs[i] = eye_dir
            continue
        temp_up = up
        if abs(np.dot(head_forward, temp_up)) > 0.98:
            temp_up = fallback_up
        right = np.cross(head_forward, temp_up)
        if np.linalg.norm(right) < 1e-8:
            temp_up = fallback_up
            right = np.cross(head_forward, temp_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-8:
            head_relative_dirs[i] = eye_dir
            continue
        right = right / right_norm
        corrected_up = np.cross(right, head_forward)
        corrected_up_norm = np.linalg.norm(corrected_up)
        if corrected_up_norm < 1e-8:
            head_relative_dirs[i] = eye_dir
            continue
        corrected_up = corrected_up / corrected_up_norm
        head_relative_dirs[i, 0] = np.dot(eye_dir, right)
        head_relative_dirs[i, 1] = np.dot(eye_dir, corrected_up)
        head_relative_dirs[i, 2] = np.dot(eye_dir, head_forward)
    return normalize_vectors(head_relative_dirs)

def choose_agtzidis_low_base_threshold(
    object_max_range,
    calibration=None,
    min_threshold=None,
    max_threshold=None
):
    """
    Chooses Agtzidis low_base_threshold from GLB bounding-box max range.
    The threshold is interpolated between calibration points, then clipped
    between AGTZIDIS_DYNAMIC_THRESHOLD_MIN and AGTZIDIS_DYNAMIC_THRESHOLD_MAX.
    """
    if calibration is None:
        calibration = AGTZIDIS_SIZE_CALIBRATION
    if min_threshold is None:
        min_threshold = AGTZIDIS_DYNAMIC_THRESHOLD_MIN
    if max_threshold is None:
        max_threshold = AGTZIDIS_DYNAMIC_THRESHOLD_MAX

    if not np.isfinite(object_max_range) or object_max_range <= 0:
        return float(min_threshold)

    calibration = sorted(calibration, key=lambda item: item[0])
    if len(calibration) == 0:
        return float(np.clip(15.0, min_threshold, max_threshold))

    ranges = np.array([item[0] for item in calibration], dtype=float)
    thresholds = np.array([item[1] for item in calibration], dtype=float)

    threshold = float(np.interp(object_max_range, ranges, thresholds))
    return float(np.clip(threshold, min_threshold, max_threshold))

# ==========================================
# 2. FIXATION DETECTION ALGORITHMS
# ==========================================

def _group_fixations(df, boolean_mask, min_duration, algo_name):
    fixations = []
    if len(df) == 0:
        return pd.DataFrame(fixations)
    boolean_mask = np.asarray(boolean_mask, dtype=bool)
    if not boolean_mask.any():
        return pd.DataFrame(fixations)
    df_temp = df.copy()
    df_temp["group"] = (~boolean_mask).cumsum()
    valid_groups = df_temp[boolean_mask].groupby("group")
    for _, group in valid_groups:
        if len(group) < 2:
            continue
        start_time = group["timestamp"].iloc[0]
        end_time = group["timestamp"].iloc[-1]
        duration = end_time - start_time
        if duration >= min_duration:
            points_str = "|".join([
                f"{px:.3f},{py:.3f},{pz:.3f}"
                for px, py, pz in zip(group["x"], group["y"], group["z"])
            ])
            fixations.append({
                "start_time": start_time, "end_time": end_time, "duration": duration,
                "device_start_time": str(group["deviceTime"].iloc[0]),
                "device_end_time": str(group["deviceTime"].iloc[-1]),
                "centroid_x": group["x"].mean(), "centroid_y": group["y"].mean(), "centroid_z": group["z"].mean(),
                "eye_origin_x": group["localEyeOriginX"].mean(),
                "eye_origin_y": group["localEyeOriginY"].mean(),
                "eye_origin_z": group["localEyeOriginZ"].mean(),
                "algorithm": algo_name,
                "gaze_points": points_str
            })
    return pd.DataFrame(fixations)

def generate_idt_fixations(df, dispersion_threshold=20.0, min_duration=0.25):
    fixations = []
    if len(df) == 0:
        return pd.DataFrame(fixations)
    coords = df[["x", "y", "z"]].to_numpy(dtype=float)
    eye_origins = df[["localEyeOriginX", "localEyeOriginY", "localEyeOriginZ"]].to_numpy(dtype=float)
    timestamps = df["timestamp"].to_numpy(dtype=float)
    i = 0
    while i < len(coords):
        window_start = i
        for j in range(window_start, len(coords)):
            window_coords = coords[window_start:j + 1]
            if len(window_coords) < 2:
                continue
            dispersion = np.sum(np.max(window_coords, axis=0) - np.min(window_coords, axis=0))
            if dispersion > dispersion_threshold:
                fixation_end = j - 1
                if fixation_end >= window_start:
                    fixation_coords = coords[window_start:fixation_end + 1]
                    fixation_eye_origins = eye_origins[window_start:fixation_end + 1]
                    duration = timestamps[fixation_end] - timestamps[window_start]
                    if duration >= min_duration and len(fixation_coords) >= 2:
                        centroid = np.mean(fixation_coords, axis=0)
                        eye_origin = np.mean(fixation_eye_origins, axis=0)
                        points_str = "|".join([f"{px:.3f},{py:.3f},{pz:.3f}" for px, py, pz in fixation_coords])
                        fixations.append({
                            "start_time": timestamps[window_start], "end_time": timestamps[fixation_end], "duration": duration,
                            "device_start_time": str(df["deviceTime"].iloc[window_start]),
                            "device_end_time": str(df["deviceTime"].iloc[fixation_end]),
                            "centroid_x": centroid[0], "centroid_y": centroid[1], "centroid_z": centroid[2],
                            "eye_origin_x": eye_origin[0], "eye_origin_y": eye_origin[1], "eye_origin_z": eye_origin[2],
                            "algorithm": "I-DT", "gaze_points": points_str
                        })
                i = j
                break
        else:
            fixation_coords = coords[window_start:]
            fixation_eye_origins = eye_origins[window_start:]
            if len(fixation_coords) >= 2:
                duration = timestamps[-1] - timestamps[window_start]
                if duration >= min_duration:
                    centroid = np.mean(fixation_coords, axis=0)
                    eye_origin = np.mean(fixation_eye_origins, axis=0)
                    points_str = "|".join([f"{px:.3f},{py:.3f},{pz:.3f}" for px, py, pz in fixation_coords])
                    fixations.append({
                        "start_time": timestamps[window_start], "end_time": timestamps[-1], "duration": duration,
                        "device_start_time": str(df["deviceTime"].iloc[window_start]),
                        "device_end_time": str(df["deviceTime"].iloc[-1]),
                        "centroid_x": centroid[0], "centroid_y": centroid[1], "centroid_z": centroid[2],
                        "eye_origin_x": eye_origin[0], "eye_origin_y": eye_origin[1], "eye_origin_z": eye_origin[2],
                        "algorithm": "I-DT", "gaze_points": points_str
                    })
            break
    return pd.DataFrame(fixations)

def generate_ivt_fixations(df, velocity_threshold=30.0, min_duration=0.25):
    if len(df) == 0:
        return pd.DataFrame()
    is_fixation = df["gaze_angular_vel"].fillna(0) < velocity_threshold
    return _group_fixations(df, is_fixation, min_duration, "I-VT")

def generate_agtzidis_I_S5T(df, min_duration=0.25, low_base_threshold=10.0,
                            saccade_peak_threshold=150.0, saccade_surround_threshold=35.0,
                            head_threshold=7.0, max_time_gap=0.10):
    fixations = []
    if len(df) == 0:
        return pd.DataFrame(fixations)

    gaze_vel = df["gaze_angular_vel"].fillna(0).to_numpy(dtype=float)
    if "local_eye_angular_vel" in df.columns:
        local_eye_vel = df["local_eye_angular_vel"].fillna(0).to_numpy(dtype=float)
    else:
        local_eye_vel = np.zeros(len(df), dtype=float)
    head_vel = df["head_angular_vel"].fillna(0).to_numpy(dtype=float)

    is_saccade_peak = gaze_vel >= saccade_peak_threshold
    is_saccade_surround = gaze_vel >= saccade_surround_threshold
    saccade_mask = np.zeros(len(df), dtype=bool)
    in_saccade = False
    has_peak = False
    start_idx = 0

    for i in range(len(df)):
        if is_saccade_surround[i]:
            if not in_saccade:
                start_idx = i
                in_saccade = True
                has_peak = False
            if is_saccade_peak[i]:
                has_peak = True
        else:
            if in_saccade:
                if has_peak:
                    saccade_mask[start_idx:i] = True
                in_saccade = False
    if in_saccade and has_peak:
        saccade_mask[start_idx:] = True

    scale = 1.0 + (head_vel / 60.0)
    low_gaze_thd = low_base_threshold * scale
    is_fixation_frame = (~saccade_mask) & (gaze_vel < low_gaze_thd)

    primary_labels = np.full(len(df), "Noise", dtype=object)
    secondary_labels = np.full(len(df), "", dtype=object)
    primary_labels[is_fixation_frame] = "Fixation"

    mask_vor = (is_fixation_frame & (head_vel > head_threshold) & (local_eye_vel > head_threshold))
    secondary_labels[mask_vor] = "VOR"

    mask_head_pursuit = (is_fixation_frame & (head_vel > head_threshold) & (local_eye_vel <= head_threshold))
    secondary_labels[mask_head_pursuit] = "Head Pursuit"

    df_temp = df.copy()
    df_temp["primary"] = primary_labels
    df_temp["secondary"] = secondary_labels

    if max_time_gap is not None and len(df_temp) > 1:
        time_diffs = df_temp["timestamp"].diff().fillna(0).to_numpy(dtype=float)
        time_gap_break = time_diffs > max_time_gap
    else:
        time_gap_break = np.zeros(len(df_temp), dtype=bool)

    df_temp["group"] = ((~is_fixation_frame) | time_gap_break).cumsum()
    valid_groups = df_temp[is_fixation_frame].groupby("group")

    for _, group in valid_groups:
        if len(group) < 2:
            continue
        start_time = group["timestamp"].iloc[0]
        end_time = group["timestamp"].iloc[-1]
        duration = end_time - start_time
        if duration < min_duration:
            continue

        prim_label = "Fixation"
        sec_labels = group["secondary"].replace("", np.nan).dropna()
        sec_label = sec_labels.mode()[0] if not sec_labels.empty else "Standard"

        if "local_eye_angular_vel" in group.columns:
            okn_velocity = group["local_eye_angular_vel"].fillna(0)
        else:
            okn_velocity = pd.Series(0.0, index=group.index)
        has_okn_spikes = (okn_velocity > 35.0).sum() > (len(group) * 0.15)

        if has_okn_spikes:
            if sec_label == "VOR":
                sec_label = "OKN+VOR"
            else:
                sec_label = "OKN"

        points_str = "|".join([f"{px:.3f},{py:.3f},{pz:.3f}" for px, py, pz in zip(group["x"], group["y"], group["z"])])

        fixations.append({
            "start_time": start_time, "end_time": end_time, "duration": duration,
            "device_start_time": str(group["deviceTime"].iloc[0]),
            "device_end_time": str(group["deviceTime"].iloc[-1]),
            "centroid_x": group["x"].mean(), "centroid_y": group["y"].mean(), "centroid_z": group["z"].mean(),
            "eye_origin_x": group["localEyeOriginX"].mean(),
            "eye_origin_y": group["localEyeOriginY"].mean(),
            "eye_origin_z": group["localEyeOriginZ"].mean(),
            "algorithm": "Agtzidis-I-S5T",
            "primary_label": prim_label,
            "secondary_label": sec_label,
            "gaze_points": points_str
        })
    return pd.DataFrame(fixations)

# ==========================================
# 3. TEXTURE BAKING & RENDERING
# ==========================================

def bake_texture_to_vertex_colors(tm_mesh):
    if isinstance(tm_mesh.visual, trimesh.visual.TextureVisuals) and tm_mesh.visual.material is not None:
        mat = tm_mesh.visual.material
        img = None
        if hasattr(mat, "baseColorTexture") and mat.baseColorTexture is not None:
            img = mat.baseColorTexture
        elif hasattr(mat, "image") and mat.image is not None:
            img = mat.image
        if img is not None:
            img_np = np.array(img.convert("RGB"))
            h, w, _ = img_np.shape
            uvs = tm_mesh.visual.uv
            if uvs is not None and len(uvs) > 0:
                u_px = (uvs[:, 0] * (w - 1)).astype(int)
                v_px = ((1.0 - uvs[:, 1]) * (h - 1)).astype(int)
                u_px = np.clip(u_px, 0, w - 1)
                v_px = np.clip(v_px, 0, h - 1)
                vertex_colors = img_np[v_px, u_px] / 255.0
                return vertex_colors
    return np.ones((len(tm_mesh.vertices), 3)) * 0.8

def load_and_bake_mesh(glb_path):
    print(f"Loading and baking textures from {glb_path}...")
    scene = trimesh.load(glb_path)
    if isinstance(scene, trimesh.Scene):
        meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("No valid meshes found.")
        tm_mesh = meshes[0]
    else:
        tm_mesh = scene
    v_colors = bake_texture_to_vertex_colors(tm_mesh)
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(tm_mesh.vertices),
        triangles=o3d.utility.Vector3iVector(tm_mesh.faces))
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(v_colors)
    return o3d_mesh

def get_outward_surface_normal(centroid, kdtree, pcd, eye_origin, search_radius):
    [k, idx, _] = kdtree.search_radius_vector_3d(centroid, search_radius)
    if k < 3:
        [k2, idx2, _] = kdtree.search_knn_vector_3d(centroid, 1)
        if k2 > 0:
            normal = np.asarray(pcd.normals)[idx2[0]]
        else:
            normal = np.array([0.0, 1.0, 0.0])
    else:
        pts = np.asarray(pcd.points)[idx]
        pts_centered = pts - np.mean(pts, axis=0)
        cov = np.cov(pts_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        normal = eigenvectors[:, 0]
    view_vec = eye_origin - centroid
    view_vec_norm = np.linalg.norm(view_vec)
    if view_vec_norm > 0:
        view_vec /= view_vec_norm
    if np.dot(normal, view_vec) < 0:
        normal = -normal
    return normal

def render_fixation_images(fixations_df, o3d_mesh, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    min_bound = o3d_mesh.get_min_bound()
    max_bound = o3d_mesh.get_max_bound()
    max_range = np.max(max_bound - min_bound)
    dynamic_roi_radius = max_range * 0.10
    plane_fit_radius = max_range * 0.05
    fixations_df_sorted = fixations_df.sort_values(by="start_time").reset_index(drop=True)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=800, visible=False)
    try:
        render_option = vis.get_render_option()
        if render_option is not None:
            render_option.point_size = 8.0
    except Exception:
        pass
    vis.add_geometry(o3d_mesh)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d_mesh.vertices
    pcd.normals = o3d_mesh.vertex_normals
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    saved_image_paths = []
    try:
        font = ImageFont.truetype("arial.ttf", 32)
        small_font = ImageFont.truetype("arial.ttf", 24)
        collage_font = ImageFont.truetype("arial.ttf", 48)
        collage_small_font = ImageFont.truetype("arial.ttf", 32)
    except Exception:
        font = ImageFont.load_default()
        small_font = font
        collage_font = font
        collage_small_font = font
    text_box_height = 160
    for idx, row in fixations_df_sorted.iterrows():
        print(f"Rendering Fixation {idx + 1}/{len(fixations_df_sorted)}...")
        vis.clear_geometries()
        vis.add_geometry(o3d_mesh, reset_bounding_box=False)
        centroid = np.array([row["centroid_x"], row["centroid_y"], row["centroid_z"]], dtype=float)
        eye_origin = np.array([row["eye_origin_x"], row["eye_origin_y"], row["eye_origin_z"]], dtype=float)
        view_vec = eye_origin - centroid
        view_norm = np.linalg.norm(view_vec)
        if view_norm > 0:
            view_dir = view_vec / view_norm
        else:
            view_dir = np.array([0.0, 1.0, 0.0])
        sphere_radius = dynamic_roi_radius * 0.15
        sphere_pos = centroid + view_dir * (sphere_radius * 0.8)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=20)
        sphere.translate(sphere_pos)
        sphere.paint_uniform_color([1, 0, 0])
        sphere.compute_vertex_normals()
        vis.add_geometry(sphere, reset_bounding_box=False)
        if "gaze_points" in row.index and pd.notna(row["gaze_points"]):
            pts = []
            for pt_str in str(row["gaze_points"]).split("|"):
                coords = pt_str.split(",")
                if len(coords) == 3:
                    try:
                        pts.append([float(coords[0]), float(coords[1]), float(coords[2])])
                    except ValueError:
                        pass
            if pts:
                pcd_hits = o3d.geometry.PointCloud()
                pcd_hits.points = o3d.utility.Vector3dVector(np.array(pts))
                pcd_hits.paint_uniform_color([0.0, 0.0, 1.0])
                vis.add_geometry(pcd_hits, reset_bounding_box=False)
        ctr = vis.get_view_control()
        forward = centroid - eye_origin
        norm = np.linalg.norm(forward)
        if norm > 0:
            forward /= norm
        else:
            forward = np.array([0, 0, 1])
        ctr.set_lookat(centroid)
        ctr.set_front(-forward)
        ctr.set_up(CAMERA_UP_VECTOR)
        ctr.set_zoom(0.5)
        vis.poll_events()
        vis.update_renderer()
        img_fp_buf = vis.capture_screen_float_buffer(do_render=True)
        img_fp = Image.fromarray((np.asarray(img_fp_buf) * 255).astype(np.uint8))
        gaze_line = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([eye_origin, sphere_pos]),
            lines=o3d.utility.Vector2iVector([[0, 1]]))
        gaze_line.colors = o3d.utility.Vector3dVector([[0, 0.8, 0]])
        vis.add_geometry(gaze_line, reset_bounding_box=False)
        outward_normal = get_outward_surface_normal(centroid, kdtree, pcd, eye_origin, plane_fit_radius)
        if np.dot(outward_normal, view_dir) < 0.5:
            cam_dir = view_dir
        else:
            cam_dir = outward_normal
        cam_pos = centroid + cam_dir * (max_range * 2.5)
        front_vec = cam_pos - centroid
        front_vec /= np.linalg.norm(front_vec)
        up = np.array(CAMERA_UP_VECTOR, dtype=float)
        if abs(np.dot(front_vec, up)) > 0.99:
            up = np.array([1.0, 0.0, 0.0])
        ctr.set_lookat(centroid)
        ctr.set_front(front_vec)
        ctr.set_up(up)
        ctr.set_zoom(0.75)
        vis.poll_events()
        vis.update_renderer()
        img_tp_buf = vis.capture_screen_float_buffer(do_render=True)
        img_tp = Image.fromarray((np.asarray(img_tp_buf) * 255).astype(np.uint8))
        combined_width = img_fp.width + img_tp.width
        combined_height = img_fp.height + text_box_height
        combined_img = Image.new("RGB", (combined_width, combined_height), (255, 255, 255))
        combined_img.paste(img_fp, (0, 0))
        combined_img.paste(img_tp, (img_fp.width, 0))
        draw = ImageDraw.Draw(combined_img)
        draw.rectangle([0, 0, 260, 45], fill=(40, 40, 40))
        draw.text((15, 8), "First-Person View", fill=(255, 255, 255), font=small_font)
        draw.rectangle([img_fp.width, 0, img_fp.width + 260, 45], fill=(40, 40, 40))
        draw.text((img_fp.width + 15, 8), "Third-Person View", fill=(255, 255, 255), font=small_font)
        safe_start = str(row["device_start_time"]).replace(":", "-").replace(".", "-")
        safe_end = str(row["device_end_time"]).replace(":", "-").replace(".", "-")
        prim_label = row.get("primary_label", "Fixation")
        sec_label = row.get("secondary_label", "")
        text_info = (
            f"Algorithm: {row['algorithm']}   |   Primary: {prim_label}\n"
            f"Duration: {row['duration']:.3f}s   |   Time: {safe_start} -> {safe_end}\n"
            f"Centroid: [{centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}]"
        )
        if sec_label and sec_label != "Standard":
            text_info += f"\nSecondary: {sec_label}"
        text_y = img_fp.height + 25
        draw.multiline_text((20, text_y), text_info, fill=(0, 0, 0), font=font)
        safe_algo = str(row["algorithm"]).replace(" ", "_")
        filename = f"{safe_algo}_{idx + 1}_{row['duration']:.3f}s_{safe_start}-{safe_end}.png"
        if sec_label and sec_label != "Standard":
            safe_sec = str(sec_label).replace("+", "_").replace(" ", "_")
            filename = f"{safe_algo}_{idx + 1}_{prim_label}_{safe_sec}_{row['duration']:.3f}s_{safe_start}-{safe_end}.png"
        img_path = os.path.join(output_dir, filename)
        combined_img.save(img_path)
        saved_image_paths.append(img_path)
    vis.destroy_window()
    print(f"Individual images saved to {output_dir}.")

    if saved_image_paths:
        print("Stitching collages...")
        images = [Image.open(p) for p in saved_image_paths]
        images_per_row = 4
        rows_per_collage = 4
        images_per_collage = images_per_row * rows_per_collage
        eeg_space_height = 400
        timeline_height = 120
        row_padding = 40
        img_w, img_h = images[0].size
        collage_width = images_per_row * img_w
        row_height = eeg_space_height + timeline_height + img_h + row_padding
        num_collages = math.ceil(len(images) / images_per_collage)
        for c_idx in range(num_collages):
            start_img_idx = c_idx * images_per_collage
            end_img_idx = min(start_img_idx + images_per_collage, len(images))
            current_images = images[start_img_idx:end_img_idx]
            current_rows_data = fixations_df_sorted.iloc[start_img_idx:end_img_idx]
            num_rows = math.ceil(len(current_images) / images_per_row)
            collage_height = num_rows * row_height
            collage = Image.new("RGB", (collage_width, collage_height), (245, 245, 245))
            draw_collage = ImageDraw.Draw(collage)
            for r in range(num_rows):
                y_offset = r * row_height
                draw_collage.rectangle([0, y_offset, collage_width, y_offset + eeg_space_height], fill=(250, 250, 250), outline=(220, 220, 220))
                draw_collage.text((30, y_offset + 30), "EEG Visualization Space", fill=(180, 180, 180), font=collage_font)
                timeline_y = y_offset + eeg_space_height
                draw_collage.line([(0, timeline_y + timeline_height // 2), (collage_width, timeline_y + timeline_height // 2)], fill=(150, 150, 150), width=5)
                for c in range(images_per_row):
                    img_idx = r * images_per_row + c
                    if img_idx < len(current_images):
                        row_data = current_rows_data.iloc[img_idx]
                        x_center = c * img_w + img_w // 2
                        draw_collage.line([(x_center, timeline_y + 15), (x_center, timeline_y + timeline_height - 15)], fill=(100, 100, 100), width=4)
                        time_str = f"{row_data['start_time']:.2f}s - {row_data['end_time']:.2f}s"
                        prim_label = row_data.get("primary_label", "")
                        sec_label = row_data.get("secondary_label", "")
                        if prim_label:
                            time_str += f"\n({prim_label}"
                            if sec_label and sec_label != "Standard":
                                time_str += f" + {sec_label}"
                            time_str += ")"
                        try:
                            bbox = draw_collage.multiline_textbbox((0, 0), time_str, font=collage_small_font, align="center")
                        except AttributeError:
                            bbox = draw_collage.textbbox((0, 0), time_str, font=collage_small_font)
                        text_w = bbox[2] - bbox[0]
                        draw_collage.multiline_text((x_center - text_w // 2, timeline_y + timeline_height // 2 + 10), time_str, fill=(50, 50, 50), font=collage_small_font, align="center")
                img_y_offset = timeline_y + timeline_height
                for c in range(images_per_row):
                    img_idx = r * images_per_row + c
                    if img_idx < len(current_images):
                        img = current_images[img_idx]
                        x_offset = c * img_w
                        collage.paste(img, (x_offset, img_y_offset))
                for c in range(1, images_per_row):
                    x_line = c * img_w
                    draw_collage.line([(x_line, y_offset), (x_line, y_offset + row_height)], fill=(0, 0, 0), width=5)
                if r < num_rows - 1:
                    line_y = img_y_offset + img_h + (row_padding // 2)
                    draw_collage.line([(0, line_y), (collage_width, line_y)], fill=(0, 0, 0), width=5)
            collage_path = os.path.join(output_dir, f"fixation_collage_{c_idx + 1}.png")
            collage.save(collage_path)
            print(f"Collage {c_idx + 1} saved to: {collage_path}")

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

def process_eye_tracking_data(csv_path, glb_path, output_dir):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not os.path.exists(glb_path):
        raise FileNotFoundError(f"3D Model file not found: {glb_path}")

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError("CSV file is empty.")

    df["delta_t"] = df["timestamp"].diff().fillna(0)
    median_dt = df["delta_t"].median()
    if median_dt > 0:
        print(f"Estimated sample rate: {1.0 / median_dt:.2f} Hz")

    local_eye_cols = ["localEyeDirX", "localEyeDirY", "localEyeDirZ"]
    local_head_cols = ["localHeadForwardX", "localHeadForwardY", "localHeadForwardZ"]
    world_eye_cols = ["eyeDirectionX", "eyeDirectionY", "eyeDirectionZ"]
    world_head_cols = ["headForwardX", "headForwardY", "headForwardZ"]

    has_local_eye = all(c in df.columns for c in local_eye_cols)
    has_local_head = all(c in df.columns for c in local_head_cols)
    has_world_eye = all(c in df.columns for c in world_eye_cols)
    has_world_head = all(c in df.columns for c in world_head_cols)

    local_eye_dirs = None
    world_eye_dirs = None
    local_head_dirs = None
    world_head_dirs = None

    if has_local_eye:
        local_eye_dirs = normalize_vectors(df[local_eye_cols].to_numpy(dtype=float))
    if has_world_eye:
        world_eye_dirs = normalize_vectors(df[world_eye_cols].to_numpy(dtype=float))
    if has_local_head:
        local_head_dirs = normalize_vectors(df[local_head_cols].to_numpy(dtype=float))
    if has_world_head:
        world_head_dirs = normalize_vectors(df[world_head_cols].to_numpy(dtype=float))

    if local_eye_dirs is not None:
        gaze_dirs = local_eye_dirs
        ivt_gaze_source = "localEyeDir"
    elif world_eye_dirs is not None:
        gaze_dirs = world_eye_dirs
        ivt_gaze_source = "eyeDirection"
    else:
        raise ValueError("No usable gaze direction columns found.")

    if local_head_dirs is not None:
        head_dirs = local_head_dirs
        head_source = "localHeadForward"
    elif world_head_dirs is not None:
        head_dirs = world_head_dirs
        head_source = "headForward"
    else:
        raise ValueError("No usable head direction columns found.")

    dt = df["delta_t"].to_numpy(dtype=float)

    df["gaze_angular_vel"] = calculate_angular_velocities(gaze_dirs, dt)
    df["head_angular_vel"] = calculate_angular_velocities(head_dirs, dt)

    if world_eye_dirs is not None:
        df["world_gaze_angular_vel"] = calculate_angular_velocities(world_eye_dirs, dt)
        agtzidis_gaze_source = "eyeDirection"
    else:
        df["world_gaze_angular_vel"] = df["gaze_angular_vel"]
        agtzidis_gaze_source = ivt_gaze_source

    if world_head_dirs is not None:
        df["world_head_angular_vel"] = calculate_angular_velocities(world_head_dirs, dt)
        agtzidis_head_source = "headForward"
    else:
        df["world_head_angular_vel"] = df["head_angular_vel"]
        agtzidis_head_source = head_source

    if world_eye_dirs is not None and world_head_dirs is not None:
        eye_for_head = world_eye_dirs
        head_for_head = world_head_dirs
        eye_in_head_source = "world"
    elif local_eye_dirs is not None and local_head_dirs is not None:
        eye_for_head = local_eye_dirs
        head_for_head = local_head_dirs
        eye_in_head_source = "local"
    else:
        eye_for_head = gaze_dirs
        head_for_head = head_dirs
        eye_in_head_source = "fallback"

    eye_dirs_head = build_head_relative_eye_dirs(eye_for_head, head_for_head, up=CAMERA_UP_VECTOR)
    df["local_eye_angular_vel"] = calculate_angular_velocities(eye_dirs_head, dt)

    print(f"I-VT gaze velocity source: {ivt_gaze_source}")
    print(f"Agtzidis gaze velocity source: {agtzidis_gaze_source}")
    print(f"Agtzidis head velocity source: {agtzidis_head_source}")
    print(f"Eye-in-head approximation source: {eye_in_head_source}")
    print("Velocity smoothing: disabled")

    print("Loading and baking 3D mesh...")
    o3d_mesh = load_and_bake_mesh(glb_path)

    # Determine object size for Agtzidis dynamic threshold
    object_max_range = float(np.max(o3d_mesh.get_max_bound() - o3d_mesh.get_min_bound()))
    if AGTZIDIS_LOW_BASE_THRESHOLD is None:
        agtzidis_low_base = choose_agtzidis_low_base_threshold(object_max_range)
        print(f"GLB max range: {object_max_range:.2f}")
        print(f"Auto-selected Agtzidis low-base threshold: {agtzidis_low_base:.2f}")
    else:
        agtzidis_low_base = float(AGTZIDIS_LOW_BASE_THRESHOLD)
        print(f"Using manual Agtzidis low-base threshold: {agtzidis_low_base:.2f}")

    print("Calculating I-DT Fixations...")
    idt_df = generate_idt_fixations(df, dispersion_threshold=IDT_DISPERSION_THRESHOLD, min_duration=MIN_DURATION)

    print("Calculating I-VT Fixations...")
    ivt_df = generate_ivt_fixations(df, velocity_threshold=IVT_VELOCITY_THRESHOLD, min_duration=MIN_DURATION)

    print("Calculating Agtzidis I-S5T Head-Eye Coupling Fixations...")
    agtzidis_df = generate_agtzidis_I_S5T(
        df,
        min_duration=MIN_DURATION,
        low_base_threshold=agtzidis_low_base,
        saccade_peak_threshold=AGTZIDIS_SACCADE_PEAK_THRESHOLD,
        saccade_surround_threshold=AGTZIDIS_SACCADE_SURROUND_THRESHOLD,
        head_threshold=AGTZIDIS_HEAD_THRESHOLD,
        max_time_gap=AGTZIDIS_MAX_TIME_GAP)

    print("\n--- Detection Summary ---")
    print(f"I-DT Fixations Found: {len(idt_df)}")
    print(f"I-VT Fixations Found: {len(ivt_df)}")
    print(f"Agtzidis I-S5T Events Found: {len(agtzidis_df)}\n")

    os.makedirs(output_dir, exist_ok=True)
    idt_df.to_csv(os.path.join(output_dir, "fixations_idt.csv"), index=False)
    ivt_df.to_csv(os.path.join(output_dir, "fixations_ivt.csv"), index=False)
    agtzidis_df.to_csv(os.path.join(output_dir, "fixations_agtzidis.csv"), index=False)
    print("Saved fixation CSVs.")

    algorithms_to_run = [("I-DT", idt_df), ("I-VT", ivt_df), ("Agtzidis-I-S5T", agtzidis_df)]

    for algo_name, algo_df in algorithms_to_run:
        if algo_df.empty:
            print(f"\nNo events detected for {algo_name}. Skipping visualization.")
            continue
        algo_output_dir = os.path.join(output_dir, algo_name)
        print(f"\n--- Starting 3D Visualization Pipeline for {algo_name} ---")
        render_fixation_images(fixations_df=algo_df, o3d_mesh=o3d_mesh, output_dir=algo_output_dir)

if __name__ == "__main__":
    process_eye_tracking_data(CSV_FILE_PATH, GLB_FILE_PATH, OUTPUT_DIRECTORY)