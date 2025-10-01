import os
import sys
import io
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
import concurrent.futures

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import japanize_matplotlib
from wordcloud import WordCloud
import open3d as o3d
import trimesh
from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy

from scipy.spatial.distance import jensenshannon
from sudachipy import tokenizer as sudachi_tokenizer
from sudachipy import dictionary as sudachi_dictionary
from tqdm import tqdm

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, HRFlowable, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from svglib.svglib import svg2rlg

try:
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_3D_AVAILABLE = True
except ImportError:
    MATPLOTLIB_3D_AVAILABLE = False

# SHARED CONSTANTS

ASSIGNED_NUMBERS_DICT = {
    'AS0001': '1', 'FH0008': '2', 'IN0003': '3', 'IN0008': '4',
    'IN0009': '5', 'IN0017': '6', 'IN0081': '7', 'IN0104': '8',
    'IN0135': '9', 'IN0148': '10', 'IN0220': '11', 'IN0228': '12',
    'IN0232': '13', 'IN0239': '14', 'IN0277': '15', 'MY0001': '16',
    'MY0002': '17', 'MY0004': '18', 'MY0006': '19', 'MY0007': '20',
    'ND0001': '21', 'NM0001': '22', 'NM0002': '23', 'NM0009': '24',
    'NM0010': '25', 'NM0014': '26', 'NM0015': '27', 'NM0017': '28',
    'NM0041': '29', 'NM0049': '30', 'NM0066': '31', 'NM0070': '32',
    'NM0072': '33', 'NM0073': '34', 'NM0079': '35', 'NM0080': '36',
    'NM0099': '37', 'NM0106': '38', 'NM0133': '39', 'NM0135': '40',
    'NM0144': '41', 'NM0154': '42', 'NM0156': '43', 'NM0159': '44',
    'NM0168': '45', 'NM0173': '46', 'NM0175': '47', 'NM0189': '48',
    'NM0191': '49', 'NM0206': '50', 'SB0002': '51', 'SB0004': '52',
    'SI0001': '53', 'SJ0503': '54', 'SJ0504': '55', 'SK0001': '56',
    'SK0002': '57', 'SK0003': '58', 'SK0004': '59', 'SK0005': '60',
    'SK0013': '61', 'SS0001': '62', 'TJ0004': '63', 'TJ0005': '64',
    'TJ0010': '65', 'TK0002': '66', 'TK0048': '67', 'TK0057': '68',
    'UD0001': '69', 'UD0003': '70', 'UD0005': '71', 'UD0006': '72',
    'UD0011': '73', 'UD0013': '74', 'UD0014': '75', 'UD0016': '76',
    'UD0023': '77', 'UD0302': '78', 'UD0304': '79', 'UD0308': '80',
    'UD0318': '81', 'UD0322': '82', 'UD0411': '83', 'UD0412': '84',
    'UK0001': '85', 'IN0295': '86', 'IN0306': '87', 'MH0037': '88',
    'NM0239': '89', 'NZ0001': '90', 'SK0035': '91', 'TK0020': '92',
    'UD0028': '93'
}

EMOTION_MAPS = {
    'japan': {
        "full_map": {
            "面白い・気になる形だ": "面白い", "美しい・芸術的だ": "美しい",
            "不思議・意味不明": "不思議", "不気味・不安・怖い": "怖い",
            "何も感じない": "何も感じない", "NO RESPONSE": "NO RESPONSE"
        },
        "colors": {
            "面白い・気になる形だ": "#00FFFF", "美しい・芸術的だ": "#00FF00",
            "不思議・意味不明": "#FFFF00", "不気味・不安・怖い": "#FF0000",
            "何も感じない": "#505050", "NO RESPONSE": "#D3D3D3"
        },
        "target_labels": ["面白い", "美しい", "不思議", "怖い", "何も感じない"]
    },
    'malaysia': {
        "full_map": {
            "Interesting and attentional shape": "Interesting", "Beautiful and artistic": "Beautiful",
            "Strange and incomprehensible": "Strange", "Creepy / unsettling / scary": "Scary",
            "Feel nothing": "Feel nothing", "NO RESPONSE": "NO RESPONSE"
        },
        "colors": {
            "Interesting and attentional shape": "#00FFFF", "Beautiful and artistic": "#00FF00",
            "Strange and incomprehensible": "#FFFF00", "Creepy / unsettling / scary": "#FF0000",
            "Feel nothing": "#505050", "NO RESPONSE": "#D3D3D3"
        },
        "target_labels": ["Interesting", "Beautiful", "Strange", "Scary", "Feel nothing"]
    }
}


# DATA LOADING & PREPARATION

def get_pottery_id_list():
    """Returns a list of formatted pottery IDs like 'AS0001(1)'."""
    return [f"{pid}({num})" for pid, num in ASSIGNED_NUMBERS_DICT.items()]

def group_data_by_pottery(root_dir: str) -> dict:
    """Finds and groups all pointcloud and model files by pottery ID."""
    print(f"Scanning for gaze data in '{root_dir}'...")
    data_paths = defaultdict(list)
    if not os.path.exists(root_dir):
        print(f"Warning: Directory not found: {root_dir}")
        return data_paths

    for group_folder in os.listdir(root_dir):
        group_path = os.path.join(root_dir, group_folder)
        if not os.path.isdir(group_path): continue
        for session_folder in os.listdir(group_path):
            session_path = os.path.join(group_path, session_folder)
            if not os.path.isdir(session_path): continue
            for pottery_folder in os.listdir(session_path):
                pottery_path = os.path.join(session_path, pottery_folder)
                if not os.path.isdir(pottery_path): continue
                pointcloud_file = os.path.join(pottery_path, "pointcloud.csv")
                model_file = os.path.join(pottery_path, "model.obj")
                if os.path.exists(pointcloud_file) and os.path.exists(model_file):
                    data_paths[pottery_folder].append({
                        'pointcloud': pointcloud_file,
                        'model': model_file
                    })
    print(f"Found data for {len(data_paths)} unique pottery IDs.")
    return data_paths

def load_transcripts(root_dir: str):
    """Loads all transcripts from a directory into a dictionary."""
    print(f"Loading transcripts from '{root_dir}'...")
    transcripts_dict = {}
    all_pottery_ids = set()

    if not os.path.exists(root_dir):
        print(f"Error: Transcript directory not found: {root_dir}")
        return {}, []

    for group_folder in os.listdir(root_dir):
        group_path = os.path.join(root_dir, group_folder)
        if not os.path.isdir(group_path): continue
        for session_folder in os.listdir(group_path):
            session_path = os.path.join(group_path, session_folder)
            if not os.path.isdir(session_path): continue
            for pottery_folder in os.listdir(session_path):
                pottery_path = os.path.join(session_path, pottery_folder)
                transcript_path = os.path.join(pottery_path, "final_transcript.txt")
                if os.path.exists(transcript_path):
                    try:
                        with open(transcript_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if content.strip():
                                key = (pottery_folder, session_folder)
                                transcripts_dict[key] = content
                                all_pottery_ids.add(pottery_folder)
                    except Exception as e:
                        print(f"Warning: Could not read {transcript_path}: {e}")

    print(f"Loaded {len(transcripts_dict)} non-empty transcripts.")
    return transcripts_dict, sorted(list(all_pottery_ids))

def load_combined_qna_data(root_dir: str) -> pd.DataFrame:
    """Loads and combines all qa_corrected.csv files into a single DataFrame."""
    print(f"Loading all Q&A data from '{root_dir}'...")
    df_list = []
    if not os.path.exists(root_dir):
        print(f"Error: Q&A data directory not found: {root_dir}")
        return pd.DataFrame()

    for group_folder in os.listdir(root_dir):
        group_path = os.path.join(root_dir, group_folder)
        if not os.path.isdir(group_path): continue
        for session_folder in os.listdir(group_path):
            session_path = os.path.join(group_path, session_folder)
            if not os.path.isdir(session_path): continue
            for pottery_folder in os.listdir(session_path):
                pottery_path = os.path.join(session_path, pottery_folder)
                qa_path = os.path.join(pottery_path, "qa_corrected.csv")
                if os.path.exists(qa_path):
                    try:
                        temp_df = pd.read_csv(qa_path, header=0, sep=",")
                        temp_df['pottery_id'] = pottery_folder
                        temp_df['session_id'] = session_folder
                        df_list.append(temp_df)
                    except Exception as e:
                        print(f"Warning: Could not process {qa_path}: {e}")

    if not df_list:
        return pd.DataFrame()

    return pd.concat(df_list, ignore_index=True)

def load_alignment_data(root_dir: str):
    """Loads paths for both QA and transcript files for alignment analysis."""
    print(f"Scanning for alignment data (QA & transcript) in '{root_dir}'...")
    data_paths = []
    if not os.path.exists(root_dir):
        print(f"Error: Directory not found: {root_dir}")
        return []

    for group_folder in os.listdir(root_dir):
        group_path = os.path.join(root_dir, group_folder)
        if not os.path.isdir(group_path): continue
        for session_folder in os.listdir(group_path):
            session_path = os.path.join(group_path, session_folder)
            if not os.path.isdir(session_path): continue
            for pottery_folder in os.listdir(session_path):
                pottery_path = os.path.join(session_path, pottery_folder)
                qa_path = os.path.join(pottery_path, "qa_corrected.csv")
                transcript_path = os.path.join(pottery_path, "final_transcript.txt")

                if os.path.exists(qa_path) and os.path.exists(transcript_path):
                    with open(transcript_path, 'r', encoding='utf-8') as f:
                        if f.read().strip():
                            data_paths.append({
                                'QA': str(qa_path),
                                'TRANSCRIPT': str(transcript_path),
                                'SESSION_ID': session_folder,
                                'ID': pottery_folder
                            })
    print(f"Found {len(data_paths)} valid pairs of QA and transcript files.")
    return data_paths

def read_ply_vertex_count(file_path: str) -> int:
    """Reads the vertex count from a PLY file header."""
    try:
        with open(file_path, 'r', errors='ignore') as f:
            for line in f:
                clean_line = line.strip()
                if clean_line.startswith('element vertex'):
                    return int(clean_line.split()[2])
                if clean_line == 'end_header':
                    break
    except (IOError, IndexError, ValueError):
        return 0
    return 0

# CORE CALCULATIONS & METRICS

def calculate_jensen_shannon_distance(vec1, vec2):
    """Calculates the Jensen-Shannon Distance (JSD) between two vectors."""
    sum1, sum2 = np.sum(vec1), np.sum(vec2)
    if sum1 == 0 and sum2 == 0: return 0.0
    if sum1 == 0 or sum2 == 0: return 1.0
    return jensenshannon(vec1, vec2, base=2)

def calculate_qa_emotion_percentages(data_paths, language):
    """Calculates emotion percentages from QA event files."""
    print("Calculating emotion percentages from QA event counts...")
    emotion_map = EMOTION_MAPS[language]['full_map']
    target_labels = EMOTION_MAPS[language]['target_labels']

    for data_path in tqdm(data_paths, desc="Processing QA files"):
        try:
            df = pd.read_csv(data_path['QA'])
            df['label'] = df['answer'].str.strip().map(emotion_map)
            counts = df['label'].value_counts()
            total = counts.sum()

            percentages = {}
            if total > 0:
                for label in target_labels:
                    percentages[label] = (counts.get(label, 0) / total) * 100
            else:
                percentages = {label: 0.0 for label in target_labels}
            data_path['qa_percentages'] = percentages
        except Exception as e:
            print(f"Warning: Could not process {data_path['QA']}. Error: {e}")
            data_path['qa_percentages'] = {label: 0.0 for label in target_labels}
    return data_paths

# 3D MODEL & IMAGE GENERATION

def create_simple_pottery_image(
    pottery_id: str, image_size: tuple = (256, 256)) -> Image.Image:
    """Creates a simple placeholder image with pottery information when 3D rendering fails."""
    img = Image.new('RGB', image_size, color=(240, 245, 250))
    draw = ImageDraw.Draw(img)
    center_x, center_y = image_size[0] // 2, image_size[1] // 2
    base_width, base_height = image_size[0] // 3, image_size[1] // 6
    draw.ellipse([
        center_x - base_width // 2, center_y + image_size[1] // 4 -
        base_height // 2, center_x + base_width // 2,
        center_y + image_size[1] // 4 + base_height // 2
    ],
                 fill=(120, 120, 120),
                 outline=(80, 80, 80))
    body_width, body_height = image_size[0] // 4, image_size[1] // 3
    draw.ellipse([
        center_x - body_width // 2, center_y - body_height // 2,
        center_x + body_width // 2, center_y + body_height // 2
    ],
                 fill=(150, 150, 150),
                 outline=(100, 100, 100))
    neck_width, neck_height = image_size[0] // 6, image_size[1] // 8
    draw.ellipse([
        center_x - neck_width // 2, center_y - image_size[1] // 4 -
        neck_height // 2, center_x + neck_width // 2,
        center_y - image_size[1] // 4 + neck_height // 2
    ],
                 fill=(130, 130, 130),
                 outline=(90, 90, 90))
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    text_bbox = draw.textbbox((0, 0), pottery_id, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[
        3] - text_bbox[1]
    text_x, text_y = (image_size[0] -
                      text_width) // 2, image_size[1] - text_height - 15
    draw.rectangle([
        text_x - 5, text_y - 3, text_x + text_width + 5,
        text_y + text_height + 3
    ],
                   fill=(255, 255, 255, 200),
                   outline=(100, 100, 100))
    draw.text((text_x, text_y), pottery_id, fill=(0, 0, 0), font=font)
    try:
        small_font = ImageFont.truetype("arial.ttf", 12)
    except:
        small_font = ImageFont.load_default()
    draw.text((10, 10), "3D Model", fill=(100, 100, 100), font=small_font)
    return img

def render_glb_matplotlib(
    glb_path: str, output_size: tuple = (512, 512)) -> np.ndarray:
    """Alternative renderer using matplotlib for 3D visualization."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        fig = plt.figure(figsize=(output_size[0] / 100, output_size[1] / 100),
                         dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        mesh = trimesh.load(glb_path)
        if hasattr(mesh, 'geometry'):
            if len(mesh.geometry) == 0: raise ValueError("No geometry found")
            mesh = list(mesh.geometry.values())[0]
        rotation_matrix = trimesh.transformations.rotation_matrix(
            angle=np.pi / 2, direction=[1, 0, 0], point=[0, 0, 0])
        mesh.apply_transform(rotation_matrix)
        vertices, faces = mesh.vertices, mesh.faces
        ax.plot_trisurf(vertices[:, 0],
                        vertices[:, 1],
                        vertices[:, 2],
                        triangles=faces,
                        alpha=0.9,
                        cmap='copper',
                        linewidth=0,
                        antialiased=True)
        ax.view_init(elev=10, azim=0)
        max_range = np.array(
            [vertices[:, i].max() - vertices[:, i].min()
             for i in range(3)]).max() / 2.0
        mid = np.mean(vertices, axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        ax.set_axis_off()
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)
        return buf
    except Exception as e:
        print(f"Error with matplotlib rendering {glb_path}: {e}")
        return np.ones((*output_size[::-1], 3), dtype=np.uint8) * 180

def render_glb_front_view(
    glb_path: str, output_size: tuple = (512, 512)) -> np.ndarray:
    """Renders a front view of a .glb file and returns it as a numpy array."""
    if MATPLOTLIB_3D_AVAILABLE:
        return render_glb_matplotlib(glb_path, output_size)
    print(
        f"All rendering methods failed for {glb_path}, creating artistic placeholder"
    )
    try:
        pottery_id = os.path.basename(glb_path).split('.')[0]
        return np.array(create_simple_pottery_image(pottery_id, output_size))
    except:
        return np.ones((*output_size[::-1], 3), dtype=np.uint8) * 200

def create_cluster_collage(pottery_ids: list,
                           pottery_dir: str,
                           cluster_id: int,
                           output_dir: str,
                           image_size: tuple = (256, 256),
                           collage_columns: int = None) -> str:
    """Creates a collage of pottery models for a given cluster."""
    print(
        f"Creating collage for Cluster {cluster_id} with {len(pottery_ids)} items..."
    )
    num_items = len(pottery_ids)
    if collage_columns is None:
        collage_columns = min(5, int(math.ceil(math.sqrt(num_items))))
    collage_rows = int(math.ceil(num_items / collage_columns))
    collage = Image.new(
        'RGB', (collage_columns * image_size[0], collage_rows * image_size[1]),
        color=(240, 240, 240))
    rendered_count = 0
    for idx, pottery_id in enumerate(
            tqdm(pottery_ids, desc=f"Cluster {cluster_id}")):
        row, col = idx // collage_columns, idx % collage_columns
        glb_files = [
            f for f in os.listdir(pottery_dir)
            if f.startswith(pottery_id) and f.endswith('.glb')
        ]
        if glb_files:
            try:
                rendered_image = render_glb_front_view(
                    os.path.join(pottery_dir, glb_files[0]), image_size)
                pil_image = Image.fromarray(rendered_image)
                rendered_count += 1
            except Exception as e:
                print(f"Failed to render {pottery_id}: {e}")
                pil_image = create_simple_pottery_image(
                    f"{pottery_id}\n(Render Error)", image_size)
            draw = ImageDraw.Draw(pil_image)
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except:
                font = ImageFont.load_default()
            text_bbox = draw.textbbox((0, 0), pottery_id, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[
                3] - text_bbox[1]
            text_x, text_y = 5, image_size[1] - text_height - 10
            draw.rectangle([
                text_x, text_y - 2, text_x + text_width + 4,
                text_y + text_height + 2
            ],
                           fill=(255, 255, 255, 180))
            draw.text((text_x + 2, text_y),
                      pottery_id,
                      fill=(0, 0, 0),
                      font=font)
        else:
            pil_image = create_simple_pottery_image(
                f"{pottery_id}\n(GLB not found)", image_size)
        collage.paste(pil_image, (col * image_size[0], row * image_size[1]))
    collage_path = os.path.join(output_dir, f"cluster_{cluster_id}.png")
    collage.save(collage_path, "PNG")
    print(
        f"Collage saved: {collage_path} ({rendered_count}/{len(pottery_ids)} models rendered)"
    )
    return collage_path

# PLOTTING & VISUALIZATION UTILITIES

def save_colored_mesh(mesh, intensities, colormap, output_path):
    """Normalizes intensities, applies a colormap, and saves the mesh."""
    mesh_copy = deepcopy(mesh)
    max_intensity = np.max(intensities)
    normalized_values = intensities / max_intensity if max_intensity > 0 else intensities
    colors = colormap(normalized_values)[:, :3]
    mesh_copy.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_triangle_mesh(output_path, mesh_copy, write_ascii=True)

def create_difference_colors(norm_jp_intensities, norm_my_intensities):
    """Creates the Red/Cyan/Grey/Black color array for the difference map."""
    num_vertices = len(norm_jp_intensities)
    colors = np.zeros((num_vertices, 3))
    jp_only = (norm_jp_intensities > 0) & (norm_my_intensities == 0)
    my_only = (norm_my_intensities > 0) & (norm_jp_intensities == 0)
    overlap = (norm_jp_intensities > 0) & (norm_my_intensities > 0)
    colors[jp_only, 0] = norm_jp_intensities[jp_only]
    colors[my_only, 1:] = norm_my_intensities[my_only][:, np.newaxis]
    avg_intensity = (norm_jp_intensities[overlap] + norm_my_intensities[overlap]) / 2.0
    colors[overlap] = avg_intensity[:, np.newaxis]
    return colors

def create_jsd_bar_chart(df, summary_stats, output_base_dir):
    """Generates and saves a bar chart of all JSD scores."""
    df_plot = df.sort_values(by='js_distance', ascending=False)
    mean_jsd, std_jsd = summary_stats['mean'], summary_stats['std']

    plt.figure(figsize=(18, 8))
    plt.bar(df_plot['pottery_id'], df_plot['js_distance'], color='darkblue', label='JSD Score')
    plt.axhline(mean_jsd, color='red', linestyle='--', label=f'Mean ({mean_jsd:.3f})')
    plt.axhline(mean_jsd + std_jsd, color='orange', linestyle=':', label='Mean +/- 1 Std Dev')
    plt.axhline(mean_jsd - std_jsd, color='orange', linestyle=':')

    plt.title('Jensen-Shannon Distance (JSD) of Gaze Heatmaps (Japan vs. Malaysia)')
    plt.ylabel('JSD Score (0.0 = Identical, 1.0 = Max Difference)')
    plt.xlabel('Pottery ID (Sorted by Dissimilarity)')
    plt.xticks(rotation=90, fontsize=6)
    plt.ylim(0, df_plot['js_distance'].max() * 1.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    chart_filepath = os.path.join(output_base_dir, "jsd_pottery_bar_graph.png")
    plt.savefig(chart_filepath)
    plt.close()
    return chart_filepath

def draw_ellipse(points, ax=None, **kwargs):
    """Draws a fitted ellipse around a set of points."""
    ax = ax or plt.gca()
    if len(points) < 2: return
    cov = np.cov(points, rowvar=False)
    if np.isclose(np.linalg.det(cov), 0): return

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))

    transformed_points = points @ eigvecs
    center_transformed = np.mean(transformed_points, axis=0)
    width, height = 2 * np.sqrt(5.991) * np.std(transformed_points, axis=0)

    center = center_transformed @ eigvecs.T
    ellipse = mpatches.Ellipse(xy=center, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)

def generate_word_cloud_and_bar_chart(all_words, font_path, output_prefix):
    """Generates and saves a word cloud and bar chart for word frequencies."""
    if not all_words:
        print("No words found for analysis.")
        return

    word_counts = Counter(all_words)

    # Save CSV
    pd.DataFrame(word_counts.most_common(), columns=['Word', 'Frequency']).to_csv(f"{output_prefix}_distribution.csv", index=False)

    # Bar Chart
    plt.figure(figsize=(18, 8))
    top_words, frequencies = zip(*word_counts.most_common(50))
    plt.bar(top_words, frequencies)
    plt.title('Top 50 Most Frequent Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_barchart.png")
    plt.close()

    # Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=font_path).generate(' '.join(all_words))
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(f"{output_prefix}_wordcloud.png")
    plt.close()
    print(f"Generated word frequency outputs with prefix: {output_prefix}")

# PDF REPORT GENERATION

worker_styles = None
def init_worker(styles_arg):
    global worker_styles
    worker_styles = styles_arg

def _create_comparison_plot_for_pdf(qa_pct, embed_pct, title, language):
    """Creates a side-by-side bar chart for PDF reporting."""
    target_labels = EMOTION_MAPS[language]['target_labels']
    qa_values = [qa_pct.get(label, 0) for label in target_labels]
    embed_values = [embed_pct.get(label, 0) for label in target_labels]

    y = np.arange(len(target_labels))
    height = 0.4
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    fig.suptitle(title, fontsize=14)

    colors = plt.cm.get_cmap('jet', len(target_labels))

    ax1.barh(y, qa_values, height, color=[colors(i) for i in range(len(target_labels))])
    ax1.set_title('QA Event Count (%)')
    ax1.set_xlim(0, 100)
    ax1.set_yticks(y)
    ax1.set_yticklabels(target_labels)
    ax1.invert_xaxis()
    ax1.yaxis.tick_right()

    ax2.barh(y, embed_values, height, color=[colors(i) for i in range(len(target_labels))])
    ax2.set_title('Transcript Classification Score (%)')
    ax2.set_xlim(0, 100)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    buf = io.BytesIO()
    plt.savefig(buf, format='svg')
    buf.seek(0)
    plt.close(fig)
    return buf

def _process_pottery_group_for_report(pottery_data_with_lang):
    """Processes a single pottery group for the alignment PDF report."""
    pottery_id, sessions, language = pottery_data_with_lang
    pottery_story = []

    pottery_story.append(Paragraph(f"Analysis for Pottery ID: {pottery_id}", worker_styles['HeaderStyle']))
    avg_alignment = np.mean([s['alignment_score'] for s in sessions if 'alignment_score' in s])
    pottery_story.append(Paragraph(f"<b>Average Alignment Score:</b> {avg_alignment:.3f}", worker_styles['BodyStyle']))
    pottery_story.append(HRFlowable(width="100%"))

    for session in sorted(sessions, key=lambda x: x['SESSION_ID']):
        pottery_story.append(Paragraph(f"<b>Session:</b> {session['SESSION_ID']} | <b>Alignment:</b> {session.get('alignment_score', 0):.3f}", worker_styles['BodyStyle']))

        plot_buffer = _create_comparison_plot_for_pdf(session['qa_percentages'], session['embedding_percentages'], f"{pottery_id} | {session['SESSION_ID']}", language)
        drawing = svg2rlg(plot_buffer)

        # Scale drawing to fit page
        desired_width = 7.5 * inch
        scale_factor = desired_width / drawing.width
        drawing.width, drawing.height = drawing.width * scale_factor, drawing.height * scale_factor
        drawing.scale(scale_factor, scale_factor)

        pottery_story.append(Table([[drawing]]))
        pottery_story.append(Paragraph("<b>Transcript:</b>", worker_styles['BodyStyle']))
        pottery_story.append(Paragraph(session.get('transcript_text', 'N/A'), worker_styles['TranscriptStyle']))
        pottery_story.append(Spacer(1, 12))

    pottery_story.append(PageBreak())
    return pottery_story

def generate_alignment_report(data_paths, model_id, output_filename, language, font_path, max_workers=8):
    """Generates the full QA vs. Transcript alignment PDF report."""
    print(f"Generating PDF report: {output_filename}")
    try:
        pdfmetrics.registerFont(TTFont('CustomFont', font_path))
        font_name = 'CustomFont'
    except Exception as e:
        print(f"Warning: Could not register font {font_path}. Using default. Error: {e}")
        font_name = 'Helvetica'

    doc = SimpleDocTemplate(output_filename)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleStyle', fontName=font_name, fontSize=18, alignment=TA_CENTER, spaceAfter=12))
    styles.add(ParagraphStyle(name='HeaderStyle', fontName=font_name, fontSize=14, spaceAfter=10))
    styles.add(ParagraphStyle(name='BodyStyle', fontName=font_name, fontSize=10, leading=14))
    styles.add(ParagraphStyle(name='TranscriptStyle', fontName=font_name, fontSize=8, leading=10, leftIndent=10))

    story = [Paragraph("QA vs. Transcript Alignment Report", styles['TitleStyle'])]
    story.append(Paragraph(f"Model: {model_id}", styles['BodyStyle']))
    avg_alignment = np.mean([dp['alignment_score'] for dp in data_paths if 'alignment_score' in dp])
    story.append(Paragraph(f"<b>Overall Average Alignment Score:</b> {avg_alignment:.3f}", styles['HeaderStyle']))
    story.append(PageBreak())

    data_by_pottery = defaultdict(list)
    for dp in data_paths:
        data_by_pottery[dp['ID']].append(dp)

    pottery_tasks = [(pid, sessions, language) for pid, sessions in sorted(data_by_pottery.items())]

    styles_for_workers = {k: v for k, v in styles.items()}

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(styles_for_workers,)) as executor:
        results = list(tqdm(executor.map(_process_pottery_group_for_report, pottery_tasks), total=len(pottery_tasks), desc="Generating PDF pages"))

    for story_chunk in results:
        story.extend(story_chunk)

    doc.build(story)
    print(f"PDF report generation complete: {output_filename}")


def generate_transcript_pdf(transcripts_by_pottery, output_filename,
                            font_path):
    """Generates a PDF containing all transcripts, grouped by pottery ID and session."""
    try:
        pdfmetrics.registerFont(TTFont('CustomFont', font_path))
        font_name = 'CustomFont'
    except Exception as e:
        print(
            f"Warning: Could not register font {font_path}. Using default. Error: {e}"
        )
        font_name = 'Helvetica'

    doc = SimpleDocTemplate(output_filename)
    styles = getSampleStyleSheet()
    # styles.add(ParagraphStyle(name='Title', fontName=font_name, fontSize=16, spaceAfter=16))
    styles.add(
        ParagraphStyle(name='Heading',
                       fontName=font_name,
                       fontSize=12,
                       spaceAfter=10,
                       spaceBefore=10))
    styles.add(
        ParagraphStyle(name='Body',
                       fontName=font_name,
                       fontSize=9,
                       leading=14,
                       spaceAfter=10))

    story = [
        Paragraph("Aggregated Transcripts by Pottery ID", styles['Title']),
        HRFlowable(width="100%")
    ]

    for pottery_id in sorted(transcripts_by_pottery.keys()):
        story.append(Paragraph(f"Pottery ID: {pottery_id}", styles['Heading']))
        for i, (session_id,
                text) in enumerate(transcripts_by_pottery[pottery_id]):
            subtitle = f"<b>Transcript {i+1} | Session ID: {session_id}</b>"
            story.append(
                Paragraph(
                    f"{subtitle}<br/><br/>{text.replace(os.linesep, '<br/>')}",
                    styles['Body']))
            if i < len(transcripts_by_pottery[pottery_id]) - 1:
                story.append(
                    HRFlowable(width="100%", thickness=0.5, color=colors.grey))
        story.append(PageBreak())

    doc.build(story)
    print(f"Transcript PDF saved to {output_filename}")
