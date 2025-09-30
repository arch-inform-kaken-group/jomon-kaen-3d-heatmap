import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import japanize_matplotlib
import os
import sys
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from PIL import Image, ImageDraw, ImageFont
import math

import trimesh

# Alternative rendering using matplotlib for 3D visualization
try:
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    MATPLOTLIB_3D_AVAILABLE = True
except ImportError:
    MATPLOTLIB_3D_AVAILABLE = False

# --- Dictionaries and Constants ---

ASSIGNED_NUMBERS_DICT = {
    'AS0001': '1',
    'FH0008': '2',
    'IN0003': '3',
    'IN0008': '4',
    'IN0009': '5',
    'IN0017': '6',
    'IN0081': '7',
    'IN0104': '8',
    'IN0135': '9',
    'IN0148': '10',
    'IN0220': '11',
    'IN0228': '12',
    'IN0232': '13',
    'IN0239': '14',
    'IN0277': '15',
    'MY0001': '16',
    'MY0002': '17',
    'MY0004': '18',
    'MY0006': '19',
    'MY0007': '20',
    'ND0001': '21',
    'NM0001': '22',
    'NM0002': '23',
    'NM0009': '24',
    'NM0010': '25',
    'NM0014': '26',
    'NM0015': '27',
    'NM0017': '28',
    'NM0041': '29',
    'NM0049': '30',
    'NM0066': '31',
    'NM0070': '32',
    'NM0072': '33',
    'NM0073': '34',
    'NM0079': '35',
    'NM0080': '36',
    'NM0099': '37',
    'NM0106': '38',
    'NM0133': '39',
    'NM0135': '40',
    'NM0144': '41',
    'NM0154': '42',
    'NM0156': '43',
    'NM0159': '44',
    'NM0168': '45',
    'NM0173': '46',
    'NM0175': '47',
    'NM0189': '48',
    'NM0191': '49',
    'NM0206': '50',
    'SB0002': '51',
    'SB0004': '52',
    'SI0001': '53',
    'SJ0503': '54',
    'SJ0504': '55',
    'SK0001': '56',
    'SK0002': '57',
    'SK0003': '58',
    'SK0004': '59',
    'SK0005': '60',
    'SK0013': '61',
    'SS0001': '62',
    'TJ0004': '63',
    'TJ0005': '64',
    'TJ0010': '65',
    'TK0002': '66',
    'TK0048': '67',
    'TK0057': '68',
    'UD0001': '69',
    'UD0003': '70',
    'UD0005': '71',
    'UD0006': '72',
    'UD0011': '73',
    'UD0013': '74',
    'UD0014': '75',
    'UD0016': '76',
    'UD0023': '77',
    'UD0302': '78',
    'UD0304': '79',
    'UD0308': '80',
    'UD0318': '81',
    'UD0322': '82',
    'UD0411': '83',
    'UD0412': '84',
    'UK0001': '85',
    'IN0295': '86',
    'IN0306': '87',
    'MH0037': '88',
    'NM0239': '89',
    'NZ0001': '90',
    'SK0035': '91',
    'TK0020': '92',
    'UD0028': '93',
}

EMOTION_COLOR_MAP = {
    "面白い・気になる形だ": "#00FFFF",
    "美しい・芸術的だ": "#00FF00",
    "不思議・意味不明": "#FFFF00",
    "不気味・不安・怖い": "#FF0000",
    "何も感じない": "#505050",
    "NO RESPONSE": "#D3D3D3",
    # "Interesting and attentional shape": "#00FFFF",
    # "Beautiful and artistic": "#00FF00",
    # "Strange and incomprehensible": "#FFFF00",
    # "Creepy / unsettling / scary": "#FF0000",
    # "Feel nothing": "#505050",
    # "NO RESPONSE": "#D3D3D3",
}

# --- 3D Model Rendering Functions ---
def create_simple_pottery_image(pottery_id: str, image_size: tuple = (256, 256)) -> Image.Image:
    """
    Creates a simple placeholder image with pottery information when 3D rendering fails.
    """
    # Create a gradient background
    img = Image.new('RGB', image_size, color=(240, 245, 250))
    draw = ImageDraw.Draw(img)
    
    # Draw a simple pottery silhouette using basic shapes
    center_x, center_y = image_size[0] // 2, image_size[1] // 2
    
    # Draw a simple pottery vessel shape
    # Base
    base_width = image_size[0] // 3
    base_height = image_size[1] // 6
    draw.ellipse([center_x - base_width//2, center_y + image_size[1]//4 - base_height//2,
                  center_x + base_width//2, center_y + image_size[1]//4 + base_height//2],
                 fill=(120, 120, 120), outline=(80, 80, 80))
    
    # Body
    body_width = image_size[0] // 4
    body_height = image_size[1] // 3
    draw.ellipse([center_x - body_width//2, center_y - body_height//2,
                  center_x + body_width//2, center_y + body_height//2],
                 fill=(150, 150, 150), outline=(100, 100, 100))
    
    # Neck
    neck_width = image_size[0] // 6
    neck_height = image_size[1] // 8
    draw.ellipse([center_x - neck_width//2, center_y - image_size[1]//4 - neck_height//2,
                  center_x + neck_width//2, center_y - image_size[1]//4 + neck_height//2],
                 fill=(130, 130, 130), outline=(90, 90, 90))
    
    # Add pottery ID label
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Background for text
    text_bbox = draw.textbbox((0, 0), pottery_id, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    text_x = (image_size[0] - text_width) // 2
    text_y = image_size[1] - text_height - 15
    
    draw.rectangle([text_x - 5, text_y - 3, text_x + text_width + 5, text_y + text_height + 3],
                   fill=(255, 255, 255, 200), outline=(100, 100, 100))
    draw.text((text_x, text_y), pottery_id, fill=(0, 0, 0), font=font)
    
    # Add "3D Model" text at top
    try:
        small_font = ImageFont.truetype("arial.ttf", 12)
    except:
        small_font = ImageFont.load_default()
    
    draw.text((10, 10), "3D Model", fill=(100, 100, 100), font=small_font)
    
    return img

def render_glb_matplotlib(glb_path: str, output_size: tuple = (512, 512)) -> np.ndarray:
    """
    Alternative renderer using matplotlib for 3D visualization.
    More stable but less realistic than pyrender.
    Includes 90-degree left rotation.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import io
        
        # Load the 3D model
        mesh = trimesh.load(glb_path)
        
        # If it's a scene, get the first mesh
        if hasattr(mesh, 'geometry'):
            if len(mesh.geometry) == 0:
                raise ValueError("No geometry found in the GLB file")
            mesh = list(mesh.geometry.values())[0]
        
        # Apply 90-degree rotation to the left (around Z-axis)
        # 90 degrees to the left = -90 degrees around Z-axis
        rotation_matrix = trimesh.transformations.rotation_matrix(
            angle=np.pi/2,  # -90 degrees in radians
            direction=[1, 0, 0],  # Z-axis
            point=[0, 0, 0]
        )
        mesh.apply_transform(rotation_matrix)
        
        # Create matplotlib 3D plot
        fig = plt.figure(figsize=(output_size[0]/100, output_size[1]/100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        # Get vertices and faces
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Plot the mesh with better visualization
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                       triangles=faces, alpha=0.9, cmap='copper', 
                       linewidth=0, antialiased=True)
        
        # Set up the view for front perspective
        ax.view_init(elev=10, azim=0)  # Slightly elevated front view
        
        # Set equal aspect ratio
        max_range = np.array([vertices[:,0].max()-vertices[:,0].min(),
                             vertices[:,1].max()-vertices[:,1].min(),
                             vertices[:,2].max()-vertices[:,2].min()]).max() / 2.0
        mid_x = (vertices[:,0].max()+vertices[:,0].min()) * 0.5
        mid_y = (vertices[:,1].max()+vertices[:,1].min()) * 0.5
        mid_z = (vertices[:,2].max()+vertices[:,2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Remove axes and make background white
        ax.set_axis_off()
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # Convert to numpy array - handle different matplotlib versions
        fig.canvas.draw()
        
        # Try the new method first (matplotlib >= 3.8)
        try:
            buf = fig.canvas.buffer_rgba()
            buf = np.asarray(buf)
            # Convert RGBA to RGB
            buf = buf[:, :, :3]
        except (AttributeError, TypeError):
            # Fallback for older matplotlib versions
            try:
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            except AttributeError:
                # Final fallback using buffer and PIL
                buf = fig.canvas.buffer_rgba()
                img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), buf, "raw", "RGBA", 0, 1)
                img = img.convert("RGB")
                buf = np.array(img)
        
        plt.close(fig)
        return buf
        
    except Exception as e:
        print(f"Error with matplotlib rendering {glb_path}: {e}")
        # Return a placeholder image
        placeholder = np.ones((*output_size[::-1], 3), dtype=np.uint8) * 180
        return placeholder


def render_glb_front_view(glb_path: str, output_size: tuple = (512, 512)) -> np.ndarray:
    """
    Renders a front view of a .glb file and returns it as a numpy array.
    Uses multiple rendering backends with fallbacks for better compatibility.
    
    Args:
        glb_path: Path to the .glb file
        output_size: Tuple of (width, height) for the output image
    
    Returns:
        numpy array representing the rendered image
    """
    
    # Fallback to matplotlib rendering
    if MATPLOTLIB_3D_AVAILABLE:
        print(f"Falling back to matplotlib rendering for {glb_path}")
        return render_glb_matplotlib(glb_path, output_size)
    
    # Final fallback: create a more informative placeholder
    print(f"All rendering methods failed for {glb_path}, creating artistic placeholder")
    try:
        # Extract pottery ID from path for the placeholder
        pottery_id = os.path.basename(glb_path).split('.')[0]
        placeholder_img = create_simple_pottery_image(pottery_id, output_size)
        return np.array(placeholder_img)
    except:
        # Absolute final fallback
        print(f"Could not create placeholder for {glb_path}")
        placeholder = np.ones((*output_size[::-1], 3), dtype=np.uint8) * 200
        return placeholder


def create_cluster_collage(pottery_ids: list, pottery_dir: str, cluster_id: int, 
                          output_dir: str, image_size: tuple = (256, 256),
                          collage_columns: int = None) -> str:
    """
    Creates a collage of pottery models for a given cluster.
    
    Args:
        pottery_ids: List of pottery IDs in this cluster
        pottery_dir: Directory containing .glb files
        cluster_id: The cluster number
        output_dir: Directory to save the collage
        image_size: Size of each individual pottery image
        collage_columns: Number of columns in the collage (auto-calculated if None)
    
    Returns:
        Path to the saved collage image
    """
    print(f"Creating collage for Cluster {cluster_id} with {len(pottery_ids)} items...")
    
    # Auto-calculate grid dimensions
    num_items = len(pottery_ids)
    if collage_columns is None:
        collage_columns = min(5, int(math.ceil(math.sqrt(num_items))))
    collage_rows = int(math.ceil(num_items / collage_columns))
    
    # Calculate collage dimensions
    collage_width = collage_columns * image_size[0]
    collage_height = collage_rows * image_size[1]
    
    # Create the collage canvas
    collage = Image.new('RGB', (collage_width, collage_height), color=(240, 240, 240))
    
    # Render each pottery model and add to collage
    rendered_count = 0
    for idx, pottery_id in enumerate(tqdm(pottery_ids, desc=f"Cluster {cluster_id}")):
        row = idx // collage_columns
        col = idx % collage_columns
        
        # Find the .glb file for this pottery
        pottery_key = pottery_id  # pottery_id already includes the number format
        glb_files = [f for f in os.listdir(pottery_dir) if f.startswith(pottery_key) and f.endswith('.glb')]
        
        if glb_files:
            glb_path = os.path.join(pottery_dir, glb_files[0])
            
            try:
                # Render the model with better error handling
                rendered_image = render_glb_front_view(glb_path, image_size)
                
                # Convert numpy array to PIL Image
                pil_image = Image.fromarray(rendered_image)
                rendered_count += 1
                
            except Exception as e:
                print(f"Failed to render {pottery_id}: {e}")
                # Create error placeholder
                pil_image = Image.new('RGB', image_size, color=(220, 220, 220))
                draw = ImageDraw.Draw(pil_image)
                try:
                    font = ImageFont.truetype("arial.ttf", 12)
                except:
                    font = ImageFont.load_default()
                draw.text((10, 10), f"{pottery_id}\n(Render Error)", fill=(100, 100, 100), font=font)
            
            # Add label with pottery ID
            draw = ImageDraw.Draw(pil_image)
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except:
                font = ImageFont.load_default()
            
            # Draw text background with better positioning
            text_bbox = draw.textbbox((0, 0), pottery_id, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Position text at bottom of image
            text_x = 5
            text_y = image_size[1] - text_height - 10
            
            draw.rectangle([text_x, text_y - 2, text_x + text_width + 4, text_y + text_height + 2], 
                         fill=(255, 255, 255, 180))
            draw.text((text_x + 2, text_y), pottery_id, fill=(0, 0, 0), font=font)
            
        else:
            # Create placeholder if .glb file not found
            pil_image = Image.new('RGB', image_size, color=(200, 200, 200))
            draw = ImageDraw.Draw(pil_image)
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
            draw.text((10, 10), f"{pottery_id}\n(GLB not found)", fill=(0, 0, 0), font=font)
        
        # Paste into collage
        x = col * image_size[0]
        y = row * image_size[1]
        collage.paste(pil_image, (x, y))
    
    # Save the collage
    collage_filename = f"cluster_{cluster_id}.png"
    collage_path = os.path.join(output_dir, collage_filename)
    collage.save(collage_path, "PNG")
    
    print(f"Collage saved: {collage_path} ({rendered_count}/{len(pottery_ids)} models rendered successfully)")
    return collage_path

# --- Data Loading Functions (unchanged from original) ---

def increment_error(key, path, errors: dict):
    """Simple helper function to track errors during data loading."""
    if errors.get(key) is None:
        errors[key] = {'count': 1, 'paths': {path}}
    else:
        errors[key]['count'] += 1
        errors[key]['paths'].add(path)
    return errors

def find_data_paths_detailed(root: str,
                             pottery_path_str: str,
                             limit: int = 1000) -> list:
    """Finds all valid data instances and returns their paths."""
    root = Path(root)
    pottery_path = Path(pottery_path_str)
    errors = {}

    # pottery_ids_to_include = ['IN0295', 'IN0306', 'MH0037', 'NM0239', 'NZ0001', 'SK0035', 'TK0020', 'UD0028']
    pottery_ids_to_include = []

    data = []
    pottery_id_to_path = {}
    if not root.exists():
        raise ValueError(f"Root directory not found: {root}")
    if not pottery_path.exists():
        raise ValueError(f"Pottery directory not found: {pottery_path}")

    pottery_ids = [
        f"{pid}({ASSIGNED_NUMBERS_DICT[pid]})" for pid in ASSIGNED_NUMBERS_DICT
        if pid not in pottery_ids_to_include
    ]

    print(f"\nCHECKING POTTERY PATHS")
    pottery_id_all = [
        f"{pid}({num})" for pid, num in ASSIGNED_NUMBERS_DICT.items()
    ]
    pottery_available = os.listdir(pottery_path)
    pottery_id_available = [p.split(".")[0] for p in pottery_available]
    for p in tqdm(pottery_id_all, desc="POTTERY & DOGU"):
        if p in pottery_id_available:
            pottery_id_to_path[p] = pottery_path / pottery_available[
                pottery_id_available.index(p)]
        else:
            pottery_id_to_path[p] = ""
            errors = increment_error('Missing pottery',
                                     str(pottery_path / f"{p}.*"), errors)

    print(f"\nCHECKING RAW DATA PATHS")
    limit_dict = {pid: 0 for pid in pottery_ids}
    group_keys = [d for d in os.listdir(root) if os.path.isdir(root / d)]
    for g in group_keys:
        group_path = root / g
        session_keys = [
            d for d in os.listdir(group_path) if os.path.isdir(group_path / d)
        ]
        for s in tqdm(session_keys, desc=g):
            session_path = group_path / s
            pottery_keys = [
                d for d in os.listdir(session_path)
                if os.path.isdir(session_path / d)
            ]
            for p in pottery_keys:
                if p not in pottery_ids:
                    continue

                qa_path = session_path / p / "qa_corrected.csv"
                if qa_path.exists() and limit_dict[p] < limit:
                    limit_dict[p] += 1
                    data.append({
                        'qa': str(qa_path),
                        'GROUP': g,
                        'SESSION_ID': s,
                        'ID': p
                    })
                else:
                    errors = increment_error('QNA path does not exist',
                                             str(qa_path), errors)

    print(f"\nLoader finished. Found {len(data)} valid data instances.")
    if errors:
        print("Encountered errors during loading:")
        for key, val in errors.items():
            print(f"- {key}: {val['count']} instance(s)")

    return data

def load_combined_qna_data(root_dir: str,
                           pottery_models_dir: str) -> pd.DataFrame:
    """Uses the detailed loader to find all qa.csv files and combines them."""
    data_to_process = find_data_paths_detailed(
        root=root_dir, pottery_path_str=pottery_models_dir)
    if not data_to_process:
        return pd.DataFrame()

    df_list = []
    for item in tqdm(data_to_process, desc="Loading and combining data"):
        try:
            temp_df = pd.read_csv(item['qa'], header=0, sep=",")
            temp_df['timestamp'] = pd.to_numeric(temp_df['timestamp'],
                                                 errors='coerce')
            temp_df.dropna(subset=['timestamp'], inplace=True)
            temp_df['pottery_id'] = item['ID']
            temp_df['session_id'] = item['SESSION_ID']
            df_list.append(temp_df)
        except Exception as e:
            print(f"Could not read or process file {item['qa']}: {e}",
                  file=sys.stderr)

    if not df_list:
        print("No data could be loaded.", file=sys.stderr)
        return pd.DataFrame()

    print("\nCombining all data sources for analysis...")
    return pd.concat(df_list, ignore_index=True)

def analyze_and_plot_stacked_emotions(combined_df: pd.DataFrame):
    """Takes a combined DataFrame and generates stacked bar graphs for all pottery."""
    fontsize = 8

    if combined_df.empty:
        print("Combined DataFrame is empty. No analysis performed.",
              file=sys.stderr)
        return

    df = combined_df.copy()
    df['answer'] = df['answer'].str.strip()
    df = df.sort_values(by=['pottery_id', 'timestamp']).reset_index(drop=True)

    # Plot 1: Percentage by Event Count (Session-Normalized)
    print(
        "Generating plot for percentage breakdown of emotions (by event count)"
    )
    session_counts_df = pd.crosstab([df['pottery_id'], df['session_id']],
                                      df['answer'])
    print(session_counts_df.head())
    session_percentage_df = session_counts_df.div(session_counts_df.sum(axis=1), axis=0) * 100
    print(session_percentage_df.head())
    percentage_df = session_percentage_df.groupby('pottery_id').mean()
    print(percentage_df.head())

    ax1 = percentage_df.plot(
        kind='bar',
        stacked=True,
        figsize=(20, 8),
        color=[
            EMOTION_COLOR_MAP.get(e, '#CCCCCC') for e in percentage_df.columns
        ],
        width=0.7,
        fontsize=fontsize,
    )
    plt.title('Average Percentage of Emotions per Pottery (by Event Count)',
              fontsize=16)
    plt.ylabel('Average Percentage (%)', fontsize=12)
    plt.xlabel('Pottery ID', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    ax1.legend(title='Emotion / Affective State',
               bbox_to_anchor=(1.02, 1),
               loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.95, 1], pad=2)
    plt.savefig('emotion_stacked_percentage_plot_by_event_count.png')
    plt.show()

    # --- Duration Calculations ---
    print("\nCalculating emotion durations...")
    df['time_diff'] = df.groupby(['pottery_id',
                                  'session_id'])['timestamp'].diff()
    emotion_changed = df['answer'] != df.groupby(['pottery_id', 'session_id'
                                                  ])['answer'].shift()
    time_gap_exceeded = df['time_diff'] > 0.05
    df['block_id'] = (emotion_changed | time_gap_exceeded).cumsum()

    block_durations = df.groupby(['pottery_id', 'session_id', 'block_id'
                                  ]).agg(start_time=('timestamp', 'min'),
                                         end_time=('timestamp', 'max'),
                                         answer=('answer',
                                                 'first')).reset_index()
    print(block_durations.head())
    block_durations['duration'] = block_durations['end_time'] - block_durations['start_time']

    #############################################################################################

    print("\nGenerating plot for total duration of emotions")
    df['time_diff'] = df.groupby(['pottery_id',
                                  'session_id'])['timestamp'].diff()

    emotion_changed = df['answer'] != df.groupby(['pottery_id', 'session_id'
                                                  ])['answer'].shift()
    time_gap_exceeded = df['time_diff'] > 0.05

    df['block_id'] = (emotion_changed | time_gap_exceeded).cumsum()

    block_durations = df.groupby(['pottery_id', 'session_id', 'block_id'
                                  ]).agg(start_time=('timestamp', 'min'),
                                         end_time=('timestamp', 'max'),
                                         answer=('answer',
                                                 'first')).reset_index()

    block_durations['duration'] = block_durations[
        'end_time'] - block_durations['start_time']

    duration_df = block_durations.pivot_table(index='pottery_id',
                                              columns='answer',
                                              values='duration',
                                              aggfunc='sum',
                                              fill_value=0)

    pottery_session_counts = df.groupby('pottery_id')['session_id'].nunique()

    average_duration_df = duration_df.div(pottery_session_counts, axis=0)

    ax2 = average_duration_df.plot(
        kind='bar',
        stacked=True,
        figsize=(20, 8),
        color=[
            EMOTION_COLOR_MAP.get(e, '#CCCCCC')
            for e in average_duration_df.columns
        ],
        width=0.7,
        fontsize=fontsize,
    )

    for container in ax2.containers:
        labels = [f'{v:.1f}' if v > 0.1 else '' for v in container.datavalues]
        ax2.bar_label(container,
                      labels=labels,
                      label_type='center',
                      fontsize=fontsize - 3,
                      color='black',
                      weight='bold')

    plt.title('Average Duration of Emotions per Pottery (50ms gap limit)',
              fontsize=16)
    plt.ylabel('Average Duration (seconds)', fontsize=12)
    plt.xlabel('Pottery ID', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    ax2.legend(title='Emotion / Affective State',
               bbox_to_anchor=(1.02, 1),
               loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.95, 1], pad=2)
    plt.savefig('emotion_stacked_duration_plot.png')
    plt.show()

    #############################################################################################

    print(
        "\nGenerating plot for percentage of viewing time (including no response)"
    )
    session_durations = df.groupby(['pottery_id', 'session_id'
                                      ])['timestamp'].agg(['min', 'max'])
    session_durations['total_duration'] = (session_durations['max'] -
                                           session_durations['min']).clip(
                                               upper=60)
    emotion_duration_per_session = block_durations.groupby(
        ['pottery_id', 'session_id'])['duration'].sum()
    session_summary = pd.merge(
        session_durations,
        emotion_duration_per_session.rename('emotion_duration'),
        on=['pottery_id', 'session_id'])
    session_summary['NO RESPONSE'] = session_summary[
        'total_duration'] - session_summary['emotion_duration']
    total_emotion_durations = block_durations.groupby(
        ['pottery_id', 'answer'])['duration'].sum().unstack(fill_value=0)
    total_not_viewing = session_summary.groupby(
        'pottery_id')['NO RESPONSE'].sum()
    final_durations = pd.concat([total_emotion_durations, total_not_viewing],
                                axis=1)
    percentage_viewing_time_df = final_durations.div(
        final_durations.sum(axis=1), axis=0) * 100

    ax3 = percentage_viewing_time_df.plot(
        kind='bar',
        stacked=True,
        figsize=(20, 8),
        color=[
            EMOTION_COLOR_MAP.get(e, '#CCCCCC')
            for e in percentage_viewing_time_df.columns
        ],
        width=0.7,
        fontsize=fontsize,
    )
    plt.title('Percentage of Viewing Time per Pottery (60s per session)',
              fontsize=16)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.xlabel('Pottery ID', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    ax3.legend(title='Emotion / Affective State',
               bbox_to_anchor=(1.02, 1),
               loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.95, 1], pad=2)
    plt.savefig('viewing_time_percentage_plot_including_no_response.png')
    plt.show()

    #############################################################################################

    output_dir = "timelines"
    os.makedirs(output_dir, exist_ok=True)
    print(
        f"\nGenerating stacked emotion timelines. Plots will be saved in '{output_dir}/'"
    )

    # Group data by each pottery ID to create a separate plot for each
    pottery_groups = combined_df.groupby('pottery_id')

    for pottery_id, pottery_df in tqdm(pottery_groups,
                                       desc="Creating Stacked Timelines"):
        sessions = sorted(pottery_df['session_id'].unique())
        num_sessions = len(sessions)

        # Create a figure with height proportional to the number of sessions
        fig, ax = plt.subplots(figsize=(20, num_sessions * 0.7))

        # Process and plot each session as a separate row in the figure
        for i, session_id in enumerate(sessions):
            session_df = pottery_df[pottery_df['session_id'] ==
                                    session_id].copy().sort_values('timestamp')

            if session_df.empty:
                continue

            # Normalize timestamps to start from 0 for this session
            session_start_time = session_df['timestamp'].min()
            session_df[
                'timestamp'] = session_df['timestamp'] - session_start_time

            # Identify continuous blocks of the same emotion
            session_df['time_diff'] = session_df['timestamp'].diff()
            emotion_changed = session_df['answer'] != session_df[
                'answer'].shift()
            time_gap_exceeded = session_df['time_diff'] > 0.05
            session_df['block_id'] = (emotion_changed
                                      | time_gap_exceeded).cumsum()

            block_df = session_df.groupby('block_id').agg(
                start_time=('timestamp', 'min'),
                end_time=('timestamp', 'max'),
                answer=('answer', 'first')).reset_index()
            block_df[
                'duration'] = block_df['end_time'] - block_df['start_time']

            colors = [
                EMOTION_COLOR_MAP.get(ans, "#808080")
                for ans in block_df["answer"]
            ]

            # Plot this session's blocks on its assigned horizontal row (y=i)
            ax.barh(
                y=[i] * len(block_df),
                width=block_df['duration'],
                left=block_df['start_time'],
                color=colors,
                height=0.8,
            )

        # --- Formatting the entire plot for the current pottery ID ---
        ax.set_yticks(range(num_sessions))
        ax.set_yticklabels(sessions, fontsize=8)
        ax.invert_yaxis()  # Puts the first session at the top
        ax.set_xlabel("Time Since Session Start (seconds)")
        ax.set_xlim(left=0)
        ax.set_title(f"Emotion Timelines for {pottery_id}")

        # Create a shared legend
        legend_patches = [
            mpatches.Patch(color=color, label=name)
            for name, color in EMOTION_COLOR_MAP.items()
        ]
        ax.legend(handles=legend_patches,
                  bbox_to_anchor=(1.02, 1),
                  loc='upper left')

        fig.tight_layout(rect=[0, 0, 0.97, 1], pad=2)

        # Save the combined figure for the pottery ID
        filename = f"timeline_stacked_{pottery_id}.png"
        save_path = os.path.join(output_dir, filename)
        fig.savefig(save_path)
        plt.close(fig)


def perform_clustering_analysis(combined_df: pd.DataFrame):
    """
    Performs clustering analysis on the pottery based on emotion percentages
    and generates a dendrogram plot.
    """
    if combined_df.empty:
        print(
            "Combined DataFrame is empty. No clustering analysis will be performed.",
            file=sys.stderr)
        return

    print("\nPerforming clustering analysis...")

    # --- 1. Prepare the Data ---
    session_counts_df = pd.crosstab(
        [combined_df['pottery_id'], combined_df['session_id']],
        combined_df['answer'])
    session_percentage_df = session_counts_df.div(
        session_counts_df.sum(axis=1), axis=0) * 100
    percentage_df = session_percentage_df.groupby('pottery_id').mean()

    for emotion in EMOTION_COLOR_MAP.keys():
        if emotion not in percentage_df.columns:
            percentage_df[emotion] = 0
    percentage_df = percentage_df[list(EMOTION_COLOR_MAP.keys())]

    # --- 2. Perform Hierarchical Clustering ---
    linked = linkage(percentage_df, method='ward')

    # --- 3. Plot the Dendrogram ---
    plt.figure(figsize=(15, 8))
    dendrogram(linked,
               orientation='top',
               labels=percentage_df.index.tolist(),
               distance_sort='descending',
               show_leaf_counts=True)

    plt.title(
        'Hierarchical Clustering Dendrogram of Pottery by Emotion Profile')
    plt.xlabel('Pottery ID')
    plt.ylabel('Distance (Similarity)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # --- 4. Save the Plot ---
    output_filename = 'pottery_clustering_dendrogram.png'
    plt.savefig(output_filename)
    print(f"Clustering plot saved to '{output_filename}'")
    plt.show()

# --- Enhanced K-Means Function with Collage Generation ---

def perform_kmeans_clustering_with_collages(combined_df: pd.DataFrame, pottery_models_dir: str):
    """
    Performs K-Means clustering on the pottery based on emotion percentages
    and generates both plots and 3D model collages for each cluster.
    """
    if combined_df.empty:
        print("Combined DataFrame is empty. No clustering analysis performed.",
              file=sys.stderr)
        return

    print("\n--- Starting K-Means Clustering Analysis with Collage Generation ---")

    # 1. Prepare the Data for Clustering
    print("Step 1: Preparing data by calculating session-level percentages and then averaging...")
    session_counts_df = pd.crosstab(
        [combined_df['pottery_id'], combined_df['session_id']],
        combined_df['answer'])
    session_percentage_df = session_counts_df.div(
        session_counts_df.sum(axis=1), axis=0) * 100
    percentage_df = session_percentage_df.groupby('pottery_id').mean()

    all_emotions = list(EMOTION_COLOR_MAP.keys())
    for emotion in all_emotions:
        if emotion not in percentage_df.columns:
            percentage_df[emotion] = 0
    percentage_df = percentage_df[all_emotions]

    # 2. Determine the Optimal Number of Clusters (K) using the Elbow Method
    print("Step 2: Finding optimal number of clusters using the Elbow Method...")
    inertia = []
    k_range = range(1, min(51, len(percentage_df)))
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=25)
        kmeans.fit(percentage_df)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (Sum of squared distances)')
    plt.title('Elbow Method for Optimal K')
    plt.grid(True)
    elbow_plot_path = 'kmeans_elbow_plot.png'
    plt.savefig(elbow_plot_path)
    plt.show()
    print(f"Elbow method plot saved to '{elbow_plot_path}'")

    # 3. Run clustering for different values of K
    for j in range(8, 9):
        optimal_k = j
        print(f"\nStep 3: Performing K-Means clustering with K={optimal_k}...")

        # Run K-Means with the current K
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=25)
        cluster_labels = kmeans.fit_predict(percentage_df)
        percentage_df['cluster'] = cluster_labels

        # Create output directory for this K
        output_dir = f'k_{j}'
        os.makedirs(output_dir, exist_ok=True)

        # 4. Visualize the Clusters using PCA for dimensionality reduction
        print("Step 4: Visualizing clusters using PCA...")
        pca = PCA(n_components=2, random_state=42)
        reduced_features = pca.fit_transform(
            percentage_df.drop('cluster', axis=1))

        kmeans_2d = KMeans(n_clusters=optimal_k, random_state=42, n_init=25)
        kmeans_2d.fit(reduced_features)

        x_min, x_max = reduced_features[:, 0].min() - 1, reduced_features[:, 0].max() + 1
        y_min, y_max = reduced_features[:, 1].min() - 1, reduced_features[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))

        Z = kmeans_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(14, 10))
        cmap_obj = plt.get_cmap('viridis', optimal_k)

        plt.imshow(Z,
                   interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=cmap_obj,
                   alpha=0.2,
                   aspect='auto',
                   origin='lower')

        scatter = plt.scatter(reduced_features[:, 0],
                              reduced_features[:, 1],
                              c=cluster_labels,
                              cmap=cmap_obj,
                              s=100,
                              alpha=0.9,
                              edgecolors='k')

        for i, txt in enumerate(percentage_df.index):
            plt.annotate(txt, (reduced_features[i, 0], reduced_features[i, 1]),
                         fontsize=9)

        plt.title(f'K-Means Clustering of Pottery (K={optimal_k})')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(handles=scatter.legend_elements()[0],
                   labels=[f'Cluster {i}' for i in range(optimal_k)],
                   title="Clusters")
        plt.grid(True)

        cluster_plot_path = os.path.join(output_dir, 'pottery_kmeans_cluster_plot.png')
        plt.savefig(cluster_plot_path)
        plt.close()
        print(f"K-Means cluster plot saved to '{cluster_plot_path}'")

        # 5. Generate cluster assignments and collages
        print("Step 5: Generating cluster assignments and 3D model collages...")
        
        output_file_path = os.path.join(output_dir, 'cluster_assignments.txt')
        collage_paths = []
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write("Cluster Assignments\n")
            print("\n--- Cluster Assignments ---")
            
            for i in range(optimal_k):
                cluster_header = f"\nCluster {i}:\n"
                f.write(cluster_header)
                print(cluster_header.strip())

                members = percentage_df[percentage_df['cluster'] == i].index.tolist()
                members_str = ", ".join(members)
                f.write(members_str + "\n")
                print(members_str)
                
                # Generate collage for this cluster
                if members:  # Only create collage if cluster has members
                    collage_path = create_cluster_collage(
                        pottery_ids=members,
                        pottery_dir=pottery_models_dir,
                        cluster_id=i,
                        output_dir=output_dir
                    )
                    collage_paths.append(collage_path)

            print("--------------------------")
            
        print(f"Cluster assignments have been saved to '{output_file_path}'")
        print(f"Generated {len(collage_paths)} cluster collages in '{output_dir}/'")
        
        # Remove the temporary cluster column for next iteration
        percentage_df = percentage_df.drop('cluster', axis=1)

# --- Main Execution Block ---

if __name__ == "__main__":
    DATASET_ROOT_DIR = "./src/jomon_kaen_dataset/japan"
    POTTERY_MODELS_DIR = "./src/pottery"

    try:
        combined_dataframe = load_combined_qna_data(DATASET_ROOT_DIR,
                                                    POTTERY_MODELS_DIR)

        if not combined_dataframe.empty:
            analyze_and_plot_stacked_emotions(combined_dataframe)
            perform_kmeans_clustering_with_collages(combined_dataframe, POTTERY_MODELS_DIR)
        else:
            print("No data was loaded, so no analysis will be performed.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Could not run analysis due to missing data/directories: {e}")
        print(
            "Please ensure the './src/data' and './src/pottery' directories are set up correctly."
        )