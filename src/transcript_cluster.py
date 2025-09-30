import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import japanize_matplotlib  # For displaying Japanese characters in plots
import os
import sys
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from PIL import Image, ImageDraw, ImageFont
import math
import trimesh
import torch
import neologdn
from transformers import pipeline

try:
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_3D_AVAILABLE = True
except ImportError:
    MATPLOTLIB_3D_AVAILABLE = False

# --- Dictionaries and Constants ---

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
    'UD0028': '93', 'rembak7': 'A'
}

EMOTION_COLOR_MAP_EN = {
    "Interesting and attentional shape": "#00FFFF",
    "Beautiful and artistic": "#00FF00",
    "Strange and incomprehensible": "#FFFF00",
    "Creepy / unsettling / scary": "#FF0000",
    "Feel nothing": "#505050",
    "NO RESPONSE": "#D3D3D3",
}

EMOTION_COLOR_MAP_JP = {
    "面白い・気になる形だ": "#00FFFF",
    "美しい・芸術的だ": "#00FF00",
    "不思議・意味不明": "#FFFF00",
    "不気味・不安・怖い": "#FF0000",
    "何も感じない": "#505050",
    "NO RESPONSE": "#D3D3D3",
}

SHORT_LABELS_JP = {
    "面白い・気になる形だ": "面白い",
    "美しい・芸術的だ": "美しい",
    "不思議・意味不明": "不思議",
    "不気味・不安・怖い": "怖い",
    "何も感じない": "何も感じない",
    "NO RESPONSE": "NO RESPONSE"
}

SHORT_LABELS_EN = {
    "Interesting and attentional shape": "Interesting",
    "Beautiful and artistic": "Beautiful",
    "Strange and incomprehensible": "Strange",
    "Creepy / unsettling / scary": "Scary",
    "Feel nothing": "Feel nothing",
    "NO RESPONSE": "NO RESPONSE"
}

# --- 3D Model Rendering Functions (unchanged from original) ---
def create_simple_pottery_image(pottery_id: str, image_size: tuple = (256, 256)) -> Image.Image:
    img = Image.new('RGB', image_size, color=(240, 245, 250))
    draw = ImageDraw.Draw(img)
    center_x, center_y = image_size[0] // 2, image_size[1] // 2
    base_width, base_height = image_size[0] // 3, image_size[1] // 6
    draw.ellipse([center_x - base_width // 2, center_y + image_size[1] // 4 - base_height // 2, center_x + base_width // 2, center_y + image_size[1] // 4 + base_height // 2], fill=(120, 120, 120), outline=(80, 80, 80))
    body_width, body_height = image_size[0] // 4, image_size[1] // 3
    draw.ellipse([center_x - body_width // 2, center_y - body_height // 2, center_x + body_width // 2, center_y + body_height // 2], fill=(150, 150, 150), outline=(100, 100, 100))
    neck_width, neck_height = image_size[0] // 6, image_size[1] // 8
    draw.ellipse([center_x - neck_width // 2, center_y - image_size[1] // 4 - neck_height // 2, center_x + neck_width // 2, center_y - image_size[1] // 4 + neck_height // 2], fill=(130, 130, 130), outline=(90, 90, 90))
    try: font = ImageFont.truetype("arial.ttf", 16)
    except: font = ImageFont.load_default()
    text_bbox = draw.textbbox((0, 0), pottery_id, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    text_x, text_y = (image_size[0] - text_width) // 2, image_size[1] - text_height - 15
    draw.rectangle([text_x - 5, text_y - 3, text_x + text_width + 5, text_y + text_height + 3], fill=(255, 255, 255, 200), outline=(100, 100, 100))
    draw.text((text_x, text_y), pottery_id, fill=(0, 0, 0), font=font)
    try: small_font = ImageFont.truetype("arial.ttf", 12)
    except: small_font = ImageFont.load_default()
    draw.text((10, 10), "3D Model", fill=(100, 100, 100), font=small_font)
    return img

def render_glb_matplotlib(glb_path: str, output_size: tuple = (512, 512)) -> np.ndarray:
    try:
        import matplotlib
        matplotlib.use('Agg')
        fig = plt.figure(figsize=(output_size[0] / 100, output_size[1] / 100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        mesh = trimesh.load(glb_path)
        if hasattr(mesh, 'geometry'):
            if len(mesh.geometry) == 0: raise ValueError("No geometry found")
            mesh = list(mesh.geometry.values())[0]
        rotation_matrix = trimesh.transformations.rotation_matrix(angle=np.pi / 2, direction=[1, 0, 0], point=[0, 0, 0])
        mesh.apply_transform(rotation_matrix)
        vertices, faces = mesh.vertices, mesh.faces
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, alpha=0.9, cmap='copper', linewidth=0, antialiased=True)
        ax.view_init(elev=10, azim=0)
        max_range = np.array([vertices[:, i].max() - vertices[:, i].min() for i in range(3)]).max() / 2.0
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

def render_glb_front_view(glb_path: str, output_size: tuple = (512, 512)) -> np.ndarray:
    if MATPLOTLIB_3D_AVAILABLE:
        return render_glb_matplotlib(glb_path, output_size)
    print(f"All rendering methods failed for {glb_path}, creating artistic placeholder")
    try:
        pottery_id = os.path.basename(glb_path).split('.')[0]
        return np.array(create_simple_pottery_image(pottery_id, output_size))
    except:
        return np.ones((*output_size[::-1], 3), dtype=np.uint8) * 200

def create_cluster_collage(pottery_ids: list, pottery_dir: str, cluster_id: int, output_dir: str, image_size: tuple = (256, 256), collage_columns: int = None) -> str:
    print(f"Creating collage for Cluster {cluster_id} with {len(pottery_ids)} items...")
    num_items = len(pottery_ids)
    if collage_columns is None:
        collage_columns = min(5, int(math.ceil(math.sqrt(num_items))))
    collage_rows = int(math.ceil(num_items / collage_columns))
    collage = Image.new('RGB', (collage_columns * image_size[0], collage_rows * image_size[1]), color=(240, 240, 240))
    rendered_count = 0
    for idx, pottery_id in enumerate(tqdm(pottery_ids, desc=f"Cluster {cluster_id}")):
        row, col = idx // collage_columns, idx % collage_columns
        glb_files = [f for f in os.listdir(pottery_dir) if f.startswith(pottery_id.split('(')[0]) and f.endswith('.glb')]
        if glb_files:
            try:
                rendered_image = render_glb_front_view(os.path.join(pottery_dir, glb_files[0]), image_size)
                pil_image = Image.fromarray(rendered_image)
                rendered_count += 1
            except Exception as e:
                print(f"Failed to render {pottery_id}: {e}")
                pil_image = create_simple_pottery_image(f"{pottery_id}\n(Render Error)", image_size)
            draw = ImageDraw.Draw(pil_image)
            try: font = ImageFont.truetype("arial.ttf", 14)
            except: font = ImageFont.load_default()
            text_bbox = draw.textbbox((0, 0), pottery_id, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            text_x, text_y = 5, image_size[1] - text_height - 10
            draw.rectangle([text_x, text_y - 2, text_x + text_width + 4, text_y + text_height + 2], fill=(255, 255, 255, 180))
            draw.text((text_x + 2, text_y), pottery_id, fill=(0, 0, 0), font=font)
        else:
            pil_image = create_simple_pottery_image(f"{pottery_id}\n(GLB not found)", image_size)
        collage.paste(pil_image, (col * image_size[0], row * image_size[1]))
    collage_path = os.path.join(output_dir, f"cluster_{cluster_id}.png")
    collage.save(collage_path, "PNG")
    print(f"Collage saved: {collage_path} ({rendered_count}/{len(pottery_ids)} models rendered)")
    return collage_path


# --- Data Loading Functions ---
def load_transcripts(root_dir: str, pottery_models_dir: str, limit: int = 1000):
    """
    Loads transcripts into a dictionary for analysis.
    Returns:
        - dict: A dictionary mapping (pottery_id, session_id) to transcript text.
        - list: A list of unique pottery_ids found.
    """
    root, pottery_path = Path(root_dir), Path(pottery_models_dir)
    if not root.exists(): raise ValueError(f"Root directory not found: {root}")
    if not pottery_path.exists(): raise ValueError(f"Pottery directory not found: {pottery_path}")

    data_paths = []
    pottery_ids_with_nums = [f"{pid}({num})" for pid, num in ASSIGNED_NUMBERS_DICT.items()]
    print(f"\nScanning for raw transcript data paths...")
    limit_dict = {pid: 0 for pid in pottery_ids_with_nums}

    for g in os.listdir(root):
        group_path = root / g
        if not os.path.isdir(group_path): continue
        for s in tqdm(os.listdir(group_path), desc=g):
            session_path = group_path / s
            if not os.path.isdir(session_path): continue
            for p in os.listdir(session_path):
                if p in pottery_ids_with_nums and limit_dict[p] < limit:
                    transcript_path = session_path / p / "final_transcript.txt"
                    if transcript_path.exists():
                        with open(transcript_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        if content.strip():
                            limit_dict[p] += 1
                            data_paths.append({
                                'transcript': str(transcript_path),
                                'ID': p,
                                'SESSION_ID': s
                            })

    print(f"\nLoader finished. Found {len(data_paths)} valid transcript instances.")
    if not data_paths:
        return {}, []

    # Load Transcripts into dictionary
    transcripts_dict = {}
    all_pottery_ids = set()
    for item in tqdm(data_paths, desc="Loading transcript data"):
        try:
            with open(item['transcript'], 'r', encoding='utf-8') as f:
                key = (item['ID'], item['SESSION_ID'])
                transcripts_dict[key] = f.read()
                all_pottery_ids.add(item['ID'])
        except Exception as e:
            print(f"Could not read transcript file {item['transcript']}: {e}", file=sys.stderr)

    return transcripts_dict, sorted(list(all_pottery_ids))


# --- Analysis Functions ---

def calculate_transcript_percentages(transcripts_dict: dict, language: str, model_id: str) -> pd.DataFrame:
    """
    Analyzes transcripts using a zero-shot model and returns emotion percentages per pottery_id.
    """
    if not transcripts_dict:
        print("No transcripts to analyze.")
        return pd.DataFrame()

    print(f"\n--- Starting Transcript Emotion Analysis using '{model_id}' ---")

    # Define labels based on language
    if language == 'japan':
        TARGET_LABELS = ["面白い", "美しい", "不思議", "怖い", "何も感じない"]
    else:
        TARGET_LABELS = ["Interesting", "Beautiful", "Strange", "Scary", "Feel nothing"]

    # Initialize Hugging Face pipeline
    print("Loading Zero-Shot classification model...")
    device_num = 0 if torch.cuda.is_available() else -1
    # device_num = -1
    classifier = pipeline("zero-shot-classification", model=model_id, device=device_num)

    # Prepare data for batch processing
    session_keys = list(transcripts_dict.keys())
    transcript_texts = [transcripts_dict[key] for key in session_keys]

    # Normalize Japanese text if applicable
    if language == 'japan':
        transcript_texts = [neologdn.normalize(text.replace('\n', ' ').replace('　', ' ')) for text in transcript_texts]
    else:
        transcript_texts = [text.replace('\n', ' ') for text in transcript_texts]

    print(f"Classifying {len(transcript_texts)} transcripts...")
    results_generator = (classifier(text, TARGET_LABELS, multi_label=False) for text in transcript_texts)
    all_results = list(tqdm(results_generator, total=len(transcript_texts)))

    # Process results into a DataFrame
    records = []
    for i, result in enumerate(all_results):
        pottery_id, session_id = session_keys[i]
        score_dict = {label: score for label, score in zip(result['labels'], result['scores'])}
        record = {'pottery_id': pottery_id, 'session_id': session_id}
        for label in TARGET_LABELS:
            record[label] = score_dict.get(label, 0) * 100 # Store as percentage
        records.append(record)
    
    session_level_df = pd.DataFrame(records)

    # Aggregate by pottery_id (average percentages across sessions)
    pottery_level_df = session_level_df.drop(columns='session_id').groupby('pottery_id').mean()
    
    print("Transcript analysis complete.")
    return pottery_level_df

def draw_ellipse(points, ax=None, **kwargs):
    ax = ax or plt.gca()
    if len(points) < 2: return
    cov = np.cov(points, rowvar=False)
    if np.isclose(np.linalg.det(cov), 0):
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
        width, height = (x_max - x_min) * 1.05, (y_max - y_min) * 1.05
        ellipse = mpatches.Ellipse(xy=center, width=width or 0.5, height=height or 0.5, angle=0, **kwargs)
        ax.add_patch(ellipse)
        return
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    transformed_points = points @ eigvecs
    x_min, y_min = np.min(transformed_points, axis=0)
    x_max, y_max = np.max(transformed_points, axis=0)
    center_transformed = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
    distances_from_center = transformed_points - center_transformed
    a, b = np.max(np.abs(distances_from_center), axis=0)
    width, height = 2 * a * 1.45, 2 * b * 1.45
    final_center = center_transformed @ eigvecs.T
    ellipse = mpatches.Ellipse(xy=final_center, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)


def perform_transcript_clustering(
    data_to_cluster: pd.DataFrame, 
    pottery_models_dir: str,
):
    """
    Performs K-Means clustering on transcript features and saves all results.
    """
    if data_to_cluster.empty:
        print("Transcript data is empty. No clustering analysis performed.", file=sys.stderr)
        return

    print("\n--- Starting K-Means Clustering Analysis (Transcript Features Only) ---")
    
    print(f"Step 1: Feature matrix created with {data_to_cluster.shape[0]} items and {data_to_cluster.shape[1]} features.")

    print("Step 2: Finding optimal number of clusters using the Elbow Method...")
    inertia = []
    max_k = min(21, len(data_to_cluster))
    k_range = range(1, max_k)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(data_to_cluster)
        inertia.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal K (Transcript Features)')
    plt.grid(True)
    plt.savefig('kmeans_elbow_plot_transcript.png')
    plt.close()
    print("Elbow method plot saved to 'kmeans_elbow_plot_transcript.png'")

    # Run clustering for a range of K values
    for k_val in range(2, max_k + 1):
        if k_val > len(data_to_cluster):
            print(f"Skipping K={k_val} as it is greater than the number of data points.")
            continue
            
        print(f"\n--- Processing for K={k_val} ---")
        output_dir = f'k_{k_val}_transcript'
        os.makedirs(output_dir, exist_ok=True)

        print(f"Step 3: Performing K-Means clustering with K={k_val}...")
        kmeans = KMeans(n_clusters=k_val, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(data_to_cluster)

        # --- SAVE CLUSTER DATA CSV ---
        results_df = data_to_cluster.copy()
        results_df['cluster_assignment'] = cluster_labels
        results_df.to_csv(os.path.join(output_dir, 'cluster_data.csv'))
        print(f"Saved detailed cluster data and assignments to '{os.path.join(output_dir, 'cluster_data.csv')}'")

        print("Step 4: Visualizing clusters using PCA...")
        pca = PCA(n_components=2, random_state=42)
        reduced_features = pca.fit_transform(data_to_cluster)

        fig, ax = plt.subplots(figsize=(14, 10))
        cmap_obj = plt.get_cmap('viridis', k_val)
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap=cmap_obj, s=100, alpha=0.9, edgecolors='k')

        for i in range(k_val):
            points = reduced_features[cluster_labels == i]
            draw_ellipse(points, ax=ax, edgecolor=cmap_obj(i / (k_val - 1) if k_val > 1 else 0), facecolor='none', lw=2, linestyle='--')
        
        for i, txt in enumerate(data_to_cluster.index):
            plt.annotate(txt, (reduced_features[i, 0], reduced_features[i, 1]), fontsize=9)

        plt.title(f'K-Means Clustering of Pottery (K={k_val}, Transcript Features)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(handles=scatter.legend_elements()[0], labels=[f'Cluster {i}' for i in range(k_val)], title="Clusters")
        
        x_min, x_max = reduced_features[:, 0].min(), reduced_features[:, 0].max()
        y_min, y_max = reduced_features[:, 1].min(), reduced_features[:, 1].max()
        x_padding, y_padding = (x_max - x_min) * 0.1, (y_max - y_min) * 0.1
        ax.set_xlim(x_min - (x_padding or 1), x_max + (x_padding or 1))
        ax.set_ylim(y_min - (y_padding or 1), y_max + (y_padding or 1))
        plt.grid(True)
        
        cluster_plot_path = os.path.join(output_dir, 'pottery_kmeans_cluster_plot.png')
        plt.savefig(cluster_plot_path)
        plt.close(fig)
        print(f"K-Means cluster plot saved to '{cluster_plot_path}'")

        print("Step 5: Generating cluster assignments text file and 3D model collages...")
        with open(os.path.join(output_dir, 'cluster_assignments.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Cluster Assignments for K={k_val}\n")
            for i in range(k_val):
                members = data_to_cluster.index[cluster_labels == i].tolist()
                f.write(f"\nCluster {i}:\n" + ", ".join(members) + "\n")
                if members:
                    create_cluster_collage(pottery_ids=members, pottery_dir=pottery_models_dir, cluster_id=i, output_dir=output_dir)

# --- Main Execution Block ---

if __name__ == "__main__":
    # --- USER CONTROLS ---
    # Language setting determines which emotion labels and models are used.
    # Options: 'japan' or 'malaysia'
    SELECTED_LANGUAGE = 'japan'
    # SELECTED_LANGUAGE = 'malaysia'

    # Zero-shot classification model for transcript analysis.
    # Choose a model appropriate for the selected language.
    if SELECTED_LANGUAGE == 'japan':
        TRANSCRIPT_MODEL_ID = 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'
        DATASET_ROOT_DIR = "./src/jomon_kaen_dataset/japan"
    else: # malaysia
        TRANSCRIPT_MODEL_ID = 'cross-encoder/nli-deberta-v3-large'
        DATASET_ROOT_DIR = "./src/jomon_kaen_dataset/malaysia"
        
    POTTERY_MODELS_DIR = "./src/pottery"

    # Pottery Selection: List of base pottery IDs to include/exclude.
    # Leave empty to process all pottery found: POTTERY_SELECTION = []
    POTTERY_SELECTION = [] 
    # POTTERY_SELECTION = ['NM0099', 'NM0175']
    
    # Mode: True to ONLY INCLUDE items in POTTERY_SELECTION,
    #       False to EXCLUDE items in POTTERY_SELECTION.
    INCLUDE_POTTERY = True
    # --- END USER CONTROLS ---

    try:
        # 1. Load Transcript data
        transcripts_dictionary, all_pottery_ids = load_transcripts(DATASET_ROOT_DIR, POTTERY_MODELS_DIR)

        # 2. Filter data if a selection is provided
        if POTTERY_SELECTION:
            print(f"\nFiltering pottery based on selection (Include mode: {INCLUDE_POTTERY})...")
            
            # Extract base IDs from the loaded pottery IDs
            base_id_map = {pid: pid.split('(')[0] for pid in all_pottery_ids}
            
            if INCLUDE_POTTERY:
                filtered_pottery_ids = {pid for pid, base_id in base_id_map.items() if base_id in POTTERY_SELECTION}
            else: # Exclude
                filtered_pottery_ids = {pid for pid, base_id in base_id_map.items() if base_id not in POTTERY_SELECTION}

            print(f"Filtered from {len(all_pottery_ids)} to {len(filtered_pottery_ids)} unique pottery items.")

            # Filter the transcripts dictionary to match the filtered list
            transcripts_dictionary = {k: v for k, v in transcripts_dictionary.items() if k[0] in filtered_pottery_ids}
            print(f"Filtered transcripts dictionary to {len(transcripts_dictionary)} entries.")

        if not transcripts_dictionary:
            raise ValueError("After loading and filtering, the transcripts dictionary is empty. No analysis can be run.")

        # 3. Calculate percentages from Transcripts (This is our feature set for clustering)
        clustering_features_df = calculate_transcript_percentages(
            transcripts_dictionary,
            language=SELECTED_LANGUAGE,
            model_id=TRANSCRIPT_MODEL_ID
        )

        # 4. Perform clustering and save all results
        perform_transcript_clustering(
            data_to_cluster=clustering_features_df,
            pottery_models_dir=POTTERY_MODELS_DIR
        )

    except (FileNotFoundError, ValueError) as e:
        print(f"\nERROR: Could not run analysis due to an error: {e}", file=sys.stderr)
        print("Please ensure your data directories and paths are set up correctly.", file=sys.stderr)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()