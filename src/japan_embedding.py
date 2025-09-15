import os
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import japanize_matplotlib

import numpy as np
from tqdm import tqdm
import pandas as pd

import neologdn
from sudachipy import tokenizer
from sudachipy import dictionary

from sentence_transformers import SentenceTransformer
import umap
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv

load_dotenv()

FONT_PATH = "C:/Windows/Fonts/msgothic.ttc"

ASSIGNED_NUMBERS_DICT = {
    'AS0001': '1', 'FH0008': '2', 'IN0003': '3', 'IN0008': '4', 'IN0009': '5', 'IN0017': '6',
    'IN0081': '7', 'IN0104': '8', 'IN0135': '9', 'IN0148': '10', 'IN0220': '11', 'IN0228': '12',
    'IN0232': '13', 'IN0239': '14', 'IN0277': '15', 'MY0001': '16', 'MY0002': '17', 'MY0004': '18',
    'MY0006': '19', 'MY0007': '20', 'ND0001': '21', 'NM0001': '22', 'NM0002': '23', 'NM0009': '24',
    'NM0010': '25', 'NM0014': '26', 'NM0015': '27', 'NM0017': '28', 'NM0041': '29', 'NM0049': '30',
    'NM0066': '31', 'NM0070': '32', 'NM0072': '33', 'NM0073': '34', 'NM0079': '35', 'NM0080': '36',
    'NM0099': '37', 'NM0106': '38', 'NM0133': '39', 'NM0135': '40', 'NM0144': '41', 'NM0154': '42',
    'NM0156': '43', 'NM0159': '44', 'NM0168': '45', 'NM0173': '46', 'NM0175': '47', 'NM0189': '48',
    'NM0191': '49', 'NM0206': '50', 'SB0002': '51', 'SB0004': '52', 'SI0001': '53', 'SJ0503': '54',
    'SJ0504': '55', 'SK0001': '56', 'SK0002': '57', 'SK0003': '58', 'SK0004': '59', 'SK0005': '60',
    'SK0013': '61', 'SS0001': '62', 'TJ0004': '63', 'TJ0005': '64', 'TJ0010': '65', 'TK0002': '66',
    'TK0048': '67', 'TK0057': '68', 'UD0001': '69', 'UD0003': '70', 'UD0005': '71', 'UD0006': '72',
    'UD0011': '73', 'UD0013': '74', 'UD0014': '75', 'UD0016': '76', 'UD0023': '77', 'UD0302': '78',
    'UD0304': '79', 'UD0308': '80', 'UD0318': '81', 'UD0322': '82', 'UD0411': '83', 'UD0412': '84',
    'UK0001': '85', 'IN0295': '86', 'IN0306': '87', 'MH0037': '88', 'NM0239': '89', 'NZ0001': '90',
    'SK0035': '91', 'TK0020': '92', 'UD0028': '93'
}

LABELS_JP = {
    "面白い・気になる形だ": "面白い", "美しい・芸術的だ": "美しい", "不思議・意味不明": "不思議",
    "不気味・不安・怖い": "怖い", "何も感じない": "何も感じない", "NO RESPONSE": "NO RESPONSE"
}

def get_pottery_id_list():
    return [f"{pid}({num})" for pid, num in ASSIGNED_NUMBERS_DICT.items()]

def load_data_paths(root=''):
    data_paths = []
    if not Path(root).exists():
        raise (ValueError(f"Root directory not found: {root}"))

    print(f"\nCHECKING RAW DATA PATHS")
    group_keys = os.listdir(root)
    for g in group_keys:
        group_path = Path(root) / g
        if not os.path.isdir(group_path): continue
        session_keys = os.listdir(group_path)
        for s in tqdm(session_keys, desc=g):
            session_path = group_path / s
            if not os.path.isdir(session_path): continue
            pottery_keys = os.listdir(session_path)
            for p in pottery_keys:
                pottery_path = session_path / p
                if not os.path.isdir(pottery_path) or p in ['language.txt', 'gender.txt']:
                    continue
                qa_save_path = pottery_path / "qa_corrected.csv"
                final_transcript_save_path = pottery_path / "final_transcript.txt"
                if qa_save_path.exists() and final_transcript_save_path.exists():
                    with open(final_transcript_save_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if content.strip() != "":
                        data_paths.append({
                            'QA': str(qa_save_path), 'TRANSCRIPT': str(final_transcript_save_path),
                            'GROUP': g, 'SESSION_ID': s, 'ID': p
                        })
    print(f"NUMBER OF VALID DATA: {len(data_paths)} [ Has QA & TRANSCRIPT that are not empty ]")
    return data_paths

def tokenize_japanese(data_paths):
    tokenizer_obj = dictionary.Dictionary().create()
    mode = tokenizer.Tokenizer.SplitMode.A
    parts_of_speech_to_keep = {"名詞", "動詞", "形容詞", "形状詞", "副詞"}
    for data_path in tqdm(data_paths, desc='TOKENIZING JAPANESE COMMENTS'):
        with open(data_path['TRANSCRIPT'], 'r', encoding='utf-8') as f:
            content = f.read()
        normalized_text = neologdn.normalize(content)
        tokens = tokenizer_obj.tokenize(normalized_text, mode)
        data_path['TOKENS'] = [m.normalized_form() for m in tokens if m.part_of_speech()[0] in parts_of_speech_to_keep]
    return data_paths

def embed_tokens(data_paths, model, model_id, mode='fulltext'):
    sentences = []
    # e5 models perform best with a "query: " or "passage: " prefix.
    # We use "query: " here for similarity comparison.
    use_prefix = 'e5' in model_id.lower()
    
    for data_path in tqdm(data_paths, desc='PREPARING FOR EMBEDDING'):
        if mode == 'fulltext':
            with open(data_path['TRANSCRIPT'], 'r', encoding='utf-8') as f:
                content = f.read()
        elif mode == 'tokens':
            content = " ".join(data_path['TOKENS'])
        
        if use_prefix:
            sentences.append("query: " + content)
        else:
            sentences.append(content)

    print(f"Encoding {len(sentences)} sentences with model: {model_id}...")
    embeddings = model.encode(
        sentences, batch_size=128, 
        convert_to_numpy=True,
        normalize_embeddings=True, 
        show_progress_bar=True
    )
    return embeddings

if __name__ == "__main__":
    # root = "./src/jomon_kaen_dataset/japan"
    root = r"D:\storage\jomon_kaen\jomon_kaen_dataset\japan"

    # Choose one of the following model IDs to experiment with.
    # The first time you run a new model, it will be downloaded automatically.
    
    # 1. Lightweight, recent model from Google.
    # SELECTED_MODEL_ID = 'google/embeddinggemma-300m'
    
    # 2. High-performance multilingual model (requires "query: " prefix, handled automatically).
    # SELECTED_MODEL_ID = 'intfloat/multilingual-e5-large'
    
    # 3. Popular, well-balanced multilingual model.
    SELECTED_MODEL_ID = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    
    # 4. A model specifically trained on Japanese.
    # SELECTED_MODEL_ID = 'pkshatech/GLuCoSE-base-ja'

    if not os.path.exists(root):
        print(f"Error: The directory '{root}' does not exist.")
    elif not os.path.exists(FONT_PATH):
        print(f"Error: The font file '{FONT_PATH}' was not found.")
    else:
        data_paths = load_data_paths(root)
        data_paths = tokenize_japanese(data_paths)

        print(f"\nLoading Sentence Transformer model: {SELECTED_MODEL_ID}...")
        device = "cuda"
        model = SentenceTransformer(SELECTED_MODEL_ID, device=device)
        
        embeddings = embed_tokens(data_paths, model, SELECTED_MODEL_ID)
        # embeddings = embed_tokens(data_paths, model, SELECTED_MODEL_ID, mode='tokens')

        # --- CLUSTERING BASED ON SIMILARITY ---
        target_labels_jp = ["面白い", "美しい", "不思議", "怖い", "何も感じない"]
        print(f"\nClustering based on vector similarity to: {target_labels_jp}")

        if 'e5' in SELECTED_MODEL_ID.lower():
            print("Applying 'query: ' prefix to labels for e5 model.")
            prefixed_labels = ["query: " + label for label in target_labels_jp]
        else:
            prefixed_labels = target_labels_jp
            
        label_embeddings = model.encode(
            prefixed_labels, convert_to_numpy=True, normalize_embeddings=True
        )
        similarities = cosine_similarity(embeddings, label_embeddings)
        labels = np.argmax(similarities, axis=1)
        
        # --- 3D UMAP VISUALIZATION ---
        print("Running UMAP for 3D visualization...")
        umap_3d = umap.UMAP(
            n_components=3, # Set to 3 for 3D
            n_neighbors=15, min_dist=0.0, metric="cosine",
            random_state=42, n_jobs=1
        ).fit_transform(embeddings)

        print("Generating 3D plot...")
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d') # Enable 3D projection
        
        colors = plt.cm.get_cmap('jet', len(target_labels_jp))

        for i, label_text in enumerate(target_labels_jp):
            idx = (labels == i)
            if np.sum(idx) > 0:
                ax.scatter(
                    umap_3d[idx, 0], umap_3d[idx, 1], umap_3d[idx, 2], # Use x, y, z
                    s=50, alpha=0.7, color=colors(i), label=label_text
                )
        
        ax.set_xlabel("UMAP-1", fontsize=12)
        ax.set_ylabel("UMAP-2", fontsize=12)
        ax.set_zlabel("UMAP-3", fontsize=12) # Add z-axis label
        
        plt.legend(loc="best", fontsize=12)
        title_text = f"類似度クエリに基づく3Dクラスタリング\nModel: {SELECTED_MODEL_ID}"
        plt.title(title_text, fontsize=16)
        plt.tight_layout()
        
        # Create a filename-safe version of the model ID
        model_name_for_file = SELECTED_MODEL_ID.replace('/', '_')
        output_filename = f"japanese_transcript_cluster_3d_{model_name_for_file}.png"
        plt.savefig(output_filename)
        print(f"\n3D Plot successfully saved to '{output_filename}'")

        plt.show()