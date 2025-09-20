from dotenv import load_dotenv
load_dotenv()

import os
import sys
import io
import math
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import concurrent.futures

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import japanize_matplotlib

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity
import umap

from google import genai

# --- PDF Report Imports ---
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, HRFlowable, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER
from svglib.svglib import svg2rlg

# --- 1. Configure Google AI API Key ---
# IMPORTANT: Replace "YOUR_API_KEY" with your actual Google AI API key.
try:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        api_key = "YOUR_API_KEY"  # <-- PASTE YOUR API KEY HERE
        if api_key == "YOUR_API_KEY":
            print("API Key not found. Please paste your key or set the GOOGLE_API_KEY environment variable.")
            sys.exit()
    genai.configure(api_key=api_key)
except Exception as e:
    print(f"Error configuring Google AI: {e}")
    sys.exit()

# --- Environment/Path Setup ---
FONT_PATH = "C:/Windows/Fonts/msgothic.ttc"
plt.rcParams['font.family'] = 'MS Gothic'
try:
    pdfmetrics.registerFont(TTFont('JapaneseFont', FONT_PATH))
    JAPANESE_FONT = 'JapaneseFont'
except Exception as e:
    print(f"Warning: Could not register font '{FONT_PATH}'. PDF may not render correctly. Error: {e}")
    JAPANESE_FONT = 'Helvetica'

# --- Global Constants ---
LABELS_JP_MAP = {
    "面白い・気になる形だ": "面白い",
    "美しい・芸術的だ": "美しい",
    "不思議・意味不明": "不思議",
    "不気味・不安・怖い": "怖い",
    "何も感じない": "何も感じない",
    "NO RESPONSE": "NO RESPONSE"
}
TARGET_LABELS_JP = ["面白い", "美しい", "不思議", "怖い", "何も感じない"]
LABEL_COLORS = plt.cm.get_cmap('jet', len(TARGET_LABELS_JP))

# --- Worker and Data Loading Functions (Unchanged) ---
worker_styles = None
def init_worker(styles_arg):
    global worker_styles
    worker_styles = styles_arg

def load_data_paths(root=''):
    data_paths = []
    if not Path(root).exists():
        raise ValueError(f"Root directory not found: {root}")
    print(f"\nCHECKING RAW DATA PATHS IN: {root}")
    # ... (rest of the function is unchanged)
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
                if not os.path.isdir(pottery_path): continue
                qa_save_path = pottery_path / "qa_corrected.csv"
                final_transcript_save_path = pottery_path / "final_transcript.txt"
                if qa_save_path.exists() and final_transcript_save_path.exists():
                    with open(final_transcript_save_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if content.strip() != "":
                        data_paths.append({
                            'QA': str(qa_save_path),
                            'TRANSCRIPT': str(final_transcript_save_path),
                            'GROUP': g, 'SESSION_ID': s, 'ID': p
                        })
    print(f"NUMBER OF VALID DATA: {len(data_paths)}")
    return data_paths


# --- NEW: Google Embedding Function ---
def embed_content_google(texts, task_type, model_id='models/embedding-001'):
    """
    Generates embeddings for a list of texts using Google's model,
    handling batching automatically.
    """
    all_embeddings = []
    # Google's API has a limit of 100 documents per request
    batch_size = 100
    
    print(f"Embedding {len(texts)} texts with model: {model_id} (Task: {task_type})")
    
    # Process texts in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Batches"):
        batch = texts[i:i+batch_size]
        try:
            response = genai.embed_content(
                model=model_id,
                content=batch,
                task_type=task_type
            )
            all_embeddings.extend(response['embedding'])
        except Exception as e:
            print(f"An error occurred during batch embedding: {e}")
            # Add placeholder embeddings for the failed batch
            all_embeddings.extend([np.zeros(768)] * len(batch))
            
    return np.array(all_embeddings)

# --- Analysis and Report Generation Functions (Unchanged) ---
def calculate_qa_emotion_percentages(data_paths):
    print("\nCalculating emotion percentages from QA event counts...")
    for data_path in tqdm(data_paths, desc="Processing QA files"):
        try:
            df = pd.read_csv(data_path['QA'])
            df['label'] = df['answer'].str.strip().map(LABELS_JP_MAP)
            counts = df['label'].value_counts()
            total = counts.drop("NO RESPONSE", errors='ignore').sum()
            percentages = {}
            if total > 0:
                for label in TARGET_LABELS_JP:
                    percentages[label] = (counts.get(label, 0) / total) * 100
            else:
                percentages = {label: 0.0 for label in TARGET_LABELS_JP}
            data_path['qa_percentages'] = percentages
        except Exception as e:
            print(f"Warning: Could not process {data_path['QA']}. Error: {e}")
            data_path['qa_percentages'] = {label: 0.0 for label in TARGET_LABELS_JP}
    return data_paths

def create_comparison_plot(qa_pct, embed_pct, title):
    labels, qa_values, embed_values = list(qa_pct.keys()), list(qa_pct.values()), list(embed_pct.values())
    y = np.arange(len(labels))
    height = 0.4
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    fig.suptitle(title, fontsize=14)
    ax1.barh(y, qa_values, height, color=[LABEL_COLORS(i / len(labels)) for i in range(len(labels))])
    ax1.set_title('QA Event Count (%)')
    ax1.set_xlabel('Percentage')
    ax1.set_xlim(0, 100)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.invert_xaxis()
    ax1.yaxis.tick_right()
    ax2.barh(y, embed_values, height, color=[LABEL_COLORS(i / len(labels)) for i in range(len(labels))])
    ax2.set_title('Transcript Embedding Similarity (%)')
    ax2.set_xlabel('Percentage')
    ax2.set_xlim(0, 100)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    buf = io.BytesIO()
    plt.savefig(buf, format='svg')
    buf.seek(0)
    plt.close(fig)
    return buf

def process_pottery_group(pottery_data):
    pottery_id, sessions = pottery_data
    pottery_story = []
    pottery_story.append(Paragraph(f"Analysis for Pottery ID: {pottery_id}", worker_styles['HeaderStyle']))
    pottery_scores = [s['alignment_score'] for s in sessions if 'alignment_score' in s]
    avg_pottery_alignment = np.mean(pottery_scores) if pottery_scores else 0
    pottery_story.append(Paragraph(f"<b>Average Alignment for this Pottery:</b> {avg_pottery_alignment:.3f}", worker_styles['BodyStyle']))
    pottery_story.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=10))
    sessions.sort(key=lambda x: x['SESSION_ID'])
    for i, session in enumerate(sessions):
        pottery_story.append(Paragraph(f"<b>Session:</b> {session['SESSION_ID']} | <b>Alignment:</b> {session.get('alignment_score', 'N/A'):.3f}", worker_styles['BodyStyle']))
        plot_buffer = create_comparison_plot(session['qa_percentages'], session['embedding_percentages'], title=f"{pottery_id} | {session['SESSION_ID']}")
        drawing = svg2rlg(plot_buffer)
        desired_width = 7.5 * inch
        scale_factor = desired_width / drawing.width
        drawing.width *= scale_factor
        drawing.height *= scale_factor
        drawing.scale(scale_factor, scale_factor)
        drawing_table = Table([[drawing]], colWidths=[desired_width])
        table_style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('LEFTPADDING', (0,0), (-1,-1), 0), ('RIGHTPADDING', (0,0), (-1,-1), 0)])
        drawing_table.setStyle(table_style)
        pottery_story.append(drawing_table)
        pottery_story.append(Spacer(1, 0.1 * inch))
        transcript_text = session['transcript_text']
        pottery_story.append(Paragraph("<b>Transcript:</b>", worker_styles['BodyStyle']))
        pottery_story.append(Paragraph(transcript_text, worker_styles['TranscriptStyle']))
        if i < len(sessions) - 1:
            pottery_story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey, spaceBefore=10, spaceAfter=10))
    pottery_story.append(PageBreak())
    return pottery_story
    
def generate_alignment_report(data_paths, model_id, output_filename, max_workers=10, debug=False):
    print(f"\nGenerating PDF report: {output_filename}")
    doc = SimpleDocTemplate(output_filename, pagesize=(8.5 * inch, 11 * inch))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleStyle', fontName=JAPANESE_FONT, fontSize=20, alignment=TA_CENTER, spaceAfter=20))
    styles.add(ParagraphStyle(name='HeaderStyle', fontName=JAPANESE_FONT, fontSize=16, spaceAfter=12, spaceBefore=20))
    styles.add(ParagraphStyle(name='BodyStyle', fontName=JAPANESE_FONT, fontSize=10, leading=14))
    styles.add(ParagraphStyle(name='TranscriptStyle', fontName=JAPANESE_FONT, fontSize=8, leading=12, leftIndent=10, rightIndent=10))
    styles_for_workers = {'HeaderStyle': styles['HeaderStyle'], 'BodyStyle': styles['BodyStyle'], 'TranscriptStyle': styles['TranscriptStyle']}
    story = []
    story.append(Paragraph("QA vs. Transcript Embedding Alignment Report", styles['TitleStyle']))
    story.append(Paragraph(f"Analysis generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')} in Niigata", styles['BodyStyle']))
    story.append(Paragraph(f"Embedding Model Used: {model_id}", styles['BodyStyle']))
    story.append(Spacer(1, 0.25 * inch))
    all_scores = [dp['alignment_score'] for dp in data_paths if 'alignment_score' in dp]
    avg_alignment = np.mean(all_scores) if all_scores else 0
    story.append(Paragraph(f"<b>Overall Average Alignment Score:</b> {avg_alignment:.3f}", styles['HeaderStyle']))
    story.append(Paragraph("<i>(Cosine similarity between QA % vector and Embedding % vector. 1.0 = perfect alignment)</i>", styles['BodyStyle']))
    story.append(PageBreak())
    print("Pre-loading transcripts into memory...")
    transcript_cache = {}
    for dp in tqdm(data_paths, desc="Reading Transcripts"):
        path = dp['TRANSCRIPT']
        if path not in transcript_cache:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    transcript_cache[path] = f.read().replace('\n', '<br/>')
            except FileNotFoundError:
                transcript_cache[path] = "<i>Error: Transcript file not found.</i>"
    data_by_pottery = defaultdict(list)
    for dp in data_paths:
        dp['transcript_text'] = transcript_cache.get(dp['TRANSCRIPT'], "<i>Error loading transcript.</i>")
        data_by_pottery[dp['ID']].append(dp)
    pottery_tasks = sorted(data_by_pottery.items())
    if debug:
        init_worker(styles_for_workers)
        pottery_stories = [process_pottery_group(task) for task in tqdm(pottery_tasks, desc="Debugging Pottery Groups")]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(styles_for_workers,)) as executor:
            results_iterator = executor.map(process_pottery_group, pottery_tasks)
            pottery_stories = list(tqdm(results_iterator, total=len(pottery_tasks), desc="Processing Pottery Groups in Parallel"))
    for story_chunk in pottery_stories:
        story.extend(story_chunk)
    print("All content generated. Building final PDF document...")
    doc.build(story)
    print("PDF report generation complete.")

if __name__ == "__main__":
    root = "./src/jomon_kaen_dataset/japan"
    SELECTED_MODEL_ID = "Google Gemini (embedding-001)"

    if not os.path.exists(root):
        print(f"Error: The directory '{root}' does not exist.")
    elif not os.path.exists(FONT_PATH):
        print(f"Error: The font file '{FONT_PATH}' was not found.")
    else:
        # Step 1: Load data paths (tokenization is no longer needed)
        data_paths = load_data_paths(root)

        # Step 2: Calculate percentages from QA files
        data_paths = calculate_qa_emotion_percentages(data_paths)

        # Step 3: Generate transcript embeddings using Google's model
        transcript_texts = []
        for data_path in tqdm(data_paths, desc='Reading Transcripts for Embedding'):
            with open(data_path['TRANSCRIPT'], 'r', encoding='utf-8') as f:
                transcript_texts.append(f.read())
        
        embeddings = embed_content_google(
            texts=transcript_texts,
            task_type="RETRIEVAL_DOCUMENT" # Use for documents to be retrieved
        )

        # Step 4: Calculate similarity to labels using Google's model
        label_embeddings = embed_content_google(
            texts=TARGET_LABELS_JP,
            task_type="RETRIEVAL_QUERY" # Use for the query/labels
        )
        similarities = cosine_similarity(embeddings, label_embeddings)

        # Step 5: Convert similarities to percentages and calculate alignment
        print("\nCalculating embedding distributions and alignment scores...")
        embedding_percentages_all = softmax(similarities, axis=1) * 100
        for i, data_path in enumerate(data_paths):
            data_path['embedding_percentages'] = {
                label: embedding_percentages_all[i, j]
                for j, label in enumerate(TARGET_LABELS_JP)
            }
            qa_vector = np.array(list(data_path['qa_percentages'].values()))
            embed_vector = np.array(list(data_path['embedding_percentages'].values()))
            if np.linalg.norm(qa_vector) > 0 and np.linalg.norm(embed_vector) > 0:
                score = np.dot(qa_vector, embed_vector) / (np.linalg.norm(qa_vector) * np.linalg.norm(embed_vector))
            else:
                score = 0.0
            data_path['alignment_score'] = score

        # Step 6: Generate the 3D Clustering Visualization
        print("\n--- Generating 3D Clustering Visualization ---")
        cluster_labels = np.argmax(similarities, axis=1)
        print("Running UMAP for 3D visualization...")
        umap_3d = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=42, n_jobs=1).fit_transform(embeddings)
        
        print("Generating 3D plot...")
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        for i, label_text in enumerate(TARGET_LABELS_JP):
            idx = (cluster_labels == i)
            if np.sum(idx) > 0:
                ax.scatter(umap_3d[idx, 0], umap_3d[idx, 1], umap_3d[idx, 2], s=50, alpha=0.7, color=LABEL_COLORS(i), label=label_text)
        
        ax.set_xlabel("UMAP-1", fontsize=12)
        ax.set_ylabel("UMAP-2", fontsize=12)
        ax.set_zlabel("UMAP-3", fontsize=12)
        ax.view_init(elev=10, azim=-45)
        plt.legend(loc="best", fontsize=12)
        title_text = f"類似度クエリに基づく3Dクラスタリング\nModel: {SELECTED_MODEL_ID}"
        plt.title(title_text, fontsize=16)
        plt.tight_layout()
        
        model_name_for_file = "google_embedding_001"
        plot_output_filename = f"cluster_plot_3d_{model_name_for_file}.png"
        plt.savefig(plot_output_filename)
        print(f"\n3D Plot successfully saved to '{plot_output_filename}'")
        plt.show()

        # Step 7: Generate the final PDF report
        matplotlib.use('Agg')
        report_output_filename = f"Alignment_Report_{model_name_for_file}.pdf"
        num_workers = 8 # Adjust as needed
        
        generate_alignment_report(data_paths, SELECTED_MODEL_ID, report_output_filename, max_workers=num_workers, debug=False)