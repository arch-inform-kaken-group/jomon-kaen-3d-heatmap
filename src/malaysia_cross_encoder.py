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
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from scipy.special import softmax
from transformers import pipeline
from sentence_transformers import SentenceTransformer, CrossEncoder
import umap
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, HRFlowable, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from svglib.svglib import svg2rlg

font_path='C:/Windows/Fonts/simhei.ttf'

try:
    pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
    pdfmetrics.registerFont(TTFont('ChineseFont-Bold', font_path))
    CHINESE_FONT = 'ChineseFont'
    CHINESE_FONT_BOLD = 'ChineseFont-Bold'
    print(
        f"Successfully registered Chinese font '{font_path}' for PDF generation."
    )
except Exception as e:
    print(
        f"Error registering font: {e}. Chinese text in PDF may not render correctly."
    )
    CHINESE_FONT = 'Helvetica'
    CHINESE_FONT_BOLD = 'Helvetica-Bold'

# --- Global Constants (ENGLISH VERSION) ---
LABELS_EN_MAP = {
    "Interesting and attentional shape": "Interesting",
    "Beautiful and artistic": "Beautiful",
    "Strange and incomprehensible": "Strange",
    "Creepy / unsettling / scary": "Scary",
    "Feel nothing": "Feel nothing",
    "NO RESPONSE": "NO RESPONSE"
}
# Using simple labels is better for Cross-Encoders and for the final report
TARGET_LABELS_EN = ["Interesting", "Beautiful", "Strange", "Scary", "Feel nothing"]
LABEL_COLORS = plt.cm.get_cmap('jet', len(TARGET_LABELS_EN))

worker_styles = None

def init_worker(styles_arg):
    global worker_styles
    worker_styles = styles_arg

# --- Data Loading and Preprocessing Functions ---
# (load_data_paths function remains the same)
def load_data_paths(root=''):
    data_paths = []
    if not Path(root).exists():
        raise ValueError(f"Root directory not found: {root}")
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
                            'QA': str(qa_save_path),
                            'TRANSCRIPT': str(final_transcript_save_path),
                            'GROUP': g,
                            'SESSION_ID': s,
                            'ID': p
                        })
    print(f"NUMBER OF VALID DATA: {len(data_paths)} [ Has QA & TRANSCRIPT that are not empty ]")
    return data_paths


# --- Analysis and Report Generation Functions ---
# (calculate_qa_emotion_percentages, create_comparison_plot, process_pottery_group, generate_alignment_report remain the same)
def calculate_qa_emotion_percentages(data_paths):
    print("\nCalculating emotion percentages from QA event counts...")
    for data_path in tqdm(data_paths, desc="Processing QA files"):
        try:
            df = pd.read_csv(data_path['QA'])
            df['label'] = df['answer'].str.strip().map(LABELS_EN_MAP)
            counts = df['label'].value_counts()
            total = counts.drop("NO RESPONSE", errors='ignore').sum()
            percentages = {}
            if total > 0:
                for label in TARGET_LABELS_EN:
                    percentages[label] = (counts.get(label, 0) / total) * 100
            else:
                percentages = {label: 0.0 for label in TARGET_LABELS_EN}
            data_path['qa_percentages'] = percentages
        except Exception as e:
            print(f"Warning: Could not process {data_path['QA']}. Error: {e}")
            data_path['qa_percentages'] = {label: 0.0 for label in TARGET_LABELS_EN}
    return data_paths

def create_comparison_plot(qa_pct, embed_pct, title):
    labels = list(qa_pct.keys())
    qa_values = list(qa_pct.values())
    embed_values = list(embed_pct.values())
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
    ax2.set_title('Transcript Classification Score (%)') # Renamed for clarity
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
        pottery_story.append(
            Paragraph(
                f"<b>Session:</b> {session['SESSION_ID']} | <b>Alignment:</b> {session.get('alignment_score', 'N/A'):.3f}",
                worker_styles['BodyStyle']))
        plot_buffer = create_comparison_plot(
            session['qa_percentages'],
            session['embedding_percentages'],
            title=f"{pottery_id} | {session['SESSION_ID']}")
        drawing = svg2rlg(plot_buffer)
        desired_width = 7.5 * inch
        scale_factor = desired_width / drawing.width
        drawing.width *= scale_factor
        drawing.height *= scale_factor
        drawing.scale(scale_factor, scale_factor)
        drawing_table = Table([[drawing]], colWidths=[desired_width])
        table_style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                  ('LEFTPADDING', (0,0), (-1,-1), 0),
                                  ('RIGHTPADDING', (0,0), (-1,-1), 0)])
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
    """Generates the complete PDF report using the specified custom font."""
    print(f"\nGenerating PDF report: {output_filename}")

    doc = SimpleDocTemplate(output_filename, pagesize=(8.5 * inch, 11 * inch))
    
    # --- Define PDF Styles using the Custom Font ---
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleStyle', fontName=CHINESE_FONT_BOLD, fontSize=20, alignment=TA_CENTER, spaceAfter=20))
    styles.add(ParagraphStyle(name='HeaderStyle', fontName=CHINESE_FONT_BOLD, fontSize=16, spaceAfter=12, spaceBefore=20))
    styles.add(ParagraphStyle(name='BodyStyle', fontName=CHINESE_FONT, fontSize=10, leading=14))
    styles.add(ParagraphStyle(name='TranscriptStyle', fontName=CHINESE_FONT, fontSize=8, leading=12, leftIndent=10, rightIndent=10))
    
    styles_for_workers = {
        'HeaderStyle': styles['HeaderStyle'],
        'BodyStyle': styles['BodyStyle'],
        'TranscriptStyle': styles['TranscriptStyle']
    }
    
    story = []
    
    # --- Report Title Page ---
    story.append(Paragraph("QA vs. Transcript Embedding Alignment Report", styles['TitleStyle']))
    story.append(Paragraph(f"Analysis generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['BodyStyle']))
    story.append(Paragraph(f"Embedding Model Used: {model_id}", styles['BodyStyle']))
    story.append(Spacer(1, 0.25 * inch))
    all_scores = [dp['alignment_score'] for dp in data_paths if 'alignment_score' in dp]
    avg_alignment = np.mean(all_scores) if all_scores else 0
    story.append(Paragraph(f"<b>Overall Average Alignment Score:</b> {avg_alignment:.3f}", styles['HeaderStyle']))
    story.append(Paragraph("<i>(Cosine similarity between QA % vector and Embedding % vector. 1.0 = perfect alignment)</i>", styles['BodyStyle']))
    story.append(PageBreak())

    # --- Pre-load transcripts ---
    print("Pre-loading transcripts into memory...")
    transcript_cache = {}
    for dp in tqdm(data_paths, desc="Reading Transcripts"):
        path = dp['TRANSCRIPT']
        if path not in transcript_cache:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    transcript_cache[path] = f.read().replace('\n', ' ')
            except FileNotFoundError:
                transcript_cache[path] = "<i>Error: Transcript file not found.</i>"
    
    # --- Group data and process in parallel ---
    data_by_pottery = defaultdict(list)
    for dp in data_paths:
        dp['transcript_text'] = transcript_cache.get(dp['TRANSCRIPT'], "<i>Error loading transcript.</i>")
        data_by_pottery[dp['ID']].append(dp)
        
    pottery_stories = []
    pottery_tasks = sorted(data_by_pottery.items())

    if debug:
        print("\nRunning in debug mode (sequentially)...")
        init_worker(styles_for_workers)
        for task in tqdm(pottery_tasks, desc="Debugging Pottery Groups"):
            pottery_stories.append(process_pottery_group(task))
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=init_worker,
            initargs=(styles_for_workers,)
        ) as executor:
            results_iterator = executor.map(process_pottery_group, pottery_tasks)
            pottery_stories = list(tqdm(results_iterator, total=len(pottery_tasks), desc="Processing Pottery Groups in Parallel"))

    # --- Assemble and build the final PDF ---
    for story_chunk in pottery_stories:
        story.extend(story_chunk)
            
    print("All content generated. Building final PDF document...")
    doc.build(story)
    print("PDF report generation complete.")


if __name__ == "__main__":
    # root = r"D:\storage\jomon_kaen\jomon_kaen_dataset\malaysia"
    root = "./src/jomon_kaen_dataset/malaysia"
    
    # --- MODEL SELECTION: Using a Cross-Encoder for higher accuracy ---
    # SELECTED_MODEL_ID = 'cross-encoder/ms-marco-MiniLM-L12-v2'
    SELECTED_MODEL_ID = 'cross-encoder/nli-deberta-v3-large'

    if not os.path.exists(root):
        print(f"Error: The directory '{root}' does not exist.")
    else:
        # Step 1 & 2: Load data paths and calculate QA percentages
        data_paths = load_data_paths(root)
        data_paths = calculate_qa_emotion_percentages(data_paths)

        # --- NEW & EFFICIENT METHOD: Step 3 & 4 using Hugging Face Pipeline ---
        print(f"\nLoading Hugging Face zero-shot classification pipeline...")
        # device_num = 0 if torch.cuda.is_available() else -1
        device_num = -1
        # Use the pipeline interface, which is more efficient for this task
        classifier = pipeline("zero-shot-classification", model=SELECTED_MODEL_ID, device=device_num)
        
        # Prepare the transcripts
        print("Preparing transcripts for classification...")
        transcripts = []
        for data_path in data_paths:
            with open(data_path['TRANSCRIPT'], 'r', encoding='utf-8') as f:
                transcripts.append(f.read().replace('\n', ' '))

        print(f"Classifying {len(transcripts)} transcripts (this is batched and much faster)...")
        # A single, batched call to the pipeline for all transcripts
        results_generator = (classifier(transcript, TARGET_LABELS_EN, multi_label=False) for transcript in transcripts)
        all_results = list(tqdm(results_generator, total=len(transcripts)))

        # Convert the results into a single score matrix
        scores_matrix = np.zeros((len(transcripts), len(TARGET_LABELS_EN)))
        for i, result in enumerate(all_results):
            # The pipeline returns the labels and scores in a certain order. We match it.
            score_dict = {label: score for label, score in zip(result['labels'], result['scores'])}
            # Fill the matrix in the correct order (matching TARGET_LABELS_EN)
            for j, label in enumerate(TARGET_LABELS_EN):
                scores_matrix[i, j] = score_dict.get(label, 0)
        
        # --- Step 5: Convert scores to percentages and calculate alignment ---
        print("\nCalculating classification distributions and alignment scores...")
        # The pipeline returns softmaxed probabilities by default, so we can use them directly
        embedding_percentages_all = scores_matrix * 100
        
        for i, data_path in enumerate(data_paths):
            data_path['embedding_percentages'] = {
                label: embedding_percentages_all[i, j] for j, label in enumerate(TARGET_LABELS_EN)
            }
            qa_vector = np.array(list(data_path['qa_percentages'].values()))
            embed_vector = np.array(list(data_path['embedding_percentages'].values()))
            if np.linalg.norm(qa_vector) > 0 and np.linalg.norm(embed_vector) > 0:
                score = np.dot(qa_vector, embed_vector) / (np.linalg.norm(qa_vector) * np.linalg.norm(embed_vector))
            else:
                score = 0.0
            data_path['alignment_score'] = score
            
        # --- Step 6 and 7 (Plotting and PDF) ---
        print("\n--- Generating Embeddings for 3D Visualization ONLY ---")
        vis_model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2', device=f"cuda:{device_num}" if device_num != -1 else "cpu")
        embeddings_for_visualization = vis_model.encode(transcripts, show_progress_bar=True)
        
        print("\n--- Generating 3D Clustering Visualization ---")
        # Cluster labels are now based on the superior cross-encoder results
        cluster_labels = np.argmax(scores_matrix, axis=1) 
        
        print("Running UMAP for 3D visualization...")
        umap_3d = umap.UMAP(n_components=3, n_neighbors=45, min_dist=0.0,
                             metric="cosine", random_state=42, n_jobs=1).fit_transform(embeddings_for_visualization)
        
        print("Generating 3D plot...")
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        for i, label_text in enumerate(TARGET_LABELS_EN):
            idx = (cluster_labels == i)
            if np.sum(idx) > 0:
                ax.scatter(umap_3d[idx, 0], umap_3d[idx, 1], umap_3d[idx, 2],
                            s=50, alpha=0.7, color=LABEL_COLORS(i), label=label_text)
        ax.set_xlabel("UMAP-1", fontsize=12)
        ax.set_ylabel("UMAP-2", fontsize=12)
        ax.set_zlabel("UMAP-3", fontsize=12)
        plt.legend(loc="best", fontsize=12)
        title_text = f"3D Clustering based on Cross-Encoder Classification\nModel: {SELECTED_MODEL_ID}"
        plt.title(title_text, fontsize=16)
        plt.tight_layout()
        model_name_for_file = SELECTED_MODEL_ID.replace('/', '_')
        plot_output_filename = f"cluster_plot_3d_{model_name_for_file}_zeroshot.png"
        plt.savefig(plot_output_filename)
        print(f"\n3D Plot successfully saved to '{plot_output_filename}'")
        plt.show()

        matplotlib.use('Agg')

        # Step 7: Generate the final PDF report
        report_output_filename = f"Cross-Encoder_MY_Alignment_Report_{model_name_for_file}_zeroshot.pdf"
        num_workers = 8
        generate_alignment_report(data_paths,
                                    SELECTED_MODEL_ID,
                                    report_output_filename,
                                    max_workers=num_workers,
                                    debug=False)