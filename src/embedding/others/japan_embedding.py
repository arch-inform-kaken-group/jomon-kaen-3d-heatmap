import os
import sys
import io
import math  # Added for chunksize calculation
from collections import defaultdict
from pathlib import Path
from datetime import datetime

import concurrent.futures  # Corrected import name
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import japanize_matplotlib

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from scipy.special import softmax

import neologdn
from sudachipy import tokenizer
from sudachipy import dictionary

from sentence_transformers import SentenceTransformer
import umap
from sklearn.metrics.pairwise import cosine_similarity

# --- PDF Report Imports ---
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, HRFlowable, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from svglib.svglib import svg2rlg

# --- Environment/Path Setup ---
FONT_PATH = "C:/Windows/Fonts/msgothic.ttc"
# Register the font for matplotlib and reportlab
plt.rcParams['font.family'] = 'MS Gothic'
try:
    pdfmetrics.registerFont(TTFont('JapaneseFont', FONT_PATH))
    JAPANESE_FONT = 'JapaneseFont'
    # print(f"Successfully registered font '{FONT_PATH}' for PDF generation.")
except Exception as e:
    print(
        f"Warning: Could not register font '{FONT_PATH}'. PDF may not render Japanese characters correctly. Error: {e}"
    )
    JAPANESE_FONT = 'Helvetica'

# --- Global Constants ---
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
    'UD0028': '93'
}

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

worker_styles = None


def init_worker(styles_arg):
    """
    Initializer for each worker process. This function receives the styles
    dictionary once and stores it in a global variable for that specific
    process, avoiding repeated and problematic serialization.
    """
    global worker_styles
    worker_styles = styles_arg


# --- Data Loading and Preprocessing Functions ---
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
                if not os.path.isdir(pottery_path) or p in [
                        'language.txt', 'gender.txt'
                ]:
                    continue
                qa_save_path = pottery_path / "qa_corrected.csv"
                final_transcript_save_path = pottery_path / "final_transcript.txt"
                if qa_save_path.exists() and final_transcript_save_path.exists(
                ):
                    with open(final_transcript_save_path,
                              'r',
                              encoding='utf-8') as f:
                        content = f.read()
                    if content.strip() != "":
                        data_paths.append({
                            'QA':
                            str(qa_save_path),
                            'TRANSCRIPT':
                            str(final_transcript_save_path),
                            'GROUP':
                            g,
                            'SESSION_ID':
                            s,
                            'ID':
                            p
                        })
    print(
        f"NUMBER OF VALID DATA: {len(data_paths)} [ Has QA & TRANSCRIPT that are not empty ]"
    )
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
        data_path['TOKENS'] = [
            m.normalized_form() for m in tokens
            if m.part_of_speech()[0] in parts_of_speech_to_keep
        ]
    return data_paths


def embed_tokens(data_paths, model, model_id, mode='fulltext'):
    sentences = []
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
        sentences,
        batch_size=128,
        convert_to_numpy=True,
        #   normalize_embeddings=True,
        normalize_embedings=False,
        show_progress_bar=True)
    return embeddings


# --- Analysis and Report Generation Functions ---


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
            data_path['qa_percentages'] = {
                label: 0.0
                for label in TARGET_LABELS_JP
            }
    return data_paths


def create_comparison_plot(qa_pct, embed_pct, title):
    """
    Creates a comparison bar plot and returns it as an in-memory SVG file.
    This function is self-contained and safe to call from multiple processes.
    """
    labels = list(qa_pct.keys())
    qa_values = list(qa_pct.values())
    embed_values = list(embed_pct.values())

    y = np.arange(len(labels))
    height = 0.4

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    fig.suptitle(title, fontsize=14)

    # Left Plot (QA)
    ax1.barh(y,
             qa_values,
             height,
             color=[LABEL_COLORS(i / len(labels)) for i in range(len(labels))])
    ax1.set_title('QA Event Count (%)')
    ax1.set_xlabel('Percentage')
    ax1.set_xlim(0, 100)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.invert_xaxis()
    ax1.yaxis.tick_right()

    # Right Plot (Embeddings)
    ax2.barh(y,
             embed_values,
             height,
             color=[LABEL_COLORS(i / len(labels)) for i in range(len(labels))])
    ax2.set_title('Transcript Embedding Similarity (%)')
    ax2.set_xlabel('Percentage')
    ax2.set_xlim(0, 100)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save to an in-memory buffer as an SVG
    buf = io.BytesIO()
    plt.savefig(buf, format='svg')
    buf.seek(0)

    # Close the figure to free up memory immediately
    plt.close(fig)
    return buf


def process_pottery_group(pottery_data):
    """
    Worker function that creates all ReportLab flowables for a SINGLE pottery_id group.
    It loops through all sessions for this group sequentially within this process.
    """
    pottery_id, sessions = pottery_data
    pottery_story = []

    # --- Add the header for this entire pottery section ---
    pottery_story.append(
        Paragraph(f"Analysis for Pottery ID: {pottery_id}",
                  worker_styles['HeaderStyle']))
    pottery_scores = [
        s['alignment_score'] for s in sessions if 'alignment_score' in s
    ]
    avg_pottery_alignment = np.mean(pottery_scores) if pottery_scores else 0
    pottery_story.append(
        Paragraph(
            f"<b>Average Alignment for this Pottery:</b> {avg_pottery_alignment:.3f}",
            worker_styles['BodyStyle']))
    pottery_story.append(
        HRFlowable(width="100%",
                   thickness=1,
                   color=colors.black,
                   spaceAfter=10))

    sessions.sort(key=lambda x: x['SESSION_ID'])

    # --- Loop through all sessions for THIS group sequentially ---
    for i, session in enumerate(sessions):
        # 1. Create the header paragraph for the session
        pottery_story.append(
            Paragraph(
                f"<b>Session:</b> {session['SESSION_ID']} | <b>Alignment:</b> {session.get('alignment_score', 'N/A'):.3f}",
                worker_styles['BodyStyle']))

        # 2. Generate the plot
        plot_buffer = create_comparison_plot(
            session['qa_percentages'],
            session['embedding_percentages'],
            title=f"{pottery_id} | {session['SESSION_ID']}")

        # 3. Convert SVG and scale it
        drawing = svg2rlg(plot_buffer)
        desired_width = 7.5 * inch
        scale_factor = desired_width / drawing.width
        drawing.width = drawing.width * scale_factor
        drawing.height = drawing.height * scale_factor
        drawing.scale(scale_factor, scale_factor)

        # --- THIS IS THE FIX FOR CENTERING ---
        # 4. Wrap the drawing in a single-cell table to control alignment
        drawing_in_list = [[drawing]]

        # Create a table with the drawing inside
        # colWidths specifies the width of the column, which we set to our desired graph width
        drawing_table = Table(drawing_in_list, colWidths=[desired_width])

        # Create a style that applies to all cells (from top-left (0,0) to bottom-right (-1,-1))
        # and sets the alignment to CENTER
        table_style = TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),  # Optional: remove padding
            ('RIGHTPADDING', (0, 0), (-1, -1), 0)
        ])  # Optional: remove padding

        drawing_table.setStyle(table_style)

        # Append the TABLE to the story, not the drawing directly
        pottery_story.append(drawing_table)
        # --- END OF FIX ---

        pottery_story.append(Spacer(1, 0.1 * inch))

        # 5. Add transcript text
        transcript_text = session['transcript_text']
        pottery_story.append(
            Paragraph("<b>Transcript:</b>", worker_styles['BodyStyle']))
        pottery_story.append(
            Paragraph(transcript_text, worker_styles['TranscriptStyle']))

        # 6. Add a separator between sessions
        if i < len(sessions) - 1:
            pottery_story.append(
                HRFlowable(width="100%",
                           thickness=0.5,
                           color=colors.grey,
                           spaceBefore=10,
                           spaceAfter=10))

    # Add a page break after the entire section is done
    pottery_story.append(PageBreak())

    return pottery_story


def generate_alignment_report(data_paths,
                              model_id,
                              output_filename,
                              max_workers=10,
                              debug=False):
    """
    Generates the complete PDF report by parallelizing work at the pottery_id level
    for much greater efficiency.
    """
    print(f"\nGenerating PDF report: {output_filename}")
    doc = SimpleDocTemplate(output_filename, pagesize=(8.5 * inch, 11 * inch))

    # --- 1. Styles Setup ---
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(name='TitleStyle',
                       fontName=JAPANESE_FONT,
                       fontSize=20,
                       alignment=TA_CENTER,
                       spaceAfter=20))
    styles.add(
        ParagraphStyle(name='HeaderStyle',
                       fontName=JAPANESE_FONT,
                       fontSize=16,
                       spaceAfter=12,
                       spaceBefore=20))
    styles.add(
        ParagraphStyle(name='BodyStyle',
                       fontName=JAPANESE_FONT,
                       fontSize=10,
                       leading=14))
    styles.add(
        ParagraphStyle(name='TranscriptStyle',
                       fontName=JAPANESE_FONT,
                       fontSize=8,
                       leading=12,
                       leftIndent=10,
                       rightIndent=10))

    # Create the simple, pickle-safe dictionary for workers
    styles_for_workers = {
        'HeaderStyle': styles['HeaderStyle'],
        'BodyStyle': styles['BodyStyle'],
        'TranscriptStyle': styles['TranscriptStyle']
    }

    story = []

    # --- 2. Report Header ---
    story.append(
        Paragraph("QA vs. Transcript Embedding Alignment Report",
                  styles['TitleStyle']))
    story.append(
        Paragraph(
            f"Analysis generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')} in Petaling Jaya",
            styles['BodyStyle']))
    story.append(
        Paragraph(f"Embedding Model Used: {model_id}", styles['BodyStyle']))
    story.append(Spacer(1, 0.25 * inch))
    all_scores = [
        dp['alignment_score'] for dp in data_paths if 'alignment_score' in dp
    ]
    avg_alignment = np.mean(all_scores) if all_scores else 0
    story.append(
        Paragraph(
            f"<b>Overall Average Alignment Score:</b> {avg_alignment:.3f}",
            styles['HeaderStyle']))
    story.append(
        Paragraph(
            "<i>(Cosine similarity between QA % vector and Embedding % vector. 1.0 = perfect alignment)</i>",
            styles['BodyStyle']))
    story.append(PageBreak())

    # --- 3. Pre-load all transcripts into memory ---
    print("Pre-loading transcripts into memory...")
    transcript_cache = {}
    for dp in tqdm(data_paths, desc="Reading Transcripts"):
        path = dp['TRANSCRIPT']
        if path not in transcript_cache:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    transcript_cache[path] = f.read().replace('\n', '<br/>')
            except FileNotFoundError:
                transcript_cache[
                    path] = "<i>Error: Transcript file not found.</i>"

    # --- 4. Prepare and Group Data ---
    data_by_pottery = defaultdict(list)
    for dp in data_paths:
        dp['transcript_text'] = transcript_cache.get(
            dp['TRANSCRIPT'], "<i>Error loading transcript.</i>")
        data_by_pottery[dp['ID']].append(dp)

    # --- 5. Process Data in Parallel by Pottery Group ---
    pottery_stories = []

    # Prepare the list of "chunky" tasks
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
                initargs=(styles_for_workers, )) as executor:

            # The map function preserves the order of the input tasks
            results_iterator = executor.map(process_pottery_group,
                                            pottery_tasks)
            pottery_stories = list(
                tqdm(results_iterator,
                     total=len(pottery_tasks),
                     desc="Processing Pottery Groups in Parallel"))

    # --- 6. Assemble the final story from the collected chunks ---
    for story_chunk in pottery_stories:
        story.extend(story_chunk)

    print("All content generated. Building final PDF document...")
    doc.build(story)
    print("PDF report generation complete.")


if __name__ == "__main__":
    root = "./src/jomon_kaen_dataset/japan"
    # root = r"D:\storage\jomon_kaen\jomon_kaen_dataset\japan"

    # --- MODEL SELECTION ---
    # 1. Lightweight, recent model from Google.
    # SELECTED_MODEL_ID = 'google/embeddinggemma-300m'

    SELECTED_MODEL_ID = 'Qwen/Qwen3-Embedding-0.6B'

    # 3. Popular, well-balanced multilingual model.
    # SELECTED_MODEL_ID = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

    if not os.path.exists(root):
        print(f"Error: The directory '{root}' does not exist.")
    elif not os.path.exists(FONT_PATH):
        print(f"Error: The font file '{FONT_PATH}' was not found.")
    else:
        # Step 1: Load data paths and perform tokenization
        data_paths = load_data_paths(root)
        data_paths = tokenize_japanese(data_paths)

        # Step 2: Calculate percentages from QA files
        data_paths = calculate_qa_emotion_percentages(data_paths)

        # Step 3: Load model and generate transcript embeddings
        print(f"\nLoading Sentence Transformer model: {SELECTED_MODEL_ID}...")
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu"
        model = SentenceTransformer(SELECTED_MODEL_ID, device=device)
        embeddings = embed_tokens(data_paths,
                                  model,
                                  SELECTED_MODEL_ID,
                                  mode='fulltext')

        # Step 4: Calculate similarity to labels
        if 'e5' in SELECTED_MODEL_ID.lower():
            prefixed_labels = ["query: " + label for label in TARGET_LABELS_JP]
        else:
            prefixed_labels = TARGET_LABELS_JP
        label_embeddings = model.encode(prefixed_labels,
                                        convert_to_numpy=True,
                                        normalize_embeddings=True)
        similarities = cosine_similarity(embeddings, label_embeddings)

        # Step 5: Convert similarities to percentages and calculate alignment for the report
        print("\nCalculating embedding distributions and alignment scores...")
        embedding_percentages_all = softmax(similarities, axis=1) * 100
        for i, data_path in enumerate(data_paths):
            data_path['embedding_percentages'] = {
                label: embedding_percentages_all[i, j]
                for j, label in enumerate(TARGET_LABELS_JP)
            }
            qa_vector = np.array(list(data_path['qa_percentages'].values()))
            embed_vector = np.array(
                list(data_path['embedding_percentages'].values()))
            if np.linalg.norm(qa_vector) > 0 and np.linalg.norm(
                    embed_vector) > 0:
                score = np.dot(qa_vector, embed_vector) / (
                    np.linalg.norm(qa_vector) * np.linalg.norm(embed_vector))
            else:
                score = 0.0
            data_path['alignment_score'] = score

        # --- NEW Step 6: Generate the 3D Clustering Visualization ---
        print("\n--- Generating 3D Clustering Visualization ---")
        # Determine the dominant cluster for coloring the plot
        cluster_labels = np.argmax(similarities, axis=1)

        # Run UMAP to reduce dimensionality to 3D
        print("Running UMAP for 3D visualization...")
        umap_3d = umap.UMAP(n_components=3,
                            n_neighbors=15,
                            min_dist=0.0,
                            metric="cosine",
                            random_state=42,
                            n_jobs=1).fit_transform(embeddings)

        # Create and save the plot
        print("Generating 3D plot...")
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')

        for i, label_text in enumerate(TARGET_LABELS_JP):
            idx = (cluster_labels == i)
            if np.sum(idx) > 0:
                ax.scatter(umap_3d[idx, 0],
                           umap_3d[idx, 1],
                           umap_3d[idx, 2],
                           s=50,
                           alpha=0.7,
                           color=LABEL_COLORS(i),
                           label=label_text)

        ax.set_xlabel("UMAP-1", fontsize=12)
        ax.set_ylabel("UMAP-2", fontsize=12)
        ax.set_zlabel("UMAP-3", fontsize=12)
        ax.view_init(elev=30, azim=20)
        plt.legend(loc="best", fontsize=12)
        title_text = f"類似度クエリに基づく3Dクラスタリング\nModel: {SELECTED_MODEL_ID}"
        plt.title(title_text, fontsize=16)
        plt.tight_layout()

        model_name_for_file = SELECTED_MODEL_ID.replace('/', '_')
        plot_output_filename = f"cluster_plot_3d_{model_name_for_file}.png"
        plt.savefig(plot_output_filename)
        print(f"\n3D Plot successfully saved to '{plot_output_filename}'")
        plt.show(
        )  # Uncomment to display the plot interactively after saving, also uncomment 'agg' at imports

        # --- Step 7: Generate the final PDF report ---
        matplotlib.use('Agg')
        report_output_filename = f"Alignment_Report_{model_name_for_file}.pdf"

        # Set desired number of workers for the PDF generation
        num_workers = 8

        # generate_alignment_report(data_paths,
        #                           SELECTED_MODEL_ID,
        #                           report_output_filename,
        #                           max_workers=num_workers,
        #                           debug=False)
