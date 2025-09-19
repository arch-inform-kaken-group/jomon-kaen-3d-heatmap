import os
import sys
import io
import math
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import concurrent.futures
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import japanize_matplotlib
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from scipy.special import softmax
import neologdn
# SentenceTransformer is now only used for the 3D plot visualization
from sentence_transformers import SentenceTransformer
# NEW, IMPORTANT IMPORT: The Hugging Face Pipeline
from transformers import pipeline
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

# --- Environment/Path Setup ---
FONT_PATH = "C:/Windows/Fonts/msgothic.ttc"
try:
    pdfmetrics.registerFont(TTFont('JapaneseFont', FONT_PATH))
    JAPANESE_FONT = 'JapaneseFont'
    print(f"Successfully registered font '{FONT_PATH}' for PDF generation.")
except Exception as e:
    print(
        f"Warning: Could not register font '{FONT_PATH}'. PDF may not render Japanese characters correctly. Error: {e}"
    )
    JAPANESE_FONT = 'Helvetica'


# --- Global Constants (JAPANESE VERSION) ---
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
    global worker_styles
    worker_styles = styles_arg

# --- Data Loading and Preprocessing Functions ---
# (load_data_paths remains the same)
def load_data_paths(root=''):
    data_paths = []
    if not Path(root).exists():
        raise ValueError(f"Root directory not found: {root}")
    print(f"\nRAWデータパスを確認中")
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
    print(f"有効なデータ数: {len(data_paths)} [QAと空でない書き起こしが存在]")
    return data_paths


# --- Analysis and Report Generation Functions ---
# (calculate_qa_emotion_percentages, create_comparison_plot, process_pottery_group, generate_alignment_report remain the same)
def calculate_qa_emotion_percentages(data_paths):
    print("\nQAイベント数から感情の割合を計算中...")
    for data_path in tqdm(data_paths, desc="QAファイルを処理中"):
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
            print(f"警告: {data_path['QA']} を処理できませんでした。エラー: {e}")
            data_path['qa_percentages'] = {label: 0.0 for label in TARGET_LABELS_JP}
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
    ax1.set_title('QAイベント数 (%)')
    ax1.set_xlabel('割合')
    ax1.set_xlim(0, 100)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.invert_xaxis()
    ax1.yaxis.tick_right()
    ax2.barh(y, embed_values, height, color=[LABEL_COLORS(i / len(labels)) for i in range(len(labels))])
    ax2.set_title('書き起こし分類スコア (%)')
    ax2.set_xlabel('割合')
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
    pottery_story.append(Paragraph(f"土器IDの分析: {pottery_id}", worker_styles['HeaderStyle']))
    pottery_scores = [s['alignment_score'] for s in sessions if 'alignment_score' in s]
    avg_pottery_alignment = np.mean(pottery_scores) if pottery_scores else 0
    pottery_story.append(Paragraph(f"<b>この土器の平均整合性スコア:</b> {avg_pottery_alignment:.3f}", worker_styles['BodyStyle']))
    pottery_story.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=10))
    sessions.sort(key=lambda x: x['SESSION_ID'])
    for i, session in enumerate(sessions):
        pottery_story.append(
            Paragraph(
                f"<b>セッション:</b> {session['SESSION_ID']} | <b>整合性:</b> {session.get('alignment_score', 'N/A'):.3f}",
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
        pottery_story.append(Paragraph("<b>書き起こし:</b>", worker_styles['BodyStyle']))
        pottery_story.append(Paragraph(transcript_text, worker_styles['TranscriptStyle']))
        if i < len(sessions) - 1:
            pottery_story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey, spaceBefore=10, spaceAfter=10))
    pottery_story.append(PageBreak())
    return pottery_story

def generate_alignment_report(data_paths, model_id, output_filename, max_workers=10, debug=False):
    print(f"\nPDFレポートを生成中: {output_filename}")
    doc = SimpleDocTemplate(output_filename, pagesize=(8.5 * inch, 11 * inch))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleStyle', fontName=JAPANESE_FONT, fontSize=20, alignment=TA_CENTER, spaceAfter=20))
    styles.add(ParagraphStyle(name='HeaderStyle', fontName=JAPANESE_FONT, fontSize=16, spaceAfter=12, spaceBefore=20))
    styles.add(ParagraphStyle(name='BodyStyle', fontName=JAPANESE_FONT, fontSize=10, leading=14))
    styles.add(ParagraphStyle(name='TranscriptStyle', fontName=JAPANESE_FONT, fontSize=8, leading=12, leftIndent=10, rightIndent=10))
    styles_for_workers = {
        'HeaderStyle': styles['HeaderStyle'],
        'BodyStyle': styles['BodyStyle'],
        'TranscriptStyle': styles['TranscriptStyle']
    }
    story = []
    story.append(Paragraph("QAと書き起こし分類の整合性レポート", styles['TitleStyle']))
    story.append(Paragraph(f"分析生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['BodyStyle']))
    story.append(Paragraph(f"使用分類モデル: {model_id}", styles['BodyStyle']))
    story.append(Spacer(1, 0.25 * inch))
    all_scores = [dp['alignment_score'] for dp in data_paths if 'alignment_score' in dp]
    avg_alignment = np.mean(all_scores) if all_scores else 0
    story.append(Paragraph(f"<b>全体の平均整合性スコア:</b> {avg_alignment:.3f}", styles['HeaderStyle']))
    story.append(Paragraph("<i>(QAの割合ベクトルと分類スコアの割合ベクトルのコサイン類似度。1.0が完全一致)</i>", styles['BodyStyle']))
    story.append(PageBreak())
    print("書き起こしをメモリに読み込み中...")
    transcript_cache = {}
    for dp in tqdm(data_paths, desc="書き起こしを読み込み中"):
        path = dp['TRANSCRIPT']
        if path not in transcript_cache:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    transcript_cache[path] = f.read().replace('\n', '<br/>')
            except FileNotFoundError:
                transcript_cache[path] = "<i>エラー: 書き起こしファイルが見つかりません。</i>"
    data_by_pottery = defaultdict(list)
    for dp in data_paths:
        dp['transcript_text'] = transcript_cache.get(dp['TRANSCRIPT'], "<i>エラー: 書き起こしを読み込めません。</i>")
        data_by_pottery[dp['ID']].append(dp)
    pottery_tasks = sorted(data_by_pottery.items())
    if debug:
        print("\nデバッグモードで実行中（逐次処理）...")
        init_worker(styles_for_workers)
        pottery_stories = [process_pottery_group(task) for task in tqdm(pottery_tasks, desc="土器グループをデバッグ中")]
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=init_worker,
            initargs=(styles_for_workers,)
        ) as executor:
            results_iterator = executor.map(process_pottery_group, pottery_tasks)
            pottery_stories = list(tqdm(results_iterator, total=len(pottery_tasks), desc="土器グループを並列処理中"))
    for story_chunk in pottery_stories:
        story.extend(story_chunk)
    print("すべてのコンテンツを生成しました。最終的なPDFドキュメントを構築中...")
    doc.build(story)
    print("PDFレポートの生成が完了しました。")


if __name__ == "__main__":
    # root = r"D:\storage\jomon_kaen\jomon_kaen_dataset\japan"
    root = "./src/jomon_kaen_dataset/japan"
    
    SELECTED_MODEL_ID = 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'

    if not os.path.exists(root):
        print(f"エラー: ディレクトリ '{root}' が存在しません。")
    else:
        # Step 1 & 2: Load data paths and calculate QA percentages
        data_paths = load_data_paths(root)
        data_paths = calculate_qa_emotion_percentages(data_paths)

        # --- Step 3 & 4 using Zero-Shot-Classification Pipeline ---
        print(f"\nZero-Shotパイプラインとモデルを読み込み中: {SELECTED_MODEL_ID}...")
        device_num = 0 if torch.cuda.is_available() else -1
        classifier = pipeline("zero-shot-classification", model=SELECTED_MODEL_ID, device=device_num)

        print("分類用の書き起こしを準備中...")
        transcripts = []
        for data_path in data_paths:
            with open(data_path['TRANSCRIPT'], 'r', encoding='utf-8') as f:
                transcripts.append(neologdn.normalize(f.read()))
        
        print(f"{len(transcripts)}件の書き起こしを分類中 (この処理には時間がかかる場合があります)...")
        # The pipeline returns a list of dictionaries. We process them in a loop.
        results_generator = (classifier(transcript, TARGET_LABELS_JP, multi_label=False) for transcript in transcripts)
        all_results = list(tqdm(results_generator, total=len(transcripts)))

        # Convert the list of dictionary results into a single score matrix
        scores_matrix = np.zeros((len(transcripts), len(TARGET_LABELS_JP)))
        for i, result in enumerate(all_results):
            # Create a dictionary for easy lookup: {'label': score}
            score_dict = {label: score for label, score in zip(result['labels'], result['scores'])}
            # Fill the matrix in the correct order
            for j, label in enumerate(TARGET_LABELS_JP):
                scores_matrix[i, j] = score_dict.get(label, 0)

        # --- Step 5: Convert scores to percentages and calculate alignment ---
        print("\n分類分布と整合性スコアを計算中...")
        # The pipeline already outputs softmaxed scores (probabilities), so we just multiply by 100
        embedding_percentages_all = scores_matrix * 100
        
        for i, data_path in enumerate(data_paths):
            data_path['embedding_percentages'] = {
                label: embedding_percentages_all[i, j] for j, label in enumerate(TARGET_LABELS_JP)
            }
            qa_vector = np.array(list(data_path['qa_percentages'].values()))
            embed_vector = np.array(list(data_path['embedding_percentages'].values()))
            if np.linalg.norm(qa_vector) > 0 and np.linalg.norm(embed_vector) > 0:
                score = np.dot(qa_vector, embed_vector) / (np.linalg.norm(qa_vector) * np.linalg.norm(embed_vector))
            else:
                score = 0.0
            data_path['alignment_score'] = score
            
        # --- Step 6 and 7 (Plotting and PDF) ---
        print("\n--- 3D可視化用の埋め込みを生成中 ---")
        vis_model = SentenceTransformer('cl-tohoku/bert-base-japanese-whole-word-masking', device=f"cuda:{device_num}" if device_num !=-1 else "cpu")
        embeddings_for_visualization = vis_model.encode(transcripts, show_progress_bar=True)
        
        print("\n--- 3Dクラスタリングの可視化を生成中 ---")
        cluster_labels = np.argmax(scores_matrix, axis=1) 
        
        print("3D可視化のためにUMAPを実行中...")
        umap_3d = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.0,
                            metric="cosine", random_state=42, n_jobs=1).fit_transform(embeddings_for_visualization)
        
        print("3Dプロットを生成中...")
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        for i, label_text in enumerate(TARGET_LABELS_JP):
            idx = (cluster_labels == i)
            if np.sum(idx) > 0:
                ax.scatter(umap_3d[idx, 0], umap_3d[idx, 1], umap_3d[idx, 2],
                           s=50, alpha=0.7, color=LABEL_COLORS(i), label=label_text)
        ax.set_xlabel("UMAP-1", fontsize=12)
        ax.set_ylabel("UMAP-2", fontsize=12)
        ax.set_zlabel("UMAP-3", fontsize=12)
        plt.legend(loc="best", fontsize=12)
        title_text = f"Zero-Shot分類に基づく3Dクラスタリング\nモデル: {SELECTED_MODEL_ID}"
        plt.title(title_text, fontsize=16)
        plt.tight_layout()
        model_name_for_file = SELECTED_MODEL_ID.replace('/', '_')
        plot_output_filename = f"cluster_plot_3d_{model_name_for_file}_jp_zeroshot.png"
        plt.savefig(plot_output_filename)
        print(f"\n3Dプロットを'{plot_output_filename}'に保存しました。")

        # Step 7: Generate the final PDF report
        report_output_filename = f"Alignment_Report_{model_name_for_file}_jp_zeroshot.pdf"
        num_workers = 8
        generate_alignment_report(data_paths,
                                  SELECTED_MODEL_ID,
                                  report_output_filename,
                                  max_workers=num_workers,
                                  debug=False)