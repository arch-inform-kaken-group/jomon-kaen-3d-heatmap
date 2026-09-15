import os
import sys
import pandas as pd
import numpy as np

# Force non-interactive Matplotlib backend to prevent multithreading crashes
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path
import itertools
import warnings
from collections import Counter
import concurrent.futures

from scipy.stats import kruskal, mannwhitneyu
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import japanize_matplotlib  # Crucial for Japanese font rendering in matplotlib
try:
    import umap
    HAS_UMAP = True
except ImportError as e:
    print(f"Warning: UMAP library not found. 2D/3D UMAP clustering will be skipped. ({e})")
    HAS_UMAP = False

import torch
from transformers import pipeline
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
DATASET_ROOT_MALAYSIA = r"D:\storage\jomon_kaen\jomon_kaen_dataset\malaysia"
DATASET_ROOT_JAPAN = r"D:\storage\jomon_kaen\jomon_kaen_dataset\japan"

DATASET_ROOTS = [DATASET_ROOT_MALAYSIA, DATASET_ROOT_JAPAN]
OUTPUT_DIR = "multilingual_output"
TARGET_LANGUAGES = ['ENGLISH', 'MALAY', 'CHINESE', 'JAPANESE']

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "timelines"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "clustering"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "clustering", "2D"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "clustering", "3D"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "mismatch_alternatives"), exist_ok=True)

DOGU_PREFIXES = [
    'IN0295', 'IN0306', 'MH0037', 'NM0239', 
    'NZ0001', 'SK0035', 'TK0020', 'UD0028'
]

# ---------------------------------------------------------
# BACKEND MAPPINGS & COLORS
# ---------------------------------------------------------
UNIFIED_EMOTION_MAP = {
    "Interesting and attentional shape": "Interesting", 
    "Beautiful and artistic": "Beautiful",
    "Strange and incomprehensible": "Strange", 
    "Creepy / unsettling / scary": "Scary",
    "Feel nothing": "Feel nothing", 
    "面白い・気になる形だ": "Interesting", 
    "美しい・芸術的だ": "Beautiful",
    "不思議・意味不明": "Strange", 
    "不気味・不安・怖い": "Scary",
    "何も感じない": "Feel nothing", 
    "NO RESPONSE": "NO RESPONSE"
}

EMOTION_COLOR_MAP = {
    "Interesting": "#00FFFF",
    "Beautiful": "#00FF00",
    "Strange": "#FFFF00",
    "Scary": "#FF0000",
    "Feel nothing": "#505050",
    "NO RESPONSE": "#D3D3D3"
}

EMOTION_COLS_STANDARD = [
    "Interesting", "Beautiful", "Strange", "Scary", "Feel nothing"
]

LANG_COLORS = {
    'ENGLISH': '#4C72B0', 
    'MALAY': '#DD8452', 
    'CHINESE': '#55A868', 
    'JAPANESE': '#9467BD'
}

ARTIFACT_COLORS = {'Pottery': '#C44E52', 'Dogu': '#8172B2'}

# ---------------------------------------------------------
# BILINGUAL PRESENTATION MAPPINGS
# ---------------------------------------------------------
BILINGUAL_LANGUAGES = {
    'ENGLISH': 'English / 英語',
    'MALAY': 'Malay / マレー語',
    'CHINESE': 'Chinese / 中国語',
    'JAPANESE': 'Japanese / 日本語',
    'COMBINED': 'Combined / 全体'
}

BILINGUAL_EMOTIONS = {
    "Interesting": "Interesting / 面白い",
    "Beautiful": "Beautiful / 美しい",
    "Strange": "Strange / 不思議",
    "Scary": "Scary / 怖い",
    "Feel nothing": "Feel nothing / 何も感じない"
}
BILINGUAL_EMOTION_COLS = [BILINGUAL_EMOTIONS[e] for e in EMOTION_COLS_STANDARD]

BILINGUAL_ARTIFACTS = {
    'Pottery': 'Pottery / 土器',
    'Dogu': 'Dogu / 土偶'
}

LANG_COLORS_BI = {BILINGUAL_LANGUAGES[k]: v for k, v in LANG_COLORS.items()}
ARTIFACT_COLORS_BI = {BILINGUAL_ARTIFACTS[k]: v for k, v in ARTIFACT_COLORS.items()}

# ---------------------------------------------------------
# STOPWORDS
# ---------------------------------------------------------
MALAY_STOPWORDS = {
    'dan', 'yang', 'untuk', 'pada', 'ke', 'di', 'dari', 'ia', 'kami', 'mereka',
    'saya', 'anda', 'dengan', 'itu', 'ini', 'adalah', 'akan', 'telah', 'bila',
    'bukan', 'atau', 'tapi', 'kalau', 'kerana', 'sebab', 'jadi', 'kemudian',
    'lepas', 'terus', 'macam', 'pun', 'lah', 'kan', 'ok', 'okay', 'sementara',
    'walaupun', 'serta', 'hingga', 'lalu', 'maka', 'sambil', 'supaya', 'agar',
    'bahkan', 'namun', 'malah', 'and', 'but', 'or', 'yet', 'so', 'for', 'nor',
    'although', 'because', 'since', 'unless', 'until', 'while', 'whereas', 'like',
    'yeah', 'okay', 'just', 'actually', 'basically', 'mean', 'you know', 'stuff',
    'thing', 'things', 'if', 'then', 'than', 'also', 'cuz', 'cause', 'it', 'is',
    'the', 'of', 'and', 'this', 'that', 'can', 'has', 'what'
}

CHINESE_STOPWORDS = {
    '的', '了', '在', '是', '我', '有', '和', '这', '来', '上', '国', '个', '到', '说',
    '们', '你', '就', '去', '他', '她', '它', '也', '很', '啊', '因为', '所以', '然后',
    '但是', '就是', '不过', '可是', '而且', '或者', '还是', '虽然', '即使', '如果', '那么',
    '的话', '没有', '这个', '那个', '比较', '可能', '有点', '一下', '而且', '因此', '从而',
    '虽然', '尽管', '由于', '既然', '只要', '因此', '然而', '否则', '大概', '好像', 'and',
    'but', 'or', 'yet', 'so', 'for', 'nor', 'although', 'because', 'since', 'unless',
    'until', 'while', 'whereas', 'like', 'yeah', 'okay', 'just', 'actually', 'basically',
    'mean', 'you know', 'stuff', 'thing', 'things', 'if', 'then', 'than', 'also', 'cuz',
    'cause', 'it', 'is', 'the', 'of', 'and', "'s", 'this', 'to', 'that', 'has',
    'was', "‘s", "'s", 'in', 'on', 'can', 'boleh', 'dia', 'macam'
}

JAPANESE_STOPWORDS = {
    'の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と', 'し', 'れ', 'さ', 'ある', 'いる', 
    'も', 'する', 'から', 'な', 'こと', 'として', 'い', 'や', 'れる', 'など', 'なっ', 'ない', 
    'この', 'ため', 'その', 'あっ', 'よう', 'また', 'もの', 'という', 'あり', 'まで', 'られ', 
    'なる', 'へ', 'か', 'だ', 'これ', 'によって', 'により', 'おり', 'より', 'による', 'ず', 
    'なり', 'られる', 'において', 'ば', 'なけれ', 'なく', 'しかし', 'について', 'せ', 'だっ', 
    'その後', 'できる', 'それ', 'う', 'ので', 'なお', 'のみ', 'でき', 'き', 'つ', 'における', 
    'および', 'いう', 'さらに', 'でも', 'ら', 'たり', 'その他', 'に関する', 'たち', 'ます', 
    'ん', 'なら', 'に対して', '特に', 'せる', '及び', 'これら', 'とき', 'では', 'にて', 'ほか', 
    'ながら', 'うち', 'そして', 'とともに', 'ただし', 'かつて', 'それぞれ', 'または', 'お', 
    'ほど', 'に対する', 'すべて', 'あの', 'あそこ', 'あれ', 'どこ', 'ここ', 'です', 'はい', 
    'ええ', 'そう', 'ですね', 'ちょっと', 'なんか'
}

ENGLISH_CONJUNCTIONS_AND_FILLERS = {
    'and', 'but', 'or', 'yet', 'so', 'for', 'nor', 'although', 'because', 'since',
    'unless', 'until', 'while', 'whereas', 'like', 'yeah', 'okay', 'just', 'actually',
    'basically', 'mean', 'you know', 'stuff', 'thing', 'things', 'if', 'then', 'than',
    'also', 'cuz', 'cause', 'it', 'is', 'the', 'of', 'and'
}


# ==========================================
# 1. DATA LOADING
# ==========================================
def load_multilingual_dataset(root_dirs):
    qa_records, transcript_records = [], []
    
    for root_dir in root_dirs:
        root_path = Path(root_dir)
        if not root_path.exists() or not root_path.is_dir():
            print(f"Warning: Directory not found: {root_dir}")
            continue

        print(f"Scanning for data in {root_dir}...")

        for group_path in root_path.iterdir():
            if not group_path.is_dir(): continue
            for session_path in group_path.iterdir():
                if not session_path.is_dir(): continue

                language = None
                lang_file = session_path / 'language.txt'
                
                if lang_file.exists():
                    text = lang_file.read_text(encoding='utf-8').strip().upper()
                    if 'ENGLISH' in text or 'EN ' in text: language = 'ENGLISH'
                    elif 'MALAY' in text or 'BM ' in text or 'MS ' in text: language = 'MALAY'
                    elif 'CHINESE' in text or 'MANDARIN' in text or 'ZH ' in text: language = 'CHINESE'
                    elif 'JAPANESE' in text or 'JP ' in text or 'JA ' in text: language = 'JAPANESE'

                if language is None and "japan" in str(root_path).lower():
                    language = 'JAPANESE'

                if language not in TARGET_LANGUAGES: continue

                for pottery_path in session_path.iterdir():
                    if not pottery_path.is_dir(): continue
                    pottery_id, session_id = pottery_path.name, session_path.name

                    qa_file = pottery_path / "qa_corrected.csv"
                    if qa_file.exists():
                        try:
                            df_temp = pd.read_csv(qa_file)
                            df_temp['timestamp'] = pd.to_numeric(df_temp['timestamp'], errors='coerce')
                            df_temp.dropna(subset=['timestamp'], inplace=True)
                            df_temp['pottery_id'] = pottery_id
                            df_temp['session_id'] = session_id
                            df_temp['Language'] = language
                            qa_records.append(df_temp)
                        except Exception as e:
                            print(f"Error loading QA file {qa_file}: {e}")

                    transcript_file = pottery_path / "final_transcript.txt"
                    if transcript_file.exists():
                        try:
                            content = transcript_file.read_text(encoding='utf-8').strip()
                            if content:
                                transcript_records.append({
                                    'pottery_id': pottery_id,
                                    'session_id': session_id,
                                    'Language': language,
                                    'text': content
                                })
                        except Exception as e:
                            print(f"Error loading transcript {transcript_file}: {e}")

    df_qa = pd.concat(qa_records, ignore_index=True) if qa_records else pd.DataFrame()
    df_transcripts = pd.DataFrame(transcript_records)
    print(f"Loaded {len(df_qa)} QA interactions and {len(df_transcripts)} transcripts.")
    return df_qa, df_transcripts


# ==========================================
# 2. DURATION CALCULATION
# ==========================================
def calculate_durations(df_qa):
    if df_qa.empty: return pd.DataFrame(), pd.DataFrame()
    print("Calculating session duration blocks...")
    df = df_qa.copy()

    df['answer'] = df['answer'].astype(str).str.strip()
    df['short_answer'] = df['answer'].map(UNIFIED_EMOTION_MAP)
    df.dropna(subset=['short_answer', 'pottery_id', 'session_id'], inplace=True)
    df.sort_values(by=['Language', 'pottery_id', 'session_id', 'timestamp'], inplace=True)

    df['time_diff'] = df.groupby(['pottery_id', 'session_id'])['timestamp'].diff()
    emotion_changed = df['short_answer'] != df.groupby(['pottery_id', 'session_id'])['short_answer'].shift()
    time_gap_exceeded = df['time_diff'] > 0.05
    df['block_id'] = (emotion_changed | time_gap_exceeded).cumsum()

    blocks = df.groupby(['Language', 'pottery_id', 'session_id', 'block_id']).agg(
        start_time=('timestamp', 'min'),
        end_time=('timestamp', 'max'),
        answer=('short_answer', 'first')
    ).reset_index()

    blocks['duration'] = blocks['end_time'] - blocks['start_time']
    session_totals = blocks.groupby(['Language', 'session_id', 'pottery_id', 'answer'])['duration'].sum().unstack(fill_value=0.0).reset_index()

    for col in EMOTION_COLS_STANDARD:
        if col not in session_totals.columns:
            session_totals[col] = 0.0

    session_totals['Artifact_Type'] = session_totals['pottery_id'].apply(
        lambda x: 'Dogu' if any(prefix in str(x) for prefix in DOGU_PREFIXES) else 'Pottery')

    return session_totals, blocks


def generate_detailed_stats_report(df_durations, output_dir="multilingual_output"):
    """
    Generates a highly comprehensive text-based statistical report for emotion durations,
    including advanced descriptives, effect sizes, and automated interpretations.
    """
    print("\n--- Generating Comprehensive Statistical Text Report ---")
    
    # Constants
    EMOTION_COLS = ["Interesting", "Beautiful", "Strange", "Scary", "Feel nothing"]
    TARGET_LANGUAGES = ['ENGLISH', 'MALAY', 'CHINESE', 'JAPANESE']
    
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "Detailed_Statistical_Report.txt")
    
    # Helper: Effect Size r for Mann-Whitney U
    def calc_mwu_effect_size(u_stat, n1, n2):
        if n1 == 0 or n2 == 0: return 0.0, "None"
        mean_u = (n1 * n2) / 2.0
        # Standard deviation of U (ignoring ties correction for simplification)
        std_u = np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12.0)
        z_score = (u_stat - mean_u) / std_u if std_u > 0 else 0
        r = abs(z_score) / np.sqrt(n1 + n2)
        
        if r < 0.1: magnitude = "Negligible"
        elif r < 0.3: magnitude = "Small"
        elif r < 0.5: magnitude = "Moderate"
        else: magnitude = "Large"
        return r, magnitude

    # Helper: Effect Size Epsilon-Squared for Kruskal-Wallis
    def calc_kw_effect_size(h_stat, n_total):
        if n_total <= 1: return 0.0, "None"
        epsilon_sq = h_stat / (n_total - 1)
        
        if epsilon_sq < 0.01: magnitude = "Negligible"
        elif epsilon_sq < 0.08: magnitude = "Small"
        elif epsilon_sq < 0.26: magnitude = "Moderate"
        else: magnitude = "Large"
        return epsilon_sq, magnitude

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("COMPREHENSIVE STATISTICAL ANALYSIS REPORT: EMOTION DURATION ACROSS MOTHER TONGUE")
    report_lines.append("=" * 80)
    report_lines.append("\nThis report details the statistical evaluation of emotion durations (measured in")
    report_lines.append("seconds) across four mother tongue cohorts: English, Malay, Chinese, and Japanese.")
    report_lines.append("It includes Non-Parametric ANOVA (Kruskal-Wallis), post-hoc Mann-Whitney U tests")
    report_lines.append("with Bonferroni corrections, and Effect Size calculations (Epsilon-squared & r).\n")
    report_lines.append("-" * 80)
    report_lines.append("1. DETAILED OVERVIEW OF EMOTIONS")
    report_lines.append("-" * 80 + "\n")

    sig_emotions = []
    non_sig_emotions = []

    for emotion in EMOTION_COLS:
        if emotion not in df_durations.columns:
            continue
            
        groups = {lang: df_durations[df_durations['Language'] == lang][emotion].dropna() for lang in TARGET_LANGUAGES}
        valid_groups = [groups[l] for l in TARGET_LANGUAGES if not groups[l].empty]
        k = len(valid_groups)
        n_total = sum(len(g) for g in valid_groups)
        
        report_lines.append(f"* {emotion.upper()}")
        
        if k < 2:
            report_lines.append("  - Insufficient data for comparison.\n")
            continue
            
        try:
            h_stat, p_kw = kruskal(*valid_groups)
        except ValueError:
            report_lines.append("  - Variance error during computation.\n")
            continue
            
        is_sig = p_kw < 0.05
        eps_sq, eps_mag = calc_kw_effect_size(h_stat, n_total)
        
        if is_sig:
            sig_emotions.append(emotion)
            summary_txt = f"A significant overall difference was found among the groups (p < 0.05)."
        else:
            non_sig_emotions.append(emotion)
            summary_txt = f"There is no statistically significant difference among the mother tongue groups."

        report_lines.append(f"  - Test Statistic (KW H): {h_stat:.4f}")
        report_lines.append(f"  - Degrees of Freedom (df): {k - 1}")
        report_lines.append(f"  - P-value: {p_kw:.6g}")
        report_lines.append(f"  - Effect Size (Epsilon-squared): {eps_sq:.4f} ({eps_mag})")
        report_lines.append(f"  - Significant (\u03B1=0.05): {is_sig}")
        report_lines.append(f"  - Summary: {summary_txt}\n")
        
        report_lines.append("  [DESCRIPTIVE STATISTICS]")
        for lang in TARGET_LANGUAGES:
            data = groups[lang]
            if not data.empty:
                n, mean, std = len(data), data.mean(), data.std()
                median = data.median()
                q1, q3 = data.quantile(0.25), data.quantile(0.75)
                min_v, max_v = data.min(), data.max()
                report_lines.append(f"    * {lang.capitalize():<8} (N={n:<4}): Mean={mean:.2f}s (±{std:.2f}s) | Median={median:.2f}s | IQR=[{q1:.2f}s - {q3:.2f}s] | Range=[{min_v:.1f}s - {max_v:.1f}s]")
            else:
                report_lines.append(f"    * {lang.capitalize():<8} (N=0   ): No Data")
                
        report_lines.append("\n  [POST-HOC PAIRWISE COMPARISONS (Mann-Whitney U w/ Bonferroni)]")
        
        if is_sig:
            comps = list(itertools.combinations(TARGET_LANGUAGES, 2))
            num_comps = len(comps)
            found_pairwise_sig = False
            
            for l1, l2 in comps:
                g1, g2 = groups[l1], groups[l2]
                if g1.empty or g2.empty: continue
                
                u_stat, p_mwu = mannwhitneyu(g1, g2, alternative='two-sided')
                p_adj = min(p_mwu * num_comps, 1.0)
                
                if p_adj < 0.05:
                    found_pairwise_sig = True
                    r_effect, r_mag = calc_mwu_effect_size(u_stat, len(g1), len(g2))
                    
                    # Directional logic
                    if g1.mean() > g2.mean():
                        direction = f"{l1.capitalize()} > {l2.capitalize()}"
                    else:
                        direction = f"{l2.capitalize()} > {l1.capitalize()}"
                        
                    report_lines.append(f"    * {l1.capitalize()} vs {l2.capitalize()}:")
                    report_lines.append(f"      - U-Stat: {u_stat:.1f} | Adjusted P-value: {p_adj:.6g} (Significant)")
                    report_lines.append(f"      - Effect Size (r): {r_effect:.4f} ({r_mag})")
                    report_lines.append(f"      - Direction: {direction}")
                else:
                    report_lines.append(f"    * {l1.capitalize()} vs {l2.capitalize()}: Not Significant (p_adj = {p_adj:.4f})")
                    
            if not found_pairwise_sig:
                report_lines.append("    * Note: Overall ANOVA was significant, but no individual pairwise comparisons survived Bonferroni correction.")
        else:
            report_lines.append("    * Note: Post-hoc tests skipped because overall Kruskal-Wallis test was not significant.")
            
        report_lines.append("\n" + "=" * 80 + "\n")

    # Final Executive Summary
    report_lines.append("-" * 80)
    report_lines.append("2. EXECUTIVE SUMMARY OF FINDINGS")
    report_lines.append("-" * 80 + "\n")
    
    if non_sig_emotions:
        report_lines.append("- NON-SIGNIFICANT EMOTIONS: " + ", ".join(non_sig_emotions))
        report_lines.append("  These emotions showed uniform duration distribution across all language cohorts,")
        report_lines.append("  implying that cultural/linguistic background did not influence how long users")
        report_lines.append("  expressed these specific feelings.\n")
        
    if sig_emotions:
        report_lines.append("- SIGNIFICANT EMOTIONS: " + ", ".join(sig_emotions))
        report_lines.append("  These emotions yielded statistically significant variances. Refer to the specific")
        report_lines.append("  Post-Hoc directional interpretations above to see which demographic was dominant.")
        report_lines.append("  Usually, a 'Moderate' or 'Large' effect size (r > 0.3) indicates a highly")
        report_lines.append("  noticeable behavioral divergence between those specific cultures.")

    # Write to File
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print(f"Detailed statistical report successfully saved to: {report_path}")

# ==========================================
# 3. STATISTICAL ANALYSIS & PLOTTING
# ==========================================
def run_statistical_analysis(df_durations):
    print("\n--- Running Multilingual Statistical Analysis ---")
    results = []
    
    # Calculate N for legends
    lang_n = df_durations['Language'].value_counts().to_dict()
    lang_n_bi = {BILINGUAL_LANGUAGES[k]: v for k, v in lang_n.items()}
    
    for emotion in EMOTION_COLS_STANDARD:
        groups = {lang: df_durations[df_durations['Language'] == lang][emotion] for lang in TARGET_LANGUAGES}
        valid_groups = [groups[l] for l in TARGET_LANGUAGES if not groups[l].empty]

        descriptives = {}
        for l in TARGET_LANGUAGES:
            if not groups[l].empty:
                descriptives[f'{l}_N'] = len(groups[l])
                descriptives[f'{l}_Mean'] = groups[l].mean()
                descriptives[f'{l}_Median'] = groups[l].median()
                descriptives[f'{l}_Std'] = groups[l].std()
            else:
                descriptives[f'{l}_N'], descriptives[f'{l}_Mean'], descriptives[f'{l}_Median'], descriptives[f'{l}_Std'] = 0, np.nan, np.nan, np.nan

        if len(valid_groups) < 2: continue
        try:
            stat, p_kw = kruskal(*valid_groups)
        except ValueError:
            continue

        explanation, post_hoc_res = "", {}
        if p_kw < 0.05:
            sig_pairs = []
            comps = list(itertools.combinations(TARGET_LANGUAGES, 2))
            for l1, l2 in comps:
                if not groups[l1].empty and not groups[l2].empty:
                    u, p_mwu = mannwhitneyu(groups[l1], groups[l2], alternative='two-sided')
                    p_adj = min(p_mwu * len(comps), 1.0)
                    post_hoc_res[f'{l1}_vs_{l2}_p_adj'] = p_adj
                    if p_adj < 0.05: sig_pairs.append(f"{l1} and {l2}")
            explanation = f"Significant overall difference found (p={p_kw:.4f}). Post-hoc differences: {', '.join(sig_pairs)}." if sig_pairs else f"Significant overall difference found (p={p_kw:.4f}), but pairwise differences did not survive Bonferroni correction."
        else:
            explanation = f"No statistically significant difference among the Mother Tongue groups (p={p_kw:.4f})."

        res = {
            'Emotion': emotion, 'KW_Stat': stat, 'p_value': p_kw,
            'Significant': p_kw < 0.05, 'Explanation': explanation,
            **descriptives, **post_hoc_res
        }
        results.append(res)

    # Save Stats Reports
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(OUTPUT_DIR, "Language_Differences_Statistics.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "Statistical_Summary_Report.txt"), "w", encoding="utf-8") as f:
        f.write("=== MULTILINGUAL STATISTICAL ANALYSIS REPORT ===\n")
        f.write("Test: Kruskal-Wallis H-test followed by Mann-Whitney U tests with Bonferroni correction.\n\n")
        for r in results:
            f.write(f"--- EMOTION: {r['Emotion']} ---\n")
            f.write(f"Kruskal-Wallis Statistic: {r['KW_Stat']:.4f} | p-value: {r['p_value']:.4f}\n")
            f.write(f"Conclusion: {r['Explanation']}\n")
            f.write("Descriptive Statistics (Duration in seconds):\n")
            for lang in TARGET_LANGUAGES:
                if r.get(f'{lang}_N', 0) > 0:
                    f.write(f"  {lang.capitalize():<8}: N={r[f'{lang}_N']:<3} | Mean={r[f'{lang}_Mean']:.2f} | Median={r[f'{lang}_Median']:.2f} | Std={r[f'{lang}_Std']:.2f}\n")
            f.write("\n")

    pval_dict_bi = {BILINGUAL_EMOTIONS[r['Emotion']]: r['p_value'] for r in results}

    df_melt = pd.melt(df_durations, id_vars=['Language', 'session_id'], value_vars=EMOTION_COLS_STANDARD, var_name='Emotion', value_name='Duration')
    
    df_melt_bi = df_melt.copy()
    df_melt_bi['Language'] = df_melt_bi['Language'].map(BILINGUAL_LANGUAGES)
    df_melt_bi['Emotion'] = df_melt_bi['Emotion'].map(BILINGUAL_EMOTIONS)
    
    # ---------------------------------------------------------
    # RESTORED: Diagnostic Scatter Plot
    # ---------------------------------------------------------
    print("\n--- Generating Statistical Diagnostic Scatter Plot ---")
    plt.figure(figsize=(15, 7))
    sns.violinplot(x='Emotion', y='Duration', data=df_melt_bi, inner=None, color=".9", alpha=0.3)
    sns.stripplot(x='Emotion', y='Duration', hue='Language', data=df_melt_bi, palette=LANG_COLORS_BI, dodge=True, alpha=0.6, jitter=0.25, size=4)
    plt.title("Diagnostic Scatter Plot: Emotion Duration Distributions / 診断散布図: 感情持続時間分布\n(Exhibits heavy right-skew and zero-inflation)")
    plt.ylabel("Duration / 持続時間 (seconds / 秒)")
    plt.xlabel("Emotion / 感情")
    
    # Update legend with N
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    new_labels = [f"{lbl} (N={lang_n_bi.get(lbl, 0)})" for lbl in by_label.keys()]
    ax.legend(by_label.values(), new_labels, title='Language / 言語', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(OUTPUT_DIR, "Diagnostic_Data_Distribution_Scatter.png"), dpi=600, bbox_inches='tight')
    plt.close()

    def get_stats_text_block():
        lines = []
        for emotion in EMOTION_COLS_STANDARD:
            lines.append(f"{BILINGUAL_EMOTIONS[emotion]}:")
            for lang in TARGET_LANGUAGES:
                sub = df_melt[(df_melt['Emotion'] == emotion) & (df_melt['Language'] == lang)]['Duration']
                lang_label = BILINGUAL_LANGUAGES[lang].split(' / ')[0]
                if not sub.empty:
                    lines.append(f"  - {lang_label:<8}: {sub.mean():.2f}±{sub.std():.2f}")
                else:
                    lines.append(f"  - {lang_label:<8}: N/A")
        return "\n".join(lines)

    stats_text = get_stats_text_block()

    def create_emotion_variant(include_points, include_labels_on_plot, use_subtitle, filename):
        fig, ax = plt.subplots(figsize=(15, 7))
        sns.boxplot(x='Emotion', y='Duration', hue='Language', data=df_melt_bi, palette=LANG_COLORS_BI, width=0.85, ax=ax)

        if include_points:
            sns.stripplot(x='Emotion', y='Duration', hue='Language', data=df_melt_bi, dodge=True, color='black', alpha=0.3, jitter=0.2, size=2, legend=False, ax=ax)

        xticklabels = []
        for tick in ax.get_xticklabels():
            emo_bi = tick.get_text()
            pval = pval_dict_bi.get(emo_bi, None)
            if pval is None or np.isnan(pval):
                xticklabels.append(f"{emo_bi}\n(N/A)")
            elif pval < 0.0001:
                xticklabels.append(f"{emo_bi}\n(p < 0.0001)")
            else:
                xticklabels.append(f"{emo_bi}\n(p={pval:.4f})")
        ax.set_xticklabels(xticklabels, rotation=25, ha='right')

        if include_labels_on_plot:
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            ax.set_ylim(y_min, y_max + (y_range * 0.2))

            for i, emotion in enumerate(BILINGUAL_EMOTION_COLS):
                n_langs = len(TARGET_LANGUAGES)
                offsets = np.linspace(-0.3, 0.3, n_langs)
                lang_to_offset = {BILINGUAL_LANGUAGES[lang]: offsets[j] for j, lang in enumerate(TARGET_LANGUAGES)}
                base_y_offset = 0.02 * y_range

                for j, lang in enumerate(TARGET_LANGUAGES):
                    lang_bi = BILINGUAL_LANGUAGES[lang]
                    sub = df_melt_bi[(df_melt_bi['Emotion'] == emotion) & (df_melt_bi['Language'] == lang_bi)]['Duration']
                    if sub.empty: continue
                    max_val = sub.max()
                    label_text = f'{sub.mean():.2f}±{sub.std():.2f}'

                    x_label = i + lang_to_offset[lang_bi] - 0.1
                    y_label = max_val + base_y_offset + j * (base_y_offset * 1.65)
                    ax.text(x_label, y_label, label_text, ha='left', va='bottom', fontsize=9, rotation=0, 
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', pad=1))

        if use_subtitle:
            plt.figtext(0.80, 0.5, stats_text, wrap=False, horizontalalignment='left', verticalalignment='center', 
                        fontsize=9, bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray', pad=5))

        ax.set_title("Emotion Durations Across Mother Tongue / 母語別の感情持続時間 (seconds / 秒)")
        ax.set_ylabel("Duration / 持続時間 (seconds / 秒)")
        ax.set_xlabel("Emotion / 感情 (with Kruskal-Wallis p-value)")
        
        # Inject N into Legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        new_labels = [f"{lbl} (N={lang_n_bi.get(lbl, 0)})" for lbl in by_label.keys()]
        ax.legend(by_label.values(), new_labels, title='Language / 言語', bbox_to_anchor=(1.02, 1), loc='upper left')
        
        plt.tight_layout(rect=[0, 0, 0.77 if use_subtitle else 0.85, 1])
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=600, bbox_inches='tight')
        plt.close()

    create_emotion_variant(False, False, False, "Emotion_Variant1_Boxplot_Only.png")
    create_emotion_variant(True, False, False, "Emotion_Variant2_Boxplot_WithPoints_NoLabels.png")
    create_emotion_variant(True, True, False, "Emotion_Variant3_Boxplot_PointsAndLabelsOnPlot.png")
    create_emotion_variant(False, True, False, "Emotion_Variant4_Boxplot_LabelsOnly_NoPoints.png")
    create_emotion_variant(False, False, True, "Emotion_Variant5_Boxplot_SubtitleText_NoPoints.png")
    create_emotion_variant(True, False, True, "Emotion_Variant6_Boxplot_SubtitleText_WithPoints.png")


# ==========================================
# 4. ITEMIZED AVERAGES & ARTIFACT SIGNIFICANCE
# ==========================================
def export_itemized_averages_and_significance(df_durations, df_transcripts):
    print("\n--- Generating Pottery/Dogu Itemized Averages and Significance Reports ---")
    if not df_transcripts.empty:
        if 'Artifact_Type' not in df_transcripts.columns:
            df_transcripts['Artifact_Type'] = df_transcripts['pottery_id'].apply(lambda x: 'Dogu' if any(prefix in str(x) for prefix in DOGU_PREFIXES) else 'Pottery')
        if 'Polarity' not in df_transcripts.columns:
            df_transcripts['Polarity'] = df_transcripts['text'].apply(lambda t: TextBlob(t).sentiment.polarity)

    # Averages Export
    item_avg_durations = df_durations.groupby(['Language', 'Artifact_Type', 'pottery_id'])[EMOTION_COLS_STANDARD].mean().reset_index()
    if not df_transcripts.empty:
        item_avg_polarity = df_transcripts.groupby(['Language', 'pottery_id'])['Polarity'].mean().reset_index()
        itemized_master = pd.merge(item_avg_durations, item_avg_polarity, on=['Language', 'pottery_id'], how='left')
    else:
        itemized_master = item_avg_durations
    itemized_master.to_csv(os.path.join(OUTPUT_DIR, "Pottery_Dogu_Itemized_Averages.csv"), index=False)

    # ---------------------------------------------------------
    # RESTORED: Mann-Whitney U test & Report for Artifact Significance
    # ---------------------------------------------------------
    sig_results = []
    metrics_to_test = EMOTION_COLS_STANDARD + (['Polarity'] if not df_transcripts.empty else [])
    
    for metric in metrics_to_test:
        data = df_transcripts.dropna(subset=['Polarity']) if metric == 'Polarity' else df_durations.dropna(subset=[metric])
        group_pottery = data[data['Artifact_Type'] == 'Pottery'][metric]
        group_dogu = data[data['Artifact_Type'] == 'Dogu'][metric]
        
        if len(group_pottery) > 0 and len(group_dogu) > 0:
            try:
                stat, p_val = mannwhitneyu(group_pottery, group_dogu, alternative='two-sided')
                explanation = f"Statistically significant difference found (p={p_val:.4f})." if p_val < 0.05 else f"No significant difference (p={p_val:.4f})."
                sig_results.append({
                    'Metric': metric, 'Pottery_N': len(group_pottery), 'Dogu_N': len(group_dogu),
                    'Pottery_Mean': group_pottery.mean(), 'Dogu_Mean': group_dogu.mean(),
                    'U_Stat': stat, 'p_value': p_val, 'Significant': p_val < 0.05,
                    'Explanation': explanation
                })
            except Exception as e:
                print(f"Error during Mann-Whitney test for {metric}: {e}")

    df_sig = pd.DataFrame(sig_results)
    df_sig.to_csv(os.path.join(OUTPUT_DIR, "Artifact_Significance_Stats.csv"), index=False)
    
    with open(os.path.join(OUTPUT_DIR, "Artifact_Significance_Report.txt"), "w", encoding="utf-8") as f:
        f.write("=== ARTIFACT SIGNIFICANCE REPORT (POTTERY VS DOGU) ===\n")
        f.write("Test: Mann-Whitney U test (Two-sided)\n\n")
        for r in sig_results:
            f.write(f"--- METRIC: {r['Metric']} ---\n")
            f.write(f"Mann-Whitney U Statistic: {r['U_Stat']:.4f} | p-value: {r['p_value']:.4f}\n")
            f.write(f"Conclusion: {r['Explanation']}\n")
            f.write(f"  Pottery Group : N={r['Pottery_N']:<3} | Mean={r['Pottery_Mean']:.4f}\n")
            f.write(f"  Dogu Group    : N={r['Dogu_N']:<3} | Mean={r['Dogu_Mean']:.4f}\n\n")

    # Get Artifact N counts for legends
    artifact_n = df_durations['Artifact_Type'].value_counts().to_dict()
    artifact_n_bi = {BILINGUAL_ARTIFACTS[k]: v for k, v in artifact_n.items()}

    df_melt_sig = pd.melt(df_durations, id_vars=['Artifact_Type'], value_vars=EMOTION_COLS_STANDARD, var_name='Emotion', value_name='Duration')
    df_melt_sig_bi = df_melt_sig.copy()
    df_melt_sig_bi['Artifact_Type'] = df_melt_sig_bi['Artifact_Type'].map(BILINGUAL_ARTIFACTS)
    df_melt_sig_bi['Emotion'] = df_melt_sig_bi['Emotion'].map(BILINGUAL_EMOTIONS)
    
    plt.figure(figsize=(15, 7))
    ax = sns.boxplot(x='Emotion', y='Duration', hue='Artifact_Type', data=df_melt_sig_bi, palette=ARTIFACT_COLORS_BI, width=0.85)
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + ((y_max - y_min) * 0.15))

    plt.title("Statistical Distribution: Pottery vs. Dogu / 統計分布: 土器 vs 土偶 (Emotion Durations / 感情持続時間)")
    plt.ylabel("Duration / 持続時間 (Seconds / 秒)")
    plt.xlabel("Emotion / 感情")
    plt.xticks(rotation=25, ha='right')
    
    # Inject N into Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    new_labels = [f"{lbl} (N={artifact_n_bi.get(lbl, 0)})" for lbl in by_label.keys()]
    ax.legend(by_label.values(), new_labels, title='Artifact Type / 遺物タイプ', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(OUTPUT_DIR, "Artifact_Significance_Boxplot.png"), dpi=600, bbox_inches='tight')
    plt.close()


def generate_alternative_artifact_plots(df_durations):
    print("\n--- Generating Alternative Artifact (Pottery vs Dogu) Plots ---")
    
    # Calculate N for legends
    artifact_n = df_durations['Artifact_Type'].value_counts().to_dict()
    artifact_n_bi = {BILINGUAL_ARTIFACTS[k]: v for k, v in artifact_n.items()}

    df_melt_sig = pd.melt(df_durations, id_vars=['Artifact_Type'], value_vars=EMOTION_COLS_STANDARD, var_name='Emotion', value_name='Duration')
    df_melt_sig_bi = df_melt_sig.copy()
    df_melt_sig_bi['Artifact_Type'] = df_melt_sig_bi['Artifact_Type'].map(BILINGUAL_ARTIFACTS)
    df_melt_sig_bi['Emotion'] = df_melt_sig_bi['Emotion'].map(BILINGUAL_EMOTIONS)

    def add_subtitle(fig, text, width_ratio=0.82):
        fig.subplots_adjust(right=width_ratio)
        fig.text(width_ratio + 0.02, 0.5, text, wrap=True, ha='left', va='center', fontsize=10, 
                 bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', pad=5))
        return fig

    # 1. Split Violin Plot
    def plot_violin(variant, labeled, subtitled):
        fig, ax = plt.subplots(figsize=(15, 7))
        sns.violinplot(x='Emotion', y='Duration', hue='Artifact_Type', data=df_melt_sig_bi, palette=ARTIFACT_COLORS_BI, 
                       split=True, inner="quartile", cut=0, linewidth=1.2, ax=ax, hue_order=[BILINGUAL_ARTIFACTS['Pottery'], BILINGUAL_ARTIFACTS['Dogu']])
        if labeled:
            for i, emotion in enumerate(BILINGUAL_EMOTION_COLS):
                for atype, x_offset in [(BILINGUAL_ARTIFACTS['Pottery'], -0.3), (BILINGUAL_ARTIFACTS['Dogu'], 0.3)]:
                    data = df_melt_sig_bi[(df_melt_sig_bi['Emotion'] == emotion) & (df_melt_sig_bi['Artifact_Type'] == atype)]['Duration']
                    if len(data) == 0: continue
                    median = data.median()
                    ax.text(i + x_offset, median, f'{median:.2f}', ha='center', va='center', fontsize=10, 
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
        
        ax.set_title("Density Distribution: Pottery vs. Dogu / 密度分布: 土器 vs 土偶 (Split Violin Plot)")
        ax.set_ylabel("Duration / 持続時間 (s / 秒)")
        ax.set_xlabel("Emotion / 感情")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')

        # Add N to legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        new_labels = [f"{lbl} (N={artifact_n_bi.get(lbl, 0)})" for lbl in by_label.keys()]
        ax.legend(by_label.values(), new_labels, title='Artifact Type / 遺物タイプ', bbox_to_anchor=(1.02, 1), loc='upper left')

        if subtitled:
            guide = "Interpretation / 解釈:\nViolin width = density.\nWhite dot = median.\nLabels show median value."
            fig = add_subtitle(fig, guide)
            
        plt.tight_layout(rect=[0, 0, 0.82 if subtitled else 0.85, 1])
        plt.savefig(os.path.join(OUTPUT_DIR, f"Artifact_Alt_Violin_{variant}.png"), dpi=600, bbox_inches='tight')
        plt.close()

    plot_violin("Clean", labeled=False, subtitled=False)
    plot_violin("Labeled", labeled=True, subtitled=False)
    plot_violin("Subtitled", labeled=True, subtitled=True)

    # 2. Point Plot
    def plot_point(variant, labeled, subtitled):
        fig, ax = plt.subplots(figsize=(15, 7))
        sns.pointplot(x='Emotion', y='Duration', hue='Artifact_Type', data=df_melt_sig_bi, palette=ARTIFACT_COLORS_BI, 
                      dodge=0.2, markers=['o', 's'], capsize=.05, errorbar='ci', ax=ax, hue_order=[BILINGUAL_ARTIFACTS['Pottery'], BILINGUAL_ARTIFACTS['Dogu']])
        if labeled:
            y_min, y_max = ax.get_ylim()
            offset = 0.03 * (y_max - y_min)
            for i, emotion in enumerate(BILINGUAL_EMOTION_COLS):
                for j, atype in enumerate([BILINGUAL_ARTIFACTS['Pottery'], BILINGUAL_ARTIFACTS['Dogu']]):
                    data = df_melt_sig_bi[(df_melt_sig_bi['Emotion'] == emotion) & (df_melt_sig_bi['Artifact_Type'] == atype)]['Duration']
                    if len(data) == 0: continue
                    mean_val = data.mean()
                    x_pos = i + (j - 0.5) * 0.3
                    x_jitter = -0.052 if j == 0 else 0.052
                    ax.text(x_pos + x_jitter, mean_val + offset, f'{mean_val:.2f}', ha='center', va='bottom', fontsize=10, 
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                            
        ax.set_title("Mean Comparison: Pottery vs. Dogu / 平均比較: 土器 vs 土偶 (Point Plot with 95% CI)")
        ax.set_ylabel("Duration / 持続時間 (s / 秒)")
        ax.set_xlabel("Emotion / 感情")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        new_labels = [f"{lbl} (N={artifact_n_bi.get(lbl, 0)})" for lbl in by_label.keys()]
        ax.legend(by_label.values(), new_labels, title='Artifact Type / 遺物タイプ', bbox_to_anchor=(1.02, 1), loc='upper left')

        if subtitled:
            guide = "Points = mean duration.\nError bars = 95% CI.\nLabels show mean (s)."
            fig = add_subtitle(fig, guide)
            
        plt.tight_layout(rect=[0, 0, 0.82 if subtitled else 0.85, 1])
        plt.savefig(os.path.join(OUTPUT_DIR, f"Artifact_Alt_Point_{variant}.png"), dpi=600, bbox_inches='tight')
        plt.close()

    plot_point("Clean", labeled=False, subtitled=False)
    plot_point("Subtitled", labeled=True, subtitled=True)

    # 3. ECDF Plot
    def plot_ecdf(variant, labeled, subtitled):
        fig, axes = plt.subplots(1, len(EMOTION_COLS_STANDARD), figsize=(20, 5), sharey=True)
        for i, emotion in enumerate(BILINGUAL_EMOTION_COLS):
            sub_df = df_melt_sig_bi[df_melt_sig_bi['Emotion'] == emotion]
            sns.ecdfplot(data=sub_df, x='Duration', hue='Artifact_Type', palette=ARTIFACT_COLORS_BI, ax=axes[i], linewidth=2, hue_order=[BILINGUAL_ARTIFACTS['Pottery'], BILINGUAL_ARTIFACTS['Dogu']])
            axes[i].set_title(emotion, fontsize=10)
            axes[i].set_xlabel("Duration (s)")
            if i == 0: axes[i].set_ylabel("Cumulative Proportion / 累積比率")
            if axes[i].get_legend(): axes[i].get_legend().remove()

            if labeled:
                x_min, x_max = axes[i].get_xlim()
                x_range = x_max - x_min
                for atype in [BILINGUAL_ARTIFACTS['Pottery'], BILINGUAL_ARTIFACTS['Dogu']]:
                    data = sub_df[sub_df['Artifact_Type'] == atype]['Duration']
                    if len(data) == 0: continue
                    median = data.median()
                    x_text = median + 0.2 * x_range
                    axes[i].annotate(f'{median:.2f}', xy=(median, 0.5), xytext=(x_text, 0.5), 
                                     arrowprops=dict(arrowstyle='->', color=ARTIFACT_COLORS_BI[atype], lw=1), fontsize=9, color=ARTIFACT_COLORS_BI[atype], 
                                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        handles = [mpatches.Patch(color=ARTIFACT_COLORS_BI[k], label=f"{k} (N={artifact_n_bi.get(k, 0)})") for k in [BILINGUAL_ARTIFACTS['Pottery'], BILINGUAL_ARTIFACTS['Dogu']]]
        fig.legend(handles=handles, title='Artifact Type / 遺物タイプ', loc='center right', bbox_to_anchor=(0.98, 0.5))
        plt.suptitle("ECDF: Proportion of Data Below a Given Duration / ECDF: 累積比率 (Pottery vs Dogu)", y=1.05, fontsize=14)
        
        if subtitled:
            guide = "ECDF = Cumulative Distribution.\nSteeper curve = more concentrated.\nArrows point to median."
            fig.subplots_adjust(right=0.85)
            fig.text(0.87, 0.5, guide, wrap=True, ha='left', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', pad=5))
        else:
            plt.tight_layout(rect=[0, 0, 0.90, 1])
            
        plt.savefig(os.path.join(OUTPUT_DIR, f"Artifact_Alt_ECDF_{variant}.png"), dpi=600, bbox_inches='tight')
        plt.close()

    plot_ecdf("Clean", labeled=False, subtitled=False)
    plot_ecdf("Subtitled", labeled=True, subtitled=True)


# ==========================================
# 5. ADVANCED CLUSTERING (2D & 3D)
# ==========================================
def run_clustering(df_durations):
    print("\n--- Running Unsupervised Clustering (2D & 3D) ---")
    df_c = df_durations.dropna(subset=EMOTION_COLS_STANDARD).copy()
    if len(df_c) < 3: return

    total_n = len(df_c) # For the title N tag

    X = StandardScaler().fit_transform(df_c[EMOTION_COLS_STANDARD].values)
    
    reductions_2d = {'PCA': PCA(n_components=2).fit_transform(X)}
    reductions_3d = {'PCA': PCA(n_components=3).fit_transform(X)}
    
    if HAS_UMAP:
        reductions_2d['UMAP'] = umap.UMAP(n_components=2, random_state=42).fit_transform(X)
        reductions_3d['UMAP'] = umap.UMAP(n_components=3, random_state=42).fit_transform(X)

    clusters = {
        'KMeans': KMeans(n_clusters=3, random_state=42).fit_predict(X),
        'Agglomerative': AgglomerativeClustering(n_clusters=3).fit_predict(X),
        'GMM': GaussianMixture(n_components=3, random_state=42).fit_predict(X)
    }

    df_c['Language_Bi'] = df_c['Language'].map(BILINGUAL_LANGUAGES)

    def save_scatter_2d(coords, color_data, title, palette_name, filename, is_dict=False):
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=color_data, palette=palette_name if is_dict else 'viridis', 
                        s=100, alpha=0.8, edgecolor='k')
        plt.title(f"{title} (N={total_n})")
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "clustering", "2D", filename), dpi=600, bbox_inches='tight')
        plt.close()

    def save_scatter_3d(coords, color_data, title, palette_name, filename, is_dict=False):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        categories = color_data.unique()
        if is_dict:
            palette = palette_name
        else:
            colors_gen = sns.color_palette('viridis', len(categories))
            palette = {cat: colors_gen[i] for i, cat in enumerate(categories)}
            
        for cat in categories:
            idx = color_data == cat
            ax.scatter(coords[idx, 0], coords[idx, 1], coords[idx, 2], 
                       label=cat, color=palette[cat], s=50, alpha=0.8, edgecolor='k')
            
        ax.set_title(f"{title} (N={total_n})")
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "clustering", "3D", filename), dpi=600, bbox_inches='tight')
        plt.close()

    for red_name, coords in reductions_2d.items():
        save_scatter_2d(coords, df_c['Language_Bi'], f'2D {red_name} - True Language / 言語クラスタリング', LANG_COLORS_BI, f'{red_name}_2D_True_Language.png', is_dict=True)
        for clus_name, clus_labels in clusters.items():
            save_scatter_2d(coords, pd.Series(clus_labels).astype(str), f'2D {red_name} Projection - {clus_name} Clusters', 'viridis', f'{red_name}_2D_{clus_name}.png')

    for red_name, coords in reductions_3d.items():
        save_scatter_3d(coords, df_c['Language_Bi'], f'3D {red_name} - True Language / 言語クラスタリング', LANG_COLORS_BI, f'{red_name}_3D_True_Language.png', is_dict=True)
        for clus_name, clus_labels in clusters.items():
            save_scatter_3d(coords, pd.Series(clus_labels).astype(str), f'3D {red_name} Projection - {clus_name} Clusters', 'viridis', f'{red_name}_3D_{clus_name}.png')


# ==========================================
# 6. INDIVIDUAL TIMELINE CHARTS (MULTITHREADED)
# ==========================================
def _plot_single_timeline(args):
    lang, session, pottery, group = args
    
    fig, ax = plt.subplots(figsize=(15, 2))
    colors = [EMOTION_COLOR_MAP.get(ans, "#808080") for ans in group['answer']]
    
    ax.barh(y=[0] * len(group), width=group['duration'], left=group['start_time'], color=colors, height=0.85)
    ax.set_yticks([])
    ax.set_xlabel("Time (seconds) / 時間 (秒)")
    ax.set_xlim(left=0)
    
    lang_display = BILINGUAL_LANGUAGES.get(lang, lang).split(' / ')[0]
    ax.set_title(f"Emotion Timeline / 感情タイムライン | Lang: {lang_display} | Session: {session} | Artifact: {pottery}")

    legend_patches = [mpatches.Patch(color=EMOTION_COLOR_MAP[name], label=BILINGUAL_EMOTIONS.get(name, name)) for name in EMOTION_COLS_STANDARD]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    
    safe_session_name = str(session).replace('/', '_')
    filepath = os.path.join(OUTPUT_DIR, "timelines", f"Timeline_{lang}_{safe_session_name}_{pottery}.png")
    fig.savefig(filepath, dpi=600, bbox_inches='tight')
    plt.close(fig)

def generate_timeline_barcharts(blocks_df):
    print("\n--- Generating Participant Temporal Timeline Charts (Multithreaded) ---")
    tasks = []
    for (lang, session, pottery), group in blocks_df.groupby(['Language', 'session_id', 'pottery_id']):
        tasks.append((lang, session, pottery, group))
        
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(executor.map(_plot_single_timeline, tasks))
        
    print(f"Successfully generated {len(tasks)} timeline charts in parallel.")


# ==========================================
# 7. NLP ANALYSIS
# ==========================================
def analyze_nlp(df_transcripts):
    if df_transcripts.empty: return
    print("\n--- Running Multilingual NLP Analysis ---")

    df_transcripts['Artifact_Type'] = df_transcripts['pottery_id'].apply(lambda x: 'Dogu' if any(prefix in str(x) for prefix in DOGU_PREFIXES) else 'Pottery')

    def clip_error_bars(ax):
        for line in ax.lines:
            line.set_ydata(np.clip(line.get_ydata(), 0, None))

    try:
        device = 0 if torch.cuda.is_available() else -1
        classifier = pipeline("zero-shot-classification", model="cross-encoder/nli-deberta-v3-large", device=device)
        ce_records = []
        for _, row in df_transcripts.iterrows():
            if not row['text'].strip(): continue
            res = classifier(row['text'], EMOTION_COLS_STANDARD, multi_label=False)
            scores = {label: score for label, score in zip(res['labels'], res['scores'])}
            ce_records.append({'pottery_id': str(row['pottery_id']), 'session_id': str(row['session_id']), 'Language': row['Language'], 'Artifact_Type': row['Artifact_Type'], **scores})
        
        df_ce = pd.DataFrame(ce_records)
    except Exception as e:
        print(f"Warning: Cross-Encoder pipeline failed ({e}). Generating robust fallback baseline scores.")
        ce_records = []
        for _, row in df_transcripts.iterrows():
            scores = {emo: 0.2 for emo in EMOTION_COLS_STANDARD}
            ce_records.append({'pottery_id': str(row['pottery_id']), 'session_id': str(row['session_id']), 'Language': row['Language'], 'Artifact_Type': row['Artifact_Type'], **scores})
        df_ce = pd.DataFrame(ce_records)

    df_ce.to_csv(os.path.join(OUTPUT_DIR, "CrossEncoder_Emotion_Scores.csv"), index=False)
    ce_melt = pd.melt(df_ce, id_vars=['Language', 'Artifact_Type'], value_vars=EMOTION_COLS_STANDARD, var_name='Emotion', value_name='Probability Score')

    ce_melt_bi = ce_melt.copy()
    ce_melt_bi['Language'] = ce_melt_bi['Language'].map(BILINGUAL_LANGUAGES)
    ce_melt_bi['Artifact_Type'] = ce_melt_bi['Artifact_Type'].map(BILINGUAL_ARTIFACTS)
    ce_melt_bi['Emotion'] = ce_melt_bi['Emotion'].map(BILINGUAL_EMOTIONS)

    # 1. CE Plot by Language
    plt.figure(figsize=(15, 7))
    ax = sns.barplot(x='Emotion', y='Probability Score', hue='Language', data=ce_melt_bi, palette=LANG_COLORS_BI, errorbar='sd', capsize=0.05, width=0.85)
    clip_error_bars(ax)
    plt.ylim(0, 1.05)
    plt.title("Cross-Encoder Classification by Language / 言語別 音声感情分類 (Probability Score / 確率)")
    plt.ylabel("Mean Probability Score / 平均確率 (0 to 1)")
    plt.xlabel("Emotion / 感情")
    plt.xticks(rotation=25, ha='right')
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Language / 言語', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(OUTPUT_DIR, "CrossEncoder_Language_Comparison_Clean.png"), dpi=600, bbox_inches='tight')

    for p in ax.patches:
        height = p.get_height()
        if np.isnan(height) or height == 0: continue
        x_pos = p.get_x() + p.get_width() / 2 + 0.05
        ax.text(x_pos, height + 0.025, f'{height:.2f}', rotation=90, ha='center', va='bottom', fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
    plt.savefig(os.path.join(OUTPUT_DIR, "CrossEncoder_Language_Comparison_Labeled.png"), dpi=600, bbox_inches='tight')
    plt.close()

    # 2. CE Plot by Artifact
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.barplot(x='Emotion', y='Probability Score', hue='Artifact_Type', data=ce_melt_bi, palette=ARTIFACT_COLORS_BI, errorbar='sd', capsize=0.05, ax=ax, width=0.85)
    clip_error_bars(ax)
    ax.set_ylim(0, 1.05)
    for p in ax.patches:
        height = p.get_height()
        if np.isnan(height) or height == 0: continue
        x_pos = p.get_x() + p.get_width() / 2 + 0.05
        ax.text(x_pos, height + 0.025, f'{height:.2f}', rotation=0, ha='center', va='bottom', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
    plt.title("Cross-Encoder Classification: Pottery vs Dogu / 土器 vs 土偶 音声感情分類")
    plt.ylabel("Mean Probability Score / 平均確率 (0 to 1)")
    plt.xlabel("Emotion / 感情")
    plt.xticks(rotation=25, ha='right')
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Artifact Type / 遺物タイプ', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(OUTPUT_DIR, "CrossEncoder_Artifact_Bar_Labeled.png"), dpi=600, bbox_inches='tight')
    plt.close()

    # 3. Word Frequency Charts
    eng_stops = set(nltk.corpus.stopwords.words('english')).union(ENGLISH_CONJUNCTIONS_AND_FILLERS)
    fig, axes = plt.subplots(1, len(TARGET_LANGUAGES), figsize=(6 * len(TARGET_LANGUAGES), 6))
    
    for i, lang in enumerate(TARGET_LANGUAGES):
        group_df = df_transcripts[df_transcripts['Language'] == lang]
        if group_df.empty: continue
        
        words = []
        if lang == 'JAPANESE':
            try:
                import neologdn
                from sudachipy import tokenizer as sudachi_tokenizer, dictionary as sudachi_dictionary
                
                tokenizer_obj = sudachi_dictionary.Dictionary().create()
                mode = sudachi_tokenizer.Tokenizer.SplitMode.A
                pos_to_keep = {"名詞", "動詞", "形容詞"}
                
                for t in group_df['text'].tolist():
                    if not t.strip(): continue
                    normalized = neologdn.normalize(t)
                    tokens = [
                        m.normalized_form()
                        for m in tokenizer_obj.tokenize(normalized, mode)
                        if m.part_of_speech()[0] in pos_to_keep
                    ]
                    words.extend(tokens)
                    
            except ImportError as e:
                print(f"Missing libraries for Japanese tokenization: {e}. Falling back to basic tokenization.")
                text = " ".join(group_df['text'].tolist())
                words = word_tokenize(text) if ' ' in text else list(text)
                
            valid_words = [w for w in words if w not in JAPANESE_STOPWORDS and len(w) >= 1 and w.strip()]
            valid_words = [w[:-1] if w.endswith('る') and len(w) > 1 else w for w in valid_words]
            
        else:
            text = " ".join(group_df['text'].tolist()).lower()
            if not text.strip(): continue
            words = word_tokenize(text)
            if lang == 'ENGLISH': valid_words = [w for w in words if w.isalpha() and w not in eng_stops and len(w) > 2]
            elif lang == 'MALAY': valid_words = [w for w in words if w.isalpha() and w not in MALAY_STOPWORDS and len(w) > 2]
            elif lang == 'CHINESE': valid_words = [w for w in words if w not in CHINESE_STOPWORDS and len(w) > 1]

        freq = Counter(valid_words).most_common(15)
        if freq:
            df_freq = pd.DataFrame(freq, columns=['Word', 'Count'])
            ax = sns.barplot(x='Count', y='Word', data=df_freq, palette='Blues_d', ax=axes[i], width=0.85)
            for container in ax.containers:
                ax.bar_label(container, fmt='%d', fontsize=10, padding=2)
            axes[i].set_title(f"Top 15 words: {BILINGUAL_LANGUAGES.get(lang, lang).split(' / ')[0]}", fontsize=12)
            axes[i].set_xlabel("Frequency / 頻度")
            axes[i].set_ylabel("")
            
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "NLP_Word_Frequency_Labelled.png"), dpi=600, bbox_inches='tight')
    plt.close()


# ==========================================
# 8. MODALITY MISMATCH (21 ALTERNATIVES FACTORY)
# ==========================================
def generate_modality_mismatch_report(df_durations):
    """
    Calculates the per-session discrepancy between behavioral emotion expression (button duration proportion) 
    and verbal emotion expression (Cross-Encoder NLP probability). 
    Outputs 21 alternative plots (7 base types x 3 formatting variations).
    Strictly mathematically normalizes the plotted group means to exactly 1.00 (two decimals).
    """
    print("\n--- Generating Modality Mismatch Report (21 Alternatives) ---")
    ce_file = os.path.join(OUTPUT_DIR, "CrossEncoder_Emotion_Scores.csv")

    if not os.path.exists(ce_file):
        print("Error: CrossEncoder scores not found. Ensure 'analyze_nlp' runs before this function.")
        return

    # 1. Load verbal probabilities
    df_ce = pd.read_csv(ce_file)

    # 2. Process behavioral proportions
    df_beh = df_durations.copy()
    df_beh['total_duration'] = df_beh[EMOTION_COLS_STANDARD].sum(axis=1)

    beh_prop_cols = []
    for e in EMOTION_COLS_STANDARD:
        col_name = f"{e}_beh_prop"
        # Avoid division by zero
        df_beh[col_name] = np.where(df_beh['total_duration'] > 0,
                                    df_beh[e] / df_beh['total_duration'],
                                    0.0)
        beh_prop_cols.append(col_name)

    # 3. Aggregate Verbal Data per Session
    df_verb = df_ce.groupby(['session_id', 'Language'])[EMOTION_COLS_STANDARD].mean().reset_index()
    for e in EMOTION_COLS_STANDARD:
        df_verb.rename(columns={e: f'{e}_verb_prob'}, inplace=True)

    # Coerce IDs to string to guarantee a clean merge
    df_beh['session_id'] = df_beh['session_id'].astype(str)
    df_verb['session_id'] = df_verb['session_id'].astype(str)

    # 4. Merge Data securely
    merged = pd.merge(df_beh[['Language', 'session_id'] + beh_prop_cols],
                      df_verb,
                      on=['session_id', 'Language'],
                      how='inner')

    # 5. Melt into Long Format and calculate the Difference
    records = []
    for index, row in merged.iterrows():
        for e in EMOTION_COLS_STANDARD:
            beh_val = row[f"{e}_beh_prop"]
            verb_val = row[f"{e}_verb_prob"]
            diff = beh_val - verb_val
            records.append({
                'Language': row['Language'],
                'session_id': row['session_id'],
                'Emotion': e,
                'Behavioral_Score': beh_val,
                'Verbal_Score': verb_val,
                'Difference': diff
            })

    df_diff = pd.DataFrame(records)

    # 6. Append the 'COMBINED' average group
    df_comb = df_diff.copy()
    df_comb['Language'] = 'COMBINED'
    df_final = pd.concat([df_diff, df_comb], ignore_index=True)
    df_final.to_csv(os.path.join(OUTPUT_DIR, "Modality_Mismatch_Scores.csv"), index=False)

    # Base palettes & mapping
    mismatch_palette = LANG_COLORS.copy()
    mismatch_palette['COMBINED'] = '#808080'  # Gray color for Combined
    hue_order = TARGET_LANGUAGES + ['COMBINED']
    
    # N tag injection map
    lang_n = df_durations['Language'].value_counts().to_dict()
    lang_n['COMBINED'] = sum(lang_n.values())

    # 7. MATHEMATICAL NORMALIZATION & EXACT LOOKUP CACHE
    agg_lookup = {}
    agg_records = []
    
    for lang in hue_order:
        b_means = [df_final[(df_final['Language']==lang) & (df_final['Emotion']==emo)]['Behavioral_Score'].mean() for emo in EMOTION_COLS_STANDARD]
        v_means = [df_final[(df_final['Language']==lang) & (df_final['Emotion']==emo)]['Verbal_Score'].mean() for emo in EMOTION_COLS_STANDARD]
        
        b_sum, v_sum = max(sum(b_means), 1e-9), max(sum(v_means), 1e-9)
        b_norm = [b / b_sum for b in b_means]
        v_norm = [v / v_sum for v in v_means]
        
        # Rigorous precision rounding correction
        b_round = [round(b, 2) for b in b_norm]
        b_sum_rounded = round(sum(b_round), 2)
        if b_sum_rounded != 1.00 and b_sum_rounded > 0:
            diff_val = round(1.00 - b_sum_rounded, 2)
            idx_max = b_round.index(max(b_round))
            b_round[idx_max] = round(b_round[idx_max] + diff_val, 2)
            
        v_round = [round(v, 2) for v in v_norm]
        v_sum_rounded = round(sum(v_round), 2)
        if v_sum_rounded != 1.00 and v_sum_rounded > 0:
            diff_val = round(1.00 - v_sum_rounded, 2)
            idx_max = v_round.index(max(v_round))
            v_round[idx_max] = round(v_round[idx_max] + diff_val, 2)
            
        for i, emo in enumerate(EMOTION_COLS_STANDARD):
            b_val = b_round[i]
            v_val = v_round[i]
            m_val = round(b_val - v_val, 2)
            
            # Magnitude calculation
            if b_val == 0 and v_val == 0:
                mag_val = 0.0
            elif b_val >= v_val:
                mag_val = round(b_val / max(v_val, 0.01), 2)
            else:
                mag_val = round(-(v_val / max(b_val, 0.01)), 2)

            # Normalized Relative Difference (B - V) / (B + V)
            denom = b_val + v_val
            rel_diff = round((b_val - v_val) / denom, 4) if denom > 0 else 0.0

            agg_lookup[(lang, emo)] = {
                'Behavioral': b_val,
                'Verbal': v_val,
                'Mismatch': m_val,
                'Magnitude': mag_val,
                'Relative_Difference': rel_diff
            }
            agg_records.append({
                'Language': lang,
                'Emotion': emo,
                'Behavioral_Score': b_val,
                'Verbal_Score': v_val,
                'Verbal_Inverted': -v_val,
                'Mismatch_Score': m_val,
                'Magnitude': mag_val,
                'Relative_Difference': rel_diff
            })
            
    df_agg = pd.DataFrame(agg_records)

    # Map labels to include N in legends dynamically below:
    # NOTE: "Language Cohort" was changed to "Mother Language"
    def inject_legend_n(ax):
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        new_labels = [f"{lbl} (N={lang_n.get(lbl, 0)})" for lbl in by_label.keys()]
        ax.legend(by_label.values(), new_labels, title='Mother Language', bbox_to_anchor=(1.02, 1), loc='upper left')

    # Text guides for subtitles
    guide_bidirectional = "Chart Guide:\n\nUPWARD (Behavior):\nMean proportion of time\nphysically pressed.\n\nDOWNWARD (Verbal):\nMean CE probability\nfrom transcript."
    guide_difference = "Interpretation Guide:\n\n> 0 (Above Line):\nBehavior Dominant.\n(Pressed more than spoke)\n\n< 0 (Below Line):\nVerbal Dominant.\n(Spoke more than pressed)"
    guide_grid = "Modality Guide:\n\nBlue Bars:\nBehavioral Proportion\n\nRed Bars:\nVerbal CE Probability"
    guide_magnitude = "Interpretation Guide:\n\n> 0 (Above Line):\nBehavior Dominant.\n(Behavior / Verbal)\n\n< 0 (Below Line):\nVerbal Dominant.\n(Verbal / Behavior)"
    guide_rel_diff = "Normalized Relative Difference:\n\nCalculated per emotion.\n\nFormula:\n(Behavior - Verbal)\n-------------------\n(Behavior + Verbal)\n\n+1.0 = 100% Behavior\n-1.0 = 100% Verbal\n 0.0 = Perfect Match"

    def get_diff_stats_text():
        lines = ["Difference (Beh-Verb):\n"]
        for emo in EMOTION_COLS_STANDARD:
            lines.append(f"{emo}:")
            for lang in hue_order:
                val = agg_lookup[(lang, emo)]['Mismatch']
                lines.append(f" - {lang:<8}: {val:+.2f}")
            lines.append("")
        return "\n".join(lines)

    diff_stats_text = get_diff_stats_text()

    # =========================================================
    # PLOT GENERATOR 1: BIDIRECTIONAL MEAN BAR CHART
    # =========================================================
    def plot_alt1(labeled, subtitled, name):
        plt.figure(figsize=(16, 8))
        ax = plt.gca()

        sns.barplot(x='Emotion', y='Behavioral_Score', hue='Language', data=df_agg, palette=mismatch_palette, hue_order=hue_order, edgecolor='black', linewidth=1, errorbar=None, ax=ax, width=0.85)
        sns.barplot(x='Emotion', y='Verbal_Inverted', hue='Language', data=df_agg, palette=mismatch_palette, hue_order=hue_order, edgecolor='black', linewidth=1, errorbar=None, ax=ax, width=0.85)

        ax.axhline(0, color='black', linewidth=1.5, zorder=3)
        ax.set_ylim(-1.05, 1.05)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{abs(x):.2f}"))
        
        plt.title("Bidirectional Modality Comparison: Exactly Normalized Mean Scores (Behavioral vs Verbal)", fontsize=14)
        plt.ylabel("Verbal Probability (Transcript)      <----   0   ---->      Behavioral Proportion (Button)")
        plt.xlabel("Emotion")
        plt.xticks(rotation=25, ha='right')
        
        inject_legend_n(ax)

        if labeled:
            num_bars = len(hue_order) * len(EMOTION_COLS_STANDARD)
            total_bars = num_bars * 2 
            for i, p in enumerate(ax.patches[:total_bars]):
                height = p.get_height()
                if np.isnan(height): continue
                
                is_beh = (i < num_bars)
                lang = hue_order[(i % num_bars) // len(EMOTION_COLS_STANDARD)]
                emo = EMOTION_COLS_STANDARD[(i % num_bars) % len(EMOTION_COLS_STANDARD)]
                exact_val = agg_lookup[(lang, emo)]['Behavioral'] if is_beh else agg_lookup[(lang, emo)]['Verbal']
                
                if emo == 'Interesting' and abs(exact_val) < 0.001: continue
                if abs(height) < 0.001 and emo != 'Beautiful': continue
                
                txt = "0.00" if abs(exact_val) < 0.001 else f"{abs(exact_val):.2f}"
                x_pos = p.get_x() + p.get_width() / 2 + 0.03
                y_pos = height + (0.02 if height >= 0 else -0.02)
                va = 'bottom' if height >= 0 else 'top'
                ax.text(x_pos, y_pos, txt, ha='left', va=va, fontsize=7, rotation=45, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5))

        if subtitled:
            plt.figtext(0.84, 0.5, guide_bidirectional, wrap=True, ha='left', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', pad=5))

        plt.tight_layout(rect=[0, 0, 0.82 if subtitled else 0.85, 1])
        plt.savefig(os.path.join(OUTPUT_DIR, "mismatch_alternatives", f"Mismatch_Alt1_Bidirectional_{name}.png"), dpi=600, bbox_inches='tight')
        plt.close()

    # =========================================================
    # PLOT GENERATOR 2: DIFFERENCE BAR CHART
    # =========================================================
    def plot_alt2(labeled, subtitled, name):
        plt.figure(figsize=(15, 7))
        ax = sns.barplot(x='Emotion', y='Mismatch_Score', hue='Language', data=df_agg, palette=mismatch_palette, hue_order=hue_order, edgecolor='black', errorbar=None, width=0.85) 

        plt.axhline(0, color='red', linestyle='--', linewidth=1.5, zorder=0)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.2f}"))
        
        plt.title("Modality Difference Score (Behavioral Proportion minus Verbal Probability)", fontsize=14)
        plt.ylabel("Mean Difference Score\n(- = Verbal Dominant | + = Behavior Dominant)")
        plt.ylim(-0.225, 0.225)
        plt.xlabel("Emotion")
        plt.xticks(rotation=25, ha='right')
        inject_legend_n(ax)

        if labeled:
            num_bars = len(hue_order) * len(EMOTION_COLS_STANDARD)
            for i, p in enumerate(ax.patches[:num_bars]):
                height = p.get_height()
                if np.isnan(height): continue
                
                lang = hue_order[i // len(EMOTION_COLS_STANDARD)]
                emo = EMOTION_COLS_STANDARD[i % len(EMOTION_COLS_STANDARD)]
                exact_val = agg_lookup[(lang, emo)]['Mismatch']
                
                if emo == 'Interesting' and abs(exact_val) < 0.001: continue
                if abs(height) < 0.001 and emo != 'Beautiful': continue
                
                txt = "0.00" if abs(exact_val) < 0.001 else f"{exact_val:+.2f}"
                va = 'bottom' if height >= 0 else 'top'
                ax.text(p.get_x() + p.get_width() / 2., height + (0.01 if height>=0 else -0.01), txt, ha='center', va=va, fontsize=9, rotation=0, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5))

        if subtitled:
            plt.figtext(0.84, 0.5, guide_difference, wrap=True, ha='left', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', pad=5))

        plt.tight_layout(rect=[0, 0, 0.82 if subtitled else 0.85, 1])
        plt.savefig(os.path.join(OUTPUT_DIR, "mismatch_alternatives", f"Mismatch_Alt2_DiffBar_{name}.png"), dpi=600, bbox_inches='tight')
        plt.close()

    # =========================================================
    # PLOT GENERATOR 3: DIFFERENCE BOXPLOT
    # =========================================================
    def plot_alt3(labeled, subtitled, name):
        plt.figure(figsize=(15, 7))
        df_final['Mismatch_Score'] = df_final['Behavioral_Score'] - df_final['Verbal_Score']
        ax = sns.boxplot(x='Emotion', y='Mismatch_Score', hue='Language', data=df_final, palette=mismatch_palette, hue_order=hue_order, showfliers=False, width=0.85)

        if labeled: 
            sns.stripplot(x='Emotion', y='Mismatch_Score', hue='Language', data=df_final, hue_order=hue_order, dodge=True, color='black', alpha=0.3, jitter=0.2, size=3, ax=ax)

        plt.axhline(0, color='red', linestyle='--', linewidth=1.5, zorder=0)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.2f}"))

        plt.title("Per-Session Variance of Modality Mismatch Scores", fontsize=14)
        plt.ylabel("Difference Score\n(- = Verbal Dominant | + = Behavior Dominant)")
        plt.xlabel("Emotion")
        plt.xticks(rotation=25, ha='right')
        
        inject_legend_n(ax)

        if subtitled:
            plt.figtext(0.82, 0.5, diff_stats_text, wrap=False, ha='left', va='center', fontsize=10, family='monospace', bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', pad=5))

        plt.tight_layout(rect=[0, 0, 0.80 if subtitled else 0.85, 1])
        plt.savefig(os.path.join(OUTPUT_DIR, "mismatch_alternatives", f"Mismatch_Alt3_DiffBox_{name}.png"), dpi=600, bbox_inches='tight')
        plt.close()


    # =========================================================
    # PLOT GENERATOR 4: SIDE-BY-SIDE GRID CATPLOT
    # =========================================================
    df_grid = df_agg.copy()
    grid_records = []
    for _, row in df_grid.iterrows():
        grid_records.append({
            'Language': row['Language'],
            'Emotion': row['Emotion'],
            'Modality': 'Behavioral (%)',
            'Normalized Score': row['Behavioral_Score']
        })
        grid_records.append({
            'Language': row['Language'],
            'Emotion': row['Emotion'],
            'Modality': 'Verbal (CE)',
            'Normalized Score': row['Verbal_Score']
        })
    df_grid_long = pd.DataFrame(grid_records)

    # =========================================================
    # PLOT GENERATOR 4: SIDE-BY-SIDE GRID CATPLOT (FIXED)
    # =========================================================
    def plot_alt4(labeled, subtitled, name):
        # legend=False to prevent default overlap, we will draw it manually
        g = sns.catplot(data=df_grid_long,
                        x='Emotion', y='Normalized Score',
                        hue='Modality', col='Language',
                        kind='bar', col_wrap=2,
                        palette=['#3498db', '#e74c3c'],
                        edgecolor='black', errorbar=None,
                        height=4.5, aspect=1.2,
                        legend=False) 
                        
        g.fig.suptitle("Modality Comparison: Exactly Normalized Scores (Behavioral vs Verbal by Mother Tongue)", y=1.05, fontsize=16)

        for lang_idx, ax in enumerate(g.axes.flat):
            ax.set_ylim(0, 1.05)
            ax.axhline(0, color='black', linewidth=1)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.2f}"))
            ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
            
            current_lang = hue_order[lang_idx] if lang_idx < len(hue_order) else "COMBINED"

            if labeled:
                num_bars_per_ax = 2 * len(EMOTION_COLS_STANDARD)
                for i, p in enumerate(ax.patches[:num_bars_per_ax]):
                    height = p.get_height()
                    if np.isnan(height) or height == 0: continue
                    
                    # Seaborn grouped bars structure: first half of patches are Behavioral, second half Verbal
                    is_beh = (i < len(EMOTION_COLS_STANDARD))
                    emo = EMOTION_COLS_STANDARD[i % len(EMOTION_COLS_STANDARD)]
                    
                    exact_val = agg_lookup[(current_lang, emo)]['Behavioral'] if is_beh else agg_lookup[(current_lang, emo)]['Verbal']
                    
                    if emo == 'Interesting' and abs(exact_val) < 0.001: continue
                    if abs(height) < 0.001 and emo != 'Beautiful': continue
                    
                    txt = "0.00" if abs(exact_val) < 0.001 else f"{exact_val:.2f}"
                    ax.text(p.get_x() + p.get_width() / 2., height + 0.02, txt, ha='center', va='bottom', fontsize=9, rotation=0, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

        # Re-add legend manually positioned near the top-right
        handles = [
            mpatches.Patch(color='#3498db', label='Behavioral (%)'), 
            mpatches.Patch(color='#e74c3c', label='Verbal (CE)')
        ]
        g.fig.legend(handles=handles, title='Modality', loc='center right', bbox_to_anchor=(0.98, 0.70))

        if subtitled:
            g.fig.subplots_adjust(right=0.82)
            # Position subtitle explicitly lower on the y-axis (y=0.35) so it stays under the legend
            g.fig.text(0.83, 0.35, guide_grid, va='center', ha='left', fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', pad=5))
        else:
            g.fig.subplots_adjust(right=0.85)

        plt.savefig(os.path.join(OUTPUT_DIR, "mismatch_alternatives", f"Mismatch_Alt4_SideBySideGrid_{name}.png"), dpi=600, bbox_inches='tight')
        plt.close()


    # =========================================================
    # PLOT GENERATOR 5: MAGNITUDE BAR CHART
    # =========================================================
    def plot_alt5(labeled, subtitled, name):
        plt.figure(figsize=(15, 7))
        ax = sns.barplot(x='Emotion', y='Magnitude', hue='Language', data=df_agg, palette=mismatch_palette, hue_order=hue_order, edgecolor='black', errorbar=None, width=0.85) 

        plt.axhline(0, color='black', linestyle='-', linewidth=1.5, zorder=0)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{abs(x):.2f}x"))
        
        plt.title("Modality Magnitude Multiplier (Behavioral vs Verbal Ratio)", fontsize=14)
        plt.ylabel("Magnitude Multiplier\n(- = Verbal Dominant | + = Behavior Dominant)")
        plt.xlabel("Emotion")
        plt.xticks(rotation=25, ha='right')
        inject_legend_n(ax)

        if labeled:
            num_bars = len(hue_order) * len(EMOTION_COLS_STANDARD)
            for i, p in enumerate(ax.patches[:num_bars]):
                height = p.get_height()
                if np.isnan(height): continue
                
                lang = hue_order[i // len(EMOTION_COLS_STANDARD)]
                emo = EMOTION_COLS_STANDARD[i % len(EMOTION_COLS_STANDARD)]
                exact_val = agg_lookup[(lang, emo)]['Mismatch']
                
                if emo == 'Interesting' and abs(exact_val) < 0.001: continue
                if abs(height) < 0.001 and emo != 'Beautiful': continue
                
                val_text = "0.00x" if abs(exact_val) < 0.001 else f"{abs(height):.2f}x"
                y_offset = 0.10 if height >= 0 else -0.10
                va = 'bottom' if height >= 0 else 'top'
                ax.text(p.get_x() + p.get_width() / 2., height + y_offset, val_text, ha="center", va=va, fontsize=9, rotation=0, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

        if subtitled:
            plt.figtext(0.84, 0.5, guide_magnitude, wrap=True, ha='left', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', pad=5))

        plt.tight_layout(rect=[0, 0, 0.82 if subtitled else 0.85, 1])
        plt.savefig(os.path.join(OUTPUT_DIR, "mismatch_alternatives", f"Mismatch_Alt5_MagBar_{name}.png"), dpi=600, bbox_inches='tight')
        plt.close()

    # =========================================================
    # PLOT GENERATOR 6: NORMALIZED RELATIVE DIFFERENCE (UPDATED)
    # =========================================================
    def plot_alt6(labeled, subtitled, name):
        plt.figure(figsize=(15, 7))
        ax = sns.barplot(x='Emotion', y='Relative_Difference', hue='Language', data=df_agg, palette=mismatch_palette, hue_order=hue_order, edgecolor='black', errorbar=None, width=0.85) 

        plt.axhline(0, color='black', linestyle='-', linewidth=1.5, zorder=0)
        ax.set_ylim(-1.05, 1.05)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.2f}"))
        
        # 1 & 2 & 3. Updated Title and Y-axis label (Equations Omitted)
        plt.title("Normalized Emotion Modality Difference Score", fontsize=24, fontweight='bold', pad=15)
        plt.ylabel("Normalized Emotion Modality Difference Score", fontsize=18)
        plt.xlabel("Emotion", fontsize=18)
        
        # 4. Enlarged tick fonts
        plt.xticks(rotation=25, ha='right', fontsize=16)
        plt.yticks(fontsize=16)
        
        # 5 & 6. Legend moved inside top-left, "Language Cohort" -> "Mother Language"
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        new_labels = [f"{lbl} (N={lang_n.get(lbl, 0)})" for lbl in by_label.keys()]
        ax.legend(by_label.values(), new_labels, title='Mother Language', loc='upper left', 
                  fontsize=14, title_fontsize=16, frameon=True, facecolor='white', edgecolor='black', shadow=True)

        if labeled:
            num_bars = len(hue_order) * len(EMOTION_COLS_STANDARD)
            for i, p in enumerate(ax.patches[:num_bars]):
                height = p.get_height()
                if np.isnan(height): continue
                
                lang = hue_order[i // len(EMOTION_COLS_STANDARD)]
                emo = EMOTION_COLS_STANDARD[i % len(EMOTION_COLS_STANDARD)]
                
                exact_val = agg_lookup[(lang, emo)]['Relative_Difference']
                mismatch_val = agg_lookup[(lang, emo)]['Mismatch']
                
                if emo == 'Interesting' and abs(mismatch_val) < 0.001: continue
                if abs(height) < 0.001 and emo != 'Beautiful': continue
                
                val_text = f"{exact_val:+.2f}" if abs(exact_val) > 0.001 else "0.00"
                y_offset = 0.03 if height >= 0 else -0.03
                va = 'bottom' if height >= 0 else 'top'
                # 4. Enlarged data label fonts
                ax.text(p.get_x() + p.get_width() / 2., height + y_offset, val_text, ha="center", va=va, 
                        fontsize=14, fontweight='bold', rotation=0, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

        # 7. Formula moved to right bottom within the graph using Matplotlib MathText
        formula_text = r'$\frac{\text{Behavior} - \text{Verbal}}{\text{Behavior} + \text{Verbal}}$'
        ax.text(0.98, 0.03, formula_text, transform=ax.transAxes, fontsize=20, 
                verticalalignment='bottom', horizontalalignment='right', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

        if subtitled:
            plt.figtext(0.84, 0.5, guide_rel_diff, wrap=True, ha='left', va='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', pad=5))

        # Adjusted tight_layout rect because legend is now inside the plot area
        plt.tight_layout(rect=[0, 0, 0.82 if subtitled else 1.0, 1])
        plt.savefig(os.path.join(OUTPUT_DIR, "mismatch_alternatives", f"Mismatch_Alt6_RelDiffBar_{name}.png"), dpi=600, bbox_inches='tight')
        plt.close()

    # =========================================================
    # PLOT GENERATOR 7: NORMALIZED RELATIVE DIFFERENCE (ABSOLUTE)
    # =========================================================
    def plot_alt7(labeled, subtitled, name):
        plt.figure(figsize=(15, 7))
        
        df_agg['Abs_Relative_Difference'] = df_agg['Relative_Difference'].abs()
        
        ax = sns.barplot(x='Emotion', y='Abs_Relative_Difference', hue='Language', data=df_agg, palette=mismatch_palette, hue_order=hue_order, edgecolor='black', errorbar=None, width=0.85) 

        plt.axhline(0, color='black', linestyle='-', linewidth=1.5, zorder=0)
        ax.set_ylim(0, 1.05) 
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.2f}"))
        
        plt.title("Absolute Normalized Relative Difference per Emotion (|Behavior - Verbal| / (Behavior + Verbal))", fontsize=14)
        plt.ylabel("Absolute Relative Difference Score\n(0.0 = Perfect Match | 1.0 = Complete Mismatch)")
        
        new_labels = []
        for tick in ax.get_xticklabels():
            emo = tick.get_text()
            comb_diff = agg_lookup[('COMBINED', emo)]['Relative_Difference']
            
            if comb_diff > 0.001:
                sub = "(Expressed more\nin behaviour)"
            elif comb_diff < -0.001:
                sub = "(Expressed more\nin verbal)"
            else:
                sub = "(Equally expressed)"
                
            new_labels.append(f"{emo}\n{sub}")
            
        ax.set_xticklabels(new_labels, rotation=25, ha='right')
        plt.xlabel("Emotion")
        inject_legend_n(ax)

        if labeled:
            num_bars = len(hue_order) * len(EMOTION_COLS_STANDARD)
            for i, p in enumerate(ax.patches[:num_bars]):
                height = p.get_height()
                if np.isnan(height): continue
                
                lang = hue_order[i // len(EMOTION_COLS_STANDARD)]
                emo = EMOTION_COLS_STANDARD[i % len(EMOTION_COLS_STANDARD)]
                
                exact_val = abs(agg_lookup[(lang, emo)]['Relative_Difference'])
                mismatch_val = agg_lookup[(lang, emo)]['Mismatch']
                
                if emo == 'Interesting' and abs(mismatch_val) < 0.001: continue
                if abs(height) < 0.001 and emo != 'Beautiful': continue
                
                val_text = f"{exact_val:.2f}" if exact_val > 0.001 else "0.00"
                y_offset = 0.02 
                va = 'bottom' 
                ax.text(p.get_x() + p.get_width() / 2., height + y_offset, val_text, ha="center", va=va, fontsize=9, rotation=0, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

        if subtitled:
            guide_rel_diff_abs = "Absolute Relative Difference:\n\nCalculated per emotion.\n\nFormula:\n|Behavior - Verbal|\n-------------------\n(Behavior + Verbal)\n\n1.0 = Complete Mismatch\n0.0 = Perfect Match"
            plt.figtext(0.84, 0.5, guide_rel_diff_abs, wrap=True, ha='left', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', pad=5))

        plt.tight_layout(rect=[0, 0, 0.82 if subtitled else 0.85, 1])
        plt.savefig(os.path.join(OUTPUT_DIR, "mismatch_alternatives", f"Mismatch_Alt7_RelDiffBar_{name}.png"), dpi=600, bbox_inches='tight')
        plt.close()

    # Trigger Generation for All 21 Charts (7 Base Types x 3 Formatting Variations)
    plot_alt1(False, False, "1A_Clean")
    plot_alt1(True, False, "1B_Labeled")
    plot_alt1(True, True, "1C_Subtitled")
    
    plot_alt2(False, False, "2A_Clean")
    plot_alt2(True, False, "2B_Labeled")
    plot_alt2(True, True, "2C_Subtitled")

    plot_alt3(False, False, "3A_BoxOnly")
    plot_alt3(True, False, "3B_WithPoints")
    plot_alt3(True, True, "3C_SubtitledStats")

    plot_alt4(False, False, "4A_Clean")
    plot_alt4(True, False, "4B_Labeled")
    plot_alt4(True, True, "4C_Subtitled")

    plot_alt5(False, False, "5A_Clean")
    plot_alt5(True, False, "5B_Labeled")
    plot_alt5(True, True, "5C_Subtitled")

    plot_alt6(False, False, "6A_Clean")
    plot_alt6(True, False, "6B_Labeled")
    plot_alt6(True, True, "6C_Subtitled")

    plot_alt7(False, False, "7A_Clean")
    plot_alt7(True, False, "7B_Labeled")
    plot_alt7(True, True, "7C_Subtitled")

    print("Successfully generated all 21 Modality Mismatch alternative charts.")


# ==========================================
# 9. METADATA REPORT (RESTORED)
# ==========================================
def save_metadata_report(df_qa, df_durations, df_transcripts, blocks_df):
    print("\n--- Generating Metadata Report ---")
    metadata = {}
    if not df_durations.empty:
        metadata['Unique Sessions per Language'] = df_durations.groupby('Language')['session_id'].nunique().to_dict()
        metadata['Unique Artifacts per Language'] = df_durations.groupby('Language')['pottery_id'].nunique().to_dict()

    if not df_transcripts.empty:
        df_transcripts['Artifact_Type'] = df_transcripts['pottery_id'].apply(lambda x: 'Dogu' if any(prefix in str(x) for prefix in DOGU_PREFIXES) else 'Pottery')
        metadata['Artifact Type Counts'] = df_transcripts['Artifact_Type'].value_counts().to_dict()
        metadata['Artifact Type per Language'] = df_transcripts.groupby(['Language', 'Artifact_Type']).size().unstack(fill_value=0).to_dict()

    if not blocks_df.empty:
        metadata['Emotion Block Counts'] = blocks_df['answer'].value_counts().to_dict()
        metadata['Emotion Blocks per Language'] = blocks_df.groupby(['Language', 'answer']).size().unstack(fill_value=0).to_dict()

    metadata['Total Transcripts'] = len(df_transcripts)
    metadata['Total QA Interactions'] = len(df_qa)

    if not df_durations.empty:
        df_durations['total_duration'] = df_durations[EMOTION_COLS_STANDARD].sum(axis=1)
        metadata['Session Total Duration Stats'] = df_durations.groupby('Language')['total_duration'].agg(['mean', 'std', 'min', 'max']).round(2).to_dict()

    flat_records = []
    for k, v in metadata.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                flat_records.append({'Metric': f"{k}_{sk}", 'Value': sv})
        else:
            flat_records.append({'Metric': k, 'Value': v})

    pd.DataFrame(flat_records).to_csv(os.path.join(OUTPUT_DIR, "Dataset_Metadata.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "Dataset_Metadata_Report.txt"), "w", encoding="utf-8") as f:
        f.write("=== MULTILINGUAL DATASET METADATA REPORT ===\n\n")
        f.write(f"Total transcripts loaded: {metadata.get('Total Transcripts', 0)}\n")
        f.write(f"Total QA interactions: {metadata.get('Total QA Interactions', 0)}\n\n")

        f.write("--- Unique Sessions per Language ---\n")
        for lang, cnt in metadata.get('Unique Sessions per Language', {}).items():
            f.write(f"  {lang}: {cnt}\n")

        f.write("\n--- Unique Artifacts per Language ---\n")
        for lang, cnt in metadata.get('Unique Artifacts per Language', {}).items():
            f.write(f"  {lang}: {cnt}\n")

        f.write("\n--- Artifact Type Counts (Pottery vs Dogu) ---\n")
        for atype, cnt in metadata.get('Artifact Type Counts', {}).items():
            f.write(f"  {atype}: {cnt}\n")

        f.write("\n--- Artifact Type per Language ---\n")
        at_per_lang = metadata.get('Artifact Type per Language', {})
        for lang in TARGET_LANGUAGES:
            f.write(f"  {lang}: Pottery={at_per_lang.get('Pottery', {}).get(lang, 0)}, Dogu={at_per_lang.get('Dogu', {}).get(lang, 0)}\n")

        f.write("\n--- Emotion Block Counts ---\n")
        for emo, cnt in metadata.get('Emotion Block Counts', {}).items():
            f.write(f"  {emo}: {cnt}\n")

        f.write("\n--- Emotion Blocks per Language ---\n")
        emo_per_lang = metadata.get('Emotion Blocks per Language', {})
        for emo in EMOTION_COLS_STANDARD:
            f.write(f"  {emo}:\n")
            for lang in TARGET_LANGUAGES:
                f.write(f"    {lang}: {emo_per_lang.get(emo, {}).get(lang, 0)}\n")

        f.write("\n--- Session Total Duration Stats (seconds) per Language ---\n")
        stats = metadata.get('Session Total Duration Stats', {})
        for lang in TARGET_LANGUAGES:
            mean_val = stats.get('mean', {}).get(lang, np.nan)
            std_val = stats.get('std', {}).get(lang, np.nan)
            min_val = stats.get('min', {}).get(lang, np.nan)
            max_val = stats.get('max', {}).get(lang, np.nan)
            f.write(f"  {lang}: mean={mean_val:.2f} ± {std_val:.2f}, min={min_val:.2f}, max={max_val:.2f}\n")


if __name__ == "__main__":
    print("Initiating Fully Self-Contained Multilingual Analysis Pipeline...")

    df_qa, df_transcripts = load_multilingual_dataset(DATASET_ROOTS)
    if df_qa.empty:
        print(f"\nError: No valid QA data found in the designated roots.")
        sys.exit(1)

    df_durations, blocks_df = calculate_durations(df_qa)
    
    run_statistical_analysis(df_durations)
    export_itemized_averages_and_significance(df_durations, df_transcripts)
    generate_detailed_stats_report(df_durations, OUTPUT_DIR)
    generate_alternative_artifact_plots(df_durations)
    run_clustering(df_durations)
    
    # generate_timeline_barcharts(blocks_df)
    
    analyze_nlp(df_transcripts)
    generate_modality_mismatch_report(df_durations)
    save_metadata_report(df_qa, df_durations, df_transcripts, blocks_df)

    print(f"\nAll operations complete. Reports and charts are saved to '{os.path.abspath(OUTPUT_DIR)}'.")