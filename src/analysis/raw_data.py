import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_ROOT_MALAYSIA = r"D:\storage\jomon_kaen\jomon_kaen_dataset\malaysia"
DATASET_ROOT_JAPAN = r"D:\storage\jomon_kaen\jomon_kaen_dataset\japan"
OUTPUT_DIR = "raw_counts_and_verification"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DOGU_PREFIXES = [
    'IN0295', 'IN0306', 'MH0037', 'NM0239',
    'NZ0001', 'SK0035', 'TK0020', 'UD0028'
]

EMOTION_MAP = {
    "Interesting and attentional shape": "Interesting",
    "Beautiful and artistic": "Beautiful",
    "Strange and incomprehensible": "Strange",
    "Creepy / unsettling / scary": "Scary",
    "Feel nothing": "Feel nothing",
    "面白い・気になる形だ": "Interesting",
    "美しい・芸術的だ": "Beautiful",
    "不思議・意味不明": "Strange",
    "不気味・不安・怖い": "Scary",
    "何も感じない": "Feel nothing"
}

EMOTION_COLS = ["Interesting", "Beautiful", "Strange", "Scary", "Feel nothing"]
TARGET_LANGUAGES = ['ENGLISH', 'MALAY', 'CHINESE', 'JAPANESE']

# Hardcoded expected values from your 2-1_ObjType_ANOVA_E sheet for cross-verification
EXPECTED_STATS = {
    'Dogu': {
        'N': 8,
        'Interesting': {'Mean': 0.2594, 'Std': 0.0288},
        'Beautiful': {'Mean': 0.0812, 'Std': 0.0225},
        'Strange': {'Mean': 0.3035, 'Std': 0.0713},
        'Scary': {'Mean': 0.2856, 'Std': 0.0741},
        'Feel nothing': {'Mean': 0.0703, 'Std': 0.0290}
    },
    'Pottery': {
        'N': 85,
        'Interesting': {'Mean': 0.2782, 'Std': 0.0948},
        'Beautiful': {'Mean': 0.3225, 'Std': 0.1443},
        'Strange': {'Mean': 0.1494, 'Std': 0.0816},
        'Scary': {'Mean': 0.0675, 'Std': 0.0481},
        'Feel nothing': {'Mean': 0.1823, 'Std': 0.1237}
    }
}

# ==========================================
# 1. DATA LOADING
# ==========================================
def load_multilingual_dataset(root_dirs):
    qa_records = []
    for root_dir in root_dirs:
        root_path = Path(root_dir)
        if not root_path.exists():
            print(f"Warning: Directory not found: {root_dir}")
            continue
        for group_path in root_path.iterdir():
            if not group_path.is_dir(): continue
            for session_path in group_path.iterdir():
                if not session_path.is_dir(): continue
                
                language = None
                lang_file = session_path / 'language.txt'
                if lang_file.exists():
                    try:
                        text = lang_file.read_text(encoding='utf-8').strip().upper()
                        if 'ENGLISH' in text or 'EN ' in text: language = 'ENGLISH'
                        elif 'MALAY' in text or 'BM ' in text: language = 'MALAY'
                        elif 'CHINESE' in text or 'MANDARIN' in text: language = 'CHINESE'
                        elif 'JAPANESE' in text or 'JP ' in text: language = 'JAPANESE'
                    except: pass
                if language is None and "japan" in str(root_path).lower():
                    language = 'JAPANESE'
                if language not in TARGET_LANGUAGES: continue
                
                for pottery_path in session_path.iterdir():
                    if not pottery_path.is_dir(): continue
                    qa_file = pottery_path / "qa_corrected.csv"
                    if qa_file.exists():
                        try:
                            df_temp = pd.read_csv(qa_file, encoding='utf-8')
                            df_temp['timestamp'] = pd.to_numeric(df_temp['timestamp'], errors='coerce')
                            df_temp.dropna(subset=['timestamp'], inplace=True)
                            df_temp['pottery_id'] = pottery_path.name
                            df_temp['session_id'] = session_path.name
                            df_temp['Language'] = language
                            qa_records.append(df_temp)
                        except Exception as e:
                            print(f"Error loading {qa_file}: {e}")
                            
    return pd.concat(qa_records, ignore_index=True) if qa_records else pd.DataFrame()

# ==========================================
# 2. RAW COUNTS CALCULATION
# ==========================================
def calculate_raw_counts(df_qa):
    print("Calculating raw event counts per instance...")
    df = df_qa.copy()
    df['emotion'] = df['answer'].astype(str).str.strip().map(EMOTION_MAP)
    df.dropna(subset=['emotion', 'pottery_id', 'session_id'], inplace=True)
    
    # Keep only the 5 valid mapped emotions
    df = df[df['emotion'].isin(EMOTION_COLS)]
    
    # Count occurrences of each emotion per session x pottery
    raw_counts = df.groupby(['pottery_id', 'session_id', 'Language'])['emotion'].value_counts().unstack(fill_value=0).reset_index()
    
    # Ensure all emotion columns exist even if count is 0
    for emo in EMOTION_COLS:
        if emo not in raw_counts.columns:
            raw_counts[emo] = 0
            
    # Reorder columns
    cols = ['pottery_id', 'session_id', 'Language'] + EMOTION_COLS
    raw_counts = raw_counts[cols]
    
    # Calculate total count per instance
    raw_counts['Total_Count'] = raw_counts[EMOTION_COLS].sum(axis=1)
    
    return raw_counts

# ==========================================
# 3. CROSS-VERIFICATION LOGIC
# ==========================================
def cross_verify_stats(raw_counts):
    print("\n" + "="*70)
    print("CROSS-VERIFICATION: Recalculating Stats from Raw Counts")
    print("="*70)
    
    # Step A: Calculate instance-level proportions (Count / Total_Count)
    for emo in EMOTION_COLS:
        raw_counts[f'Prop_{emo}'] = raw_counts[emo] / raw_counts['Total_Count'].replace(0, np.nan)
        
    # Step B: Calculate pottery-level means (mean of session proportions per artifact)
    prop_cols = [f'Prop_{emo}' for emo in EMOTION_COLS]
    pottery_means = raw_counts.groupby('pottery_id')[prop_cols].mean().reset_index()
    pottery_means.rename(columns={f'Prop_{emo}': emo for emo in EMOTION_COLS}, inplace=True)
    
    # Step C: Assign Artifact Type (Dogu vs Pottery)
    pottery_means['Type'] = pottery_means['pottery_id'].apply(
        lambda x: 'Dogu' if any(p in str(x) for p in DOGU_PREFIXES) else 'Pottery')
        
    # Step D: Compare against expected values from your ANOVA sheet
    all_match = True
    for t in ['Dogu', 'Pottery']:
        group_data = pottery_means[pottery_means['Type'] == t]
        calc_n = len(group_data)
        exp_n = EXPECTED_STATS[t]['N']
        
        n_status = 'MATCH' if calc_n == exp_n else 'MISMATCH'
        print(f"\n[{t}] N Check: Expected = {exp_n}, Calculated = {calc_n} -> {n_status}")
        if calc_n != exp_n: all_match = False
        
        for emo in EMOTION_COLS:
            calc_mean = group_data[emo].mean()
            calc_std = group_data[emo].std(ddof=1)
            
            exp_mean = EXPECTED_STATS[t][emo]['Mean']
            exp_std = EXPECTED_STATS[t][emo]['Std']
            
            # Use np.isclose for floating point comparison (tolerance 1e-4)
            mean_match = np.isclose(calc_mean, exp_mean, atol=1e-4)
            std_match = np.isclose(calc_std, exp_std, atol=1e-4)
            
            status = "MATCH ✔" if (mean_match and std_match) else "MISMATCH ✘"
            if not (mean_match and std_match): all_match = False
            
            print(f"  {emo:<14} | Expected: Mean={exp_mean:.4f}, Std={exp_std:.4f} | "
                  f"Calculated: Mean={calc_mean:.4f}, Std={calc_std:.4f} -> {status}")
                  
    print("\n" + "="*70)
    if all_match:
        print("VERIFICATION SUCCESSFUL: All calculated stats perfectly match the ANOVA output!")
    else:
        print("VERIFICATION WARNING: Some values did not match exactly.")
    print("="*70 + "\n")

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    df_qa = load_multilingual_dataset([DATASET_ROOT_MALAYSIA, DATASET_ROOT_JAPAN])
    if df_qa.empty:
        print("Error: No data loaded.")
        return
        
    raw_counts = calculate_raw_counts(df_qa)
    
    # Export to Excel
    out_file = os.path.join(OUTPUT_DIR, "Raw_Instance_Counts.xlsx")
    raw_counts.to_excel(out_file, index=False)
    print(f"✓ Raw counts exported to: {out_file}")
    
    # Run Verification
    cross_verify_stats(raw_counts)

if __name__ == "__main__":
    main()