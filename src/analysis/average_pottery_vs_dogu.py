import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import os
from pathlib import Path
from tqdm import tqdm

# Dictionaries and Constants
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

# English/Malaysian Emotion Map
EMOTION_COLOR_MAP_EN = {
    "Interesting and attentional shape": "#00FFFF",
    "Beautiful and artistic": "#00FF00",
    "Strange and incomprehensible": "#FFFF00",
    "Creepy / unsettling / scary": "#FF0000",
    "Feel nothing": "#505050",
    "NO RESPONSE": "#D3D3D3",
}

SHORT_LABELS_EN = {
    "Interesting and attentional shape": "Interesting",
    "Beautiful and artistic": "Beautiful",
    "Strange and incomprehensible": "Strange",
    "Creepy / unsettling / scary": "Scary",
    "Feel nothing": "Feel nothing",
    "NO RESPONSE": "NO RESPONSE"
}

EMOTION_STACK_ORDER_EN = [
    "Interesting and attentional shape", "Beautiful and artistic",
    "Strange and incomprehensible", "Creepy / unsettling / scary",
    "Feel nothing", "NO RESPONSE"
]

# Japanese Emotion Map
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


# Data Loading Functions
def find_data_paths_detailed(root: str,
                             pottery_path_str: str,
                             limit: int = 1000) -> list:
    """Finds all qa_corrected.csv paths."""
    root, pottery_path = Path(root), Path(pottery_path_str)
    if not root.exists():
        raise ValueError(f"Root directory not found: {root}")
    if not pottery_path.exists():
        raise ValueError(f"Pottery directory not found: {pottery_path}")

    data = []
    pottery_ids = [
        f"{pid}({num})" for pid, num in ASSIGNED_NUMBERS_DICT.items()
    ]

    print(f"\nCHECKING RAW DATA PATHS")
    limit_dict = {pid: 0 for pid in pottery_ids}

    for g in os.listdir(root):
        group_path = root / g
        if not os.path.isdir(group_path):
            continue
        for s in tqdm(os.listdir(group_path), desc=g):
            session_path = group_path / s
            if not os.path.isdir(session_path):
                continue
            for p in os.listdir(session_path):
                if p in pottery_ids and limit_dict[p] < limit:
                    qa_path = session_path / p / "qa_corrected.csv"
                    if qa_path.exists():
                        limit_dict[p] += 1
                        data.append({
                            'qa': str(qa_path),
                            'GROUP': g,
                            'SESSION_ID': s,
                            'ID': p
                        })

    print(f"\nLoader finished. Found {len(data)} valid data instances.")
    return data


def load_combined_qna_data(root_dir: str,
                             pottery_models_dir: str) -> pd.DataFrame:
    """Loads and combines all QnA data from CSV files."""
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
            print(f"Could not read or process file {item['qa']}: {e}")

    if not df_list:
        return pd.DataFrame()

    print("\nCombining all data sources for analysis...")
    return pd.concat(df_list, ignore_index=True)


def analyze_pottery_vs_dogu(combined_df: pd.DataFrame,
                                language: str = 'malaysia'):
    """
    Analyzes and plots emotion responses, comparing Pottery (1-85) vs. 
    dogus (86-93).
    
    Args:
        combined_df: DataFrame with emotion response data
        language: Language setting for labels ('malaysia' or 'japan')
    """

    if language == 'japan':
        EMOTION_COLOR_MAP = EMOTION_COLOR_MAP_JP
        EMOTION_STACK_ORDER = [
            "何も感じない", "不気味・不安・怖い", "不思議・意味不明", "美しい・芸術的だ", "面白い・気になる形だ",
            "NO RESPONSE"
        ]
        EMOTION_SHORT_LABEL_MAP = SHORT_LABELS_JP
    else:
        EMOTION_COLOR_MAP = EMOTION_COLOR_MAP_EN
        EMOTION_STACK_ORDER = [
            "Feel nothing", "Creepy / unsettling / scary",
            "Strange and incomprehensible", "Beautiful and artistic",
            "Interesting and attentional shape", "NO RESPONSE"
        ]
        EMOTION_SHORT_LABEL_MAP = SHORT_LABELS_EN

    if combined_df.empty:
        print("Combined DataFrame is empty. No analysis performed.")
        return

    # Add short_answer column
    combined_df['answer'] = combined_df['answer'].str.strip()
    combined_df['short_answer'] = combined_df['answer'].map(
        EMOTION_SHORT_LABEL_MAP)
    
    # --- New Grouping Logic ---
    # Extract assigned number from pottery_id (e.g., "AS0001(1)" -> "1")
    combined_df['assigned_num_str'] = combined_df['pottery_id'].str.split(
        '(', expand=True)[1].str.replace(')', '')
    combined_df['assigned_num'] = pd.to_numeric(
        combined_df['assigned_num_str'], errors='coerce')
    
    # Assign type based on number
    def assign_type(num):
        if pd.isna(num):
            return 'Other'
        if 1 <= num <= 85:
            return 'Pottery'
        elif 86 <= num <= 93:
            return 'dogu'
        else:
            return 'Other'

    combined_df['Type'] = combined_df['assigned_num'].apply(assign_type)
    
    # Filter out 'Other' types
    analysis_df = combined_df[combined_df['Type'].isin(
        ['Pottery', 'dogu'])].copy()
    
    if analysis_df.empty:
        print("No data found for Pottery or dogu types. Exiting.")
        return
        
    print(f"\nFound {len(analysis_df[analysis_df['Type'] == 'Pottery'])} responses for Pottery.")
    print(f"Found {len(analysis_df[analysis_df['Type'] == 'dogu'])} responses for dogu.")

    # Calculate percentage by event count (session-normalized)
    session_counts_df = pd.crosstab(
        [analysis_df['Type'], analysis_df['session_id']],
        analysis_df['short_answer'])
    
    # Add missing emotion columns if any
    all_emotions = EMOTION_SHORT_LABEL_MAP.values()
    for col in all_emotions:
        if col not in session_counts_df.columns:
            session_counts_df[col] = 0
            
    session_percentage_df = session_counts_df.div(
        session_counts_df.sum(axis=1), axis=0) * 100
    
    # Group by our new 'Type' to get the final average percentages
    percentage_df = session_percentage_df.groupby('Type').mean()

    # --- Plotting ---
    output_dir = "pottery_dogu_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Map colors to short labels
    short_label_color_map = {
        EMOTION_SHORT_LABEL_MAP[k]: v
        for k, v in EMOTION_COLOR_MAP.items()
    }
    emotion_order = [
        EMOTION_SHORT_LABEL_MAP[e] for e in EMOTION_STACK_ORDER
        if e in EMOTION_COLOR_MAP.keys() and e != "NO RESPONSE"
    ]
    
    # Ensure all emotions are in the dataframe, fill with 0 if not
    for emotion in emotion_order:
        if emotion not in percentage_df.columns:
            percentage_df[emotion] = 0.0
            
    plot_colors = [
        short_label_color_map.get(e, '#CCCCCC') for e in emotion_order
    ]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot stacked bar chart
    percentage_df[emotion_order].plot(kind='bar',
                                      stacked=True,
                                      ax=ax,
                                      color=plot_colors,
                                      width=0.6)

    # Add value labels
    for container in ax.containers:
        labels = [
            f'{v:.1f}' if v > 2 else '' for v in container.datavalues
        ]
        ax.bar_label(container,
                     labels=labels,
                     label_type='center',
                     fontsize=12,
                     color='black',
                     weight='bold')

    ax.set_title('Emotion Response: Pottery (No. 1-85) vs. dogu (No. 86-93)',
                 fontsize=14,
                 pad=20)
    ax.set_ylabel('Average Percentage (%)', fontsize=12)
    ax.set_xlabel('Artifact Type', fontsize=12)
    ax.set_ylim(0, 100)
    ax.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

    # Save plot
    filename = "pottery_vs_dogu_emotion_analysis.png"
    plt.savefig(os.path.join(output_dir, filename),
                dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # Save summary CSV
    percentage_df.to_csv(os.path.join(output_dir, 'pottery_vs_dogu_summary.csv'))

    print(f"\nAnalysis complete! Results saved to '{output_dir}/' directory")
    print(f"Generated plot: {filename}")


# Main Execution
if __name__ == "__main__":
    # === USER CONTROLS ===
    # SELECTED_LANGUAGE = 'malaysia'
    SELECTED_LANGUAGE = 'japan'

    # DATASET_ROOT_DIR = "./src/jomon_kaen_dataset/malaysia"
    DATASET_ROOT_DIR = "./src/jomon_kaen_dataset/japan"
    POTTERY_MODELS_DIR = "./src/pottery"
    # === END USER CONTROLS ===

    try:
        # Load emotion data
        combined_dataframe = load_combined_qna_data(DATASET_ROOT_DIR,
                                                    POTTERY_MODELS_DIR)

        # Run the new pottery vs. dogu analysis
        if not combined_dataframe.empty:
            analyze_pottery_vs_dogu(
                combined_dataframe,
                language=SELECTED_LANGUAGE)
        else:
            print("No data available for analysis.")

    except (FileNotFoundError, ValueError) as e:
        print(f"Could not run analysis due to an error: {e}")
        print(
            "Please ensure all directories are set up correctly."
        )
