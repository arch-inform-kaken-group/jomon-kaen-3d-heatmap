import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import os
from pathlib import Path
from tqdm import tqdm

# --- Dictionaries and Constants ---
ASSIGNED_NUMBERS_DICT = {
    'AS0001': '1', 'FH0008': '2', 'IN0003': '3', 'IN0008': '4', 'IN0009': '5',
    'IN0017': '6', 'IN0081': '7', 'IN0104': '8', 'IN0135': '9', 'IN0148': '10',
    'IN0220': '11', 'IN0228': '12', 'IN0232': '13', 'IN0239': '14', 'IN0277': '15',
    'MY0001': '16', 'MY0002': '17', 'MY0004': '18', 'MY0006': '19', 'MY0007': '20',
    'ND0001': '21', 'NM0001': '22', 'NM0002': '23', 'NM0009': '24', 'NM0010': '25',
    'NM0014': '26', 'NM0015': '27', 'NM0017': '28', 'NM0041': '29', 'NM0049': '30',
    'NM0066': '31', 'NM0070': '32', 'NM0072': '33', 'NM0073': '34', 'NM0079': '35',
    'NM0080': '36', 'NM0099': '37', 'NM0106': '38', 'NM0133': '39', 'NM0135': '40',
    'NM0144': '41', 'NM0154': '42', 'NM0156': '43', 'NM0159': '44', 'NM0168': '45',
    'NM0173': '46', 'NM0175': '47', 'NM0189': '48', 'NM0191': '49', 'NM0206': '50',
    'SB0002': '51', 'SB0004': '52', 'SI0001': '53', 'SJ0503': '54', 'SJ0504': '55',
    'SK0001': '56', 'SK0002': '57', 'SK0003': '58', 'SK0004': '59', 'SK0005': '60',
    'SK0013': '61', 'SS0001': '62', 'TJ0004': '63', 'TJ0005': '64', 'TJ0010': '65',
    'TK0002': '66', 'TK0048': '67', 'TK0057': '68', 'UD0001': '69', 'UD0003': '70',
    'UD0005': '71', 'UD0006': '72', 'UD0011': '73', 'UD0013': '74', 'UD0014': '75',
    'UD0016': '76', 'UD0023': '77', 'UD0302': '78', 'UD0304': '79', 'UD0308': '80',
    'UD0318': '81', 'UD0322': '82', 'UD0411': '83', 'UD0412': '84', 'UK0001': '85',
    'IN0295': '86', 'IN0306': '87', 'MH0037': '88', 'NM0239': '89', 'NZ0001': '90',
    'SK0035': '91', 'TK0020': '92', 'UD0028': '93', 'rembak7': 'A'
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
    "Feel nothing", "Creepy / unsettling / scary",
    "Strange and incomprehensible", "Beautiful and artistic",
    "Interesting and attentional shape", "NO RESPONSE"
]

# Japanese Emotion Map
EMOTION_COLOR_MAP_JP = {
    "面白い・気になる形だ": "#00FFFF",
    "美しい・芸術的だ": "#00FF00",
    "不思議・意味不明": "#FFFF00",
    "不気味・不安・怖い": "#FF0000",
    "何も感じない": "#505050",
    "NO RESPONSE": "#D3D3D3",  # Assuming "NO RESPONSE" stays constant
}

# Mapping long labels to short labels for plots and CSVs
SHORT_LABELS_JP = {
    "面白い・気になる形だ": "面白い",
    "美しい・芸術的だ": "美しい",
    "不思議・意味不明": "不思議",
    "不気味・不安・怖い": "怖い",
    "何も感じない": "何も感じない",
    "NO RESPONSE": "NO RESPONSE"
}

# --- Data Loading Functions ---
def find_data_paths_detailed(root: str, pottery_path_str: str, limit: int = 1000) -> list:
    root, pottery_path = Path(root), Path(pottery_path_str)
    if not root.exists(): 
        raise ValueError(f"Root directory not found: {root}")
    if not pottery_path.exists():
        raise ValueError(f"Pottery directory not found: {pottery_path}")
    
    data = []
    pottery_ids = [f"{pid}({num})" for pid, num in ASSIGNED_NUMBERS_DICT.items()]
    
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


def load_combined_qna_data(root_dir: str, pottery_models_dir: str) -> pd.DataFrame:
    data_to_process = find_data_paths_detailed(root=root_dir, pottery_path_str=pottery_models_dir)
    if not data_to_process: 
        return pd.DataFrame()
    
    df_list = []
    for item in tqdm(data_to_process, desc="Loading and combining data"):
        try:
            temp_df = pd.read_csv(item['qa'], header=0, sep=",")
            temp_df['timestamp'] = pd.to_numeric(temp_df['timestamp'], errors='coerce')
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


def analyze_emotions_by_features(combined_df: pd.DataFrame, features_csv: str, 
                                language: str = 'malaysia', selected_features: list = None,
                                include_shape_analysis: bool = True):
    """Analyzes and plots emotion responses categorized by pottery features.
    
    Args:
        combined_df: DataFrame with emotion response data
        features_csv: Path to CSV file with pottery features
        language: Language setting for labels
        selected_features: List of specific features to analyze. If None, analyzes all features.
        include_shape_analysis: Whether to include shape type analysis
    """

    if language == 'japan':
        EMOTION_COLOR_MAP = EMOTION_COLOR_MAP_JP
        EMOTION_STACK_ORDER = [
            "何も感じない", "不気味・不安・怖い", "不思議・意味不明", "美しい・芸術的だ", "面白い・気になる形だ",
            "NO RESPONSE"
        ]
        EMOTION_SYMBOL_MAP = {
            "面白い・気になる形だ": "◇",
            "美しい・芸術的だ": "□",
            "不思議・意味不明": "△",
            "不気味・不安・怖い": "X",
            "何も感じない": "○",
            "NO RESPONSE": "・"
        }
        EMOTION_SHORT_LABEL_MAP = SHORT_LABELS_JP
    else:
        EMOTION_COLOR_MAP = EMOTION_COLOR_MAP_EN
        EMOTION_STACK_ORDER = [
            "Feel nothing", "Creepy / unsettling / scary",
            "Strange and incomprehensible", "Beautiful and artistic",
            "Interesting and attentional shape", "NO RESPONSE"
        ]
        EMOTION_SYMBOL_MAP = {
            "Interesting and attentional shape": "◇",
            "Beautiful and artistic": "□",
            "Strange and incomprehensible": "△",
            "Creepy / unsettling / scary": "X",
            "Feel nothing": "○",
            "NO RESPONSE": "・"
        }
        EMOTION_SHORT_LABEL_MAP = SHORT_LABELS_EN
    
    if combined_df.empty:
        print("Combined DataFrame is empty. No analysis performed.")
        return
    
    # Load features CSV
    features_df = pd.read_csv(features_csv)
    
    # Extract pottery code from the CODE column (e.g., "AS0001(1).ply" -> "AS0001(1)")
    features_df['pottery_id'] = features_df['CODE'].str.replace('.ply', '')
    
    # Add short_answer column
    combined_df['answer'] = combined_df['answer'].str.strip()
    combined_df['short_answer'] = combined_df['answer'].map(EMOTION_SHORT_LABEL_MAP)
    
    # Calculate percentage by event count (session-normalized)
    session_counts_df = pd.crosstab([combined_df['pottery_id'], combined_df['session_id']], 
                                    combined_df['short_answer'])
    session_percentage_df = session_counts_df.div(session_counts_df.sum(axis=1), axis=0) * 100
    percentage_df = session_percentage_df.groupby('pottery_id').mean()
    
    # Merge with features
    merged_df = features_df.merge(percentage_df, on='pottery_id', how='inner')
    
    # Create output directory
    output_dir = "feature_emotion_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get feature columns (exclude CODE and pottery_id)
    all_feature_columns = [col for col in features_df.columns 
                          if col not in ['CODE', 'pottery_id'] and not col.startswith('SHAPE_TYPE_')]
    
    # Filter to selected features if specified
    if selected_features is not None and len(selected_features) > 0:
        feature_columns = [f for f in selected_features if f in all_feature_columns]
        if len(feature_columns) == 0:
            print(f"Warning: None of the selected features found in CSV. Available features:")
            print(", ".join(all_feature_columns))
            return
        print(f"\nAnalyzing {len(feature_columns)} selected features...")
        print(f"Features: {', '.join(feature_columns)}")
    else:
        feature_columns = all_feature_columns
        print(f"\nAnalyzing all {len(feature_columns)} binary features...")
    
    # Add SHAPE_TYPE as a special aggregated feature
    shape_type_columns = [col for col in features_df.columns if col.startswith('SHAPE_TYPE_')]
    
    # Map colors to short labels
    short_label_color_map = {EMOTION_SHORT_LABEL_MAP[k]: v for k, v in EMOTION_COLOR_MAP.items()}
    emotion_order = [EMOTION_SHORT_LABEL_MAP[e] for e in EMOTION_STACK_ORDER 
                     if e in EMOTION_COLOR_MAP.keys() and e != "NO RESPONSE"]
    plot_colors = [short_label_color_map.get(e, '#CCCCCC') for e in emotion_order]
    
    # Analyze binary features
    for feature in tqdm(feature_columns, desc="Processing binary features"):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Split data by feature value (0 or 1)
        feature_groups = merged_df.groupby(feature)
        
        results = []
        for value, group in feature_groups:
            if len(group) == 0:
                continue
            
            # Calculate average emotion percentages for this group
            emotion_avgs = group[emotion_order].mean()
            results.append({
                'value': 'Yes' if value == 1.0 else 'No',
                **emotion_avgs.to_dict()
            })
        
        if not results:
            plt.close(fig)
            continue
        
        results_df = pd.DataFrame(results).set_index('value')
        
        # Plot stacked bar chart
        results_df[emotion_order].plot(kind='bar', stacked=True, ax=ax, 
                                       color=plot_colors, width=0.6)
        
        # Add value labels
        for container in ax.containers:
            labels = [f'{v:.1f}' if v > 2 else '' for v in container.datavalues]
            ax.bar_label(container, labels=labels, label_type='center', 
                        fontsize=9, color='black', weight='bold')
        
        ax.set_title(f'Emotion Response by Feature: {feature}', fontsize=14, pad=20)
        ax.set_ylabel('Average Percentage (%)', fontsize=11)
        ax.set_xlabel('Feature Present', fontsize=11)
        ax.set_ylim(0, 100)
        ax.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        filename = f"{feature.lower().replace(' ', '_')}_emotion_analysis.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # Analyze SHAPE_TYPE
    if include_shape_analysis:
        print("\nAnalyzing shape types...")
        
        # Create a single column for shape type
        def get_shape_type(row):
            for col in shape_type_columns:
                if row[col] == 1:
                    return col.replace('SHAPE_TYPE_', '')
            return 'Unknown'
        
        merged_df['shape_type'] = merged_df.apply(get_shape_type, axis=1)
        
        # Group by shape type
        shape_groups = merged_df.groupby('shape_type')
        
        shape_results = []
        for shape, group in shape_groups:
            if len(group) < 2:  # Skip if too few samples
                continue
            emotion_avgs = group[emotion_order].mean()
            shape_results.append({
                'shape_type': shape,
                'count': len(group),
                **emotion_avgs.to_dict()
            })
        
        if shape_results:
            shape_df = pd.DataFrame(shape_results).set_index('shape_type')
            
            fig, ax = plt.subplots(figsize=(14, 8))
            shape_df[emotion_order].plot(kind='bar', stacked=True, ax=ax, 
                                         color=plot_colors, width=0.7)
            
            for container in ax.containers:
                labels = [f'{v:.1f}' if v > 2 else '' for v in container.datavalues]
                ax.bar_label(container, labels=labels, label_type='center', 
                            fontsize=8, color='black', weight='bold')
            
            ax.set_title('Emotion Response by Pottery Shape Type', fontsize=16, pad=20)
            ax.set_ylabel('Average Percentage (%)', fontsize=12)
            ax.set_xlabel('Shape Type', fontsize=12)
            ax.set_ylim(0, 100)
            ax.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, 'shape_type_emotion_analysis.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Save summary CSV
            shape_df.to_csv(os.path.join(output_dir, 'shape_type_summary.csv'))
    
    print(f"\nAnalysis complete! Results saved to '{output_dir}/' directory")
    feature_count = len(feature_columns)
    shape_count = 1 if include_shape_analysis else 0
    print(f"Generated {feature_count} binary feature plot(s) + {shape_count} shape type plot(s)")


# --- Main Execution ---
if __name__ == "__main__":
    # === USER CONTROLS ===
    # SELECTED_LANGUAGE = 'malaysia'
    SELECTED_LANGUAGE = 'japan'
    POTTERY_SELECTION = []  # Empty list for all pottery
    INCLUDE_POTTERY = True
    
    # DATASET_ROOT_DIR = "./src/jomon_kaen_dataset/malaysia"
    DATASET_ROOT_DIR = "./src/jomon_kaen_dataset/japan"
    POTTERY_MODELS_DIR = "./src/pottery"
    FEATURES_CSV = "./src/analysis/DS_Labels_Cleaned.csv"
    
    # SELECT FEATURES TO ANALYZE
    # Option 1: Analyze ALL features (set to None or empty list)
    SELECTED_FEATURES = None
    
    # Option 2: Analyze specific features (uncomment and modify as needed)
    # SELECTED_FEATURES = [
    #     'HAS_FLAME_LIKE_DECORATION',
    #     'HAS_CROWN_LIKE_DECORATION', 
    #     'HAS_HANDLES',
    #     'HAS_SPIRAL_PATTERN'
    # ]
    
    # Include shape type analysis?
    INCLUDE_SHAPE_TYPE_ANALYSIS = True
    # === END USER CONTROLS ===
    
    try:
        # Load emotion data
        combined_dataframe = load_combined_qna_data(DATASET_ROOT_DIR, POTTERY_MODELS_DIR)
        
        # Filter pottery if selection is specified
        if not combined_dataframe.empty and POTTERY_SELECTION:
            print(f"\nFiltering pottery based on selection (Include mode: {INCLUDE_POTTERY})...")
            base_ids = combined_dataframe['pottery_id'].str.split('(', expand=True)[0]
            initial_count = len(combined_dataframe['pottery_id'].unique())
            
            if INCLUDE_POTTERY:
                combined_dataframe = combined_dataframe[base_ids.isin(POTTERY_SELECTION)]
                print(f"Included {len(combined_dataframe['pottery_id'].unique())} of {initial_count} unique pottery items.")
            else:
                combined_dataframe = combined_dataframe[~base_ids.isin(POTTERY_SELECTION)]
                print(f"Excluded IDs, {len(combined_dataframe['pottery_id'].unique())} of {initial_count} unique items remaining.")
        
        # Run feature-based analysis
        if not combined_dataframe.empty:
            analyze_emotions_by_features(combined_dataframe, FEATURES_CSV, 
                                        language=SELECTED_LANGUAGE,
                                        selected_features=SELECTED_FEATURES,
                                        include_shape_analysis=INCLUDE_SHAPE_TYPE_ANALYSIS)
        else:
            print("No data available for analysis.")
            
    except (FileNotFoundError, ValueError) as e:
        print(f"Could not run analysis due to an error: {e}")
        print("Please ensure all directories and the features CSV are set up correctly.")