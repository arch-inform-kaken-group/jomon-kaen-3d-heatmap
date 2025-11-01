import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import os
from pathlib import Path
from tqdm import tqdm

# Dictionaries and Constants
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
    'SK0035': '91', 'TK0020': '92', 'UD0028': '93',
}

# English/Malaysian Emotion Map
SHORT_LABELS_EN = {
    "Interesting and attentional shape": "Interesting",
    "Beautiful and artistic": "Beautiful",
    "Strange and incomprehensible": "Strange",
    "Creepy / unsettling / scary": "Scary",
    "Feel nothing": "Feel nothing",
    "NO RESPONSE": "NO RESPONSE"
}

# Japanese Emotion Map
SHORT_LABELS_JP_MAP = {
    "面白い・気になる形だ": "Interesting",
    "美しい・芸術的だ": "Beautiful",
    "不思議・意味不明": "Strange",
    "不気味・不安・怖い": "Scary",
    "何も感じない": "Feel nothing",
    "NO RESPONSE": "NO RESPONSE"
}

# Standard 5 emotions for analysis
EMOTION_COLS_STANDARD = [
    "Interesting", "Beautiful", "Strange", "Scary", "Feel nothing"
]


# Data Loading Functions
def find_data_paths_detailed(root: str,
                             pottery_path_str: str,
                             limit: int = 1000) -> list:
    root, pottery_path = Path(root), Path(pottery_path_str)
    if not root.exists():
        raise ValueError(f"Root directory not found: {root}")
    if not pottery_path.exists():
        raise ValueError(f"Pottery directory not found: {pottery_path}")

    data = []
    pottery_ids = [
        f"{pid}({num})" for pid, num in ASSIGNED_NUMBERS_DICT.items()
    ]

    print(f"\nCHECKING RAW DATA PATHS at {root}")
    limit_dict = {pid: 0 for pid in pottery_ids}

    for g in os.listdir(root):
        group_path = root / g
        if not os.path.isdir(group_path):
            continue
        for s in tqdm(os.listdir(group_path), desc=g, leave=False):
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

    print(f"Loader finished for {root}. Found {len(data)} valid data instances.")
    return data


def load_combined_qna_data(root_dir: str,
                           pottery_models_dir: str) -> pd.DataFrame:
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
            temp_df.sort_values(by='timestamp', inplace=True) # Ensure data is sorted
            temp_df['pottery_id'] = item['ID']
            temp_df['session_id'] = item['SESSION_ID']
            df_list.append(temp_df)
        except Exception as e:
            print(f"Could not read or process file {item['qa']}: {e}")

    if not df_list:
        return pd.DataFrame()

    return pd.concat(df_list, ignore_index=True)


# STATISTICS FUNCTIONS

def _get_emotion_stats_by_country(df: pd.DataFrame,
                                  language: str,
                                  country_prefix: str) -> pd.DataFrame:
    """
    Calculates descriptive stats (avg, std, min, max, count) for
    session-normalized emotion *percentages* (from event counts).
    """
    if df.empty:
        return pd.DataFrame()

    print(f"Processing Event Count (Percentage) Stats for {country_prefix}")
    
    # 1. Map answers to standard labels
    if language == 'japan':
        def map_jp(answer):
            return SHORT_LABELS_JP_MAP.get(str(answer).strip(), None)
        df['short_answer'] = df['answer'].apply(map_jp)
    else:  # malaysia
        def map_en(answer):
            return SHORT_LABELS_EN.get(str(answer).strip(), None)
        df['short_answer'] = df['answer'].apply(map_en)

    df.dropna(subset=['short_answer', 'pottery_id', 'session_id'],
              inplace=True)

    if df.empty:
        print(f"No valid emotion data found for {country_prefix}")
        return pd.DataFrame()

    # 2. Calculate Percentages per session
    session_counts = pd.crosstab(
        [df['pottery_id'], df['session_id']],
        df['short_answer'])
    session_pct = session_counts.div(session_counts.sum(axis=1),
                                     axis=0) * 100

    # 3. Ensure all 5 emotion columns exist before aggregating
    for col in EMOTION_COLS_STANDARD:
        if col not in session_pct.columns:
            session_pct[col] = 0.0
    session_pct = session_pct[EMOTION_COLS_STANDARD] # Enforce column order

    # 4. Aggregate stats (mean, std, min, max, COUNT) across sessions
    pottery_stats = session_pct.groupby('pottery_id').agg(
        ['mean', 'std', 'min', 'max', 'count'] # ADDED 'count'
    ).fillna(0.0)

    # 5. Flatten MultiIndex columns and rename
    final_df = pd.DataFrame(index=pottery_stats.index)
    
    # Add Session Count (same for all columns, so we pick the first)
    final_df[f"{country_prefix}Session_Count"] = pottery_stats[
        (EMOTION_COLS_STANDARD[0], 'count')
    ].astype(int)
    
    for emotion in EMOTION_COLS_STANDARD:
        final_df[f"{country_prefix}Avg_{emotion}_Pct"] = pottery_stats[(emotion, 'mean')]
        final_df[f"{country_prefix}Std_{emotion}_Pct"] = pottery_stats[(emotion, 'std')]
        final_df[f"{country_prefix}Min_{emotion}_Pct"] = pottery_stats[(emotion, 'min')]
        final_df[f"{country_prefix}Max_{emotion}_Pct"] = pottery_stats[(emotion, 'max')]

    return final_df


def _get_gaze_stats_by_country(df_in: pd.DataFrame,
                               language: str,
                               country_prefix: str) -> pd.DataFrame:
    """
    Calculates descriptive stats (avg, std, min, max, count) for
    accumulated gaze *duration* (in seconds) AND *duration percentage*.
    """
    if df_in.empty:
        return pd.DataFrame()

    print(f"Processing Actual Duration (Seconds & Pct) Stats for {country_prefix}")
    
    df = df_in.copy() # Avoid modifying the original dataframe

    # 1. Map answers to standard labels
    if language == 'japan':
        def map_jp(answer):
            return SHORT_LABELS_JP_MAP.get(str(answer).strip(), None)
        df['short_answer'] = df['answer'].apply(map_jp)
    else:  # malaysia
        def map_en(answer):
            return SHORT_LABELS_EN.get(str(answer).strip(), None)
        df['short_answer'] = df['answer'].apply(map_en)

    df.dropna(subset=['short_answer', 'pottery_id', 'session_id'],
              inplace=True)
              
    if df.empty:
        print(f"No valid emotion data found for {country_prefix} gaze.")
        return pd.DataFrame()

    # 2. Calculate duration blocks
    df.sort_values(by=['pottery_id', 'session_id', 'timestamp'], inplace=True)
    df['time_diff'] = df.groupby(['pottery_id', 'session_id'])['timestamp'].diff()
    emotion_changed = df['short_answer'] != df.groupby(
        ['pottery_id', 'session_id'])['short_answer'].shift()
    time_gap_exceeded = df['time_diff'] > 0.05 
    df['block_id'] = (emotion_changed | time_gap_exceeded).cumsum()

    # 3. Aggregate block durations
    block_durations = df.groupby(
        ['pottery_id', 'session_id', 'block_id']
    ).agg(
        start_time=('timestamp', 'min'),
        end_time=('timestamp', 'max'),
        answer=('short_answer', 'first')
    ).reset_index()
    block_durations['duration'] = block_durations['end_time'] - block_durations['start_time']
    
    # 4. Sum durations per emotion *per session* (ACTUAL DURATION IN SECONDS)
    session_emotion_durations = block_durations.groupby(
        ['pottery_id', 'session_id', 'answer']
    )['duration'].sum().unstack(fill_value=0.0)

    # 5. Calculate total session duration (min to max timestamp)
    session_total_dur_agg = df.groupby(
        ['pottery_id', 'session_id']
    )['timestamp'].agg(['min', 'max'])
    session_total_duration = (session_total_dur_agg['max'] - 
                              session_total_dur_agg['min'])
    # Replace 0 duration sessions with NaN to avoid divide-by-zero
    session_total_duration = session_total_duration.replace(0, np.nan) 

    # 6. Calculate DURATION PERCENTAGE
    # Ensure all 5 emotion columns exist in duration stats
    for col in EMOTION_COLS_STANDARD:
        if col not in session_emotion_durations.columns:
            session_emotion_durations[col] = 0.0
    session_emotion_durations = session_emotion_durations[EMOTION_COLS_STANDARD]

    # Divide emotion duration by total session duration
    session_emotion_duration_pct = session_emotion_durations.div(
        session_total_duration, axis=0
    ) * 100
    
    # 7. Aggregate stats for ACTUAL DURATION (SECONDS)
    pottery_stats_dur = session_emotion_durations.groupby('pottery_id').agg(
        ['mean', 'std', 'min', 'max', 'count'] # ADDED 'count'
    ).fillna(0.0)

    # 8. Aggregate stats for DURATION PERCENTAGE
    pottery_stats_pct = session_emotion_duration_pct.groupby('pottery_id').agg(
        ['mean', 'std', 'min', 'max'] # No need for count here, already got it
    ).fillna(0.0)

    # 9. Flatten and rename columns
    final_df_dur = pd.DataFrame(index=pottery_stats_dur.index)
    final_df_pct = pd.DataFrame(index=pottery_stats_pct.index)

    # Add Session Count (from duration stats)
    final_df_dur[f"{country_prefix}Session_Count"] = pottery_stats_dur[
        (EMOTION_COLS_STANDARD[0], 'count')
    ].astype(int)

    for emotion in EMOTION_COLS_STANDARD:
        # Actual Duration (Seconds)
        final_df_dur[f"{country_prefix}Avg_{emotion}_Dur"] = pottery_stats_dur[(emotion, 'mean')]
        final_df_dur[f"{country_prefix}Std_{emotion}_Dur"] = pottery_stats_dur[(emotion, 'std')]
        final_df_dur[f"{country_prefix}Min_{emotion}_Dur"] = pottery_stats_dur[(emotion, 'min')]
        final_df_dur[f"{country_prefix}Max_{emotion}_Dur"] = pottery_stats_dur[(emotion, 'max')]
        
        # Duration Percentage
        final_df_pct[f"{country_prefix}Avg_{emotion}_DurPct"] = pottery_stats_pct[(emotion, 'mean')]
        final_df_pct[f"{country_prefix}Std_{emotion}_DurPct"] = pottery_stats_pct[(emotion, 'std')]
        final_df_pct[f"{country_prefix}Min_{emotion}_DurPct"] = pottery_stats_pct[(emotion, 'min')]
        final_df_pct[f"{country_prefix}Max_{emotion}_DurPct"] = pottery_stats_pct[(emotion, 'max')]
    
    # 10. Combine the two dataframes (Dur and DurPct)
    final_df = pd.concat([final_df_dur, final_df_pct], axis=1)
    
    return final_df


def generate_all_pottery_statistics(
        root_japan: str,
        root_malaysia: str,
        pottery_models_dir: str,
        output_event_pct_file: str = "pottery_event_count_statistics.xlsx",
        output_duration_file: str = "pottery_duration_statistics.xlsx"):
    """
    Generates two separate Excel files: one for event count stats
    and one for duration (seconds and percentage) stats.
    """

    print(" Starting Combined Statistics Generation")

    # 1. Load Raw Data
    print("\nLoading Japan QnA data")
    df_japan = load_combined_qna_data(root_japan, pottery_models_dir)
    print("\nLoading Malaysia QnA data")
    df_malaysia = load_combined_qna_data(root_malaysia, pottery_models_dir)

    # 2. Process Event Count (Percentage) Stats
    stats_emo_jp = _get_emotion_stats_by_country(df_japan, 'japan', "JAPAN_")
    stats_emo_my = _get_emotion_stats_by_country(df_malaysia, 'malaysia', "MALAYSIA_")

    # 3. Process Actual Duration (Seconds & Pct) Stats
    stats_gaze_jp = _get_gaze_stats_by_country(df_japan,
                                               "japan",
                                               "JAPAN_")
    stats_gaze_my = _get_gaze_stats_by_country(df_malaysia,
                                               "malaysia",
                                               "MALAYSIA_")

    # 4. Get Master Pottery List
    all_pottery_ids = [
        f"{pid}({num})" for pid, num in ASSIGNED_NUMBERS_DICT.items()
    ]
    base_df = pd.DataFrame(index=all_pottery_ids)
    base_df.index.name = "pottery_id"

    # 5. Create and Save FILE 1: Event Count Stats
    print(f"\nCombining Event Count stats")
    df_event_pct = base_df.join(
        [stats_emo_jp, stats_emo_my], how='left'
    ).fillna(0.0)
    
    # Re-order columns to put Count first
    cols_event = sorted(df_event_pct.columns, key=lambda x: "Session_Count" not in x)
    df_event_pct = df_event_pct[cols_event]
    
    df_event_pct.to_excel(output_event_pct_file)
    print("\n" + "="*50)
    print("SUCCESS: Event Count statistics file generated!")
    print(f"File saved as: {output_event_pct_file}")
    print(f"Total Columns: {len(df_event_pct.columns)} (40 stats + 2 counts)")
    print("="*50)

    # 6. Create and Save FILE 2: Duration Stats
    print(f"\nCombining Duration (Seconds & Pct) stats")
    df_duration = base_df.join(
        [stats_gaze_jp, stats_gaze_my], how='left'
    ).fillna(0.0)

    # Re-order columns to put Count first
    cols_dur = sorted(df_duration.columns, key=lambda x: "Session_Count" not in x)
    df_duration = df_duration[cols_dur]

    df_duration.to_excel(output_duration_file)
    print("\n" + "="*50)
    print("SUCCESS: Duration statistics file generated!")
    print(f"File saved as: {output_duration_file}")
    print(f"Total Columns: {len(df_duration.columns)} (80 stats + 2 counts)")
    print("="*50)
    
    # Return combined dataframes for sanity check
    return df_japan, df_malaysia


# SANITY CHECK FUNCTIONS

def _load_features(features_csv: str):
    """Loads and prepares the features dataframe."""
    try:
        features_df = pd.read_csv(features_csv)
        features_df['pottery_id'] = features_df['CODE'].str.replace(
            '.ply', '', regex=False
        )
        feature_columns = [
            col for col in features_df.columns if
            col not in ['CODE', 'pottery_id'] and not col.startswith('SHAPE_TYPE_')
        ]
        return features_df, feature_columns
    except FileNotFoundError:
        print(f"Sanity Check ERROR: Cannot find features file: {features_csv}")
        return None, []

def run_sanity_check_from_sessions(df_japan: pd.DataFrame,
                                   df_malaysia: pd.DataFrame,
                                   features_csv: str):
    """
    METHOD A: Calculates feature stats directly from all session-level data.
    This is equivalent to the `analyze_emotions_by_features` script.
    """
    
    print("\n" + "="*50)
    print(" SANITY CHECK (METHOD A: From SESSIONS)")
    print("This method gives EQUAL WEIGHT to every SESSION.")
    print("="*50)

    # 1. Combine and Standardize Data
    print("Standardizing Japan and Malaysia data")
    if not df_japan.empty:
        df_japan['short_answer'] = df_japan['answer'].apply(
            lambda x: SHORT_LABELS_JP_MAP.get(str(x).strip(), None)
        )
    if not df_malaysia.empty:
        df_malaysia['short_answer'] = df_malaysia['answer'].apply(
            lambda x: SHORT_LABELS_EN.get(str(x).strip(), None)
        )
    combined_df = pd.concat([df_japan, df_malaysia], ignore_index=True)
    combined_df.dropna(subset=['short_answer', 'pottery_id', 'session_id'],
                       inplace=True)
    if combined_df.empty:
        print("Sanity Check ERROR: No combined data to analyze.")
        return

    # 2. Calculate Session-Level EVENT COUNT Pct
    print("Calculating session-level Event Count Pct")
    session_counts = pd.crosstab(
        [combined_df['pottery_id'], combined_df['session_id']],
        combined_df['short_answer'])
    session_event_pct = session_counts.div(session_counts.sum(axis=1),
                                           axis=0) * 100
    for col in EMOTION_COLS_STANDARD:
        if col not in session_event_pct.columns:
            session_event_pct[col] = 0.0
    session_event_pct = session_event_pct[EMOTION_COLS_STANDARD]

    # 3. Load Features
    features_df, feature_columns = _load_features(features_csv)
    if features_df is None: return

    # 4. Merge and Analyze
    print("Merging session data with features")
    merged_event_pct = features_df.merge(
        session_event_pct.reset_index(), on='pottery_id', how='inner'
    )
    
    results_data = []
    for feature in feature_columns:
        if feature not in merged_event_pct.columns:
            continue
        event_groups = merged_event_pct.groupby(feature)
        
        for value, group in event_groups:
            label = "Yes" if value == 1.0 else "No"
            event_avgs = group[EMOTION_COLS_STANDARD].mean()
            event_stds = group[EMOTION_COLS_STANDARD].std().fillna(0.0)
            
            results_data.append({
                'Feature': feature,
                'Value': label,
                'Method': 'Event Count Pct',
                'Session_Count': len(group), # Count of sessions
                **{f"Avg_{e}": event_avgs[e] for e in EMOTION_COLS_STANDARD},
                **{f"Std_{e}": event_stds[e] for e in EMOTION_COLS_STANDARD}
            })
    
    # 5. Print Results Table
    results_df = pd.DataFrame(results_data)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 200)
    
    for feature in feature_columns:
        print("\n" + "-"*80)
        print(f"Feature: {feature} (Calculated from SESSIONS)")
        print("-"*80)
        
        print(results_df[results_df['Feature'] == feature].to_string(
            index=False,
            columns=[
                'Value', 'Method', 'Session_Count',
                'Avg_Interesting', 'Std_Interesting',
                'Avg_Beautiful', 'Std_Beautiful',
            ],
            float_format="%.2f"
        ))
    print("\n" + "="*50)
    print(" METHOD A COMPLETE")
    print("="*50)


def run_recalculation_from_pottery_stats(
        event_pct_file: str,
        duration_file: str,
        features_csv: str):
    """
    METHOD B: Loads the per-pottery stats from the Excel files,
    merges with features, and recalculates feature stats.
    This gives EQUAL WEIGHT to every POTTERY.
    """
    
    print("\n" + "="*50)
    print(" RECALCULATION CHECK (METHOD B: From POTTERIES)")
    print("This method gives EQUAL WEIGHT to every POTTERY.")
    print("="*50)

    # 1. Load Data
    print("Loading generated Excel files and features")
    try:
        df_event_pct = pd.read_excel(event_pct_file, index_col=0)
        df_duration = pd.read_excel(duration_file, index_col=0)
    except FileNotFoundError as e:
        print(f"ERROR: Could not find generated Excel file. {e}")
        print("Please run the script to generate files first.")
        return
        
    features_df, feature_columns = _load_features(features_csv)
    if features_df is None: return

    # 2. Merge Features with Pottery Stats
    merged_event = features_df.merge(
        df_event_pct, on='pottery_id', how='inner'
    )
    merged_duration = features_df.merge(
        df_duration, on='pottery_id', how='inner'
    )
    
    # 3. Analyze Event Count Pct
    print("\nAnalyzing Event Count Pct (from pottery stats)")
    event_results = []
    for feature in feature_columns:
        if feature not in merged_event.columns:
            continue
        
        # Group potteries by feature
        groups = merged_event.groupby(feature)
        for value, group in groups:
            label = "Yes" if value == 1.0 else "No"
            
            # Get the mean and std of the *pottery average* columns
            jp_avg_cols = [f"JAPAN_Avg_{e}_Pct" for e in EMOTION_COLS_STANDARD]
            my_avg_cols = [f"MALAYSIA_Avg_{e}_Pct" for e in EMOTION_COLS_STANDARD]
            
            # Calculate the mean OF the averages
            jp_feature_avgs = group[jp_avg_cols].mean() 
            my_feature_avgs = group[my_avg_cols].mean()
            # Calculate the std OF the averages
            jp_feature_stds = group[jp_avg_cols].std().fillna(0.0) 
            my_feature_stds = group[my_avg_cols].std().fillna(0.0)
            
            event_results.append({
                'Feature': feature,
                'Value': label,
                'Method': 'Event Count Pct',
                'Pottery_Count': len(group), # Count of potteries
                **{col: jp_feature_avgs[col] for col in jp_avg_cols},
                **{col.replace("Avg", "Std"): jp_feature_stds[col] for col in jp_avg_cols},
                **{col: my_feature_avgs[col] for col in my_avg_cols},
                **{col.replace("Avg", "Std"): my_feature_stds[col] for col in my_avg_cols}
            })

    # 4. Print Event Count Results
    results_df = pd.DataFrame(event_results)
    for feature in feature_columns:
        print("\n" + "-"*80)
        print(f"Feature: {feature} (Recalculated from POTTERIES)")
        print("-"*80)
        
        print(results_df[results_df['Feature'] == feature].to_string(
            index=False,
            columns=[
                'Value', 'Method', 'Pottery_Count',
                'JAPAN_Avg_Interesting_Pct', 'JAPAN_Std_Interesting_Pct',
                'MALAYSIA_Avg_Interesting_Pct', 'MALAYSIA_Std_Interesting_Pct',
            ],
            float_format="%.2f"
        ))
        
    # 5. Analyze Duration Pct
    print("\nAnalyzing Duration Pct (from pottery stats)")
    duration_results = []
    for feature in feature_columns:
        if feature not in merged_duration.columns:
            continue
            
        groups = merged_duration.groupby(feature)
        for value, group in groups:
            label = "Yes" if value == 1.0 else "No"
            
            # Get the avg and std of the pottery average duration pct columns
            jp_avg_cols = [f"JAPAN_Avg_{e}_DurPct" for e in EMOTION_COLS_STANDARD]
            my_avg_cols = [f"MALAYSIA_Avg_{e}_DurPct" for e in EMOTION_COLS_STANDARD]
            
            jp_feature_avgs = group[jp_avg_cols].mean()
            my_feature_avgs = group[my_avg_cols].mean()
            jp_feature_stds = group[jp_avg_cols].std().fillna(0.0)
            my_feature_stds = group[my_avg_cols].std().fillna(0.0)
            
            duration_results.append({
                'Feature': feature,
                'Value': label,
                'Method': 'Duration Pct',
                'Pottery_Count': len(group), # Count of potteries
                **{col: jp_feature_avgs[col] for col in jp_avg_cols},
                **{col.replace("Avg", "Std"): jp_feature_stds[col] for col in jp_avg_cols},
                **{col: my_feature_avgs[col] for col in my_avg_cols},
                **{col.replace("Avg", "Std"): my_feature_stds[col] for col in my_avg_cols}
            })

    # 6. Print Duration Pct Results
    results_df = pd.DataFrame(duration_results)
    for feature in feature_columns:
        print("\n" + "-"*80)
        print(f"Feature: {feature} (Recalculated from POTTERIES)")
        print("-"*80)
        
        print(results_df[results_df['Feature'] == feature].to_string(
            index=False,
            columns=[
                'Value', 'Method', 'Pottery_Count',
                'JAPAN_Avg_Interesting_DurPct', 'JAPAN_Std_Interesting_DurPct',
                'MALAYSIA_Avg_Interesting_DurPct', 'MALAYSIA_Std_Interesting_DurPct',
            ],
            float_format="%.2f"
        ))

    print("\n" + "="*50)
    print(" METHOD B COMPLETE")
    print("="*50)


# Main Execution
if __name__ == "__main__":
    
    # Define Paths
    DATASET_ROOT_JAPAN = "./src/jomon_kaen_dataset/japan"
    DATASET_ROOT_MALAYSIA = "./src/jomon_kaen_dataset/malaysia"
    POTTERY_MODELS_DIR = "./src/pottery"
    FEATURES_CSV = "./src/analysis/DS_Labels_Cleaned.csv"
    
    EVENT_PCT_FILE = "pottery_event_count_statistics.xlsx"
    DURATION_FILE = "pottery_duration_statistics.xlsx"

    # Run the Main Statistics Generation
    try:
        # This function generates the files AND returns the loaded data
        df_jp, df_my = generate_all_pottery_statistics(
            root_japan=DATASET_ROOT_JAPAN,
            root_malaysia=DATASET_ROOT_MALAYSIA,
            pottery_models_dir=POTTERY_MODELS_DIR,
            output_event_pct_file=EVENT_PCT_FILE,
            output_duration_file=DURATION_FILE
        )
        
        # Run Sanity Check (Method A: From Sessions)
        # This uses the data loaded above, saving time
        # This is equivalent to your `analyze_emotions_by_features` script
        run_sanity_check_from_sessions(
            df_japan=df_jp.copy(), # Use copies to prevent modification
            df_malaysia=df_my.copy(),
            features_csv=FEATURES_CSV
        )
        
        # Run Recalculation (Method B: From Potteries)
        # This loads the Excel files we just created and recalculates
        run_recalculation_from_pottery_stats(
            event_pct_file=EVENT_PCT_FILE,
            duration_file=DURATION_FILE,
            features_csv=FEATURES_CSV
        )
        
    except Exception as e:
        print(f"\nAn error occurred during statistics generation: {e}")
        import traceback
        traceback.print_exc()