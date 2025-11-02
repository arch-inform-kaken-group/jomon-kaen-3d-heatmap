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


# ===============================================
# NEW STATISTICS FUNCTIONS (As per user request)
# ===============================================

def _get_emotion_event_average_by_country(df: pd.DataFrame,
                                          language: str) -> pd.DataFrame:
    """
    Calculates the Sum, Avg, and Std Dev of emotion button-press EVENTS
    per participant.
    """
    if df.empty:
        return pd.DataFrame()

    print(f"Processing Event Count Stats (Sum, Avg, Std) for {language}")
    
    # 1. Get all unique (pottery, session) pairs BEFORE filtering
    # This ensures participants with zero presses are included
    participant_sessions = df[['pottery_id', 'session_id']].drop_duplicates()
    if participant_sessions.empty:
        return pd.DataFrame()
    participant_sessions = participant_sessions.set_index(
        ['pottery_id', 'session_id']
    )
    
    # 2. Map answers to standard labels
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

    # 3. Get total event counts PER SESSION
    session_event_counts = pd.crosstab(
        [df['pottery_id'], df['session_id']],
        df['short_answer']
    )
    
    # 4. Ensure all 5 emotion columns exist
    for col in EMOTION_COLS_STANDARD:
        if col not in session_event_counts.columns:
            session_event_counts[col] = 0
    session_event_counts = session_event_counts[EMOTION_COLS_STANDARD]

    # 5. Re-index to include participants with zero events
    session_event_counts = session_event_counts.reindex(
        participant_sessions.index, fill_value=0
    )

    # 6. Aggregate stats (sum, mean, std, count) per pottery
    pottery_stats = session_event_counts.groupby('pottery_id')[
        EMOTION_COLS_STANDARD
    ].agg(['sum', 'mean', 'std', 'count']).fillna(0.0) # fillna for std on N=1

    # 7. Flatten columns and add to a final DataFrame
    final_stats = pd.DataFrame(index=pottery_stats.index)
    
    # Get Participant Count (same for all columns, so pick one)
    final_stats['Participant_Count'] = pottery_stats[
        (EMOTION_COLS_STANDARD[0], 'count')
    ].astype(int)

    for emotion in EMOTION_COLS_STANDARD:
        final_stats[f'Total_{emotion}_Events'] = pottery_stats[(emotion, 'sum')]
        final_stats[f'Avg_{emotion}_Events'] = pottery_stats[(emotion, 'mean')]
        final_stats[f'Std_{emotion}_Events'] = pottery_stats[(emotion, 'std')]
    
    return final_stats


def _get_emotion_duration_average_by_country(df_in: pd.DataFrame,
                                             language: str) -> pd.DataFrame:
    """
    Calculates the Sum, Avg, and Std Dev of gaze DURATION (in seconds)
    per participant. Includes "No Reaction" time calculated from a 60s max.
    """
    # Define all columns to be processed, including the new one
    EMOTION_COLS_WITH_NO_REACTION = EMOTION_COLS_STANDARD + ["No Reaction"]
    MAX_DURATION = 60.0 # Maximum time per participant
    
    if df_in.empty:
        return pd.DataFrame()

    print(f"Processing Gaze Duration Stats (Sum, Avg, Std) for {language}")
    
    # 1. Get all unique (pottery, session) pairs BEFORE filtering
    participant_sessions = df_in[['pottery_id', 'session_id']].drop_duplicates()
    if participant_sessions.empty:
        return pd.DataFrame()
    participant_sessions = participant_sessions.set_index(
        ['pottery_id', 'session_id']
    )

    df = df_in.copy() # Avoid modifying the original dataframe

    # 2. Map answers to standard labels
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
        print(f"No valid emotion data found for {language} gaze.")
        # Create an empty dataframe with the standard columns
        session_emotion_durations = pd.DataFrame(
            0.0, 
            index=pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=['pottery_id', 'session_id']), 
            columns=EMOTION_COLS_STANDARD
        )
    else:
        # 3. Calculate duration blocks
        df.sort_values(by=['pottery_id', 'session_id', 'timestamp'], inplace=True)
        df['time_diff'] = df.groupby(['pottery_id', 'session_id'])['timestamp'].diff()
        emotion_changed = df['short_answer'] != df.groupby(
            ['pottery_id', 'session_id'])['short_answer'].shift()
        time_gap_exceeded = df['time_diff'] > 0.05 
        df['block_id'] = (emotion_changed | time_gap_exceeded).cumsum()

        # 4. Aggregate block durations
        block_durations = df.groupby(
            ['pottery_id', 'session_id', 'block_id']
        ).agg(
            start_time=('timestamp', 'min'),
            end_time=('timestamp', 'max'),
            answer=('short_answer', 'first')
        ).reset_index()
        block_durations['duration'] = block_durations['end_time'] - block_durations['start_time']
        
        # 5. Sum total durations per emotion PER SESSION
        session_emotion_durations = block_durations.groupby(
            ['pottery_id', 'session_id', 'answer']
        )['duration'].sum().unstack(fill_value=0.0)

        # 6. Ensure all 5 *standard* emotion columns exist
        for col in EMOTION_COLS_STANDARD:
            if col not in session_emotion_durations.columns:
                session_emotion_durations[col] = 0.0
        session_emotion_durations = session_emotion_durations[EMOTION_COLS_STANDARD]

    # 7. Re-index to include participants with zero duration
    # (e.g., those who had no qa file for this pottery)
    session_emotion_durations = session_emotion_durations.reindex(
        participant_sessions.index, fill_value=0.0
    )
    
    # 8. Calculate "No Reaction" duration per session
    # Sum of all *recorded* reaction durations
    total_reacted_duration = session_emotion_durations[EMOTION_COLS_STANDARD].sum(axis=1)
    # No Reaction = 60s (max) - total reacted time. Clip at 0 just in case.
    session_emotion_durations['No Reaction'] = (MAX_DURATION - total_reacted_duration).clip(lower=0)


    # 9. Aggregate stats (sum, mean, std, count) per pottery
    pottery_stats = session_emotion_durations.groupby('pottery_id')[
        EMOTION_COLS_WITH_NO_REACTION # Use the new list
    ].agg(['sum', 'mean', 'std', 'count']).fillna(0.0) # fillna for std on N=1

    # 10. Flatten columns and add to a final DataFrame
    final_stats = pd.DataFrame(index=pottery_stats.index)
    
    # Get Participant Count
    final_stats['Participant_Count'] = pottery_stats[
        (EMOTION_COLS_STANDARD[0], 'count')
    ].astype(int)

    for emotion in EMOTION_COLS_WITH_NO_REACTION: # Use the new list
        final_stats[f'Total_{emotion}_Duration'] = pottery_stats[(emotion, 'sum')]
        final_stats[f'Avg_{emotion}_Duration'] = pottery_stats[(emotion, 'mean')]
        final_stats[f'Std_{emotion}_Duration'] = pottery_stats[(emotion, 'std')]

    return final_stats


def generate_all_pottery_statistics(
        root_japan: str,
        root_malaysia: str,
        pottery_models_dir: str,
        output_event_jp_file: str = "japan_event_averages.xlsx",
        output_event_my_file: str = "malaysia_event_averages.xlsx",
        output_duration_jp_file: str = "japan_duration_averages.xlsx",
        output_duration_my_file: str = "malaysia_duration_averages.xlsx"):
    """
    Generates four separate Excel files for event and duration stats,
    split by country. Includes Sum, Avg, Std Dev, and Count.
    """

    print(" Starting Combined Statistics Generation")

    # 1. Load Raw Data
    print("\nLoading Japan QnA data")
    df_japan = load_combined_qna_data(root_japan, pottery_models_dir)
    print("\nLoading Malaysia QnA data")
    df_malaysia = load_combined_qna_data(root_malaysia, pottery_models_dir)

    # 2. Process Event Count Stats
    print("\nProcessing Event Stats")
    stats_event_jp = _get_emotion_event_average_by_country(df_japan, 'japan')
    stats_event_my = _get_emotion_event_average_by_country(df_malaysia, 'malaysia')

    # 3. Process Gaze Duration Stats
    print("\nProcessing Duration Stats")
    stats_duration_jp = _get_emotion_duration_average_by_country(df_japan, "japan")
    stats_duration_my = _get_emotion_duration_average_by_country(df_malaysia, "malaysia")

    # 4. Get Master Pottery List
    all_pottery_ids = [
        f"{pid}({num})" for pid, num in ASSIGNED_NUMBERS_DICT.items()
    ]
    base_df = pd.DataFrame(index=all_pottery_ids)
    base_df.index.name = "pottery_id"

    # Helper for sorting columns
    def sort_cols(x):
        if x == 'Participant_Count': return (0, x)
        if 'Total_' in x: return (1, x)
        if 'Avg_' in x: return (2, x)
        if 'Std_' in x: return (3, x)
        return (4, x)

    # 5. Create and Save FILE 1: Japan Event Averages
    print(f"\nCombining Japan Event Averages stats")
    df_event_jp = base_df.join(stats_event_jp, how='left').fillna(0.0)
    # Re-order columns: Count, then Totals, then Avgs, then Stds
    cols_event_jp = sorted(df_event_jp.columns, key=sort_cols)
    df_event_jp = df_event_jp[cols_event_jp]
    df_event_jp.to_excel(output_event_jp_file)
    print("\n" + "="*50)
    print("SUCCESS: Japan Event Stats file generated!")
    print(f"File saved as: {output_event_jp_file}")
    print("="*50)

    # 6. Create and Save FILE 2: Malaysia Event Averages
    print(f"\nCombining Malaysia Event Averages stats")
    df_event_my = base_df.join(stats_event_my, how='left').fillna(0.0)
    cols_event_my = sorted(df_event_my.columns, key=sort_cols)
    df_event_my = df_event_my[cols_event_my]
    df_event_my.to_excel(output_event_my_file)
    print("\n" + "="*50)
    print("SUCCESS: Malaysia Event Stats file generated!")
    print(f"File saved as: {output_event_my_file}")
    print("="*50)

    # 7. Create and Save FILE 3: Japan Duration Averages
    print(f"\nCombining Japan Duration Averages stats")
    df_duration_jp = base_df.join(stats_duration_jp, how='left').fillna(0.0)
    cols_dur_jp = sorted(df_duration_jp.columns, key=sort_cols)
    df_duration_jp = df_duration_jp[cols_dur_jp]
    df_duration_jp.to_excel(output_duration_jp_file)
    print("\n" + "="*50)
    print("SUCCESS: Japan Duration Stats file generated!")
    print(f"File saved as: {output_duration_jp_file}")
    print("="*50)

    # 8. Create and Save FILE 4: Malaysia Duration Averages
    print(f"\nCombining Malaysia Duration Averages stats")
    df_duration_my = base_df.join(stats_duration_my, how='left').fillna(0.0)
    cols_dur_my = sorted(df_duration_my.columns, key=sort_cols)
    df_duration_my = df_duration_my[cols_dur_my]
    df_duration_my.to_excel(output_duration_my_file)
    print("\n" + "="*50)
    print("SUCCESS: Malaysia Duration Stats file generated!")
    print(f"File saved as: {output_duration_my_file}")
    print("="*50)
    
    print("\nAll statistics files generated successfully.")


# Main Execution
if __name__ == "__main__":
    
    # Define Paths
    DATASET_ROOT_JAPAN = "./src/jomon_kaen_dataset/japan"
    DATASET_ROOT_MALAYSIA = "./src/jomon_kaen_dataset/malaysia"
    POTTERY_MODELS_DIR = "./src/pottery"
    
    # Define new output filenames
    EVENT_JP_FILE = "japan_event_statistics.xlsx"
    EVENT_MY_FILE = "malaysia_event_statistics.xlsx"
    DURATION_JP_FILE = "japan_duration_statistics.xlsx"
    DURATION_MY_FILE = "malaysia_duration_statistics.xlsx"

    # Run the Main Statistics Generation
    try:
        # This function generates the 4 files
        generate_all_pottery_statistics(
            root_japan=DATASET_ROOT_JAPAN,
            root_malaysia=DATASET_ROOT_MALAYSIA,
            pottery_models_dir=POTTERY_MODELS_DIR,
            output_event_jp_file=EVENT_JP_FILE,
            output_event_my_file=EVENT_MY_FILE,
            output_duration_jp_file=DURATION_JP_FILE,
            output_duration_my_file=DURATION_MY_FILE
        )
        
    except Exception as e:
        print(f"\nAn error occurred during statistics generation: {e}")
        import traceback
        traceback.print_exc()