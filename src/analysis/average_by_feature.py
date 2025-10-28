# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import japanize_matplotlib
# import os
# from pathlib import Path
# from tqdm import tqdm

# # Dictionaries and Constants
# ASSIGNED_NUMBERS_DICT = {
#     'AS0001': '1',
#     'FH0008': '2',
#     'IN0003': '3',
#     'IN0008': '4',
#     'IN0009': '5',
#     'IN0017': '6',
#     'IN0081': '7',
#     'IN0104': '8',
#     'IN0135': '9',
#     'IN0148': '10',
#     'IN0220': '11',
#     'IN0228': '12',
#     'IN0232': '13',
#     'IN0239': '14',
#     'IN0277': '15',
#     'MY0001': '16',
#     'MY0002': '17',
#     'MY0004': '18',
#     'MY0006': '19',
#     'MY0007': '20',
#     'ND0001': '21',
#     'NM0001': '22',
#     'NM0002': '23',
#     'NM0009': '24',
#     'NM0010': '25',
#     'NM0014': '26',
#     'NM0015': '27',
#     'NM0017': '28',
#     'NM0041': '29',
#     'NM0049': '30',
#     'NM0066': '31',
#     'NM0070': '32',
#     'NM0072': '33',
#     'NM0073': '34',
#     'NM0079': '35',
#     'NM0080': '36',
#     'NM0099': '37',
#     'NM0106': '38',
#     'NM0133': '39',
#     'NM0135': '40',
#     'NM0144': '41',
#     'NM0154': '42',
#     'NM0156': '43',
#     'NM0159': '44',
#     'NM0168': '45',
#     'NM0173': '46',
#     'NM0175': '47',
#     'NM0189': '48',
#     'NM0191': '49',
#     'NM0206': '50',
#     'SB0002': '51',
#     'SB0004': '52',
#     'SI0001': '53',
#     'SJ0503': '54',
#     'SJ0504': '55',
#     'SK0001': '56',
#     'SK0002': '57',
#     'SK0003': '58',
#     'SK0004': '59',
#     'SK0005': '60',
#     'SK0013': '61',
#     'SS0001': '62',
#     'TJ0004': '63',
#     'TJ0005': '64',
#     'TJ0010': '65',
#     'TK0002': '66',
#     'TK0048': '67',
#     'TK0057': '68',
#     'UD0001': '69',
#     'UD0003': '70',
#     'UD0005': '71',
#     'UD0006': '72',
#     'UD0011': '73',
#     'UD0013': '74',
#     'UD0014': '75',
#     'UD0016': '76',
#     'UD0023': '77',
#     'UD0302': '78',
#     'UD0304': '79',
#     'UD0308': '80',
#     'UD0318': '81',
#     'UD0322': '82',
#     'UD0411': '83',
#     'UD0412': '84',
#     'UK0001': '85',
#     'IN0295': '86',
#     'IN0306': '87',
#     'MH0037': '88',
#     'NM0239': '89',
#     'NZ0001': '90',
#     'SK0035': '91',
#     'TK0020': '92',
#     'UD0028': '93',
# }

# # English/Malaysian Emotion Map
# EMOTION_COLOR_MAP_EN = {
#     "Interesting and attentional shape": "#00FFFF",
#     "Beautiful and artistic": "#00FF00",
#     "Strange and incomprehensible": "#FFFF00",
#     "Creepy / unsettling / scary": "#FF0000",
#     "Feel nothing": "#505050",
#     "NO RESPONSE": "#D3D3D3",
# }

# SHORT_LABELS_EN = {
#     "Interesting and attentional shape": "Interesting",
#     "Beautiful and artistic": "Beautiful",
#     "Strange and incomprehensible": "Strange",
#     "Creepy / unsettling / scary": "Scary",
#     "Feel nothing": "Feel nothing",
#     "NO RESPONSE": "NO RESPONSE"
# }

# EMOTION_STACK_ORDER_EN = [
#     "Interesting and attentional shape", "Beautiful and artistic",
#     "Strange and incomprehensible", "Creepy / unsettling / scary",
#     "Feel nothing", "NO RESPONSE"
# ]

# # Japanese Emotion Map
# EMOTION_COLOR_MAP_JP = {
#     "面白い・気になる形だ": "#00FFFF",
#     "美しい・芸術的だ": "#00FF00",
#     "不思議・意味不明": "#FFFF00",
#     "不気味・不安・怖い": "#FF0000",
#     "何も感じない": "#505050",
#     "NO RESPONSE": "#D3D3D3",  # Assuming "NO RESPONSE" stays constant
# }

# # Mapping long labels to short labels for plots and CSVs
# SHORT_LABELS_JP = {
#     "面白い・気になる形だ": "面白い",
#     "美しい・芸術的だ": "美しい",
#     "不思議・意味不明": "不思議",
#     "不気味・不安・怖い": "怖い",
#     "何も感じない": "何も感じない",
#     "NO RESPONSE": "NO RESPONSE"
# }


# # Data Loading Functions
# def find_data_paths_detailed(root: str,
#                              pottery_path_str: str,
#                              limit: int = 1000) -> list:
#     root, pottery_path = Path(root), Path(pottery_path_str)
#     if not root.exists():
#         raise ValueError(f"Root directory not found: {root}")
#     if not pottery_path.exists():
#         raise ValueError(f"Pottery directory not found: {pottery_path}")

#     data = []
#     pottery_ids = [
#         f"{pid}({num})" for pid, num in ASSIGNED_NUMBERS_DICT.items()
#     ]

#     print(f"\nCHECKING RAW DATA PATHS")
#     limit_dict = {pid: 0 for pid in pottery_ids}

#     for g in os.listdir(root):
#         group_path = root / g
#         if not os.path.isdir(group_path):
#             continue
#         for s in tqdm(os.listdir(group_path), desc=g):
#             session_path = group_path / s
#             if not os.path.isdir(session_path):
#                 continue
#             for p in os.listdir(session_path):
#                 if p in pottery_ids and limit_dict[p] < limit:
#                     qa_path = session_path / p / "qa_corrected.csv"
#                     if qa_path.exists():
#                         limit_dict[p] += 1
#                         data.append({
#                             'qa': str(qa_path),
#                             'GROUP': g,
#                             'SESSION_ID': s,
#                             'ID': p
#                         })

#     print(f"\nLoader finished. Found {len(data)} valid data instances.")
#     return data


# def load_combined_qna_data(root_dir: str,
#                            pottery_models_dir: str) -> pd.DataFrame:
#     data_to_process = find_data_paths_detailed(
#         root=root_dir, pottery_path_str=pottery_models_dir)
#     if not data_to_process:
#         return pd.DataFrame()

#     df_list = []
#     for item in tqdm(data_to_process, desc="Loading and combining data"):
#         try:
#             temp_df = pd.read_csv(item['qa'], header=0, sep=",")
#             temp_df['timestamp'] = pd.to_numeric(temp_df['timestamp'],
#                                                  errors='coerce')
#             temp_df.dropna(subset=['timestamp'], inplace=True)
#             temp_df['pottery_id'] = item['ID']
#             temp_df['session_id'] = item['SESSION_ID']
#             df_list.append(temp_df)
#         except Exception as e:
#             print(f"Could not read or process file {item['qa']}: {e}")

#     if not df_list:
#         return pd.DataFrame()

#     print("\nCombining all data sources for analysis...")
#     return pd.concat(df_list, ignore_index=True)


# def analyze_emotions_by_features(combined_df: pd.DataFrame,
#                                  features_csv: str,
#                                  language: str = 'malaysia',
#                                  selected_features: list = None,
#                                  include_shape_analysis: bool = True):
#     """Analyzes and plots emotion responses categorized by pottery features.
    
#     Args:
#         combined_df: DataFrame with emotion response data
#         features_csv: Path to CSV file with pottery features
#         language: Language setting for labels
#         selected_features: List of specific features to analyze. If None, analyzes all features.
#         include_shape_analysis: Whether to include shape type analysis
#     """

#     if language == 'japan':
#         EMOTION_COLOR_MAP = EMOTION_COLOR_MAP_JP
#         EMOTION_STACK_ORDER = [
#             "何も感じない", "不気味・不安・怖い", "不思議・意味不明", "美しい・芸術的だ", "面白い・気になる形だ",
#             "NO RESPONSE"
#         ]
#         EMOTION_SYMBOL_MAP = {
#             "面白い・気になる形だ": "◇",
#             "美しい・芸術的だ": "□",
#             "不思議・意味不明": "△",
#             "不気味・不安・怖い": "X",
#             "何も感じない": "○",
#             "NO RESPONSE": "・"
#         }
#         EMOTION_SHORT_LABEL_MAP = SHORT_LABELS_JP
#     else:
#         EMOTION_COLOR_MAP = EMOTION_COLOR_MAP_EN
#         EMOTION_STACK_ORDER = [
#             "Feel nothing", "Creepy / unsettling / scary",
#             "Strange and incomprehensible", "Beautiful and artistic",
#             "Interesting and attentional shape", "NO RESPONSE"
#         ]
#         EMOTION_SYMBOL_MAP = {
#             "Interesting and attentional shape": "◇",
#             "Beautiful and artistic": "□",
#             "Strange and incomprehensible": "△",
#             "Creepy / unsettling / scary": "X",
#             "Feel nothing": "○",
#             "NO RESPONSE": "・"
#         }
#         EMOTION_SHORT_LABEL_MAP = SHORT_LABELS_EN

#     if combined_df.empty:
#         print("Combined DataFrame is empty. No analysis performed.")
#         return

#     # Load features CSV
#     features_df = pd.read_csv(features_csv)

#     # Extract pottery code from the CODE column (e.g., "AS0001(1).ply" -> "AS0001(1)")
#     features_df['pottery_id'] = features_df['CODE'].str.replace('.ply', '')

#     # Add short_answer column
#     combined_df['answer'] = combined_df['answer'].str.strip()
#     combined_df['short_answer'] = combined_df['answer'].map(
#         EMOTION_SHORT_LABEL_MAP)

#     # Calculate percentage by event count (session-normalized)
#     session_counts_df = pd.crosstab(
#         [combined_df['pottery_id'], combined_df['session_id']],
#         combined_df['short_answer'])
#     session_percentage_df = session_counts_df.div(
#         session_counts_df.sum(axis=1), axis=0) * 100
#     percentage_df = session_percentage_df.groupby('pottery_id').mean()

#     # Merge with features
#     merged_df = features_df.merge(percentage_df, on='pottery_id', how='inner')

#     # Create output directory
#     output_dir = "feature_emotion_analysis"
#     os.makedirs(output_dir, exist_ok=True)

#     # Get feature columns (exclude CODE and pottery_id)
#     all_feature_columns = [
#         col for col in features_df.columns if
#         col not in ['CODE', 'pottery_id'] and not col.startswith('SHAPE_TYPE_')
#     ]

#     # Filter to selected features if specified
#     if selected_features is not None and len(selected_features) > 0:
#         feature_columns = [
#             f for f in selected_features if f in all_feature_columns
#         ]
#         if len(feature_columns) == 0:
#             print(
#                 f"Warning: None of the selected features found in CSV. Available features:"
#             )
#             print(", ".join(all_feature_columns))
#             return
#         print(f"\nAnalyzing {len(feature_columns)} selected features...")
#         print(f"Features: {', '.join(feature_columns)}")
#     else:
#         feature_columns = all_feature_columns
#         print(f"\nAnalyzing all {len(feature_columns)} binary features...")

#     # Add SHAPE_TYPE as a special aggregated feature
#     shape_type_columns = [
#         col for col in features_df.columns if col.startswith('SHAPE_TYPE_')
#     ]

#     # Map colors to short labels
#     short_label_color_map = {
#         EMOTION_SHORT_LABEL_MAP[k]: v
#         for k, v in EMOTION_COLOR_MAP.items()
#     }
#     emotion_order = [
#         EMOTION_SHORT_LABEL_MAP[e] for e in EMOTION_STACK_ORDER
#         if e in EMOTION_COLOR_MAP.keys() and e != "NO RESPONSE"
#     ]
#     plot_colors = [
#         short_label_color_map.get(e, '#CCCCCC') for e in emotion_order
#     ]

#     # Analyze binary features
#     for feature in tqdm(feature_columns, desc="Processing binary features"):
#         fig, ax = plt.subplots(figsize=(10, 6))

#         # Split data by feature value (0 or 1)
#         feature_groups = merged_df.groupby(feature)

#         results = []
#         for value, group in feature_groups:
#             if len(group) == 0:
#                 continue

#             # Calculate average emotion percentages for this group
#             emotion_avgs = group[emotion_order].mean()
#             results.append({
#                 'value': 'Yes' if value == 1.0 else 'No',
#                 **emotion_avgs.to_dict()
#             })

#         if not results:
#             plt.close(fig)
#             continue

#         results_df = pd.DataFrame(results).set_index('value')

#         # Plot stacked bar chart
#         results_df[emotion_order].plot(kind='bar',
#                                        stacked=True,
#                                        ax=ax,
#                                        color=plot_colors,
#                                        width=0.6)

#         # Add value labels
#         for container in ax.containers:
#             labels = [
#                 f'{v:.1f}' if v > 2 else '' for v in container.datavalues
#             ]
#             ax.bar_label(container,
#                          labels=labels,
#                          label_type='center',
#                          fontsize=12,
#                          color='black',
#                          weight='bold')

#         ax.set_title(f'Emotion Response by Feature: {feature}',
#                      fontsize=14,
#                      pad=20)
#         ax.set_ylabel('Average Percentage (%)', fontsize=12)
#         ax.set_xlabel('Feature Present', fontsize=12)
#         ax.set_ylim(0, 100)
#         ax.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
#         plt.xticks(rotation=0)
#         plt.grid(axis='y', linestyle='--', alpha=0.3)
#         plt.tight_layout()

#         # Save plot
#         filename = f"{feature.lower().replace(' ', '_')}_emotion_analysis.png"
#         plt.savefig(os.path.join(output_dir, filename),
#                     dpi=150,
#                     bbox_inches='tight')
#         plt.close(fig)

#     # Analyze SHAPE_TYPE
#     if include_shape_analysis:
#         print("\nAnalyzing shape types...")

#         # Create a single column for shape type
#         def get_shape_type(row):
#             for col in shape_type_columns:
#                 if row[col] == 1:
#                     return col.replace('SHAPE_TYPE_', '')
#             return 'Unknown'

#         merged_df['shape_type'] = merged_df.apply(get_shape_type, axis=1)

#         # Group by shape type
#         shape_groups = merged_df.groupby('shape_type')

#         shape_results = []
#         for shape, group in shape_groups:
#             if len(group) < 2:  # Skip if too few samples
#                 continue
#             emotion_avgs = group[emotion_order].mean()
#             shape_results.append({
#                 'shape_type': shape,
#                 'count': len(group),
#                 **emotion_avgs.to_dict()
#             })

#         if shape_results:
#             shape_df = pd.DataFrame(shape_results).set_index('shape_type')

#             fig, ax = plt.subplots(figsize=(14, 8))
#             shape_df[emotion_order].plot(kind='bar',
#                                          stacked=True,
#                                          ax=ax,
#                                          color=plot_colors,
#                                          width=0.7)

#             for container in ax.containers:
#                 labels = [
#                     f'{v:.1f}' if v > 2 else '' for v in container.datavalues
#                 ]
#                 ax.bar_label(container,
#                              labels=labels,
#                              label_type='center',
#                              fontsize=12,
#                              color='black',
#                              weight='bold')

#             ax.set_title('Emotion Response by Pottery Shape Type',
#                          fontsize=14,
#                          pad=20)
#             ax.set_ylabel('Average Percentage (%)', fontsize=12)
#             ax.set_xlabel('Shape Type', fontsize=12)
#             ax.set_ylim(0, 100)
#             ax.legend(title='Emotion',
#                       bbox_to_anchor=(1.05, 1),
#                       loc='upper left')
#             plt.xticks(rotation=45, ha='right')
#             plt.grid(axis='y', linestyle='--', alpha=0.3)
#             plt.tight_layout()

#             plt.savefig(os.path.join(output_dir,
#                                      'shape_type_emotion_analysis.png'),
#                         dpi=150,
#                         bbox_inches='tight')
#             plt.close(fig)

#             # Save summary CSV
#             shape_df.to_csv(os.path.join(output_dir, 'shape_type_summary.csv'))

#     print(f"\nAnalysis complete! Results saved to '{output_dir}/' directory")
#     feature_count = len(feature_columns)
#     shape_count = 1 if include_shape_analysis else 0
#     print(
#         f"Generated {feature_count} binary feature plot(s) + {shape_count} shape type plot(s)"
#     )


# def compute_cosine_similarities_with_interesting(
#     combined_df_japan: pd.DataFrame,
#     combined_df_malaysia: pd.DataFrame,
#     features_csv: str,
#     output_csv: str = "cosine_similarities_with_interesting.csv"
# ):
#     """
#     Compute cosine similarities between Japan and Malaysia emotion profiles 
#     INCLUDING the 'Interesting' emotion, for each binary pottery feature.
    
#     Args:
#         combined_df_japan: DataFrame with Japanese responses
#         combined_df_malaysia: DataFrame with Malaysian responses
#         features_csv: Path to features CSV
#         output_csv: Output file to save results
#     """
#     import numpy as np
#     from sklearn.metrics.pairwise import cosine_similarity

#     # Load features
#     features_df = pd.read_csv(features_csv)
#     features_df['pottery_id'] = features_df['CODE'].str.replace('.ply', '', regex=False)

#     # Define emotion columns (FULL 5 emotions, excluding NO RESPONSE)
#     emotion_cols_en = [
#         "Interesting and attentional shape",
#         "Beautiful and artistic",
#         "Strange and incomprehensible",
#         "Creepy / unsettling / scary",
#         "Feel nothing"
#     ]
#     short_to_long_en = {v: k for k, v in SHORT_LABELS_EN.items()}
#     emotion_cols_short = ["Interesting", "Beautiful", "Strange", "Scary", "Feel nothing"]
#     emotion_cols_full = [short_to_long_en[s] for s in emotion_cols_short]

#     # Map Japanese answers to English-equivalent short labels
#     def map_jp_to_short(answer):
#         mapping = {
#             "面白い・気になる形だ": "Interesting",
#             "美しい・芸術的だ": "Beautiful",
#             "不思議・意味不明": "Strange",
#             "不気味・不安・怖い": "Scary",
#             "何も感じない": "Feel nothing"
#         }
#         return mapping.get(answer.strip(), None)

#     # Process Japan
#     combined_df_japan['short_answer'] = combined_df_japan['answer'].apply(map_jp_to_short)
#     combined_df_japan = combined_df_japan.dropna(subset=['short_answer'])

#     # Process Malaysia (already uses English)
#     combined_df_malaysia['short_answer'] = combined_df_malaysia['answer'].map(SHORT_LABELS_EN)
#     combined_df_malaysia = combined_df_malaysia.dropna(subset=['short_answer'])

#     # Compute mean emotion vectors per pottery (session-normalized)
#     def get_mean_emotions_per_pottery(df):
#         session_counts = pd.crosstab(
#             [df['pottery_id'], df['session_id']],
#             df['short_answer']
#         )
#         session_pct = session_counts.div(session_counts.sum(axis=1), axis=0) * 100
#         pottery_means = session_pct.groupby('pottery_id').mean()
#         # Ensure all 5 emotions are present (fill missing with 0)
#         for col in emotion_cols_short:
#             if col not in pottery_means.columns:
#                 pottery_means[col] = 0.0
#         return pottery_means[emotion_cols_short].fillna(0.0)

#     japan_means = get_mean_emotions_per_pottery(combined_df_japan)
#     malaysia_means = get_mean_emotions_per_pottery(combined_df_malaysia)

#     # Merge with features
#     japan_merged = features_df[['pottery_id'] + [c for c in features_df.columns if c.startswith('HAS_') or c.startswith('NO_')]].merge(
#         japan_means, on='pottery_id', how='inner'
#     )
#     malaysia_merged = features_df[['pottery_id'] + [c for c in features_df.columns if c.startswith('HAS_') or c.startswith('NO_')]].merge(
#         malaysia_means, on='pottery_id', how='inner'
#     )

#     # Identify binary feature columns
#     feature_cols = [col for col in features_df.columns if col.startswith('HAS_')]

#     results = []
#     for feature in feature_cols:
#         # Japan: group by feature = 1 and = 0
#         j_has = japan_merged[japan_merged[feature] == 1][emotion_cols_short]
#         j_no = japan_merged[japan_merged[feature] == 0][emotion_cols_short]
        
#         m_has = malaysia_merged[malaysia_merged[feature] == 1][emotion_cols_short]
#         m_no = malaysia_merged[malaysia_merged[feature] == 0][emotion_cols_short]

#         # Compute mean vectors
#         j_has_vec = j_has.mean().values if len(j_has) > 0 else np.zeros(5)
#         j_no_vec = j_no.mean().values if len(j_no) > 0 else np.zeros(5)
#         m_has_vec = m_has.mean().values if len(m_has) > 0 else np.zeros(5)
#         m_no_vec = m_no.mean().values if len(m_no) > 0 else np.zeros(5)

#         # Cosine similarity (handle zero vectors)
#         def safe_cosine(a, b):
#             if np.all(a == 0) or np.all(b == 0):
#                 return np.nan
#             return cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0]

#         sim_has = safe_cosine(j_has_vec, m_has_vec)
#         sim_no = safe_cosine(j_no_vec, m_no_vec)

#         results.append({
#             'Feature': feature,
#             'Japan_Has_Vector': j_has_vec.round(2),
#             'Malaysia_Has_Vector': m_has_vec.round(2),
#             'Cosine_Sim_Has': round(sim_has, 4) if not np.isnan(sim_has) else np.nan,
#             'Japan_No_Vector': j_no_vec.round(2),
#             'Malaysia_No_Vector': m_no_vec.round(2),
#             'Cosine_Sim_No': round(sim_no, 4) if not np.isnan(sim_no) else np.nan,
#         })

#     results_df = pd.DataFrame(results)
    
#     # Save full results
#     results_df.to_csv(output_csv, index=False)
#     print(f"\nCosine similarities (with 'Interesting') saved to: {output_csv}")
    
#     # Print clean summary table
#     print("\n" + "="*80)
#     print("COSINE SIMILARITY (INCLUDING 'INTERESTING') BY FEATURE")
#     print("="*80)
#     print(f"{'Feature':<25} {'Has Feature':<12} {'No Feature':<12}")
#     print("-"*80)
#     for _, row in results_df.iterrows():
#         feat = row['Feature'].replace('HAS_', '').replace('_', ' ').title()
#         has_sim = f"{row['Cosine_Sim_Has']:.4f}" if pd.notna(row['Cosine_Sim_Has']) else "N/A"
#         no_sim = f"{row['Cosine_Sim_No']:.4f}" if pd.notna(row['Cosine_Sim_No']) else "N/A"
#         print(f"{feat:<25} {has_sim:<12} {no_sim:<12}")
#     print("="*80)

#     return results_df


# # Main Execution
# if __name__ == "__main__":
#     # === USER CONTROLS ===
#     SELECTED_LANGUAGE = 'malaysia'
#     # SELECTED_LANGUAGE = 'japan'
#     POTTERY_SELECTION = []  # Empty list for all pottery
#     INCLUDE_POTTERY = True

#     DATASET_ROOT_DIR = "./src/jomon_kaen_dataset/malaysia"
#     # DATASET_ROOT_DIR = "./src/jomon_kaen_dataset/japan"
#     POTTERY_MODELS_DIR = "./src/pottery"
#     FEATURES_CSV = "./src/analysis/DS_Labels_Cleaned.csv"

#     # SELECT FEATURES TO ANALYZE
#     # Option 1: Analyze ALL features (set to None or empty list)
#     SELECTED_FEATURES = None

#     # Option 2: Analyze specific features (uncomment and modify as needed)
#     # SELECTED_FEATURES = [
#     #     'HAS_FLAME_LIKE_DECORATION',
#     #     'HAS_CROWN_LIKE_DECORATION',
#     #     'HAS_HANDLES',
#     #     'HAS_SPIRAL_PATTERN'
#     # ]

#     # Include shape type analysis?
#     INCLUDE_SHAPE_TYPE_ANALYSIS = True
#     # === END USER CONTROLS ===

#     try:
#         # Load emotion data
#         combined_dataframe = load_combined_qna_data(DATASET_ROOT_DIR,
#                                                     POTTERY_MODELS_DIR)

#         # Filter pottery if selection is specified
#         if not combined_dataframe.empty and POTTERY_SELECTION:
#             print(
#                 f"\nFiltering pottery based on selection (Include mode: {INCLUDE_POTTERY})..."
#             )
#             base_ids = combined_dataframe['pottery_id'].str.split(
#                 '(', expand=True)[0]
#             initial_count = len(combined_dataframe['pottery_id'].unique())

#             if INCLUDE_POTTERY:
#                 combined_dataframe = combined_dataframe[base_ids.isin(
#                     POTTERY_SELECTION)]
#                 print(
#                     f"Included {len(combined_dataframe['pottery_id'].unique())} of {initial_count} unique pottery items."
#                 )
#             else:
#                 combined_dataframe = combined_dataframe[
#                     ~base_ids.isin(POTTERY_SELECTION)]
#                 print(
#                     f"Excluded IDs, {len(combined_dataframe['pottery_id'].unique())} of {initial_count} unique items remaining."
#                 )

#         # Run feature-based analysis
#         if not combined_dataframe.empty:
#             analyze_emotions_by_features(
#                 combined_dataframe,
#                 FEATURES_CSV,
#                 language=SELECTED_LANGUAGE,
#                 selected_features=SELECTED_FEATURES,
#                 include_shape_analysis=INCLUDE_SHAPE_TYPE_ANALYSIS)
#         else:
#             print("No data available for analysis.")

#     except (FileNotFoundError, ValueError) as e:
#         print(f"Could not run analysis due to an error: {e}")
#         print(
#             "Please ensure all directories and the features CSV are set up correctly."
#         )

#     try:
#         # Load both datasets
#         DATASET_ROOT_JAPAN = "./src/jomon_kaen_dataset/japan"
#         DATASET_ROOT_MALAYSIA = "./src/jomon_kaen_dataset/malaysia"
        
#         print("\nLoading Japan dataset for cross-cultural comparison...")
#         df_japan = load_combined_qna_data(DATASET_ROOT_JAPAN, POTTERY_MODELS_DIR)
#         print("Loading Malaysia dataset...")
#         df_malaysia = load_combined_qna_data(DATASET_ROOT_MALAYSIA, POTTERY_MODELS_DIR)

#         if not df_japan.empty and not df_malaysia.empty:
#             print("\nComputing cosine similarities INCLUDING 'Interesting' emotion...")
#             sim_results = compute_cosine_similarities_with_interesting(
#                 df_japan, df_malaysia, FEATURES_CSV
#             )
#         else:
#             print("Skipping cosine similarity (with Interesting) due to missing data.")
#     except Exception as e:
#         print(f"Error during extended cosine similarity computation: {e}")

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


def analyze_emotions_by_features(combined_df: pd.DataFrame,
                                 features_csv: str,
                                 language: str = 'malaysia',
                                 selected_features: list = None,
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
    combined_df['short_answer'] = combined_df['answer'].map(
        EMOTION_SHORT_LABEL_MAP)

    # Calculate percentage by event count (session-normalized)
    session_counts_df = pd.crosstab(
        [combined_df['pottery_id'], combined_df['session_id']],
        combined_df['short_answer'])
    session_percentage_df = session_counts_df.div(
        session_counts_df.sum(axis=1), axis=0) * 100
    percentage_df = session_percentage_df.groupby('pottery_id').mean()

    # Merge with features
    merged_df = features_df.merge(percentage_df, on='pottery_id', how='inner')

    # Create output directory
    output_dir = "feature_emotion_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Get feature columns (exclude CODE and pottery_id)
    all_feature_columns = [
        col for col in features_df.columns if
        col not in ['CODE', 'pottery_id'] and not col.startswith('SHAPE_TYPE_')
    ]

    # Filter to selected features if specified
    if selected_features is not None and len(selected_features) > 0:
        feature_columns = [
            f for f in selected_features if f in all_feature_columns
        ]
        if len(feature_columns) == 0:
            print(
                f"Warning: None of the selected features found in CSV. Available features:"
            )
            print(", ".join(all_feature_columns))
            return
        print(f"\nAnalyzing {len(feature_columns)} selected features...")
        print(f"Features: {', '.join(feature_columns)}")
    else:
        feature_columns = all_feature_columns
        print(f"\nAnalyzing all {len(feature_columns)} binary features...")

    # Add SHAPE_TYPE as a special aggregated feature
    shape_type_columns = [
        col for col in features_df.columns if col.startswith('SHAPE_TYPE_')
    ]

    # Map colors to short labels
    short_label_color_map = {
        EMOTION_SHORT_LABEL_MAP[k]: v
        for k, v in EMOTION_COLOR_MAP.items()
    }
    emotion_order = [
        EMOTION_SHORT_LABEL_MAP[e] for e in EMOTION_STACK_ORDER
        if e in EMOTION_COLOR_MAP.keys() and e != "NO RESPONSE"
    ]
    plot_colors = [
        short_label_color_map.get(e, '#CCCCCC') for e in emotion_order
    ]

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
        results_df[emotion_order].plot(kind='bar',
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

        ax.set_title(f'Emotion Response by Feature: {feature}',
                     fontsize=14,
                     pad=20)
        ax.set_ylabel('Average Percentage (%)', fontsize=12)
        ax.set_xlabel('Feature Present', fontsize=12)
        ax.set_ylim(0, 100)
        ax.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()

        # Save plot
        filename = f"{feature.lower().replace(' ', '_')}_emotion_analysis.png"
        plt.savefig(os.path.join(output_dir, filename),
                    dpi=150,
                    bbox_inches='tight')
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
            shape_df[emotion_order].plot(kind='bar',
                                         stacked=True,
                                         ax=ax,
                                         color=plot_colors,
                                         width=0.7)

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

            ax.set_title('Emotion Response by Pottery Shape Type',
                         fontsize=14,
                         pad=20)
            ax.set_ylabel('Average Percentage (%)', fontsize=12)
            ax.set_xlabel('Shape Type', fontsize=12)
            ax.set_ylim(0, 100)
            ax.legend(title='Emotion',
                      bbox_to_anchor=(1.05, 1),
                      loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            plt.tight_layout()

            plt.savefig(os.path.join(output_dir,
                                     'shape_type_emotion_analysis.png'),
                        dpi=150,
                        bbox_inches='tight')
            plt.close(fig)

            # Save summary CSV
            shape_df.to_csv(os.path.join(output_dir, 'shape_type_summary.csv'))

    print(f"\nAnalysis complete! Results saved to '{output_dir}/' directory")
    feature_count = len(feature_columns)
    shape_count = 1 if include_shape_analysis else 0
    print(
        f"Generated {feature_count} binary feature plot(s) + {shape_count} shape type plot(s)"
    )


def compute_cosine_similarities_with_interesting(
    combined_df_japan: pd.DataFrame,
    combined_df_malaysia: pd.DataFrame,
    features_csv: str,
    output_csv: str = "cosine_similarities_with_interesting.csv"
):
    """
    Compute normalized (Cosine) and unnormalized (Dot Product) similarities
    between Japan and Malaysia emotion profiles.
    
    Outputs a CSV and console table in the specified rich format.
    
    Args:
        combined_df_japan: DataFrame with Japanese responses
        combined_df_malaysia: DataFrame with Malaysian responses
        features_csv: Path to features CSV
        output_csv: Output file to save results
    """
    import numpy as np

    # Load features
    features_df = pd.read_csv(features_csv)
    features_df['pottery_id'] = features_df['CODE'].str.replace('.ply', '', regex=False)

    # Define emotion columns (FULL 5 emotions, excluding NO RESPONSE)
    short_to_long_en = {v: k for k, v in SHORT_LABELS_EN.items()}
    emotion_cols_short = ["Interesting", "Beautiful", "Strange", "Scary", "Feel nothing"]
    emotion_cols_full = [short_to_long_en[s] for s in emotion_cols_short]

    # Map Japanese answers to English-equivalent short labels
    def map_jp_to_short(answer):
        mapping = {
            "面白い・気になる形だ": "Interesting",
            "美しい・芸術的だ": "Beautiful",
            "不思議・意味不明": "Strange",
            "不気味・不安・怖い": "Scary",
            "何も感じない": "Feel nothing"
        }
        return mapping.get(answer.strip(), None)

    # Process Japan
    combined_df_japan['short_answer'] = combined_df_japan['answer'].apply(map_jp_to_short)
    combined_df_japan = combined_df_japan.dropna(subset=['short_answer'])

    # Process Malaysia (already uses English)
    combined_df_malaysia['short_answer'] = combined_df_malaysia['answer'].map(SHORT_LABELS_EN)
    combined_df_malaysia = combined_df_malaysia.dropna(subset=['short_answer'])

    # Compute mean emotion vectors per pottery (session-normalized)
    def get_mean_emotions_per_pottery(df):
        session_counts = pd.crosstab(
            [df['pottery_id'], df['session_id']],
            df['short_answer']
        )
        session_pct = session_counts.div(session_counts.sum(axis=1), axis=0) * 100
        pottery_means = session_pct.groupby('pottery_id').mean()
        # Ensure all 5 emotions are present (fill missing with 0)
        for col in emotion_cols_short:
            if col not in pottery_means.columns:
                pottery_means[col] = 0.0
        return pottery_means[emotion_cols_short].fillna(0.0)

    japan_means = get_mean_emotions_per_pottery(combined_df_japan)
    malaysia_means = get_mean_emotions_per_pottery(combined_df_malaysia)

    # Merge with features
    feature_only_cols = [c for c in features_df.columns if c.startswith('HAS_') or c.startswith('NO_')]
    japan_merged = features_df[['pottery_id'] + feature_only_cols].merge(
        japan_means, on='pottery_id', how='inner'
    )
    malaysia_merged = features_df[['pottery_id'] + feature_only_cols].merge(
        malaysia_means, on='pottery_id', how='inner'
    )

    # Identify binary feature columns
    feature_cols = [col for col in features_df.columns if col.startswith('HAS_')]

    # --- Define Helpers for new format ---
    def format_vec_str(v, precision=2):
        """Formats a numpy array into 'X.XX | Y.YY | ...'"""
        if v is None or (isinstance(v, np.ndarray) and v.size == 0):
            return "N/A"
        return " | ".join([f"{x:.{precision}f}" for x in v])

    def normalize_vec(v):
        """Performs L2 normalization on a vector, handling zero vectors."""
        if v is None: return None
        norm = np.linalg.norm(v)
        if norm == 0:
            return np.zeros_like(v)
        return v / norm
        
    def safe_cosine(a, b):
        """Calculates cosine similarity (normalized angle), handling zero vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return np.nan
        dot_product = np.dot(a, b)
        similarity = dot_product / (norm_a * norm_b)
        return similarity
    # --- End Helpers ---

    # Define the headers for the new format
    # Using 'St' for Strange and 'Sc' for Scary to avoid 'S' collision
    emotion_header_str = "(I | B | St | Sc | F)"
    
    col_feature = "Feature"
    col_jp_orig = f"Japan (Orig)\n{emotion_header_str}"
    col_my_orig = f"Malaysia (Orig)\n{emotion_header_str}"
    col_norm = f"Normalized (J | M)\n{emotion_header_str} | {emotion_header_str}"
    
    # --- RENAMED COLUMNS FOR CLARITY ---
    col_sim_unnorm = "Unnormalized Sim. (Dot Product)" # This is the Dot Product
    col_sim_norm = "Normalized Sim. (Cosine)" # This is the Cosine Similarity
    # ---

    final_output_data = [] # This will hold dicts for the final DataFrame

    for feature in feature_cols:
        # Get mean vectors for 'Has' and 'No' groups
        j_has = japan_merged[japan_merged[feature] == 1][emotion_cols_short]
        j_no = japan_merged[japan_merged[feature] == 0][emotion_cols_short]
        m_has = malaysia_merged[malaysia_merged[feature] == 1][emotion_cols_short]
        m_no = malaysia_merged[malaysia_merged[feature] == 0][emotion_cols_short]

        j_has_vec = j_has.mean().values if len(j_has) > 0 else np.zeros(5)
        j_no_vec = j_no.mean().values if len(j_no) > 0 else np.zeros(5)
        m_has_vec = m_has.mean().values if len(m_has) > 0 else np.zeros(5)
        m_no_vec = m_no.mean().values if len(m_no) > 0 else np.zeros(5)

        # Calculate NORMALIZED similarity (Cosine Similarity)
        sim_norm_has = safe_cosine(j_has_vec, m_has_vec)
        sim_norm_no = safe_cosine(j_no_vec, m_no_vec)
        
        # Calculate UNNORMALIZED similarity (Dot Product)
        sim_unnorm_has = np.dot(j_has_vec, m_has_vec) if (len(j_has) > 0 and len(m_has) > 0) else np.nan
        sim_unnorm_no = np.dot(j_no_vec, m_no_vec) if (len(j_no) > 0 and len(m_no) > 0) else np.nan
        
        # Calculate normalized vectors (for display)
        norm_j_has = normalize_vec(j_has_vec)
        norm_m_has = normalize_vec(m_has_vec)
        norm_j_no = normalize_vec(j_no_vec)
        norm_m_no = normalize_vec(m_no_vec)

        # Prettify feature names
        feature_base_name = feature.replace('HAS_', '').replace('_', ' ').title()
        
        # --- Row 1 (Has Feature) ---
        final_output_data.append({
            col_feature: f"Has {feature_base_name}",
            col_jp_orig: format_vec_str(j_has_vec),
            col_my_orig: format_vec_str(m_has_vec),
            col_norm: f"{format_vec_str(norm_j_has)} | {format_vec_str(norm_m_has)}",
            col_sim_unnorm: f"{sim_unnorm_has:.2f}" if pd.notna(sim_unnorm_has) else "N/A",
            col_sim_norm: f"{sim_norm_has:.4f}" if pd.notna(sim_norm_has) else "N/A"
        })
        
        # --- Row 2 (No Feature) ---
        final_output_data.append({
            col_feature: f"No {feature_base_name}",
            col_jp_orig: format_vec_str(j_no_vec),
            col_my_orig: format_vec_str(m_no_vec),
            col_norm: f"{format_vec_str(norm_j_no)} | {format_vec_str(norm_m_no)}",
            col_sim_unnorm: f"{sim_unnorm_no:.2f}" if pd.notna(sim_unnorm_no) else "N/A",
            col_sim_norm: f"{sim_norm_no:.4f}" if pd.notna(sim_norm_no) else "N/A"
        })

    # Create the final DataFrame
    final_df = pd.DataFrame(final_output_data)
    
    # Save the new formatted CSV
    final_df.to_csv(output_csv, index=False)
    print(f"\nFormatted similarity scores (with 'Interesting') saved to: {output_csv}")
    
    # --- Print new formatted table to console ---
    total_width = 190
    print("\n" + "="*total_width)
    print("CROSS-CULTURAL SIMILARITY (INCLUDING 'Interesting') BY FEATURE")
    print("="*total_width)
    
    # Define column widths for padding
    w_feat = 30
    w_orig = 30 # Width for (I | B | St | Sc | F) vectors
    w_norm = 63 # Width for two normalized vectors
    w_sim_unnorm = 30 # Renamed/resized
    w_sim_norm = 26 # Renamed/resized

    # Create header strings
    h1 = f"{'Feature':<{w_feat}} {'Japan (Orig)':<{w_orig}} {'Malaysia (Orig)':<{w_orig}} {'Normalized (J | M)':<{w_norm}} {col_sim_unnorm:<{w_sim_unnorm}} {col_sim_norm:<{w_sim_norm}}"
    h2 = f"{'':<{w_feat}} {emotion_header_str:<{w_orig}} {emotion_header_str:<{w_orig}} {emotion_header_str + ' | ' + emotion_header_str:<{w_norm}} {'(Angle * Magnitude)':<{w_sim_unnorm}} {'(Angle Only)':<{w_sim_norm}}"
    print(h1)
    print(h2)
    print("-" * (w_feat + w_orig + w_orig + w_norm + w_sim_unnorm + w_sim_norm + 5))

    for _, row in final_df.iterrows():
        # Get data, ensuring 'None' or 'N/A' doesn't break ljust
        feat = str(row[col_feature] or "N/A")
        jp_orig = str(row[col_jp_orig] or "N/A")
        my_orig = str(row[col_my_orig] or "N/A")
        norm = str(row[col_norm] or "N/A")
        sim_unnorm = str(row[col_sim_unnorm] or "N/A")
        sim_norm = str(row[col_sim_norm] or "N/A")

        print(f"{feat:<{w_feat}} {jp_orig:<{w_orig}} {my_orig:<{w_orig}} {norm:<{w_norm}} {sim_unnorm:<{w_sim_unnorm}} {sim_norm:<{w_sim_norm}}")
    
    print("="*total_width)
    
    return final_df


# Main Execution
if __name__ == "__main__":
    # === USER CONTROLS ===
    SELECTED_LANGUAGE = 'malaysia'
    # SELECTED_LANGUAGE = 'japan'
    POTTERY_SELECTION = []  # Empty list for all pottery
    INCLUDE_POTTERY = True

    DATASET_ROOT_DIR = "./src/jomon_kaen_dataset/malaysia"
    # DATASET_ROOT_DIR = "./src/jomon_kaen_dataset/japan"
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
        combined_dataframe = load_combined_qna_data(DATASET_ROOT_DIR,
                                                    POTTERY_MODELS_DIR)

        # Filter pottery if selection is specified
        if not combined_dataframe.empty and POTTERY_SELECTION:
            print(
                f"\nFiltering pottery based on selection (Include mode: {INCLUDE_POTTERY})..."
            )
            base_ids = combined_dataframe['pottery_id'].str.split(
                '(', expand=True)[0]
            initial_count = len(combined_dataframe['pottery_id'].unique())

            if INCLUDE_POTTERY:
                combined_dataframe = combined_dataframe[base_ids.isin(
                    POTTERY_SELECTION)]
                print(
                    f"Included {len(combined_dataframe['pottery_id'].unique())} of {initial_count} unique pottery items."
                )
            else:
                combined_dataframe = combined_dataframe[
                    ~base_ids.isin(POTTERY_SELECTION)]
                print(
                    f"Excluded IDs, {len(combined_dataframe['pottery_id'].unique())} of {initial_count} unique items remaining."
                )

        # Run feature-based analysis
        if not combined_dataframe.empty:
            analyze_emotions_by_features(
                combined_dataframe,
                FEATURES_CSV,
                language=SELECTED_LANGUAGE,
                selected_features=SELECTED_FEATURES,
                include_shape_analysis=INCLUDE_SHAPE_TYPE_ANALYSIS)
        else:
            print("No data available for analysis.")

    except (FileNotFoundError, ValueError) as e:
        print(f"Could not run analysis due to an error: {e}")
        print(
            "Please ensure all directories and the features CSV are set up correctly."
        )

    try:
        # Load both datasets
        DATASET_ROOT_JAPAN = "./src/jomon_kaen_dataset/japan"
        DATASET_ROOT_MALAYSIA = "./src/jomon_kaen_dataset/malaysia"
        
        print("\nLoading Japan dataset for cross-cultural comparison...")
        df_japan = load_combined_qna_data(DATASET_ROOT_JAPAN, POTTERY_MODELS_DIR)
        print("\nLoading Malaysia dataset for cross-cultural comparison...")
        df_malaysia = load_combined_qna_data(DATASET_ROOT_MALAYSIA, POTTERY_MODELS_DIR)

        if not df_japan.empty and not df_malaysia.empty:
            print("\nComputing cross-cultural similarities (INCLUDING 'Interesting')...")
            sim_results = compute_cosine_similarities_with_interesting(
                df_japan, df_malaysia, FEATURES_CSV
            )
        else:
            print("Skipping similarity analysis (with Interesting) due to missing data.")
    except Exception as e:
        print(f"Error during extended similarity computation: {e}")