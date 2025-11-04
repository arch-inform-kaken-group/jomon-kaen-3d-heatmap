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

# Japanese Emotion Map - mapped to English labels
EMOTION_COLOR_MAP_JP = {
    "面白い・気になる形だ": "#00FFFF",
    "美しい・芸術的だ": "#00FF00",
    "不思議・意味不明": "#FFFF00",
    "不気味・不安・怖い": "#FF0000",
    "何も感じない": "#505050",
    "NO RESPONSE": "#D3D3D3",
}

# Map Japanese to English labels
SHORT_LABELS_JP_TO_EN = {
    "面白い・気になる形だ": "Interesting",
    "美しい・芸術的だ": "Beautiful",
    "不思議・意味不明": "Strange",
    "不気味・不安・怖い": "Scary",
    "何も感じない": "Feel nothing",
    "NO RESPONSE": "NO RESPONSE"
}

# CSV column name mapping
FEATURE_NAME_MAPPING = {
    'Flame-like decoration': 'HAS_FLAME_LIKE_DECORATION',
    'Crown-like decoration': 'HAS_CROWN_LIKE_DECORATION',
    'Handles': 'HAS_HANDLES',
    'Cord-marked pattern': 'HAS_CORD_MARKED_PATTERN',
    'Nail engraving': 'HAS_NAIL_ENGRAVING',
    'Spiral pattern': 'HAS_SPIRAL_PATTERN',
    'Flat base': 'HAS_FLAT_BASE'
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


def compare_japan_malaysia_by_features(combined_df_japan: pd.DataFrame,
                                       combined_df_malaysia: pd.DataFrame,
                                       features_csv: str,
                                       selected_features: list = None,
                                       include_protrusions: bool = True,
                                       include_shape_types: bool = True):
    """Compare emotion responses between Japan and Malaysia by pottery features."""

    # Use English labels for everything
    emotion_order = [
        "Feel nothing", "Scary", "Strange", "Beautiful", "Interesting"
    ]
    colors = ["#505050", "#FF0000", "#FFFF00", "#00FF00", "#00FFFF"]

    if combined_df_japan.empty or combined_df_malaysia.empty:
        print("One or both DataFrames are empty. No analysis performed.")
        return

    # Load features CSV
    features_df = pd.read_csv(features_csv)
    features_df['pottery_id'] = features_df['Pottery ID']

    # Process Japan data - map to English labels
    combined_df_japan['answer'] = combined_df_japan['answer'].str.strip()
    combined_df_japan['short_answer'] = combined_df_japan['answer'].map(
        SHORT_LABELS_JP_TO_EN)

    # Process Malaysia data - map to English labels
    combined_df_malaysia['answer'] = combined_df_malaysia['answer'].str.strip()
    combined_df_malaysia['short_answer'] = combined_df_malaysia['answer'].map(
        SHORT_LABELS_EN)

    # Calculate percentages for both datasets
    def calculate_percentages(df):
        session_counts_df = pd.crosstab([df['pottery_id'], df['session_id']],
                                        df['short_answer'])
        session_percentage_df = session_counts_df.div(
            session_counts_df.sum(axis=1), axis=0) * 100
        return session_percentage_df

    japan_session_pct = calculate_percentages(combined_df_japan)
    malaysia_session_pct = calculate_percentages(combined_df_malaysia)

    # Merge with features
    japan_merged = features_df.merge(japan_session_pct.reset_index(),
                                     on='pottery_id',
                                     how='inner')
    malaysia_merged = features_df.merge(malaysia_session_pct.reset_index(),
                                        on='pottery_id',
                                        how='inner')

    # Create output directory
    output_dir = "cross_cultural_comparison"
    os.makedirs(output_dir, exist_ok=True)

    # Get feature columns from CSV
    all_feature_columns = list(FEATURE_NAME_MAPPING.keys())

    # Filter to selected features if specified
    if selected_features is not None and len(selected_features) > 0:
        feature_columns = [
            f for f in selected_features if f in all_feature_columns
        ]
        if len(feature_columns) == 0:
            print(f"Warning: None of the selected features found in CSV.")
            return
        print(f"\nAnalyzing {len(feature_columns)} selected features...")
    else:
        feature_columns = all_feature_columns
        print(f"\nAnalyzing all {len(feature_columns)} binary features...")

    # Analyze each feature
    for feature in tqdm(feature_columns, desc="Processing features"):
        fig, ax = plt.subplots(figsize=(12, 8))

        # Process Japan data
        jp_feature_groups = japan_merged[japan_merged[feature] == 1]
        my_feature_groups = malaysia_merged[malaysia_merged[feature] == 1]

        if len(jp_feature_groups) == 0 or len(my_feature_groups) == 0:
            plt.close(fig)
            continue

        # Japan analysis (feature present only)
        jp_count = len(jp_feature_groups['session_id'].unique())
        jp_emotion_avgs = jp_feature_groups[emotion_order].mean()
        jp_emotion_stds = jp_feature_groups[emotion_order].std()

        # Malaysia analysis (feature present only)
        my_count = len(my_feature_groups['session_id'].unique())
        my_emotion_avgs = my_feature_groups[emotion_order].mean()
        my_emotion_stds = my_feature_groups[emotion_order].std()

        # Create combined dataframe
        combined_results = pd.DataFrame({
            f'Japan (n={jp_count})':
            jp_emotion_avgs,
            f'Malaysia (n={my_count})':
            my_emotion_avgs
        }).T

        combined_stds = pd.DataFrame({
            f'Japan (n={jp_count})':
            jp_emotion_stds,
            f'Malaysia (n={my_count})':
            my_emotion_stds
        }).T

        # Plot combined data
        combined_results[emotion_order].plot(kind='bar',
                                             stacked=True,
                                             ax=ax,
                                             color=colors,
                                             width=0.7,
                                             legend=False)

        # Add error bars
        x_positions = np.arange(len(combined_results))
        cumulative_bottoms = combined_results.cumsum(axis=1) - combined_results

        for i, emotion in enumerate(emotion_order):
            means = combined_results[emotion]
            stds = combined_stds[emotion]
            bottoms = cumulative_bottoms[emotion]
            y_midpoints = bottoms + means / 2.0

            for j, x_pos in enumerate(x_positions):
                ax.text(x=x_pos + 0.06 + i * 0.05,
                        y=y_midpoints.iloc[j],
                        s=f"{stds.iloc[j]:.2f}",
                        fontsize=12)
                ax.errorbar(x=[x_pos + i * 0.05 + 0.06],
                            y=[y_midpoints.iloc[j]],
                            yerr=stds.iloc[j] / 2.0,
                            fmt='none',
                            ecolor='black',
                            capsize=4,
                            elinewidth=1.2,
                            alpha=0.7)

        # Add value labels
        for container in ax.containers:
            if not hasattr(container, 'datavalues'):
                continue
            labels = [
                f'{v:.1f}' if v > 2 else '' for v in container.datavalues
            ]
            ax.bar_label(container,
                         labels=labels,
                         label_type='center',
                         fontsize=14,
                         color='black',
                         weight='bold')

        ax.set_title(f'Potteries with {feature}',
                     fontsize=24,
                     pad=20,
                     weight='bold')
        ax.set_ylabel('Average Percentage (%)', fontsize=18)
        ax.set_xlabel('', fontsize=18)
        ax.set_ylim(-10, 110)
        ax.set_xticklabels(combined_results.index, rotation=0, fontsize=16)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        # Create horizontal legend below
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors[i], label=emotion_order[i])
            for i in range(len(emotion_order))
        ]
        ax.legend(handles=legend_elements,
                  loc='lower center',
                  ncol=5,
                  fontsize=16,
                  frameon=True,
                  bbox_to_anchor=(0.5, -0.15))

        plt.tight_layout()

        # Save plot
        filename = f"{feature.lower().replace(' ', '_').replace('-', '_')}_comparison.png"
        plt.savefig(os.path.join(output_dir, filename),
                    dpi=1000,
                    bbox_inches='tight')
        plt.close(fig)

    # Analyze number of protrusions - SEPARATE GRAPHS
    if include_protrusions:
        print("\nAnalyzing number of protrusions...")
        protrusion_cols = [
            col for col in features_df.columns
            if col.startswith('Number_of_protrusions_')
        ]

        def get_protrusion_count(row):
            for col in protrusion_cols:
                if row[col] == 1:
                    return col.replace('Number_of_protrusions_',
                                       '').replace('.0', '')
            return '0'

        japan_merged['protrusion_count'] = japan_merged.apply(
            get_protrusion_count, axis=1)
        malaysia_merged['protrusion_count'] = malaysia_merged.apply(
            get_protrusion_count, axis=1)

        # Create separate graphs for each country
        for country, merged_df in [('Japan', japan_merged),
                                   ('Malaysia', malaysia_merged)]:

            all_protrusion_counts = sorted(
                merged_df['protrusion_count'].unique(),
                key=lambda x: float(x) if x != 'Unknown' else -1)

            fig, ax = plt.subplots(figsize=(12, 8))

            results_list = []
            for count in all_protrusion_counts:
                group = merged_df[merged_df['protrusion_count'] == count]
                if len(group) < 2:
                    continue

                n_sessions = len(group['session_id'].unique())
                emotion_avgs = group[emotion_order].mean()
                emotion_stds = group[emotion_order].std()

                results_list.append({
                    'label': f'{count} protrusions\n(n={n_sessions})',
                    'avgs': emotion_avgs,
                    'stds': emotion_stds
                })

            if results_list:
                labels = [r['label'] for r in results_list]
                results_df = pd.DataFrame([r['avgs'] for r in results_list],
                                          index=labels)
                stds_df = pd.DataFrame([r['stds'] for r in results_list],
                                       index=labels)

                results_df[emotion_order].plot(kind='bar',
                                               stacked=True,
                                               ax=ax,
                                               color=colors,
                                               width=0.7,
                                               legend=False)

                # Add error bars
                x_positions = np.arange(len(results_df))
                cumulative_bottoms = results_df.cumsum(axis=1) - results_df

                for i, emotion in enumerate(emotion_order):
                    means = results_df[emotion]
                    stds = stds_df[emotion]
                    bottoms = cumulative_bottoms[emotion]
                    y_midpoints = bottoms + means / 2.0

                    for j, x_pos in enumerate(x_positions):
                        ax.text(x=x_pos + 0.06 + i * 0.05,
                                y=y_midpoints.iloc[j],
                                s=f"{stds.iloc[j]:.2f}",
                                fontsize=10)
                        ax.errorbar(x=[x_pos + i * 0.05 + 0.06],
                                    y=[y_midpoints.iloc[j]],
                                    yerr=stds.iloc[j] / 2.0,
                                    fmt='none',
                                    ecolor='black',
                                    capsize=4,
                                    elinewidth=1.2,
                                    alpha=0.7)

                # Add value labels
                for container in ax.containers:
                    if not hasattr(container, 'datavalues'):
                        continue
                    labels_vals = [
                        f'{v:.1f}' if v > 2 else ''
                        for v in container.datavalues
                    ]
                    ax.bar_label(container,
                                 labels=labels_vals,
                                 label_type='center',
                                 fontsize=12,
                                 color='black',
                                 weight='bold')

                ax.set_title(
                    f'{country} - Emotion Response by Number of Protrusions',
                    fontsize=24,
                    pad=20,
                    weight='bold')
                ax.set_ylabel('Average Percentage (%)', fontsize=18)
                ax.set_xlabel('', fontsize=18)
                ax.set_ylim(-10, 110)
                ax.set_xticklabels(results_df.index,
                                   rotation=45,
                                   ha='right',
                                   fontsize=14)
                ax.grid(axis='y', linestyle='--', alpha=0.3)

                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor=colors[i], label=emotion_order[i])
                    for i in range(len(emotion_order))
                ]
                ax.legend(handles=legend_elements,
                          loc='lower center',
                          ncol=5,
                          fontsize=16,
                          frameon=True,
                          bbox_to_anchor=(0.5, -0.4))

                plt.tight_layout()
                filename = f'protrusions_{country.lower()}.png'
                plt.savefig(os.path.join(output_dir, filename),
                            dpi=1000,
                            bbox_inches='tight')
                plt.close(fig)

    # Analyze shape types - SEPARATE GRAPHS
    if include_shape_types:
        print("\nAnalyzing shape types...")
        shape_type_cols = [
            col for col in features_df.columns if col.endswith(' type')
        ]

        def get_shape_type(row):
            for col in shape_type_cols:
                if row[col] == 1:
                    return col.replace(' type', '')
            return 'Unknown'

        japan_merged['shape_type'] = japan_merged.apply(get_shape_type, axis=1)
        malaysia_merged['shape_type'] = malaysia_merged.apply(get_shape_type,
                                                              axis=1)

        # Create separate graphs for each country
        for country, merged_df in [('Japan', japan_merged),
                                   ('Malaysia', malaysia_merged)]:

            all_shape_types = sorted(merged_df['shape_type'].unique())

            fig, ax = plt.subplots(figsize=(18, 8))

            results_list = []
            for shape_type in all_shape_types:
                group = merged_df[merged_df['shape_type'] == shape_type]
                if len(group) < 2:
                    continue

                n_sessions = len(group['session_id'].unique())
                emotion_avgs = group[emotion_order].mean()
                emotion_stds = group[emotion_order].std()

                results_list.append({
                    'label': f'{shape_type}\n(n={n_sessions})',
                    'avgs': emotion_avgs,
                    'stds': emotion_stds
                })

            if results_list:
                labels = [r['label'] for r in results_list]
                results_df = pd.DataFrame([r['avgs'] for r in results_list],
                                          index=labels)
                stds_df = pd.DataFrame([r['stds'] for r in results_list],
                                       index=labels)

                results_df[emotion_order].plot(kind='bar',
                                               stacked=True,
                                               ax=ax,
                                               color=colors,
                                               width=0.7,
                                               legend=False)

                # Add error bars
                x_positions = np.arange(len(results_df))
                cumulative_bottoms = results_df.cumsum(axis=1) - results_df

                for i, emotion in enumerate(emotion_order):
                    means = results_df[emotion]
                    stds = stds_df[emotion]
                    bottoms = cumulative_bottoms[emotion]
                    y_midpoints = bottoms + means / 2.0

                    for j, x_pos in enumerate(x_positions):
                        ax.text(x=x_pos + 0.06 + i * 0.05,
                                y=y_midpoints.iloc[j],
                                s=f"{stds.iloc[j]:.2f}",
                                fontsize=9)
                        ax.errorbar(x=[x_pos + i * 0.05 + 0.06],
                                    y=[y_midpoints.iloc[j]],
                                    yerr=stds.iloc[j] / 2.0,
                                    fmt='none',
                                    ecolor='black',
                                    capsize=4,
                                    elinewidth=1.2,
                                    alpha=0.7)

                # Add value labels
                for container in ax.containers:
                    if not hasattr(container, 'datavalues'):
                        continue
                    labels_vals = [
                        f'{v:.1f}' if v > 2 else ''
                        for v in container.datavalues
                    ]
                    ax.bar_label(container,
                                 labels=labels_vals,
                                 label_type='center',
                                 fontsize=11,
                                 color='black',
                                 weight='bold')

                ax.set_title(f'{country} - Emotion Response by Shape Type',
                             fontsize=24,
                             pad=20,
                             weight='bold')
                ax.set_ylabel('Average Percentage (%)', fontsize=18)
                ax.set_xlabel('', fontsize=18)
                ax.set_ylim(-10, 110)
                ax.set_xticklabels(results_df.index,
                                   rotation=45,
                                   ha='right',
                                   fontsize=12)
                ax.grid(axis='y', linestyle='--', alpha=0.3)

                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor=colors[i], label=emotion_order[i])
                    for i in range(len(emotion_order))
                ]
                ax.legend(handles=legend_elements,
                          loc='lower center',
                          ncol=5,
                          fontsize=16,
                          frameon=True,
                          bbox_to_anchor=(0.5, -0.4))

                plt.tight_layout()
                filename = f'shape_type_{country.lower()}.png'
                plt.savefig(os.path.join(output_dir, filename),
                            dpi=1000,
                            bbox_inches='tight')
                plt.close(fig)

    print(f"\nAnalysis complete! Results saved to '{output_dir}/' directory")


# Main Execution
if __name__ == "__main__":
    POTTERY_MODELS_DIR = "./src/pottery"
    FEATURES_CSV = "./DS_Labels_Cleaned.csv"

    DATASET_ROOT_JAPAN = "./src/jomon_kaen_dataset/japan"
    DATASET_ROOT_MALAYSIA = "./src/jomon_kaen_dataset/malaysia"

    # SELECT FEATURES TO ANALYZE
    SELECTED_FEATURES = None  # None for all features, or list specific ones
    # SELECTED_FEATURES = ['Flame-like decoration', 'Crown-like decoration', 'Handles']

    INCLUDE_PROTRUSIONS = True
    INCLUDE_SHAPE_TYPES = True

    try:
        print("\nLoading Japan dataset...")
        df_japan = load_combined_qna_data(DATASET_ROOT_JAPAN,
                                          POTTERY_MODELS_DIR)

        print("\nLoading Malaysia dataset...")
        df_malaysia = load_combined_qna_data(DATASET_ROOT_MALAYSIA,
                                             POTTERY_MODELS_DIR)

        if not df_japan.empty and not df_malaysia.empty:
            print(
                "\nComparing Japan and Malaysia emotion responses by features..."
            )
            compare_japan_malaysia_by_features(
                df_japan,
                df_malaysia,
                FEATURES_CSV,
                selected_features=SELECTED_FEATURES,
                include_protrusions=INCLUDE_PROTRUSIONS,
                include_shape_types=INCLUDE_SHAPE_TYPES)
        else:
            print("One or both datasets are empty. Cannot perform comparison.")

    except (FileNotFoundError, ValueError) as e:
        print(f"Could not run analysis due to an error: {e}")
        print(
            "Please ensure all directories and the features CSV are set up correctly."
        )
