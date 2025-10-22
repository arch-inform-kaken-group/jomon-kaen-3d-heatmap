"""
Author: Gemini
Date: 22 October 2025
Description:
This script performs a demographic analysis by reading 'language.txt' and 'gender.txt'
files from a nested session directory structure (root/group/session).
It aggregates the data, saves the statistics to a CSV file, and generates
a bar plot of the distributions.
"""

import os
import sys
import argparse
from pathlib import Path
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib  # Ensures Japanese characters render correctly
from tqdm import tqdm

def increment_error(key, path_str, errors):
    """Helper function to log errors."""
    if key not in errors:
        errors[key] = {'count': 0, 'paths': set()}
    errors[key]['count'] += 1
    errors[key]['paths'].add(path_str)

def analyze_demographics(root_dir, output_dir):
    """
    Analyzes language and gender data from session files.

    Args:
        root_dir (str): The root directory containing group folders.
        output_dir (str): The directory to save the output CSV and plot.
    """
    print(f"Starting demographic analysis on: {root_dir}")
    root_path = Path(root_dir)
    output_path = Path(output_dir)
    
    if not root_path.is_dir():
        print(f"Error: Root directory not found: {root_dir}", file=sys.stderr)
        return

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    language_data = []
    gender_data = []
    errors = {}
    
    session_paths = []
    
    # First, collect all session paths to use with tqdm
    print("Scanning directories...")
    for group_path in root_path.iterdir():
        if not group_path.is_dir():
            continue
        for session_path in group_path.iterdir():
            if session_path.is_dir():
                session_paths.append(session_path)

    print(f"Found {len(session_paths)} session directories. Processing files...")

    # Process each session
    for session_path in tqdm(session_paths, desc="Processing Sessions"):
        session_id_str = f"{session_path.parent.name}/{session_path.name}"
        
        # --- Process language.txt ---
        lang_file = session_path / 'language.txt'
        if lang_file.exists():
            try:
                language = lang_file.read_text(encoding='utf-8').strip()
                if language:
                    language_data.append(language)
                else:
                    increment_error('Empty language.txt', str(lang_file), errors)
            except Exception as e:
                increment_error(f'Read error language.txt: {e}', str(lang_file), errors)
        else:
            increment_error('Missing language.txt', str(lang_file), errors)
            
        # --- Process gender.txt ---
        gender_file = session_path / 'gender.txt'
        if gender_file.exists():
            try:
                gender = gender_file.read_text(encoding='utf-8').strip()
                if gender:
                    gender_data.append(gender)
                else:
                    # File is empty
                    gender_data.append("Cannot Identify, \nDue to Missing Voice Data")
                    increment_error('Empty gender.txt (logged as Cannot Identify)', str(gender_file), errors)
            except Exception as e:
                # Read error
                gender_data.append("Cannot Identify, \nDue to Missing Voice Data")
                increment_error(f'Read error gender.txt (logged as Cannot Identify): {e}', str(gender_file), errors)
        else:
            # File is missing
            gender_data.append("Cannot Identify, \nDue to Missing Voice Data")
            increment_error('Missing gender.txt (logged as Cannot Identify)', str(gender_file), errors)

    print("Aggregation complete. Generating outputs...")

    # --- Aggregate and Save CSV ---
    if not language_data and not gender_data:
        print("No demographic data found. Exiting.", file=sys.stderr)
        return

    # Create DataFrames from counts
    lang_counts = Counter(language_data)
    lang_df = pd.DataFrame(lang_counts.items(), columns=['category', 'count']).assign(type='language')
    
    gender_counts = Counter(gender_data)
    gender_df = pd.DataFrame(gender_counts.items(), columns=['category', 'count']).assign(type='gender')

    # Combine and save stats
    combined_stats_df = pd.concat([lang_df, gender_df], ignore_index=True)
    csv_path = output_path / 'demographic_stats.csv'
    combined_stats_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Stats saved to: {csv_path}")

    # --- Generate and Save Plot ---
    # Update font sizes for better readability
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'figure.titlesize': 20
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Language plot
    if not lang_df.empty:
        lang_df_sorted = lang_df.sort_values('count', ascending=False)
        ax1.bar(lang_df_sorted['category'], lang_df_sorted['count'], color='blue')
        ax1.set_title(f'Language Distribution (n={len(language_data)})')
        ax1.set_xlabel('Language', labelpad=20)
        ax1.set_ylabel('Count', labelpad=20)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add text labels inside bars
        for bar in ax1.patches:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f'{int(bar.get_height())}',
                ha='center',
                va='center',
                color='white',
                fontsize=20,
                fontweight='bold'
            )
    else:
        ax1.set_title("No Language Data Found")
        ax1.text(0.5, 0.5, "No data", horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)

    # Gender plot
    if not gender_df.empty:
        gender_df_sorted = gender_df.sort_values('count', ascending=False)
        ax2.bar(gender_df_sorted['category'], gender_df_sorted['count'], color='red')
        ax2.set_title(f'Gender Distribution (n={len(gender_data)})')
        ax2.set_xlabel('Gender', labelpad=20)
        ax2.set_ylabel('Count', labelpad=20)

        # Add text labels inside bars
        for bar in ax2.patches:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f'{int(bar.get_height())}',
                ha='center',
                va='center',
                color='white',
                fontsize=20,
                fontweight='bold'
            )
    else:
        ax2.set_title("No Gender Data Found")
        ax2.text(0.5, 0.5, "No data", horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        
    plt.suptitle('Participant Demographics', fontsize=22)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plot_path = output_path / 'demographic_plot.png'
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Plot saved to: {plot_path}")

    # --- Print Errors ---
    if errors:
        print("\n--- Errors Encountered ---")
        for key, info in errors.items():
            print(f"  {key}: {info['count']} instances.")
            if info['count'] < 10:
                for path_str in info['paths']:
                    print(f"    - {path_str}")
            else:
                print(f"    - (See first path) {next(iter(info['paths']))} and {info['count'] - 1} others.")
        print("--------------------------")
        
    print("Analysis finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze demographics (language and gender) from session data."
    )
    parser.add_argument(
        "root", 
        type=str, 
        help="Root directory containing group folders (e.g., './src/jomon_kaen_dataset/japan')."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=False,
        default="output_data",
        help="Directory to save 'demographic_stats.csv' and 'demographic_plot.png' (e.g., './analysis_output')."
    )
    
    args = parser.parse_args()
    
    analyze_demographics(args.root, args.output_dir)

