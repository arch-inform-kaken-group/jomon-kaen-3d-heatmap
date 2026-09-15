# # import os
# # import argparse
# # import numpy as np
# # import pandas as pd
# # import itertools
# # import warnings
# # from scipy.stats import chi2_contingency, ttest_ind
# # from scipy.spatial.distance import pdist, squareform
# # from pathlib import Path
# # from sklearn.decomposition import PCA
# # from sklearn.preprocessing import StandardScaler
# # import matplotlib.pyplot as plt
# # from mpl_toolkits.mplot3d import Axes3D
# # from matplotlib.patches import Patch

# # # Suppress scipy/numpy warnings for clean console output
# # warnings.filterwarnings('ignore')

# # # ==========================================
# # # CONSTANTS & MAPPINGS
# # # ==========================================
# # DOGU_PREFIXES = [
# #     'IN0295', 'IN0306', 'MH0037', 'NM0239',
# #     'NZ0001', 'SK0035', 'TK0020', 'UD0028'
# # ]

# # EMOTION_MAP_JP = {
# #     "面白い・気になる形だ": "Interesting",
# #     "美しい・芸術的だ": "Beautiful",
# #     "不思議・意味不明": "Strange",
# #     "不気味・不安・怖い": "Scary",
# #     "何も感じない": "Feel nothing",
# #     "NO RESPONSE": "NO RESPONSE"
# # }

# # EMOTION_MAP_MY = {
# #     "Interesting and attentional shape": "Interesting",
# #     "Beautiful and artistic": "Beautiful",
# #     "Strange and incomprehensible": "Strange",
# #     "Creepy / unsettling / scary": "Scary",
# #     "Feel nothing": "Feel nothing",
# #     "NO RESPONSE": "NO RESPONSE"
# # }

# # TARGET_EMOTIONS = [
# #     'Interesting', 'Beautiful', 'Strange', 'Scary', 'Feel nothing'
# # ]

# # FEATURE_CSV_TO_INTERNAL = {
# #     'HAS_FLAME_LIKE_DECORATION': 'HAS_FLAME',
# #     'HAS_CROWN_LIKE_DECORATION': 'HAS_CROWN',
# #     'HAS_HANDLES': 'HAS_HANDLES',
# #     'HAS_CORD_MARKED_PATTERN': 'HAS_CORD',
# #     'HAS_NAIL_ENGRAVING': 'HAS_NAIL',
# #     'HAS_SPIRAL_PATTERN': 'HAS_SPIRAL',
# #     'HAS_FLAT_BASE': 'HAS_FLAT_BASE'
# # }

# # FEATURE_DISPLAY_NAMES = {
# #     'HAS_FLAME': 'Flame-like decoration',
# #     'HAS_CROWN': 'Crown-like decoration',
# #     'HAS_HANDLES': 'Handles',
# #     'HAS_CORD': 'Cord-marked pattern',
# #     'HAS_NAIL': 'Nail engraving',
# #     'HAS_SPIRAL': 'Spiral pattern',
# #     'HAS_FLAT_BASE': 'Flat base'
# # }

# # # ==========================================
# # # HIGH-PRECISION P-VALUE CALCULATOR
# # # ==========================================
# # def get_chi2_p_value(chi2_stat, df):
# #     """Calculates chi-squared p-value. Uses mpmath for arbitrary precision if scipy underflows to 0.0."""
# #     try:
# #         import mpmath
# #         mpmath.mp.dps = 50
# #         p_val = mpmath.gammainc(df / 2, chi2_stat / 2, mpmath.inf) / mpmath.gamma(df / 2)
# #         if p_val == 0:
# #             return "< 1e-300 (arbitrary precision underflow)"
# #         return str(p_val)
# #     except ImportError:
# #         from scipy.stats import chi2
# #         p = chi2.sf(chi2_stat, df)
# #         if p == 0.0:
# #             return "< 2.22e-308 (float64 underflow)"
# #         return f"{p:.15e}"

# # def format_p(p_val):
# #     """Formats p-value to preserve full precision using scientific notation."""
# #     if pd.isna(p_val) or p_val == 'N/A':
# #         return "N/A"
# #     if isinstance(p_val, str):
# #         return p_val
# #     return f"{p_val:.15e}"

# # # ==========================================
# # # EXPLANATION GENERATORS
# # # ==========================================
# # def explain_chi2(is_sig, comp_name):
# #     if is_sig:
# #         return (f"SIGNIFICANT: The overall distribution of raw emotion event counts differs significantly. "
# #                 f"This indicates that the condition ({comp_name}) fundamentally shifts the general pattern and frequency of emotional responses.")
# #     else:
# #         return (f"INSIGNIFICANT: The overall distribution of raw emotion event counts does not differ significantly. "
# #                 f"The general pattern of emotional responses remains statistically similar across the compared groups.")

# # def explain_ttest(is_sig, comp_name, emotion, mean_g1, mean_g2):
# #     if is_sig:
# #         higher_group = "the first group" if mean_g1 > mean_g2 else "the second group"
# #         return (f"SIGNIFICANT: The relative frequency of the '{emotion}' emotion differs significantly. "
# #                 f"{higher_group.capitalize()} exhibited a notably higher session-normalized percentage for this specific emotion "
# #                 f"(Mean 1: {mean_g1:.2f}%, Mean 2: {mean_g2:.2f}%).")
# #     else:
# #         return (f"INSIGNIFICANT: The relative frequency of the '{emotion}' emotion does not differ significantly between the groups, "
# #                 f"indicating a similar level of this specific emotional response across the compared conditions.")

# # def explain_permanova(is_sig, comp_name):
# #     if is_sig:
# #         return (f"SIGNIFICANT: The multivariate emotion profiles differ significantly. "
# #                 f"This means the combined, overall emotional experience (the multidimensional 'shape' of all 5 emotions) "
# #                 f"is distinctly different between the groups.")
# #     else:
# #         return (f"INSIGNIFICANT: The multivariate emotion profiles do not differ significantly. "
# #                 f"The overall emotional experience, considering all emotions simultaneously, is statistically similar across the compared groups.")

# # # ==========================================
# # # 1. DATA LOADING & AGGREGATION
# # # ==========================================
# # def load_qa_data(root_dir: str, country_label: str) -> pd.DataFrame:
# #     print(f"Loading {country_label} data from '{root_dir}'...")
# #     df_list = []
# #     root_path = Path(root_dir)
# #     if not root_path.exists():
# #         print(f"Warning: Directory not found: {root_dir}")
# #         return pd.DataFrame()
        
# #     for group_folder in root_path.iterdir():
# #         if not group_folder.is_dir(): continue
# #         for session_folder in group_folder.iterdir():
# #             if not session_folder.is_dir(): continue
# #             for pottery_folder in session_folder.iterdir():
# #                 if not pottery_folder.is_dir(): continue
# #                 qa_path = pottery_folder / "qa_corrected.csv"
# #                 if qa_path.exists():
# #                     try:
# #                         temp_df = pd.read_csv(qa_path, header=0, sep=",")
# #                         temp_df['timestamp'] = pd.to_numeric(temp_df['timestamp'], errors='coerce')
# #                         temp_df.dropna(subset=['timestamp'], inplace=True)
# #                         temp_df['pottery_id'] = pottery_folder.name
# #                         temp_df['session_id'] = session_folder.name
# #                         temp_df['country'] = country_label
# #                         df_list.append(temp_df)
# #                     except Exception as e:
# #                         print(f"Warning: Could not process {qa_path}: {e}")
                        
# #     if not df_list: return pd.DataFrame()
# #     return pd.concat(df_list, ignore_index=True)

# # def prepare_and_aggregate(df_jp: pd.DataFrame, df_my: pd.DataFrame, features_csv: str) -> tuple:
# #     print("Mapping emotions and aggregating session events...")
# #     df_jp['short_answer'] = df_jp['answer'].astype(str).str.strip().map(EMOTION_MAP_JP)
# #     df_my['short_answer'] = df_my['answer'].astype(str).str.strip().map(EMOTION_MAP_MY)
    
# #     df_jp = df_jp.dropna(subset=['short_answer'])
# #     df_my = df_my.dropna(subset=['short_answer'])
# #     combined_df = pd.concat([df_jp, df_my], ignore_index=True)
# #     combined_df = combined_df[combined_df['short_answer'] != 'NO RESPONSE']
    
# #     # STEP 1: Aggregate raw event counts for EACH SESSION & EACH OBJECT
# #     instance_event_counts = pd.crosstab([
# #         combined_df['country'], combined_df['session_id'], combined_df['pottery_id']
# #     ], combined_df['short_answer'])
    
# #     for emo in TARGET_EMOTIONS:
# #         if emo not in instance_event_counts.columns:
# #             instance_event_counts[emo] = 0
# #     instance_event_counts = instance_event_counts[TARGET_EMOTIONS]
    
# #     # STEP 2: Calculate the event count as a PERCENTAGE for EACH INSTANCE
# #     instance_totals = instance_event_counts.sum(axis=1)
# #     valid_instances_mask = instance_totals > 0
# #     instance_event_counts = instance_event_counts[valid_instances_mask]
# #     instance_totals = instance_totals[valid_instances_mask]
    
# #     instance_pct = instance_event_counts.div(instance_totals, axis=0) * 100.0
    
# #     instance_event_counts = instance_event_counts.reset_index()
# #     instance_pct = instance_pct.reset_index()
    
# #     instance_event_counts['pottery_id'] = instance_event_counts['pottery_id'].astype(str).str.replace('.ply', '', regex=False)
# #     instance_pct['pottery_id'] = instance_pct['pottery_id'].astype(str).str.replace('.ply', '', regex=False)
    
# #     instance_pct['artifact_type'] = instance_pct['pottery_id'].apply(
# #         lambda x: 'Dogu' if any(prefix in str(x) for prefix in DOGU_PREFIXES) else 'Pottery')
# #     instance_event_counts['artifact_type'] = instance_event_counts['pottery_id'].apply(
# #         lambda x: 'Dogu' if any(prefix in str(x) for prefix in DOGU_PREFIXES) else 'Pottery')
    
# #     if features_csv and os.path.exists(features_csv):
# #         print(f"Merging with features from: {features_csv}")
# #         features_df = pd.read_csv(features_csv)
# #         id_col = features_df.columns[0]
# #         features_df['pottery_id'] = features_df[id_col].astype(str).str.replace('.ply', '', regex=False)
        
# #         rename_dict = {k: v for k, v in FEATURE_CSV_TO_INTERNAL.items() if k in features_df.columns}
# #         features_df = features_df.rename(columns=rename_dict)
        
# #         internal_feature_cols = list(rename_dict.values())
# #         feature_cols = ['pottery_id'] + [c for c in internal_feature_cols if c in features_df.columns]
# #         features_df = features_df[feature_cols]
        
# #         instance_pct = instance_pct.merge(features_df, on='pottery_id', how='left')
# #         instance_event_counts = instance_event_counts.merge(features_df, on='pottery_id', how='left')
        
# #         for col in internal_feature_cols:
# #             if col in instance_pct.columns:
# #                 instance_pct[col] = instance_pct[col].fillna(0).astype(int)
# #                 instance_event_counts[col] = instance_event_counts[col].fillna(0).astype(int)
                
# #     return instance_event_counts, instance_pct

# # # ==========================================
# # # 2. STATISTICAL TESTS
# # # ==========================================
# # def run_chi_squared(counts_df: pd.DataFrame, group_col: str, g1_val, g2_val) -> dict:
# #     try:
# #         g1_counts = counts_df[counts_df[group_col] == g1_val][TARGET_EMOTIONS].sum()
# #         g2_counts = counts_df[counts_df[group_col] == g2_val][TARGET_EMOTIONS].sum()
        
# #         contingency_table = np.array([g1_counts.values, g2_counts.values])
# #         col_sums = contingency_table.sum(axis=0)
# #         valid_cols = col_sums > 0
# #         if valid_cols.sum() < 2:
# #             return {'chi2': np.nan, 'p_value': 'N/A', 'dof': 0, 'error': 'Not enough variance'}
            
# #         contingency_table = contingency_table[:, valid_cols]
# #         chi2, p_value_scipy, dof, expected = chi2_contingency(contingency_table)
# #         p_value_str = get_chi2_p_value(chi2, dof)
        
# #         return {'chi2': chi2, 'p_value': p_value_str, 'dof': dof, 'error': None}
# #     except Exception as e:
# #         return {'chi2': np.nan, 'p_value': 'N/A', 'dof': 0, 'error': str(e)}

# # def run_t_tests(pct_df: pd.DataFrame, group_col: str, g1_val, g2_val) -> dict:
# #     results = {}
# #     g1_data = pct_df[pct_df[group_col] == g1_val]
# #     g2_data = pct_df[pct_df[group_col] == g2_val]
    
# #     for emo in TARGET_EMOTIONS:
# #         try:
# #             v1 = g1_data[emo].values
# #             v2 = g2_data[emo].values
# #             if len(v1) < 2 or len(v2) < 2:
# #                 results[emo] = {
# #                     't_stat': np.nan, 'p_value': 'N/A',
# #                     'mean_g1': np.nanmean(v1) if len(v1) > 0 else 0,
# #                     'std_g1': np.nanstd(v1, ddof=1) if len(v1) > 1 else 0,
# #                     'mean_g2': np.nanmean(v2) if len(v2) > 0 else 0,
# #                     'std_g2': np.nanstd(v2, ddof=1) if len(v2) > 1 else 0,
# #                     'error': 'Sample size < 2'
# #                 }
# #                 continue
                
# #             t_stat, p_value = ttest_ind(v1, v2, equal_var=False, nan_policy='omit')
# #             results[emo] = {
# #                 't_stat': t_stat, 'p_value': p_value,
# #                 'mean_g1': np.nanmean(v1), 'std_g1': np.nanstd(v1, ddof=1),
# #                 'mean_g2': np.nanmean(v2), 'std_g2': np.nanstd(v2, ddof=1),
# #                 'error': None
# #             }
# #         except Exception as e:
# #             results[emo] = {
# #                 't_stat': np.nan, 'p_value': 'N/A',
# #                 'mean_g1': np.nan, 'std_g1': np.nan,
# #                 'mean_g2': np.nan, 'std_g2': np.nan,
# #                 'error': str(e)
# #             }
# #     return results

# # def run_permanova(pct_df: pd.DataFrame, group_col: str, g1_val, g2_val, n_perm=999) -> dict:
# #     try:
# #         g1_data = pct_df[pct_df[group_col] == g1_val][TARGET_EMOTIONS].values
# #         g2_data = pct_df[pct_df[group_col] == g2_val][TARGET_EMOTIONS].values
        
# #         if len(g1_data) < 2 or len(g2_data) < 2:
# #             return {'pseudo_f': np.nan, 'p_value': 'N/A', 'error': 'Sample size < 2'}
            
# #         data = np.vstack([g1_data, g2_data])
# #         groups = np.array([0] * len(g1_data) + [1] * len(g2_data))
# #         n = len(data)
        
# #         dist_matrix = squareform(pdist(data, metric='euclidean'))
# #         d_sq = dist_matrix ** 2
# #         S_T = np.sum(d_sq) / n
        
# #         def calc_sw(grps):
# #             sw = 0
# #             for g in [0, 1]:
# #                 mask = grps == g
# #                 n_g = np.sum(mask)
# #                 if n_g > 1:
# #                     d_g = d_sq[np.ix_(mask, mask)]
# #                     sw += np.sum(d_g) / n_g
# #             return sw

# #         S_W = calc_sw(groups)
# #         S_B = S_T - S_W
# #         df_b = 1
# #         df_w = n - 2
        
# #         if S_W == 0 or df_w == 0:
# #             pseudo_f = 0.0
# #         else:
# #             pseudo_f = (S_B / df_b) / (S_W / df_w)
            
# #         perm_f = np.zeros(n_perm)
# #         for i in range(n_perm):
# #             np.random.shuffle(groups)
# #             S_W_perm = calc_sw(groups)
# #             S_B_perm = S_T - S_W_perm
# #             if S_W_perm == 0:
# #                 perm_f[i] = 0.0
# #             else:
# #                 perm_f[i] = (S_B_perm / df_b) / (S_W_perm / df_w)
                
# #         p_value = (np.sum(perm_f >= pseudo_f) + 1) / (n_perm + 1)
# #         return {'pseudo_f': pseudo_f, 'p_value': p_value, 'error': None}
# #     except Exception as e:
# #         return {'pseudo_f': np.nan, 'p_value': 'N/A', 'error': str(e)}

# # # ==========================================
# # # 3. REPORTING & OUTPUT
# # # ==========================================
# # def generate_outputs(results_dict: dict, output_dir: str):
# #     os.makedirs(output_dir, exist_ok=True)
# #     chi2_rows = []
# #     ttest_rows = []
# #     permanova_rows = []
# #     report_lines = []
    
# #     report_lines.append("=" * 100)
# #     report_lines.append("STATISTICAL ANALYSIS REPORT: EMOTION COMPONENTS BY EVENT")
# #     report_lines.append("=" * 100)
# #     report_lines.append("\nAggregation Logic:")
# #     report_lines.append("1. Raw event counts are calculated for EACH SESSION and EACH OBJECT.")
# #     report_lines.append("2. These counts are converted to PERCENTAGES for each individual instance.")
# #     report_lines.append("3. T-tests and PERMANOVA compare the distributions of these instance-level percentages.")
# #     report_lines.append("\nTests performed per comparison:")
# #     report_lines.append("- Chi-Squared: Overall distribution of SUMMED raw multi-emotion event counts.")
# #     report_lines.append("- Welch's T-Test: Session-normalized percentages per individual emotion.")
# #     report_lines.append("- PERMANOVA: Multivariate emotion profile divergence via permutations.\n")
    
# #     for comp_name, tests in results_dict.items():
# #         report_lines.append("-" * 100)
# #         report_lines.append(f"COMPARISON: {comp_name}")
# #         report_lines.append("-" * 100)
        
# #         # 1. Chi-Squared
# #         chi2 = tests['chi_squared']
# #         report_lines.append("\n[1] CHI-SQUARED TEST (Summed raw multi-emotion event counts)")
# #         if chi2['error']:
# #             report_lines.append(f"    Error: {chi2['error']}")
# #             chi2_rows.append({'Comparison': comp_name, 'Chi2': np.nan, 'P_Value': 'N/A', 'DOF': 0, 'Explanation': f"Error: {chi2['error']}"})
# #         else:
# #             is_sig = False
# #             try:
# #                 p_val_float = float(chi2['p_value']) if not chi2['p_value'].startswith('<') else 0.0
# #                 is_sig = p_val_float < 0.05
# #             except:
# #                 is_sig = '<' in str(chi2['p_value'])
# #             explanation = explain_chi2(is_sig, comp_name)
# #             report_lines.append(f"    - Chi2 Statistic : {chi2['chi2']:.6f}")
# #             report_lines.append(f"    - Degrees of Freedom: {chi2['dof']}")
# #             report_lines.append(f"    - Full P-value   : {format_p(chi2['p_value'])}")
# #             report_lines.append(f"    - Interpretation : {explanation}")
# #             chi2_rows.append({'Comparison': comp_name, 'Chi2': chi2['chi2'], 'P_Value': format_p(chi2['p_value']), 'DOF': chi2['dof'], 'Explanation': explanation})
            
# #         # 2. T-Tests
# #         ttests = tests['t_tests']
# #         report_lines.append("\n[2] WELCH'S T-TESTS (Instance-level emotion percentages)")
# #         for emo, res in ttests.items():
# #             if res['error']:
# #                 report_lines.append(f"    - {emo:<12}: Error ({res['error']})")
# #                 ttest_rows.append({'Comparison': comp_name, 'Emotion': emo, 'T_Stat': np.nan, 'P_Value': 'N/A', 'Mean_G1': np.nan, 'Std_G1': np.nan, 'Mean_G2': np.nan, 'Std_G2': np.nan, 'Explanation': f"Error: {res['error']}"})
# #             else:
# #                 is_sig = False
# #                 try:
# #                     p_val_float = float(res['p_value']) if not str(res['p_value']).startswith('<') else 0.0
# #                     is_sig = p_val_float < 0.05
# #                 except:
# #                     is_sig = '<' in str(res['p_value'])
# #                 explanation = explain_ttest(is_sig, comp_name, emo, res['mean_g1'], res['mean_g2'])
# #                 sig_marker = "*" if is_sig else ""
# #                 std_g1 = res.get('std_g1', 0.0)
# #                 std_g2 = res.get('std_g2', 0.0)
# #                 report_lines.append(f"    - {emo:<12}: t={res['t_stat']:.4f}, Full P-value={format_p(res['p_value'])} {sig_marker}")
# #                 report_lines.append(f"                   (Mean G1: {res['mean_g1']:.2f}% ± {std_g1:.2f} | Mean G2: {res['mean_g2']:.2f}% ± {std_g2:.2f})")
# #                 report_lines.append(f"                   Interpretation: {explanation}")
# #                 ttest_rows.append({'Comparison': comp_name, 'Emotion': emo, 'T_Stat': res['t_stat'], 'P_Value': format_p(res['p_value']), 'Mean_G1': res['mean_g1'], 'Std_G1': res.get('std_g1', np.nan), 'Mean_G2': res['mean_g2'], 'Std_G2': res.get('std_g2', np.nan), 'Explanation': explanation})
            
# #         # 3. PERMANOVA
# #         perm = tests['permanova']
# #         report_lines.append("\n[3] PERMANOVA (Multivariate instance-level emotion profile divergence)")
# #         if perm['error']:
# #             report_lines.append(f"    Error: {perm['error']}")
# #             permanova_rows.append({'Comparison': comp_name, 'Pseudo_F': np.nan, 'P_Value': 'N/A', 'Explanation': f"Error: {perm['error']}"})
# #         else:
# #             is_sig = False
# #             try:
# #                 p_val_float = float(perm['p_value']) if not str(perm['p_value']).startswith('<') else 0.0
# #                 is_sig = p_val_float < 0.05
# #             except:
# #                 is_sig = '<' in str(perm['p_value'])
# #             explanation = explain_permanova(is_sig, comp_name)
# #             report_lines.append(f"    - Pseudo-F Statistic: {perm['pseudo_f']:.6f}")
# #             report_lines.append(f"    - Full P-value      : {format_p(perm['p_value'])}")
# #             report_lines.append(f"    - Interpretation    : {explanation}")
# #             permanova_rows.append({'Comparison': comp_name, 'Pseudo_F': perm['pseudo_f'], 'P_Value': format_p(perm['p_value']), 'Explanation': explanation})
# #         report_lines.append("\n")

# #     # ==========================================
# #     # EXECUTIVE SUMMARY: INTRA-COUNTRY DOGU DIFFERENCES
# #     # ==========================================
# #     summary_lines = []
# #     summary_lines.append("\n" + "=" * 100)
# #     summary_lines.append("EXECUTIVE SUMMARY: SIGNIFICANT INTRA-COUNTRY DOGU DIFFERENCES")
# #     summary_lines.append("=" * 100)
# #     summary_lines.append("Quick reference for which specific Dogu pairs evoke significantly different emotional responses.\n")
    
# #     found_any = False
# #     for comp_name, tests in results_dict.items():
# #         if "Within Japan" in comp_name or "Within Malaysia" in comp_name:
# #             perm = tests['permanova']
# #             perm_sig = False
# #             if not perm['error']:
# #                 try:
# #                     p_val = float(perm['p_value']) if not str(perm['p_value']).startswith('<') else 0.0
# #                     perm_sig = p_val < 0.05
# #                 except:
# #                     perm_sig = '<' in str(perm['p_value'])
                    
# #             sig_emos = []
# #             for emo, res in tests['t_tests'].items():
# #                 if not res['error']:
# #                     try:
# #                         p_val = float(res['p_value']) if not str(res['p_value']).startswith('<') else 0.0
# #                         if p_val < 0.05: sig_emos.append(emo)
# #                     except:
# #                         if '<' in str(res['p_value']): sig_emos.append(emo)
                        
# #             if perm_sig or sig_emos:
# #                 found_any = True
# #                 summary_lines.append(f"* {comp_name}")
# #                 if perm_sig:
# #                     summary_lines.append(f"  -> Overall Profile (PERMANOVA): SIGNIFICANT (p={format_p(perm['p_value'])})")
# #                 if sig_emos:
# #                     summary_lines.append(f"  -> Specific Emotions (T-tests): {', '.join(sig_emos)} are significantly different.")
# #                 summary_lines.append("")
                
# #     if not found_any:
# #         summary_lines.append("No statistically significant differences found between any individual Dogu pairs within the same country.")
        
# #     report_lines.extend(summary_lines)

# #     # Save Text Report
# #     txt_path = os.path.join(output_dir, "statistical_analysis_report.txt")
# #     with open(txt_path, "w", encoding="utf-8") as f:
# #         f.write("\n".join(report_lines))
# #     print(f"✓ Descriptive report saved to: {txt_path}")
    
# #     # Save CSVs
# #     pd.DataFrame(chi2_rows).to_csv(os.path.join(output_dir, "chi_squared_results.csv"), index=False)
# #     pd.DataFrame(ttest_rows).to_csv(os.path.join(output_dir, "t_test_results.csv"), index=False)
# #     pd.DataFrame(permanova_rows).to_csv(os.path.join(output_dir, "permanova_results.csv"), index=False)
# #     print("✓ Statistical results saved to CSV files with full precision p-values and explanations.")

# # # ==========================================
# # # 4. 3D ANALYSIS AND VISUALIZATION
# # # ==========================================
# # def run_3d_analysis_and_explanations(results_dict: dict, instance_pct: pd.DataFrame, comparison_configs: list, output_dir: str):
# #     """
# #     Generates 3D visualizations for each comparison and saves detailed explanations to TXT files.
# #     Filters out individual Dogu comparisons to focus on Pottery/Dogu cross-country and Features.
# #     """
# #     os.makedirs(output_dir, exist_ok=True)
# #     plots_dir = os.path.join(output_dir, "3d_plots")
# #     explanations_dir = os.path.join(output_dir, "3d_explanations")
# #     os.makedirs(plots_dir, exist_ok=True)
# #     os.makedirs(explanations_dir, exist_ok=True)
    
# #     print("\n--- Generating 3D Visualizations and Detailed Explanations ---")
# #     emotion_cols = TARGET_EMOTIONS
    
# #     for config in comparison_configs:
# #         comp_name = config["name"]
        
# #         # Skip individual Dogu comparisons for 3D plots as requested
# #         if "Itemized Dogu" in comp_name or "Pairwise Dogu" in comp_name:
# #             continue
            
# #         if comp_name not in results_dict:
# #             continue
            
# #         tests = results_dict[comp_name]
# #         print(f"  Processing 3D visualization for: {comp_name} ...")
        
# #         # Create safe filename
# #         safe_name = comp_name.replace(" ", "_").replace(":", "").replace("/", "_").replace("(", "").replace(")", "")
        
# #         # Generate explanation text
# #         explanation_text = generate_3d_analysis_explanation(comp_name, tests, instance_pct, emotion_cols)
# #         explanation_file = os.path.join(explanations_dir, f"3d_analysis_{safe_name}.txt")
# #         with open(explanation_file, 'w', encoding='utf-8') as f:
# #             f.write(explanation_text)
            
# #         # Generate and save 3D plot
# #         try:
# #             generate_3d_plot_for_comparison(comp_name, instance_pct, emotion_cols, config, plots_dir, safe_name)
# #         except Exception as e:
# #             print(f"   Could not generate 3D plot for {comp_name}: {e}")
            
# #     print("✓ 3D analysis complete.")

# # def generate_3d_analysis_explanation(comp_name: str, tests: dict, instance_pct: pd.DataFrame, emotion_cols: list) -> str:
# #     """
# #     Generates a detailed explanation of the 3D analysis for a specific comparison.
# #     """
# #     lines = []
# #     lines.append("=" * 100)
# #     lines.append(f"3D ANALYSIS EXPLANATION: {comp_name}")
# #     lines.append("=" * 100)
# #     lines.append("")
    
# #     # Statistical Methods Section
# #     lines.append("-" * 100)
# #     lines.append("STATISTICAL METHODS AND ANALYSIS")
# #     lines.append("-" * 100)
# #     lines.append("")
    
# #     lines.append("1. PRINCIPAL COMPONENT ANALYSIS (PCA) FOR 3D VISUALIZATION")
# #     lines.append("   Purpose: Reduce 5-dimensional emotion data to 3 dimensions for visualization")
# #     lines.append("   Method: Linear dimensionality reduction that maximizes variance")
# #     lines.append("   - PC1: Captures the direction of maximum variance in emotion profiles")
# #     lines.append("   - PC2: Captures the second highest variance, orthogonal to PC1")
# #     lines.append("   - PC3: Captures the third highest variance, orthogonal to PC1 and PC2")
# #     lines.append("   Data Standardization: All emotion percentages are standardized (z-score)")
# #     lines.append("   Formula: z = (x - μ) / σ")
# #     lines.append("")
    
# #     lines.append("2. STATISTICAL TESTS PERFORMED")
# #     lines.append("   a) Chi-Squared Test:")
# #     chi2 = tests['chi_squared']
# #     if not chi2['error']:
# #         lines.append(f"      - Statistic: χ² = {chi2['chi2']:.4f}")
# #         lines.append(f"      - Degrees of Freedom: {chi2['dof']}")
# #         lines.append(f"      - P-value: {chi2['p_value']}")
# #         lines.append(f"      - Purpose: Tests if overall emotion distribution differs between groups")
# #     else:
# #         lines.append(f"      - Error: {chi2['error']}")
# #     lines.append("")
    
# #     lines.append("   b) Welch's T-Tests (Individual Emotions):")
# #     for emo, res in tests['t_tests'].items():
# #         if not res['error']:
# #             sig_marker = "*" if float(res['p_value']) < 0.05 else ""
# #             lines.append(f"      - {emo}:")
# #             lines.append(f"        t-statistic: {res['t_stat']:.4f}")
# #             lines.append(f"        P-value: {res['p_value']} {sig_marker}")
# #             lines.append(f"        Mean Group 1: {res['mean_g1']:.2f}% ± {res.get('std_g1', 0):.2f}")
# #             lines.append(f"        Mean Group 2: {res['mean_g2']:.2f}% ± {res.get('std_g2', 0):.2f}")
# #     lines.append("")
    
# #     lines.append("   c) PERMANOVA (Permutational Multivariate Analysis of Variance):")
# #     perm = tests['permanova']
# #     if not perm['error']:
# #         lines.append(f"      - Pseudo-F Statistic: {perm['pseudo_f']:.4f}")
# #         lines.append(f"      - P-value: {perm['p_value']}")
# #         lines.append(f"      - Permutations: 999")
# #         lines.append(f"      - Distance Metric: Euclidean distance")
# #         lines.append(f"      - Purpose: Tests if multivariate emotion profiles differ between groups")
# #     else:
# #         lines.append(f"      - Error: {perm['error']}")
# #     lines.append("")
    
# #     # 3D Visualization Interpretation
# #     lines.append("-" * 100)
# #     lines.append("3D VISUALIZATION INTERPRETATION")
# #     lines.append("-" * 100)
# #     lines.append("")
# #     lines.append("The 3D scatter plot shows each session as a point in 3D space based on PCA.")
# #     lines.append("Points closer together have similar emotional response profiles.")
# #     lines.append("Points farther apart have different emotional response profiles.")
# #     lines.append("")
# #     lines.append("What to look for:")
# #     lines.append("  - Clustering: Groups that form distinct clusters indicate different emotion patterns")
# #     lines.append("  - Separation: Clear separation along PC1, PC2, or PC3 indicates strong differences")
# #     lines.append("  - Overlap: Overlapping points suggest similar emotional responses")
# #     lines.append("  - Outliers: Points far from their group may indicate unusual responses")
# #     lines.append("")
    
# #     # Data Summary
# #     lines.append("-" * 100)
# #     lines.append("DATA SUMMARY & GROUP MEANS")
# #     lines.append("-" * 100)
# #     lines.append("")
    
# #     # Extract group information
# #     if "Japan" in comp_name and "Malaysia" in comp_name:
# #         lines.append("Comparison Type: Cross-Cultural (Japan vs Malaysia)")
# #     elif "Pottery" in comp_name and "Dogu" in comp_name:
# #         lines.append("Comparison Type: Artifact Type (Pottery vs Dogu)")
# #     elif "vs" in comp_name:
# #         lines.append("Comparison Type: Pairwise Comparison")
# #     else:
# #         lines.append("Comparison Type: Feature-based Analysis")
# #     lines.append("")
    
# #     lines.append("Emotion Dimensions Analyzed:")
# #     for emo in emotion_cols:
# #         lines.append(f"  - {emo}")
# #     lines.append("")
    
# #     # Calculate and display means and stds for the groups in this comparison
# #     lines.append("Group Statistics (from T-tests):")
# #     for emo, res in tests['t_tests'].items():
# #         if not res['error']:
# #             lines.append(f"  {emo}:")
# #             lines.append(f"    Group 1 Mean: {res['mean_g1']:.2f}% (Std: {res.get('std_g1', 0):.2f})")
# #             lines.append(f"    Group 2 Mean: {res['mean_g2']:.2f}% (Std: {res.get('std_g2', 0):.2f})")
# #     lines.append("")
    
# #     # Statistical Significance Summary
# #     lines.append("-" * 100)
# #     lines.append("STATISTICAL SIGNIFICANCE SUMMARY")
# #     lines.append("-" * 100)
# #     lines.append("")
    
# #     if not perm['error']:
# #         try:
# #             p_val = float(perm['p_value']) if not str(perm['p_value']).startswith('<') else 0.0
# #             is_sig = p_val < 0.05
# #             lines.append(f"Overall Multivariate Difference (PERMANOVA):")
# #             lines.append(f"  - Significant: {'YES' if is_sig else 'NO'} (p = {perm['p_value']})")
# #             if is_sig:
# #                 lines.append(f"  - Interpretation: The groups have SIGNIFICANTLY different emotion profiles")
# #             else:
# #                 lines.append(f"  - Interpretation: No significant difference in overall emotion profiles")
# #         except:
# #             lines.append(f"Overall Multivariate Difference: {perm['p_value']}")
# #     lines.append("")
    
# #     sig_emotions = []
# #     for emo, res in tests['t_tests'].items():
# #         if not res['error']:
# #             try:
# #                 p_val = float(res['p_value']) if not str(res['p_value']).startswith('<') else 0.0
# #                 if p_val < 0.05:
# #                     sig_emotions.append(emo)
# #             except:
# #                 if '<' in str(res['p_value']):
# #                     sig_emotions.append(emo)
    
# #     if sig_emotions:
# #         lines.append(f"Significantly Different Emotions (T-tests, p<0.05):")
# #         for emo in sig_emotions:
# #             lines.append(f"  - {emo}")
# #     else:
# #         lines.append("No individual emotions showed significant differences (p<0.05)")
# #     lines.append("")
    
# #     # Visualization Notes
# #     lines.append("-" * 100)
# #     lines.append("VISUALIZATION NOTES")
# #     lines.append("-" * 100)
# #     lines.append("")
# #     lines.append("The 3D plot uses the following conventions:")
# #     lines.append("  - X-axis: Principal Component 1 (PC1)")
# #     lines.append("  - Y-axis: Principal Component 2 (PC2)")
# #     lines.append("  - Z-axis: Principal Component 3 (PC3)")
# #     lines.append("  - Colors: Different colors represent different groups/conditions")
# #     lines.append("  - Point size: Uniform size for clarity")
# #     lines.append("  - Transparency: Alpha=0.6 to show overlapping points")
# #     lines.append("")
# #     lines.append("Percentage of variance explained by each PC is shown in the plot title.")
# #     lines.append("Higher variance explained means the 3D plot better represents the original data.")
# #     lines.append("")
    
# #     lines.append("=" * 100)
# #     lines.append("END OF 3D ANALYSIS EXPLANATION")
# #     lines.append("=" * 100)
    
# #     return "\n".join(lines)

# # def generate_3d_plot_for_comparison(comp_name: str, instance_pct: pd.DataFrame, emotion_cols: list, config: dict, plots_dir: str, safe_name: str):
# #     """
# #     Generates a 3D PCA plot for a specific comparison.
# #     """
# #     # Prepare data
# #     group_col = config["group_col"]
# #     g1_val = config["g1"]
# #     g2_val = config["g2"]
    
# #     # Filter data to only include the two groups
# #     mask = (instance_pct[group_col] == g1_val) | (instance_pct[group_col] == g2_val)
# #     data = instance_pct.loc[mask, emotion_cols].copy()
# #     data = data.dropna()
    
# #     if len(data) < 3:
# #         raise ValueError("Not enough data points for 3D visualization")
        
# #     # Standardize the data
# #     scaler = StandardScaler()
# #     data_scaled = scaler.fit_transform(data)
    
# #     # Perform PCA
# #     pca = PCA(n_components=3)
# #     principal_components = pca.fit_transform(data_scaled)
    
# #     # Create 3D plot
# #     fig = plt.figure(figsize=(12, 10))
# #     ax = fig.add_subplot(111, projection='3d')
    
# #     # Determine colors
# #     groups = instance_pct.loc[mask, group_col]
# #     colors = ['red' if g == g1_val else 'blue' for g in groups]
# #     labels = [str(g1_val), str(g2_val)]
    
# #     # Plot
# #     scatter = ax.scatter(principal_components[:, 0], 
# #                         principal_components[:, 1], 
# #                         principal_components[:, 2], 
# #                         c=colors, alpha=0.6, s=50)
    
# #     # Add labels
# #     var_explained = pca.explained_variance_ratio_
# #     ax.set_xlabel(f'PC1 ({var_explained[0]:.2%} variance)', fontsize=10, labelpad=10)
# #     ax.set_ylabel(f'PC2 ({var_explained[1]:.2%} variance)', fontsize=10, labelpad=10)
# #     ax.set_zlabel(f'PC3 ({var_explained[2]:.2%} variance)', fontsize=10, labelpad=10)
    
# #     # Title
# #     title = f"3D PCA Visualization\n{comp_name}"
# #     ax.set_title(title, fontsize=12, pad=20)
    
# #     # Add legend
# #     if len(labels) > 1:
# #         legend_elements = [Patch(facecolor='red', label=labels[0]), 
# #                           Patch(facecolor='blue', label=labels[1])]
# #         ax.legend(handles=legend_elements, loc='upper left')
    
# #     # Adjust view angle
# #     ax.view_init(elev=20, azim=45)
    
# #     # Save
# #     plt.tight_layout()
# #     plot_file = os.path.join(plots_dir, f"3d_plot_{safe_name}.png")
# #     plt.savefig(plot_file, dpi=300, bbox_inches='tight')
# #     plt.close()

# # # ==========================================
# # # 5. MAIN EXECUTION
# # # ==========================================
# # def get_comparisons(df: pd.DataFrame) -> list:
# #     """Generates the list of comparison configurations to run."""
# #     configs = [
# #         {"name": "Japan Pottery vs Malaysia Pottery", "mask": (df['country'].isin(['Japan', 'Malaysia'])) & (df['artifact_type'] == 'Pottery'), "group_col": "country", "g1": "Japan", "g2": "Malaysia"},
# #         {"name": "Japan Dogu vs Malaysia Dogu (Combined)", "mask": (df['country'].isin(['Japan', 'Malaysia'])) & (df['artifact_type'] == 'Dogu'), "group_col": "country", "g1": "Japan", "g2": "Malaysia"},
# #         {"name": "Japan: Pottery vs Dogu", "mask": df['country'] == 'Japan', "group_col": "artifact_type", "g1": "Pottery", "g2": "Dogu"},
# #         {"name": "Malaysia: Pottery vs Dogu", "mask": df['country'] == 'Malaysia', "group_col": "artifact_type", "g1": "Pottery", "g2": "Dogu"}
# #     ]
    
# #     # Add Itemized Dogu Comparisons (Japan vs Malaysia for each of the 8 Dogus)
# #     for dogu_prefix in DOGU_PREFIXES:
# #         configs.append({"name": f"Itemized Dogu: {dogu_prefix} (Japan vs Malaysia)", "mask": df['pottery_id'].str.startswith(dogu_prefix), "group_col": "country", "g1": "Japan", "g2": "Malaysia"})
    
# #     # NEW: INTRA-COUNTRY DOGU PAIRWISE COMPARISONS
# #     for country in ['Japan', 'Malaysia']:
# #         country_dogu_mask = (df['country'] == country) & (df['artifact_type'] == 'Dogu')
# #         dogu_ids = df[country_dogu_mask]['pottery_id'].unique()
# #         if len(dogu_ids) > 1:
# #             # Generate all unique pairwise combinations (e.g., 8 Dogus = 28 pairs)
# #             for d1, d2 in itertools.combinations(dogu_ids, 2):
# #                 configs.append({"name": f"Within {country}: {d1} vs {d2} (Pairwise Dogu)", "mask": (df['country'] == country) & (df['artifact_type'] == 'Dogu') & (df['pottery_id'].isin([d1, d2])), "group_col": "pottery_id", "g1": d1, "g2": d2})
    
# #     # Add Feature Comparisons dynamically
# #     for col_name, display_name in FEATURE_DISPLAY_NAMES.items():
# #         if col_name in df.columns:
# #             configs.append({"name": f"Feature: {display_name} (Present vs Absent) - Japan", "mask": df['country'] == 'Japan', "group_col": col_name, "g1": 1, "g2": 0})
# #             configs.append({"name": f"Feature: {display_name} (Present vs Absent) - Malaysia", "mask": df['country'] == 'Malaysia', "group_col": col_name, "g1": 1, "g2": 0})
# #             configs.append({"name": f"Feature: {display_name} (Japan Present vs Malaysia Present)", "mask": (df['country'].isin(['Japan', 'Malaysia'])) & (df[col_name] == 1), "group_col": "country", "g1": "Japan", "g2": "Malaysia"})
            
# #     return configs

# # def main():
# #     parser = argparse.ArgumentParser(description="Statistical analysis of emotion components by event & features.")
# #     parser.add_argument('--jp_dir', type=str, default=r"D:\storage\jomon_kaen\jomon_kaen_dataset\japan", help="Path to Japan dataset")
# #     parser.add_argument('--my_dir', type=str, default=r"D:\storage\jomon_kaen\jomon_kaen_dataset\malaysia", help="Path to Malaysia dataset")
# #     parser.add_argument('--features_csv', type=str, default="./src/DS_Labels_Cleaned.csv", help="Path to pottery features CSV")
# #     parser.add_argument('--output_dir', type=str, default="./stats_output", help="Output directory")
# #     parser.add_argument('--permutations', type=int, default=999, help="Number of permutations for PERMANOVA")
# #     args = parser.parse_args()
    
# #     # 1. Load & Prepare Data
# #     df_jp = load_qa_data(args.jp_dir, 'Japan')
# #     df_my = load_qa_data(args.my_dir, 'Malaysia')
# #     if df_jp.empty and df_my.empty:
# #         print("Error: No data loaded. Please check paths.")
# #         return
        
# #     instance_event_counts, instance_pct = prepare_and_aggregate(df_jp, df_my, args.features_csv)
# #     comparison_configs = get_comparisons(instance_pct)
    
# #     # 2. Run Tests
# #     results_dict = {}
# #     for config in comparison_configs:
# #         comp_name = config["name"]
# #         mask = config["mask"]
# #         group_col = config["group_col"]
# #         g1 = config["g1"]
# #         g2 = config["g2"]
# #         print(f"Running tests for: {comp_name} ...")
# #         valid_counts = instance_event_counts[mask]
# #         valid_pct = instance_pct[mask]
# #         if len(valid_pct[group_col].unique()) < 2:
# #             print(f"  -> Skipping {comp_name} (Not enough groups found in data)")
# #             continue
# #         results_dict[comp_name] = {
# #             'chi_squared': run_chi_squared(valid_counts, group_col, g1, g2),
# #             't_tests': run_t_tests(valid_pct, group_col, g1, g2),
# #             'permanova': run_permanova(valid_pct, group_col, g1, g2, n_perm=args.permutations)
# #         }
        
# #     # 3. Generate Standard Outputs
# #     generate_outputs(results_dict, args.output_dir)
    
# #     # 4. Generate 3D Analysis
# #     run_3d_analysis_and_explanations(results_dict, instance_pct, comparison_configs, args.output_dir)
    
# #     print("\nAnalysis complete!")

# # if __name__ == "__main__":
# #     main()




# import os
# import sys
# import pandas as pd
# import numpy as np
# from pathlib import Path
# from scipy import stats
# import warnings
# warnings.filterwarnings('ignore')

# # ==========================================
# # CONFIGURATION
# # ==========================================
# DATASET_ROOT_MALAYSIA = r"D:\storage\jomon_kaen\jomon_kaen_dataset\malaysia"
# DATASET_ROOT_JAPAN = r"D:\storage\jomon_kaen\jomon_kaen_dataset\japan"
# OUTPUT_DIR = "anova_statistical_analysis"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# DOGU_PREFIXES = [
#     'IN0295', 'IN0306', 'MH0037', 'NM0239', 
#     'NZ0001', 'SK0035', 'TK0020', 'UD0028'
# ]

# EMOTION_MAP = {
#     "Interesting and attentional shape": "Interesting", 
#     "Beautiful and artistic": "Beautiful",
#     "Strange and incomprehensible": "Strange", 
#     "Creepy / unsettling / scary": "Scary",
#     "Feel nothing": "Feel nothing", 
#     "面白い・気になる形だ": "Interesting", 
#     "美しい・芸術的だ": "Beautiful",
#     "不思議・意味不明": "Strange", 
#     "不気味・不安・怖い": "Scary",
#     "何も感じない": "Feel nothing"
# }

# EMOTION_COLS = ["Interesting", "Beautiful", "Strange", "Scary", "Feel nothing"]
# TARGET_LANGUAGES = ['ENGLISH', 'MALAY', 'CHINESE', 'JAPANESE']

# # ==========================================
# # 1. DATA LOADING
# # ==========================================
# def load_multilingual_dataset(root_dirs):
#     """Load QA data from multiple directories."""
#     qa_records = []
    
#     for root_dir in root_dirs:
#         root_path = Path(root_dir)
#         if not root_path.exists():
#             print(f"Warning: Directory not found: {root_dir}")
#             continue

#         print(f"Loading data from {root_dir}...")

#         for group_path in root_path.iterdir():
#             if not group_path.is_dir(): continue
#             for session_path in group_path.iterdir():
#                 if not session_path.is_dir(): continue

#                 language = None
#                 lang_file = session_path / 'language.txt'
                
#                 if lang_file.exists():
#                     try:
#                         text = lang_file.read_text(encoding='utf-8').strip().upper()
#                         if 'ENGLISH' in text or 'EN ' in text: language = 'ENGLISH'
#                         elif 'MALAY' in text or 'BM ' in text: language = 'MALAY'
#                         elif 'CHINESE' in text or 'MANDARIN' in text: language = 'CHINESE'
#                         elif 'JAPANESE' in text or 'JP ' in text: language = 'JAPANESE'
#                     except:
#                         pass

#                 if language is None and "japan" in str(root_path).lower():
#                     language = 'JAPANESE'

#                 if language not in TARGET_LANGUAGES: continue

#                 for pottery_path in session_path.iterdir():
#                     if not pottery_path.is_dir(): continue
#                     pottery_id = pottery_path.name

#                     qa_file = pottery_path / "qa_corrected.csv"
#                     if qa_file.exists():
#                         try:
#                             df_temp = pd.read_csv(qa_file, encoding='utf-8')
#                             df_temp['timestamp'] = pd.to_numeric(df_temp['timestamp'], errors='coerce')
#                             df_temp.dropna(subset=['timestamp'], inplace=True)
#                             df_temp['pottery_id'] = pottery_id
#                             df_temp['session_id'] = session_path.name
#                             df_temp['Language'] = language
#                             qa_records.append(df_temp)
#                         except Exception as e:
#                             print(f"Error loading {qa_file}: {e}")

#     df_qa = pd.concat(qa_records, ignore_index=True) if qa_records else pd.DataFrame()
#     print(f"Loaded {len(df_qa)} QA interactions.")
#     return df_qa

# # ==========================================
# # 2. CALCULATE NORMALIZED METRICS
# # ==========================================
# def calculate_metrics(df_qa):
#     print("Calculating session duration blocks and normalized metrics...")
    
#     df = df_qa.copy()
#     df['short_answer'] = df['answer'].astype(str).str.strip().map(EMOTION_MAP)
#     df.dropna(subset=['short_answer', 'pottery_id', 'session_id'], inplace=True)
#     df.sort_values(by=['pottery_id', 'session_id', 'timestamp'], inplace=True)
    
#     df['time_diff'] = df.groupby(['pottery_id', 'session_id'])['timestamp'].diff()
#     emotion_changed = df['short_answer'] != df.groupby(['pottery_id', 'session_id'])['short_answer'].shift()
#     time_gap_exceeded = df['time_diff'] > 0.05
#     df['block_id'] = (emotion_changed | time_gap_exceeded).cumsum()
    
#     blocks = df.groupby(['pottery_id', 'session_id', 'Language', 'block_id']).agg(
#         emotion=('short_answer', 'first'),
#         start_time=('timestamp', 'min'),
#         end_time=('timestamp', 'max'),
#         event_count=('timestamp', 'count')
#     ).reset_index()
    
#     blocks['duration'] = (blocks['end_time'] - blocks['start_time']) * 1000 
    
#     # Session Level Metrics
#     session_events = blocks.groupby(['pottery_id', 'session_id', 'Language', 'emotion'])['event_count'].sum().unstack(fill_value=0).reset_index()
#     session_duration = blocks.groupby(['pottery_id', 'session_id', 'Language', 'emotion'])['duration'].sum().unstack(fill_value=0).reset_index()
    
#     for emo in EMOTION_COLS:
#         if emo not in session_events.columns: session_events[emo] = 0
#         if emo not in session_duration.columns: session_duration[emo] = 0
            
#     session_events['total'] = session_events[EMOTION_COLS].sum(axis=1)
#     session_duration['total'] = session_duration[EMOTION_COLS].sum(axis=1)
    
#     for emo in EMOTION_COLS:
#         session_events[emo] = session_events[emo] / session_events['total'].replace(0, np.nan)
#         session_duration[emo] = session_duration[emo] / session_duration['total'].replace(0, np.nan)
        
#     session_events.fillna(0, inplace=True)
#     session_duration.fillna(0, inplace=True)
    
#     # Pottery Level (93 Items Aggregation)
#     pottery_events = session_events.groupby('pottery_id')[EMOTION_COLS].mean().reset_index()
#     pottery_duration = session_duration.groupby('pottery_id')[EMOTION_COLS].mean().reset_index()
    
#     # Formatting to match requested structure
#     pottery_events.rename(columns={'pottery_id': 'Pottery_ID'}, inplace=True)
#     pottery_duration.rename(columns={'pottery_id': 'Pottery_ID'}, inplace=True)
    
#     pottery_events['Samples'] = range(1, len(pottery_events) + 1)
#     pottery_duration['Samples'] = range(1, len(pottery_duration) + 1)
    
#     pottery_events['Type'] = pottery_events['Pottery_ID'].apply(lambda x: 'Dogu' if any(p in str(x) for p in DOGU_PREFIXES) else 'Pottery')
#     pottery_duration['Type'] = pottery_duration['Pottery_ID'].apply(lambda x: 'Dogu' if any(p in str(x) for p in DOGU_PREFIXES) else 'Pottery')
    
#     # Reorder columns
#     cols = ['Samples', 'Pottery_ID', 'Type'] + EMOTION_COLS
    
#     return pottery_events[cols], pottery_duration[cols], session_events, session_duration

# # ==========================================
# # 3. FEATURES & TYPOLOGY INTEGRATION
# # ==========================================
# def get_features_path():
#     """Finds the correct labels file."""
#     possible_paths = [
#         "./src/DS_Labels_Cleaned.xlsx"
#     ]
#     for p in possible_paths:
#         if os.path.exists(p):
#             return p
#     return None

# def attach_features(pottery_df, features_file):
#     if features_file is None or not os.path.exists(features_file):
#         print("Warning: Features file not found! Please check path.")
#         return pottery_df.copy(), [], []
        
#     try:
#         if features_file.endswith('.xlsx'):
#             feat_df = pd.read_excel(features_file)
#         else:
#             try:
#                 feat_df = pd.read_csv(features_file, encoding='utf-8-sig')
#             except:
#                 feat_df = pd.read_csv(features_file, encoding='shift_jis')
#     except Exception as e:
#         print(f"Error loading features file: {e}")
#         return pottery_df.copy(), [], []
        
#     # Standardize Pottery ID
#     feat_df['Pottery_ID'] = feat_df.iloc[:, 0].astype(str).str.replace('.ply', '', regex=False)
    
#     # Extract Features (Binary HAS_ columns)
#     feature_cols = [c for c in feat_df.columns if c.startswith('HAS_')]
    
#     # Extract Typology (Binary SHAPE_TYPE_ columns)
#     # Exclude NAN to focus only on actual typologies
#     shape_cols = [c for c in feat_df.columns if c.startswith('SHAPE_TYPE_') and 'NAN' not in c]
    
#     # Merge with 93 items dataframe
#     merged_df = pottery_df.merge(feat_df[['Pottery_ID'] + shape_cols + feature_cols], on='Pottery_ID', how='left')
    
#     # Ensure binary format (convert TRUE/FALSE/strings to 1/0)
#     for col in shape_cols + feature_cols:
#         merged_df[col] = pd.to_numeric(merged_df[col].replace({'TRUE': 1, 'FALSE': 0, True: 1, False: 0}), errors='coerce').fillna(0)
        
#     return merged_df, feature_cols, shape_cols

# # ==========================================
# # 4. STATISTICAL TESTS (ANOVA & KRUSKAL)
# # ==========================================
# def run_stats_on_groups(df, group_col, metric_cols):
#     """Runs BOTH Standard ANOVA and Kruskal-Wallis Tests."""
#     results = []
    
#     for emo in metric_cols:
#         # Extract valid groups where values exist
#         groups = [group[emo].dropna().values for name, group in df.groupby(group_col) if len(group[emo].dropna()) > 0]
#         valid_groups = [g for g in groups if len(g) >= 2] # Need at least 2 items per group for variance
        
#         if len(valid_groups) >= 2:
#             # 1. Standard ANOVA
#             try:
#                 f_stat, f_p = stats.f_oneway(*valid_groups)
#             except:
#                 f_stat, f_p = np.nan, np.nan
                
#             # 2. Kruskal-Wallis (Non-parametric)
#             try:
#                 h_stat, h_p = stats.kruskal(*valid_groups)
#             except:
#                 h_stat, h_p = np.nan, np.nan
                
#             results.append({
#                 'Emotion': emo,
#                 'ANOVA_F_Stat': f_stat,
#                 'ANOVA_p_value': f_p,
#                 'ANOVA_Significant': f_p < 0.05 if pd.notna(f_p) else False,
#                 'Kruskal_H_Stat': h_stat,
#                 'Kruskal_p_value': h_p,
#                 'Kruskal_Significant': h_p < 0.05 if pd.notna(h_p) else False
#             })
            
#     return pd.DataFrame(results)

# # ==========================================
# # MAIN EXECUTION
# # ==========================================
# def main():
#     print("=" * 80)
#     print("STATISTICAL ANALYSIS: ANOVA & KRUSKAL-WALLIS")
#     print("=" * 80)
    
#     df_qa = load_multilingual_dataset([DATASET_ROOT_MALAYSIA, DATASET_ROOT_JAPAN])
#     if df_qa.empty:
#         print("Error: No data loaded.")
#         return
        
#     # Calculate 93-Item Metrics and Session Metrics
#     pot_events, pot_dur, sess_events, sess_dur = calculate_metrics(df_qa)
    
#     # Locate features CSV safely
#     features_file = get_features_path()
#     if features_file:
#         print(f"Loading Features and Typologies from: {features_file}")
    
#     # Attach features to 93-item sets
#     pot_events_feat, feature_cols, shape_cols = attach_features(pot_events, features_file)
#     pot_dur_feat, _, _ = attach_features(pot_dur, features_file)
    
#     # ---------------------------------------------------------
#     # 2-1: Artifact Type (Pottery vs Dogu - 93 items level)
#     # ---------------------------------------------------------
#     print("Running 2-1: Artifact Type Analysis...")
#     a21_e = run_stats_on_groups(pot_events_feat, 'Type', EMOTION_COLS)
#     a21_d = run_stats_on_groups(pot_dur_feat, 'Type', EMOTION_COLS)
    
#     # ---------------------------------------------------------
#     # 2-2: Typologies (Loop through all SHAPE_TYPE_ separately)
#     # ---------------------------------------------------------
#     print("Running 2-2: Typology Analysis (Itemized separately)...")
#     a22_e_list, a22_d_list = [], []
#     for shape in shape_cols:
#         res_e = run_stats_on_groups(pot_events_feat, shape, EMOTION_COLS)
#         res_d = run_stats_on_groups(pot_dur_feat, shape, EMOTION_COLS)
        
#         # Add the Typology name directly in the output dataframe
#         if not res_e.empty:
#             res_e.insert(0, 'Typology', shape.replace('SHAPE_TYPE_', ''))
#             a22_e_list.append(res_e)
#         if not res_d.empty:
#             res_d.insert(0, 'Typology', shape.replace('SHAPE_TYPE_', ''))
#             a22_d_list.append(res_d)
            
#     a22_e = pd.concat(a22_e_list, ignore_index=True) if a22_e_list else pd.DataFrame()
#     a22_d = pd.concat(a22_d_list, ignore_index=True) if a22_d_list else pd.DataFrame()
    
#     # ---------------------------------------------------------
#     # 2-3: Languages (Session level)
#     # ---------------------------------------------------------
#     print("Running 2-3: Language Analysis...")
#     a23_e = run_stats_on_groups(sess_events, 'Language', EMOTION_COLS)
#     a23_d = run_stats_on_groups(sess_dur, 'Language', EMOTION_COLS)
    
#     # ---------------------------------------------------------
#     # 2-4: Features (Loop through all HAS_ features - 93 items level)
#     # ---------------------------------------------------------
#     print("Running 2-4: Feature Analysis (Itemized separately)...")
#     a24_e_list, a24_d_list = [], []
#     for feat in feature_cols:
#         res_e = run_stats_on_groups(pot_events_feat, feat, EMOTION_COLS)
#         res_d = run_stats_on_groups(pot_dur_feat, feat, EMOTION_COLS)
        
#         # Add the Feature name directly in the output dataframe
#         if not res_e.empty:
#             res_e.insert(0, 'Feature', feat)
#             a24_e_list.append(res_e)
#         if not res_d.empty:
#             res_d.insert(0, 'Feature', feat)
#             a24_d_list.append(res_d)
            
#     a24_e = pd.concat(a24_e_list, ignore_index=True) if a24_e_list else pd.DataFrame()
#     a24_d = pd.concat(a24_d_list, ignore_index=True) if a24_d_list else pd.DataFrame()

#     # ---------------------------------------------------------
#     # EXCEL EXPORT
#     # ---------------------------------------------------------
#     out_file = os.path.join(OUTPUT_DIR, "ANOVA_and_Kruskal_Analysis.xlsx")
#     print(f"\nWriting to Excel: {out_file}...")
    
#     with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
#         # RAW DATA (93 ITEMS)
#         pot_events.to_excel(writer, sheet_name='1-1_Norm_Events', index=False)
#         pot_dur.to_excel(writer, sheet_name='1-2_Norm_Duration', index=False)
        
#         # 2-1 ARTIFACT TYPE STATS (Pottery vs Dogu)
#         if not a21_e.empty: a21_e.to_excel(writer, sheet_name='2-1_ArtType_Stats_E', index=False)
#         if not a21_d.empty: a21_d.to_excel(writer, sheet_name='2-1_ArtType_Stats_D', index=False)
        
#         # 2-2 TYPOLOGY STATS (Each Typology separated)
#         if not a22_e.empty: a22_e.to_excel(writer, sheet_name='2-2_Typology_Stats_E', index=False)
#         if not a22_d.empty: a22_d.to_excel(writer, sheet_name='2-2_Typology_Stats_D', index=False)
            
#         # 2-3 LANGUAGE STATS
#         if not a23_e.empty: a23_e.to_excel(writer, sheet_name='2-3_Language_Stats_E', index=False)
#         if not a23_d.empty: a23_d.to_excel(writer, sheet_name='2-3_Language_Stats_D', index=False)
            
#         # 2-4 FEATURE STATS (Each Feature separated)
#         if not a24_e.empty: a24_e.to_excel(writer, sheet_name='2-4_Features_Stats_E', index=False)
#         if not a24_d.empty: a24_d.to_excel(writer, sheet_name='2-4_Features_Stats_D', index=False)

#     print("✓ Analysis successfully exported.")
#     print("\nSHEETS CREATED:")
#     print(" - 1-1_Norm_Events       (93 items, Event Data)")
#     print(" - 1-2_Norm_Duration     (93 items, Duration Data)")
#     print(" - 2-1_ArtType_Stats_E/D (Pottery vs Dogu: ANOVA & Kruskal)")
#     print(" - 2-2_Typology_Stats_E/D(Typologies itemized separately: ANOVA & Kruskal)")
#     print(" - 2-3_Language_Stats_E/D(4 Languages: ANOVA & Kruskal)")
#     print(" - 2-4_Features_Stats_E/D(Present vs Absent features: ANOVA & Kruskal)")

# if __name__ == "__main__":
#     main()



import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_ROOT_MALAYSIA = r"D:\storage\jomon_kaen\jomon_kaen_dataset\malaysia"
DATASET_ROOT_JAPAN = r"D:\storage\jomon_kaen\jomon_kaen_dataset\japan"
OUTPUT_DIR = "anova_statistical_analysis"
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

# ==========================================
# 1. DATA LOADING
# ==========================================
def load_multilingual_dataset(root_dirs):
    """Load QA data from multiple directories."""
    qa_records = []
    
    for root_dir in root_dirs:
        root_path = Path(root_dir)
        if not root_path.exists():
            print(f"Warning: Directory not found: {root_dir}")
            continue

        print(f"Loading data from {root_dir}...")

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
                    except:
                        pass

                if language is None and "japan" in str(root_path).lower():
                    language = 'JAPANESE'

                if language not in TARGET_LANGUAGES: continue

                for pottery_path in session_path.iterdir():
                    if not pottery_path.is_dir(): continue
                    pottery_id = pottery_path.name

                    qa_file = pottery_path / "qa_corrected.csv"
                    if qa_file.exists():
                        try:
                            df_temp = pd.read_csv(qa_file, encoding='utf-8')
                            df_temp['timestamp'] = pd.to_numeric(df_temp['timestamp'], errors='coerce')
                            df_temp.dropna(subset=['timestamp'], inplace=True)
                            df_temp['pottery_id'] = pottery_id
                            df_temp['session_id'] = session_path.name
                            df_temp['Language'] = language
                            qa_records.append(df_temp)
                        except Exception as e:
                            print(f"Error loading {qa_file}: {e}")

    df_qa = pd.concat(qa_records, ignore_index=True) if qa_records else pd.DataFrame()
    print(f"Loaded {len(df_qa)} QA interactions.")
    return df_qa

# ==========================================
# 2. CALCULATE NORMALIZED METRICS
# ==========================================
def calculate_metrics(df_qa):
    print("Calculating session duration blocks and normalized metrics...")
    
    df = df_qa.copy()
    df['short_answer'] = df['answer'].astype(str).str.strip().map(EMOTION_MAP)
    df.dropna(subset=['short_answer', 'pottery_id', 'session_id'], inplace=True)
    df.sort_values(by=['pottery_id', 'session_id', 'timestamp'], inplace=True)
    
    df['time_diff'] = df.groupby(['pottery_id', 'session_id'])['timestamp'].diff()
    emotion_changed = df['short_answer'] != df.groupby(['pottery_id', 'session_id'])['short_answer'].shift()
    time_gap_exceeded = df['time_diff'] > 0.05
    df['block_id'] = (emotion_changed | time_gap_exceeded).cumsum()
    
    blocks = df.groupby(['pottery_id', 'session_id', 'Language', 'block_id']).agg(
        emotion=('short_answer', 'first'),
        start_time=('timestamp', 'min'),
        end_time=('timestamp', 'max'),
        event_count=('timestamp', 'count')
    ).reset_index()
    
    blocks['duration'] = (blocks['end_time'] - blocks['start_time']) * 1000 
    
    # Session Level Metrics
    session_events = blocks.groupby(['pottery_id', 'session_id', 'Language', 'emotion'])['event_count'].sum().unstack(fill_value=0).reset_index()
    session_duration = blocks.groupby(['pottery_id', 'session_id', 'Language', 'emotion'])['duration'].sum().unstack(fill_value=0).reset_index()
    
    for emo in EMOTION_COLS:
        if emo not in session_events.columns: session_events[emo] = 0
        if emo not in session_duration.columns: session_duration[emo] = 0
            
    session_events['total'] = session_events[EMOTION_COLS].sum(axis=1)
    session_duration['total'] = session_duration[EMOTION_COLS].sum(axis=1)
    
    for emo in EMOTION_COLS:
        session_events[emo] = session_events[emo] / session_events['total'].replace(0, np.nan)
        session_duration[emo] = session_duration[emo] / session_duration['total'].replace(0, np.nan)
        
    session_events.fillna(0, inplace=True)
    session_duration.fillna(0, inplace=True)
    
    # Pottery Level (93 Items Aggregation)
    pottery_events = session_events.groupby('pottery_id')[EMOTION_COLS].mean().reset_index()
    pottery_duration = session_duration.groupby('pottery_id')[EMOTION_COLS].mean().reset_index()
    
    # Formatting to match requested structure
    pottery_events.rename(columns={'pottery_id': 'Pottery_ID'}, inplace=True)
    pottery_duration.rename(columns={'pottery_id': 'Pottery_ID'}, inplace=True)
    
    pottery_events['Samples'] = range(1, len(pottery_events) + 1)
    pottery_duration['Samples'] = range(1, len(pottery_duration) + 1)
    
    pottery_events['Type'] = pottery_events['Pottery_ID'].apply(lambda x: 'Dogu' if any(p in str(x) for p in DOGU_PREFIXES) else 'Pottery')
    pottery_duration['Type'] = pottery_duration['Pottery_ID'].apply(lambda x: 'Dogu' if any(p in str(x) for p in DOGU_PREFIXES) else 'Pottery')
    
    # Reorder columns
    cols = ['Samples', 'Pottery_ID', 'Type'] + EMOTION_COLS
    
    return pottery_events[cols], pottery_duration[cols], session_events, session_duration

# ==========================================
# 3. FEATURES & TYPOLOGY INTEGRATION
# ==========================================
def get_features_path():
    """Finds the correct labels file."""
    possible_paths = [
        "./src/DS_Labels_Cleaned.xlsx",
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return None

def attach_features(pottery_df, features_file):
    if features_file is None or not os.path.exists(features_file):
        print("Warning: Features file not found! Please check path.")
        return pottery_df.copy(), [], []
        
    try:
        if features_file.endswith('.xlsx'):
            feat_df = pd.read_excel(features_file)
        else:
            try:
                feat_df = pd.read_csv(features_file, encoding='utf-8-sig')
            except:
                feat_df = pd.read_csv(features_file, encoding='shift_jis')
    except Exception as e:
        print(f"Error loading features file: {e}")
        return pottery_df.copy(), [], []
        
    # Standardize Pottery ID
    feat_df['Pottery_ID'] = feat_df.iloc[:, 0].astype(str).str.replace('.ply', '', regex=False)
    
    # Extract Features (Binary HAS_ columns)
    feature_cols = [c for c in feat_df.columns if c.startswith('HAS_')]
    
    # Extract Typology (Binary SHAPE_TYPE_ columns)
    # Exclude NAN to focus only on actual typologies
    shape_cols = [c for c in feat_df.columns if c.startswith('SHAPE_TYPE_') and 'NAN' not in c]
    
    # Merge with 93 items dataframe
    merged_df = pottery_df.merge(feat_df[['Pottery_ID'] + shape_cols + feature_cols], on='Pottery_ID', how='left')
    
    # Ensure binary format (convert TRUE/FALSE/strings to 1/0)
    for col in shape_cols + feature_cols:
        merged_df[col] = pd.to_numeric(merged_df[col].replace({'TRUE': 1, 'FALSE': 0, True: 1, False: 0}), errors='coerce').fillna(0)
        
    return merged_df, feature_cols, shape_cols

# ==========================================
# 4. STATISTICAL TESTS (WITH METADATA)
# ==========================================
def run_stats_on_groups(df, group_col, metric_cols):
    """Runs BOTH Standard ANOVA and Kruskal-Wallis Tests and calculates Metadata."""
    results = []
    
    for emo in metric_cols:
        # Group data and extract valid groups (removing NaNs)
        group_data = {name: group[emo].dropna().values for name, group in df.groupby(group_col) if len(group[emo].dropna()) > 0}
        
        # Need at least 2 items per group for variance
        valid_groups = {name: vals for name, vals in group_data.items() if len(vals) >= 2}
        
        if len(valid_groups) >= 2:
            arrays = list(valid_groups.values())
            
            # Calculate Metadata
            total_n = sum(len(arr) for arr in arrays)
            group_details = []
            
            for name, arr in valid_groups.items():
                mean_val = np.mean(arr)
                std_val = np.std(arr, ddof=1)
                
                # Convert 1.0 / 0.0 into True / False, leave strings alone
                if str(name) in ['0.0', '0']:
                    display_name = 'False'
                elif str(name) in ['1.0', '1']:
                    display_name = 'True'
                else:
                    display_name = str(name)
                
                group_details.append(f"{display_name} (N={len(arr)}): Mean={mean_val:.4f}, Std={std_val:.4f}")
                
            details_str = " | ".join(group_details)
            
            # 1. Standard ANOVA
            try:
                f_stat, f_p = stats.f_oneway(*arrays)
            except:
                f_stat, f_p = np.nan, np.nan
                
            # 2. Kruskal-Wallis (Non-parametric)
            try:
                h_stat, h_p = stats.kruskal(*arrays)
            except:
                h_stat, h_p = np.nan, np.nan
                
            results.append({
                'Emotion': emo,
                'Total_N': total_n,
                'Group_Details': details_str,
                'ANOVA_F_Stat': f_stat,
                'ANOVA_p_value': f_p,
                'ANOVA_Significant': f_p < 0.05 if pd.notna(f_p) else False,
                'Kruskal_H_Stat': h_stat,
                'Kruskal_p_value': h_p,
                'Kruskal_Significant': h_p < 0.05 if pd.notna(h_p) else False
            })
            
    return pd.DataFrame(results)

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("=" * 80)
    print("STATISTICAL ANALYSIS: ANOVA, KRUSKAL & METADATA")
    print("=" * 80)
    
    df_qa = load_multilingual_dataset([DATASET_ROOT_MALAYSIA, DATASET_ROOT_JAPAN])
    if df_qa.empty:
        print("Error: No data loaded.")
        return
        
    # Calculate 93-Item Metrics and Session Metrics
    pot_events, pot_dur, sess_events, sess_dur = calculate_metrics(df_qa)
    
    # Locate features CSV safely
    features_file = get_features_path()
    if features_file:
        print(f"Loading Features and Typologies from: {features_file}")
    
    # Attach features to 93-item sets
    pot_events_feat, feature_cols, shape_cols = attach_features(pot_events, features_file)
    pot_dur_feat, _, _ = attach_features(pot_dur, features_file)
    
    # ---------------------------------------------------------
    # 2-1: Artifact Type (Pottery vs Dogu - 93 items level)
    # ---------------------------------------------------------
    print("Running 2-1: Artifact Type Analysis...")
    a21_e = run_stats_on_groups(pot_events_feat, 'Type', EMOTION_COLS)
    a21_d = run_stats_on_groups(pot_dur_feat, 'Type', EMOTION_COLS)
    
    # ---------------------------------------------------------
    # 2-2: Typologies (Loop through all SHAPE_TYPE_ separately)
    # ---------------------------------------------------------
    print("Running 2-2: Typology Analysis (Itemized separately)...")
    a22_e_list, a22_d_list = [], []
    for shape in shape_cols:
        res_e = run_stats_on_groups(pot_events_feat, shape, EMOTION_COLS)
        res_d = run_stats_on_groups(pot_dur_feat, shape, EMOTION_COLS)
        
        # Add the Typology name directly in the output dataframe
        if not res_e.empty:
            res_e.insert(0, 'Typology', shape.replace('SHAPE_TYPE_', ''))
            a22_e_list.append(res_e)
        if not res_d.empty:
            res_d.insert(0, 'Typology', shape.replace('SHAPE_TYPE_', ''))
            a22_d_list.append(res_d)
            
    a22_e = pd.concat(a22_e_list, ignore_index=True) if a22_e_list else pd.DataFrame()
    a22_d = pd.concat(a22_d_list, ignore_index=True) if a22_d_list else pd.DataFrame()
    
    # ---------------------------------------------------------
    # 2-3: Languages (Session level)
    # ---------------------------------------------------------
    print("Running 2-3: Language Analysis...")
    a23_e = run_stats_on_groups(sess_events, 'Language', EMOTION_COLS)
    a23_d = run_stats_on_groups(sess_dur, 'Language', EMOTION_COLS)
    
    # ---------------------------------------------------------
    # 2-4: Features (Loop through all HAS_ features - 93 items level)
    # ---------------------------------------------------------
    print("Running 2-4: Feature Analysis (Itemized separately)...")
    a24_e_list, a24_d_list = [], []
    for feat in feature_cols:
        res_e = run_stats_on_groups(pot_events_feat, feat, EMOTION_COLS)
        res_d = run_stats_on_groups(pot_dur_feat, feat, EMOTION_COLS)
        
        # Add the Feature name directly in the output dataframe
        if not res_e.empty:
            res_e.insert(0, 'Feature', feat)
            a24_e_list.append(res_e)
        if not res_d.empty:
            res_d.insert(0, 'Feature', feat)
            a24_d_list.append(res_d)
            
    a24_e = pd.concat(a24_e_list, ignore_index=True) if a24_e_list else pd.DataFrame()
    a24_d = pd.concat(a24_d_list, ignore_index=True) if a24_d_list else pd.DataFrame()

    # ---------------------------------------------------------
    # EXCEL EXPORT
    # ---------------------------------------------------------
    out_file = os.path.join(OUTPUT_DIR, "ANOVA_and_Kruskal_Analysis.xlsx")
    print(f"\nWriting to Excel: {out_file}...")
    
    with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
        # RAW DATA (93 ITEMS)
        pot_events.to_excel(writer, sheet_name='1-1_Norm_Events', index=False)
        pot_dur.to_excel(writer, sheet_name='1-2_Norm_Duration', index=False)
        
        # 2-1 ARTIFACT TYPE STATS (Pottery vs Dogu)
        if not a21_e.empty: a21_e.to_excel(writer, sheet_name='2-1_ObjectType_Stats_E', index=False)
        if not a21_d.empty: a21_d.to_excel(writer, sheet_name='2-1_ObjectType_Stats_D', index=False)
        
        # 2-2 TYPOLOGY STATS (Each Typology separated)
        if not a22_e.empty: a22_e.to_excel(writer, sheet_name='2-2_Typology_Stats_E', index=False)
        if not a22_d.empty: a22_d.to_excel(writer, sheet_name='2-2_Typology_Stats_D', index=False)
            
        # 2-3 LANGUAGE STATS
        if not a23_e.empty: a23_e.to_excel(writer, sheet_name='2-3_Language_Stats_E', index=False)
        if not a23_d.empty: a23_d.to_excel(writer, sheet_name='2-3_Language_Stats_D', index=False)
            
        # 2-4 FEATURE STATS (Each Feature separated)
        if not a24_e.empty: a24_e.to_excel(writer, sheet_name='2-4_Features_Stats_E', index=False)
        if not a24_d.empty: a24_d.to_excel(writer, sheet_name='2-4_Features_Stats_D', index=False)

    print("✓ Analysis successfully exported.")
    print("\nSHEETS CREATED:")
    print(" - 1-1_Norm_Events       (93 items, Event Data)")
    print(" - 1-2_Norm_Duration     (93 items, Duration Data)")
    print(" - 2-1_ObjectType_Stats_E/D (Pottery vs Dogu: ANOVA & Kruskal)")
    print(" - 2-2_Typology_Stats_E/D(Typologies itemized separately: ANOVA & Kruskal)")
    print(" - 2-3_Language_Stats_E/D(4 Languages: ANOVA & Kruskal)")
    print(" - 2-4_Features_Stats_E/D(Present vs Absent features: ANOVA & Kruskal)")

if __name__ == "__main__":
    main()