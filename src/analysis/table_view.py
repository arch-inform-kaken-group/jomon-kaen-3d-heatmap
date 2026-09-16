import pandas as pd
import numpy as np
import re
import os

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = "anova_lmm_statistical_analysis/ANOVA_and_LMM_Analysis.xlsx"
OUTPUT_FILE = "anova_lmm_statistical_analysis/Paper_Tables_Ready.xlsx"
EMOTIONS = ["Interesting", "Beautiful", "Strange", "Scary", "Feel nothing"]


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def parse_p_value(val):
    """Safely parse p-values which might be floats, strings, or '< 1e-300'."""
    if pd.isna(val): return 1.0
    if isinstance(val, (int, float)): return float(val)
    s = str(val).strip()
    if s.startswith('<'): return 0.0
    try:
        return float(s)
    except:
        return 1.0


def get_asterisks(p_val):
    """Convert p-value to academic significance stars."""
    if p_val < 0.001: return "***"
    if p_val < 0.01: return "**"
    if p_val < 0.05: return "*"
    return ""


def parse_means_from_details(text):
    """Extract the two Mean values from the Group_Details string."""
    if pd.isna(text): return np.nan, np.nan
    matches = re.findall(r"Mean=([-+]?\d*\.\d+|[-+]?\d+)", str(text))
    if len(matches) >= 2:
        return float(matches[0]), float(matches[1])
    return np.nan, np.nan


def format_cell(mean1, mean2, p_val):
    """Generate the final table cell string (e.g., '↑***', '↓*', 'ns').
    Note: mean1 is always the first alphabetical group (e.g., Dogu, Japan, Present)."""
    is_sig = parse_p_value(p_val) < 0.05
    if not is_sig:
        return "ns"

    if pd.isna(mean1) or pd.isna(mean2):
        return "?"

    arrow = "↑" if mean1 > mean2 else "↓"
    stars = get_asterisks(parse_p_value(p_val))
    return f"{arrow}{stars}"


# ==========================================
# 1. GENERATE TABLE 1: SIGNIFICANCE MATRIX (Overall)
# ==========================================
def generate_table1(xlsx):
    print("Generating Table 1: Overall Significance Matrix...")
    matrix_data = {}
    xl = pd.ExcelFile(xlsx)

    if '2-1_ObjType_LMM_E' in xl.sheet_names:
        df_21 = pd.read_excel(xlsx, sheet_name='2-1_ObjType_LMM_E')
        for _, row in df_21.iterrows():
            m1, m2 = parse_means_from_details(row['Group_Details'])
            matrix_data.setdefault("Object Type (Dogu vs Pottery)",
                                   {})[row['Emotion']] = format_cell(
                                       m1,
                                       m2,
                                       row['LMM_p_value'])

    if '2-2_Typology_LMM_E' in xl.sheet_names:
        df_22 = pd.read_excel(xlsx, sheet_name='2-2_Typology_LMM_E')
        for _, row in df_22.iterrows():
            m1, m2 = parse_means_from_details(row['Group_Details'])
            matrix_data.setdefault(row['Typology'],
                                   {})[row['Emotion']] = format_cell(
                                       m1,
                                       m2,
                                       row['LMM_p_value'])

    if '2-4_Features_LMM_E' in xl.sheet_names:
        df_24 = pd.read_excel(xlsx, sheet_name='2-4_Features_LMM_E')
        for _, row in df_24.iterrows():
            feat = row['Feature'].replace("HAS_", "").replace("_", " ").title()
            m1, m2 = parse_means_from_details(row['Group_Details'])
            matrix_data.setdefault(feat,
                                   {})[row['Emotion']] = format_cell(
                                       m1,
                                       m2,
                                       row['LMM_p_value'])

    df_table1 = pd.DataFrame(matrix_data).T[EMOTIONS]
    df_table1.index.name = "Factor / Group"
    return df_table1


# ==========================================
# 2. GENERATE TABLE 2: METHODOLOGICAL DIVERGENCE
# ==========================================
def generate_table2(xlsx):
    print("Generating Table 2: Methodological Divergence...")
    xl = pd.ExcelFile(xlsx)
    sheets_map = {
        "Object Type": ("2-1_ObjType_ANOVA_E",
                        "2-1_ObjType_LMM_E"),
        "Typology": ("2-2_Typology_ANOVA_E",
                     "2-2_Typology_LMM_E"),
        "Language": ("2-3_Language_ANOVA_E",
                     "2-3_Language_LMM_E"),
        "Features": ("2-4_Features_ANOVA_E",
                     "2-4_Features_LMM_E")
    }

    summary_rows = []
    for category, (a_sheet, l_sheet) in sheets_map.items():
        if a_sheet not in xl.sheet_names or l_sheet not in xl.sheet_names:
            continue
        df_a = pd.read_excel(xlsx, sheet_name=a_sheet)
        df_l = pd.read_excel(xlsx, sheet_name=l_sheet)

        merge_keys = ['Emotion']
        if 'Typology' in df_a.columns: merge_keys.append('Typology')
        if 'Feature' in df_a.columns: merge_keys.append('Feature')

        merged = df_a.merge(df_l, on=merge_keys, suffixes=('_ANOVA', '_LMM'))
        total = len(merged)
        agree = lmm_only = anova_only = 0

        for _, row in merged.iterrows():
            a_sig = str(row['ANOVA_Significant']).upper() == 'TRUE'
            l_sig = str(row['LMM_Significant']).upper() == 'TRUE'
            if a_sig == l_sig: agree += 1
            elif l_sig and not a_sig: lmm_only += 1
            elif a_sig and not l_sig: anova_only += 1

        direction = "—"
        if lmm_only > 0 and anova_only == 0: direction = "All LMM-only SIG"
        elif anova_only > 0 and lmm_only == 0: direction = "All ANOVA-only SIG"

        summary_rows.append({
            "Factor Category": category,
            "Total Tests": total,
            "Agreed (SIG or NS)": agree,
            "Diverged": lmm_only + anova_only,
            "Direction of Divergence": direction
        })

    df_table2 = pd.DataFrame(summary_rows)
    totals = df_table2.sum(numeric_only=True)
    totals['Factor Category'] = "Total"
    totals[
        'Direction of Divergence'] = f"{int(totals['Diverged']) - 2} LMM-only vs 2 ANOVA-only"
    df_table2 = pd.concat([df_table2,
                           pd.DataFrame([totals])],
                          ignore_index=True)
    return df_table2


# ==========================================
# 3. GENERATE TABLE 3: DETAILED LMM STATS (Overall SI)
# ==========================================
def generate_table3(xlsx):
    print("Generating Table 3: Detailed Overall LMM Stats (Supplementary)...")
    xl = pd.ExcelFile(xlsx)
    rows = []
    sheet_map = {
        '2-2_Typology_LMM_E': 'Typology',
        '2-4_Features_LMM_E': 'Feature'
    }

    for sheet, factor_type in sheet_map.items():
        if sheet not in xl.sheet_names: continue
        df = pd.read_excel(xlsx, sheet_name=sheet)
        for _, row in df.iterrows():
            if str(row['LMM_Significant']).upper() != 'TRUE': continue
            m1, m2 = parse_means_from_details(row['Group_Details'])
            delta = m1 - m2 if not (pd.isna(m1) or pd.isna(m2)) else np.nan

            factor_name = row['Typology'] if factor_type == 'Typology' else row[
                'Feature'].replace("HAS_",
                                   "").replace("_",
                                               " ").title()
            rows.append({
                "Factor Category": factor_type,
                "Factor / Group": factor_name,
                "Emotion": row['Emotion'],
                "Baseline Mean (Absent)": f"{m2:.3f}",
                "Target Mean (Present)": f"{m1:.3f}",
                "Δ (Effect Size)": f"{delta:+.3f}",
                "LMM F-stat": f"{row['LMM_F_Stat']:.2f}",
                "p-value": str(row['LMM_p_value']),
                "ICC": f"{row['ICC']:.3f}"
            })
    return pd.DataFrame(rows)


# ==========================================
# 4. GENERATE TABLE 4: REGIONAL MATRIX (Japan vs Malaysia)
# ==========================================
def generate_table4_region_matrix(xlsx):
    print(
        "Generating Table 4: Regional Significance Matrix (Japan vs Malaysia)..."
    )
    matrix_data = {}
    xl = pd.ExcelFile(xlsx)

    # 3-1 Pottery
    if '3-1_Region_Pottery_LMM_E' in xl.sheet_names:
        df = pd.read_excel(xlsx, sheet_name='3-1_Region_Pottery_LMM_E')
        for _, row in df.iterrows():
            m1, m2 = parse_means_from_details(row['Group_Details']) # JAPAN, MALAYSIA
            matrix_data.setdefault("Pottery (Japan vs Malaysia)",
                                   {})[row['Emotion']] = format_cell(
                                       m1,
                                       m2,
                                       row['LMM_p_value'])

    # 3-2 Dogu
    if '3-2_Region_Dogu_LMM_E' in xl.sheet_names:
        df = pd.read_excel(xlsx, sheet_name='3-2_Region_Dogu_LMM_E')
        for _, row in df.iterrows():
            m1, m2 = parse_means_from_details(row['Group_Details'])
            matrix_data.setdefault("Dogu (Japan vs Malaysia)",
                                   {})[row['Emotion']] = format_cell(
                                       m1,
                                       m2,
                                       row['LMM_p_value'])

    # 3-3 Typology
    if '3-3_Region_Typology_LMM_E' in xl.sheet_names:
        df = pd.read_excel(xlsx, sheet_name='3-3_Region_Typology_LMM_E')
        for _, row in df.iterrows():
            m1, m2 = parse_means_from_details(row['Group_Details'])
            matrix_data.setdefault(f"Typology: {row['Typology']}",
                                   {})[row['Emotion']] = format_cell(
                                       m1,
                                       m2,
                                       row['LMM_p_value'])

    # 3-4 Features
    if '3-4_Region_Features_LMM_E' in xl.sheet_names:
        df = pd.read_excel(xlsx, sheet_name='3-4_Region_Features_LMM_E')
        for _, row in df.iterrows():
            feat = row['Feature'].replace("HAS_", "").replace("_", " ").title()
            m1, m2 = parse_means_from_details(row['Group_Details'])
            matrix_data.setdefault(f"Feature: {feat}",
                                   {})[row['Emotion']] = format_cell(
                                       m1,
                                       m2,
                                       row['LMM_p_value'])

    df_table4 = pd.DataFrame(matrix_data).T[EMOTIONS]
    df_table4.index.name = "Factor / Group (Japan vs Malaysia)"
    return df_table4


# ==========================================
# 5. GENERATE TABLE S2: DETAILED REGIONAL STATS
# ==========================================
def generate_tableS2_region_detailed(xlsx):
    print(
        "Generating Table S2: Detailed Regional LMM Stats (Japan vs Malaysia)..."
    )
    xl = pd.ExcelFile(xlsx)
    rows = []
    sheet_map = {
        '3-1_Region_Pottery_LMM_E': 'Pottery',
        '3-2_Region_Dogu_LMM_E': 'Dogu',
        '3-3_Region_Typology_LMM_E': 'Typology',
        '3-4_Region_Features_LMM_E': 'Feature'
    }

    for sheet, factor_type in sheet_map.items():
        if sheet not in xl.sheet_names: continue
        df = pd.read_excel(xlsx, sheet_name=sheet)
        for _, row in df.iterrows():
            if str(row['LMM_Significant']).upper() != 'TRUE': continue

            m1, m2 = parse_means_from_details(row['Group_Details']) # JAPAN, MALAYSIA
            delta = m1 - m2 if not (pd.isna(m1) or pd.isna(m2)) else np.nan

            if factor_type == 'Feature':
                factor_name = row['Feature'].replace("HAS_",
                                                     "").replace("_",
                                                                 " ").title()
            elif factor_type == 'Typology':
                factor_name = row['Typology']
            else:
                factor_name = factor_type

            rows.append({
                "Factor Category":
                factor_type,
                "Factor / Group":
                factor_name,
                "Emotion":
                row['Emotion'],
                "Japan Mean":
                f"{m1:.3f}" if not pd.isna(m1) else "N/A",
                "Malaysia Mean":
                f"{m2:.3f}" if not pd.isna(m2) else "N/A",
                "Δ (Japan - Malaysia)":
                f"{delta:+.3f}" if not pd.isna(delta) else "N/A",
                "LMM F-stat":
                f"{row['LMM_F_Stat']:.2f}",
                "p-value":
                str(row['LMM_p_value']),
                "ICC":
                f"{row['ICC']:.3f}" if not pd.isna(row['ICC']) else "N/A"
            })
    return pd.DataFrame(rows)


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    if not os.path.exists(INPUT_FILE):
        print(
            f"Error: {INPUT_FILE} not found. Please run the statistical analysis script first."
        )
        return

    print("Loading data and generating publication tables...")

    t1 = generate_table1(INPUT_FILE)
    t2 = generate_table2(INPUT_FILE)
    t3 = generate_table3(INPUT_FILE)
    t4 = generate_table4_region_matrix(INPUT_FILE)
    tS2 = generate_tableS2_region_detailed(INPUT_FILE)

    # Export to Excel
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        t1.to_excel(writer, sheet_name='Table1_Significance_Matrix')
        t2.to_excel(writer, sheet_name='Table2_Method_Divergence', index=False)
        t3.to_excel(writer,
                    sheet_name='TableS1_Detailed_LMM_Stats',
                    index=False)
        t4.to_excel(writer, sheet_name='Table4_Region_Matrix')
        tS2.to_excel(writer,
                     sheet_name='TableS2_Region_Detailed_LMM',
                     index=False)

    print(f"\n✅ SUCCESS! Tables exported to: {OUTPUT_FILE}")

    # Print Markdown previews for quick verification (with fallback if tabulate is missing)
    print("\n" + "=" * 60)
    print("PREVIEW: Table 4 (Regional Significance Matrix)")
    print("=" * 60)
    try:
        print(t4.to_markdown())
    except ImportError:
        print(t4.to_string())

    print("\n" + "=" * 60)
    print("PREVIEW: Table S2 (Detailed Regional Stats - First 5 rows)")
    print("=" * 60)
    try:
        print(tS2.head().to_markdown(index=False))
    except ImportError:
        print(tS2.head().to_string(index=False))


if __name__ == "__main__":
    main()
