import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

try:
    import mpmath
    MP_OK = True
except ImportError:
    MP_OK = False

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_ROOT_MALAYSIA = r"D:\storage\jomon_kaen\jomon_kaen_dataset\malaysia"
DATASET_ROOT_JAPAN = r"D:\storage\jomon_kaen\jomon_kaen_dataset\japan"
OUTPUT_DIR = "anova_lmm_statistical_analysis"
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
# HIGH-PRECISION P-VALUES
# ==========================================
def _fmt_mp(p_mp):
    if p_mp == 0 or p_mp < mpmath.mpf('1e-320'):
        return "< 1.000000000000000e-300"
    exp = int(mpmath.floor(mpmath.log10(p_mp)))
    mant = p_mp / (mpmath.mpf(10) ** exp)
    sign = '-' if exp < 0 else '+'
    return f"{float(mant):.15f}e{sign}{abs(exp):02d}"


def hp_p_from_z(z):
    if z is None or not np.isfinite(z):
        return np.nan

    if MP_OK:
        mpmath.mp.dps = 50
        return _fmt_mp(mpmath.erfc(abs(mpmath.mpf(z)) / mpmath.sqrt(2)))

    p = stats.norm.sf(abs(z)) * 2
    return f"{p:.15e}" if p > 0 else "< 2.225073858507201e-308"


def hp_p_from_f(f, d1, d2):
    if f is None or not np.isfinite(f):
        return np.nan

    try:
        d1 = float(d1)
        d2 = float(d2)
    except Exception:
        return np.nan

    if not np.isfinite(d1) or not np.isfinite(d2) or d1 <= 0 or d2 <= 0:
        return np.nan

    if f <= 0:
        return "1.000000000000000e+00"

    if MP_OK:
        mpmath.mp.dps = 50
        x = mpmath.mpf(d2) / (mpmath.mpf(d2) + mpmath.mpf(d1) * mpmath.mpf(f))
        return _fmt_mp(
            mpmath.betainc(
                mpmath.mpf(d2) / 2,
                mpmath.mpf(d1) / 2,
                0,
                x,
                regularized=True
            )
        )

    p = stats.f.sf(f, d1, d2)
    return f"{p:.15e}" if p > 0 else "< 2.225073858507201e-308"


def p_is_significant(p):
    if p is None:
        return False
    if isinstance(p, float) and np.isnan(p):
        return False

    s = str(p)
    if s.startswith('<'):
        return True

    try:
        return float(s) < 0.05
    except ValueError:
        return False


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

        if 'malaysia' in str(root_path).lower():
            country = 'MALAYSIA'
        elif 'japan' in str(root_path).lower():
            country = 'JAPAN'
        else:
            country = 'UNKNOWN'

        print(f"Loading data from {root_dir} (Region: {country})...")

        for group_path in root_path.iterdir():
            if not group_path.is_dir():
                continue

            for session_path in group_path.iterdir():
                if not session_path.is_dir():
                    continue

                language = None
                lang_file = session_path / 'language.txt'

                if lang_file.exists():
                    try:
                        text = lang_file.read_text(encoding='utf-8').strip().upper()
                        if 'ENGLISH' in text or 'EN ' in text or text == 'EN':
                            language = 'ENGLISH'
                        elif 'MALAY' in text or 'BM ' in text or text == 'BM':
                            language = 'MALAY'
                        elif 'CHINESE' in text or 'MANDARIN' in text:
                            language = 'CHINESE'
                        elif 'JAPANESE' in text or 'JP ' in text or text == 'JP':
                            language = 'JAPANESE'
                    except Exception:
                        pass

                if language is None and "japan" in str(root_path).lower():
                    language = 'JAPANESE'

                if language not in TARGET_LANGUAGES:
                    continue

                subject_id = group_path.name

                for pottery_path in session_path.iterdir():
                    if not pottery_path.is_dir():
                        continue

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
                            df_temp['Subject'] = subject_id
                            df_temp['Country'] = country

                            qa_records.append(df_temp)

                        except Exception as e:
                            print(f"Error loading {qa_file}: {e}")

    df_qa = pd.concat(qa_records, ignore_index=True) if qa_records else pd.DataFrame()
    print(f"Loaded {len(df_qa)} QA interactions.")
    return df_qa


# ==========================================
# 2. CALCULATE NORMALIZED METRICS
# ==========================================
def calculate_metrics_long(df_qa):
    print("Calculating session duration blocks and normalized metrics (Long Format)...")

    df = df_qa.copy()
    df['short_answer'] = df['answer'].astype(str).str.strip().map(EMOTION_MAP)
    df.dropna(subset=['short_answer', 'pottery_id', 'session_id', 'Subject', 'Country'], inplace=True)
    df.sort_values(by=['pottery_id', 'session_id', 'timestamp'], inplace=True)

    df['time_diff'] = df.groupby(['pottery_id', 'session_id'])['timestamp'].diff()
    emotion_changed = df['short_answer'] != df.groupby(['pottery_id', 'session_id'])['short_answer'].shift()
    time_gap_exceeded = df['time_diff'] > 0.05
    df['block_id'] = (emotion_changed | time_gap_exceeded).cumsum()

    blocks = df.groupby(['pottery_id', 'session_id', 'Subject', 'Language', 'Country', 'block_id']).agg(
        emotion=('short_answer', 'first'),
        start_time=('timestamp', 'min'),
        end_time=('timestamp', 'max'),
        event_count=('timestamp', 'count')
    ).reset_index()

    blocks['duration'] = (blocks['end_time'] - blocks['start_time']) * 1000

    session_events = blocks.groupby(
        ['pottery_id', 'session_id', 'Subject', 'Language', 'Country', 'emotion']
    )['event_count'].sum().unstack(fill_value=0).reset_index()

    session_duration = blocks.groupby(
        ['pottery_id', 'session_id', 'Subject', 'Language', 'Country', 'emotion']
    )['duration'].sum().unstack(fill_value=0).reset_index()

    for emo in EMOTION_COLS:
        if emo not in session_events.columns:
            session_events[emo] = 0
        if emo not in session_duration.columns:
            session_duration[emo] = 0

    session_events['total'] = session_events[EMOTION_COLS].sum(axis=1)
    session_duration['total'] = session_duration[EMOTION_COLS].sum(axis=1)

    for emo in EMOTION_COLS:
        session_events[emo] = session_events[emo] / session_events['total'].replace(0, np.nan)
        session_duration[emo] = session_duration[emo] / session_duration['total'].replace(0, np.nan)

    session_events.fillna(0, inplace=True)
    session_duration.fillna(0, inplace=True)

    pottery_events = session_events.groupby('pottery_id')[EMOTION_COLS].mean().reset_index()
    pottery_duration = session_duration.groupby('pottery_id')[EMOTION_COLS].mean().reset_index()

    pottery_events.rename(columns={'pottery_id': 'Pottery_ID'}, inplace=True)
    pottery_duration.rename(columns={'pottery_id': 'Pottery_ID'}, inplace=True)

    pottery_events['Samples'] = range(1, len(pottery_events) + 1)
    pottery_duration['Samples'] = range(1, len(pottery_duration) + 1)

    pottery_events['Type'] = pottery_events['Pottery_ID'].apply(
        lambda x: 'Dogu' if any(p in str(x) for p in DOGU_PREFIXES) else 'Pottery'
    )
    pottery_duration['Type'] = pottery_duration['Pottery_ID'].apply(
        lambda x: 'Dogu' if any(p in str(x) for p in DOGU_PREFIXES) else 'Pottery'
    )

    cols = ['Samples', 'Pottery_ID', 'Type'] + EMOTION_COLS

    id_vars_sess_e = [c for c in session_events.columns if c not in EMOTION_COLS]
    id_vars_sess_d = [c for c in session_duration.columns if c not in EMOTION_COLS]

    sess_events_long = session_events.melt(
        id_vars=id_vars_sess_e,
        value_vars=EMOTION_COLS,
        var_name='Emotion',
        value_name='Response'
    )

    sess_dur_long = session_duration.melt(
        id_vars=id_vars_sess_d,
        value_vars=EMOTION_COLS,
        var_name='Emotion',
        value_name='Response'
    )

    type_map = pottery_events.set_index('Pottery_ID')['Type'].to_dict()
    sess_events_long['Type'] = sess_events_long['pottery_id'].map(type_map)
    sess_dur_long['Type'] = sess_dur_long['pottery_id'].map(type_map)

    return pottery_events[cols], pottery_duration[cols], sess_events_long, sess_dur_long


# ==========================================
# 3. FEATURES & TYPOLOGY INTEGRATION
# ==========================================
def get_features_path():
    possible_paths = ["./src/DS_Labels_Cleaned.xlsx"]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return None


def attach_features(pottery_df, features_file):
    if features_file is None or not os.path.exists(features_file):
        print("Warning: Features file not found!")
        return pottery_df.copy(), [], []

    try:
        if features_file.endswith('.xlsx'):
            feat_df = pd.read_excel(features_file)
        else:
            try:
                feat_df = pd.read_csv(features_file, encoding='utf-8-sig')
            except Exception:
                feat_df = pd.read_csv(features_file, encoding='shift_jis')
    except Exception as e:
        print(f"Error loading features file: {e}")
        return pottery_df.copy(), [], []

    feat_df['Pottery_ID'] = feat_df.iloc[:, 0].astype(str).str.replace('.ply', '', regex=False)
    feat_df = feat_df.drop_duplicates(subset='Pottery_ID', keep='first')

    feature_cols = [c for c in feat_df.columns if c.startswith('HAS_')]
    shape_cols = [c for c in feat_df.columns if c.startswith('SHAPE_TYPE_') and 'NAN' not in c]

    merged_df = pottery_df.merge(
        feat_df[['Pottery_ID'] + shape_cols + feature_cols],
        on='Pottery_ID',
        how='left'
    )

    for col in shape_cols + feature_cols:
        merged_df[col] = pd.to_numeric(
            merged_df[col].replace({'TRUE': 1, 'FALSE': 0, True: 1, False: 0}),
            errors='coerce'
        ).fillna(0)

    return merged_df, feature_cols, shape_cols


def attach_features_to_long(long_df, features_file):
    pot_wide = long_df[['pottery_id']].drop_duplicates()
    pot_wide, f_cols, s_cols = attach_features(
        pot_wide.rename(columns={'pottery_id': 'Pottery_ID'}),
        features_file
    )
    pot_wide.rename(columns={'Pottery_ID': 'pottery_id'}, inplace=True)

    merged = long_df.merge(pot_wide, on='pottery_id', how='left')
    return merged, f_cols, s_cols


# ==========================================
# 4. LMM ADJUSTED MEANS FOR 93 ITEMS
# ==========================================
def get_lmm_adjusted_means(long_df):
    print("Calculating LMM Adjusted Means for 93 Artifacts...")

    df = long_df.copy()
    df.dropna(subset=['Response', 'Emotion', 'pottery_id', 'Subject'], inplace=True)

    df['pottery_id'] = df['pottery_id'].astype(str)
    df['Subject'] = df['Subject'].astype(str)

    # Country-nested subject ID prevents shared random effects across countries
    if 'Country' in df.columns:
        df['Subject_UID'] = df['Country'].astype(str).str.upper() + '::' + df['Subject']
    else:
        df['Subject_UID'] = df['Subject']

    out = pd.DataFrame({'Pottery_ID': sorted(df['pottery_id'].unique())})

    for emo in EMOTION_COLS:
        emo_df = df[df['Emotion'] == emo].copy()
        adjusted_means = {}

        try:
            if emo_df['Subject_UID'].nunique() >= 2 and emo_df['pottery_id'].nunique() >= 2:
                model = smf.mixedlm(
                    "Response ~ C(pottery_id)",
                    emo_df,
                    groups=emo_df["Subject_UID"]
                )
                result = model.fit(reml=True, method='powell', maxiter=100)

                params = result.fe_params
                intercept = params.iloc[0]

                ref_pottery = sorted(emo_df['pottery_id'].unique())[0]
                adjusted_means[ref_pottery] = intercept

                for idx, val in params.iloc[1:].items():
                    pid = str(idx).replace('C(pottery_id)[T.', '').replace(']', '')
                    adjusted_means[pid] = intercept + val
            else:
                adjusted_means = emo_df.groupby('pottery_id')['Response'].mean().to_dict()

        except Exception as e:
            print(f"   LMM Mean Warning for {emo}: {e}. Falling back to raw means.")
            adjusted_means = emo_df.groupby('pottery_id')['Response'].mean().to_dict()

        out[f'LMM_Adjusted_{emo}'] = out['Pottery_ID'].map(adjusted_means)

    return out


# ==========================================
# 5. NEW: COUNTRY-SPECIFIC LMM MEANS
# ==========================================
def _fit_lmm_adjusted_means_one_country(emo_df):
    """
    Fits:
        Response ~ C(pottery_id)
    with random intercept:
        Subject_UID

    Returns:
        raw_means, lmm_adjusted_means, lmm_success
    """
    if emo_df.empty:
        return {}, {}, False

    raw = emo_df.groupby('pottery_id')['Response'].mean().to_dict()

    if emo_df['pottery_id'].nunique() < 2 or emo_df['Subject_UID'].nunique() < 2:
        return raw, raw, False

    try:
        model = smf.mixedlm(
            "Response ~ C(pottery_id)",
            emo_df,
            groups=emo_df["Subject_UID"]
        )
        result = model.fit(reml=True, method='powell', maxiter=100)

        params = result.fe_params
        intercept = params.iloc[0]

        ref_pottery = sorted(emo_df['pottery_id'].unique())[0]
        adjusted = {ref_pottery: intercept}

        for idx, val in params.iloc[1:].items():
            pid = str(idx).replace('C(pottery_id)[T.', '').replace(']', '')
            adjusted[pid] = intercept + val

        return raw, adjusted, True

    except Exception:
        return raw, raw, False


def get_lmm_adjusted_means_by_country(long_df, countries=None):
    """
    Creates Sheet-1 style pottery-level LMM adjusted means separately for each country.
    """
    print("Calculating Japan/Malaysia LMM adjusted means (Sheet-1 style)...")

    df = long_df.copy()
    df.dropna(subset=['Response', 'Emotion', 'pottery_id', 'Subject', 'Country'], inplace=True)

    df['Country'] = df['Country'].astype(str).str.upper()
    df['pottery_id'] = df['pottery_id'].astype(str)
    df['Subject_UID'] = df['Country'] + '::' + df['Subject'].astype(str)

    if countries is None:
        countries = [c for c in ['JAPAN', 'MALAYSIA'] if c in df['Country'].unique()]
        if not countries:
            countries = sorted(df['Country'].unique())

    base = pd.DataFrame({'Pottery_ID': sorted(df['pottery_id'].unique())})

    if 'Type' in df.columns:
        type_map = df.drop_duplicates('pottery_id').set_index('pottery_id')['Type'].to_dict()
        base['Type'] = base['Pottery_ID'].map(type_map)

    meta = []

    for emo in EMOTION_COLS:
        raw_maps = {}
        adj_maps = {}
        n_maps = {}

        for c in countries:
            sub = df[(df['Emotion'] == emo) & (df['Country'] == c)].copy()

            n_maps[c] = sub.groupby('pottery_id')['Response'].size().to_dict()

            raw, adj, used = _fit_lmm_adjusted_means_one_country(sub)

            raw_maps[c] = raw
            adj_maps[c] = adj

            meta.append({
                'Emotion': emo,
                'Country': c,
                'Observations': len(sub),
                'Subjects': sub['Subject_UID'].nunique(),
                'Potteries': sub['pottery_id'].nunique(),
                'LMM_Used': used,
                'Model': 'LMM' if used else 'Raw mean fallback'
            })

        for c in countries:
            base[f'Raw_{emo}_{c}'] = base['Pottery_ID'].map(raw_maps[c])
            base[f'LMM_Adjusted_{emo}_{c}'] = base['Pottery_ID'].map(adj_maps[c])
            base[f'N_{emo}_{c}'] = base['Pottery_ID'].map(n_maps[c]).fillna(0).astype(int)

        if 'JAPAN' in countries and 'MALAYSIA' in countries:
            base[f'LMM_Diff_{emo}_JPN_minus_MYS'] = (
                base[f'LMM_Adjusted_{emo}_JAPAN'] -
                base[f'LMM_Adjusted_{emo}_MALAYSIA']
            )
            base[f'Raw_Diff_{emo}_JPN_minus_MYS'] = (
                base[f'Raw_{emo}_JAPAN'] -
                base[f'Raw_{emo}_MALAYSIA']
            )

    meta_df = pd.DataFrame(meta)

    fixed_cols = ['Pottery_ID'] + (['Type'] if 'Type' in base.columns else [])
    base = base[fixed_cols + [c for c in base.columns if c not in fixed_cols]]

    return base, meta_df


# ==========================================
# 6. STATISTICAL TESTS: ANOVA
# ==========================================
def run_anova_on_groups(df, group_col, metric_cols):
    results = []
    is_long_format = 'Emotion' in df.columns and 'Response' in df.columns

    for emo in metric_cols:
        if is_long_format:
            emo_df = df[df['Emotion'] == emo]
            group_data = {
                name: group['Response'].dropna().values
                for name, group in emo_df.groupby(group_col)
                if len(group['Response'].dropna()) > 0
            }
        else:
            group_data = {
                name: group[emo].dropna().values
                for name, group in df.groupby(group_col)
                if len(group[emo].dropna()) > 0
            }

        valid_groups = {name: vals for name, vals in group_data.items() if len(vals) >= 1}

        if len(valid_groups) >= 2:
            arrays = list(valid_groups.values())
            total_n = sum(len(arr) for arr in arrays)

            d1 = len(arrays) - 1
            d2 = total_n - len(arrays)

            if d2 >= 1:
                group_details = []

                for name, arr in valid_groups.items():
                    display_name = (
                        'False' if str(name) in ('0.0', '0')
                        else ('True' if str(name) in ('1.0', '1') else str(name))
                    )
                    std_txt = f"{np.std(arr, ddof=1):.4f}" if len(arr) > 1 else "n/a"
                    group_details.append(
                        f"{display_name} (N={len(arr)}): Mean={np.mean(arr):.4f}, Std={std_txt}"
                    )

                try:
                    f_stat, _ = stats.f_oneway(*arrays)
                    f_p = hp_p_from_f(f_stat, d1, d2)
                except Exception:
                    f_stat, f_p = np.nan, np.nan

                results.append({
                    'Emotion': emo,
                    'Total_N': total_n,
                    'Group_Details': " | ".join(group_details),
                    'ANOVA_F_Stat': f_stat,
                    'ANOVA_df1': d1,
                    'ANOVA_df2': d2,
                    'ANOVA_p_value': f_p,
                    'ANOVA_Significant': p_is_significant(f_p)
                })

    return pd.DataFrame(results)


# ==========================================
# 7. STATISTICAL TESTS: LMM
# ==========================================
def run_lmm_on_long(long_df, fixed_effect_col, is_categorical=True):
    results = []
    df = long_df.copy()
    df.dropna(subset=['Response', 'Emotion', fixed_effect_col, 'Subject'], inplace=True)

    if len(df) == 0:
        return pd.DataFrame()

    # Country-nested subject ID prevents Japan/Malaysia subject mixing
    if 'Subject_UID' not in df.columns:
        if 'Country' in df.columns:
            df['Subject_UID'] = df['Country'].astype(str).str.upper() + '::' + df['Subject'].astype(str)
        else:
            df['Subject_UID'] = df['Subject'].astype(str)
    else:
        df['Subject_UID'] = df['Subject_UID'].astype(str)

    df[fixed_effect_col] = df[fixed_effect_col].astype(str)
    df['Subject'] = df['Subject'].astype(str)

    counts = df.groupby([fixed_effect_col, 'Emotion']).size()
    valid_combos = counts[counts >= 1].index
    df = df.set_index([fixed_effect_col, 'Emotion']).loc[valid_combos].reset_index()

    for emo in EMOTION_COLS:
        emo_df = df[df['Emotion'] == emo].copy()
        unique_vals = emo_df[fixed_effect_col].unique()

        if len(unique_vals) < 2:
            continue

        total_n = len(emo_df)

        group_details = []
        for name, grp in emo_df.groupby(fixed_effect_col):
            display_name = (
                'Absent' if str(name) in ['0', '0.0']
                else ('Present' if str(name) in ['1', '1.0'] else str(name))
            )
            group_details.append(
                f"{display_name} (N={len(grp)}): Mean={grp['Response'].mean():.4f}"
            )

        formula = f"Response ~ C({fixed_effect_col})"

        lmm_success = False
        f_stat, p_val = np.nan, np.nan
        subj_var, resid_var = np.nan, np.nan
        model_used = 'Failed'

        try:
            if emo_df['Subject_UID'].nunique() >= 2:
                model = smf.mixedlm(
                    formula,
                    emo_df,
                    groups=emo_df["Subject_UID"]
                )
                result = model.fit(reml=True, method='powell', maxiter=100)

                summ = result.summary().tables[1]
                target_row = None

                for idx in summ.index:
                    if fixed_effect_col in str(idx):
                        target_row = idx
                        break

                if target_row is not None:
                    t_stat = float(summ.loc[target_row, 'z'])
                    p_val = hp_p_from_z(t_stat)
                    f_stat = t_stat ** 2

                    cov_struct = result.cov_re
                    subj_var = cov_struct.iloc[0, 0] if not cov_struct.empty else 0.0
                    resid_var = result.scale

                    lmm_success = True
                    model_used = 'LMM_SubjectNestedByCountryWhenAvailable'

        except Exception:
            pass

        if not lmm_success:
            try:
                ols_model = smf.ols(formula, data=emo_df).fit()
                anova_table = sm.stats.anova_lm(ols_model, typ=2)

                fe_row = [r for r in anova_table.index if fixed_effect_col in r][0]

                f_stat = anova_table.loc[fe_row, 'F']
                d1 = anova_table.loc[fe_row, 'df']
                d2 = anova_table.loc['Residual', 'df'] if 'Residual' in anova_table.index else np.nan

                p_val = hp_p_from_f(f_stat, d1, d2)

                subj_var = np.nan
                resid_var = (
                    anova_table.loc['Residual', 'sum_sq'] / d2
                    if 'Residual' in anova_table.index and pd.notna(d2) and d2 != 0
                    else np.nan
                )

                model_used = 'OLS_ANOVA_Fallback'

            except Exception:
                model_used = 'Failed'

        results.append({
            'Emotion': emo,
            'Total_Observations': total_n,
            'Group_Details': " | ".join(group_details),
            'LMM_F_Stat': f_stat,
            'LMM_p_value': p_val,
            'LMM_Significant': p_is_significant(p_val),
            'Subject_Variance': subj_var,
            'Residual_Variance': resid_var,
            'ICC': (
                subj_var / (subj_var + resid_var)
                if pd.notna(subj_var) and pd.notna(resid_var) and (subj_var + resid_var) > 0
                else np.nan
            ),
            'Model_Used': model_used
        })

    return pd.DataFrame(results)


# ==========================================
# 8. REGION COMPARISON HELPER
# ==========================================
def analyze_region_subset(df_events, df_dur, mask, subset_name):
    """
    Runs ANOVA and LMM comparing MALAYSIA vs JAPAN on a specific subset of data.

    This version explicitly matches duration rows to event rows using key columns,
    instead of relying only on row order.
    """
    df_e_sub = df_events.loc[mask].copy()

    if df_e_sub.empty or 'Country' not in df_e_sub.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if df_e_sub['Country'].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    key_cols = [
        c for c in ['pottery_id', 'session_id', 'Subject', 'Language', 'Country', 'Emotion']
        if c in df_e_sub.columns and c in df_dur.columns
    ]

    if key_cols:
        df_d_sub = df_dur.merge(
            df_e_sub[key_cols].drop_duplicates(),
            on=key_cols,
            how='inner'
        )
    else:
        df_d_sub = df_dur.loc[mask].copy()

    # Event-based analyses
    a_e = run_anova_on_groups(df_e_sub, 'Country', EMOTION_COLS)
    l_e = run_lmm_on_long(df_e_sub, 'Country')

    # Duration-based analyses
    if df_d_sub.empty or df_d_sub['Country'].nunique() < 2:
        return a_e, pd.DataFrame(), l_e, pd.DataFrame()

    a_d = run_anova_on_groups(df_d_sub, 'Country', EMOTION_COLS)
    l_d = run_lmm_on_long(df_d_sub, 'Country')

    return a_e, a_d, l_e, l_d


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("=" * 80)
    print("STATISTICAL ANALYSIS: ANOVA + LMM + JAPAN/MALAYSIA REGION OUTPUT")
    print("=" * 80)

    df_qa = load_multilingual_dataset([DATASET_ROOT_MALAYSIA, DATASET_ROOT_JAPAN])

    if df_qa.empty:
        print("Error: No data loaded.")
        return

    # Optional audit: check whether subject IDs overlap across countries
    if {'Country', 'Subject'}.issubset(df_qa.columns):
        japan_subjects = set(
            df_qa.loc[df_qa['Country'].astype(str).str.upper() == 'JAPAN', 'Subject'].astype(str)
        )
        malaysia_subjects = set(
            df_qa.loc[df_qa['Country'].astype(str).str.upper() == 'MALAYSIA', 'Subject'].astype(str)
        )
        subject_overlap = japan_subjects & malaysia_subjects

        print("Subject-ID overlap between Japan and Malaysia:")
        if subject_overlap:
            print(subject_overlap)
        else:
            print("None")

    pot_events, pot_dur, sess_events_long, sess_dur_long = calculate_metrics_long(df_qa)

    features_file = get_features_path()

    pot_events_feat, feature_cols, shape_cols = attach_features(pot_events, features_file)
    pot_dur_feat, _, _ = attach_features(pot_dur, features_file)

    if features_file:
        sess_events_feat_long, _, _ = attach_features_to_long(sess_events_long, features_file)
        sess_dur_feat_long, _, _ = attach_features_to_long(sess_dur_long, features_file)
    else:
        sess_events_feat_long = sess_events_long.copy()
        sess_dur_feat_long = sess_dur_long.copy()

    # Country-nested subject IDs for all long-format LMMs
    sess_events_feat_long['Subject_UID'] = (
        sess_events_feat_long['Country'].astype(str).str.upper() + '::' +
        sess_events_feat_long['Subject'].astype(str)
    )
    sess_dur_feat_long['Subject_UID'] = (
        sess_dur_feat_long['Country'].astype(str).str.upper() + '::' +
        sess_dur_feat_long['Subject'].astype(str)
    )

    # ==========================================================
    # Sheet-1 style LMM adjusted means: combined and by country
    # ==========================================================
    lmm_means_events = get_lmm_adjusted_means(sess_events_feat_long)
    lmm_means_dur = get_lmm_adjusted_means(sess_dur_feat_long)

    pot_events_with_lmm = pot_events_feat.merge(lmm_means_events, on='Pottery_ID', how='left')
    pot_dur_with_lmm = pot_dur_feat.merge(lmm_means_dur, on='Pottery_ID', how='left')

    print("\nRunning 3-0: Japan/Malaysia Sheet-1 style LMM adjusted means...")
    region_events_lmm_sheet, region_events_lmm_meta = get_lmm_adjusted_means_by_country(sess_events_feat_long)
    region_dur_lmm_sheet, region_dur_lmm_meta = get_lmm_adjusted_means_by_country(sess_dur_feat_long)

    # ==========================================================
    # Standard 2-x Analyses
    # ==========================================================
    print("\nRunning 2-1: Artifact Type Analysis...")
    a21_e_anova = run_anova_on_groups(pot_events_feat, 'Type', EMOTION_COLS)
    a21_d_anova = run_anova_on_groups(pot_dur_feat, 'Type', EMOTION_COLS)
    a21_e_lmm = run_lmm_on_long(sess_events_feat_long, 'Type')
    a21_d_lmm = run_lmm_on_long(sess_dur_feat_long, 'Type')

    print("Running 2-2: Typology Analysis...")
    a22_e_anova_list, a22_d_anova_list = [], []
    a22_e_lmm_list, a22_d_lmm_list = [], []

    for shape in shape_cols:
        res_e_a = run_anova_on_groups(pot_events_feat, shape, EMOTION_COLS)
        res_d_a = run_anova_on_groups(pot_dur_feat, shape, EMOTION_COLS)

        if not res_e_a.empty:
            res_e_a.insert(0, 'Typology', shape.replace('SHAPE_TYPE_', ''))
            a22_e_anova_list.append(res_e_a)

        if not res_d_a.empty:
            res_d_a.insert(0, 'Typology', shape.replace('SHAPE_TYPE_', ''))
            a22_d_anova_list.append(res_d_a)

        res_e_l = run_lmm_on_long(sess_events_feat_long, shape)
        res_d_l = run_lmm_on_long(sess_dur_feat_long, shape)

        if not res_e_l.empty:
            res_e_l.insert(0, 'Typology', shape.replace('SHAPE_TYPE_', ''))
            a22_e_lmm_list.append(res_e_l)

        if not res_d_l.empty:
            res_d_l.insert(0, 'Typology', shape.replace('SHAPE_TYPE_', ''))
            a22_d_lmm_list.append(res_d_l)

    a22_e_anova = pd.concat(a22_e_anova_list, ignore_index=True) if a22_e_anova_list else pd.DataFrame()
    a22_d_anova = pd.concat(a22_d_anova_list, ignore_index=True) if a22_d_anova_list else pd.DataFrame()
    a22_e_lmm = pd.concat(a22_e_lmm_list, ignore_index=True) if a22_e_lmm_list else pd.DataFrame()
    a22_d_lmm = pd.concat(a22_d_lmm_list, ignore_index=True) if a22_d_lmm_list else pd.DataFrame()

    print("Running 2-3: Language Analysis...")
    a23_e_anova = run_anova_on_groups(sess_events_long, 'Language', EMOTION_COLS)
    a23_d_anova = run_anova_on_groups(sess_dur_long, 'Language', EMOTION_COLS)
    a23_e_lmm = run_lmm_on_long(sess_events_long, 'Language')
    a23_d_lmm = run_lmm_on_long(sess_dur_long, 'Language')

    print("Running 2-4: Feature Analysis...")
    a24_e_anova_list, a24_d_anova_list = [], []
    a24_e_lmm_list, a24_d_lmm_list = [], []

    for feat in feature_cols:
        res_e_a = run_anova_on_groups(pot_events_feat, feat, EMOTION_COLS)
        res_d_a = run_anova_on_groups(pot_dur_feat, feat, EMOTION_COLS)

        if not res_e_a.empty:
            res_e_a.insert(0, 'Feature', feat)
            a24_e_anova_list.append(res_e_a)

        if not res_d_a.empty:
            res_d_a.insert(0, 'Feature', feat)
            a24_d_anova_list.append(res_d_a)

        res_e_l = run_lmm_on_long(sess_events_feat_long, feat)
        res_d_l = run_lmm_on_long(sess_dur_feat_long, feat)

        if not res_e_l.empty:
            res_e_l.insert(0, 'Feature', feat)
            a24_e_lmm_list.append(res_e_l)

        if not res_d_l.empty:
            res_d_l.insert(0, 'Feature', feat)
            a24_d_lmm_list.append(res_d_l)

    a24_e_anova = pd.concat(a24_e_anova_list, ignore_index=True) if a24_e_anova_list else pd.DataFrame()
    a24_d_anova = pd.concat(a24_d_anova_list, ignore_index=True) if a24_d_anova_list else pd.DataFrame()
    a24_e_lmm = pd.concat(a24_e_lmm_list, ignore_index=True) if a24_e_lmm_list else pd.DataFrame()
    a24_d_lmm = pd.concat(a24_d_lmm_list, ignore_index=True) if a24_d_lmm_list else pd.DataFrame()

    # ==========================================================
    # 3-x Region Comparisons: Japan vs Malaysia
    # ==========================================================
    print("\nRunning 3-1 to 3-4: Region Comparisons (Japan vs Malaysia)...")

    # 3-1 Pottery vs Pottery
    r_pot_e_a, r_pot_d_a, r_pot_e_l, r_pot_d_l = analyze_region_subset(
        sess_events_feat_long,
        sess_dur_feat_long,
        sess_events_feat_long['Type'] == 'Pottery',
        'Pottery'
    )

    # 3-2 Dogu vs Dogu
    r_dog_e_a, r_dog_d_a, r_dog_e_l, r_dog_d_l = analyze_region_subset(
        sess_events_feat_long,
        sess_dur_feat_long,
        sess_events_feat_long['Type'] == 'Dogu',
        'Dogu'
    )

    # 3-3 Typology vs Typology
    r_typ_e_a, r_typ_d_a, r_typ_e_l, r_typ_d_l = [], [], [], []

    for shape in shape_cols:
        mask = sess_events_feat_long[shape] == 1
        a_e, a_d, l_e, l_d = analyze_region_subset(
            sess_events_feat_long,
            sess_dur_feat_long,
            mask,
            shape
        )

        if not a_e.empty:
            a_e.insert(0, 'Typology', shape.replace('SHAPE_TYPE_', ''))
            r_typ_e_a.append(a_e)

        if not a_d.empty:
            a_d.insert(0, 'Typology', shape.replace('SHAPE_TYPE_', ''))
            r_typ_d_a.append(a_d)

        if not l_e.empty:
            l_e.insert(0, 'Typology', shape.replace('SHAPE_TYPE_', ''))
            r_typ_e_l.append(l_e)

        if not l_d.empty:
            l_d.insert(0, 'Typology', shape.replace('SHAPE_TYPE_', ''))
            r_typ_d_l.append(l_d)

    r_typ_e_a = pd.concat(r_typ_e_a, ignore_index=True) if r_typ_e_a else pd.DataFrame()
    r_typ_d_a = pd.concat(r_typ_d_a, ignore_index=True) if r_typ_d_a else pd.DataFrame()
    r_typ_e_l = pd.concat(r_typ_e_l, ignore_index=True) if r_typ_e_l else pd.DataFrame()
    r_typ_d_l = pd.concat(r_typ_d_l, ignore_index=True) if r_typ_d_l else pd.DataFrame()

    # 3-4 Feature vs Feature
    r_feat_e_a, r_feat_d_a, r_feat_e_l, r_feat_d_l = [], [], [], []

    for feat in feature_cols:
        mask = sess_events_feat_long[feat] == 1
        a_e, a_d, l_e, l_d = analyze_region_subset(
            sess_events_feat_long,
            sess_dur_feat_long,
            mask,
            feat
        )

        if not a_e.empty:
            a_e.insert(0, 'Feature', feat)
            r_feat_e_a.append(a_e)

        if not a_d.empty:
            a_d.insert(0, 'Feature', feat)
            r_feat_d_a.append(a_d)

        if not l_e.empty:
            l_e.insert(0, 'Feature', feat)
            r_feat_e_l.append(l_e)

        if not l_d.empty:
            l_d.insert(0, 'Feature', feat)
            r_feat_d_l.append(l_d)

    r_feat_e_a = pd.concat(r_feat_e_a, ignore_index=True) if r_feat_e_a else pd.DataFrame()
    r_feat_d_a = pd.concat(r_feat_d_a, ignore_index=True) if r_feat_d_a else pd.DataFrame()
    r_feat_e_l = pd.concat(r_feat_e_l, ignore_index=True) if r_feat_e_l else pd.DataFrame()
    r_feat_d_l = pd.concat(r_feat_d_l, ignore_index=True) if r_feat_d_l else pd.DataFrame()

    # ==========================================================
    # EXCEL EXPORT: NEW FILE
    # ==========================================================
    out_file = os.path.join(
        OUTPUT_DIR,
        "ANOVA_LMM_Japan_vs_Malaysia_Analysis.xlsx"
    )

    print(f"\nWriting to new Excel file: {out_file}...")

    with pd.ExcelWriter(out_file, engine='openpyxl') as writer:

        # 1-x Sheets
        pot_events_with_lmm.to_excel(writer, sheet_name='1-1_Norm_Events_LMM', index=False)
        pot_dur_with_lmm.to_excel(writer, sheet_name='1-2_Norm_Duration_LMM', index=False)

        # New 3-0 country-separated Sheet-1 style LMM sheets
        if not region_events_lmm_sheet.empty:
            region_events_lmm_sheet.to_excel(
                writer,
                sheet_name='3-0_Region_Ev_LMM_ByCountry',
                index=False
            )

        if not region_dur_lmm_sheet.empty:
            region_dur_lmm_sheet.to_excel(
                writer,
                sheet_name='3-0_Region_Dur_LMM_ByCountry',
                index=False
            )

        if not region_events_lmm_meta.empty:
            region_events_lmm_meta.to_excel(
                writer,
                sheet_name='3-0_Region_Ev_LMM_Meta',
                index=False
            )

        if not region_dur_lmm_meta.empty:
            region_dur_lmm_meta.to_excel(
                writer,
                sheet_name='3-0_Region_Dur_LMM_Meta',
                index=False
            )

        # 2-1 Object Type
        if not a21_e_anova.empty:
            a21_e_anova.to_excel(writer, sheet_name='2-1_ObjType_ANOVA_E', index=False)
        if not a21_d_anova.empty:
            a21_d_anova.to_excel(writer, sheet_name='2-1_ObjType_ANOVA_D', index=False)
        if not a21_e_lmm.empty:
            a21_e_lmm.to_excel(writer, sheet_name='2-1_ObjType_LMM_E', index=False)
        if not a21_d_lmm.empty:
            a21_d_lmm.to_excel(writer, sheet_name='2-1_ObjType_LMM_D', index=False)

        # 2-2 Typology
        if not a22_e_anova.empty:
            a22_e_anova.to_excel(writer, sheet_name='2-2_Typology_ANOVA_E', index=False)
        if not a22_d_anova.empty:
            a22_d_anova.to_excel(writer, sheet_name='2-2_Typology_ANOVA_D', index=False)
        if not a22_e_lmm.empty:
            a22_e_lmm.to_excel(writer, sheet_name='2-2_Typology_LMM_E', index=False)
        if not a22_d_lmm.empty:
            a22_d_lmm.to_excel(writer, sheet_name='2-2_Typology_LMM_D', index=False)

        # 2-3 Language
        if not a23_e_anova.empty:
            a23_e_anova.to_excel(writer, sheet_name='2-3_Language_ANOVA_E', index=False)
        if not a23_d_anova.empty:
            a23_d_anova.to_excel(writer, sheet_name='2-3_Language_ANOVA_D', index=False)
        if not a23_e_lmm.empty:
            a23_e_lmm.to_excel(writer, sheet_name='2-3_Language_LMM_E', index=False)
        if not a23_d_lmm.empty:
            a23_d_lmm.to_excel(writer, sheet_name='2-3_Language_LMM_D', index=False)

        # 2-4 Features
        if not a24_e_anova.empty:
            a24_e_anova.to_excel(writer, sheet_name='2-4_Features_ANOVA_E', index=False)
        if not a24_d_anova.empty:
            a24_d_anova.to_excel(writer, sheet_name='2-4_Features_ANOVA_D', index=False)
        if not a24_e_lmm.empty:
            a24_e_lmm.to_excel(writer, sheet_name='2-4_Features_LMM_E', index=False)
        if not a24_d_lmm.empty:
            a24_d_lmm.to_excel(writer, sheet_name='2-4_Features_LMM_D', index=False)

        # 3-1 Region: Pottery
        if not r_pot_e_a.empty:
            r_pot_e_a.to_excel(writer, sheet_name='3-1_Region_Pottery_ANOVA_E', index=False)
        if not r_pot_d_a.empty:
            r_pot_d_a.to_excel(writer, sheet_name='3-1_Region_Pottery_ANOVA_D', index=False)
        if not r_pot_e_l.empty:
            r_pot_e_l.to_excel(writer, sheet_name='3-1_Region_Pottery_LMM_E', index=False)
        if not r_pot_d_l.empty:
            r_pot_d_l.to_excel(writer, sheet_name='3-1_Region_Pottery_LMM_D', index=False)

        # 3-2 Region: Dogu
        if not r_dog_e_a.empty:
            r_dog_e_a.to_excel(writer, sheet_name='3-2_Region_Dogu_ANOVA_E', index=False)
        if not r_dog_d_a.empty:
            r_dog_d_a.to_excel(writer, sheet_name='3-2_Region_Dogu_ANOVA_D', index=False)
        if not r_dog_e_l.empty:
            r_dog_e_l.to_excel(writer, sheet_name='3-2_Region_Dogu_LMM_E', index=False)
        if not r_dog_d_l.empty:
            r_dog_d_l.to_excel(writer, sheet_name='3-2_Region_Dogu_LMM_D', index=False)

        # 3-3 Region: Typology
        if not r_typ_e_a.empty:
            r_typ_e_a.to_excel(writer, sheet_name='3-3_Region_Typology_ANOVA_E', index=False)
        if not r_typ_d_a.empty:
            r_typ_d_a.to_excel(writer, sheet_name='3-3_Region_Typology_ANOVA_D', index=False)
        if not r_typ_e_l.empty:
            r_typ_e_l.to_excel(writer, sheet_name='3-3_Region_Typology_LMM_E', index=False)
        if not r_typ_d_l.empty:
            r_typ_d_l.to_excel(writer, sheet_name='3-3_Region_Typology_LMM_D', index=False)

        # 3-4 Region: Features
        if not r_feat_e_a.empty:
            r_feat_e_a.to_excel(writer, sheet_name='3-4_Region_Features_ANOVA_E', index=False)
        if not r_feat_d_a.empty:
            r_feat_d_a.to_excel(writer, sheet_name='3-4_Region_Features_ANOVA_D', index=False)
        if not r_feat_e_l.empty:
            r_feat_e_l.to_excel(writer, sheet_name='3-4_Region_Features_LMM_E', index=False)
        if not r_feat_d_l.empty:
            r_feat_d_l.to_excel(writer, sheet_name='3-4_Region_Features_LMM_D', index=False)

    print("✓ Analysis successfully exported.")
    print("\nSHEETS CREATED:")
    print(" - 1-x: 93 items (Raw + LMM Adjusted)")
    print(" - 3-0: Japan/Malaysia Sheet-1 style LMM adjusted means")
    print(" - 2-x: Standard ANOVA/LMM (Type, Typology, Language, Features)")
    print(" - 3-1: Region Comparison (Pottery: Japan vs Malaysia)")
    print(" - 3-2: Region Comparison (Dogu: Japan vs Malaysia)")
    print(" - 3-3: Region Comparison (Typologies: Japan vs Malaysia)")
    print(" - 3-4: Region Comparison (Features: Japan vs Malaysia)")
    print(f"\nNew Excel file saved as:\n{out_file}")


if __name__ == "__main__":
    main()