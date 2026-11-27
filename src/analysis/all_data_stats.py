import os
from pathlib import Path
import numpy as np
import pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from LMM_stats_Japan_vs_Malaysia import (
    DATASET_ROOT_MALAYSIA,
    DATASET_ROOT_JAPAN,
    EMOTION_COLS,
    DOGU_PREFIXES,
    EMOTION_MAP,
    TARGET_LANGUAGES,
    calculate_metrics_long as orig_calculate_metrics_long,
    get_features_path as orig_get_features_path,
    attach_features as orig_attach_features,
    attach_features_to_long as orig_attach_features_to_long,
    get_lmm_adjusted_means as orig_get_lmm_adjusted_means,
    get_lmm_adjusted_means_by_country as
    orig_get_lmm_adjusted_means_by_country,
    run_anova_on_groups as orig_run_anova_on_groups,
    run_lmm_on_long as orig_run_lmm_on_long,
    analyze_region_subset as orig_analyze_region_subset,
)
from average_by_feature import (
    ASSIGNED_NUMBERS_DICT,
    PLOT_TITLE,
    SHAPE_TYPE_LABELS,
    FEATURE_NAME_MAPPING,
)
from table_view import (
    parse_means_from_details,
    format_cell,
    format_cell_absent_present,
)

OUTPUT_XLSX = os.path.join(
    "anova_lmm_statistical_analysis",
    "Public_Aggregated_Quantification_Tables.xlsx",
)
FEATURES_XLSX_PATH = "./src/DS_Labels_Cleaned_backup.xlsx"


def resolve_pottery_id_with_num(raw_folder_name: str) -> str:
    name = str(raw_folder_name).strip()
    if "(" in name and name.endswith(")"):
        return name
    base = name.split("(")[0].strip()
    if base in ASSIGNED_NUMBERS_DICT:
        return f"{base}({ASSIGNED_NUMBERS_DICT[base]})"
    return name


def parse_session_language(lang_file: Path, country: str):
    language = None
    if lang_file.exists():
        try:
            text = lang_file.read_text(encoding="utf-8", errors="replace").strip().upper()
            if "ENGLISH" in text or "EN " in text or text == "EN":
                language = "ENGLISH"
            elif "MALAY" in text or "BM " in text or text == "BM":
                language = "MALAY"
            elif "CHINESE" in text or "MANDARIN" in text:
                language = "CHINESE"
            elif "TAMIL" in text:
                language = "TAMIL"
            elif "JAPANESE" in text or "JP " in text or text == "JP":
                language = "JAPANESE"
            elif text:
                language = "OTHER"
        except Exception:
            pass
    if language is None and country == "JAPAN":
        language = "JAPANESE"
    return language


def load_clean_features_df(features_path: str) -> pd.DataFrame:
    df = pd.read_excel(features_path)
    df["Pottery_ID"] = (df.iloc[:,
                                0].astype(str).str.replace(
                                    ".ply",
                                    "",
                                    regex=False).str.strip().apply(
                                        resolve_pottery_id_with_num))
    df = df.drop_duplicates(subset="Pottery_ID", keep="first")
    indicator_cols = [
        c for c in df.columns
        if c.startswith(("HAS_", "NUMBER_OF_PERTRUSIONS_", "SHAPE_TYPE_"))
    ]
    for col in indicator_cols:
        df[col] = (pd.to_numeric(
            df[col].replace({
                "TRUE": 1,
                "FALSE": 0,
                True: 1,
                False: 0
            }),
            errors="coerce",
        ).fillna(0).astype(int))
    return df


def format_typology_english(shape_col: str) -> str:
    if shape_col in SHAPE_TYPE_LABELS:
        return SHAPE_TYPE_LABELS[shape_col]
    prefixed = shape_col if shape_col.startswith(
        "SHAPE_TYPE_") else f"SHAPE_TYPE_{shape_col}"
    return SHAPE_TYPE_LABELS.get(prefixed,
                                 shape_col.replace("SHAPE_TYPE_",
                                                   ""))


def _typology_display_name(row) -> str:
    jp = str(row.get("Typology", "")).strip()
    en = str(row.get("Typology_English", "")).strip()
    return f"{jp} ({en})" if en and en != jp else jp


# ==============================================================================
# STEP 0: RAW LOADER -> SHEET 0-1 & SHEET 0-2
# ==============================================================================
def load_raw_interactions(root_dirs, filter_target_languages: bool = False):
    qa_records = []
    jp_root_resolved = Path(DATASET_ROOT_JAPAN).resolve()
    my_root_resolved = Path(DATASET_ROOT_MALAYSIA).resolve()

    attrition = {
        "JAPAN": {
            "initial_qa": 0,
            "dropped_lang": 0
        },
        "MALAYSIA": {
            "initial_qa": 0,
            "dropped_lang": 0
        },
    }

    for root_dir in root_dirs:
        root_path = Path(root_dir)
        if not root_path.exists():
            print(f"Warning: Directory not found: {root_dir}")
            continue

        resolved = root_path.resolve()
        if resolved == jp_root_resolved:
            country = "JAPAN"
        elif resolved == my_root_resolved:
            country = "MALAYSIA"
        else:
            raise ValueError(f"Unrecognized dataset root path: {root_dir}")

        for group_path in sorted(root_path.iterdir()):
            if not group_path.is_dir():
                continue
            group_id = group_path.name

            for session_path in sorted(group_path.iterdir()):
                if not session_path.is_dir():
                    continue

                language = parse_session_language(
                    session_path / "language.txt",
                    country)

                for pottery_path in sorted(session_path.iterdir()):
                    if not pottery_path.is_dir():
                        continue

                    qa_file = pottery_path / "qa_corrected.csv"
                    if not qa_file.exists():
                        continue

                    attrition[country]["initial_qa"] += 1

                    if filter_target_languages and language not in TARGET_LANGUAGES:
                        attrition[country]["dropped_lang"] += 1
                        continue

                    transcript_file = pottery_path / "final_transcript.txt"
                    transcript_text = (transcript_file.read_text(
                        encoding="utf-8",
                        errors="replace").strip()
                                       if transcript_file.exists() else "")

                    try:
                        df_temp = pd.read_csv(qa_file, encoding="utf-8")
                        df_temp["timestamp"] = pd.to_numeric(
                            df_temp["timestamp"],
                            errors="coerce")
                        df_temp.dropna(subset=["timestamp"], inplace=True)

                        df_temp["pottery_id"] = resolve_pottery_id_with_num(
                            pottery_path.name)
                        df_temp["session_id"] = session_path.name
                        df_temp[
                            "Language"] = language if language is not None else "UNKNOWN"
                        df_temp["Group_ID"] = group_id
                        # Individual participant = (Group_ID, session_id) -> 143 Japan + 152 Malaysia = 295
                        df_temp["Subject"] = f"{group_id}::{session_path.name}"
                        df_temp["Country"] = country
                        df_temp["transcript"] = transcript_text
                        qa_records.append(df_temp)
                    except Exception as e:
                        print(f"Error loading {qa_file}: {e}")

    df_out = pd.concat(qa_records,
                       ignore_index=True) if qa_records else pd.DataFrame()
    return df_out, attrition


def build_sheet_0_tables(df_qa: pd.DataFrame):
    df = df_qa.copy()
    df["short_answer"] = df["answer"].astype(str).str.strip().map(EMOTION_MAP)
    df.dropna(subset=[
        "short_answer",
        "pottery_id",
        "session_id",
        "Subject",
        "Country"
    ],
              inplace=True)
    df.sort_values(by=["pottery_id", "session_id", "timestamp"], inplace=True)

    df["time_diff"] = df.groupby(["pottery_id",
                                  "session_id"])["timestamp"].diff()
    emotion_changed = df["short_answer"] != df.groupby(
        ["pottery_id",
         "session_id"])["short_answer"].shift()
    time_gap_exceeded = df["time_diff"] > 0.05
    df["block_id"] = (emotion_changed | time_gap_exceeded).cumsum()

    group_keys = [
        "pottery_id",
        "session_id",
        "Group_ID",
        "Subject",
        "Language",
        "Country"
    ]
    blocks = (df.groupby(group_keys + ["block_id"],
                         as_index=False).agg(
                             emotion=("short_answer",
                                      "first"),
                             start_time=("timestamp",
                                         "min"),
                             end_time=("timestamp",
                                       "max"),
                             event_count=("timestamp",
                                          "count"),
                         ))
    blocks["duration_ms"] = (blocks["end_time"] -
                             blocks["start_time"]) * 1000.0

    ev_raw = blocks.groupby(group_keys +
                            ["emotion"])["event_count"].sum().unstack(
                                fill_value=0).reset_index()
    dur_raw = blocks.groupby(group_keys +
                             ["emotion"])["duration_ms"].sum().unstack(
                                 fill_value=0.0).reset_index()

    for emo in EMOTION_COLS:
        if emo not in ev_raw.columns:
            ev_raw[emo] = 0
        if emo not in dur_raw.columns:
            dur_raw[emo] = 0.0

    sess_df = ev_raw[group_keys + EMOTION_COLS].merge(
        dur_raw[group_keys + EMOTION_COLS],
        on=group_keys,
        suffixes=("_ev",
                  "_dur"),
    )

    # Anonymize Participant IDs (JPN_P001..JPN_P143, MYS_P001..MYS_P152)
    sess_df["Participant_UID"] = sess_df["Country"].str.upper(
    ) + "::" + sess_df["Subject"].astype(str)
    subj_map = {}
    jp_idx, my_idx = 1, 1
    for uid in sorted(sess_df["Participant_UID"].unique()):
        if uid.startswith("JAPAN::"):
            subj_map[uid] = f"JPN_P{jp_idx:03d}"
            jp_idx += 1
        else:
            subj_map[uid] = f"MYS_P{my_idx:03d}"
            my_idx += 1

    sess_df["Anon_Participant_ID"] = sess_df["Participant_UID"].map(subj_map)
    sess_df.sort_values(by=["pottery_id",
                            "session_id",
                            "Anon_Participant_ID"],
                        inplace=True)
    sess_df.reset_index(drop=True, inplace=True)
    sess_df["Anon_Instance_ID"] = [
        f"INST_{i+1:04d}" for i in range(len(sess_df))
    ]

    ev_mat = sess_df[[f"{e}_ev" for e in EMOTION_COLS]].values
    dur_mat = sess_df[[f"{e}_dur" for e in EMOTION_COLS]].values

    sess_df["Total_Valid_Events"] = ev_mat.sum(axis=1)
    sess_df["Total_Gaze_Duration_ms"] = dur_mat.sum(axis=1)
    sess_df["Artifact_Type"] = sess_df["pottery_id"].apply(
        lambda x: "Dogu"
        if any(p in str(x) for p in DOGU_PREFIXES) else "Pottery")

    for emo in EMOTION_COLS:
        sess_df[f"Events_Raw_{emo}"] = sess_df[f"{emo}_ev"]
        sess_df[f"Events_Norm_{emo}"] = (
            sess_df[f"{emo}_ev"] /
            sess_df["Total_Valid_Events"].replace(0,
                                                  np.nan)).fillna(0.0)
        sess_df[f"Duration_ms_{emo}"] = sess_df[f"{emo}_dur"]
        sess_df[f"Duration_Norm_{emo}"] = (
            sess_df[f"{emo}_dur"] /
            sess_df["Total_Gaze_Duration_ms"].replace(0,
                                                      np.nan)).fillna(0.0)

    trans_meta = (df.groupby(
        ["pottery_id",
         "session_id",
         "Subject",
         "Country"],
        as_index=False).agg(Transcript=("transcript",
                                        "first")))
    trans_meta["Transcript"] = trans_meta["Transcript"].fillna("").astype(
        str).str.strip()
    trans_meta["Has_Transcript"] = (trans_meta["Transcript"].str.len()
                                    > 0).astype(int)
    trans_meta["Transcript_Char_Length"] = trans_meta["Transcript"].str.len()
    trans_meta["Transcript_Word_Token_Count"] = trans_meta["Transcript"].apply(
        lambda t: len(t.split()) if t else 0)

    sess_df = sess_df.merge(
        trans_meta,
        on=["pottery_id",
            "session_id",
            "Subject",
            "Country"],
        how="left",
    )
    sess_df.rename(columns={"pottery_id": "Pottery_ID"}, inplace=True)

    ordered_cols = ([
        "Anon_Instance_ID",
        "Anon_Participant_ID",
        "Country",
        "Language",
        "Pottery_ID",
        "Artifact_Type",
        "Total_Valid_Events",
        "Total_Gaze_Duration_ms",
    ] + [f"Events_Raw_{e}"
         for e in EMOTION_COLS] + [f"Events_Norm_{e}" for e in EMOTION_COLS] +
                    [f"Duration_ms_{e}" for e in EMOTION_COLS] +
                    [f"Duration_Norm_{e}" for e in EMOTION_COLS] + [
                        "Has_Transcript",
                        "Transcript_Char_Length",
                        "Transcript_Word_Token_Count",
                        "Transcript",
                    ])
    sheet_0_1 = sess_df[ordered_cols].copy()
    sheet_0_1_internal = sess_df[ordered_cols + ["session_id",
                                                 "Group_ID"]].copy()

    norm_ev_cols = [f"Events_Norm_{e}" for e in EMOTION_COLS]
    norm_dur_cols = [f"Duration_Norm_{e}" for e in EMOTION_COLS]
    sheet_0_2 = (sheet_0_1.groupby(
        ["Anon_Participant_ID",
         "Country",
         "Language"],
        as_index=False).agg(
            Sessions_Completed=("Anon_Instance_ID",
                                "count"),
            Mean_Valid_Events_Per_Session=("Total_Valid_Events",
                                           "mean"),
            Mean_Gaze_Duration_ms_Per_Session=("Total_Gaze_Duration_ms",
                                               "mean"),
            **{c: (c,
                   "mean")
               for c in norm_ev_cols + norm_dur_cols},
        ))
    return sheet_0_1, sheet_0_1_internal, sheet_0_2


# ==============================================================================
# STEP 1: COMPUTE ALL DOWNSTREAM SHEETS STRICTLY FROM SHEET 0-1
# ==============================================================================
def compute_all_sheets_from_sheet_0_1(sheet_0_1: pd.DataFrame,
                                      features_df: pd.DataFrame):
    feature_cols = [c for c in features_df.columns if c.startswith("HAS_")]
    shape_cols = [
        c for c in features_df.columns
        if c.startswith("SHAPE_TYPE_") and "NAN" not in c.upper()
    ]
    meta_cols = ["Pottery_ID"] + shape_cols + feature_cols

    ev_norm_cols = [f"Events_Norm_{e}" for e in EMOTION_COLS]
    dur_norm_cols = [f"Duration_Norm_{e}" for e in EMOTION_COLS]

    pot_events = (sheet_0_1.groupby(
        "Pottery_ID",
        as_index=False)[ev_norm_cols].mean().rename(columns={
            f"Events_Norm_{e}": e
            for e in EMOTION_COLS
        }).sort_values("Pottery_ID").reset_index(drop=True))
    pot_events["Samples"] = range(1, len(pot_events) + 1)
    pot_events["Type"] = pot_events["Pottery_ID"].apply(
        lambda x: "Dogu"
        if any(p in str(x) for p in DOGU_PREFIXES) else "Pottery")
    pot_events_feat = pot_events[["Samples",
                                  "Pottery_ID",
                                  "Type"] + EMOTION_COLS].merge(
                                      features_df[meta_cols],
                                      on="Pottery_ID",
                                      how="left")
    for col in shape_cols + feature_cols:
        pot_events_feat[col] = pot_events_feat[col].fillna(0).astype(int)

    pot_dur = (sheet_0_1.groupby(
        "Pottery_ID",
        as_index=False)[dur_norm_cols].mean().rename(columns={
            f"Duration_Norm_{e}": e
            for e in EMOTION_COLS
        }).sort_values("Pottery_ID").reset_index(drop=True))
    pot_dur["Samples"] = range(1, len(pot_dur) + 1)
    pot_dur["Type"] = pot_dur["Pottery_ID"].apply(lambda x: "Dogu" if any(
        p in str(x) for p in DOGU_PREFIXES) else "Pottery")
    pot_dur_feat = pot_dur[["Samples",
                            "Pottery_ID",
                            "Type"] + EMOTION_COLS].merge(
                                features_df[meta_cols],
                                on="Pottery_ID",
                                how="left")
    for col in shape_cols + feature_cols:
        pot_dur_feat[col] = pot_dur_feat[col].fillna(0).astype(int)

    id_vars = [
        c for c in [
            "Anon_Instance_ID", "Anon_Participant_ID", "Country", "Language",
            "Pottery_ID", "Artifact_Type", "session_id"
        ] if c in sheet_0_1.columns
    ]

    sess_events_long = sheet_0_1.melt(
        id_vars=id_vars,
        value_vars=ev_norm_cols,
        var_name="Emotion",
        value_name="Response",
    )
    sess_events_long["Emotion"] = sess_events_long["Emotion"].str.replace(
        "Events_Norm_",
        "",
        regex=False)
    sess_events_long.rename(
        columns={
            "Pottery_ID": "pottery_id",
            "Anon_Participant_ID": "Subject",
            "Artifact_Type": "Type"
        },
        inplace=True,
    )
    sess_events_long["Subject_UID"] = sess_events_long["Subject"]

    sess_dur_long = sheet_0_1.melt(
        id_vars=id_vars,
        value_vars=dur_norm_cols,
        var_name="Emotion",
        value_name="Response",
    )
    sess_dur_long["Emotion"] = sess_dur_long["Emotion"].str.replace(
        "Duration_Norm_",
        "",
        regex=False)
    sess_dur_long.rename(
        columns={
            "Pottery_ID": "pottery_id",
            "Anon_Participant_ID": "Subject",
            "Artifact_Type": "Type"
        },
        inplace=True,
    )
    sess_dur_long["Subject_UID"] = sess_dur_long["Subject"]

    feat_lookup = features_df[meta_cols].rename(
        columns={"Pottery_ID": "pottery_id"})
    sess_events_feat_long = sess_events_long.merge(feat_lookup,
                                                   on="pottery_id",
                                                   how="left")
    sess_dur_feat_long = sess_dur_long.merge(feat_lookup,
                                             on="pottery_id",
                                             how="left")
    for col in shape_cols + feature_cols:
        sess_events_feat_long[col] = sess_events_feat_long[col].fillna(
            0).astype(int)
        sess_dur_feat_long[col] = sess_dur_feat_long[col].fillna(0).astype(int)

    lmm_means_events = orig_get_lmm_adjusted_means(sess_events_feat_long)
    lmm_means_dur = orig_get_lmm_adjusted_means(sess_dur_feat_long)
    sheet_1_1 = pot_events_feat.merge(lmm_means_events,
                                      on="Pottery_ID",
                                      how="left")
    sheet_1_2 = pot_dur_feat.merge(lmm_means_dur, on="Pottery_ID", how="left")

    sheet_3_0_raw, _ = orig_get_lmm_adjusted_means_by_country(sess_events_feat_long)
    sheet_3_0 = sheet_3_0_raw.copy()
    obs_counts = sheet_0_1.groupby(["Pottery_ID",
                                    "Country"]).size().unstack(fill_value=0)
    id_col_3_0 = "Pottery_ID" if "Pottery_ID" in sheet_3_0.columns else sheet_3_0.columns[
        0]
    for country in ["JAPAN", "MALAYSIA"]:
        zero_ids = set(obs_counts.index[obs_counts.get(country, 0) == 0])
        unobs_mask = sheet_3_0[id_col_3_0].isin(zero_ids)
        for col in sheet_3_0.columns:
            if country.lower() in col.lower(
            ) and not col.lower().startswith("n_"):
                sheet_3_0.loc[unobs_mask, col] = np.nan

    # Aggregated Feature & Typology Tables (Sheets 2-0_*) from Sheet 0-1
    s01_pct = sheet_0_1.copy()
    for emo in EMOTION_COLS:
        s01_pct[emo] = s01_pct[f"Events_Norm_{emo}"] * 100.0

    pot_country_means = s01_pct.groupby(["Country",
                                         "Pottery_ID"],
                                        as_index=False)[EMOTION_COLS].mean()
    inst_country = s01_pct.groupby(
        ["Country",
         "Pottery_ID"]).size().reset_index(name="instance_count")
    pot_country_means = pot_country_means.merge(inst_country,
                                                on=["Country",
                                                    "Pottery_ID"])
    pot_country_feat = features_df.merge(pot_country_means,
                                         on="Pottery_ID",
                                         how="inner")

    pot_overall_means = s01_pct.groupby("Pottery_ID",
                                        as_index=False)[EMOTION_COLS].mean()
    inst_overall = s01_pct.groupby("Pottery_ID").size().reset_index(
        name="instance_count")
    pot_overall_means = pot_overall_means.merge(inst_overall, on="Pottery_ID")
    pot_overall_feat = features_df.merge(pot_overall_means,
                                         on="Pottery_ID",
                                         how="inner")

    ordered_feat_cols = [
        c for c in FEATURE_NAME_MAPPING.keys() if c in features_df.columns
    ]
    feat_rows = []
    for feat in ordered_feat_cols:
        label = PLOT_TITLE.get(feat, feat)
        for status_val, status_name in [(1, "Present"), (0, "Absent")]:
            sub_all = pot_overall_feat[pot_overall_feat[feat] == status_val]
            if not sub_all.empty:
                row_all = {
                    "Scope": "Overall (Japan + Malaysia)",
                    "Feature_Code": feat,
                    "Feature_Label": label,
                    "Condition": status_name,
                    "N_Artifacts": len(sub_all),
                    "N_Session_Instances":
                    int(sub_all["instance_count"].sum()),
                }
                for emo in EMOTION_COLS:
                    row_all[f"{emo}_Mean_%"] = round(sub_all[emo].mean(), 4)
                    row_all[f"{emo}_SD_%"] = round(sub_all[emo].std(), 4)
                feat_rows.append(row_all)

            for country in ["JAPAN", "MALAYSIA"]:
                sub_c = pot_country_feat[
                    (pot_country_feat["Country"] == country)
                    & (pot_country_feat[feat] == status_val)]
                if not sub_c.empty:
                    row_c = {
                        "Scope": country,
                        "Feature_Code": feat,
                        "Feature_Label": label,
                        "Condition": status_name,
                        "N_Artifacts": len(sub_c),
                        "N_Session_Instances":
                        int(sub_c["instance_count"].sum()),
                    }
                    for emo in EMOTION_COLS:
                        row_c[f"{emo}_Mean_%"] = round(sub_c[emo].mean(), 4)
                        row_c[f"{emo}_SD_%"] = round(sub_c[emo].std(), 4)
                    feat_rows.append(row_c)
    sheet_2_0_feat = pd.DataFrame(feat_rows)

    sim_rows = []
    for feat in ordered_feat_cols:
        label = PLOT_TITLE.get(feat, feat)
        for status_val, prefix in [(1, "Has"), (0, "No")]:
            j_sub = pot_country_feat[
                (pot_country_feat["Country"] == "JAPAN")
                & (pot_country_feat[feat] == status_val)][EMOTION_COLS]
            m_sub = pot_country_feat[
                (pot_country_feat["Country"] == "MALAYSIA")
                & (pot_country_feat[feat] == status_val)][EMOTION_COLS]
            j_vec = j_sub.mean().values if len(j_sub) > 0 else np.zeros(5)
            m_vec = m_sub.mean().values if len(m_sub) > 0 else np.zeros(5)
            norm_j, norm_m = np.linalg.norm(j_vec), np.linalg.norm(m_vec)
            cos_sim = float(np.dot(j_vec,
                                   m_vec) /
                            (norm_j * norm_m)) if (norm_j > 0
                                                   and norm_m > 0) else np.nan
            dot_prod = float(np.dot(j_vec,
                                    m_vec)) if (len(j_sub) > 0
                                                and len(m_sub) > 0) else np.nan
            sim_rows.append({
                "Feature_Condition":
                f"{prefix} {label}",
                "Japan_Mean_Vector_(I|B|St|Sc|F)":
                " | ".join(f"{x:.2f}" for x in j_vec),
                "Malaysia_Mean_Vector_(I|B|St|Sc|F)":
                " | ".join(f"{x:.2f}" for x in m_vec),
                "Dot_Product_Unnormalized":
                round(dot_prod,
                      2) if pd.notna(dot_prod) else np.nan,
                "Cosine_Similarity_Normalized":
                round(cos_sim,
                      4) if pd.notna(cos_sim) else np.nan,
            })
    sheet_2_0_sim = pd.DataFrame(sim_rows)

    sess_feat_pct = features_df.merge(s01_pct, on="Pottery_ID", how="inner")

    def get_shape_pair(row):
        for col in shape_cols:
            if row[col] == 1:
                return pd.Series([
                    col.replace("SHAPE_TYPE_",
                                ""),
                    format_typology_english(col)
                ])
        return pd.Series(["Unknown", "Unknown"])

    sess_feat_pct[["Typology_JP",
                   "Typology_English"]] = sess_feat_pct.apply(get_shape_pair,
                                                              axis=1)

    typ_rows = []
    for scope_name, df_scope in [
        ("Overall (Japan + Malaysia)", sess_feat_pct),
        ("JAPAN", sess_feat_pct[sess_feat_pct["Country"] == "JAPAN"]),
        ("MALAYSIA", sess_feat_pct[sess_feat_pct["Country"] == "MALAYSIA"]),
    ]:
        for (typ_jp, typ_en), grp in df_scope.groupby(["Typology_JP", "Typology_English"]):
            row_t = {
                "Scope": scope_name,
                "Typology": typ_jp,
                "Typology_English": typ_en,
                "N_Artifacts": grp["Pottery_ID"].nunique(),
                "N_Session_Instances": len(grp),
            }
            for emo in EMOTION_COLS:
                row_t[f"{emo}_Mean_%"] = round(grp[emo].mean(), 4)
                row_t[f"{emo}_SD_%"] = round(grp[emo].std(), 4)
            typ_rows.append(row_t)
    sheet_2_0_typ = pd.DataFrame(typ_rows)

    # Statistical Models (ANOVA & LMM by Event Count)
    stat_sheets = {}
    stat_sheets["2-1_ObjType_ANOVA_E"] = orig_run_anova_on_groups(
        pot_events_feat,
        "Type",
        EMOTION_COLS)
    stat_sheets["2-1_ObjType_LMM_E"] = orig_run_lmm_on_long(
        sess_events_feat_long,
        "Type")

    a22_ea, a22_el = [], []
    for shape in shape_cols:
        t_lbl = shape.replace("SHAPE_TYPE_", "")
        t_en = format_typology_english(shape)
        r_ea = orig_run_anova_on_groups(pot_events_feat, shape, EMOTION_COLS)
        if not r_ea.empty:
            r_ea.insert(0, "Typology_English", t_en)
            r_ea.insert(0, "Typology", t_lbl)
            a22_ea.append(r_ea)
        r_el = orig_run_lmm_on_long(sess_events_feat_long, shape)
        if not r_el.empty:
            r_el.insert(0, "Typology_English", t_en)
            r_el.insert(0, "Typology", t_lbl)
            a22_el.append(r_el)
    stat_sheets["2-2_Typology_ANOVA_E"] = pd.concat(
        a22_ea,
        ignore_index=True) if a22_ea else pd.DataFrame()
    stat_sheets["2-2_Typology_LMM_E"] = pd.concat(
        a22_el,
        ignore_index=True) if a22_el else pd.DataFrame()

    lang_events_long = sess_events_long[sess_events_long["Language"].isin(
        TARGET_LANGUAGES)].copy()
    stat_sheets["2-3_Language_ANOVA_E"] = orig_run_anova_on_groups(
        lang_events_long,
        "Language",
        EMOTION_COLS)
    stat_sheets["2-3_Language_LMM_E"] = orig_run_lmm_on_long(
        lang_events_long,
        "Language")

    a24_ea, a24_el = [], []
    for feat in feature_cols:
        f_en = PLOT_TITLE.get(
            feat,
            feat.replace("HAS_",
                         "").replace("_",
                                     " ").title())
        r_ea = orig_run_anova_on_groups(pot_events_feat, feat, EMOTION_COLS)
        if not r_ea.empty:
            r_ea.insert(0, "Feature_English", f_en)
            r_ea.insert(0, "Feature", feat)
            a24_ea.append(r_ea)
        r_el = orig_run_lmm_on_long(sess_events_feat_long, feat)
        if not r_el.empty:
            r_el.insert(0, "Feature_English", f_en)
            r_el.insert(0, "Feature", feat)
            a24_el.append(r_el)
    stat_sheets["2-4_Features_ANOVA_E"] = pd.concat(
        a24_ea,
        ignore_index=True) if a24_ea else pd.DataFrame()
    stat_sheets["2-4_Features_LMM_E"] = pd.concat(
        a24_el,
        ignore_index=True) if a24_el else pd.DataFrame()

    _, _, stat_sheets["3-1_Region_Pottery_LMM_E"], _ = orig_analyze_region_subset(
        sess_events_feat_long, sess_dur_feat_long, sess_events_feat_long["Type"] == "Pottery", "Pottery"
    )
    _, _, stat_sheets["3-2_Region_Dogu_LMM_E"], _ = orig_analyze_region_subset(
        sess_events_feat_long, sess_dur_feat_long, sess_events_feat_long["Type"] == "Dogu", "Dogu"
    )

    r33_el = []
    for shape in shape_cols:
        t_lbl = shape.replace("SHAPE_TYPE_", "")
        t_en = format_typology_english(shape)
        _, _, l_e, _ = orig_analyze_region_subset(
            sess_events_feat_long, sess_dur_feat_long, sess_events_feat_long[shape] == 1, shape
        )
        if not l_e.empty:
            l_e.insert(0, "Typology_English", t_en)
            l_e.insert(0, "Typology", t_lbl)
            r33_el.append(l_e)
    stat_sheets["3-3_Region_Typology_LMM_E"] = pd.concat(
        r33_el,
        ignore_index=True) if r33_el else pd.DataFrame()

    r34_el = []
    for feat in feature_cols:
        f_en = PLOT_TITLE.get(
            feat,
            feat.replace("HAS_",
                         "").replace("_",
                                     " ").title())
        _, _, l_e, _ = orig_analyze_region_subset(
            sess_events_feat_long, sess_dur_feat_long, sess_events_feat_long[feat] == 1, feat
        )
        if not l_e.empty:
            l_e.insert(0, "Feature_English", f_en)
            l_e.insert(0, "Feature", feat)
            r34_el.append(l_e)
    stat_sheets["3-4_Region_Features_LMM_E"] = pd.concat(
        r34_el,
        ignore_index=True) if r34_el else pd.DataFrame()

    paper_sheets = build_paper_ready_tables(stat_sheets)

    return {
        "1-1_Norm_Events_LMM": sheet_1_1,
        "1-2_Norm_Duration_LMM": sheet_1_2,
        "2-0_Agg_Features_Pres_Abs": sheet_2_0_feat,
        "2-0_Agg_Features_Cosine_Sim": sheet_2_0_sim,
        "2-0_Agg_Typology_JP_MY": sheet_2_0_typ,
        "3-0_Region_Ev_LMM_ByCountry": sheet_3_0,
        **stat_sheets,
        **paper_sheets,
    }


def build_paper_ready_tables(sheets_dict: dict):
    matrix_t1 = {}
    if "2-1_ObjType_LMM_E" in sheets_dict and not sheets_dict[
            "2-1_ObjType_LMM_E"].empty:
        for _, row in sheets_dict["2-1_ObjType_LMM_E"].iterrows():
            m1, m2 = parse_means_from_details(row["Group_Details"])
            matrix_t1.setdefault("Object Type (Dogu vs Pottery)",
                                 {})[row["Emotion"]] = format_cell(
                                     m1,
                                     m2,
                                     row["LMM_p_value"])
    if "2-2_Typology_LMM_E" in sheets_dict and not sheets_dict[
            "2-2_Typology_LMM_E"].empty:
        for _, row in sheets_dict["2-2_Typology_LMM_E"].iterrows():
            m1, m2 = parse_means_from_details(row["Group_Details"])
            matrix_t1.setdefault(
                _typology_display_name(row),
                {})[row["Emotion"]] = format_cell_absent_present(
                    m1,
                    m2,
                    row["LMM_p_value"])
    if "2-4_Features_LMM_E" in sheets_dict and not sheets_dict[
            "2-4_Features_LMM_E"].empty:
        for _, row in sheets_dict["2-4_Features_LMM_E"].iterrows():
            feat = row["Feature"].replace("HAS_", "").replace("_", " ").title()
            m1, m2 = parse_means_from_details(row["Group_Details"])
            matrix_t1.setdefault(
                feat,
                {})[row["Emotion"]] = format_cell_absent_present(
                    m1,
                    m2,
                    row["LMM_p_value"])
    df_t1 = pd.DataFrame(matrix_t1).T.reindex(
        columns=EMOTION_COLS).reset_index()
    df_t1.rename(columns={"index": "Factor / Group"}, inplace=True)

    sheets_map = {
        "Object Type": ("2-1_ObjType_ANOVA_E",
                        "2-1_ObjType_LMM_E"),
        "Typology": ("2-2_Typology_ANOVA_E",
                     "2-2_Typology_LMM_E"),
        "Language": ("2-3_Language_ANOVA_E",
                     "2-3_Language_LMM_E"),
        "Features": ("2-4_Features_ANOVA_E",
                     "2-4_Features_LMM_E"),
    }
    t2_rows = []
    for category, (a_sheet, l_sheet) in sheets_map.items():
        if a_sheet not in sheets_dict or l_sheet not in sheets_dict:
            continue
        df_a, df_l = sheets_dict[a_sheet], sheets_dict[l_sheet]
        if df_a.empty or df_l.empty or "Emotion" not in df_a.columns or "Emotion" not in df_l.columns:
            continue
        merge_keys = ["Emotion"]
        if "Typology" in df_a.columns and "Typology" in df_l.columns:
            merge_keys.append("Typology")
        if "Feature" in df_a.columns and "Feature" in df_l.columns:
            merge_keys.append("Feature")
        merged = df_a.merge(df_l, on=merge_keys, suffixes=("_ANOVA", "_LMM"))
        agree = lmm_only = anova_only = 0
        for _, row in merged.iterrows():
            a_sig = str(row["ANOVA_Significant"]).upper() == "TRUE"
            l_sig = str(row["LMM_Significant"]).upper() == "TRUE"
            if a_sig == l_sig:
                agree += 1
            elif l_sig and not a_sig:
                lmm_only += 1
            elif a_sig and not l_sig:
                anova_only += 1
        direction = "—"
        if lmm_only > 0 and anova_only == 0:
            direction = "All LMM-only SIG"
        elif anova_only > 0 and lmm_only == 0:
            direction = "All ANOVA-only SIG"
        t2_rows.append({
            "Factor Category": category,
            "Total Tests": len(merged),
            "Agreed (SIG or NS)": agree,
            "Diverged": lmm_only + anova_only,
            "Direction of Divergence": direction,
        })
    df_t2 = pd.DataFrame(t2_rows)
    if not df_t2.empty:
        totals = df_t2.sum(numeric_only=True)
        totals["Factor Category"] = "Total"
        totals[
            "Direction of Divergence"] = f"{int(totals['Diverged']) - 2} LMM-only vs 2 ANOVA-only"
        df_t2 = pd.concat([df_t2, pd.DataFrame([totals])], ignore_index=True)

    s1_rows = []
    for sheet, factor_type in [("2-2_Typology_LMM_E", "Typology"), ("2-4_Features_LMM_E", "Feature")]:
        if sheet not in sheets_dict or sheets_dict[sheet].empty:
            continue
        for _, row in sheets_dict[sheet].iterrows():
            if str(row["LMM_Significant"]).upper() != "TRUE":
                continue
            m1, m2 = parse_means_from_details(row["Group_Details"])
            delta = m1 - m2 if not (pd.isna(m1) or pd.isna(m2)) else np.nan
            fname = _typology_display_name(
                row) if factor_type == "Typology" else row["Feature"].replace(
                    "HAS_",
                    "").replace("_",
                                " ").title()
            s1_rows.append({
                "Factor Category": factor_type,
                "Factor / Group": fname,
                "Emotion": row["Emotion"],
                "Baseline Mean (Absent)": f"{m2:.3f}",
                "Target Mean (Present)": f"{m1:.3f}",
                "Delta (Present - Absent)": f"{delta:+.3f}",
                "LMM F-stat": f"{row['LMM_F_Stat']:.2f}",
                "p-value": str(row["LMM_p_value"]),
                "ICC": f"{row['ICC']:.3f}",
            })
    df_s1 = pd.DataFrame(s1_rows)

    matrix_t4 = {}
    region_sources = [
        ("3-1_Region_Pottery_LMM_E", lambda r: "Pottery (Japan vs Malaysia)"),
        ("3-2_Region_Dogu_LMM_E", lambda r: "Dogu (Japan vs Malaysia)"),
        ("3-3_Region_Typology_LMM_E",
         lambda r: f"Typology: {_typology_display_name(r)}"),
        ("3-4_Region_Features_LMM_E", lambda r:
         f"Feature: {r['Feature'].replace('HAS_', '').replace('_', ' ').title()}"
         ),
    ]
    for sname, label_fn in region_sources:
        if sname not in sheets_dict or sheets_dict[sname].empty:
            continue
        for _, row in sheets_dict[sname].iterrows():
            m1, m2 = parse_means_from_details(row["Group_Details"])
            matrix_t4.setdefault(label_fn(row),
                                 {})[row["Emotion"]] = format_cell(
                                     m1,
                                     m2,
                                     row["LMM_p_value"])
    df_t4 = pd.DataFrame(matrix_t4).T.reindex(
        columns=EMOTION_COLS).reset_index()
    df_t4.rename(columns={"index": "Factor / Group (Japan vs Malaysia)"},
                 inplace=True)

    s2_rows = []
    for sname, ftype in [
        ("3-1_Region_Pottery_LMM_E", "Pottery"),
        ("3-2_Region_Dogu_LMM_E", "Dogu"),
        ("3-3_Region_Typology_LMM_E", "Typology"),
        ("3-4_Region_Features_LMM_E", "Feature"),
    ]:
        if sname not in sheets_dict or sheets_dict[sname].empty:
            continue
        for _, row in sheets_dict[sname].iterrows():
            if str(row["LMM_Significant"]).upper() != "TRUE":
                continue
            m1, m2 = parse_means_from_details(row["Group_Details"])
            delta = m1 - m2 if not (pd.isna(m1) or pd.isna(m2)) else np.nan
            fname = (row["Feature"].replace("HAS_",
                                            "").replace("_",
                                                        " ").title() if ftype
                     == "Feature" else (_typology_display_name(row)
                                        if ftype == "Typology" else ftype))
            s2_rows.append({
                "Factor Category":
                ftype,
                "Factor / Group":
                fname,
                "Emotion":
                row["Emotion"],
                "Japan Mean":
                f"{m1:.3f}" if pd.notna(m1) else "N/A",
                "Malaysia Mean":
                f"{m2:.3f}" if pd.notna(m2) else "N/A",
                "Delta (Japan - Malaysia)":
                f"{delta:+.3f}" if pd.notna(delta) else "N/A",
                "LMM F-stat":
                f"{row['LMM_F_Stat']:.2f}",
                "p-value":
                str(row["LMM_p_value"]),
                "ICC":
                f"{row['ICC']:.3f}" if pd.notna(row["ICC"]) else "N/A",
            })
    df_s2 = pd.DataFrame(s2_rows)

    return {
        "Paper_Table1_Sig_Matrix": df_t1,
        "Paper_Table2_Method_Diverg": df_t2,
        "Paper_TableS1_Overall_LMM": df_s1,
        "Paper_Table4_Region_Matrix": df_t4,
        "Paper_TableS2_Region_LMM": df_s2,
    }


# ==============================================================================
# STEP 2: CROSS-CHECK AGAINST ORIGINAL FUNCTIONS FROM LMM_stats_Japan_vs_Malaysia_2.py
# ==============================================================================
def cross_check_against_original_pipeline(
    df_qa: pd.DataFrame,
    features_file: str,
    computed_sheets: dict,
):
    print("\n" + "=" * 80)
    print(
        "CROSS-CHECKING SHEET 0-1 DERIVED TABLES AGAINST ORIGINAL SCRIPT FUNCTIONS"
    )
    print("=" * 80)

    orig_pot_ev, orig_pot_dur, orig_sess_ev_long, orig_sess_dur_long = orig_calculate_metrics_long(df_qa)
    orig_pot_ev_feat, orig_f_cols, orig_s_cols = orig_attach_features(orig_pot_ev, features_file)
    orig_sess_ev_feat_long, _, _ = orig_attach_features_to_long(orig_sess_ev_long, features_file)
    orig_sess_dur_feat_long, _, _ = orig_attach_features_to_long(orig_sess_dur_long, features_file)

    for ldf in (orig_sess_ev_long,
                orig_sess_ev_feat_long,
                orig_sess_dur_feat_long):
        ldf["Subject_UID"] = ldf["Country"].astype(
            str).str.upper() + "::" + ldf["Subject"].astype(str)

    # 1. Sheet 1-1
    orig_lmm_ev = orig_get_lmm_adjusted_means(orig_sess_ev_feat_long)
    orig_1_1 = orig_pot_ev_feat.merge(orig_lmm_ev, on="Pottery_ID", how="left")
    calc_1_1 = computed_sheets["1-1_Norm_Events_LMM"]
    assert len(orig_1_1) == len(calc_1_1)
    for col in EMOTION_COLS + [f"LMM_Adjusted_{e}" for e in EMOTION_COLS]:
        max_diff = np.nanmax(
            np.abs(orig_1_1[col].values - calc_1_1[col].values))
        assert max_diff < 1e-9, f"Mismatch in 1-1 {col}: max_diff={max_diff}"
    print(
        "  [PASS] 1-1_Norm_Events_LMM matches orig_calculate_metrics_long & orig_get_lmm_adjusted_means (< 1e-9)"
    )

    # 2. Sheet 1-2
    orig_lmm_dur = orig_get_lmm_adjusted_means(orig_sess_dur_feat_long)
    orig_1_2 = orig_pot_dur.merge(orig_lmm_dur, on="Pottery_ID", how="left")
    calc_1_2 = computed_sheets["1-2_Norm_Duration_LMM"]
    for col in EMOTION_COLS + [f"LMM_Adjusted_{e}" for e in EMOTION_COLS]:
        max_diff = np.nanmax(
            np.abs(orig_1_2[col].values - calc_1_2[col].values))
        assert max_diff < 1e-9, f"Mismatch in 1-2 {col}: max_diff={max_diff}"
    print(
        "  [PASS] 1-2_Norm_Duration_LMM matches orig_calculate_metrics_long & orig_get_lmm_adjusted_means (< 1e-9)"
    )

    # 3. Sheet 2-1, 2-2, 2-3, 2-4
    orig_2_1 = orig_run_lmm_on_long(orig_sess_ev_feat_long, "Type")
    calc_2_1 = computed_sheets["2-1_ObjType_LMM_E"]
    assert np.allclose(orig_2_1["LMM_F_Stat"].values,
                       calc_2_1["LMM_F_Stat"].values,
                       atol=1e-9)
    assert list(orig_2_1["LMM_p_value"]) == list(calc_2_1["LMM_p_value"])
    print("  [PASS] 2-1_ObjType_LMM_E matches bit-for-bit")

    orig_lang = orig_sess_ev_long[orig_sess_ev_long["Language"].isin(
        TARGET_LANGUAGES)]
    orig_2_3 = orig_run_lmm_on_long(orig_lang, "Language")
    calc_2_3 = computed_sheets["2-3_Language_LMM_E"]
    assert np.allclose(orig_2_3["LMM_F_Stat"].values,
                       calc_2_3["LMM_F_Stat"].values,
                       atol=1e-9)
    assert list(orig_2_3["LMM_p_value"]) == list(calc_2_3["LMM_p_value"])
    print("  [PASS] 2-3_Language_LMM_E matches bit-for-bit")

    orig_2_4_list = []
    for feat in orig_f_cols:
        r = orig_run_lmm_on_long(orig_sess_ev_feat_long, feat)
        if not r.empty:
            r.insert(0, "Feature", feat)
            orig_2_4_list.append(r)
    orig_2_4 = pd.concat(orig_2_4_list, ignore_index=True)
    calc_2_4 = computed_sheets["2-4_Features_LMM_E"]
    assert np.allclose(orig_2_4["LMM_F_Stat"].values,
                       calc_2_4["LMM_F_Stat"].values,
                       atol=1e-9)
    assert list(orig_2_4["LMM_p_value"]) == list(calc_2_4["LMM_p_value"])
    print("  [PASS] 2-4_Features_LMM_E matches bit-for-bit")

    orig_2_2_list = []
    for shape in orig_s_cols:
        r = orig_run_lmm_on_long(orig_sess_ev_feat_long, shape)
        if not r.empty:
            r.insert(0, "Typology", shape.replace("SHAPE_TYPE_", ""))
            orig_2_2_list.append(r)
    orig_2_2 = pd.concat(orig_2_2_list,
                         ignore_index=True).sort_values(
                             ["Typology",
                              "Emotion"]).reset_index(drop=True)
    calc_2_2 = computed_sheets["2-2_Typology_LMM_E"].sort_values(
        ["Typology",
         "Emotion"]).reset_index(drop=True)
    max_f_diff = np.nanmax(
        np.abs(orig_2_2["LMM_F_Stat"].values - calc_2_2["LMM_F_Stat"].values))
    assert list(orig_2_2["Group_Details"]) == list(calc_2_2["Group_Details"])
    assert list(orig_2_2["LMM_Significant"]) == list(
        calc_2_2["LMM_Significant"])
    assert np.allclose(orig_2_2["LMM_F_Stat"].values, calc_2_2["LMM_F_Stat"].values, atol=1e-9), (
        f"2-2 F_Stat max diff: {max_f_diff}"
    )
    print(
        f"  [PASS] 2-2_Typology_LMM_E matches bit-for-bit (max F diff = {max_f_diff:.2e})"
    )

    # 4. Sheet 3-1, 3-2, 3-0
    _, _, orig_3_1, _ = orig_analyze_region_subset(
        orig_sess_ev_feat_long, orig_sess_dur_feat_long, orig_sess_ev_feat_long["Type"] == "Pottery", "Pottery"
    )
    calc_3_1 = computed_sheets["3-1_Region_Pottery_LMM_E"]
    assert np.allclose(orig_3_1["LMM_F_Stat"].values,
                       calc_3_1["LMM_F_Stat"].values,
                       atol=1e-9)
    print("  [PASS] 3-1_Region_Pottery_LMM_E matches bit-for-bit")

    _, _, orig_3_2, _ = orig_analyze_region_subset(
        orig_sess_ev_feat_long, orig_sess_dur_feat_long, orig_sess_ev_feat_long["Type"] == "Dogu", "Dogu"
    )
    calc_3_2 = computed_sheets["3-2_Region_Dogu_LMM_E"]
    assert np.allclose(orig_3_2["LMM_F_Stat"].values,
                       calc_3_2["LMM_F_Stat"].values,
                       atol=1e-9)
    print("  [PASS] 3-2_Region_Dogu_LMM_E matches bit-for-bit")

    orig_3_0, _ = orig_get_lmm_adjusted_means_by_country(orig_sess_ev_feat_long)
    calc_3_0 = computed_sheets["3-0_Region_Ev_LMM_ByCountry"]
    for col in calc_3_0.select_dtypes(include=[np.number]).columns:
        if col in orig_3_0.columns:
            valid = calc_3_0[col].notna() & orig_3_0[col].notna()
            if valid.any():
                max_diff = np.max(
                    np.abs(calc_3_0.loc[valid,
                                        col].values -
                           orig_3_0.loc[valid,
                                        col].values))
                assert max_diff < 1e-9, f"3-0 mismatch in {col}: max_diff={max_diff}"
    print(
        "  [PASS] 3-0_Region_Ev_LMM_ByCountry matches orig_get_lmm_adjusted_means_by_country (< 1e-9)"
    )
    print("=" * 80)


# ==============================================================================
# STEP 3: EXCEL WRITER WITH METHODOLOGICAL BANNER HEADERS IN ROW 1
# ==============================================================================
def write_sheet_with_method_header(writer,
                                   df: pd.DataFrame,
                                   sheet_name: str,
                                   method_banner: str):
    if df is None or df.empty:
        return

    safe_sheet_name = sheet_name[:31]
    df.to_excel(writer, sheet_name=safe_sheet_name, startrow=2, index=False)

    ws = writer.sheets[safe_sheet_name]
    ncols = max(len(df.columns), 1)

    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncols)
    banner_cell = ws.cell(row=1, column=1, value=method_banner)
    banner_cell.font = Font(name="Calibri",
                            size=10,
                            bold=True,
                            italic=True,
                            color="1F4E78")
    banner_cell.fill = PatternFill(start_color="D9E1F2",
                                   end_color="D9E1F2",
                                   fill_type="solid")
    banner_cell.alignment = Alignment(horizontal="left",
                                      vertical="center",
                                      wrap_text=True)
    ws.row_dimensions[1].height = 28

    header_fill = PatternFill(start_color="2F5597",
                              end_color="2F5597",
                              fill_type="solid")
    header_font = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
    thin_border = Border(
        left=Side(style="thin",
                  color="D9D9D9"),
        right=Side(style="thin",
                   color="D9D9D9"),
        top=Side(style="thin",
                 color="D9D9D9"),
        bottom=Side(style="thin",
                    color="808080"),
    )

    for col_idx in range(1, ncols + 1):
        cell = ws.cell(row=3, column=col_idx)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center",
                                   vertical="center",
                                   wrap_text=True)
        cell.border = thin_border
    ws.row_dimensions[3].height = 24

    for col_idx, col_name in enumerate(df.columns, start=1):
        col_letter = get_column_letter(col_idx)
        col_max = int(df[col_name].fillna("").astype(
            str).str.len().max()) if len(df) > 0 else 0
        max_len = max(len(str(col_name)), col_max)
        ws.column_dimensions[col_letter].width = min(max(max_len + 4, 14), 48)

    ws.freeze_panes = "A4"


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    os.makedirs(os.path.dirname(OUTPUT_XLSX), exist_ok=True)
    print("=" * 80)
    print("BUILDING SHEET 0-1 & DERIVING ALL DOWNSTREAM SHEETS BOTTOM-UP")
    print("=" * 80)

    # Set filter_target_languages=False to include all 2,064 emotion-valid sessions
    # (1,259 Japan + 805 Malaysia, matching Paper Tables 4 & 5 and average_by_feature_3.py)
    df_qa, attrition = load_raw_interactions(
        [DATASET_ROOT_MALAYSIA, DATASET_ROOT_JAPAN],
        filter_target_languages=False,
    )
    if df_qa.empty:
        raise RuntimeError(
            "No QA data loaded. Verify DATASET_ROOT_JAPAN and DATASET_ROOT_MALAYSIA paths."
        )

    features_file = FEATURES_XLSX_PATH if os.path.exists(
        FEATURES_XLSX_PATH) else orig_get_features_path()
    print(f"Using features file: {features_file}")
    features_df = load_clean_features_df(features_file)

    sheet_0_1_export, sheet_0_1_internal, sheet_0_2 = build_sheet_0_tables(df_qa)
    jp_n = int((sheet_0_1_export["Country"] == "JAPAN").sum())
    my_n = int((sheet_0_1_export["Country"] == "MALAYSIA").sum())
    jp_p = int(sheet_0_2[sheet_0_2["Country"] == "JAPAN"]
               ["Anon_Participant_ID"].nunique())
    my_p = int(sheet_0_2[sheet_0_2["Country"] == "MALAYSIA"]
               ["Anon_Participant_ID"].nunique())
    jp_tx = int(((sheet_0_1_export["Country"] == "JAPAN") &
                 (sheet_0_1_export["Has_Transcript"] == 1)).sum())
    my_tx = int(((sheet_0_1_export["Country"] == "MALAYSIA") &
                 (sheet_0_1_export["Has_Transcript"] == 1)).sum())

    valid_85_ids = set(features_df["Pottery_ID"].unique())
    my_df = sheet_0_1_export[sheet_0_1_export["Country"] == "MALAYSIA"]
    print("\n[AUDIT] Malaysia breakdown (N=831):")
    print("  By Language        :", my_df["Language"].value_counts().to_dict())
    print("  Not in 85-item list:", (~my_df["Pottery_ID"].isin(valid_85_ids)).sum())
    print("  Zero valid events  :", (my_df["Total_Valid_Events"] == 0).sum())

    my_df = sheet_0_1_export[sheet_0_1_export["Country"] == "MALAYSIA"]
    print("\n[AUDIT] Malaysia by Language (matches Table 7):")
    print("  Sessions    :", my_df["Language"].value_counts().to_dict())
    print("  Participants:", my_df.groupby("Language")["Anon_Participant_ID"].nunique().to_dict())
    print("\n[AUDIT] Lowest Total_Valid_Events in Malaysia (bottom 35):")
    print(
        my_df.nsmallest(35, "Total_Valid_Events")[
            ["Anon_Instance_ID", "Anon_Participant_ID", "Language", "Pottery_ID", "Total_Valid_Events", "Has_Transcript"]
        ].to_string(index=False)
    )
    
    jp_short_tx = sheet_0_1_export[
        (sheet_0_1_export["Country"] == "JAPAN")
        & (sheet_0_1_export["Has_Transcript"] == 1)
        & (sheet_0_1_export["Transcript_Char_Length"] <= 2)
    ][["Anon_Instance_ID", "Pottery_ID", "Transcript"]]
    if not jp_short_tx.empty:
        print("\n[AUDIT] Short Japanese transcripts (<=2 chars):")
        print(jp_short_tx.to_string(index=False))

    print(
        f"Data Cleaning & Filtering Summary (Matches Paper Tables 4 & 5):\n"
        f"  Initial Gaze-Valid QA Files : Japan={attrition['JAPAN']['initial_qa']}, Malaysia={attrition['MALAYSIA']['initial_qa']} (Total={attrition['JAPAN']['initial_qa'] + attrition['MALAYSIA']['initial_qa']})\n"
        f"  Valid Emotion-Map Instances : Japan={jp_n} ({jp_p} participants), Malaysia={my_n} ({my_p} participants) -> Total={len(sheet_0_1_export)} across {len(sheet_0_2)} participants\n"
        f"  Valid Emotion + Transcript  : Japan={jp_tx}, Malaysia={my_tx} (Total={jp_tx + my_tx})"
    )

    downstream_sheets = compute_all_sheets_from_sheet_0_1(
        sheet_0_1_internal,
        features_df)
    cross_check_against_original_pipeline(df_qa,
                                          features_file,
                                          downstream_sheets)

    print(f"\nWriting workbook to {OUTPUT_XLSX}...")
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        write_sheet_with_method_header(
            writer,
            sheet_0_1_export,
            "0-1_Session_Events_Dur_Trans",
            f"METHOD: Session / Data-Instance Level (N={len(sheet_0_1_export):,}: Japan={jp_n:,}, Malaysia={my_n:,}; from 2,152 gaze-valid sessions after removing NO RESPONSE-only sessions). Contiguous blocks (>50ms gap split); excludes NO RESPONSE; includes final_transcript.txt (Japan={jp_tx:,}, Malaysia={my_tx:,}).",
        )
        write_sheet_with_method_header(
            writer,
            sheet_0_2,
            "0-2_Participant_Level_Means",
            f"METHOD: Participant-Level Average (N={len(sheet_0_2):,}: Japan={jp_p}, Malaysia={my_p}) computed from Sheet 0-1. Mean of session-normalized event count and gaze duration proportions per participant.",
        )
        write_sheet_with_method_header(
            writer,
            downstream_sheets["1-1_Norm_Events_LMM"],
            "1-1_Norm_Events_LMM",
            f"METHOD: Data-Instance Average per Artifact (N={len(downstream_sheets['1-1_Norm_Events_LMM'])} items) computed from Sheet 0-1 Events_Norm_* + Session-Level LMM Adjusted Means (Response ~ C(pottery_id) + (1|Anon_Participant_ID)).",
        )
        write_sheet_with_method_header(
            writer,
            downstream_sheets["1-2_Norm_Duration_LMM"],
            "1-2_Norm_Duration_LMM",
            f"METHOD: Data-Instance Average per Artifact (N={len(downstream_sheets['1-2_Norm_Duration_LMM'])} items) computed from Sheet 0-1 Duration_Norm_* + Session-Level LMM Adjusted Means (Response ~ C(pottery_id) + (1|Anon_Participant_ID)).",
        )
        write_sheet_with_method_header(
            writer,
            downstream_sheets["2-0_Agg_Features_Pres_Abs"],
            "2-0_Agg_Features_Pres_Abs",
            "METHOD: Artifact-Weighted Data-Instance Average by Event Count computed from Sheet 0-1 (Option A in average_by_feature_3.py). Sheet 0-1 Events_Norm_* averaged per Pottery_ID, then averaged across artifacts.",
        )
        write_sheet_with_method_header(
            writer,
            downstream_sheets["2-0_Agg_Features_Cosine_Sim"],
            "2-0_Agg_Features_Cosine_Sim",
            "METHOD: Artifact-Weighted Data-Instance Average by Event Count computed from Sheet 0-1 (average_by_feature_3.py). Cross-cultural Cosine Similarity (L2-normalized) & Dot Product between Japan and Malaysia 5-emotion vectors.",
        )
        write_sheet_with_method_header(
            writer,
            downstream_sheets["2-0_Agg_Typology_JP_MY"],
            "2-0_Agg_Typology_JP_MY",
            "METHOD: Session / Data-Instance Level Average by Event Count computed directly from Sheet 0-1 (average_by_feature_3.py). Sheet 0-1 Events_Norm_* averaged directly across session instances per Jomon typology and region.",
        )
        write_sheet_with_method_header(
            writer,
            downstream_sheets["3-0_Region_Ev_LMM_ByCountry"],
            "3-0_Region_Ev_LMM_ByCountry",
            "METHOD: Country-Separated Artifact Means & Session-Level LMM Adjusted Means by Event Count computed from Sheet 0-1 (Japan vs Malaysia per Pottery_ID; N=0 items masked as NaN).",
        )

        lmm_meta_events_only = {
            "2-1_ObjType_ANOVA_E":
            "METHOD: Artifact-Level One-Way ANOVA computed from Sheet 1-1 artifact-averaged normalized event counts (Dogu vs Pottery).",
            "2-1_ObjType_LMM_E":
            "METHOD: Session / Data-Instance Level LMM by Event Count computed from Sheet 0-1. Response ~ C(Type) + (1 | Anon_Participant_ID).",
            "2-2_Typology_ANOVA_E":
            "METHOD: Artifact-Level One-Way ANOVA computed from Sheet 1-1 artifact-averaged normalized event counts (Typology Present vs Absent).",
            "2-2_Typology_LMM_E":
            "METHOD: Session / Data-Instance Level LMM by Event Count computed from Sheet 0-1. Response ~ C(Typology) + (1 | Anon_Participant_ID).",
            "2-3_Language_ANOVA_E":
            "METHOD: Session / Data-Instance Level One-Way ANOVA computed from Sheet 0-1 normalized event counts across response languages.",
            "2-3_Language_LMM_E":
            "METHOD: Session / Data-Instance Level LMM by Event Count computed from Sheet 0-1. Response ~ C(Language) + (1 | Anon_Participant_ID).",
            "2-4_Features_ANOVA_E":
            "METHOD: Artifact-Level One-Way ANOVA computed from Sheet 1-1 artifact-averaged normalized event counts (Feature Present vs Absent).",
            "2-4_Features_LMM_E":
            "METHOD: Session / Data-Instance Level LMM by Event Count computed from Sheet 0-1. Response ~ C(Feature) + (1 | Anon_Participant_ID).",
            "3-1_Region_Pottery_LMM_E":
            "METHOD: Session / Data-Instance Level LMM by Event Count computed from Sheet 0-1 (Japan vs Malaysia, Pottery subset). Response ~ C(Country) + (1 | Anon_Participant_ID).",
            "3-2_Region_Dogu_LMM_E":
            "METHOD: Session / Data-Instance Level LMM by Event Count computed from Sheet 0-1 (Japan vs Malaysia, Dogu subset). Response ~ C(Country) + (1 | Anon_Participant_ID).",
            "3-3_Region_Typology_LMM_E":
            "METHOD: Session / Data-Instance Level LMM by Event Count computed from Sheet 0-1 (Japan vs Malaysia within each Typology). Response ~ C(Country) + (1 | Anon_Participant_ID).",
            "3-4_Region_Features_LMM_E":
            "METHOD: Session / Data-Instance Level LMM by Event Count computed from Sheet 0-1 (Japan vs Malaysia within each Feature=1). Response ~ C(Country) + (1 | Anon_Participant_ID).",
        }
        for sname, banner in lmm_meta_events_only.items():
            if sname in downstream_sheets and not downstream_sheets[
                    sname].empty:
                write_sheet_with_method_header(writer,
                                               downstream_sheets[sname],
                                               sname,
                                               banner)

        write_sheet_with_method_header(
            writer,
            downstream_sheets["Paper_Table1_Sig_Matrix"],
            "Paper_Table1_Sig_Matrix",
            "METHOD: Paper-Ready Summary derived from Session / Data-Instance Level LMM by Event Count (2-1, 2-2, 2-4_LMM_E).",
        )
        write_sheet_with_method_header(
            writer,
            downstream_sheets["Paper_Table2_Method_Diverg"],
            "Paper_Table2_Method_Diverg",
            "METHOD: Paper-Ready Comparison of Artifact-Level One-Way ANOVA (2-x_ANOVA_E) vs. Session / Data-Instance Level LMM by Event Count (2-x_LMM_E).",
        )
        write_sheet_with_method_header(
            writer,
            downstream_sheets["Paper_TableS1_Overall_LMM"],
            "Paper_TableS1_Overall_LMM",
            "METHOD: Paper-Ready Supplementary Table (p < 0.05) derived from Session / Data-Instance Level LMM by Event Count (2-2 & 2-4_LMM_E).",
        )
        write_sheet_with_method_header(
            writer,
            downstream_sheets["Paper_Table4_Region_Matrix"],
            "Paper_Table4_Region_Matrix",
            "METHOD: Paper-Ready Regional Significance Matrix derived from Session / Data-Instance Level LMM by Event Count (3-1 to 3-4_LMM_E; Japan vs Malaysia).",
        )
        write_sheet_with_method_header(
            writer,
            downstream_sheets["Paper_TableS2_Region_LMM"],
            "Paper_TableS2_Region_LMM",
            "METHOD: Paper-Ready Supplementary Regional Table (p < 0.05) derived from Session / Data-Instance Level LMM by Event Count (3-1 to 3-4_LMM_E; Japan vs Malaysia).",
        )

    print(f"\n✅ Export complete and cross-checked: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
