import json
import os
from typing import Dict, List, Optional
from dataset.utils import ASSIGNED_NUMBERS_DICT, DEFAULT_QNA_ANSWER_COLOR_MAP

KNOWN_DOGU_PREFIXES_OR_IDS = {
    "ND0001",  # Shakoki-dogu / figurine series
    "MH0037",  # Dogu specimen
    "NZ0001",  # Dogu specimen
    "UK0001",  # Dogu specimen
    "IN0295",  # Dogu specimen
    "IN0306",  # Dogu specimen
    "SK0035",  # Dogu specimen
    "TK0020",  # Dogu specimen
}


def build_artifact_taxonomy(custom_json_path: Optional[str] = None) -> Dict[str, str]:
    """
    Returns a dictionary mapping formatted IDs like 'NM0239(89)' -> 'pottery' or 'dogu'.
    """
    taxonomy = {}
    for raw_id, num in ASSIGNED_NUMBERS_DICT.items():
        formatted_id = f"{raw_id}({num})"
        if raw_id in KNOWN_DOGU_PREFIXES_OR_IDS or formatted_id in KNOWN_DOGU_PREFIXES_OR_IDS:
            taxonomy[formatted_id] = "dogu"
        else:
            taxonomy[formatted_id] = "pottery"

    if custom_json_path and os.path.exists(custom_json_path):
        with open(custom_json_path, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        for k, v in overrides.items():
            v_clean = v.strip().lower()
            if v_clean in ("pottery", "dogu"):
                taxonomy[k] = v_clean
                if k in ASSIGNED_NUMBERS_DICT:
                    taxonomy[f"{k}({ASSIGNED_NUMBERS_DICT[k]})"] = v_clean

    return taxonomy


def filter_paths_by_artifact_type(
    data_paths: List[dict],
    artifact_type: str = "all",
    custom_json_path: Optional[str] = None,
) -> List[dict]:
    """Filters dataset entries by 'pottery', 'dogu', or 'all'."""
    taxonomy = build_artifact_taxonomy(custom_json_path)
    filtered = []
    for item in data_paths:
        pid = item.get("ID", "")
        atype = taxonomy.get(pid, "pottery")
        item["ARTIFACT_TYPE"] = atype
        if artifact_type in ("all", "both") or atype == artifact_type:
            filtered.append(item)
    return filtered


# Canonical 5-channel emotion mapping aligned across Japan & Malaysia datasets
CANONICAL_EMOTIONS = [
    {
        "color": "cyan",
        "jp": "面白い・気になる形だ",
        "en": "Interesting and attentional shape",
    },
    {
        "color": "green",
        "jp": "美しい・芸術的だ",
        "en": "Beautiful and artistic",
    },
    {
        "color": "yellow",
        "jp": "不思議・意味不明",
        "en": "Strange and incomprehensible",
    },
    {
        "color": "red",
        "jp": "不気味・不安・怖い",
        "en": "Creepy / unsettling / scary",
    },
    {
        "color": "grey",
        "jp": "何も感じない",
        "en": "Feel nothing",
    },
]