import json
import pandas as pd
from pathlib import Path

def load_json(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_donki_cme(path: Path) -> pd.DataFrame:
    data = load_json(path)
    df = pd.json_normalize(data)

    # Keep a manageable subset (expand later as needed)
    keep = [
        "activityID",
        "startTime",
        "sourceLocation",
        "activeRegionNum",
        "cmeAnalyses",
        "note",
        "link",
    ]
    cols = [c for c in keep if c in df.columns]
    df = df[cols].copy()

    # Flatten first analysis entry if present
    if "cmeAnalyses" in df.columns:
        def first_analysis(x):
            if isinstance(x, list) and len(x) > 0 and isinstance(x[0], dict):
                return x[0]
            return {}
        ana = df["cmeAnalyses"].apply(first_analysis)
        ana_df = pd.json_normalize(ana)
        # Common fields found in DONKI CME analyses
        for c in ["speed", "type", "latitude", "longitude", "halfAngle", "isMostAccurate", "note"]:
            if c in ana_df.columns:
                df[f"analysis_{c}"] = ana_df[c]
        df = df.drop(columns=["cmeAnalyses"])

    return df

def load_donki_flare(path: Path) -> pd.DataFrame:
    data = load_json(path)
    df = pd.json_normalize(data)
    keep = [
        "flrID",
        "beginTime",
        "peakTime",
        "endTime",
        "classType",
        "sourceLocation",
        "activeRegionNum",
        "link",
    ]
    cols = [c for c in keep if c in df.columns]
    return df[cols].copy()

def load_kp_csv(path: Path) -> pd.DataFrame:
    """
    Expect columns like:
      - time (UTC) OR datetime
      - kp (numeric)
    If your file differs, adjust here.
    """
    df = pd.read_csv(path)
    # Normalize expected column names
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ["time", "datetime", "date_time", "timestamp", "utc_time"]:
            rename_map[c] = "time"
        if cl in ["kp", "kp_index", "kpvalue"]:
            rename_map[c] = "kp"
    df = df.rename(columns=rename_map)

    if "time" not in df.columns or "kp" not in df.columns:
        raise ValueError(f"Kp CSV must contain time and kp columns. Found: {df.columns.tolist()}")

    return df

def load_omni_csv(path: Path) -> pd.DataFrame:
    """
    Expect a time column and solar wind / IMF columns.
    Common columns (examples): time, V, N, Bz, Bt, etc.
    Adjust mapping based on your OMNI file format.
    """
    df = pd.read_csv(path)
    # Try to normalize a time column
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ["time", "datetime", "timestamp", "utc_time"]:
            rename_map[c] = "time"
    df = df.rename(columns=rename_map)

    if "time" not in df.columns:
        raise ValueError("OMNI CSV must contain a time/datetime column.")
    return df
