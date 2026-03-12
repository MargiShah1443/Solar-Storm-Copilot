import pandas as pd
import numpy as np

def to_utc_datetime(series: pd.Series) -> pd.Series:
    # DONKI timestamps usually are ISO strings, sometimes already UTC
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    return dt

def clean_kp(df_kp: pd.DataFrame) -> pd.DataFrame:
    df = df_kp.copy()
    df["time"] = to_utc_datetime(df["time"])
    df["kp"] = pd.to_numeric(df["kp"], errors="coerce")
    df = df.dropna(subset=["time", "kp"]).sort_values("time").reset_index(drop=True)
    # Kp is typically 0–9, but some sources include thirds; keep within [0, 9]
    df = df[(df["kp"] >= 0) & (df["kp"] <= 9)]
    return df

def clean_omni(df_omni: pd.DataFrame) -> pd.DataFrame:
    df = df_omni.copy()
    df["time"] = to_utc_datetime(df["time"])
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    # Convert all non-time columns to numeric where possible
    for c in df.columns:
        if c == "time":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def clean_cme(df_cme: pd.DataFrame) -> pd.DataFrame:
    df = df_cme.copy()
    if "startTime" in df.columns:
        df["startTime"] = to_utc_datetime(df["startTime"])
    # Cast likely numeric analysis fields
    for c in df.columns:
        if c.startswith("analysis_"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def clean_flare(df_flr: pd.DataFrame) -> pd.DataFrame:
    df = df_flr.copy()
    for c in ["beginTime", "peakTime", "endTime"]:
        if c in df.columns:
            df[c] = to_utc_datetime(df[c])
    return df

def summarize_missingness(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().mean().sort_values(ascending=False)
    return miss.reset_index().rename(columns={"index": "column", 0: "missing_fraction"})
