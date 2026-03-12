"""
src/io/loaders.py
-----------------
All data-loading functions for the pipeline.
Each loader returns a *clean* DataFrame — type coercion, datetime
normalisation, and fill-value handling are applied internally so that
callers never need to import clean.py separately.

clean.py is therefore no longer needed and has been removed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.utils.logging_utils import get_logger

log = get_logger(__name__)


# ------------------------------------------------------------------ #
# Internal helpers (replaces src/preprocessing/clean.py)             #
# ------------------------------------------------------------------ #

def _to_utc(series: pd.Series) -> pd.Series:
    """Parse a Series of date strings or mixed types to UTC datetime."""
    return pd.to_datetime(series, errors="coerce", utc=True)


def _to_numeric_cols(df: pd.DataFrame, exclude: tuple[str, ...] = ()) -> pd.DataFrame:
    """Cast all non-excluded columns to numeric, coercing errors to NaN."""
    for c in df.columns:
        if c not in exclude:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ------------------------------------------------------------------ #
# JSON helper                                                         #
# ------------------------------------------------------------------ #

def _load_json(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ #
# Public loaders                                                      #
# ------------------------------------------------------------------ #

def load_donki_cme(path: Path) -> pd.DataFrame:
    """
    Load and clean the DONKI CME JSON file.

    Returned columns (where present):
      activityID, startTime (UTC), sourceLocation, activeRegionNum,
      analysis_speed, analysis_type, analysis_latitude,
      analysis_longitude, analysis_halfAngle, analysis_isMostAccurate
    """
    log.info("Loading DONKI CME from %s", path)
    data = _load_json(path)
    df = pd.json_normalize(data)

    keep = [
        "activityID",
        "startTime",
        "sourceLocation",
        "activeRegionNum",
        "cmeAnalyses",
        "note",
        "link",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()

    # Flatten the first (most accurate) CME analysis entry
    if "cmeAnalyses" in df.columns:
        def _first(x):
            if isinstance(x, list) and x and isinstance(x[0], dict):
                return x[0]
            return {}

        ana_df = pd.json_normalize(df["cmeAnalyses"].apply(_first))
        for c in ["speed", "type", "latitude", "longitude", "halfAngle", "isMostAccurate", "note"]:
            if c in ana_df.columns:
                df[f"analysis_{c}"] = ana_df[c].values
        df = df.drop(columns=["cmeAnalyses"])

    # --- clean ---
    df["startTime"] = _to_utc(df["startTime"])
    for c in df.columns:
        if c.startswith("analysis_") and c != "analysis_type":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    log.info("  CME rows loaded: %d", len(df))
    return df


def load_donki_flare(path: Path) -> pd.DataFrame:
    """
    Load and clean the DONKI solar flare JSON file.

    Returned columns (where present):
      flrID, beginTime, peakTime, endTime (all UTC),
      classType, flare_strength (numeric), sourceLocation, activeRegionNum
    """
    log.info("Loading DONKI FLR from %s", path)
    data = _load_json(path)
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
    df = df[[c for c in keep if c in df.columns]].copy()

    # --- clean ---
    for c in ["beginTime", "peakTime", "endTime"]:
        if c in df.columns:
            df[c] = _to_utc(df[c])

    # Numeric flare strength  (replaces features.parse_flare_class_strength)
    if "classType" in df.columns:
        df["flare_strength"] = _parse_flare_strength(df["classType"])

    log.info("  FLR rows loaded: %d", len(df))
    return df


def _parse_flare_strength(class_series: pd.Series) -> pd.Series:
    """
    Convert GOES flare class strings ('M1.4', 'X2.0', …) to a
    continuous numeric scale where X1.0 = 1.0.

    Letter multipliers: A=1e-4, B=1e-3, C=1e-2, M=1e-1, X=1.0
    """
    _mult = {"A": 1e-4, "B": 1e-3, "C": 1e-2, "M": 1e-1, "X": 1.0}

    def _convert(x):
        if not isinstance(x, str) or len(x) < 2:
            return None
        letter = x[0].upper()
        try:
            number = float(x[1:])
        except ValueError:
            return None
        return _mult.get(letter, None) and _mult[letter] * number

    return class_series.apply(_convert)


def load_kp_csv(path: Path) -> pd.DataFrame:
    """
    Load and clean the Kp 3-hourly CSV.

    Accepts any reasonable column naming; normalises to:
      time (UTC datetime), kp (float, clipped to [0, 9])
    """
    log.info("Loading Kp CSV from %s", path)
    df = pd.read_csv(path)

    # Normalise column names
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl in {"time", "datetime", "date_time", "timestamp", "utc_time"}:
            rename[c] = "time"
        elif cl in {"kp", "kp_index", "kpvalue"}:
            rename[c] = "kp"
    df = df.rename(columns=rename)

    if "time" not in df.columns or "kp" not in df.columns:
        raise ValueError(
            f"Kp CSV must have 'time' and 'kp' columns. Found: {df.columns.tolist()}"
        )

    # --- clean ---
    df["time"] = _to_utc(df["time"])
    df["kp"] = pd.to_numeric(df["kp"], errors="coerce")
    df = (
        df.dropna(subset=["time", "kp"])
        .query("0 <= kp <= 9")
        .sort_values("time")
        .reset_index(drop=True)
    )

    log.info("  Kp rows loaded: %d", len(df))
    return df


def load_omni_csv(path: Path) -> pd.DataFrame:
    """
    Load and clean the combined OMNI CSV produced by build_omni_csv().

    Normalises the time column and casts all other columns to numeric.
    """
    log.info("Loading OMNI CSV from %s", path)
    df = pd.read_csv(path)

    rename = {}
    for c in df.columns:
        if c.lower() in {"time", "datetime", "timestamp", "utc_time"}:
            rename[c] = "time"
    df = df.rename(columns=rename)

    if "time" not in df.columns:
        raise ValueError("OMNI CSV must contain a time/datetime column.")

    # --- clean ---
    df["time"] = _to_utc(df["time"])
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    df = _to_numeric_cols(df, exclude=("time",))

    log.info("  OMNI rows loaded: %d", len(df))
    return df