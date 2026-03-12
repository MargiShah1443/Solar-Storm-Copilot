"""
src/preprocessing/features.py
------------------------------
Adds interpretable derived features to CME and flare DataFrames
before model training.

These features are wired into run_eda.py so they are included in
the saved modeling table (data/processed/cme_features_labeled.csv).

Note: flare_strength is now computed inside load_donki_flare()
(loaders.py), so it does not need to be repeated here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def add_cme_features(df_cme: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to the CME DataFrame.

    New columns:
      log_speed      — log1p of analysis_speed (reduces right-skew)
      start_hour     — hour of day of CME eruption (0–23)
      start_dow      — day of week (0=Mon … 6=Sun)
      is_fast_cme    — 1 if analysis_speed > 1000 km/s, else 0
      abs_latitude   — absolute value of analysis_latitude
    """
    df = df_cme.copy()

    if "analysis_speed" in df.columns:
        speed = pd.to_numeric(df["analysis_speed"], errors="coerce")
        df["log_speed"] = np.log1p(speed)
        df["is_fast_cme"] = (speed > 1000).astype("Int8")

    if "analysis_latitude" in df.columns:
        lat = pd.to_numeric(df["analysis_latitude"], errors="coerce")
        df["abs_latitude"] = lat.abs()

    if "startTime" in df.columns:
        dt = pd.to_datetime(df["startTime"], utc=True, errors="coerce")
        df["start_hour"] = dt.dt.hour
        df["start_dow"] = dt.dt.dayofweek

    log.info(
        "CME features added: log_speed, is_fast_cme, abs_latitude, start_hour, start_dow"
    )
    return df


def add_flare_features(df_flr: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to the flare DataFrame.

    New columns:
      flare_duration_min — duration from beginTime to endTime in minutes
      start_hour         — hour of day of flare start
      start_dow          — day of week of flare start
    """
    df = df_flr.copy()

    if "beginTime" in df.columns and "endTime" in df.columns:
        begin = pd.to_datetime(df["beginTime"], utc=True, errors="coerce")
        end = pd.to_datetime(df["endTime"], utc=True, errors="coerce")
        df["flare_duration_min"] = (end - begin).dt.total_seconds() / 60.0

    if "beginTime" in df.columns:
        dt = pd.to_datetime(df["beginTime"], utc=True, errors="coerce")
        df["start_hour"] = dt.dt.hour
        df["start_dow"] = dt.dt.dayofweek

    log.info("Flare features added: flare_duration_min, start_hour, start_dow")
    return df