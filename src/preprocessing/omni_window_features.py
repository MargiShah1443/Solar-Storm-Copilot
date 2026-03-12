from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OmniWindowConfig:
    # window is [T + start_hours, T + end_hours]
    start_hours: int = 24
    end_hours: int = 72

    # What OMNI columns to summarize
    omni_columns: Tuple[str, ...] = (
        "Bmag",
        "By_gsm",
        "Bz_gsm",
        "V",
        "Np",
        "T",
        "Pdyn",
        "Efield",
        "beta",
        "MachA",
        "AE",
        "SYM_H",
    )

    # Which summary statistics to compute
    stats: Tuple[str, ...] = ("mean", "min", "max", "std")


def _safe_stat(arr: np.ndarray, stat: str) -> float:
    if arr.size == 0:
        return np.nan

    if stat == "mean":
        return float(np.nanmean(arr))
    if stat == "min":
        return float(np.nanmin(arr))
    if stat == "max":
        return float(np.nanmax(arr))
    if stat == "std":
        return float(np.nanstd(arr))

    raise ValueError(f"Unknown stat: {stat}")


def add_omni_arrival_window_features(
    cme_df: pd.DataFrame,
    omni_df: pd.DataFrame,
    cme_time_col: str = "startTime",
    omni_time_col: str = "time",
    cfg: OmniWindowConfig = OmniWindowConfig(),
) -> pd.DataFrame:
    """
    For each CME event at time T, compute summary stats of OMNI features in the window:
      [T + cfg.start_hours, T + cfg.end_hours]

    Returns a copy of cme_df with new columns like:
      omni_w24_72_mean_Bz_gsm, omni_w24_72_min_Bz_gsm, ...
    """

    if cme_time_col not in cme_df.columns:
        raise ValueError(f"cme_df missing required time column: {cme_time_col}")
    if omni_time_col not in omni_df.columns:
        raise ValueError(f"omni_df missing required time column: {omni_time_col}")

    df_cme = cme_df.copy()
    df_omni = omni_df.copy()

    # Ensure datetimes
    df_cme[cme_time_col] = pd.to_datetime(df_cme[cme_time_col], utc=True, errors="coerce")
    df_omni[omni_time_col] = pd.to_datetime(df_omni[omni_time_col], utc=True, errors="coerce")

    # Sort OMNI by time (needed for fast slicing)
    df_omni = df_omni.dropna(subset=[omni_time_col]).sort_values(omni_time_col).reset_index(drop=True)

    # Filter to only columns that exist
    omni_cols = [c for c in cfg.omni_columns if c in df_omni.columns]

    if len(omni_cols) == 0:
        raise ValueError(
            "None of the requested OMNI columns exist in omni_df. "
            f"Requested: {cfg.omni_columns}. Available: {list(df_omni.columns)}"
        )

    # Convert OMNI time to numpy for searchsorted
    omni_times = df_omni[omni_time_col].to_numpy(dtype="datetime64[ns]")
    omni_values = {col: df_omni[col].to_numpy(dtype=float) for col in omni_cols}

    # Prepare new columns
    prefix = f"omni_w{cfg.start_hours}_{cfg.end_hours}"
    new_cols: Dict[str, List[float]] = {}
    for col in omni_cols:
        for stat in cfg.stats:
            new_cols[f"{prefix}_{stat}_{col}"] = []

    # Build windows and compute stats quickly with searchsorted
    start_delta = pd.Timedelta(hours=cfg.start_hours)
    end_delta = pd.Timedelta(hours=cfg.end_hours)

    for _, row in df_cme.iterrows():
        t = row[cme_time_col]
        if pd.isna(t):
            for k in new_cols:
                new_cols[k].append(np.nan)
            continue

        w_start = (t + start_delta).to_datetime64()
        w_end = (t + end_delta).to_datetime64()

        left = np.searchsorted(omni_times, w_start, side="left")
        right = np.searchsorted(omni_times, w_end, side="right")

        # Slice window arrays
        for col in omni_cols:
            window_arr = omni_values[col][left:right]
            for stat in cfg.stats:
                k = f"{prefix}_{stat}_{col}"
                new_cols[k].append(_safe_stat(window_arr, stat))

    # Attach columns
    for k, v in new_cols.items():
        df_cme[k] = v

    return df_cme


def omni_coverage_flag(
    df: pd.DataFrame,
    cfg: OmniWindowConfig = OmniWindowConfig(),
    require_any_of: Tuple[str, ...] = ("mean_Bz_gsm", "mean_V", "mean_Bmag"),
) -> pd.Series:
    """
    Returns a boolean Series indicating whether a row has usable OMNI features.
    We consider a row covered if at least one key window feature is present.

    We map require_any_of items to full names using prefix:
      mean_Bz_gsm -> omni_w24_72_mean_Bz_gsm
    """
    prefix = f"omni_w{cfg.start_hours}_{cfg.end_hours}_"
    candidates = []
    for short in require_any_of:
        # short looks like "mean_Bz_gsm"
        full = prefix + short
        if full in df.columns:
            candidates.append(full)

    if not candidates:
        # If none of the expected cols exist, treat as no coverage
        return pd.Series(False, index=df.index)

    # Covered if at least one candidate column is not NaN
    return df[candidates].notna().any(axis=1)
