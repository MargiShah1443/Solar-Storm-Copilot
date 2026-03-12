"""
src/preprocessing/omni_window_features.py
------------------------------------------
All event-level feature construction that joins time series data
(OMNI solar wind, Kp) onto CME events.

Previously, Kp labeling lived in both scripts/run_eda.py
(label_storms_from_kp) and src/preprocessing/merge.py
(add_kp_labels_for_cmes).  Both are now removed and replaced by
add_kp_labels() here — the single authoritative implementation.

The OMNI window feature builder from merge.py
(merge_omni_window_features) is also superseded by the faster
add_omni_arrival_window_features() below, which uses searchsorted
instead of per-row boolean filters.

Public API:
  add_kp_labels(cme_df, kp_df, ...)          -> cme_df + kp_max + label_storm
  add_omni_arrival_window_features(...)       -> cme_df + omni window stats
  omni_coverage_flag(df, cfg)                 -> boolean Series
  OmniWindowConfig                            -> config dataclass
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

log = get_logger(__name__)


# ------------------------------------------------------------------ #
# Configuration                                                       #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class OmniWindowConfig:
    """
    Defines the OMNI arrival window relative to each CME start time T:
      window = [T + start_hours,  T + end_hours]

    The default T+24h → T+72h captures the solar wind conditions
    during the expected CME transit / geomagnetic response period.
    """
    start_hours: int = 24
    end_hours: int = 72

    # OMNI columns to summarise (must exist in omni_df)
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

    # Summary statistics to compute per column per window
    stats: Tuple[str, ...] = ("mean", "min", "max", "std")


# ------------------------------------------------------------------ #
# 1.  Kp labeling  (replaces merge.py + run_eda.label_storms_from_kp)#
# ------------------------------------------------------------------ #

def add_kp_labels(
    cme_df: pd.DataFrame,
    kp_df: pd.DataFrame,
    cme_time_col: str = "startTime",
    kp_time_col: str = "time",
    lookahead_hours: int = 72,
    storm_threshold: float = 5.0,
) -> pd.DataFrame:
    """
    For each CME at time T, compute:
      kp_max_h{lookahead_hours}  — maximum Kp in [T, T + lookahead_hours]
      label_storm                — 1 if kp_max >= storm_threshold, else 0

    Uses numpy searchsorted for fast binary-search windowing instead of
    per-row boolean filtering.

    Parameters
    ----------
    cme_df : CME DataFrame with a datetime column (cme_time_col).
    kp_df  : Kp DataFrame with columns (kp_time_col, 'kp').
    cme_time_col : name of the time column in cme_df.
    kp_time_col  : name of the time column in kp_df.
    lookahead_hours : window length in hours after each CME.
    storm_threshold : Kp value that defines a geomagnetic storm (≥ this).

    Returns
    -------
    Copy of cme_df with two new columns appended.
    """
    if cme_time_col not in cme_df.columns:
        raise ValueError(f"cme_df missing column: '{cme_time_col}'")
    if kp_time_col not in kp_df.columns or "kp" not in kp_df.columns:
        raise ValueError("kp_df must have columns: time, kp")

    df = cme_df.copy()
    df[cme_time_col] = pd.to_datetime(df[cme_time_col], utc=True, errors="coerce")

    kp = (
        kp_df[["time" if kp_time_col == "time" else kp_time_col, "kp"]]
        .copy()
        .rename(columns={kp_time_col: "time"})
    )
    kp["time"] = pd.to_datetime(kp["time"], utc=True, errors="coerce")
    kp = kp.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    kp_times = kp["time"].to_numpy(dtype="datetime64[ns]")
    kp_vals = kp["kp"].to_numpy(dtype=float)
    end_delta = pd.Timedelta(hours=lookahead_hours)

    kp_max_list: List[float] = []
    for t in df[cme_time_col]:
        if pd.isna(t):
            kp_max_list.append(float("nan"))
            continue
        w_start = t.to_datetime64()
        w_end = (t + end_delta).to_datetime64()
        left = kp_times.searchsorted(w_start, side="left")
        right = kp_times.searchsorted(w_end, side="right")
        window = kp_vals[left:right]
        kp_max_list.append(float(np.nanmax(window)) if window.size else float("nan"))

    col = f"kp_max_h{lookahead_hours}"
    df[col] = kp_max_list
    df["label_storm"] = (df[col] >= storm_threshold).astype(int)

    n_storms = int(df["label_storm"].sum())
    n_total = len(df)
    log.info(
        "Kp labels added: %d / %d CMEs are storms (%.1f%%).",
        n_storms, n_total, 100 * n_storms / max(n_total, 1),
    )
    return df


# ------------------------------------------------------------------ #
# 2.  OMNI arrival-window features                                    #
# ------------------------------------------------------------------ #

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
    raise ValueError(f"Unknown stat: '{stat}'")


def add_omni_arrival_window_features(
    cme_df: pd.DataFrame,
    omni_df: pd.DataFrame,
    cme_time_col: str = "startTime",
    omni_time_col: str = "time",
    cfg: OmniWindowConfig = OmniWindowConfig(),
) -> pd.DataFrame:
    """
    For each CME event at time T, compute summary statistics of OMNI
    solar wind variables in the window [T + start_hours, T + end_hours].

    New column names follow the pattern:
      omni_w{start}_{end}_{stat}_{variable}
    e.g. omni_w24_72_mean_Bz_gsm

    Parameters
    ----------
    cme_df  : one row per CME.
    omni_df : 5-min OMNI solar wind time series.
    cfg     : OmniWindowConfig controlling window bounds, variables, stats.

    Returns
    -------
    Copy of cme_df with new feature columns appended.
    """
    if cme_time_col not in cme_df.columns:
        raise ValueError(f"cme_df missing column: '{cme_time_col}'")
    if omni_time_col not in omni_df.columns:
        raise ValueError(f"omni_df missing column: '{omni_time_col}'")

    df_cme = cme_df.copy()
    df_omni = omni_df.copy()

    df_cme[cme_time_col] = pd.to_datetime(df_cme[cme_time_col], utc=True, errors="coerce")
    df_omni[omni_time_col] = pd.to_datetime(df_omni[omni_time_col], utc=True, errors="coerce")
    df_omni = df_omni.dropna(subset=[omni_time_col]).sort_values(omni_time_col).reset_index(drop=True)

    # Only keep OMNI columns that actually exist
    omni_cols = [c for c in cfg.omni_columns if c in df_omni.columns]
    if not omni_cols:
        raise ValueError(
            f"None of the requested OMNI columns exist. "
            f"Requested: {cfg.omni_columns}. "
            f"Available: {list(df_omni.columns)}"
        )

    missing_cols = set(cfg.omni_columns) - set(omni_cols)
    if missing_cols:
        log.warning("OMNI columns not found (will be skipped): %s", missing_cols)

    omni_times = df_omni[omni_time_col].to_numpy(dtype="datetime64[ns]")
    omni_values: Dict[str, np.ndarray] = {
        col: df_omni[col].to_numpy(dtype=float) for col in omni_cols
    }

    prefix = f"omni_w{cfg.start_hours}_{cfg.end_hours}"
    new_cols: Dict[str, List[float]] = {
        f"{prefix}_{stat}_{col}": []
        for col in omni_cols
        for stat in cfg.stats
    }

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

        for col in omni_cols:
            window = omni_values[col][left:right]
            for stat in cfg.stats:
                new_cols[f"{prefix}_{stat}_{col}"].append(_safe_stat(window, stat))

    for k, v in new_cols.items():
        df_cme[k] = v

    log.info(
        "OMNI window features added: %d columns for %d CME events.",
        len(new_cols), len(df_cme),
    )
    return df_cme


# ------------------------------------------------------------------ #
# 3.  Coverage flag                                                   #
# ------------------------------------------------------------------ #

def omni_coverage_flag(
    df: pd.DataFrame,
    cfg: OmniWindowConfig = OmniWindowConfig(),
    require_any_of: Tuple[str, ...] = ("mean_Bz_gsm", "mean_V", "mean_Bmag"),
) -> pd.Series:
    """
    Return a boolean Series: True if a row has at least one usable OMNI
    feature in the arrival window.

    Checks the fully-qualified column names derived from cfg:
      "mean_Bz_gsm"  ->  "omni_w24_72_mean_Bz_gsm"
    """
    prefix = f"omni_w{cfg.start_hours}_{cfg.end_hours}_"
    candidates = [prefix + s for s in require_any_of if (prefix + s) in df.columns]

    if not candidates:
        log.warning(
            "omni_coverage_flag: none of %s found in DataFrame — returning all False.",
            [prefix + s for s in require_any_of],
        )
        return pd.Series(False, index=df.index)

    return df[candidates].notna().any(axis=1)