"""
src/eda/coverage.py
--------------------
Data coverage and year-window scoring utilities.

Previously, explore_time_window.py defined its own local version of
yearly_coverage_table().  That duplicate has been removed; callers
should now import directly from here.

Public API:
  make_yearly_coverage_table(cme_df, kp_df, omni_df, ...)  -> DataFrame
  score_year_window(coverage_df, start_year, end_year, ...) -> dict
"""
from __future__ import annotations

import pandas as pd

from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def make_yearly_coverage_table(
    df_cme: pd.DataFrame,
    df_kp: pd.DataFrame,
    df_omni: pd.DataFrame,
    horizon_hours: int = 72,
    omni_window_hours: int = 24,
    kp_threshold: float = 5.0,
) -> pd.DataFrame:
    """
    Build a per-year summary table showing data coverage and storm rates.

    Columns returned:
      year, cme_total, cme_with_kp, kp_coverage_pct,
      cme_with_omni, omni_coverage_pct, storm_count, storm_rate_pct

    Parameters
    ----------
    df_cme  : cleaned CME DataFrame (must have 'startTime').
    df_kp   : cleaned Kp DataFrame (must have 'time', 'kp').
    df_omni : cleaned OMNI DataFrame (must have 'time').
    horizon_hours    : how many hours after CME to look for Kp data.
    omni_window_hours: how many hours before CME to check for OMNI data.
    kp_threshold     : Kp ≥ this value counts as a storm.
    """
    cme = df_cme.copy()
    cme["startTime"] = pd.to_datetime(cme["startTime"], utc=True, errors="coerce")
    cme = cme.dropna(subset=["startTime"]).copy()
    cme["year"] = cme["startTime"].dt.year

    kp = df_kp[["time", "kp"]].copy()
    kp["time"] = pd.to_datetime(kp["time"], utc=True, errors="coerce")
    kp = kp.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    omni = df_omni[["time"]].copy()
    omni["time"] = pd.to_datetime(omni["time"], utc=True, errors="coerce")
    omni = omni.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    # Pre-convert to numpy for fast searchsorted
    kp_times = kp["time"].to_numpy(dtype="datetime64[ns]")
    kp_vals = kp["kp"].to_numpy(dtype=float)
    omni_times = omni["time"].to_numpy(dtype="datetime64[ns]")

    rows = []
    for y in sorted(cme["year"].unique()):
        cme_y = cme[cme["year"] == y]
        total = len(cme_y)
        kp_labeled = 0
        storm_count = 0
        omni_available = 0

        for t0 in cme_y["startTime"]:
            # Kp: future window [t0, t0 + horizon]
            t1 = t0 + pd.Timedelta(hours=horizon_hours)
            l = kp_times.searchsorted(t0.to_datetime64(), side="left")
            r = kp_times.searchsorted(t1.to_datetime64(), side="right")
            if r > l:
                kp_labeled += 1
                kp_max = float(kp_vals[l:r].max())
                if kp_max >= kp_threshold:
                    storm_count += 1

            # OMNI: past window [t0 - window, t0]
            t_prev = t0 - pd.Timedelta(hours=omni_window_hours)
            lo = omni_times.searchsorted(t_prev.to_datetime64(), side="left")
            hi = omni_times.searchsorted(t0.to_datetime64(), side="right")
            if hi > lo:
                omni_available += 1

        rows.append({
            "year": y,
            "cme_total": total,
            "cme_with_kp": kp_labeled,
            "kp_coverage_pct": round(100 * kp_labeled / total, 1) if total else 0.0,
            "cme_with_omni": omni_available,
            "omni_coverage_pct": round(100 * omni_available / total, 1) if total else 0.0,
            "storm_count": storm_count,
            "storm_rate_pct": round(100 * storm_count / kp_labeled, 1) if kp_labeled else 0.0,
        })

    df_out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    log.info("Coverage table built for %d years.", len(df_out))
    return df_out


def score_year_window(
    coverage_df: pd.DataFrame,
    start_year: int,
    end_year: int,
    min_cmes: int = 300,
    min_storms: int = 20,
    min_kp_cov_pct: float = 85.0,
    min_omni_cov_pct: float = 85.0,
) -> dict:
    """
    Score a candidate contiguous year window against simple sufficiency
    thresholds.  Useful for choosing the final training date range.

    Returns a dict with totals, average coverages, and a boolean
    'passes_thresholds' flag.
    """
    sub = coverage_df[
        coverage_df["year"].between(start_year, end_year)
    ].copy()

    total_cmes = int(sub["cme_total"].sum())
    total_storms = int(sub["storm_count"].sum())
    avg_kp_cov = float(sub["kp_coverage_pct"].mean()) if len(sub) else 0.0
    avg_omni_cov = float(sub["omni_coverage_pct"].mean()) if len(sub) else 0.0

    passes = (
        total_cmes >= min_cmes
        and total_storms >= min_storms
        and avg_kp_cov >= min_kp_cov_pct
        and avg_omni_cov >= min_omni_cov_pct
    )

    result = {
        "start_year": start_year,
        "end_year": end_year,
        "total_cmes": total_cmes,
        "total_storms": total_storms,
        "avg_kp_coverage_pct": round(avg_kp_cov, 1),
        "avg_omni_coverage_pct": round(avg_omni_cov, 1),
        "passes_thresholds": passes,
    }
    log.info("Window %d–%d: %s", start_year, end_year, result)
    return result