from __future__ import annotations

import pandas as pd


def year_counts_from_events(df_events: pd.DataFrame, time_col: str) -> pd.Series:
    years = pd.to_datetime(df_events[time_col], utc=True, errors="coerce").dt.year
    return years.value_counts().sort_index()


def make_yearly_coverage_table(
    df_cme: pd.DataFrame,
    df_kp: pd.DataFrame,
    df_omni: pd.DataFrame,
    horizon_hours: int = 72,
    omni_window_hours: int = 24,
    kp_threshold: float = 5.0,
) -> pd.DataFrame:
    """
    Produces a per-year table:
      - total CMEs
      - CMEs with any Kp data in [t0, t0+horizon]
      - CMEs with OMNI data in [t0-window, t0]
      - storm rate among labeled CMEs
    """
    cme = df_cme.copy()
    cme["startTime"] = pd.to_datetime(cme["startTime"], utc=True, errors="coerce")
    cme = cme.dropna(subset=["startTime"]).copy()
    cme["year"] = cme["startTime"].dt.year

    kp = df_kp.copy()
    kp["time"] = pd.to_datetime(kp["time"], utc=True, errors="coerce")
    kp = kp.dropna(subset=["time"]).sort_values("time")

    omni = df_omni.copy()
    omni["time"] = pd.to_datetime(omni["time"], utc=True, errors="coerce")
    omni = omni.dropna(subset=["time"]).sort_values("time")

    years = sorted(cme["year"].unique())
    rows = []

    for y in years:
        cme_y = cme[cme["year"] == y].copy()
        total = len(cme_y)

        kp_labeled = 0
        storm_count = 0
        omni_available = 0

        for t0 in cme_y["startTime"]:
            # Kp availability in future window
            t1 = t0 + pd.Timedelta(hours=horizon_hours)
            kp_window = kp[(kp["time"] >= t0) & (kp["time"] <= t1)]
            if len(kp_window) > 0:
                kp_labeled += 1
                kp_max = kp_window["kp"].max()
                if pd.notna(kp_max) and kp_max >= kp_threshold:
                    storm_count += 1

            # OMNI availability in past window
            t_prev = t0 - pd.Timedelta(hours=omni_window_hours)
            omni_window = omni[(omni["time"] >= t_prev) & (omni["time"] <= t0)]
            if len(omni_window) > 0:
                omni_available += 1

        storm_rate = (storm_count / kp_labeled) if kp_labeled > 0 else 0.0
        kp_coverage = (kp_labeled / total) if total > 0 else 0.0
        omni_coverage = (omni_available / total) if total > 0 else 0.0

        rows.append(
            {
                "year": y,
                "cme_total": total,
                "cme_with_kp": kp_labeled,
                "kp_coverage_pct": round(100 * kp_coverage, 1),
                "cme_with_omni": omni_available,
                "omni_coverage_pct": round(100 * omni_coverage, 1),
                "storm_count": storm_count,
                "storm_rate_pct": round(100 * storm_rate, 1),
            }
        )

    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


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
    Scores a candidate contiguous year window based on simple thresholds.
    """
    sub = coverage_df[(coverage_df["year"] >= start_year) & (coverage_df["year"] <= end_year)].copy()
    total_cmes = int(sub["cme_total"].sum())
    total_storms = int(sub["storm_count"].sum())

    # average coverages across years (simple mean)
    avg_kp_cov = float(sub["kp_coverage_pct"].mean()) if len(sub) else 0.0
    avg_omni_cov = float(sub["omni_coverage_pct"].mean()) if len(sub) else 0.0

    passes = (
        total_cmes >= min_cmes
        and total_storms >= min_storms
        and avg_kp_cov >= min_kp_cov_pct
        and avg_omni_cov >= min_omni_cov_pct
    )

    return {
        "start_year": start_year,
        "end_year": end_year,
        "total_cmes": total_cmes,
        "total_storms": total_storms,
        "avg_kp_coverage_pct": round(avg_kp_cov, 1),
        "avg_omni_coverage_pct": round(avg_omni_cov, 1),
        "passes_thresholds": passes,
    }
