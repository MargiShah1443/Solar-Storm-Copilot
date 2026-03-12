from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Config
from src.io.loaders import load_donki_cme, load_kp_csv, load_omni_csv
from src.preprocessing.omni_window_features import OmniWindowConfig, add_omni_arrival_window_features
from src.eda.plots import (
    plot_cme_speed_hist,
    plot_corr_heatmap,
    plot_kp_outcome_by_label,
    plot_kp_timeline,
    plot_missingness_top,
)


def label_storms_from_kp(
    cme_df: pd.DataFrame,
    kp_df: pd.DataFrame,
    cme_time_col: str = "startTime",
    kp_time_col: str = "time",
    lookahead_hours: int = 72,
    storm_threshold: float = 5.0,
) -> pd.DataFrame:
    """
    For each CME at time T:
      kp_max_h72 = max Kp in [T, T+72h]
      label_storm = 1 if kp_max_h72 >= 5 else 0
    """
    df = cme_df.copy()
    df[cme_time_col] = pd.to_datetime(df[cme_time_col], utc=True, errors="coerce")

    kp = kp_df.copy()
    kp[kp_time_col] = pd.to_datetime(kp[kp_time_col], utc=True, errors="coerce")
    kp = kp.dropna(subset=[kp_time_col]).sort_values(kp_time_col).reset_index(drop=True)

    kp_times = kp[kp_time_col].to_numpy(dtype="datetime64[ns]")
    kp_vals = kp["kp"].to_numpy(dtype=float)

    end_delta = pd.Timedelta(hours=lookahead_hours)

    kp_max_list = []
    for _, row in df.iterrows():
        t = row[cme_time_col]
        if pd.isna(t):
            kp_max_list.append(float("nan"))
            continue

        w_start = t.to_datetime64()
        w_end = (t + end_delta).to_datetime64()

        left = kp_times.searchsorted(w_start, side="left")
        right = kp_times.searchsorted(w_end, side="right")

        window = kp_vals[left:right]
        kp_max_list.append(float(pd.Series(window).max()) if window.size else float("nan"))

    df["kp_max_h72"] = kp_max_list
    df["label_storm"] = (df["kp_max_h72"] >= storm_threshold).astype(int)
    return df


def main():
    cfg = Config()
    fig_dir: Path = cfg.figures_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading raw data...")
    df_cme = load_donki_cme(cfg.donki_cme_file)
    df_kp = load_kp_csv(cfg.kp_file)
    df_omni = load_omni_csv(cfg.omni_file)

    print("[INFO] Cleaning datasets...")

    # Basic CME cleanup: ensure startTime exists
    if "startTime" not in df_cme.columns:
        raise ValueError("CME dataframe missing 'startTime'. Check your CME loader output.")

    # If analysis_speed exists, keep it numeric
    if "analysis_speed" in df_cme.columns:
        df_cme["analysis_speed"] = pd.to_numeric(df_cme["analysis_speed"], errors="coerce")

    # Convert lat/long numeric if present
    for c in ["analysis_latitude", "analysis_longitude", "analysis_halfAngle"]:
        if c in df_cme.columns:
            df_cme[c] = pd.to_numeric(df_cme[c], errors="coerce")

    print("[INFO] Creating labels using Kp (max in next 72h)...")
    df_labeled = label_storms_from_kp(df_cme, df_kp)

    print("[INFO] Adding OMNI arrival-window features (T+24h to T+72h)...")
    omni_cfg = OmniWindowConfig(start_hours=24, end_hours=72)
    df_features = add_omni_arrival_window_features(
        df_labeled,
        df_omni,
        cme_time_col="startTime",
        omni_time_col="time",
        cfg=omni_cfg,
    )

    print(f"[INFO] CME labeled feature table: {df_features.shape}")
    print(f"[INFO] Kp table: {df_kp.shape}")

    # Summary info
    missing_fraction_mean = float(df_features.isna().mean().mean())
    missing_cols_gt_50pct = int((df_features.isna().mean() > 0.5).sum())
    dtypes = df_features.dtypes.value_counts().to_dict()

    print("[INFO] Basic EDA summary:", {
        "n_rows": int(df_features.shape[0]),
        "n_cols": int(df_features.shape[1]),
        "dtypes": {str(k): int(v) for k, v in dtypes.items()},
        "missing_fraction_mean": missing_fraction_mean,
        "missing_cols_gt_50pct": missing_cols_gt_50pct,
    })

    print("[INFO] Generating EDA plots...")

    # Save figures
    plot_cme_speed_hist(df_features, outpath=fig_dir / "cme_speed_hist.png")

    # Correlation: only numeric
    plot_corr_heatmap(df_features, outpath=fig_dir / "corr_heatmap.png")

    plot_kp_outcome_by_label(df_features, outpath=fig_dir / "kp_outcome_by_label.png")

    plot_kp_timeline(df_kp, outpath=fig_dir / "kp_timeline.png")

    plot_missingness_top(df_features, outpath=fig_dir / "missingness_top.png")

    # Optionally save the final modeling table for next stage
    out_table = cfg.raw_dir / "cme_feature_table.csv"
    df_features.to_csv(out_table, index=False)
    print(f"[INFO] Saved modeling table to: {out_table}")

    print(f"[INFO] Done. Figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
