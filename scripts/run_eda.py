"""
scripts/run_eda.py
------------------
Main EDA script.  Run after download_data.py has populated data/raw/.

Pipeline:
  1. Load clean data (CME, Kp, OMNI)
  2. Add Kp storm labels
  3. Add CME derived features  (log_speed, is_fast_cme, etc.)
  4. Add OMNI arrival-window features  (T+24h → T+72h)
  5. Generate EDA figures → outputs/figures/
  6. Save modeling table  → data/processed/cme_features_labeled.csv
  7. Save missingness report → data/interim/missingness_cme_features.csv

Key changes from previous version:
  - label_storms_from_kp() removed; now uses add_kp_labels() from
    src/preprocessing/omni_window_features.py
  - CME features (log_speed, is_fast_cme, etc.) now wired in via
    src/preprocessing/features.py (previously unused)
  - Four new EDA plots: class_imbalance, storm_rate_by_speed,
    omni_by_label, storm_rate_by_year
  - Feature table saved to data/processed/ (not data/raw/)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Config
from src.eda.plots import (
    plot_class_imbalance,
    plot_cme_speed_hist,
    plot_corr_heatmap,
    plot_kp_outcome_by_label,
    plot_kp_timeline,
    plot_missingness_top,
    plot_omni_by_label,
    plot_storm_rate_by_speed,
    plot_storm_rate_by_year,
)
from src.io.loaders import load_donki_cme, load_kp_csv, load_omni_csv
from src.preprocessing.features import add_cme_features
from src.preprocessing.omni_window_features import (
    OmniWindowConfig,
    add_kp_labels,
    add_omni_arrival_window_features,
)
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def main() -> None:
    cfg = Config()
    cfg.makedirs()
    fig_dir: Path = cfg.figures_dir

    # ---------------------------------------------------------------- #
    # 1. Load                                                           #
    # ---------------------------------------------------------------- #
    log.info("Loading raw data...")
    df_cme = load_donki_cme(cfg.donki_cme_file)
    df_kp = load_kp_csv(cfg.kp_file)
    df_omni = load_omni_csv(cfg.omni_file)

    log.info(
        "Loaded: CME=%d rows, Kp=%d rows, OMNI=%d rows",
        len(df_cme), len(df_kp), len(df_omni),
    )

    # ---------------------------------------------------------------- #
    # 2. Kp storm labels                                               #
    # ---------------------------------------------------------------- #
    log.info("Adding Kp storm labels (lookahead=%dh, threshold=%.1f)...",
             cfg.kp_lookahead_hours, cfg.kp_storm_threshold)
    df_labeled = add_kp_labels(
        df_cme,
        df_kp,
        lookahead_hours=cfg.kp_lookahead_hours,
        storm_threshold=cfg.kp_storm_threshold,
    )

    # ---------------------------------------------------------------- #
    # 3. CME derived features                                          #
    # ---------------------------------------------------------------- #
    log.info("Adding CME derived features (log_speed, is_fast_cme, ...)...")
    df_labeled = add_cme_features(df_labeled)

    # ---------------------------------------------------------------- #
    # 4. OMNI arrival-window features  (T+24h → T+72h)                #
    # ---------------------------------------------------------------- #
    omni_cfg = OmniWindowConfig(
        start_hours=cfg.omni_window_start_hours,
        end_hours=cfg.omni_window_end_hours,
    )
    log.info(
        "Adding OMNI window features (T+%dh → T+%dh)...",
        omni_cfg.start_hours, omni_cfg.end_hours,
    )
    df_features = add_omni_arrival_window_features(
        df_labeled,
        df_omni,
        cme_time_col="startTime",
        omni_time_col="time",
        cfg=omni_cfg,
    )

    log.info("Feature table shape: %s", df_features.shape)

    # Quick summary
    miss_mean = df_features.isna().mean().mean()
    miss_gt50 = int((df_features.isna().mean() > 0.5).sum())
    storm_rate = df_features["label_storm"].mean()
    log.info(
        "Summary: missing_mean=%.1f%%, cols_gt50pct_missing=%d, storm_rate=%.1f%%",
        100 * miss_mean, miss_gt50, 100 * storm_rate,
    )

    # ---------------------------------------------------------------- #
    # 5. EDA figures                                                   #
    # ---------------------------------------------------------------- #
    log.info("Generating EDA figures → %s", fig_dir)

    # --- original plots ---
    plot_cme_speed_hist(df_features, fig_dir / "cme_speed_hist.png", dpi=cfg.fig_dpi)
    plot_corr_heatmap(df_features, fig_dir / "corr_heatmap.png", dpi=cfg.fig_dpi)
    plot_kp_outcome_by_label(df_features, fig_dir / "kp_outcome_by_label.png", dpi=cfg.fig_dpi)
    plot_kp_timeline(df_kp, fig_dir / "kp_timeline.png", dpi=cfg.fig_dpi)
    plot_missingness_top(df_features, fig_dir / "missingness_top.png", dpi=cfg.fig_dpi)

    # --- new plots ---
    plot_class_imbalance(df_features, fig_dir / "class_imbalance.png", dpi=cfg.fig_dpi)
    plot_storm_rate_by_speed(df_features, fig_dir / "storm_rate_by_speed.png", dpi=cfg.fig_dpi)
    plot_omni_by_label(df_features, fig_dir / "omni_by_label.png", dpi=cfg.fig_dpi)
    plot_storm_rate_by_year(df_features, fig_dir / "storm_rate_by_year.png", dpi=cfg.fig_dpi)

    # ---------------------------------------------------------------- #
    # 6. Save modeling table                                           #
    # ---------------------------------------------------------------- #
    out_table = cfg.feature_table_file   # data/processed/cme_features_labeled.csv
    df_features.to_csv(out_table, index=False)
    log.info("Saved modeling table → %s  (%d rows, %d cols)",
             out_table, *df_features.shape)

    # ---------------------------------------------------------------- #
    # 7. Save missingness report                                       #
    # ---------------------------------------------------------------- #
    miss_report = (
        df_features.isna()
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_fraction"})
    )
    miss_out = cfg.missingness_file   # data/interim/missingness_cme_features.csv
    miss_report.to_csv(miss_out, index=False)
    log.info("Saved missingness report → %s", miss_out)

    log.info("=== EDA complete. ===")
    log.info("Next step: open data/processed/cme_features_labeled.csv for modeling.")


if __name__ == "__main__":
    main()