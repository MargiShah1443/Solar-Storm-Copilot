"""
scripts/explore_time_window.py
-------------------------------
Diagnostic script to explore data coverage and storm rates by year.
Useful for choosing the final training date range and verifying that
OMNI / Kp data is adequately populated after running download_data.py.

Previously, this script defined its own local yearly_coverage_table()
function which was a near-duplicate of make_yearly_coverage_table()
in src/eda/coverage.py.  That duplicate has been removed.
All coverage logic now lives in src/eda/coverage.py.
"""
from __future__ import annotations

from src.config import Config
from src.eda.coverage import make_yearly_coverage_table, score_year_window
from src.io.loaders import load_donki_cme, load_kp_csv, load_omni_csv
from src.preprocessing.omni_window_features import OmniWindowConfig, add_kp_labels
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def main() -> None:
    cfg = Config()
    omni_cfg = OmniWindowConfig(
        start_hours=cfg.omni_window_start_hours,
        end_hours=cfg.omni_window_end_hours,
    )

    log.info("Loading datasets for time-window exploration...")
    df_cme = load_donki_cme(cfg.donki_cme_file)
    df_kp = load_kp_csv(cfg.kp_file)
    df_omni = load_omni_csv(cfg.omni_file)

    log.info("Labeling storms using Kp lookahead window...")
    df_labeled = add_kp_labels(
        df_cme,
        df_kp,
        lookahead_hours=cfg.kp_lookahead_hours,
        storm_threshold=cfg.kp_storm_threshold,
    )

    log.info("Computing per-year coverage table...")
    cov = make_yearly_coverage_table(
        df_labeled,
        df_kp,
        df_omni,
        horizon_hours=cfg.kp_lookahead_hours,
        omni_window_hours=omni_cfg.start_hours,
        kp_threshold=cfg.kp_storm_threshold,
    )

    print("\n=== Yearly Coverage Summary ===")
    print(cov.to_string(index=False))

    # Score the full 2015–2023 window as a candidate training range
    print("\n=== Window Scoring: 2015–2023 ===")
    score = score_year_window(
        cov,
        start_year=cfg.start_year,
        end_year=cfg.end_year,
    )
    for k, v in score.items():
        print(f"  {k}: {v}")

    # Also score a shorter candidate window (e.g. 2017–2023)
    print("\n=== Window Scoring: 2017–2023 ===")
    score2 = score_year_window(cov, start_year=2017, end_year=2023)
    for k, v in score2.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()