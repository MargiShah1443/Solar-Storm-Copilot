"""
src/preprocessing/splitter.py
-------------------------------
Time-based train / validation / test splitting for the CME dataset.

WHY TIME-BASED (not random):
    The dataset spans Solar Cycles 24 and 25.  Storm rates vary from 0%
    (solar minimum 2020) to 34% (solar maximum 2015).  A random split would
    scatter solar-cycle effects across train/test, causing the model to appear
    better than it is on out-of-distribution future data.

    Time-based splitting tests the realistic question:
        "Can a model trained on past solar cycles predict storms in a new one?"

Default split (configurable):
    Train      : 2015 – 2020  (covers Cycle 24 peak + decline + minimum)
    Validation : 2021          (early Cycle 25 ramp — hyperparameter tuning)
    Test       : 2022 – 2023  (active Cycle 25 — final evaluation, touch once)

Public API:
    time_split(df)              -> (train_df, val_df, test_df)
    extract_Xy(df, feature_cols) -> (X, y)
    summarise_split(train, val, test) -> prints split statistics
"""
from __future__ import annotations

from typing import Optional
import pandas as pd

from src.utils.logging_utils import get_logger
from src.preprocessing.feature_engineering import LABEL_COL, METADATA_COLS

log = get_logger(__name__)

# Default year boundaries — change here or pass as arguments
TRAIN_END_YEAR   = 2020
VAL_YEAR         = 2021
TEST_START_YEAR  = 2022


def time_split(
    df: pd.DataFrame,
    train_end: int = TRAIN_END_YEAR,
    val_year:  int = VAL_YEAR,
    test_start: int = TEST_START_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split an engineered feature DataFrame into train / val / test
    based on the 'year' column.

    Parameters
    ----------
    df          : Output of engineer_features() — must contain 'year' column.
    train_end   : Last year (inclusive) in training set.
    val_year    : Single year used as validation set.
    test_start  : First year (inclusive) in test set.

    Returns
    -------
    (train_df, val_df, test_df) — all index-reset DataFrames.
    """
    if "year" not in df.columns:
        raise ValueError(
            "'year' column not found. Run engineer_features() before splitting."
        )

    train_df = df[df["year"] <= train_end].copy().reset_index(drop=True)
    val_df   = df[df["year"] == val_year].copy().reset_index(drop=True)
    test_df  = df[df["year"] >= test_start].copy().reset_index(drop=True)

    n_total = len(train_df) + len(val_df) + len(test_df)
    if n_total != len(df):
        lost = len(df) - n_total
        log.warning(
            "%d rows fell outside train/val/test boundaries "
            "(years %d–%d, %d, %d+) and were excluded.",
            lost, df["year"].min(), train_end, val_year, test_start,
        )

    summarise_split(train_df, val_df, test_df)
    return train_df, val_df, test_df


def extract_Xy(
    df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    drop_metadata: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate a split DataFrame into feature matrix X and label vector y.

    Parameters
    ----------
    df            : One of (train_df, val_df, test_df) from time_split().
    feature_cols  : Explicit list of feature columns to keep.
                    If None, uses all columns except label, metadata, and year.
    drop_metadata : Whether to drop METADATA_COLS (startTime, year).

    Returns
    -------
    (X, y) — X is a DataFrame, y is a Series.
    """
    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in DataFrame.")

    y = df[LABEL_COL].astype(int)

    if feature_cols is not None:
        # Use explicit list — verify all exist
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            log.warning("Requested feature cols not found: %s", missing)
        X = df[[c for c in feature_cols if c in df.columns]]
    else:
        # Auto-detect: drop label, metadata, year, and any leftover id cols
        exclude = {LABEL_COL, "year", "startTime", "kp_max_h72"}
        if drop_metadata:
            exclude.update(METADATA_COLS)
        X = df[[c for c in df.columns if c not in exclude]]

    log.debug("extract_Xy: X shape %s, y shape %s, positive rate %.1f%%",
              X.shape, y.shape, 100 * y.mean())
    return X, y


def summarise_split(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> None:
    """Print a formatted summary table of the three splits."""
    log.info("=" * 60)
    log.info("TIME-BASED SPLIT SUMMARY")
    log.info("=" * 60)
    log.info("%-12s %8s %10s %12s %12s", "Split", "Rows", "Years", "Storms", "Storm rate")
    log.info("-" * 60)

    for name, split_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        if len(split_df) == 0:
            log.info("%-12s %8s", name, "EMPTY")
            continue
        years = f"{split_df['year'].min()}–{split_df['year'].max()}"
        n_storms = int(split_df[LABEL_COL].sum()) if LABEL_COL in split_df.columns else -1
        rate = 100 * n_storms / len(split_df) if n_storms >= 0 else float("nan")
        log.info("%-12s %8d %10s %12d %11.1f%%", name, len(split_df), years, n_storms, rate)

    log.info("=" * 60)