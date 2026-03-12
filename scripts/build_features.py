"""
scripts/build_features.py
---------------------------
Step 1 of the modeling pipeline: Feature Engineering & Preprocessing.

Reads  : data/raw/cme_features_labeled.csv
Writes : data/processed/
            X_train.csv, y_train.csv
            X_val.csv,   y_val.csv
            X_test.csv,  y_test.csv
            preprocessor.pkl        ← fitted scaler/imputer (use at inference)
            feature_names.txt       ← ordered list of final feature names
            split_summary.csv       ← row counts and storm rates per split

Usage:
    python -m scripts.build_features
    python -m scripts.build_features --input data/raw/cme_features_labeled.csv
    python -m scripts.build_features --train-end 2020 --val-year 2021 --test-start 2022
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# ── make sure project root is on sys.path when run as __main__ ──────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.feature_engineering import engineer_features, get_feature_groups
from src.preprocessing.splitter import time_split, extract_Xy
from src.preprocessing.preprocessor import (
    build_preprocessor, apply_preprocessor,
    get_feature_names, save_preprocessor,
)
from src.preprocessing.validation import validate_splits, check_no_leakage, print_preprocessing_report
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# ------------------------------------------------------------------ #
# Paths                                                               #
# ------------------------------------------------------------------ #
DEFAULT_INPUT  = PROJECT_ROOT / "data" / "raw" / "cme_features_labeled.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed"


# ------------------------------------------------------------------ #
# Main pipeline                                                       #
# ------------------------------------------------------------------ #

def build_features(
    input_path:  Path = DEFAULT_INPUT,
    output_dir:  Path = DEFAULT_OUTPUT,
    train_end:   int  = 2020,
    val_year:    int  = 2021,
    test_start:  int  = 2022,
    scale:       bool = True,
) -> None:
    """
    Full feature engineering and preprocessing pipeline.
    All outputs are written to output_dir.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ────────────────────────────────────────────────────────
    log.info("=== Step 1/6  Load raw labeled features ===")
    df_raw = pd.read_csv(input_path)
    log.info("Loaded %d rows × %d cols from %s", *df_raw.shape, input_path.name)

    # ── Feature engineering ─────────────────────────────────────────
    log.info("=== Step 2/6  Feature engineering ===")
    df = engineer_features(df_raw)

    # ── Time-based split ────────────────────────────────────────────
    log.info("=== Step 3/6  Time-based train / val / test split ===")
    train_df, val_df, test_df = time_split(
        df, train_end=train_end, val_year=val_year, test_start=test_start
    )

    # ── Extract X, y ────────────────────────────────────────────────
    log.info("=== Step 4/6  Extract features and labels ===")

    # Determine feature columns: exclude label, metadata, year
    exclude = {"label_storm", "year", "startTime", "kp_max_h72"}
    feature_cols = [c for c in train_df.columns if c not in exclude]
    log.info("Using %d feature columns", len(feature_cols))

    X_train_raw, y_train = extract_Xy(train_df, feature_cols=feature_cols)
    X_val_raw,   y_val   = extract_Xy(val_df,   feature_cols=feature_cols)
    X_test_raw,  y_test  = extract_Xy(test_df,  feature_cols=feature_cols)

    # Leakage check before scaling
    check_no_leakage(X_train_raw, X_val_raw, X_test_raw)

    # ── Fit preprocessor (impute + scale) ───────────────────────────
    log.info("=== Step 5/6  Fit preprocessor on X_train only ===")
    if scale:
        preprocessor = build_preprocessor(X_train_raw)
        X_train = apply_preprocessor(preprocessor, X_train_raw, return_df=True)
        X_val   = apply_preprocessor(preprocessor, X_val_raw,   return_df=True)
        X_test  = apply_preprocessor(preprocessor, X_test_raw,  return_df=True)
        save_preprocessor(preprocessor, output_dir / "preprocessor.pkl")
    else:
        log.info("  Scaling skipped (scale=False). Only imputation step applied.")
        # For tree models, scaling is not needed.  We still need imputation.
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy="median")
        X_train = pd.DataFrame(
            imputer.fit_transform(X_train_raw),
            columns=X_train_raw.columns, index=X_train_raw.index
        )
        X_val = pd.DataFrame(
            imputer.transform(X_val_raw),
            columns=X_val_raw.columns, index=X_val_raw.index
        )
        X_test = pd.DataFrame(
            imputer.transform(X_test_raw),
            columns=X_test_raw.columns, index=X_test_raw.index
        )

    # ── Validate ────────────────────────────────────────────────────
    log.info("=== Step 6/6  Validate and write outputs ===")
    validate_splits(X_train, X_val, X_test, y_train, y_val, y_test)
    print_preprocessing_report(X_train, X_val, X_test, y_train, y_val, y_test)

    # ── Write outputs ────────────────────────────────────────────────
    _write_outputs(
        output_dir,
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        feature_cols,
        train_df, val_df, test_df,
    )

    log.info("=== build_features complete.  Outputs in %s ===", output_dir)
    log.info("Next step: python -m scripts.train_model")


def _write_outputs(
    out: Path,
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    feature_cols,
    train_df, val_df, test_df,
) -> None:
    """Write all processed splits and metadata to disk."""

    # Feature matrices
    for name, X in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        path = out / f"{name}.csv"
        X.to_csv(path, index=False)
        log.info("  Wrote %s  (%d × %d)", path.name, *X.shape)

    # Labels
    for name, y in [("y_train", y_train), ("y_val", y_val), ("y_test", y_test)]:
        path = out / f"{name}.csv"
        y.to_csv(path, index=False, header=True)
        log.info("  Wrote %s  (%d rows, %.1f%% storms)", path.name, len(y), 100 * y.mean())

    # Feature names
    feat_path = out / "feature_names.txt"
    feat_path.write_text("\n".join(X_train.columns.tolist()))
    log.info("  Wrote feature_names.txt (%d features)", len(X_train.columns))

    # Split summary CSV for downstream reporting
    rows = []
    for name, X, y, df_split in [
        ("train",      X_train, y_train, train_df),
        ("validation", X_val,   y_val,   val_df),
        ("test",       X_test,  y_test,  test_df),
    ]:
        rows.append({
            "split":      name,
            "n_rows":     len(y),
            "n_features": X.shape[1],
            "n_storms":   int(y.sum()),
            "storm_rate": round(float(y.mean()), 4),
            "year_min":   int(df_split["year"].min()) if "year" in df_split.columns else None,
            "year_max":   int(df_split["year"].max()) if "year" in df_split.columns else None,
        })
    summary_df = pd.DataFrame(rows)
    summary_path = out / "split_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info("  Wrote split_summary.csv")


# ------------------------------------------------------------------ #
# CLI                                                                 #
# ------------------------------------------------------------------ #

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature engineering and preprocessing pipeline."
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="Path to cme_features_labeled.csv",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Directory to write processed splits",
    )
    parser.add_argument(
        "--train-end", type=int, default=2020,
        help="Last year (inclusive) in training set",
    )
    parser.add_argument(
        "--val-year", type=int, default=2021,
        help="Year used as validation set",
    )
    parser.add_argument(
        "--test-start", type=int, default=2022,
        help="First year (inclusive) in test set",
    )
    parser.add_argument(
        "--no-scale", action="store_true",
        help="Skip StandardScaler (use for tree-based models only)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_features(
        input_path  = args.input,
        output_dir  = args.output,
        train_end   = args.train_end,
        val_year    = args.val_year,
        test_start  = args.test_start,
        scale       = not args.no_scale,
    )