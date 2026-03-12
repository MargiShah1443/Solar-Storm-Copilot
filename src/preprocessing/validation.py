"""
src/preprocessing/validation.py
---------------------------------
Sanity checks to run after preprocessing, before model training.
Catches common mistakes (leakage, wrong shapes, NaNs slipping through)
early rather than getting mysterious model failures later.

Public API:
    validate_splits(X_train, X_val, X_test, y_train, y_val, y_test)
    check_no_leakage(X_train, X_val, X_test)
    check_no_nulls(X, name)
    print_preprocessing_report(X_train, X_val, X_test, y_train, y_val, y_test)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def validate_splits(
    X_train: np.ndarray | pd.DataFrame,
    X_val:   np.ndarray | pd.DataFrame,
    X_test:  np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    y_val:   np.ndarray | pd.Series,
    y_test:  np.ndarray | pd.Series,
) -> bool:
    """
    Run all validation checks. Returns True if all pass, raises on failure.
    """
    passed = True

    # Shape consistency
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], (
        f"Feature count mismatch: train={X_train.shape[1]}, "
        f"val={X_val.shape[1]}, test={X_test.shape[1]}"
    )
    assert len(X_train) == len(y_train), "X_train and y_train row count mismatch"
    assert len(X_val)   == len(y_val),   "X_val and y_val row count mismatch"
    assert len(X_test)  == len(y_test),  "X_test and y_test row count mismatch"

    # No NaNs after preprocessing (imputer should have handled them all)
    for name, X in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        passed &= check_no_nulls(X, name)

    # No infinite values
    for name, X in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        n_inf = np.isinf(X_arr).sum()
        if n_inf > 0:
            log.error("  ✗ %s contains %d infinite values!", name, n_inf)
            passed = False
        else:
            log.info("  ✓ %s — no infinite values", name)

    # Label is binary
    for name, y in [("y_train", y_train), ("y_val", y_val), ("y_test", y_test)]:
        unique = np.unique(y)
        if not set(unique).issubset({0, 1}):
            log.error("  ✗ %s has non-binary values: %s", name, unique)
            passed = False

    # Class balance warning
    for name, y in [("y_train", y_train), ("y_val", y_val), ("y_test", y_test)]:
        rate = np.mean(y)
        level = "WARNING" if rate < 0.1 or rate > 0.9 else "INFO"
        getattr(log, level.lower())(
            "  Storm rate in %-10s %.1f%%  (n=%d)", name, 100 * rate, len(y)
        )

    if passed:
        log.info("All preprocessing validation checks PASSED ✓")
    else:
        log.error("One or more preprocessing validation checks FAILED ✗")

    return passed


def check_no_leakage(
    X_train: pd.DataFrame,
    X_val:   pd.DataFrame,
    X_test:  pd.DataFrame,
) -> None:
    """
    Warn if any obviously leaky column names are present in feature matrices.
    Leaky columns: kp_max_h72, label_storm, or any raw 'SYM_H' without ablation.
    """
    leaky_patterns = ["kp_max_h72", "label_storm", "kp_"]

    for name, X in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        if not isinstance(X, pd.DataFrame):
            continue
        for col in X.columns:
            for pat in leaky_patterns:
                if pat in col:
                    log.error(
                        "LEAKAGE RISK: column '%s' in %s matches pattern '%s'. "
                        "Remove it before training!", col, name, pat
                    )


def check_no_nulls(X: np.ndarray | pd.DataFrame, name: str = "X") -> bool:
    """Check for NaN values and log result. Returns True if clean."""
    if isinstance(X, pd.DataFrame):
        n_null = X.isna().sum().sum()
    else:
        n_null = int(np.isnan(X).sum())

    if n_null > 0:
        log.error("  ✗ %s contains %d NaN values after preprocessing!", name, n_null)
        if isinstance(X, pd.DataFrame):
            per_col = X.isna().sum()
            per_col = per_col[per_col > 0]
            for col, cnt in per_col.items():
                log.error("      %s: %d NaNs", col, cnt)
        return False

    log.info("  ✓ %s — no NaN values", name)
    return True


def print_preprocessing_report(
    X_train: np.ndarray | pd.DataFrame,
    X_val:   np.ndarray | pd.DataFrame,
    X_test:  np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    y_val:   np.ndarray | pd.Series,
    y_test:  np.ndarray | pd.Series,
) -> None:
    """Print a full human-readable preprocessing summary."""
    total = len(y_train) + len(y_val) + len(y_test)

    log.info("")
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║         PREPROCESSING REPORT                        ║")
    log.info("╠══════════════════════════════════════════════════════╣")
    log.info("║  %-20s %8s %8s %8s      ║", "Split", "Rows", "Features", "Storm%")
    log.info("╠══════════════════════════════════════════════════════╣")

    for name, X, y in [
        ("Train", X_train, y_train),
        ("Validation", X_val, y_val),
        ("Test", X_test, y_test),
    ]:
        n_feat = X.shape[1]
        rate = 100 * np.mean(y)
        pct_of_total = 100 * len(y) / total
        log.info(
            "║  %-20s %8d %8d %7.1f%%      ║",
            f"{name} ({pct_of_total:.0f}%)", len(y), n_feat, rate,
        )

    log.info("╚══════════════════════════════════════════════════════╝")
    log.info("")