"""
src/preprocessing/preprocessor.py
-----------------------------------
Fit-on-train-only imputation and scaling pipeline.

CRITICAL DESIGN RULE:
    All sklearn transformers (imputers, scalers) must be FIT only on X_train,
    then APPLIED to val and test.  Fitting on the full dataset before splitting
    is a form of data leakage — the model would have seen test-set statistics
    during training.

Public API:
    build_preprocessor(X_train)          -> fitted sklearn Pipeline
    apply_preprocessor(pipeline, X)      -> transformed numpy array
    get_feature_names(pipeline, X)       -> list of output feature names
    save_preprocessor(pipeline, path)    -> persists to disk
    load_preprocessor(path)              -> loads from disk
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.utils.logging_utils import get_logger

log = get_logger(__name__)


# ------------------------------------------------------------------ #
# Column type detection                                               #
# ------------------------------------------------------------------ #

def _split_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Separate columns into numeric and categorical based on dtype.
    Returns (numeric_cols, categorical_cols).
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return numeric_cols, categorical_cols


# ------------------------------------------------------------------ #
# Public API                                                          #
# ------------------------------------------------------------------ #

def build_preprocessor(X_train: pd.DataFrame) -> Pipeline:
    """
    Build and fit a preprocessing pipeline on X_train.

    Numeric columns:
        - Median imputation (robust to OMNI data gaps)
        - StandardScaler (zero mean, unit variance)

    Categorical columns:
        - Most-frequent imputation
        - OneHotEncoder (handle_unknown='ignore' for unseen values at test time)

    Parameters
    ----------
    X_train : DataFrame
        Training features ONLY — never pass val/test data here.

    Returns
    -------
    Fitted sklearn Pipeline
    """
    numeric_cols, categorical_cols = _split_column_types(X_train)

    log.info(
        "Building preprocessor — %d numeric cols, %d categorical cols",
        len(numeric_cols), len(categorical_cols),
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    transformers = [("num", numeric_pipeline, numeric_cols)]

    if categorical_cols:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",       # silently drop any unlisted columns
        verbose_feature_names_out=False,
    )

    full_pipeline = Pipeline([("preprocessor", preprocessor)])
    full_pipeline.fit(X_train)

    log.info("Preprocessor fitted on %d training rows", len(X_train))
    _log_imputation_summary(X_train, numeric_cols)

    return full_pipeline


def apply_preprocessor(
    pipeline: Pipeline,
    X: pd.DataFrame,
    return_df: bool = False,
) -> np.ndarray | pd.DataFrame:
    """
    Apply a fitted preprocessing pipeline to any split (train/val/test).

    Parameters
    ----------
    pipeline : fitted Pipeline from build_preprocessor()
    X        : features DataFrame (same columns as X_train)
    return_df: if True, return a DataFrame with feature names preserved

    Returns
    -------
    numpy array (default) or DataFrame
    """
    X_transformed = pipeline.transform(X)

    if return_df:
        feature_names = get_feature_names(pipeline, X)
        return pd.DataFrame(X_transformed, columns=feature_names, index=X.index)

    return X_transformed


def get_feature_names(pipeline: Pipeline, X_ref: pd.DataFrame) -> list[str]:
    """
    Extract output feature names from the fitted ColumnTransformer.
    Works for both numeric (passthrough after scaling) and one-hot categorical.
    """
    ct = pipeline.named_steps["preprocessor"]
    try:
        names = ct.get_feature_names_out().tolist()
    except AttributeError:
        # Fallback for older sklearn versions
        numeric_cols, _ = _split_column_types(X_ref)
        names = numeric_cols
    return names


def save_preprocessor(pipeline: Pipeline, path: Path) -> None:
    """Persist fitted pipeline to disk using pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    log.info("Preprocessor saved → %s", path)


def load_preprocessor(path: Path) -> Pipeline:
    """Load a previously saved preprocessing pipeline."""
    with open(Path(path), "rb") as f:
        pipeline = pickle.load(f)
    log.info("Preprocessor loaded ← %s", path)
    return pipeline


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def _log_imputation_summary(X: pd.DataFrame, numeric_cols: list[str]) -> None:
    """Log how many values will be imputed per column (for audit trail)."""
    missing = X[numeric_cols].isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        log.info("  No missing values in numeric columns — no imputation needed.")
    else:
        log.info("  Columns requiring imputation (median strategy):")
        for col, n in missing.items():
            log.info("    %-45s  %d missing (%.1f%%)", col, n, 100 * n / len(X))