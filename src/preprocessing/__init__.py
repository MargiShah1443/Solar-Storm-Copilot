"""
src/preprocessing/__init__.py
Preprocessing sub-package: feature engineering, imputation, scaling, splitting.
"""
from src.preprocessing.feature_engineering import engineer_features, get_feature_groups, LABEL_COL
from src.preprocessing.splitter import time_split, extract_Xy, summarise_split
from src.preprocessing.preprocessor import (
    build_preprocessor,
    apply_preprocessor,
    get_feature_names,
    save_preprocessor,
    load_preprocessor,
)

__all__ = [
    "engineer_features",
    "get_feature_groups",
    "LABEL_COL",
    "time_split",
    "extract_Xy",
    "summarise_split",
    "build_preprocessor",
    "apply_preprocessor",
    "get_feature_names",
    "save_preprocessor",
    "load_preprocessor",
]