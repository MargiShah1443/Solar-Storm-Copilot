"""
src/preprocessing/feature_engineering.py
------------------------------------------
All feature-level transformations applied BEFORE the train/val/test split.
Operates purely on the raw labeled DataFrame and returns an enriched one.

No imputation or scaling happens here — those are fit-on-train-only operations
that live in preprocessor.py to prevent data leakage.

Public API:
    engineer_features(df)  -> DataFrame with new/cleaned columns
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# ------------------------------------------------------------------ #
# Column groups                                                       #
# ------------------------------------------------------------------ #

# These columns carry no predictive signal and must be removed
_DROP_ALWAYS = [
    "analysis_note",        # 100% missing
    "activityID",           # identifier, not a feature
    "link",                 # URL string
    "note",                 # free-text, not parsed
    "sourceLocation",       # raw string, superseded by engineered lat/lon
    "analysis_isMostAccurate",  # ~99% True — zero variance
]

# Efield = V × Bz / 1000 (r=0.99 confirmed) — pure redundancy
_DROP_REDUNDANT = [
    "omni_w24_72_mean_Efield",
    "omni_w24_72_min_Efield",
    "omni_w24_72_max_Efield",
    "omni_w24_72_std_Efield",
]

# Columns used only for labeling / bookkeeping, not as model inputs
_DROP_LEAKAGE = [
    "kp_max_h72",   # direct source of the label — must never be a feature
]

# Target column
LABEL_COL = "label_storm"

# Columns to keep as-is for downstream use but not feed to the model
METADATA_COLS = ["startTime", "year"]


# ------------------------------------------------------------------ #
# Public API                                                          #
# ------------------------------------------------------------------ #

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations to the raw labeled DataFrame.

    Steps (in order):
      1. Parse timestamps and extract year
      2. Drop always-remove and redundant columns
      3. Handle activeRegionNum → binary flag
      4. Encode analysis_type as one-hot
      5. Flag and drop all-OMNI-missing rows
      6. Flag SYM_H leakage risk (audit column, not dropped automatically)
      7. Add interaction features
      8. Clip extreme outliers in CME speed

    Returns a new DataFrame — does not modify the input.
    """
    df = df.copy()
    n_start = len(df)
    log.info("Feature engineering — starting with %d rows, %d cols", n_start, df.shape[1])

    # ── 1. Timestamps ──────────────────────────────────────────────
    df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
    df["year"] = df["startTime"].dt.year
    log.info("  [1/8] Parsed startTime → year extracted")

    # ── 2. Drop always-remove + redundant + leakage columns ────────
    to_drop = [c for c in _DROP_ALWAYS + _DROP_REDUNDANT + _DROP_LEAKAGE
               if c in df.columns]
    df = df.drop(columns=to_drop)
    log.info("  [2/8] Dropped %d columns (%s...)", len(to_drop), to_drop[:3])

    # ── 3. activeRegionNum → binary flag ───────────────────────────
    # 78.7% missing — imputing a solar region number makes no sense.
    # Presence/absence is the real signal (CME from a named AR vs. not).
    if "activeRegionNum" in df.columns:
        df["has_active_region"] = df["activeRegionNum"].notna().astype(int)
        df = df.drop(columns=["activeRegionNum"])
        rate = df["has_active_region"].mean()
        log.info("  [3/8] activeRegionNum → has_active_region flag  (%.1f%% positive)", rate * 100)
    else:
        log.warning("  [3/8] activeRegionNum not found — skipping flag creation")

    # ── 4. Encode analysis_type ────────────────────────────────────
    # Values: S (standard), C (halo/partial-halo), O (other), R (rare)
    # R has only 7 rows — merge into O to avoid sparse dummy columns
    if "analysis_type" in df.columns:
        df["analysis_type"] = df["analysis_type"].fillna("S")
        df["analysis_type"] = df["analysis_type"].replace({"R": "O"})
        type_dummies = pd.get_dummies(
            df["analysis_type"], prefix="cme_type", drop_first=True, dtype=int
        )
        df = pd.concat([df.drop(columns=["analysis_type"]), type_dummies], axis=1)
        log.info("  [4/8] analysis_type one-hot encoded → cols: %s", type_dummies.columns.tolist())

    # ── 5. Drop rows where ALL OMNI features are simultaneously NaN ─
    # These CMEs have zero solar wind context — imputing every single
    # feature for an entire row manufactures data, not fills gaps.
    omni_cols = [c for c in df.columns if c.startswith("omni_")]
    all_omni_missing = df[omni_cols].isna().all(axis=1)
    n_dropped = all_omni_missing.sum()
    df = df[~all_omni_missing].copy()
    log.info(
        "  [5/8] Dropped %d rows with all OMNI features missing  (%d → %d rows)",
        n_dropped, n_start, len(df),
    )

    # ── 6. SYM_H leakage audit column ──────────────────────────────
    # SYM_H measures geomagnetic disturbance in real-time.
    # If the OMNI window overlaps the storm itself (not just preconditions),
    # these features are partially measuring the target, not predicting it.
    # We keep them but add a flag so the modeler can easily ablate them.
    symh_cols = [c for c in df.columns if "SYM_H" in c]
    if symh_cols:
        log.info(
            "  [6/8] SYM_H leakage audit: %d SYM_H features retained. "
            "Run ablation study with/without these cols before final model.",
            len(symh_cols),
        )

    # ── 7. Interaction features ─────────────────────────────────────
    # Physics-motivated cross terms that tree models can learn but
    # linear models benefit from having explicitly.
    df = _add_interaction_features(df)
    log.info("  [7/8] Interaction features added")

    # ── 8. CME speed outlier clip ───────────────────────────────────
    # Max observed: 2650 km/s — extreme outliers can destabilise
    # distance-based models.  Cap at 99th percentile of training data.
    # We clip here (pre-split) using the global 99th pct as a soft cap.
    if "analysis_speed" in df.columns:
        cap = df["analysis_speed"].quantile(0.99)
        n_clipped = (df["analysis_speed"] > cap).sum()
        df["analysis_speed"] = df["analysis_speed"].clip(upper=cap)
        log.info("  [8/8] analysis_speed clipped at 99th pct (%.0f km/s), %d rows affected",
                 cap, n_clipped)

    log.info(
        "Feature engineering complete — %d rows, %d cols  (label='%s')",
        len(df), df.shape[1], LABEL_COL,
    )
    return df


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add physics-motivated interaction and derived features.

    All new columns are prefixed with 'feat_' so they can be identified
    easily in feature importance plots.
    """
    # Southward Bz intensity: how negative the minimum excursion is.
    # Positive = southward dip depth; zero if Bz never went south.
    if "omni_w24_72_min_Bz_gsm" in df.columns:
        df["feat_bz_south_depth"] = (-df["omni_w24_72_min_Bz_gsm"]).clip(lower=0)

    # Solar wind ram energy proxy: V² × Np  (∝ kinetic energy flux)
    if "omni_w24_72_mean_V" in df.columns and "omni_w24_72_mean_Np" in df.columns:
        df["feat_sw_ram_proxy"] = (
            df["omni_w24_72_mean_V"] ** 2 * df["omni_w24_72_mean_Np"]
        ) / 1e6   # scale to ~O(1)

    # Bz variability relative to magnitude: high std + low mean = rotating Bz
    if "omni_w24_72_std_Bz_gsm" in df.columns and "omni_w24_72_mean_Bmag" in df.columns:
        denom = df["omni_w24_72_mean_Bmag"].replace(0, np.nan)
        df["feat_bz_variability_ratio"] = df["omni_w24_72_std_Bz_gsm"] / denom

    # Speed × southward Bz depth = geoeffective coupling proxy
    if "feat_bz_south_depth" in df.columns and "omni_w24_72_mean_V" in df.columns:
        df["feat_coupling"] = (
            df["omni_w24_72_mean_V"] * df["feat_bz_south_depth"]
        ) / 1e4   # scale

    # CME angular width × speed = "energy footprint"
    if "analysis_speed" in df.columns and "analysis_halfAngle" in df.columns:
        df["feat_cme_energy"] = (
            df["analysis_speed"] * df["analysis_halfAngle"]
        ) / 1e4

    return df


def get_feature_groups() -> dict[str, list[str]]:
    """
    Return named groups of features for ablation studies and reporting.
    Call AFTER engineer_features() so the column names exist.
    """
    return {
        "cme_properties": [
            "analysis_speed", "log_speed", "analysis_halfAngle",
            "analysis_latitude", "analysis_longitude", "abs_latitude",
            "is_fast_cme", "has_active_region",
            "cme_type_C", "cme_type_O", "cme_type_S",   # one-hot cols
            "start_hour", "start_dow",
            "feat_cme_energy",
        ],
        "imf_features": [
            "omni_w24_72_mean_Bmag", "omni_w24_72_min_Bmag",
            "omni_w24_72_max_Bmag", "omni_w24_72_std_Bmag",
            "omni_w24_72_mean_By_gsm", "omni_w24_72_min_By_gsm",
            "omni_w24_72_max_By_gsm", "omni_w24_72_std_By_gsm",
            "omni_w24_72_mean_Bz_gsm", "omni_w24_72_min_Bz_gsm",
            "omni_w24_72_max_Bz_gsm", "omni_w24_72_std_Bz_gsm",
            "feat_bz_south_depth", "feat_bz_variability_ratio",
        ],
        "solar_wind_features": [
            "omni_w24_72_mean_V", "omni_w24_72_min_V",
            "omni_w24_72_max_V", "omni_w24_72_std_V",
            "omni_w24_72_mean_Np", "omni_w24_72_min_Np",
            "omni_w24_72_max_Np", "omni_w24_72_std_Np",
            "omni_w24_72_mean_T", "omni_w24_72_min_T",
            "omni_w24_72_max_T", "omni_w24_72_std_T",
            "omni_w24_72_mean_Pdyn", "omni_w24_72_min_Pdyn",
            "omni_w24_72_max_Pdyn", "omni_w24_72_std_Pdyn",
            "omni_w24_72_mean_beta", "omni_w24_72_min_beta",
            "omni_w24_72_max_beta", "omni_w24_72_std_beta",
            "omni_w24_72_mean_MachA", "omni_w24_72_min_MachA",
            "omni_w24_72_max_MachA", "omni_w24_72_std_MachA",
            "feat_sw_ram_proxy", "feat_coupling",
        ],
        "geomagnetic_indices": [
            "omni_w24_72_mean_AE", "omni_w24_72_min_AE",
            "omni_w24_72_max_AE", "omni_w24_72_std_AE",
            "omni_w24_72_mean_SYM_H", "omni_w24_72_min_SYM_H",
            "omni_w24_72_max_SYM_H", "omni_w24_72_std_SYM_H",
        ],
    }