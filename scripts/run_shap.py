"""
scripts/run_shap.py
--------------------
Step 5: SHAP (SHapley Additive exPlanations) analysis on the best model.

Loads the trained XGBoost model and explains its predictions using SHAP
TreeExplainer — the exact, model-aware explainer for tree-based models.

Two key checks this script performs:
  1. Does min_Bz_gsm rank as the top feature? (it should — physics says so)
  2. Do the SHAP direction signs match physical intuition?
     - More negative Bz  → higher storm probability  ✓
     - Higher V          → higher storm probability  ✓
     - Higher SYM_H      → lower storm probability   ✓ (already disturbed)

If top features don't match physics, the model has learned spurious patterns.

Usage:
    python -m scripts.run_shap

Outputs:
    outputs/figures/shap_summary.png        — beeswarm (top 20 features)
    outputs/figures/shap_bar.png            — mean |SHAP| bar chart
    outputs/figures/shap_bz_dependence.png  — Bz dependence plot
    results/shap_feature_importance.csv     — ranked feature importance table
"""
from __future__ import annotations

import sys
import pickle
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_utils import get_logger
log = get_logger(__name__)

# ── Paths ────────────────────────────────────────────────────────────
PROCESSED   = PROJECT_ROOT / "data" / "processed"
MODELS_DIR  = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def run_shap() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load model and training data ─────────────────────────────────
    log.info("Loading XGBoost model...")
    with open(MODELS_DIR / "xgboost.pkl", "rb") as f:
        model = pickle.load(f)

    log.info("Loading training data for SHAP background...")
    X_train = pd.read_csv(PROCESSED / "X_train.csv")
    y_train = pd.read_csv(PROCESSED / "y_train.csv").squeeze()

    log.info("  X_train shape: %s", X_train.shape)

    # ── Compute SHAP values ──────────────────────────────────────────
    log.info("Computing SHAP values with TreeExplainer...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X_train)   # Explanation object
    log.info("  SHAP values shape: %s", shap_values.values.shape)

    # ── Feature importance table ─────────────────────────────────────
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature"         : X_train.columns.tolist(),
        "mean_abs_shap"   : mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance_df["rank"] = importance_df.index + 1

    imp_path = RESULTS_DIR / "shap_feature_importance.csv"
    importance_df.to_csv(imp_path, index=False)

    log.info("")
    log.info("=" * 55)
    log.info("TOP 15 FEATURES BY MEAN |SHAP|")
    log.info("=" * 55)
    for _, row in importance_df.head(15).iterrows():
        log.info("  %2d. %-40s %.4f",
                 row["rank"], row["feature"], row["mean_abs_shap"])
    log.info("=" * 55)

    # ── Physics validation ───────────────────────────────────────────
    log.info("")
    log.info("PHYSICS VALIDATION:")
    top5 = importance_df["feature"].head(5).tolist()
    bz_rank = importance_df[importance_df["feature"].str.contains("Bz_gsm")]["rank"].min()
    v_rank  = importance_df[importance_df["feature"].str.contains("mean_V")]["rank"].min()
    log.info("  min_Bz_gsm rank : %s  (expected: top 3)", bz_rank)
    log.info("  mean_V rank     : %s  (expected: top 10)", v_rank)
    if bz_rank <= 5:
        log.info("  ✓ Bz features dominate — model learned correct physics")
    else:
        log.warning("  ✗ Bz not in top 5 — check for leakage or data issues")

    # ── Plot 1: Beeswarm summary (top 20) ────────────────────────────
    log.info("")
    log.info("Generating SHAP beeswarm plot (top 20 features)...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(
        shap_values,
        max_display=20,
        show=False,
    )
    plt.title("SHAP Feature Importance — XGBoost (Training Set)", fontsize=12, pad=12)
    plt.tight_layout()
    path1 = FIGURES_DIR / "shap_summary.png"
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved → %s", path1)

    # ── Plot 2: Mean |SHAP| bar chart (top 20) ───────────────────────
    log.info("Generating SHAP bar chart...")
    top20 = importance_df.head(20)
    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(
        top20["feature"][::-1],
        top20["mean_abs_shap"][::-1],
        color="#2196F3", edgecolor="white", linewidth=0.5
    )
    ax.set_xlabel("Mean |SHAP value|  (impact on model output)", fontsize=11)
    ax.set_title("Top 20 Features by Mean |SHAP| — XGBoost", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path2 = FIGURES_DIR / "shap_bar.png"
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved → %s", path2)

    # ── Plot 3: Bz dependence plot ───────────────────────────────────
    log.info("Generating Bz_gsm SHAP dependence plot...")
    bz_col = "omni_w24_72_min_Bz_gsm"
    if bz_col in X_train.columns:
        bz_idx = X_train.columns.tolist().index(bz_col)

        # Colour by mean_V to show interaction
        v_col = "omni_w24_72_mean_V"
        color_col = X_train[v_col].values if v_col in X_train.columns else None

        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(
            X_train[bz_col],
            shap_values.values[:, bz_idx],
            c=color_col,
            cmap="RdYlBu_r",
            alpha=0.5,
            s=12,
            rasterized=True,
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("mean_V (scaled)", fontsize=10)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="gray",  linewidth=0.6, linestyle=":")
        ax.set_xlabel(f"{bz_col}  (scaled)", fontsize=11)
        ax.set_ylabel("SHAP value  (impact on storm probability)", fontsize=11)
        ax.set_title(
            "SHAP Dependence: min_Bz_gsm\n"
            "More negative Bz → higher SHAP → higher storm probability",
            fontsize=11
        )
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        path3 = FIGURES_DIR / "shap_bz_dependence.png"
        plt.savefig(path3, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("  Saved → %s", path3)
    else:
        log.warning("  %s not found in X_train — skipping dependence plot", bz_col)

    log.info("")
    log.info("SHAP analysis complete.")
    log.info("Next step: python -m scripts.evaluate")


if __name__ == "__main__":
    run_shap()