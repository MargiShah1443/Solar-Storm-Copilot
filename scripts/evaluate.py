"""
scripts/evaluate.py
--------------------
Step 6: Final evaluation on the held-out test set.

IMPORTANT: This script touches the test set EXACTLY ONCE.
           Do not run this script until all model selection,
           hyperparameter tuning, and ablation studies are complete.

Evaluates the best model (XGBoost) on test data and also runs
an ablation study comparing:
  - XGBoost WITH SYM_H and AE features  (current model)
  - XGBoost WITHOUT SYM_H and AE        (leakage-free model)

This ablation is critical because SHAP revealed that SYM_H (rank 2)
and AE (rank 1) are geomagnetic disturbance indices that may partially
measure the storm rather than predict it from preconditions alone.

Usage:
    python -m scripts.evaluate

Outputs:
    results/test_scores.csv                  — final metrics on test set
    results/ablation_scores.csv              — with vs without SYM_H/AE
    outputs/figures/test_confusion_matrix.png
    outputs/figures/test_roc_pr_curves.png
    outputs/figures/ablation_comparison.png
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

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
)
from xgboost import XGBClassifier

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

# ── Potentially leaky features identified by SHAP ───────────────────
# SYM_H: real-time ring current index (measures ongoing storm)
# AE: auroral electrojet index (measures ongoing magnetospheric activity)
LEAKY_FEATURES = [
    "omni_w24_72_mean_SYM_H",
    "omni_w24_72_min_SYM_H",
    "omni_w24_72_max_SYM_H",
    "omni_w24_72_std_SYM_H",
    "omni_w24_72_mean_AE",
    "omni_w24_72_min_AE",
    "omni_w24_72_max_AE",
    "omni_w24_72_std_AE",
]

SCALE_POS_WEIGHT = 3.08


# ── Helpers ──────────────────────────────────────────────────────────

def load_all_splits():
    log.info("Loading all splits from %s", PROCESSED)
    X_train = pd.read_csv(PROCESSED / "X_train.csv")
    y_train = pd.read_csv(PROCESSED / "y_train.csv").squeeze()
    X_val   = pd.read_csv(PROCESSED / "X_val.csv")
    y_val   = pd.read_csv(PROCESSED / "y_val.csv").squeeze()
    X_test  = pd.read_csv(PROCESSED / "X_test.csv")
    y_test  = pd.read_csv(PROCESSED / "y_test.csv").squeeze()
    log.info("  Train : %s  storms=%.1f%%", X_train.shape, 100*y_train.mean())
    log.info("  Val   : %s  storms=%.1f%%", X_val.shape,   100*y_val.mean())
    log.info("  Test  : %s  storms=%.1f%%", X_test.shape,  100*y_test.mean())
    return X_train, y_train, X_val, y_val, X_test, y_test


def compute_metrics(name: str, y_true, y_pred, y_prob, split: str = "test") -> dict:
    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)
    ap   = average_precision_score(y_true, y_prob)
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    report = classification_report(
        y_true, y_pred,
        target_names=["No Storm (0)", "Storm (1)"],
        digits=4,
    )
    log.info("")
    log.info("─" * 58)
    log.info("  %s  [%s]", name.upper(), split.upper())
    log.info("─" * 58)
    log.info("\n%s", report)
    log.info("  AUC-ROC       : %.4f", auc)
    log.info("  Avg Precision : %.4f", ap)
    log.info("  F1  (storm)   : %.4f", f1)
    log.info("  Precision     : %.4f", prec)
    log.info("  Recall        : %.4f", rec)
    log.info("  TP=%d  FP=%d  TN=%d  FN=%d", tp, fp, tn, fn)
    return {
        "model": name, "split": split,
        "f1_storm": round(f1, 4), "precision": round(prec, 4),
        "recall": round(rec, 4),  "auc_roc": round(auc, 4),
        "avg_prec": round(ap, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "_y_prob": y_prob, "_y_pred": y_pred,
    }


def train_xgboost(X_train, y_train, tag: str = "") -> XGBClassifier:
    log.info("  Training XGBoost%s on %d rows, %d features...",
             f" ({tag})" if tag else "", len(X_train), X_train.shape[1])
    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=SCALE_POS_WEIGHT,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


# ── Plots ─────────────────────────────────────────────────────────────

def plot_test_confusion(result: dict, y_test) -> None:
    cm = confusion_matrix(y_test, result["_y_pred"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Storm", "Storm"]
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(
        f"Confusion Matrix — XGBoost on Test Set\n"
        f"F1={result['f1_storm']:.4f}  "
        f"AUC={result['auc_roc']:.4f}  "
        f"Recall={result['recall']:.4f}",
        fontsize=10
    )
    plt.tight_layout()
    path = FIGURES_DIR / "test_confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved → %s", path)


def plot_test_roc_pr(result: dict, y_test) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    RocCurveDisplay.from_predictions(
        y_test, result["_y_prob"],
        name=f"XGBoost  (AUC={result['auc_roc']:.3f})",
        ax=axes[0], color="#2196F3",
    )
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.500)")
    axes[0].set_title("ROC Curve — Test Set", fontsize=12)
    axes[0].legend(fontsize=9)

    PrecisionRecallDisplay.from_predictions(
        y_test, result["_y_prob"],
        name=f"XGBoost  (AP={result['avg_prec']:.3f})",
        ax=axes[1], color="#2196F3",
    )
    storm_rate = y_test.mean()
    axes[1].axhline(storm_rate, color="k", linestyle="--", linewidth=1,
                    label=f"No-skill ({storm_rate:.3f})")
    axes[1].set_title("Precision-Recall Curve — Test Set", fontsize=12)
    axes[1].legend(fontsize=9)

    plt.suptitle("XGBoost Final Evaluation — Test Set (2022–2023)", fontsize=12)
    plt.tight_layout()
    path = FIGURES_DIR / "test_roc_pr_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved → %s", path)


def plot_ablation(full_res: dict, clean_res: dict, y_test) -> None:
    """Bar chart comparing full model vs leakage-free model."""
    metrics  = ["f1_storm", "precision", "recall", "auc_roc"]
    labels   = ["F1 (Storm)", "Precision", "Recall", "AUC-ROC"]
    full_vals  = [full_res[m]  for m in metrics]
    clean_vals = [clean_res[m] for m in metrics]

    x   = np.arange(len(labels))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, full_vals,  w, label="With SYM_H & AE  (possible leakage)",
                color="#2196F3", alpha=0.85)
    b2 = ax.bar(x + w/2, clean_vals, w, label="Without SYM_H & AE  (leakage-free)",
                color="#FF5722", alpha=0.85)

    ax.bar_label(b1, fmt="%.3f", fontsize=9, padding=3)
    ax.bar_label(b2, fmt="%.3f", fontsize=9, padding=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("SYM_H / AE Ablation Study — Test Set\n"
                 "Do geomagnetic indices leak the label?", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = FIGURES_DIR / "ablation_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved → %s", path)


# ── Main ──────────────────────────────────────────────────────────────

def evaluate() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = load_all_splits()

    # ── Load saved XGBoost model ─────────────────────────────────────
    log.info("")
    log.info("=== PART 1: Final test evaluation (full feature set) ===")
    with open(MODELS_DIR / "xgboost.pkl", "rb") as f:
        model_full = pickle.load(f)

    y_pred_full = model_full.predict(X_test)
    y_prob_full = model_full.predict_proba(X_test)[:, 1]
    res_full    = compute_metrics(
        "XGBoost (full features)", y_test, y_pred_full, y_prob_full, split="test"
    )

    plot_test_confusion(res_full, y_test)
    plot_test_roc_pr(res_full, y_test)

    # ── Ablation: retrain without SYM_H and AE ───────────────────────
    log.info("")
    log.info("=== PART 2: Ablation — remove SYM_H and AE features ===")

    leaky_present = [c for c in LEAKY_FEATURES if c in X_train.columns]
    log.info("  Dropping %d features: %s", len(leaky_present), leaky_present)

    X_train_clean = X_train.drop(columns=leaky_present)
    X_val_clean   = X_val.drop(columns=leaky_present)
    X_test_clean  = X_test.drop(columns=leaky_present)

    model_clean = train_xgboost(X_train_clean, y_train, tag="no SYM_H/AE")

    y_pred_clean = model_clean.predict(X_test_clean)
    y_prob_clean = model_clean.predict_proba(X_test_clean)[:, 1]
    res_clean    = compute_metrics(
        "XGBoost (no SYM_H/AE)", y_test, y_pred_clean, y_prob_clean, split="test"
    )

    plot_ablation(res_full, res_clean, y_test)

    # ── Save all scores ──────────────────────────────────────────────
    log.info("")
    log.info("=== FINAL SUMMARY ===")

    # Load val scores for context
    val_scores = pd.read_csv(RESULTS_DIR / "val_scores.csv") \
        if (RESULTS_DIR / "val_scores.csv").exists() else pd.DataFrame()

    test_rows = [
        {k: v for k, v in res_full.items()  if not k.startswith("_")},
        {k: v for k, v in res_clean.items() if not k.startswith("_")},
    ]
    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(RESULTS_DIR / "test_scores.csv", index=False)
    log.info("Saved test_scores.csv → %s", RESULTS_DIR / "test_scores.csv")

    # Print final comparison
    log.info("")
    log.info("=" * 62)
    log.info("FINAL RESULTS SUMMARY")
    log.info("=" * 62)
    log.info("%-32s %7s %7s %7s", "Model", "F1", "AUC", "Recall")
    log.info("-" * 62)

    if not val_scores.empty:
        for _, row in val_scores.iterrows():
            log.info("  [VAL]  %-27s %7.4f %7.4f %7.4f",
                     row["model"], row["f1_storm"], row["auc_roc"], row["recall"])
        log.info("  " + "-" * 58)

    for row in test_rows:
        log.info("  [TEST] %-27s %7.4f %7.4f %7.4f",
                 row["model"], row["f1_storm"], row["auc_roc"], row["recall"])

    log.info("=" * 62)
    log.info("")
    log.info("Key finding: SYM_H/AE ablation shows how much of the")
    log.info("performance depends on real-time geomagnetic indices")
    log.info("vs. purely predictive solar wind features.")
    log.info("")
    log.info("Evaluation complete. Project pipeline finished.")


if __name__ == "__main__":
    evaluate()