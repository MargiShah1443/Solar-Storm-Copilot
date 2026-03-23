"""
scripts/run_baseline.py
------------------------
Step 3: Physics-threshold baseline.

Predicts storm = 1 if BOTH conditions hold:
  - min_Bz_gsm  < -8 nT   (sustained southward magnetic field)
  - mean_V      > 450 km/s (elevated solar wind speed)

This encodes the primary physical mechanism driving geomagnetic storms
and sets the performance floor every ML model must exceed.

Evaluates on the VALIDATION set only.
Test set is never touched until final evaluation in evaluate.py.

Usage:
    python -m scripts.run_baseline

Outputs (written to results/):
    baseline_val_report.txt   — full classification report
    baseline_val_scores.csv   — single-row metrics summary
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── project root on path ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_utils import get_logger
log = get_logger(__name__)

# ── paths ───────────────────────────────────────────────────────────
PROCESSED   = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

# ── threshold constants ─────────────────────────────────────────────
BZ_THRESHOLD = -8.0    # nT  — southward Bz threshold
V_THRESHOLD  = 450.0   # km/s — solar wind speed threshold


def run_baseline() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load validation split ────────────────────────────────────────
    log.info("Loading validation split from %s", PROCESSED)
    X_val = pd.read_csv(PROCESSED / "X_val.csv")
    y_val = pd.read_csv(PROCESSED / "y_val.csv").squeeze()

    log.info("  X_val shape : %s", X_val.shape)
    log.info("  y_val storms: %d / %d  (%.1f%%)",
             y_val.sum(), len(y_val), 100 * y_val.mean())

    # ── Apply physics threshold rule ────────────────────────────────
    bz_col = "omni_w24_72_min_Bz_gsm"
    v_col  = "omni_w24_72_mean_V"

    for col in [bz_col, v_col]:
        if col not in X_val.columns:
            raise KeyError(f"Required column '{col}' not found in X_val.csv")

    bz_cond = X_val[bz_col] < BZ_THRESHOLD
    v_cond  = X_val[v_col]  > V_THRESHOLD
    y_pred  = (bz_cond & v_cond).astype(int)

    log.info("Threshold rule: %s < %.1f nT  AND  %s > %.1f km/s",
             bz_col, BZ_THRESHOLD, v_col, V_THRESHOLD)
    log.info("  Predicted storms : %d  (%.1f%% of val set)",
             y_pred.sum(), 100 * y_pred.mean())

    # ── Metrics ─────────────────────────────────────────────────────
    report = classification_report(
        y_val, y_pred,
        target_names=["No Storm (0)", "Storm (1)"],
        digits=4,
    )

    # AUC-ROC uses the binary prediction as score (no probabilities)
    auc    = roc_auc_score(y_val, y_pred)
    f1     = f1_score(y_val, y_pred, zero_division=0)
    prec   = precision_score(y_val, y_pred, zero_division=0)
    rec    = recall_score(y_val, y_pred, zero_division=0)
    cm     = confusion_matrix(y_val, y_pred)

    tn, fp, fn, tp = cm.ravel()

    log.info("\n%s", "=" * 55)
    log.info("BASELINE RESULTS — Validation Set")
    log.info("=" * 55)
    log.info("\n%s", report)
    log.info("AUC-ROC   : %.4f", auc)
    log.info("F1 (storm): %.4f", f1)
    log.info("Precision : %.4f", prec)
    log.info("Recall    : %.4f", rec)
    log.info("Confusion matrix:")
    log.info("  TN=%d  FP=%d", tn, fp)
    log.info("  FN=%d  TP=%d", fn, tp)
    log.info("=" * 55)
    log.info("NOTE: Every ML model trained in train_model.py must")
    log.info("      exceed F1=%.4f and AUC=%.4f to be useful.", f1, auc)
    log.info("=" * 55)

    # ── Save text report ────────────────────────────────────────────
    report_path = RESULTS_DIR / "baseline_val_report.txt"
    with open(report_path, "w") as f:
        f.write("PHYSICS BASELINE — Validation Set\n")
        f.write(f"Rule: {bz_col} < {BZ_THRESHOLD} nT  AND  "
                f"{v_col} > {V_THRESHOLD} km/s\n")
        f.write("=" * 55 + "\n")
        f.write(report)
        f.write(f"\nAUC-ROC   : {auc:.4f}\n")
        f.write(f"F1 (storm): {f1:.4f}\n")
        f.write(f"Precision : {prec:.4f}\n")
        f.write(f"Recall    : {rec:.4f}\n")
        f.write(f"\nConfusion matrix:\n")
        f.write(f"  TN={tn}  FP={fp}\n  FN={fn}  TP={tp}\n")
    log.info("Saved report → %s", report_path)

    # ── Save metrics CSV ────────────────────────────────────────────
    scores_df = pd.DataFrame([{
        "model"    : "physics_baseline",
        "split"    : "val",
        "f1_storm" : round(f1,   4),
        "precision": round(prec, 4),
        "recall"   : round(rec,  4),
        "auc_roc"  : round(auc,  4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }])
    scores_path = RESULTS_DIR / "baseline_val_scores.csv"
    scores_df.to_csv(scores_path, index=False)
    log.info("Saved scores → %s", scores_path)

    # ── Confusion matrix plot ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Storm", "Storm"]
    )
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion Matrix — Physics Baseline (Val)", fontsize=11)

    # Right: precision-recall
    pr_disp = PrecisionRecallDisplay.from_predictions(
        y_val, y_pred,
        name="Physics baseline",
        ax=axes[1],
    )
    axes[1].set_title("Precision-Recall — Physics Baseline (Val)", fontsize=11)
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    fig_path = FIGURES_DIR / "baseline_val.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved figure → %s", fig_path)

    log.info("Baseline complete. Next: python -m scripts.train_model")


if __name__ == "__main__":
    run_baseline()