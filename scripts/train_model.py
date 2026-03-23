"""
scripts/train_model.py
-----------------------
Step 4: Train and compare all ML models on the validation set.

Models trained (in order of complexity):
  1. Logistic Regression  — linear baseline, interpretable coefficients
  2. Random Forest         — non-linear ensemble, handles interactions
  3. XGBoost               — gradient boosting, expected best performer

All models use class weighting to handle the 22.4% storm rate imbalance.
All hyperparameter decisions are made using the VALIDATION set only.
The test set is never touched here.

Usage:
    python -m scripts.train_model

Outputs:
    models/logistic_regression.pkl
    models/random_forest.pkl
    models/xgboost.pkl
    results/val_scores.csv          — all models + baseline comparison table
    outputs/figures/val_pr_curves.png
    outputs/figures/val_roc_curves.png
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

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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

# ── Class imbalance ratio (train set: 1010 no-storm / 328 storm) ─────
SCALE_POS_WEIGHT = 3.08   # used by XGBoost


# ── Helpers ──────────────────────────────────────────────────────────

def load_splits():
    """Load all processed splits from disk."""
    log.info("Loading processed splits from %s", PROCESSED)
    X_train = pd.read_csv(PROCESSED / "X_train.csv")
    y_train = pd.read_csv(PROCESSED / "y_train.csv").squeeze()
    X_val   = pd.read_csv(PROCESSED / "X_val.csv")
    y_val   = pd.read_csv(PROCESSED / "y_val.csv").squeeze()
    log.info("  Train: %s  |  storms: %d (%.1f%%)",
             X_train.shape, y_train.sum(), 100 * y_train.mean())
    log.info("  Val  : %s  |  storms: %d (%.1f%%)",
             X_val.shape, y_val.sum(), 100 * y_val.mean())
    return X_train, y_train, X_val, y_val


def evaluate(name: str, model, X_val, y_val) -> dict:
    """Compute all metrics for one model on the validation set."""
    y_pred  = model.predict(X_val)
    y_prob  = model.predict_proba(X_val)[:, 1]

    f1   = f1_score(y_val, y_pred, zero_division=0)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec  = recall_score(y_val, y_pred, zero_division=0)
    auc  = roc_auc_score(y_val, y_prob)
    ap   = average_precision_score(y_val, y_prob)
    cm   = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()

    report = classification_report(
        y_val, y_pred,
        target_names=["No Storm (0)", "Storm (1)"],
        digits=4,
    )

    log.info("")
    log.info("─" * 55)
    log.info("  %s — Validation Results", name.upper())
    log.info("─" * 55)
    log.info("\n%s", report)
    log.info("  AUC-ROC        : %.4f", auc)
    log.info("  Avg Precision  : %.4f", ap)
    log.info("  F1  (storm)    : %.4f", f1)
    log.info("  Precision      : %.4f", prec)
    log.info("  Recall         : %.4f", rec)
    log.info("  TP=%d  FP=%d  TN=%d  FN=%d", tp, fp, tn, fn)

    return {
        "model"    : name,
        "split"    : "val",
        "f1_storm" : round(f1,   4),
        "precision": round(prec, 4),
        "recall"   : round(rec,  4),
        "auc_roc"  : round(auc,  4),
        "avg_prec" : round(ap,   4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "_y_prob"  : y_prob,   # kept for plotting, dropped before CSV save
        "_y_pred"  : y_pred,
    }


def save_model(model, name: str) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    log.info("  Saved model → %s", path)


def plot_curves(results: list[dict], X_val, y_val) -> None:
    """Plot ROC and Precision-Recall curves for all models together."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors = ["#2196F3", "#4CAF50", "#FF5722"]

    for res, color in zip(results, colors):
        name   = res["model"]
        y_prob = res["_y_prob"]

        RocCurveDisplay.from_predictions(
            y_val, y_prob, name=f"{name}  (AUC={res['auc_roc']:.3f})",
            ax=axes[0], color=color,
        )
        PrecisionRecallDisplay.from_predictions(
            y_val, y_prob, name=f"{name}  (AP={res['avg_prec']:.3f})",
            ax=axes[1], color=color,
        )

    # Baseline reference lines
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.500)")
    axes[0].set_title("ROC Curves — Validation Set", fontsize=12)
    axes[0].legend(fontsize=9)

    storm_rate = y_val.mean()
    axes[1].axhline(storm_rate, color="k", linestyle="--", linewidth=1,
                    label=f"No-skill baseline ({storm_rate:.3f})")
    axes[1].set_title("Precision-Recall Curves — Validation Set", fontsize=12)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    path = FIGURES_DIR / "val_roc_pr_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved curves → %s", path)


def plot_confusion_matrices(results: list[dict], X_val, y_val) -> None:
    """Side-by-side confusion matrices for all three models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, res in zip(axes, results):
        cm = confusion_matrix(y_val, res["_y_pred"])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No Storm", "Storm"]
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(
            f"{res['model']}\nF1={res['f1_storm']:.3f}  AUC={res['auc_roc']:.3f}",
            fontsize=10
        )

    plt.suptitle("Confusion Matrices — Validation Set", fontsize=12, y=1.02)
    plt.tight_layout()
    path = FIGURES_DIR / "val_confusion_matrices.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved confusion matrices → %s", path)


def save_scores(results: list[dict]) -> None:
    """Save model comparison table, appending baseline if it exists."""
    # Strip plotting artifacts before saving
    clean = [{k: v for k, v in r.items() if not k.startswith("_")}
             for r in results]
    scores_df = pd.DataFrame(clean)

    # Append baseline row if it was already saved
    baseline_path = RESULTS_DIR / "baseline_val_scores.csv"
    if baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path)
        scores_df = pd.concat([baseline_df, scores_df], ignore_index=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "val_scores.csv"
    scores_df.to_csv(out, index=False)

    log.info("")
    log.info("=" * 55)
    log.info("MODEL COMPARISON — Validation Set")
    log.info("=" * 55)
    log.info("%-28s %8s %8s %8s", "Model", "F1", "AUC-ROC", "Recall")
    log.info("-" * 55)
    for _, row in scores_df.iterrows():
        log.info("%-28s %8.4f %8.4f %8.4f",
                 row["model"], row["f1_storm"], row["auc_roc"], row["recall"])
    log.info("=" * 55)
    log.info("Saved comparison → %s", out)


# ── Model definitions ────────────────────────────────────────────────

def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    log.info("")
    log.info("Training Logistic Regression...")
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        C=0.1,              # L2 regularisation — prevents overfitting on 1338 rows
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train, y_train)
    log.info("  Done. Coefficients shape: %s", model.coef_.shape)
    return model


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    log.info("")
    log.info("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,           # limit depth — avoids overfitting on small train set
        min_samples_leaf=10,   # each leaf needs at least 10 samples
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    log.info("  Done. OOB not computed (oob_score=False for speed).")
    return model


def train_xgboost(X_train, y_train) -> XGBClassifier:
    log.info("")
    log.info("Training XGBoost...")
    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=SCALE_POS_WEIGHT,   # handles class imbalance
        eval_metric="aucpr",                 # optimise area under PR curve
        early_stopping_rounds=None,          # disabled — no val labels leaked
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    log.info("  Done. Best iteration: %s", model.best_iteration
             if hasattr(model, "best_iteration") else "N/A")
    return model


# ── Main ─────────────────────────────────────────────────────────────

def train_all() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val = load_splits()

    # ── Train ────────────────────────────────────────────────────────
    lr  = train_logistic_regression(X_train, y_train)
    rf  = train_random_forest(X_train, y_train)
    xgb = train_xgboost(X_train, y_train)

    # ── Save models ──────────────────────────────────────────────────
    save_model(lr,  "logistic_regression")
    save_model(rf,  "random_forest")
    save_model(xgb, "xgboost")

    # ── Evaluate all on val set ──────────────────────────────────────
    log.info("")
    log.info("Evaluating all models on validation set...")
    results = [
        evaluate("Logistic Regression", lr,  X_val, y_val),
        evaluate("Random Forest",        rf,  X_val, y_val),
        evaluate("XGBoost",              xgb, X_val, y_val),
    ]

    # ── Plots ────────────────────────────────────────────────────────
    plot_curves(results, X_val, y_val)
    plot_confusion_matrices(results, X_val, y_val)

    # ── Save comparison table ────────────────────────────────────────
    save_scores(results)

    # ── Identify best model ──────────────────────────────────────────
    best = max(results, key=lambda r: r["f1_storm"])
    log.info("")
    log.info("Best model on val set: %s  (F1=%.4f, AUC=%.4f)",
             best["model"], best["f1_storm"], best["auc_roc"])
    log.info("")
    log.info("Next step: python -m scripts.run_shap")


if __name__ == "__main__":
    train_all()