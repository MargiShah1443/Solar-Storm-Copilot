"""
src/eda/plots.py
----------------
All EDA visualisation functions.  Each function is self-contained:
it accepts a DataFrame, saves a figure to outpath, and closes cleanly.

Existing plots (from original codebase):
  plot_cme_speed_hist
  plot_corr_heatmap
  plot_kp_outcome_by_label
  plot_kp_timeline
  plot_missingness_top

New plots added:
  plot_class_imbalance       — storm vs no-storm bar chart
  plot_storm_rate_by_speed   — storm rate across CME speed quartiles
  plot_omni_by_label         — box plots of key OMNI features split by label
  plot_storm_rate_by_year    — yearly storm rate bar chart
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# ------------------------------------------------------------------ #
# Existing plots                                                      #
# ------------------------------------------------------------------ #

def plot_cme_speed_hist(df: pd.DataFrame, outpath: Path, dpi: int = 140) -> None:
    """Histogram of CME analysis_speed."""
    if "analysis_speed" not in df.columns:
        log.warning("plot_cme_speed_hist: 'analysis_speed' not found — skipping.")
        return
    speeds = pd.to_numeric(df["analysis_speed"], errors="coerce").dropna()
    if speeds.empty:
        log.warning("plot_cme_speed_hist: no speed data — skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(speeds, bins=40, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("CME speed (km/s)")
    ax.set_ylabel("Count")
    ax.set_title("CME speed distribution")
    ax.axvline(1000, color="tomato", linestyle="--", linewidth=1.2, label="1000 km/s threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    log.info("Saved %s", outpath)


def plot_corr_heatmap(df: pd.DataFrame, outpath: Path, dpi: int = 140) -> None:
    """Correlation heatmap of all numeric features."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        log.warning("plot_corr_heatmap: not enough numeric columns — skipping.")
        return

    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(corr.columns, fontsize=6)
    ax.set_title("Correlation heatmap (numeric features)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    log.info("Saved %s", outpath)


def plot_kp_outcome_by_label(df: pd.DataFrame, outpath: Path, dpi: int = 140) -> None:
    """Overlapping histograms of max Kp for storm vs no-storm CMEs."""
    if "kp_max_h72" not in df.columns or "label_storm" not in df.columns:
        log.warning("plot_kp_outcome_by_label: required columns missing — skipping.")
        return

    storm = df[df["label_storm"] == 1]["kp_max_h72"].dropna()
    no_storm = df[df["label_storm"] == 0]["kp_max_h72"].dropna()
    if storm.empty and no_storm.empty:
        log.warning("plot_kp_outcome_by_label: no data — skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(no_storm, bins=30, alpha=0.7, color="steelblue", label=f"No storm (n={len(no_storm)})")
    ax.hist(storm, bins=30, alpha=0.7, color="darkorange", label=f"Storm (n={len(storm)})")
    ax.axvline(5.0, color="black", linestyle="--", linewidth=1, label="Threshold Kp=5")
    ax.set_xlabel("Max Kp in next 72 h")
    ax.set_ylabel("Count")
    ax.set_title("Kp outcome distribution by label")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    log.info("Saved %s", outpath)


def plot_kp_timeline(df_kp: pd.DataFrame, outpath: Path, dpi: int = 140) -> None:
    """Line plot of Kp index over time."""
    if "time" not in df_kp.columns or "kp" not in df_kp.columns:
        log.warning("plot_kp_timeline: 'time' or 'kp' column missing — skipping.")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_kp["time"], df_kp["kp"], linewidth=0.6, color="steelblue")
    ax.axhline(5.0, color="tomato", linestyle="--", linewidth=1, label="Storm threshold (Kp=5)")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Kp (0–9)")
    ax.set_title("Kp index timeline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    log.info("Saved %s", outpath)


def plot_missingness_top(
    df: pd.DataFrame, outpath: Path, top_n: int = 25, dpi: int = 140
) -> None:
    """Bar chart of the top-N columns by missing fraction."""
    miss_frac = df.isna().mean().sort_values(ascending=False).head(top_n)
    if miss_frac.empty:
        log.warning("plot_missingness_top: no missingness data — skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    miss_frac.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_ylabel("Missing fraction")
    ax.set_title(f"Top {top_n} columns by missingness")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=75, fontsize=8)
    ax.axhline(0.5, color="tomato", linestyle="--", linewidth=1, label="50% threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    log.info("Saved %s", outpath)


# ------------------------------------------------------------------ #
# New plots                                                           #
# ------------------------------------------------------------------ #

def plot_class_imbalance(df: pd.DataFrame, outpath: Path, dpi: int = 140) -> None:
    """
    Bar chart showing storm vs no-storm counts and the imbalance ratio.
    Important for choosing appropriate evaluation metrics and resampling
    strategies during model training.
    """
    if "label_storm" not in df.columns:
        log.warning("plot_class_imbalance: 'label_storm' not found — skipping.")
        return

    counts = df["label_storm"].value_counts().sort_index()
    labels = ["No storm (0)", "Storm (1)"]
    colors = ["steelblue", "darkorange"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", width=0.5)

    for bar, v in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f"{v:,}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    total = counts.sum()
    ratio = counts.iloc[1] / total if len(counts) > 1 else float("nan")
    ax.set_title(f"Class distribution  (storm rate = {ratio:.1%})")
    ax.set_ylabel("Count")
    ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    log.info("Saved %s", outpath)


def plot_storm_rate_by_speed(
    df: pd.DataFrame,
    outpath: Path,
    n_bins: int = 5,
    dpi: int = 140,
) -> None:
    """
    Bar chart of storm rate (%) across CME speed quintiles.

    Demonstrates whether fast CMEs are more likely to cause storms,
    providing physical motivation for including analysis_speed as a
    model feature.
    """
    if "analysis_speed" not in df.columns or "label_storm" not in df.columns:
        log.warning("plot_storm_rate_by_speed: required columns missing — skipping.")
        return

    tmp = df[["analysis_speed", "label_storm"]].copy()
    tmp["analysis_speed"] = pd.to_numeric(tmp["analysis_speed"], errors="coerce")
    tmp = tmp.dropna()
    if tmp.empty:
        log.warning("plot_storm_rate_by_speed: no data after dropna — skipping.")
        return

    tmp["speed_bin"] = pd.qcut(tmp["analysis_speed"], q=n_bins, duplicates="drop")
    grouped = tmp.groupby("speed_bin", observed=True)["label_storm"]
    storm_rate = (grouped.mean() * 100).reset_index()
    storm_rate.columns = ["speed_bin", "storm_rate_pct"]
    bin_labels = [str(b) for b in storm_rate["speed_bin"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        range(len(storm_rate)),
        storm_rate["storm_rate_pct"],
        color="steelblue",
        edgecolor="white",
    )
    ax.set_xticks(range(len(storm_rate)))
    ax.set_xticklabels(bin_labels, rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("CME speed bin (km/s)")
    ax.set_ylabel("Storm rate (%)")
    ax.set_title(f"Storm rate by CME speed ({n_bins} equal-frequency bins)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    log.info("Saved %s", outpath)


def plot_omni_by_label(
    df: pd.DataFrame,
    outpath: Path,
    features: Optional[list[str]] = None,
    dpi: int = 140,
) -> None:
    """
    Side-by-side box plots of key OMNI window features split by label_storm.

    Physically, southward Bz (negative) and high solar wind speed are the
    primary drivers of geomagnetic storms, so we expect clear separation
    in these distributions.

    Only plots features that exist in df and have at least some non-NaN data.
    """
    if "label_storm" not in df.columns:
        log.warning("plot_omni_by_label: 'label_storm' not found — skipping.")
        return

    if features is None:
        features = [
            "omni_w24_72_mean_Bz_gsm",
            "omni_w24_72_min_Bz_gsm",
            "omni_w24_72_mean_V",
            "omni_w24_72_mean_Pdyn",
            "omni_w24_72_mean_Np",
            "omni_w24_72_mean_SYM_H",
        ]

    available = [f for f in features if f in df.columns and df[f].notna().any()]
    if not available:
        log.warning("plot_omni_by_label: none of the requested OMNI features available — skipping.")
        return

    n = len(available)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for i, feat in enumerate(available):
        ax = axes_flat[i]
        groups = [
            df.loc[df["label_storm"] == 0, feat].dropna().values,
            df.loc[df["label_storm"] == 1, feat].dropna().values,
        ]
        bp = ax.boxplot(
            groups,
            labels=["No storm", "Storm"],
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        bp["boxes"][0].set_facecolor("steelblue")
        bp["boxes"][1].set_facecolor("darkorange")
        ax.set_title(feat.replace("omni_w24_72_", ""), fontsize=9)
        ax.set_ylabel("Value")

    # Hide unused axes
    for j in range(len(available), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("OMNI window features split by storm label", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", outpath)


def plot_storm_rate_by_year(
    df: pd.DataFrame,
    outpath: Path,
    cme_time_col: str = "startTime",
    dpi: int = 140,
) -> None:
    """
    Bar chart of storm rate per year.

    Reveals solar cycle variation — storm rates peak near solar maximum.
    Critically demonstrates why temporal (not random) train/test splitting
    is necessary to avoid data leakage.
    """
    if cme_time_col not in df.columns or "label_storm" not in df.columns:
        log.warning("plot_storm_rate_by_year: required columns missing — skipping.")
        return

    tmp = df[[cme_time_col, "label_storm"]].copy()
    tmp[cme_time_col] = pd.to_datetime(tmp[cme_time_col], utc=True, errors="coerce")
    tmp["year"] = tmp[cme_time_col].dt.year
    tmp = tmp.dropna(subset=["year", "label_storm"])

    if tmp.empty:
        log.warning("plot_storm_rate_by_year: no data — skipping.")
        return

    yearly = (
        tmp.groupby("year")["label_storm"]
        .agg(total="count", storms="sum")
        .assign(storm_rate_pct=lambda x: 100 * x["storms"] / x["total"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        yearly["year"].astype(str),
        yearly["storm_rate_pct"],
        color="steelblue",
        edgecolor="white",
    )
    for _, row in yearly.iterrows():
        ax.text(
            str(int(row["year"])),
            row["storm_rate_pct"] + 0.5,
            f"{row['storm_rate_pct']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xlabel("Year")
    ax.set_ylabel("Storm rate (%)")
    ax.set_title("Geomagnetic storm rate by year (solar cycle variation)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    log.info("Saved %s", outpath)