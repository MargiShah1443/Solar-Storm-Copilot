from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_cme_speed_hist(df: pd.DataFrame, outpath):
    """
    Histogram of CME speeds (analysis_speed).
    """
    if "analysis_speed" not in df.columns:
        print("[WARN] analysis_speed not found — skipping CME speed histogram.")
        return

    speeds = df["analysis_speed"].dropna()
    if speeds.empty:
        print("[WARN] No CME speed data available.")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(speeds, bins=40)
    plt.xlabel("CME speed (km/s)")
    plt.ylabel("Count")
    plt.title("CME speed distribution")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_corr_heatmap(df: pd.DataFrame, outpath):
    """
    Correlation heatmap for numeric features.
    """
    num_df = df.select_dtypes(include=[np.number])

    if num_df.shape[1] < 2:
        print("[WARN] Not enough numeric columns for correlation heatmap.")
        return

    corr = num_df.corr()

    plt.figure(figsize=(12, 10))
    plt.imshow(corr, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation heatmap (numeric features)")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_kp_outcome_by_label(df: pd.DataFrame, outpath):
    """
    Distribution of max Kp (72h) for storm vs no-storm CMEs.
    """
    if "kp_max_h72" not in df.columns or "label_storm" not in df.columns:
        print("[WARN] Missing kp_max_h72 or label_storm — skipping plot.")
        return

    storm = df[df["label_storm"] == 1]["kp_max_h72"].dropna()
    no_storm = df[df["label_storm"] == 0]["kp_max_h72"].dropna()

    if storm.empty and no_storm.empty:
        print("[WARN] No Kp data to plot.")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(no_storm, bins=30, alpha=0.7, label="No storm")
    plt.hist(storm, bins=30, alpha=0.7, label="Storm")
    plt.xlabel("Max Kp in next 72h")
    plt.ylabel("Count")
    plt.title("Kp outcome distribution by label")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_kp_timeline(df_kp: pd.DataFrame, outpath):
    """
    Timeline of Kp index over time.
    """
    if "time" not in df_kp.columns or "kp" not in df_kp.columns:
        print("[WARN] Missing time or kp column — skipping Kp timeline.")
        return

    plt.figure(figsize=(10, 4))
    plt.plot(df_kp["time"], df_kp["kp"], linewidth=0.7)
    plt.xlabel("Time (UTC)")
    plt.ylabel("Kp (0–9)")
    plt.title("Kp timeline")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_missingness_top(df: pd.DataFrame, outpath, top_n: int = 25):
    """
    Bar plot of columns with highest missing fraction.
    """
    miss_frac = df.isna().mean().sort_values(ascending=False).head(top_n)

    if miss_frac.empty:
        print("[WARN] No missingness to plot.")
        return

    plt.figure(figsize=(10, 5))
    miss_frac.plot(kind="bar")
    plt.ylabel("Missing fraction")
    plt.title(f"Top {top_n} columns by missingness")
    plt.xticks(rotation=75)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
