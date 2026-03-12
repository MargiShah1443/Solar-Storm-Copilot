from __future__ import annotations

import pandas as pd

from src.config import Config
from src.io.loaders import load_donki_cme, load_kp_csv, load_omni_csv
from src.preprocessing.omni_window_features import OmniWindowConfig, add_omni_arrival_window_features, omni_coverage_flag
from scripts.run_eda import label_storms_from_kp


def yearly_coverage_table(
    df: pd.DataFrame,
    time_col: str = "startTime",
    omni_cfg: OmniWindowConfig = OmniWindowConfig(),
) -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df["year"] = df[time_col].dt.year

    # coverage flags
    has_kp = df["kp_max_h72"].notna()
    has_omni = omni_coverage_flag(df, cfg=omni_cfg)

    # group summary
    g = df.groupby("year")
    out = g.size().rename("cme_total").to_frame()
    out["cme_with_kp"] = g.apply(lambda x: int(x["kp_max_h72"].notna().sum()))
    out["kp_coverage_pct"] = (out["cme_with_kp"] / out["cme_total"] * 100).round(1)

    out["cme_with_omni"] = g.apply(lambda x: int(omni_coverage_flag(x, cfg=omni_cfg).sum()))
    out["omni_coverage_pct"] = (out["cme_with_omni"] / out["cme_total"] * 100).round(1)

    out["storm_count"] = g["label_storm"].sum().astype(int)
    out["storm_rate_pct"] = (out["storm_count"] / out["cme_total"] * 100).round(1)

    return out.reset_index()


def main():
    cfg = Config()
    omni_cfg = OmniWindowConfig(start_hours=24, end_hours=72)

    print("[INFO] Loading datasets for time-window exploration...")
    df_cme = load_donki_cme(cfg.donki_cme_file)
    df_kp = load_kp_csv(cfg.kp_file)
    df_omni = load_omni_csv(cfg.omni_file)

    print("[INFO] Labeling storms using Kp lookahead window...")
    df_labeled = label_storms_from_kp(df_cme, df_kp)

    print("[INFO] Adding OMNI arrival-window features (T+24 to T+72)...")
    df_feat = add_omni_arrival_window_features(
        df_labeled,
        df_omni,
        cme_time_col="startTime",
        omni_time_col="time",
        cfg=omni_cfg,
    )

    print("[INFO] Computing per-year coverage table...")
    cov = yearly_coverage_table(df_feat, time_col="startTime", omni_cfg=omni_cfg)

    print("\n=== Yearly Coverage Summary ===")
    print(cov.to_string(index=False))


if __name__ == "__main__":
    main()
