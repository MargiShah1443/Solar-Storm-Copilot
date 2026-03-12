from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


COL_NAMES = [
    "year", "day", "hour", "minute",
    "imf_id", "plasma_id",
    "imf_npts", "plasma_npts", "interp_pct",
    "timeshift_sec", "timeshift_rms", "pfn_rms", "dbot1",
    "Bmag", "Bx_gse", "By_gse", "Bz_gse", "By_gsm", "Bz_gsm",
    "B_rms", "Bvec_rms",
    "V", "Vx_gse", "Vy_gse", "Vz_gse",
    "Np", "T",
    "Pdyn", "Efield", "beta", "MachA",
    "Xgse", "Ygse", "Zgse",
    "BSN_Xgse", "BSN_Ygse", "BSN_Zgse",
    "AE", "AL", "AU", "SYM_D", "SYM_H", "ASY_D", "ASY_H",
    "NaNp", "MachMS",
]

# Fill values seen in the SPDF OMNI modified-format docs
FILL_VALUES = {
    9999999, 999999, 99999,
    9999.99, 99999.9, 999.99,
    99.99, 9.999, 99.9
}


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _build_time(df: pd.DataFrame) -> pd.Series:
    # year + day-of-year + hour + minute -> UTC timestamp
    base = pd.to_datetime(df["year"].astype(int).astype(str), format="%Y", utc=True)
    return (
        base
        + pd.to_timedelta(df["day"].astype(int) - 1, unit="D")
        + pd.to_timedelta(df["hour"].astype(int), unit="h")
        + pd.to_timedelta(df["minute"].astype(int), unit="m")
    )


def load_omni_5min_asc(path: Path, year_min: int = 1995, year_max: int = 2026) -> pd.DataFrame:
    """
    Reads OMNI modified 5-min ASCII file (space-separated).
    Robust to header/metadata lines: drops rows where year/day/hour/minute are invalid.

    Files: omni_5minYYYY.asc
    """
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COL_NAMES,
        engine="python",
        on_bad_lines="skip",  # skip any weird lines
    )

    # Coerce key time columns to numeric; invalid rows become NaN
    df = _coerce_numeric(df, ["year", "day", "hour", "minute"])

    # Drop non-data rows (headers/blank/etc.)
    df = df.dropna(subset=["year", "day", "hour", "minute"]).copy()

    # Filter to valid ranges (per the format doc)
    df = df[
        (df["year"] >= year_min) & (df["year"] <= year_max) &
        (df["day"] >= 1) & (df["day"] <= 366) &
        (df["hour"] >= 0) & (df["hour"] <= 23) &
        (df["minute"] >= 0) & (df["minute"] <= 59)
    ].copy()

    # Replace numeric fill values with NA (only for numeric-like columns)
    for c in df.columns:
        df[c] = df[c].replace(list(FILL_VALUES), pd.NA)

    # Build timestamp
    df["time"] = _build_time(df)

    # Sort
    df = df.sort_values("time").reset_index(drop=True)
    return df


def combine_omni_years(files: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for p in files:
        frames.append(load_omni_5min_asc(p))
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("time").reset_index(drop=True)
    return df


def export_omni_csv(
    omni_files: Iterable[Path],
    out_csv: Path,
    keep_cols: Optional[list[str]] = None,
) -> Path:
    """
    Combines yearly OMNI 5-min files and exports a clean omni.csv for the pipeline.
    """
    if keep_cols is None:
        keep_cols = [
            "time",
            "Bmag", "By_gsm", "Bz_gsm",
            "V", "Np", "T",
            "Pdyn", "Efield", "beta", "MachA",
            "AE", "SYM_H"
        ]

    df = combine_omni_years(omni_files)

    # Keep only columns present
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # Convert numerics (except time)
    for c in df.columns:
        if c == "time":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv
