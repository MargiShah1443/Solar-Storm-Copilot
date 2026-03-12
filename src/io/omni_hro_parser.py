"""
src/io/omni_hro_parser.py
--------------------------
Parses NASA/SPDF OMNI modified 5-minute ASCII files (.asc) into
clean pandas DataFrames.

This module is deliberately kept as pure parsing logic — no network
calls, no pipeline awareness.  The download step lives in downloaders.py
and the load-from-CSV step lives in loaders.py.

Public API:
  load_omni_5min_asc(path)          -> DataFrame for one year
  combine_omni_years(files)         -> DataFrame for multiple years
  export_omni_csv(files, out_csv)   -> writes combined CSV, returns Path
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# ------------------------------------------------------------------ #
# Column schema for the SPDF OMNI modified 5-min format              #
# ------------------------------------------------------------------ #

# Known named columns in order — covers the standard 55-col OMNI HRO schema.
# If a file has MORE columns than this list, extras get auto-named _extra_N.
# If a file has FEWER, only the present ones are used.  Either way, no warning.
_COL_NAMES_BASE = [
    # --- Time (cols 1–4) ---
    "year", "day", "hour", "minute",
    # --- Spacecraft / data-quality (cols 5–13) ---
    "imf_id", "plasma_id",
    "imf_npts", "plasma_npts", "interp_pct",
    "timeshift_sec", "timeshift_rms", "pfn_rms", "dbot1",
    # --- IMF (cols 14–21) ---
    "Bmag", "Bx_gse", "By_gse", "Bz_gse", "By_gsm", "Bz_gsm",
    "B_rms", "Bvec_rms",
    # --- Plasma (cols 22–31) ---
    "V", "Vx_gse", "Vy_gse", "Vz_gse",
    "Np", "T",
    "Pdyn", "Efield", "beta", "MachA",
    # --- Position (cols 32–37) ---
    "Xgse", "Ygse", "Zgse",
    "BSN_Xgse", "BSN_Ygse", "BSN_Zgse",
    # --- Geomagnetic indices (cols 38–45) ---
    "AE", "AL", "AU", "SYM_D", "SYM_H", "ASY_D", "ASY_H",
    # --- Derived (cols 46–49) ---
    "NaNp", "MachMS", "QI", "QF",
    # --- Extended fields present in some file versions (cols 50+) ---
    "Lyman_alpha", "QF2", "QF3", "QF4", "QF5", "QF6",
]


def _detect_col_names(path: Path) -> list[str]:
    """Read the first non-empty line to count actual columns, then build names."""
    with open(path) as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                n_actual = len(stripped.split())
                break
        else:
            return _COL_NAMES_BASE  # empty file fallback

    if n_actual <= len(_COL_NAMES_BASE):
        return _COL_NAMES_BASE[:n_actual]
    # File has more columns than our schema — pad with generic names
    extras = [f"_extra_{i}" for i in range(n_actual - len(_COL_NAMES_BASE))]
    return _COL_NAMES_BASE + extras

# Sentinel fill values defined in the SPDF OMNI format documentation
_FILL_VALUES = {
    9_999_999, 999_999, 99_999,
    9_999.99, 99_999.9, 999.99,
    99.99, 9.999, 99.9,
}

# Columns to export to omni.csv (keeps file size manageable)
_DEFAULT_KEEP_COLS = [
    "time",
    "Bmag", "By_gsm", "Bz_gsm",
    "V", "Np", "T",
    "Pdyn", "Efield", "beta", "MachA",
    "AE", "SYM_H",
]


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _build_time(df: pd.DataFrame) -> pd.Series:
    """Convert year + day-of-year + hour + minute → UTC Timestamp."""
    base = pd.to_datetime(df["year"].astype(int).astype(str), format="%Y", utc=True)
    return (
        base
        + pd.to_timedelta(df["day"].astype(int) - 1, unit="D")
        + pd.to_timedelta(df["hour"].astype(int), unit="h")
        + pd.to_timedelta(df["minute"].astype(int), unit="m")
    )


# ------------------------------------------------------------------ #
# Public API                                                          #
# ------------------------------------------------------------------ #

def load_omni_5min_asc(
    path: Path,
    year_min: int = 1995,
    year_max: int = 2030,
) -> pd.DataFrame:
    """
    Read a single OMNI modified 5-min ASCII file into a clean DataFrame.

    - Skips malformed / header lines automatically.
    - Replaces all SPDF fill values with NaN.
    - Returns a DataFrame sorted by the 'time' column (UTC).
    """
    log.info("  Parsing %s", path.name)
    col_names = _detect_col_names(path)
    log.debug("    %s: detected %d columns", path.name, len(col_names))
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=col_names,
        engine="python",
        on_bad_lines="skip",
        index_col=False,        # never absorb columns into the row index
    )

    df = _coerce_numeric(df, ["year", "day", "hour", "minute"])
    df = df.dropna(subset=["year", "day", "hour", "minute"]).copy()

    # Drop rows outside valid calendar ranges
    df = df[
        df["year"].between(year_min, year_max)
        & df["day"].between(1, 366)
        & df["hour"].between(0, 23)
        & df["minute"].between(0, 59)
    ].copy()

    # Build timestamp FIRST (before fill-value replacement wipes time columns)
    df["time"] = _build_time(df)

    # Now replace sentinel fill values — skip the four time-index columns
    # and the newly built 'time' column so we don't corrupt them
    time_cols = {"year", "day", "hour", "minute", "time"}
    fill_list = list(_FILL_VALUES)
    for c in df.columns:
        if c not in time_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].replace(fill_list, pd.NA)

    df = df.sort_values("time").reset_index(drop=True)
    return df


def combine_omni_years(files: Iterable[Path]) -> pd.DataFrame:
    """Concatenate multiple yearly OMNI .asc files into one DataFrame."""
    frames = [load_omni_5min_asc(p) for p in files]
    if not frames:
        raise ValueError("No OMNI files provided to combine_omni_years().")
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    return df


def export_omni_csv(
    omni_files: Iterable[Path],
    out_csv: Path,
    keep_cols: Optional[list[str]] = None,
) -> Path:
    """
    Combine yearly OMNI .asc files and export a trimmed CSV.

    Only the columns in keep_cols are written; defaults to the set of
    solar wind / IMF variables most relevant for geomagnetic storm
    prediction (see _DEFAULT_KEEP_COLS).
    """
    if keep_cols is None:
        keep_cols = _DEFAULT_KEEP_COLS

    df = combine_omni_years(list(omni_files))

    present = [c for c in keep_cols if c in df.columns]
    missing_req = set(keep_cols) - set(present)
    if missing_req:
        log.warning("Requested OMNI columns not found and will be skipped: %s", missing_req)

    df = df[present].copy()

    # Ensure numerics (except time)
    for c in df.columns:
        if c != "time":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log.info("Exported omni.csv → %s  (%d rows, %d cols)", out_csv, len(df), len(present))
    return out_csv