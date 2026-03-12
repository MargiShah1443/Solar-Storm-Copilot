"""
src/io/downloaders.py
---------------------
All network-fetch logic for the three raw data sources:
  1. NASA DONKI  (CME + solar flares)
  2. GFZ Kp index
  3. NASA/SPDF OMNI HRO modified (5-min solar wind)

After downloading OMNI .asc files, call build_omni_csv() to produce the
single omni.csv used by the rest of the pipeline.  This replaces the old
standalone scripts/build_omni_csv.py.
"""
from __future__ import annotations

import datetime as dt
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.io.omni_hro_parser import export_omni_csv
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# ------------------------------------------------------------------ #
# Shared HTTP session with automatic retries                          #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class DownloadResult:
    name: str
    url: str
    saved_to: Path
    n_rows: Optional[int] = None


def _make_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_SESSION = _make_session()


def _http_get(
    url: str,
    params: dict | None = None,
    timeout: int = 180,
) -> requests.Response:
    headers = {"User-Agent": "solar-storm-copilot/1.0 (course-project)"}
    resp = _SESSION.get(url, headers=headers, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp


# ------------------------------------------------------------------ #
# 1. DONKI  (CME / FLR) — chunked per calendar year                  #
# ------------------------------------------------------------------ #

def _year_chunks(start_year: int, end_year: int) -> List[Tuple[str, str]]:
    return [(f"{y}-01-01", f"{y}-12-31") for y in range(start_year, end_year + 1)]


def download_donki_json_chunked(
    activity: str,
    start_year: int,
    end_year: int,
    out_json: Path,
    timeout: int = 240,
    sleep_s: float = 0.6,
    overwrite: bool = False,
) -> DownloadResult:
    """
    Download a DONKI activity type ('CME' or 'FLR') year by year and
    concatenate into a single JSON list file.

    Chunking avoids gateway timeouts on large date ranges.
    """
    base_url = f"https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/{activity}"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # Return cached file if overwrite is not requested
    if out_json.exists() and not overwrite:
        try:
            with open(out_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            n = len(data) if isinstance(data, list) else None
        except Exception:
            n = None
        log.info("DONKI %s already exists (%s rows) — skipping download.", activity, n)
        return DownloadResult(
            name=f"donki_{activity.lower()}",
            url=base_url,
            saved_to=out_json,
            n_rows=n,
        )

    all_items: list[dict] = []
    for s, e in _year_chunks(start_year, end_year):
        log.info("  Fetching DONKI %s  %s → %s", activity, s, e)
        resp = _http_get(base_url, params={"startDate": s, "endDate": e}, timeout=timeout)
        chunk = resp.json()
        if isinstance(chunk, list):
            all_items.extend(chunk)
        else:
            all_items.append(chunk)
        time.sleep(sleep_s)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_items, f, indent=2)

    log.info("Saved DONKI %s → %s  (%d items)", activity, out_json, len(all_items))
    return DownloadResult(
        name=f"donki_{activity.lower()}",
        url=base_url,
        saved_to=out_json,
        n_rows=len(all_items),
    )


# ------------------------------------------------------------------ #
# 2. GFZ Kp index — complete series TXT                              #
# ------------------------------------------------------------------ #

def download_kp_from_complete_series_txt(
    start_date: str,
    end_date: str,
    out_csv: Path,
    timeout: int = 300,
    overwrite: bool = True,
) -> DownloadResult:
    """
    Download Kp 3-hourly data from the GFZ Potsdam complete-series file
    (records from 1932 onward) and filter to [start_date, end_date].

    Output CSV columns: time (UTC), kp (float).

    Fixed-width column offsets in the source file:
      year   [0:4]   month  [5:7]   day   [8:10]
      start_hour (hh.h) [11:15]     kp (ff.fff) [46:52]
    """
    url = "https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_since_1932.txt"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if out_csv.exists() and not overwrite:
        df = pd.read_csv(out_csv)
        log.info("Kp CSV already exists (%d rows) — skipping download.", len(df))
        return DownloadResult(name="kp_3hr", url=url, saved_to=out_csv, n_rows=len(df))

    log.info("Downloading GFZ Kp complete-series file...")
    resp = _http_get(url, timeout=timeout)
    lines = resp.text.splitlines()

    start_dt = pd.to_datetime(start_date, utc=True)
    end_dt_excl = pd.to_datetime(end_date, utc=True) + pd.Timedelta(days=1)

    rows = []
    for line in lines:
        if not line or line.startswith("#"):
            continue
        try:
            year = int(line[0:4])
            month = int(line[5:7])
            day = int(line[8:10])
            start_hour = float(line[11:15])
            kp = float(line[46:52])
        except Exception:
            continue

        t = dt.datetime(year, month, day, tzinfo=dt.timezone.utc) + dt.timedelta(
            hours=start_hour
        )
        if t < start_dt.to_pydatetime() or t >= end_dt_excl.to_pydatetime():
            continue
        rows.append((t, kp))

    df = pd.DataFrame(rows, columns=["time", "kp"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = (
        df.sort_values("time")
        .drop_duplicates(subset=["time"])
        .reset_index(drop=True)
    )
    df.to_csv(out_csv, index=False)
    log.info("Saved Kp CSV → %s  (%d rows)", out_csv, len(df))
    return DownloadResult(name="kp_3hr", url=url, saved_to=out_csv, n_rows=len(df))


# ------------------------------------------------------------------ #
# 3. OMNI HRO modified — yearly .asc files                           #
# ------------------------------------------------------------------ #

def _download_omni_year(
    year: int,
    out_dir: Path,
    resolution: str = "5min",
    overwrite: bool = False,
    timeout: int = 240,
    sleep_s: float = 0.2,
) -> DownloadResult:
    if resolution not in {"min", "5min"}:
        raise ValueError("resolution must be 'min' or '5min'")

    filename = f"omni_{resolution}{year}.asc"
    url = (
        f"https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/modified/{filename}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    if out_path.exists() and not overwrite:
        log.info("  OMNI %d already exists — skipping.", year)
        return DownloadResult(name="omni_hro_modified", url=url, saved_to=out_path)

    log.info("  Downloading OMNI %d ...", year)
    resp = _http_get(url, timeout=timeout)
    out_path.write_bytes(resp.content)
    time.sleep(sleep_s)
    return DownloadResult(name="omni_hro_modified", url=resp.url, saved_to=out_path)


def download_omni_hro_modified_years(
    years: Iterable[int],
    out_dir: Path,
    resolution: str = "5min",
    overwrite: bool = False,
) -> list[DownloadResult]:
    """Download one .asc file per year into out_dir."""
    results = []
    for y in years:
        results.append(_download_omni_year(y, out_dir, resolution=resolution, overwrite=overwrite))
    return results


# ------------------------------------------------------------------ #
# 4. Build omni.csv from downloaded .asc files                       #
#    (replaces the old scripts/build_omni_csv.py)                    #
# ------------------------------------------------------------------ #

def build_omni_csv(
    omni_raw_dir: Path,
    out_csv: Path,
    start_year: int,
    end_year: int,
    resolution: str = "5min",
    overwrite: bool = False,
) -> Path:
    """
    Combine all yearly OMNI .asc files in omni_raw_dir into a single
    clean omni.csv.  Only processes years in [start_year, end_year].

    This is a one-time operation: if out_csv already exists and
    overwrite=False the function returns immediately.
    """
    if out_csv.exists() and not overwrite:
        log.info("omni.csv already exists — skipping build. Pass overwrite=True to rebuild.")
        return out_csv

    asc_files = sorted(
        omni_raw_dir / f"omni_{resolution}{y}.asc"
        for y in range(start_year, end_year + 1)
    )
    missing = [f for f in asc_files if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing OMNI .asc files — run download_omni_hro_modified_years() first.\n"
            f"Missing: {[str(m) for m in missing]}"
        )

    log.info("Building omni.csv from %d .asc files...", len(asc_files))
    path = export_omni_csv(asc_files, out_csv)
    log.info("Saved omni.csv → %s", path)
    return path