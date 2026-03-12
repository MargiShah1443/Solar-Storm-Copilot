from __future__ import annotations

import datetime as dt
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, List, Dict, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ----------------------------
# Common utilities
# ----------------------------

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


def _http_get(url: str, params: dict | None = None, timeout: int = 180) -> requests.Response:
    headers = {"User-Agent": "solar-storm-copilot/1.0 (course-project)"}
    resp = _SESSION.get(url, headers=headers, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp


# ----------------------------
# DONKI (chunked by year)
# ----------------------------

def _date_chunks_by_year(start_year: int, end_year: int) -> List[Tuple[str, str]]:
    chunks: List[Tuple[str, str]] = []
    for y in range(start_year, end_year + 1):
        chunks.append((f"{y}-01-01", f"{y}-12-31"))
    return chunks


def download_donki_json_chunked(
    activity: str,              # "CME" or "FLR"
    start_year: int,
    end_year: int,
    out_json: Path,
    timeout: int = 240,
    sleep_s: float = 0.6,
    overwrite: bool = False,
) -> DownloadResult:
    """
    Downloads DONKI activity across multiple years by fetching per-year chunks,
    then concatenating into a single JSON list.

    Prevents timeouts for large date ranges.
    """
    base = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get"
    url = f"{base}/{activity}"

    out_json.parent.mkdir(parents=True, exist_ok=True)

    if out_json.exists() and not overwrite:
        try:
            with open(out_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            n = len(data) if isinstance(data, list) else None
        except Exception:
            n = None
        return DownloadResult(name=f"donki_{activity.lower()}", url=url, saved_to=out_json, n_rows=n)

    all_items: List[Dict] = []
    for s, e in _date_chunks_by_year(start_year, end_year):
        params = {"startDate": s, "endDate": e}
        resp = _http_get(url, params=params, timeout=timeout)
        items = resp.json()

        if isinstance(items, list):
            all_items.extend(items)
        else:
            all_items.append(items)

        time.sleep(sleep_s)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_items, f, indent=2)

    return DownloadResult(name=f"donki_{activity.lower()}", url=url, saved_to=out_json, n_rows=len(all_items))


# ----------------------------
# Kp (GFZ complete-series TXT)
# ----------------------------

def download_kp_from_complete_series_txt(
    start_date: str,   # "YYYY-MM-DD"
    end_date: str,     # "YYYY-MM-DD"
    out_csv: Path,
    timeout: int = 300,
    overwrite: bool = True,
) -> DownloadResult:
    """
    Downloads Kp from GFZ complete-series file:
      https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_since_1932.txt

    Parses fixed-width lines:
      year  [0:4]
      month [5:7]
      day   [8:10]
      start hour (hh.h) [11:15]
      Kp float (ff.fff) [46:52]

    Then filters to [start_date, end_date] inclusive and outputs CSV with:
      time (UTC), kp (float)
    """
    url = "https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_since_1932.txt"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if out_csv.exists() and not overwrite:
        df = pd.read_csv(out_csv)
        return DownloadResult(name="kp_3hr", url=url, saved_to=out_csv, n_rows=len(df))

    resp = _http_get(url, timeout=timeout)
    lines = resp.text.splitlines()

    start_dt = pd.to_datetime(start_date, utc=True)
    end_dt_exclusive = pd.to_datetime(end_date, utc=True) + pd.Timedelta(days=1)

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

        base = dt.datetime(year, month, day, tzinfo=dt.timezone.utc)
        t = base + dt.timedelta(hours=start_hour)

        if t < start_dt.to_pydatetime() or t >= end_dt_exclusive.to_pydatetime():
            continue

        rows.append((t, kp))

    df = pd.DataFrame(rows, columns=["time", "kp"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)

    df.to_csv(out_csv, index=False)
    return DownloadResult(name="kp_3hr", url=url, saved_to=out_csv, n_rows=len(df))


# ----------------------------
# OMNI HRO (modified) yearly downloads
# ----------------------------

def download_omni_hro_modified_year(
    year: int,
    out_dir: Path,
    resolution: str = "5min",
    overwrite: bool = False,
    sleep_s: float = 0.2,
    timeout: int = 240,
) -> DownloadResult:
    base = "https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/modified"
    if resolution not in {"min", "5min"}:
        raise ValueError("resolution must be 'min' or '5min'")

    filename = f"omni_{resolution}{year}.asc"
    url = f"{base}/{filename}"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    if out_path.exists() and not overwrite:
        return DownloadResult(name="omni_hro_modified", url=url, saved_to=out_path, n_rows=None)

    resp = _http_get(url, timeout=timeout)
    with open(out_path, "wb") as f:
        f.write(resp.content)

    time.sleep(sleep_s)
    return DownloadResult(name="omni_hro_modified", url=resp.url, saved_to=out_path, n_rows=None)


def download_omni_hro_modified_years(
    years: Iterable[int],
    out_dir: Path,
    resolution: str = "5min",
    overwrite: bool = False,
) -> list[DownloadResult]:
    results: list[DownloadResult] = []
    for y in years:
        results.append(
            download_omni_hro_modified_year(
                y, out_dir, resolution=resolution, overwrite=overwrite
            )
        )
    return results
