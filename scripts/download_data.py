from __future__ import annotations

from src.config import Config
from src.io.downloaders import (
    download_donki_json_chunked,
    download_kp_from_complete_series_txt,
    download_omni_hro_modified_years,
)

def main():
    cfg = Config()

    START_YEAR = 2015
    END_YEAR = 2023
    START_DATE = "2015-01-01"
    END_DATE = "2023-12-31"

    # 1) DONKI (CME + FLR) — chunked by year
    r_cme = download_donki_json_chunked(
        activity="CME",
        start_year=START_YEAR,
        end_year=END_YEAR,
        out_json=cfg.raw_dir / "donki_cme_2015_2023.json",
        timeout=240,
        sleep_s=0.6,
        overwrite=False,
    )
    print(f"Saved DONKI CME: {r_cme.saved_to} (items={r_cme.n_rows})")

    r_flr = download_donki_json_chunked(
        activity="FLR",
        start_year=START_YEAR,
        end_year=END_YEAR,
        out_json=cfg.raw_dir / "donki_flr_2015_2023.json",
        timeout=240,
        sleep_s=0.6,
        overwrite=False,
    )
    print(f"Saved DONKI FLR: {r_flr.saved_to} (items={r_flr.n_rows})")

    # 2) Kp (3-hourly) — GFZ complete-series TXT
    r_kp = download_kp_from_complete_series_txt(
        start_date=START_DATE,
        end_date=END_DATE,
        out_csv=cfg.raw_dir / "kp_3hr_2015_2023.csv",
        timeout=300,
        overwrite=True,   # important: overwrite the previous empty CSV
    )
    print(f"Saved Kp 3-hour (GFZ TXT): {r_kp.saved_to} (rows={r_kp.n_rows})")

    # 3) OMNI HRO modified (5-min)
    years = range(START_YEAR, END_YEAR + 1)
    results = download_omni_hro_modified_years(
        years,
        cfg.raw_dir / "omni_hro_modified",
        resolution="5min",
        overwrite=False,
    )
    for r in results:
        print(f"Saved OMNI HRO: {r.saved_to}")

if __name__ == "__main__":
    main()
