from pathlib import Path
from src.config import Config
from src.io.omni_hro_parser import export_omni_csv

def main():
    cfg = Config()

    omni_dir = cfg.raw_dir / "omni_hro_modified"
    omni_files = [
        omni_dir / "omni_5min2017.asc",
        omni_dir / "omni_5min2018.asc",
    ]

    out_csv = cfg.raw_dir / "omni.csv"
    export_omni_csv(omni_files, out_csv)
    print(f"Built OMNI CSV at: {out_csv}")

if __name__ == "__main__":
    main()
