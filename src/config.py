from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    # Project paths
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"
    raw_dir: Path = data_dir / "raw"
    interim_dir: Path = data_dir / "interim"
    processed_dir: Path = data_dir / "processed"

    outputs_dir: Path = project_root / "outputs"
    figures_dir: Path = outputs_dir / "figures"

    # File names (start with local downloads)
    donki_cme_file: Path = raw_dir / "donki_cme_2015_2023.json"
    donki_flr_file: Path = raw_dir / "donki_flr_2015_2023.json"
    kp_file: Path = raw_dir / "kp_3hr_2015_2023.csv"
    omni_file: Path = raw_dir / "omni.csv"

    # Time settings
    timezone: str = "UTC"

    # EDA controls
    fig_dpi: int = 140
    save_figs: bool = True
