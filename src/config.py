from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # ------------------------------------------------------------------ #
    # Project root is two levels above this file: src/ -> project root    #
    # ------------------------------------------------------------------ #
    project_root: Path = Path(__file__).resolve().parents[1]

    # Data directories
    data_dir: Path = project_root / "data"
    raw_dir: Path = data_dir / "raw"
    interim_dir: Path = data_dir / "interim"
    processed_dir: Path = data_dir / "processed"

    # Output directories
    outputs_dir: Path = project_root / "outputs"
    figures_dir: Path = outputs_dir / "figures"

    # ------------------------------------------------------------------ #
    # Raw source files                                                     #
    # ------------------------------------------------------------------ #
    donki_cme_file: Path = raw_dir / "donki_cme_2015_2023.json"
    donki_flr_file: Path = raw_dir / "donki_flr_2015_2023.json"
    kp_file: Path = raw_dir / "kp_3hr_2015_2023.csv"

    # OMNI: raw yearly .asc files live in a sub-folder
    omni_raw_dir: Path = raw_dir / "omni_hro_modified"
    # Cleaned, combined CSV produced by download_data.py (run once)
    omni_file: Path = raw_dir / "omni.csv"

    # ------------------------------------------------------------------ #
    # Processed / interim files                                           #
    # ------------------------------------------------------------------ #
    # Final modeling-ready feature table written by run_eda.py
    feature_table_file: Path = processed_dir / "cme_features_labeled.csv"
    # Missingness summary written by run_eda.py for quick inspection
    missingness_file: Path = interim_dir / "missingness_cme_features.csv"

    # ------------------------------------------------------------------ #
    # Download settings                                                   #
    # ------------------------------------------------------------------ #
    start_year: int = 2015
    end_year: int = 2023

    # ------------------------------------------------------------------ #
    # EDA / labeling settings                                             #
    # ------------------------------------------------------------------ #
    kp_storm_threshold: float = 5.0
    kp_lookahead_hours: int = 72
    omni_window_start_hours: int = 24
    omni_window_end_hours: int = 72
    fig_dpi: int = 140
    save_figs: bool = True

    # ------------------------------------------------------------------ #
    # Helper: ensure a directory exists (replaces paths.ensure_dir)      #
    # ------------------------------------------------------------------ #
    def makedirs(self) -> None:
        """Create all project directories if they do not exist."""
        for d in (
            self.raw_dir,
            self.omni_raw_dir,
            self.interim_dir,
            self.processed_dir,
            self.figures_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)