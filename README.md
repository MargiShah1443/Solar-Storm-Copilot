# Solar Storm Copilot

**Predicting Geomagnetic Storms from Coronal Mass Ejections and Solar Wind Data**

> CS5100 Foundations of Artificial Intelligence — Northeastern University  
> Final Project

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Scientific Background](#2-scientific-background)
3. [Dataset](#3-dataset)
4. [Pipeline Overview](#4-pipeline-overview)
5. [Project Structure](#5-project-structure)
6. [Setup and Installation](#6-setup-and-installation)
7. [How to Run](#7-how-to-run)
8. [Results](#8-results)
9. [Key Findings](#9-key-findings)
10. [References](#10-references)

---

## 1. Problem Statement

Geomagnetic storms are disturbances in Earth's magnetic field driven by solar eruptions. They can disrupt satellites, communication systems, power grids, and GPS navigation. Early and accurate prediction of these events is a critical space weather problem.

This project builds a supervised machine learning pipeline that answers a single binary question for each observed Coronal Mass Ejection (CME):

> **Will this CME cause a geomagnetic storm (Kp ≥ 5) at Earth within the next 72 hours?**

The model ingests real NASA observational data, engineers physically meaningful features, and produces a calibrated storm probability score — acting as a data-driven copilot for space weather forecasters.

---

## 2. Scientific Background

### Coronal Mass Ejections (CMEs)

CMEs are large expulsions of magnetised plasma from the Sun's corona. They travel through the solar system at speeds ranging from 80 to 2,650 km/s and can reach Earth in 1–4 days. Not every CME causes a geomagnetic storm — the key determining factor is the orientation of the interplanetary magnetic field (IMF) embedded in the CME when it arrives at Earth.

### The Bz Component

When a CME arrives, if its southward IMF component (Bz_gsm) is negative (pointing south), it reconnects with Earth's northward magnetic field and drives energy into the magnetosphere. The more negative and sustained the Bz, the stronger the storm. This is why `min_Bz_gsm` is the single strongest storm predictor in this dataset.

### The Kp Index

The Kp index (0–9) is a global measure of geomagnetic disturbance, updated every 3 hours. A Kp ≥ 5 event is classified as a geomagnetic storm. This project uses the **maximum Kp within 72 hours after a CME** as the prediction target.

### Why Machine Learning?

Physics-based threshold rules (e.g. "storm if Bz < −8 nT") are fragile — they ignore interactions between features, cannot adapt to solar cycle variation, and miss storms driven by sustained moderate Bz. ML models can capture these non-linear dependencies across 61 solar wind and CME features simultaneously.

---

## 3. Dataset

Three NASA data sources are integrated, covering **2015–2023** (9 years, Solar Cycles 24 and 25).

| Source | Description | Size |
|---|---|---|
| [NASA DONKI](https://kauai.ccmc.gsfc.nasa.gov/DONKI/) | CME catalog: speed, direction, half-angle, type | 4,107 events |
| [NASA DONKI](https://kauai.ccmc.gsfc.nasa.gov/DONKI/) | Solar flare catalog | 1,148 events |
| [GFZ Potsdam](https://kp.gfz-potsdam.de/) | 3-hour Kp geomagnetic index | 26,296 rows |
| [NASA OMNI HRO](https://omniweb.gsfc.nasa.gov/) | 5-minute solar wind measurements | 946,656 rows |

### OMNI Features (per CME, 24–72h window)

For each CME, 48 summary statistics (mean, min, max, std) are computed from OMNI solar wind measurements in the 24–72 hour window after CME launch:

| Variable | Physical meaning |
|---|---|
| `Bz_gsm` | Southward IMF component — primary storm driver |
| `By_gsm` | East-west IMF component |
| `Bmag` | Total IMF magnitude |
| `V` | Solar wind bulk velocity |
| `Np` | Proton number density |
| `T` | Proton temperature |
| `Pdyn` | Dynamic (ram) pressure |
| `beta` | Plasma beta (thermal/magnetic pressure ratio) |
| `MachA` | Alfvénic Mach number |
| `AE` | Auroral electrojet index |
| `SYM_H` | Symmetric ring current index |

### Label Definition

```
label_storm = 1  if  max(Kp) >= 5  within 72 hours of CME onset
label_storm = 0  otherwise
```

**Class distribution:** 921 storms (22.4%) and 3,186 non-storms (77.6%) across 4,107 CME events.

### Train / Validation / Test Split

A **time-based split** is used to prevent solar cycle leakage. A random split would scatter Solar Cycle 24 maximum (2015, 34% storm rate) and minimum (2020, 0% storm rate) events across train and test, causing the model to appear better than it would on future data.

| Split | Years | Events | Storm Rate |
|---|---|---|---|
| Train | 2015–2020 | 1,338 | 24.5% |
| Validation | 2021 | 510 | 11.2% |
| **Test** | **2022–2023** | **2,252** | **23.8%** |

The test set spans Solar Cycle 25 — a genuinely different solar cycle from the training data — making this a realistic out-of-distribution evaluation.

---

## 4. Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Download raw data                                  │
│  python -m scripts.download_data                            │
│  → data/raw/ (OMNI .asc files, DONKI JSON, Kp CSV)          │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│  Step 2: Exploratory Data Analysis                          │
│  python -m scripts.run_eda                                  │
│  → outputs/figures/ (9 EDA plots)                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│  Step 3: Feature Engineering & Preprocessing                │
│  python -m scripts.build_features                           │
│  → data/processed/ (X/y train/val/test splits + scaler)     │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│  Step 4: Physics Baseline                                   │
│  python -m scripts.run_baseline                             │
│  → results/baseline_val_scores.csv                          │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│  Step 5: Train ML Models                                    │
│  python -m scripts.train_model                              │
│  → models/ (3 .pkl files) + results/val_scores.csv          │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│  Step 6: SHAP Explainability                                │
│  python -m scripts.run_shap                                 │
│  → outputs/figures/shap_*.png + results/shap_importance.csv │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│  Step 7: Final Evaluation + Ablation Study                  │
│  python -m scripts.evaluate                                 │
│  → results/test_scores.csv + ablation_scores.csv            │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Project Structure

```
Solar-Storm-Copilot/
│
├── data/
│   ├── raw/                          # Downloaded source files (not in git)
│   │   ├── omni_hro_modified/        # OMNI .asc files, one per year (2015–2023)
│   │   ├── donki_cme_2015_2023.json  # NASA DONKI CME catalog
│   │   ├── donki_flr_2015_2023.json  # NASA DONKI solar flare catalog
│   │   ├── kp_3hr_2015_2023.csv      # GFZ Kp geomagnetic index
│   │   └── omni.csv                  # Parsed & combined OMNI solar wind data
│   │
│   ├── interim/
│   │   └── missingness_cme_features.csv   # Missing value audit per column
│   │
│   └── processed/                    # Model-ready splits (not in git)
│       ├── cme_features_labeled.csv  # Full labeled feature matrix (4,107 × 68)
│       ├── X_train.csv               # Training features  (1,338 × 61)
│       ├── X_val.csv                 # Validation features (510 × 61)
│       ├── X_test.csv                # Test features       (2,252 × 61)
│       ├── y_train.csv               # Training labels
│       ├── y_val.csv                 # Validation labels
│       ├── y_test.csv                # Test labels
│       ├── preprocessor.pkl          # Fitted imputer + StandardScaler
│       ├── feature_names.txt         # Ordered list of 61 feature names
│       └── split_summary.csv         # Row counts and storm rates per split
│
├── models/                           # Trained model files (not in git)
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
│
├── outputs/
│   └── figures/                      # All generated plots
│       ├── class_imbalance.png
│       ├── cme_speed_hist.png
│       ├── corr_heatmap.png
│       ├── kp_outcome_by_label.png
│       ├── kp_timeline.png
│       ├── missingness_top.png
│       ├── omni_by_label.png
│       ├── storm_rate_by_speed.png
│       ├── storm_rate_by_year.png
│       ├── baseline_val.png
│       ├── val_roc_pr_curves.png
│       ├── val_confusion_matrices.png
│       ├── shap_summary.png
│       ├── shap_bar.png
│       ├── shap_bz_dependence.png
│       ├── test_confusion_matrix.png
│       ├── test_roc_pr_curves.png
│       └── ablation_comparison.png
│
├── results/                          # Metrics and evaluation outputs
│   ├── baseline_val_report.txt
│   ├── baseline_val_scores.csv
│   ├── val_scores.csv                # All models compared on validation set
│   ├── test_scores.csv               # Final test set results
│   ├── ablation_scores.csv           # With vs without SYM_H/AE
│   └── shap_feature_importance.csv   # All 61 features ranked by mean |SHAP|
│
├── scripts/                          # Executable pipeline scripts
│   ├── download_data.py              # Step 1: fetch all raw data from APIs
│   ├── run_eda.py                    # Step 2: EDA figures and coverage analysis
│   ├── build_features.py             # Step 3: feature engineering + splits
│   ├── run_baseline.py               # Step 4: physics threshold baseline
│   ├── train_model.py                # Step 5: train LR, RF, XGBoost
│   ├── run_shap.py                   # Step 6: SHAP explainability analysis
│   └── evaluate.py                   # Step 7: final test evaluation + ablation
│
├── src/                              # Core library modules
│   ├── eda/
│   │   ├── coverage.py               # Yearly coverage statistics
│   │   └── plots.py                  # EDA plotting functions
│   │
│   ├── io/
│   │   ├── downloaders.py            # NASA API and SPDF data fetchers
│   │   ├── loaders.py                # Load processed data from disk
│   │   └── omni_hro_parser.py        # OMNI ASCII format parser
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── feature_engineering.py    # All pre-split transformations
│   │   ├── features.py               # OMNI window feature computation
│   │   ├── omni_window_features.py   # Kp labeling + OMNI aggregation
│   │   ├── preprocessor.py           # Fit-on-train imputer + scaler
│   │   ├── splitter.py               # Time-based train/val/test split
│   │   └── validation.py             # Post-preprocessing sanity checks
│   │
│   └── utils/
│       └── logging_utils.py          # Shared logger configuration
│
├── config.py                         # Project-wide constants and paths
├── debug_omni.py                     # Diagnostic script for OMNI parser
├── README.md                         # This file
└── requirements.txt                  # Python dependencies
├── Predicting_Geomagnetic_Storms_from_Solar_Eruptions_Using_Machine_Learning.pdf
```

---

## 6. Setup and Installation

### Prerequisites

- Python 3.10 or higher
- pip

### 1. Clone the repository

```bash
git clone https://github.com/MargiShah1443/Solar-Storm-Copilot.git
cd Solar-Storm-Copilot
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### requirements.txt

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
shap>=0.44
matplotlib>=3.7
requests>=2.31
```

---

## 7. How to Run

Run every step from the **project root directory**. Each script is a self-contained module that reads from and writes to the paths defined in `config.py`.

### Step 1 — Download raw data

Fetches all source data from NASA APIs and SPDF. Skips files that already exist.

```bash
python -m scripts.download_data
```

**Output:** `data/raw/` — OMNI `.asc` files (2015–2023), DONKI JSON, Kp CSV, `omni.csv`

---

### Step 2 — Exploratory Data Analysis

```bash
python -m scripts.run_eda
```

**Output:** 9 figures in `outputs/figures/` — class balance, CME speed distribution, Kp timeline, feature separation by label, storm rate by year, correlation heatmap, missingness audit.

---

### Step 3 — Feature Engineering and Preprocessing

```bash
python -m scripts.build_features --input data/processed/cme_features_labeled.csv
```

**Optional flags:**
```bash
--train-end 2020    # last year in training set (default: 2020)
--val-year  2021    # validation year (default: 2021)
--test-start 2022   # first year in test set (default: 2022)
--no-scale          # skip StandardScaler (use for tree models only)
```

**Output:** `data/processed/` — `X_train.csv`, `X_val.csv`, `X_test.csv`, `y_*.csv`, `preprocessor.pkl`, `feature_names.txt`, `split_summary.csv`

---

### Step 4 — Physics Baseline

Evaluates a two-condition rule on the validation set. Sets the performance floor for all ML models.

```bash
python -m scripts.run_baseline
```

**Output:** `results/baseline_val_scores.csv`, `outputs/figures/baseline_val.png`

---

### Step 5 — Train ML Models

Trains Logistic Regression, Random Forest, and XGBoost in sequence. All evaluated on the validation set.

```bash
python -m scripts.train_model
```

**Output:** `models/*.pkl`, `results/val_scores.csv`, `outputs/figures/val_roc_pr_curves.png`, `outputs/figures/val_confusion_matrices.png`

---

### Step 6 — SHAP Explainability

Runs SHAP TreeExplainer on the best model (XGBoost) using the training set as background. Validates that learned feature importances are physically consistent.

```bash
python -m scripts.run_shap
```

**Output:** `outputs/figures/shap_summary.png`, `shap_bar.png`, `shap_bz_dependence.png`, `results/shap_feature_importance.csv`

---

### Step 7 — Final Evaluation

**Run this exactly once.** Evaluates the best model on the held-out test set and runs a SYM_H/AE ablation study.

```bash
python -m scripts.evaluate
```

**Output:** `results/test_scores.csv`, `outputs/figures/test_confusion_matrix.png`, `test_roc_pr_curves.png`, `ablation_comparison.png`

---

### Run the full pipeline in sequence

```bash
python -m scripts.download_data
python -m scripts.run_eda
python -m scripts.build_features --input data/processed/cme_features_labeled.csv
python -m scripts.run_baseline
python -m scripts.train_model
python -m scripts.run_shap
python -m scripts.evaluate
```

---

## 8. Results

### Model Comparison — Validation Set (2021)

| Model | F1 (Storm) | AUC-ROC | Precision | Recall |
|---|---|---|---|---|
| Physics baseline | 0.000 | 0.500 | — | — |
| Logistic Regression | 0.640 | 0.953 | 0.522 | 0.825 |
| Random Forest | 0.662 | 0.947 | 0.589 | 0.754 |
| **XGBoost** | **0.729** | **0.945** | **0.705** | 0.754 |

XGBoost was selected as the best model based on F1 score on the validation set.

---

### Final Results — Test Set (2022–2023)

| Model | F1 (Storm) | AUC-ROC | Precision | Recall |
|---|---|---|---|---|
| XGBoost (full features) | **0.685** | **0.909** | 0.624 | 0.759 |
| XGBoost (no SYM_H/AE) | 0.648 | 0.867 | 0.563 | 0.763 |

On the test set, XGBoost correctly identified **407 out of 536 geomagnetic storms** with 245 false alarms and 129 missed events.

---

### SYM_H / AE Ablation Study

SHAP analysis revealed that the AE auroral electrojet index (rank 1) and SYM_H ring current index (rank 2) were the most impactful features. Because these indices measure real-time geomagnetic disturbance, there was a concern they might partially measure the storm rather than predict it from preconditions.

The ablation study showed F1 decreased by only **3.7 points** (0.685 → 0.648) when all 8 SYM_H and AE features were removed. Recall was essentially unchanged (0.759 → 0.763). This confirms these indices carry genuine predictive signal — capturing pre-storm magnetospheric pre-conditioning — rather than simply leaking the label.

---

### SHAP Top 15 Features

| Rank | Feature | Mean \|SHAP\| | Physical interpretation |
|---|---|---|---|
| 1 | `max_AE` | 1.392 | Peak auroral activity — early storm onset signal |
| 2 | `min_SYM_H` | 1.377 | Ring current dip — storm pre-conditioning |
| 3 | `mean_SYM_H` | 0.657 | Sustained ring current depression |
| 4 | `max_Bmag` | 0.476 | Strong IMF — enhanced coupling |
| 5 | `feat_cme_energy` | 0.399 | CME speed × width proxy |
| 6 | `feat_coupling` | 0.333 | V × southward Bz coupling |
| 7 | `min_Bz_gsm` | 0.312 | Southward IMF — primary physical driver |
| 8 | `max_SYM_H` | 0.306 | Ring current variability |
| 9 | `min_T` | 0.297 | Solar wind temperature minimum |
| 10 | `mean_By_gsm` | 0.287 | East-west IMF orientation |

---

## 9. Key Findings

**1. ML substantially outperforms physics rules.**
The two-condition physics baseline (Bz < −8 nT AND V > 450 km/s) predicted zero storms on the 2021 validation set, achieving F1=0.000. XGBoost achieved F1=0.729 on the same data, demonstrating that learned non-linear combinations of 61 features capture storm-driving conditions that simple threshold rules miss entirely.

**2. The solar cycle is the dominant confound.**
Storm rates vary from 0% (2020, solar minimum) to 34% (2015, solar maximum). A random train/test split would distribute these dramatically different conditions across both sets, producing an artificially optimistic evaluation. The time-based split — train on Cycle 24 (2015–2020), test on Cycle 25 (2022–2023) — provides a realistic assessment of cross-cycle generalization.

**3. SYM_H and AE are legitimate predictors, not leakage.**
Despite ranking as the top two SHAP features, removing SYM_H and AE only reduces F1 by 3.7 points on the test set. These indices capture pre-storm magnetospheric pre-conditioning (quiet-time ring current state, baseline auroral activity) that genuinely predicts whether an incoming CME will produce a storm.

**4. Southward Bz is physically confirmed.**
The SHAP dependence plot for `min_Bz_gsm` shows a monotonically increasing relationship between more negative Bz and higher storm probability — exactly as geophysics predicts. The model has learned real physics, not statistical artefacts.

**5. CME speed alone is a weak predictor.**
Solar wind velocity (`mean_V`) ranked 52nd out of 61 features by SHAP importance. Storm rate varies only from 17.5% to 25.6% across CME speed quintiles. The orientation of the magnetic field in the CME — not its speed — determines geoeffectiveness.

---

## 10. References

1. **NASA DONKI CME and Flare Catalog**  
   https://kauai.ccmc.gsfc.nasa.gov/DONKI/

2. **NASA OMNI High-Resolution Solar Wind Data**  
   https://omniweb.gsfc.nasa.gov/

3. **GFZ Potsdam Kp Geomagnetic Index**  
   https://kp.gfz-potsdam.de/

4. Camporeale, E. (2019). The challenge of machine learning in space weather: Nowcasting and forecasting. *Space Weather*, 17(8), 1166–1207.  
   https://doi.org/10.1029/2018SW002061

5. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

6. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.

---

*Built as a final project for CS5100 Foundations of Artificial Intelligence, Northeastern University.*