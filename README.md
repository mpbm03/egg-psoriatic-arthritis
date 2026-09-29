# Electrogastrography Biomarkers for Psoriatic Arthritis

Signal-processing and machine-learning pipeline for identifying electrogastrography (EGG) biomarkers in patients with Psoriatic Arthritis (PsA).

For each recording (per subject and recording type - `baseline` / `long`), the pipeline:

1. Loads the raw EGG signal
2. Detects and replaces motion artefacts (manually annotated)
3. Centers and normalizes the EGG channels
4. Applies a low-pass filter and downsamples
5. Segments the clean signal into variable-length sub-segments
6. Applies EMD (Empirical Mode Decomposition) and selects the gastric/intestinal IMFs
7. Reconstructs the gastric + intestinal signals and assigns labels (Control / Pathological)
8. Extracts features (temporal, statistical, spectral, and Morlet wavelet similarity)
9. Trains and evaluates ML models (Logistic Regression, Random Forest, Support Vector Machine,  K-Nearest Neighbors, Gradient Boosting)

## Repository structure

```
PsA_code/
├── main.py                      # entry point - runs the full pipeline (steps 1–9 above)
├── run_top_splits.py            # re-runs training on the best representative train/test splits
├── loso_evaluation.py           # Leave-One-Subject-Out (LOSO) evaluation
├── constants.py                 # all project configuration (including DATA_DIR)
│
├── data_io/                     # loading raw data and artefact annotations
├── preprocessing/               # artefact replacement, centering, normalization, filtering, downsampling
├── segmentation/                # splitting the clean signal into sub-segments
├── decomposition/                # EMD and IMF frequency estimation
├── imf_selection/                # gastric/intestinal IMF selection and final DataFrame assembly
├── features/                     # feature extraction (TSFEL, spectral, Morlet)
├── pipeline/                     # orchestrator - combines steps 1–8 per recording
├── ml/                            # train/test split, feature selection, grid search, plots
├── utils.py                       # shared helper functions (e.g. segment-level comparison)
│
├── analyse_segments.py           # post-hoc analysis - top_splits
├── analyse_segments_LOSO.py      # post-hoc analysis - LOSO
├── shap_analysis.py              # post-hoc analysis - top_splits
├── shap_analysis_LOSO.py         # post-hoc analysis - LOSO
├── statistical_analysis.py       # post-hoc analysis - either (feature statistics)
├── permutation_test.py           # post-hoc analysis - top_splits
├── permutation_test_LOSO.py      # post-hoc analysis - LOSO
└── age_analysis.py               # post-hoc analysis - either (age as a covariate)
```

## Requirements

- Python 3.12 (tested with this version)
- Main dependencies:

```
pandas
numpy
scipy
scikit-learn
emd
pycwt
tsfel
tqdm
joblib
matplotlib
shap
seaborn
statsmodels
```

Install with:

```bash
pip install pandas numpy scipy scikit-learn emd pycwt tsfel tqdm joblib matplotlib shap seaborn statsmodels
```

## Data setup

The pipeline expects the data to be organized as follows:

```
<data_folder>/                              e.g. "iProlepsis EGG data"
├── Control/
│   ├── ID31/
│   │   ├── baseline.txt
│   │   ├── long.txt
│   │   ├── ID31_baseline_noisy_intervals.txt
│   │   └── ID31_noisy_intervals.txt
│   └── ...
├── Pathological/
│   ├── ID3/
│   │   ├── baseline.txt
│   │   ├── long.txt
│   │   ├── ID3_baseline_noisy_intervals.txt
│   │   └── ID3_noisy_intervals.txt
│   └── ...
├── annotations.md
└── prandial_offsets.csv
```

- The two top-level group folders must be named exactly `Control` and `Pathological`.
- Each recording `.txt` file has whitespace/tab-separated columns, as defined by `ALL_COLUMNS` in `constants.py` (or `ALL_COLUMNS_NO_CH5` for subjects recorded without the CH8 channel - currently only `ID7`).
- Manual artefact annotation files are **required** for every recording - the pipeline raises a `FileNotFoundError` for any `baseline`/`long` recording that doesn't have one. Each subject needs both:
  - `<subject_id>_baseline_noisy_intervals.txt` - for the `baseline` recording
  - `<subject_id>_noisy_intervals.txt` - for the `long` recording

  Each file lists noisy/questionable sample intervals per channel as Python literals (`noisy_intervals1 = [(start, end), ...]`, etc.) - see `data_io/artifact_loader.py` for the full format.
- `prandial_offsets.csv` must have the columns `subject_id`, `meal_offset_min`. It defines, per subject, the exact offset (in minutes) marking when the meal/prandial window starts within the `long` recording.

## How to run

**`main.py` always has to be run first**, for either of the two downstream evaluation strategies below (`run_top_splits.py` or `loso_evaluation.py`) - it is the only script that does the raw signal processing (artefact replacement, filtering, segmentation, EMD, IMF selection) and feature extraction. Every other script in this repository loads its inputs from what `main.py` saves; none of them re-process the raw signal.

1. Clone the repository and install the dependencies (see above).
2. Open `constants.py` and change **only** the `DATA_DIR` variable to point to your local copy of the dataset:

   ```python
   DATA_DIR = Path(r"C:\path\to\your\iProlepsis EGG data")
   ```

   All other constants (sampling frequency, frequency bands, segmentation mode, ML models, etc.) are already set to the values used in the project and don't need to be changed to reproduce the results.
3. Run the pipeline from the repository root:

   ```bash
   python main.py
   ```

This processes every recording found in `DATA_DIR` and extracts the features. It then evaluates model performance across several candidate train/test splits by subject (500 by default), each trained and scored independently, to check how sensitive the results are to which subjects end up in the test set. The split whose best model achieves the highest test Macro F1 is picked as the representative split, and only that one is analysed in detail: it is trained both on the full feature set and through the SelectKBest feature-selection sweep, and its learning curves, confusion matrices, and feature-selection plots are the ones saved.
   - `df_features.joblib` - full DataFrame of extracted features (this is the input every other script in the repository needs)
   - `split_summary.csv` - summary of all candidate splits considered, with their test Macro F1 
   - Trained models (`.joblib`) for the representative split, both with the full feature set and with the best per-model feature subset (`best k`)
   - Plots for learning curves, confusion matrices, and the feature-selection sweep (`.png`) - for the representative split

4. **Then choose one (or both) of the two downstream evaluation strategies:**

   - **`run_top_splits.py`** - re-runs the full training/evaluation on the `TOP_N` best-performing train/test splits found in `split_summary.csv` (one single representative split per seed), saving each to its own `top_splits/seed_<seed>/` folder.
   - **`loso_evaluation.py`** - runs full Leave-One-Subject-Out cross-validation: every one of the 25 subjects is held out and predicted once, using the other 24 for training. This is slower (25 full training runs per model) but gives a more robust estimate of generalisation to unseen subjects. Results are saved to `loso_results/full/` and `loso_results/bestk_sweep/`.

   Both scripts only need `saved_models/df_features.joblib` from `main.py` - they do not depend on each other and can be run independently, in any order, or both.

## Post-hoc analysis scripts

Once you have results from `run_top_splits.py` and/or `loso_evaluation.py`, the following scripts provide additional analysis. Each is written for one specific evaluation strategy (matching its `_LOSO` suffix, or lack of one), except `statistical_analysis.py` and `age_analysis.py`, which only need `df_features.joblib` and work with either:

| Script | Requires                                                                                   | What it does |
|---|--------------------------------------------------------------------------------------------|---|
| `analyse_segments.py` | `run_top_splits.py`, `df_features.joblib` and `df_all_meta.joblib` (from `main.py`)  output | Per-subject figure of which segments were classified correctly/incorrectly along the recording's timeline |
| `analyse_segments_LOSO.py` | `loso_evaluation.py` output                                                                | Same analysis, aggregated across all 25 LOSO folds |
| `shap_analysis.py` | `run_top_splits.py` and `df_features.joblib` (from `main.py`) output                       | SHAP feature-importance plots for each trained model |
| `shap_analysis_LOSO.py` | `loso_evaluation.py` and `df_features.joblib` (from `main.py`) output                      | Same SHAP analysis, aggregated across LOSO folds (only valid for models where the selected features are consistent across all 25 folds) |
| `permutation_test.py` | `run_top_splits.py` and `df_features.joblib` (from `main.py`) output                       | Label-permutation significance test for the representative-split model |
| `permutation_test_LOSO.py` |              `df_features.joblib` (from `main.py`)                                                   | Label-permutation significance test for the aggregated LOSO Macro-F1 |
| `statistical_analysis.py` | `main.py` output only                                                                      | Shapiro-Wilk normality test + Welch's t-test / Mann-Whitney U per feature, with FDR correction |
| `age_analysis.py` | `main.py` output + `age_map.csv`                                                           | ANCOVA-style comparison of the group effect on key features, with and without age as a covariate |

## Key settings (`constants.py`)

| Constant | Description |
|---|---|
| `DATA_DIR` | **The only change needed** - path to the data folder |
| `FS` / `FS_DOWNSAMPLE` | Original and post-downsampling sampling frequency (Hz) |
| `BANDS_CPM` | Frequency bands (gastric, intestinal, etc.) in cycles per minute |
| `SEGMENTATION_MODE` | Signal segmentation mode (`"signal"` - signal-level segmentation) |
| `MIN_SIGNAL_SUBSEG_SEC` / `MAX_SIGNAL_SUBSEG_SEC` | Minimum/maximum sub-segment duration |
| `LABEL_MODE` | Label source (`"group"` - derived from the group folder) |
| `IMF_SELECTION_MODE` | IMF selection criterion (`"band_power_peaks"`) |
| `MODELS_DICT` | ML models and their hyperparameter grids for `GridSearchCV` |

`SEGMENTATION_MODE`, `LABEL_MODE`, and `IMF_SELECTION_MODE` are implemented as named modes rather than hardcoded logic, specifically so that future work can try a different segmentation, labelling, or IMF-selection strategy without restructuring the pipeline - just add a new branch for the new mode and point the constant at it. Their current status.
