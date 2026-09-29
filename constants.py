from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR = Path(r"C:\Users\08mar\OneDrive\Desktop\Disco_externo\iProlepsis EGG data")          # Root directory with ID folders and annotations.md
PRANDIAL_OFFSETS_FILE = DATA_DIR/"prandial_offsets.csv"


# ── Sampling ─────────────────────────────────────────────────────────────────

FS             = 1000    # Original sampling frequency (Hz)
FS_DOWNSAMPLE  = 10      # Target sampling frequency after downsampling (Hz)

# ── Columns ──────────────────────────────────────────────────────────────────

TIME_COL    = "time"
ALL_COLUMNS = ["nSeq", "DI", "CH1", "CH2", "CH3", "CH4", "CH5", "CH6", "CH7", "CH8"]
ALL_COLUMNS_NO_CH5 = ["nSeq", "DI", "CH1", "CH2", "CH3", "CH4", "CH6", "CH7", "CH8"]

CHANNELS        = ["CH1", "CH2", "CH3", "CH4", "CH6", "CH7", "CH8"]
EGG_CHANNELS    = ["CH1", "CH2", "CH3", "CH4"]
ACC_CHANNELS    = ["CH6", "CH7", "CH8"]

# ── Frequency ─────────────────────────────────────────────────────────────────

FREQ_MAX_FFT = 0.2   # Hz  (~12 CPM)

BANDS_CPM = {
    "gastric":     (1,  5),
    "intestinal":  (7, 12),
    "heart_rate":  (48.0, 120.0),
    "breathing":   (12.0,  25.0),
}

# ── Artifacts ─────────────────────────────────────────────────────────────────
#   False → discard the entire segment if ANY EGG channel exceeds
#            ARTIFACT_RATIO_THRESHOLD. Produces fewer but cleaner segments.
CHANNEL_NULLING = False

# Questionable artefact handling
#   If True, questionable intervals are treated as noisy and counted towards
#   the artifact_ratio used to discard segments. If False, questionable
#   intervals are stored for visualisation only and do not affect segmentation.
TREAT_QUESTIONABLE_AS_NOISY = True

# ── Segmentation mode ─────────────────────────────────────────────────────────
# SEGMENTATION_MODE: controls how the EGG signal is divided into segments.
#   "signal" → signal-level splitting: artifact and questionable intervals
#              from all channels are unioned, short isolated gaps are repaired
#              by cubic interpolation, and the remaining clean signal is split
#              into variable-length blocks of [MIN_SUBSEG_SEC, MAX_SUBSEG_SEC].
#              Recovers more usable signal than "window" mode, especially for
#              subjects with scattered artifacts, at the cost of variable
#              segment durations and loss of temporal comparability.
SEGMENTATION_MODE = "signal"

# ── Signal-level segmentation parameters (only used when SEGMENTATION_MODE="signal") ──
# MIN/MAX sub-segment duration for signal-level splitting.
# Each clean block is divided into sub-segments within this range.
# Blocks shorter than MIN are discarded. Blocks between MIN and MAX produce
# one sub-segment. Blocks longer than MAX are divided into multiple sub-segments
# of up to MAX seconds each (last one may be shorter but >= MIN).
MIN_SIGNAL_SUBSEG_SEC = 120.0  # 2 minutes minimum
MAX_SIGNAL_SUBSEG_SEC = 300.0  # 5 minutes maximum

# ── Segmentation ──────────────────────────────────────────────────────────────

SEGMENT_MINUTES = 5

# ── Labeling mode ─────────────────────────────────────────────────────────────
#   'group'    → label comes from the group folder (0 = control, 1 = pathological)
LABEL_MODE = "group"

# ── Pipeline mode ─────────────────────────────────────────────────────────────
#   'ml'   → only classical ML models (Logistic Regression, RF, SVM, KNN)
PIPELINE_MODE = "ml"


# ── Prandial window (only used when LABEL_MODE = 'prandial') ──────────────────
PRANDIAL_START_MIN = 10  # minutes after start of 'long' recording
PRANDIAL_END_MIN = 70  # minutes after start of 'long' recording


# ── IMF Selection Mode ────────────────────────────────────────────────────────
#   "band_power_peaks"      → Evaluates IMFs whose dominant frequency falls within the target band
#                             and requires the IMF to have a physiologically plausible number
#                             of peaks for the target band. Then, select IMF with the highest band power in each band
IMF_SELECTION_MODE = "band_power_peaks"


# ── Morlet wavelet similarity feature ──────────────────────────────────────────
# MORLET_OMEGA0: non-dimensional frequency parameter of the Morlet wavelet,
# following Komorowski & Pietraszek (2016), J Med Syst 40:10. The authors
# tested several wavelets for EGG analysis via CWTFT and found that the
# non-analytic Morlet wavelet with omega0=6 produced scalograms consistent
# with known EGG properties (dominant peak ~3 CPM, expected postprandial
# power changes). omega0=6 also satisfies the wavelet admissibility condition.
MORLET_OMEGA0 = 6.0

# ── Feature Extraction ─────────────────────────────────────────────────────────────
FEATURES_TO_NORMALIZE =[
    "Absolute Energy",
    "Area under the curve",
    "Band Power",
    "Neighbourhood Peaks",
    "Negative Turning Points",
    "Positive Turning Points",
    "Signal distance",
    "Sum absolute diff",
]

# Features affected by normalization — remove after extraction
FEATURES_TO_DROP_SUBSTRINGS = [
    "0_Mean",
    "0_Median",
    "0_Area under the curve",
    "0_Mean diff",
    "0_Median diff",
    "0_Max",
    "0_Min",
    "0_Absolute energy",
    "0_Average power",
    "0_Root mean square",
    "0_Variance",
    "0_Standard deviation",
    "0_Interquartile range",
    "0_Mean absolute deviation",
    "0_Median absolute deviation",
    "0_Mean absolute diff",
    "0_Median absolute diff",
    "0_Sum absolute diff",
    "0_Signal distance",
    "0_Histogram mode",
    "0_ECDF Percentile_0",
    "0_ECDF Percentile_1",
    "0_ECDF_",
    "0_Peak to peak distance",
]


# ── Train set segment balance threshold ────────────────────────────────────────
# MIN_TRAIN_BALANCE: minimum segment balance ratio in the training set
# BEFORE upsampling. If the original training set is too imbalanced,
# upsampling duplicates too many segments from the minority class,
# which can cause overfitting. A value of 0.5 means the minority class
# must have at least 50% as many segments as the majority class before
# upsampling is applied.
# Set to 0.0 to disable this filter entirely. Usualy I use 0.80
MIN_TRAIN_BALANCE = 0.90

# ── Test set segment proportion filter ────────────────────────────────────────
# MIN_TEST_SEG_RATIO / MAX_TEST_SEG_RATIO: acceptable range for the
# proportion of segments in the test set relative to all segments.
# Since test_size=0.2 refers to subjects not segments, the actual
# segment ratio can vary widely. This filter ensures the test set
# contains approximately 20% of all segments (e.g. between 15% and 25%).
# Set MIN=0.0 and MAX=1.0 to disable this filter entirely.
MIN_TEST_SEG_RATIO = 0.15
MAX_TEST_SEG_RATIO = 0.25


# ── Train and teste of ML models ──────────────────────────────────────────────────────────────
MODELS_DICT = {

    "Logistic Regression": {
        "_class": "sklearn.linear_model.LogisticRegression",
        "_params": {
            "max_iter":     1000,
            "random_state": 42,
            "class_weight": "balanced",
        },
        "_grid": {
            "C":       [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l1", "l2"],
            "solver":  ["liblinear"],
        },
    },

    "Random Forest": {
        "_class": "sklearn.ensemble.RandomForestClassifier",
        "_params": {
            "random_state": 42,
            "class_weight": "balanced",
        },
        "_grid": {
            "n_estimators":      [50, 100, 200],
            "max_depth":         [5, 10, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf":  [1, 2],
            "max_features":      ["sqrt", "log2"],
        },
    },

    "SVM": {
        "_class": "sklearn.svm.SVC",
        "_params": {
            "C":            1.0,
            "kernel":       "rbf",
            "degree":       2,
            "tol":          1e-2,
            "gamma":        "auto",
            "class_weight": "balanced",
            "probability":  True,
        },
        "_grid": {
            "C":      [0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"],
            "gamma":  ["auto", "scale"],
        },
    },

    "KNN": {
        "_class": "sklearn.neighbors.KNeighborsClassifier",
        "_params": {
            "n_neighbors": 3,
            "weights":     "uniform",
            "algorithm":   "auto",
            "p":           2,
            "leaf_size":   10,
            "metric":      "euclidean",
        },
        "_grid": {
            "n_neighbors": [3, 5, 7],
            "weights":     ["uniform", "distance"],
            "p":           [1, 2],
            "leaf_size":   [10, 30, 50],
            "metric":      ["euclidean", "manhattan"],
        },
    },
    "Gradient Boosting": {
        "_class": "sklearn.ensemble.GradientBoostingClassifier",
        "_params": {
            "random_state": 42,
        },
        "_grid": {
            "n_estimators":  [50, 100, 200],
            "max_depth":     [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample":     [0.8, 1.0],
            "max_features":  ["sqrt", "log2"],
        },
    },

}

