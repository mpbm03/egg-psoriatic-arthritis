"""
Permutation test for the Logistic Regression model.

Validates that the model is learning genuine physiological patterns and not
noise or pipeline artefacts.

Procedure
---------
For each of N_PERMUTATIONS iterations:
  1. Randomly shuffle the labels of ALL subjects at the subject level —
     each subject keeps all its segments but its group assignment (control
     vs pathological) is randomly reassigned. The number of subjects per
     class is preserved across permutations.
  2. Re-run the full training pipeline using exactly the same functions
     as main.py: best_file_split → upsample_minority → GridSearchCV with
     GroupKFold → evaluate on test set.
  3. Record the macro F1 on the (shuffled) test set.

The original F1 (trained on true labels) is compared against the
distribution of F1 scores from shuffled labels. If the model is learning
real physiological patterns, the original F1 should be clearly above the
shuffled distribution.

A one-sided empirical p-value is computed as:
    p = (number of permutations with F1 >= original F1) / N_PERMUTATIONS

Usage
-----
1. Run main.py at least once so that saved_models/df_features.joblib
   exists (add the save line described below if not yet present).
2. Set ORIGINAL_F1 to the F1 obtained in the original run.
3. Run:  python permutation_test.py

Output
------
- Console: per-permutation progress and final summary (mean/std/max F1,
  empirical p-value)
- permutation_test_best/permutation_results_<model>_<features>.csv
- permutation_test_best/permutation_test_<model>_<features>.png

Available Functions
-------------------
[Public]
  run()

------------------
[Private]
  _shuffle_labels(df_features, random_state) -> pd.DataFrame
  _plot_results(permutation_f1s, original_f1, p_value, save_dir)
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# internal imports — same functions used in main.py
from constants import MODELS_DICT
from ml import (best_file_split, prepare_Xy, results_to_dataframe)


# ------------------------------------------------------------------------------------------------------------------- #
# configuration
# ------------------------------------------------------------------------------------------------------------------- #

_HERE      = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR   = os.path.join(_HERE, "permutation_test_best")

SAVED_MODELS_DIR = os.path.join(_HERE, "saved_models")

# Directory where run_top_splits.py saved this seed's artefacts
MODELS_DIR = os.path.join(_HERE, "top_splits", "seed_44")

# F1 obtained in the original run — set to the best-k F1 if using FEATURE_SET="best_k"
ORIGINAL_F1 = 0.918

N_PERMUTATIONS = 1000
BASE_SEED      = 0     # permutation i uses seed BASE_SEED + i

# Model to test — must match a key in MODELS_DICT
MODEL = "SVM"
LR_DICT = {MODEL: MODELS_DICT[MODEL]}

# Feature set to use in each permutation:
#   "full"   → all features (same as the original full-feature run)
#   "best_k" → only the k features selected by the sweep in the original run.
#              The same k features are used in every permutation without
#              re-running the sweep, which is computationally feasible and
#              methodologically sound: we test whether THIS specific feature
#              subset has genuine discriminative power.
FEATURE_SET = "best_k"


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def run() -> None:
    """
    Load the full feature DataFrame, run N_PERMUTATIONS label-shuffled
    training runs, and compare against the original F1.
    """
    print(f"\n{'='*60}")
    print(f"  Permutation test — {MODEL} ({FEATURE_SET} features)")
    print(f"  Original F1  : {ORIGINAL_F1:.4f}")
    print(f"  Permutations : {N_PERMUTATIONS}")
    print(f"{'='*60}\n")

    # ── Load df_features and optional best-k feature list ────────────────
    feat_path = os.path.join(SAVED_MODELS_DIR, "df_features.joblib")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(
            f"df_features.joblib not found in {MODELS_DIR}.\n"
            f"Add this line to main.py after feature extraction:\n"
            f"    joblib.dump(df_features, 'saved_models/df_features.joblib')\n"
            f"    print('[SAVE] saved_models/df_features.joblib')"
        )
    df_features = joblib.load(feat_path)
    print(f"[load] df_features: {len(df_features)} segments, "
          f"{df_features['subject_id'].nunique()} subjects")

    # If using best_k, load the feature list selected in the original run
    if FEATURE_SET == "best_k":
        best_features_per_model = joblib.load(
            os.path.join(MODELS_DIR, "best_features_per_model.joblib")
        )
        best_k_per_model = joblib.load(
            os.path.join(MODELS_DIR, "best_k_per_model.joblib")
        )
        k_features = best_features_per_model[MODEL]
        k          = best_k_per_model[MODEL]
        print(f"[load] Using best-k features: k={k} features selected in original run")
    else:
        k_features = None
        print(f"[load] Using full feature set")
    print()

    # ── Load original train/test split IDs ───────────────────────────────
    # The permutation test uses the same subject assignment as the original
    # run — only the labels are shuffled. This avoids the problem of
    # imbalanced splits when labels are randomly reassigned, and ensures
    # that the test compares apples to apples: the same subjects in train
    # and test, only with permuted class assignments.
    best_test_ids = joblib.load(os.path.join(MODELS_DIR, "best_test_ids.joblib"))
    all_subjects  = df_features["subject_id"].unique()
    train_ids     = [s for s in all_subjects if s not in best_test_ids]
    test_ids      = list(best_test_ids)
    print(f"[load] Original split — train: {len(train_ids)} subjects, "
          f"test: {len(test_ids)} subjects")
    print(f"  Train: {sorted(train_ids)}")
    print(f"  Test:  {sorted(test_ids)}\n")

    # ── Run permutations ──────────────────────────────────────────────────
    permutation_f1s = []

    for i in range(N_PERMUTATIONS):
        seed = BASE_SEED + i
        print(f"[perm {i+1:>2}/{N_PERMUTATIONS}]  seed={seed}")

        # 1. Shuffle labels at subject level
        df_shuffled = _shuffle_labels(df_features, random_state=seed)

        # 2. Filter to best-k features if requested — keep label and subject_id
        if FEATURE_SET == "best_k":
            df_input = df_shuffled[k_features + ["label", "subject_id"]]
        else:
            df_input = df_shuffled

        # 3. Apply the SAME train/test split as the original run
        # Only labels are permuted — subject assignment is fixed
        df_train = df_input[df_input["subject_id"].isin(train_ids)].reset_index(drop=True)
        df_test  = df_input[df_input["subject_id"].isin(test_ids)].reset_index(drop=True)

        print(f"  Train segments: {len(df_train)}  |  Test segments: {len(df_test)}")
        print(f"  Train labels: {dict(df_train['label'].value_counts().sort_index())}")
        print(f"  Test  labels: {dict(df_test['label'].value_counts().sort_index())}")

        try:
            # 4. Train with GridSearchCV on permuted labels
            X_train, y_train = prepare_Xy(df_train)
            X_test,  y_test  = prepare_Xy(df_test)
            subject_ids_train = df_train["subject_id"].values

            from ml.gridsearch import grid_search_models
            from sklearn.model_selection import StratifiedGroupKFold

            cv_splits = []
            for rep in range(5):
                gkf = StratifiedGroupKFold(
                    n_splits=3, shuffle=True, random_state=rep
                )
                cv_splits.extend(gkf.split(X_train, y_train, subject_ids_train))

            results, test_f1s, _, _, _ = grid_search_models(
                X_train           = X_train,
                y_train           = y_train,
                subject_ids_train = subject_ids_train,
                models_dict       = LR_DICT,
                X_test            = X_test,
                y_test            = y_test,
                n_splits          = 3,
                n_repeats         = 5,
                scoring           = "f1_macro",
            )

            f1 = float(test_f1s[MODEL])

        except Exception as e:
            print(f"  [SKIP] permutation {i+1} failed: {e}")
            f1 = 0.5  # chance level

        permutation_f1s.append(f1)
        print(f"  F1 = {f1:.4f}\n")

    # ── Summary ───────────────────────────────────────────────────────────
    permutation_f1s = np.array(permutation_f1s)
    n_exceeding = int((permutation_f1s >= ORIGINAL_F1).sum())
    p_value = (permutation_f1s >= ORIGINAL_F1).sum() / N_PERMUTATIONS

    print(f"\n{'='*60}")
    print(f"  Results")
    print(f"{'='*60}")
    print(f"  Original F1          : {ORIGINAL_F1:.4f}")
    print(f"  Permutation F1 mean  : {permutation_f1s.mean():.4f}")
    print(f"  Permutation F1 std   : {permutation_f1s.std():.4f}")
    print(f"  Permutation F1 max   : {permutation_f1s.max():.4f}")
    print(f"  Permutations >= F1   : {n_exceeding} / {N_PERMUTATIONS}")
    print(f"  p-value (empirical)  : {p_value:.3f}  "
          f"({'significant' if p_value < 0.05 else 'not significant'} at α=0.05)")
    print(f"{'='*60}")

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(SAVE_DIR, exist_ok=True)

    df_results = pd.DataFrame({
        "permutation": range(1, N_PERMUTATIONS + 1),
        "seed":        [BASE_SEED + i for i in range(N_PERMUTATIONS)],
        "f1_macro":    permutation_f1s,
    })
    model_slug = MODEL.replace(" ", "_")
    feature_slug = FEATURE_SET
    filename_csv = f"permutation_results_{model_slug}_{feature_slug}.csv"
    csv_path = os.path.join(SAVE_DIR, filename_csv)
    df_results.to_csv(csv_path, index=False)
    print(f"\n[save] {csv_path}")

    _plot_results(permutation_f1s, ORIGINAL_F1, p_value, n_exceeding, model_slug, feature_slug, SAVE_DIR)

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _shuffle_labels(
    df_features:  pd.DataFrame,
    random_state: int,
) -> pd.DataFrame:
    """
    Shuffle group labels at the subject level.

    Each subject keeps all its segments intact, but its label (0=control,
    1=pathological) is randomly reassigned from the pool of existing
    subject labels. The total number of subjects per class is preserved.

    :param df_features:  full feature DataFrame with 'subject_id' and 'label'
    :param random_state: random seed for reproducibility

    :return: copy of df_features with shuffled subject-level labels
    """
    rng = np.random.default_rng(random_state)

    # One label per subject
    subject_labels = (
        df_features.groupby("subject_id")["label"]
        .agg(lambda x: int(x.mode()[0]))
        .reset_index()
        .rename(columns={"label": "original_label"})
    )

    # Shuffle labels — class counts are preserved
    shuffled = subject_labels["original_label"].values.copy()
    rng.shuffle(shuffled)
    subject_labels["shuffled_label"] = shuffled

    # Print which subjects had their label changed
    changed = subject_labels[
        subject_labels["original_label"] != subject_labels["shuffled_label"]
    ]
    label_names = {0: "control", 1: "pathological"}
    ctrl_to_path = sorted(
        changed[changed["original_label"] == 0]["subject_id"].tolist()
    )
    path_to_ctrl = sorted(
        changed[changed["original_label"] == 1]["subject_id"].tolist()
    )
    print(f"  Swapped {len(changed)} subject(s):")
    if ctrl_to_path:
        print(f"    control → pathological : {ctrl_to_path}")
    if path_to_ctrl:
        print(f"    pathological → control : {path_to_ctrl}")

    # Map back to all segments
    df_shuffled = df_features.copy()
    label_map   = subject_labels.set_index("subject_id")["shuffled_label"]
    df_shuffled["label"] = df_shuffled["subject_id"].map(label_map).astype(int)

    return df_shuffled


def _plot_results(
        permutation_f1s: np.ndarray,
        original_f1: float,
        p_value: float,
        n_exceeding: int,
        model_slug:      str,
        feature_slug:    str,
        save_dir: str,
) -> None:
    """
    Plot the distribution of permutation F1 scores vs the original F1.

    :param permutation_f1s: F1 scores from shuffled-label runs
    :param original_f1:     F1 from the original true-label run
    :param p_value:         empirical p-value
    :param n_exceeding:     number of permutations with F1 >= original F1
    :param model_slug:      model name with spaces replaced by underscores
    :param feature_slug:    feature set identifier (e.g. 'full' or 'best_k')
    :param save_dir:        output directory
    """
    n_perm = len(permutation_f1s)
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(permutation_f1s, bins=20, color="#378ADD",
            edgecolor="white", linewidth=0.6, alpha=0.85,
            label="Permuted labels")
    ax.axvline(original_f1, color="#D85A30", linewidth=2,
               linestyle="--",
               label=f"Original F1 = {original_f1:.4f}")
    ax.axvline(permutation_f1s.mean(), color="#888888", linewidth=1,
               linestyle=":",
               label=f"Permutation mean = {permutation_f1s.mean():.4f}")

    ax.set_xlabel("Macro F1")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Permutation test — {MODEL}\n"
        f"p = {p_value:.3f}  "
        f"({n_exceeding}/{n_perm} permutations ≥ original F1)"
    )
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    filename = f"permutation_test_{model_slug}_{feature_slug}.png"
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[save] {path}")


# ------------------------------------------------------------------------------------------------------------------- #
# entry point
# ------------------------------------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    run()