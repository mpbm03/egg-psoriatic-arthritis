"""
Permutation test for the aggregated LOSO subject-level Macro-F1 (KNN, k=2,
Morlet similarity of EGG2 and EGG3 — the features and hyperparameters
already established as optimal on the real data).

To keep this computationally feasible, each permutation retrains the model
with the SAME fixed features and hyperparameters found on the real data,
instead of re-running the full k-sweep and grid search for every
permutation. This still tests the core question — "is the observed LOSO
performance better than expected by chance?" — without the (impractical)
cost of ~1000 full LOSO optimisations.

Usage
-----
Run from the project root after main.py has produced
saved_models/df_features.joblib:

    python permutation_test_LOSO.py

Output
------
- Console: observed Macro-F1, permutation mean, and empirical p-value
- loso_permutation_test_result.joblib — dict with observed_f1, permuted_f1s,
  and p_value

Available Functions
-------------------
[Public]
run
    Compute the observed LOSO Macro-F1 on the real labels, then repeat the
    same LOSO procedure N_PERMUTATIONS times with the subject labels
    shuffled, and save the empirical p-value.
run_single_loso
    Run one full LOSO evaluation with fixed hyperparameters/features,
    returning the subject-level aggregated Macro-F1.

------------------
[Private]

------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
from collections import Counter

import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
N_PERMUTATIONS = 1000

# Fixed features and KNN hyperparameters used for every LOSO fold in the
# permutation test. These are NOT re-optimised per permutation — they were
# fixed beforehand to the values most frequently selected across the 25
# per-subject LOSO models: for each of the 25 held-out subjects, KNN went
# through its own hyperparameter grid search and feature selection, and
# the values below are the mode (most common choice) across all 25 folds,
# rather than the values from any single fold.
FIXED_FEATURES = ["CH2_morlet_sim", "CH3_morlet_sim"]  # adjust if necessary
FIXED_KNN_PARAMS = {
    "n_neighbors": 7,
    "metric": "manhattan",
    "weights": "uniform",
    "leaf_size": 10,
    "p": 1,
}


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def run_single_loso(df_features: pd.DataFrame, subjects: list,
                     label_col: str = "label") -> float:
    """
    Run one full LOSO evaluation with fixed hyperparameters/features,
    returning the subject-level aggregated Macro-F1.

    :param df_features: feature table, one row per segment
    :param subjects: list of subject IDs to iterate over (each held out once)
    :param label_col: name of the label column
    :return: subject-level aggregated Macro-F1
    """
    y_true_subj = []
    y_pred_subj = []

    for test_subject in subjects:
        df_train = df_features[df_features["subject_id"] != test_subject]
        df_test  = df_features[df_features["subject_id"] == test_subject]

        X_train = df_train[FIXED_FEATURES].values
        y_train = df_train[label_col].values
        X_test  = df_test[FIXED_FEATURES].values

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(**FIXED_KNN_PARAMS)),
        ])
        pipe.fit(X_train, y_train)
        y_pred_segs = pipe.predict(X_test)

        # majority vote → subject-level prediction
        cnt = Counter(y_pred_segs)
        subject_pred = cnt.most_common(1)[0][0]
        true_label = int(df_test[label_col].mode()[0])

        y_true_subj.append(true_label)
        y_pred_subj.append(subject_pred)

    return f1_score(y_true_subj, y_pred_subj, average="macro", zero_division=0)


def run() -> None:
    """
    Compute the observed LOSO Macro-F1 on the real labels, then repeat the
    same LOSO procedure N_PERMUTATIONS times with the subject labels
    shuffled, and save the empirical p-value.
    """
    df_features = joblib.load("saved_models/df_features.joblib")
    subjects = sorted(df_features["subject_id"].unique())

    # ── observed value, with the original labels ───────────────────────────
    observed_f1 = run_single_loso(df_features, subjects)
    print(f"Observed Macro-F1 (real labels): {observed_f1:.4f}")

    # ── subject → original label map, to shuffle correctly ─────────────────
    subject_labels = (
        df_features.drop_duplicates("subject_id")
        .set_index("subject_id")["label"]
    )

    permuted_f1s = []
    rng = np.random.default_rng(seed=42)

    for i in range(N_PERMUTATIONS):
        shuffled_labels = subject_labels.sample(frac=1, random_state=rng.integers(1e9))
        shuffled_labels.index = subject_labels.index  # reassign to the same subjects

        df_perm = df_features.copy()
        df_perm["label"] = df_perm["subject_id"].map(shuffled_labels)

        f1_perm = run_single_loso(df_perm, subjects)
        permuted_f1s.append(f1_perm)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{N_PERMUTATIONS}] permutations completed")

    permuted_f1s = np.array(permuted_f1s)
    p_value = np.mean(permuted_f1s >= observed_f1)

    print(f"\nObserved Macro-F1: {observed_f1:.4f}")
    print(f"Permutation mean: {permuted_f1s.mean():.4f}")
    print(f"Empirical p-value: {p_value:.4f}")

    joblib.dump(
        {"observed_f1": observed_f1, "permuted_f1s": permuted_f1s, "p_value": p_value},
        "loso_permutation_test_result.joblib"
    )


# ------------------------------------------------------------------------------------------------------------------- #
# entry point
# ------------------------------------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    run()