"""
Re-run the full training and evaluation pipeline for the top N splits
by mean test F1, as recorded in split_summary.csv.

For each split, the pipeline is re-run with the corresponding seed and
all artefacts (models, figures, CSVs) are saved to a dedicated folder:

    top_splits/seed_<seed>/

Usage
-----
Run from the project root after main.py has produced:
    saved_models/df_features.joblib
    saved_models/split_summary.csv

    python run_top_splits.py

Configuration
-------------
Adjust TOP_N and other constants at the top of this file.

Available Functions
-------------------
[Public]
run                 - Load split_summary.csv and df_features.joblib, select the top TOP_N splits by mean test F1, and re-run each through _run_single_split

------------------
[Private]
_run_single_split   - Re-run the full train/evaluate pipeline (full feature set + best-k sweep) for one seed and save all artefacts to save_dir

------------------
"""
# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
import pandas as pd
import joblib
import traceback
import matplotlib
matplotlib.use("Agg")

from sklearn.model_selection import StratifiedGroupKFold

# internal imports
from constants import MODELS_DICT
from ml import (prepare_Xy, results_to_dataframe,
                results_to_dataframe_bestk, plot_learning_curves,
                plot_confusion_matrices)
from ml.splitting import file_split
from ml.feature_selection import sweep_feature_selection
from ml.gridsearch import grid_search_models

# ------------------------------------------------------------------------------------------------------------------- #
# configuration
# ------------------------------------------------------------------------------------------------------------------- #

SUMMARY_CSV   = "saved_models/split_summary.csv"
DF_FEATURES   = "saved_models/df_features.joblib"
OUTPUT_ROOT   = "top_splits"

TOP_N         = 10
N_CV_SPLITS   = 3
N_CV_REPEATS  = 5
SCORING       = "f1_macro"

CORRELATION_THRESHOLD = 0.85

# ------------------------------------------------------------------------------------------------------------------- #
# suppress harmless sklearn warnings (single-class CV folds that are ignored)
# ------------------------------------------------------------------------------------------------------------------- #
import warnings
from sklearn.exceptions import FitFailedWarning

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.feature_selection._univariate_selection")
warnings.filterwarnings("ignore", category=FitFailedWarning)
# ------------------------------------------------------------------------------------------------------------------- #
# helpers
# ------------------------------------------------------------------------------------------------------------------- #

def _run_single_split(seed, df_features, save_dir):
    """
        Re-run the full train/evaluate pipeline (full feature set + best-k
        sweep) for one seed and save all artefacts to save_dir.

        :param seed:        random seed identifying the train/test split to reproduce
        :param df_features: full feature DataFrame (all subjects/segments)
        :param save_dir:    output directory for this seed's models, figures and CSVs
        :return: None
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n{'=' * 65}")
    print(f"  Seed {seed}  ->  {save_dir}")
    print(f"{'=' * 65}\n")

    # ── 1. Split ──────────────────────────────────────────────────────────
    df_train, df_test = file_split(df_features, test_size=0.2, random_state=seed)
    df_train_bal = df_train

    train_ids = sorted(df_train["subject_id"].unique())
    test_ids = sorted(df_test["subject_id"].unique())
    print(f"  Train subjects ({len(train_ids)}): {train_ids}")
    print(f"  Test  subjects ({len(test_ids)}):  {test_ids}")
    print(f"  Train segments: {len(df_train_bal)}  |  Test segments: {len(df_test)}")

    from utils import _majority_vote_subjects

    X_train, y_train = prepare_Xy(df_train_bal)
    X_test, y_test = prepare_Xy(df_test)
    subject_ids_train = df_train_bal["subject_id"].values
    subject_ids_test = df_test["subject_id"].values

    # ── 2. Full feature set ───────────────────────────────────────────────
    cv_splits = []
    for rep in range(N_CV_REPEATS):
        gkf = StratifiedGroupKFold(
            n_splits=N_CV_SPLITS, shuffle=True, random_state=rep
        )
        cv_splits.extend(gkf.split(X_train, y_train, subject_ids_train))

    results, test_f1s, cv_f1s, cv_mins, cv_maxs = grid_search_models(
        X_train=X_train,
        y_train=y_train,
        subject_ids_train=subject_ids_train,
        models_dict=MODELS_DICT,
        X_test=X_test,
        y_test=y_test,
        n_splits=N_CV_SPLITS,
        n_repeats=N_CV_REPEATS,
        scoring=SCORING,
    )

    results_df_full = results_to_dataframe(results, X_test, y_test)
    fitted_models_full = {name: gs.best_estimator_ for name, gs in results.items()}

    print("\n[full features]")
    print(results_df_full.to_string())

    # Subject-level majority vote — full features
    for name, model in fitted_models_full.items():
        y_pred = model.predict(X_test)
        _majority_vote_subjects(
            model_name=f"{name} (full)",
            y_pred=y_pred,
            y_true=y_test,
            subject_ids=subject_ids_test,
        )

    # Save full models
    for name, model in fitted_models_full.items():
        joblib.dump(model,
                    os.path.join(save_dir, f"{name.replace(' ', '_')}_full.joblib"))

    # Figures — full
    plot_learning_curves(
        fitted_models_full, X_train, y_train,
        subject_ids_train=subject_ids_train,
        save_path=os.path.join(save_dir, "learning_curves_full.png"),
    )
    plot_confusion_matrices(
        fitted_models_full, X_test, y_test,
        save_path=os.path.join(save_dir, "confusion_matrices_full.png"),
    )

    # ── 3. Feature selection sweep ────────────────────────────────────────
    feature_cols = [c for c in df_train_bal.columns
                    if c not in {"label", "subject_id", "_is_clean",
                                 "n_segments_total", "duration_sec"}]
    X_train_df = df_train_bal[feature_cols]
    X_test_df = df_test[feature_cols]

    (df_sweep, df_cv_results, best_gs_per_model,
     X_train_per_model, X_test_per_model,
     best_features_per_model, f_score_ranking) = sweep_feature_selection(
        X_train=X_train_df,
        y_train=y_train,
        X_test=X_test_df,
        y_test=y_test,
        subject_ids_train=subject_ids_train,
        models_dict=MODELS_DICT,
        max_k=20,
        n_cv_splits=N_CV_SPLITS,
        scoring=SCORING,
        correlation_threshold=CORRELATION_THRESHOLD,
    )


    best_k_per_model = {
        name: gs.best_estimator_.named_steps["selector"].k
        for name, gs in best_gs_per_model.items()
    }


    fitted_models_bestk = {name: gs.best_estimator_
                           for name, gs in best_gs_per_model.items()}

    results_df_bestk = results_to_dataframe_bestk(
        fitted_models_bestk=fitted_models_bestk,
        X_test_per_model=X_test_per_model,
        best_gs_per_model=best_gs_per_model,
        best_k_per_model=best_k_per_model,
        y_test=y_test,
    )

    print("\n[best-k features]")
    print(results_df_bestk.to_string())

    # Subject-level majority vote — best-k features
    for name, model in fitted_models_bestk.items():
        y_pred = model.predict(X_test_per_model[name])
        _majority_vote_subjects(
            model_name=f"{name} (k={best_k_per_model[name]})",
            y_pred=y_pred,
            y_true=y_test,
            subject_ids=subject_ids_test,
        )

    # Save best-k models
    for name, model in fitted_models_bestk.items():
        k = best_k_per_model[name]
        joblib.dump(model,
                    os.path.join(save_dir,
                                 f"{name.replace(' ', '_')}_bestk{k}.joblib"))

    # Figures — best-k (pipeline handles transformations internally)
    from ml.plots import plot_confusion_matrices_pipeline, plot_learning_curves_pipeline

    plot_confusion_matrices_pipeline(
        fitted_models=fitted_models_bestk,
        X_test_per_model=X_test_per_model,
        y_test=y_test,
        best_k_per_model=best_k_per_model,
        save_path=os.path.join(save_dir, "confusion_matrices_bestk.png"),
    )

    plot_learning_curves_pipeline(
        fitted_models=fitted_models_bestk,
        X_train=X_train_df.values,
        y_train=y_train,
        subject_ids_train=subject_ids_train,
        best_k_per_model=best_k_per_model,
        save_path=os.path.join(save_dir, "learning_curves_bestk.png"),
    )

    # ── 4. Save all artefacts ─────────────────────────────────────────────
    joblib.dump(test_ids,
                os.path.join(save_dir, "best_test_ids.joblib"))
    joblib.dump(df_test,
                os.path.join(save_dir, "best_test.joblib"))
    joblib.dump(best_features_per_model,
                os.path.join(save_dir, "best_features_per_model.joblib"))
    joblib.dump(best_k_per_model,
                os.path.join(save_dir, "best_k_per_model.joblib"))
    joblib.dump(f_score_ranking,
                os.path.join(save_dir, "f_score_ranking.joblib"))

    results_df_full.to_csv(os.path.join(save_dir, "results_full.csv"))
    results_df_bestk.to_csv(os.path.join(save_dir, "results_bestk.csv"))

    print(f"\n[save] All artefacts saved to {save_dir}/")


# ------------------------------------------------------------------------------------------------------------------- #
# entry point
# ------------------------------------------------------------------------------------------------------------------- #

def run():
    """
        Load split_summary.csv and df_features.joblib, select the top TOP_N
        splits by mean test F1, and re-run each through _run_single_split.

        :return: None
    """
    summary_df = pd.read_csv(SUMMARY_CSV, index_col=0)
    df_features = joblib.load(DF_FEATURES)

    # Select top N splits by mean_test_f1
    top_splits = summary_df.nlargest(TOP_N, "mean_test_f1")

    print(f"\nTop {TOP_N} splits by mean test F1:")
    print(top_splits[["seed", "mean_cv_f1", "mean_test_f1", "test_ids"]].to_string())

    for rank, (_, row) in enumerate(top_splits.iterrows(), 1):
        seed = int(row["seed"])
        save_dir = os.path.join(OUTPUT_ROOT, f"seed_{seed}")

        print(f"\n{'#' * 65}")
        print(f"  Rank {rank}/{TOP_N}  —  seed={seed}  "
              f"test F1={row['mean_test_f1']:.4f}")
        print(f"{'#' * 65}")

        try:
            _run_single_split(seed, df_features, save_dir)
        except Exception as e:
            print(f"  [ERROR] seed={seed} failed: {e}")
            traceback.print_exc()
            continue

    print(f"\n{'=' * 65}")
    print(f"  Done. Results in {OUTPUT_ROOT}/")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    run()