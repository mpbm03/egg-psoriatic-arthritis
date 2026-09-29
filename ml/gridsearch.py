"""
GridSearchCV with GroupKFold cross-validation.

GroupKFold ensures that segments from the same subject never appear in
both the train and validation folds during cross-validation — mirroring
the same logic used for the final train/test split.

Available Functions
-------------------
[Public]
grid_search_models   - Run (repeated) StratifiedGroupKFold GridSearchCV for every model in models_dict, evaluating best params on train/test sets
results_to_dataframe - Build a macro-metrics summary DataFrame (and print a per-class breakdown) from fitted GridSearchCV results

------------------
[Private]
_build_model    - Instantiate a model from its config dict by resolving its class path at runtime
_print_metrics  - Print and return precision/recall/F1/accuracy for a given prediction split

------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import importlib
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (
    precision_score, recall_score,
    f1_score, accuracy_score,
)
from typing import Dict, Tuple


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def grid_search_models(
        X_train: np.ndarray,
        y_train: np.ndarray,
        subject_ids_train: np.ndarray,
        models_dict: Dict[str, dict],
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_splits: int = 5,
        n_repeats: int = 1,
        base_seed: int = 42,
        scoring: str = "f1_macro",
        n_jobs: int = -1,
) -> Tuple[Dict[str, GridSearchCV], Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Run GridSearchCV with (repeated) StratifiedGroupKFold for every model in
    models_dict.

    StratifiedGroupKFold combines two constraints:
      - Group constraint: segments from the same subject never appear
        simultaneously in the train and validation folds, preventing
        data leakage.
      - Stratification constraint: each fold preserves the IDs class proportion
        (control vs pathological).

    When n_repeats > 1, the StratifiedGroupKFold split (n_splits folds) is
    repeated n_repeats times with different random group assignments
    (shuffle=True, seeds base_seed, base_seed+1, ...), and all
    n_splits * n_repeats folds are pooled into a single list of
    (train_idx, val_idx) pairs passed to GridSearchCV. This reduces the
    variance of the CV F1 estimate compared to a single split — with a
    small number of subjects, the specific group assignment of one split
    can heavily bias the result.

    The best hyperparameter combination is selected by the mean CV F1 across
    all n_splits * n_repeats folds.

    :param X_train: training features
    :param y_train: training labels
    :param subject_ids_train: subject ID per training sample
                         (used as groups for GroupKFold)
    :param models_dict: model configs from model_configs.py
    :param X_test: test features
    :param y_test: test labels
    :param n_splits: number of StratifiedGroupKFold splits per repeat. Default 5.
    :param n_repeats: number of times to repeat the n_splits-fold split with
                      different random group assignments. Default 1
                      (no repetition — original behaviour).
    :param base_seed: base random seed for repeats. Repeat i uses
                      random_state=base_seed + i. Default 42.
    :param scoring: str         GridSearchCV scoring metric. Default 'f1_macro'.
    :param n_jobs: int         parallel jobs. Default -1 (all cores).

    :return:
    results          : dict[str, GridSearchCV]   fitted GridSearchCV objects
    test_f1_scores   : dict[str, float]          macro F1 on test set per model
    cv_f1_scores     : dict[str, float]          mean CV F1 across all pooled
                                                 folds for best params
    cv_f1_min_scores : dict[str, float]          min CV F1 across folds for best params
    cv_f1_max_scores : dict[str, float]          max CV F1 across folds for best params
    """
    # ── Build the pooled list of (train_idx, val_idx) folds ───────────────
    # Repeat the StratifiedGroupKFold split n_repeats times with shuffle=True
    # and different seeds, then pool all folds together. GridSearchCV accepts
    # any iterable of (train_idx, val_idx) pairs via cv=, so the repeated
    # folds are treated exactly like a larger single CV split.
    cv_splits = []
    for rep in range(n_repeats):
        group_kfold = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=base_seed + rep
        )
        cv_splits.extend(group_kfold.split(X_train, y_train, subject_ids_train))

    total_folds = n_splits * n_repeats
    print(f"  CV folds: {n_splits} splits x {n_repeats} repeat(s) = {total_folds} total")

    results = {}
    test_f1_scores = {}
    cv_f1_scores = {}
    cv_f1_min_scores = {}
    cv_f1_max_scores = {}

    for model_name, config in tqdm(models_dict.items(), desc="Models"):
        print(f"\n{'=' * 55}")
        print(f"  GridSearch — {model_name}")
        print(f"{'=' * 55}")

        # Build model and wrap in a scaling pipeline
        # SimpleImputer handles residual NaN values that may appear when
        # SIMULATE_REAL_DEPLOYMENT=True accepts noisy segments where EMD
        # fails to reconstruct some channels.
        model = _build_model(config)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", model),
        ])
        # Prefix param grid keys with 'clf__' for the pipeline
        param_grid = {
            f"clf__{k}": v
            for k, v in config["_grid"].items()
        }

        # ── For KNN: cap n_neighbors to the smallest fold size ─────────
        # GroupKFold can produce small folds — n_neighbors must be < n_samples_fit
        if "clf__n_neighbors" in param_grid:
            min_fold_size = min(len(train_idx) for train_idx, _ in cv_splits)
            safe_k = [k for k in param_grid["clf__n_neighbors"] if k < min_fold_size]
            if not safe_k:
                safe_k = [1]  # fallback: at least 1 neighbour always works
            param_grid["clf__n_neighbors"] = safe_k
            print(f"  KNN n_neighbors capped to {safe_k} (min fold size = {min_fold_size})")

        # Run GridSearchCV
        grid_search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid if param_grid else {},
            cv=cv_splits,
            scoring=scoring,
            verbose=1,
            n_jobs=n_jobs,
            refit=True,
            error_score="raise",
        )
        grid_search.fit(X_train, y_train)

        cv_f1_scores[model_name] = grid_search.best_score_

        # Min and max F1 across all pooled folds for the best hyperparameter
        # combination. best_index_ is the row in cv_results_ corresponding
        # to the best params. split{j}_test_score contains the score for
        # fold j — there are total_folds = n_splits * n_repeats such splits.
        best_idx = grid_search.best_index_
        fold_scores = np.array([
            grid_search.cv_results_[f"split{j}_test_score"][best_idx]
            for j in range(total_folds)
        ])
        cv_f1_min_scores[model_name] = float(fold_scores.min())
        cv_f1_max_scores[model_name] = float(fold_scores.max())

        print(f"\n  Best params : {grid_search.best_params_}")
        print(f"  Best CV {scoring}: {grid_search.best_score_:.4f}  "
              f"[min={cv_f1_min_scores[model_name]:.4f}, "
              f"max={cv_f1_max_scores[model_name]:.4f}]")

        # Evaluate on test set
        y_test_pred = grid_search.predict(X_test)
        test_metrics = _print_metrics(y_test, y_test_pred, "Test set")

        # Evaluate on train set (overfitting check)
        y_train_pred = grid_search.predict(X_train)
        _print_metrics(y_train, y_train_pred, "Train set")

        results[model_name] = grid_search
        test_f1_scores[model_name] = test_metrics["f1"]

    # Summary table
    print(f"\n{'=' * 55}")
    print("  Summary — Test F1 scores")
    print(f"{'=' * 55}")
    for name, score in sorted(test_f1_scores.items(), key=lambda x: -x[1]):
        print(f"  {name:<25} F1 = {score:.4f}")

    return results, test_f1_scores, cv_f1_scores, cv_f1_min_scores, cv_f1_max_scores


def results_to_dataframe(
    results:       Dict[str, GridSearchCV],
    X_test:        np.ndarray,
    y_test:        np.ndarray,
) -> pd.DataFrame:
    """
    Build a summary DataFrame from fitted GridSearchCV results.

    Also prints a per-class metrics breakdown table showing Precision,
    Recall and F1 for each class separately — making it transparent where
    each model succeeds or fails (e.g. high recall on controls but low
    recall on pathological).

    :param results: dict[str, GridSearchCV]  output of grid_search_models
    :param X_test: np.ndarray
    :param y_test: np.ndarray

    :return: Dataframe  rows = models, cols = best_params + test metrics
    """
    rows       = []
    rows_class = []
    for model_name, gs in results.items():
        y_pred = gs.predict(X_test)
        row = {
            "model":        model_name,
            "best_params":  gs.best_params_,
            "cv_score":     gs.best_score_,
            "accuracy":     accuracy_score(y_test, y_pred),
            "precision":    precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall":       recall_score(   y_test, y_pred, average="macro", zero_division=0),
            "f1":           f1_score(       y_test, y_pred, average="macro", zero_division=0),
        }
        rows.append(row)
        # ── Per-class metrics ─────────────────────────────────────────────
        prec_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
        rec_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

        row_class = {"model": model_name}
        for cls_idx, cls_name in enumerate(["Control (0)", "Pathological (1)"]):
            if cls_idx < len(prec_per_class):
                row_class[f"Prec {cls_name}"] = round(float(prec_per_class[cls_idx]), 4)
                row_class[f"Recall {cls_name}"] = round(float(rec_per_class[cls_idx]), 4)
                row_class[f"F1 {cls_name}"] = round(float(f1_per_class[cls_idx]), 4)
        rows_class.append(row_class)

    df_macro = pd.DataFrame(rows).set_index("model").round(4)
    df_class = pd.DataFrame(rows_class).set_index("model")

    # ── Print per-class breakdown table ───────────────────────────────────
    print(f"\n{'=' * 90}")
    print("  Per-class metrics breakdown (full feature set)")
    print(f"{'=' * 90}")
    print(f"  {'Model':<25} {'Prec(0)':>9}  {'Rec(0)':>8}  {'F1(0)':>7}  "
          f"{'Prec(1)':>9}  {'Rec(1)':>8}  {'F1(1)':>7}")
    print("  " + "-" * 86)
    for model_name, row in df_class.iterrows():
        print(f"  {model_name:<25} "
              f"{row.get('Prec Control (0)', float('nan')):>9.4f}  "
              f"{row.get('Recall Control (0)', float('nan')):>8.4f}  "
              f"{row.get('F1 Control (0)', float('nan')):>7.4f}  "
              f"{row.get('Prec Pathological (1)', float('nan')):>9.4f}  "
              f"{row.get('Recall Pathological (1)', float('nan')):>8.4f}  "
              f"{row.get('F1 Pathological (1)', float('nan')):>7.4f}")
    print(f"{'=' * 90}")

    return df_macro
# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _build_model(config: dict):
    """
    Instantiate a model from its config dict.

    Resolves the class string (e.g. 'sklearn.svm.SVC') at runtime
    without using exec/eval on arbitrary code.

    :param config: model config dict with '_class' (dotted class path) and
                   '_params' (constructor kwargs) keys
    :return: instantiated (unfitted) estimator
    """
    class_path = config["_class"]
    module_str, class_str = class_path.rsplit(".", 1)
    module = importlib.import_module(module_str)
    cls    = getattr(module, class_str)
    return cls(**config["_params"])

def _print_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    """
    Print and return a metrics dict for a given split.

    :param y_true: true labels
    :param y_pred: predicted labels
    :param label:  label describing this split, used in the printed header
    :return: dict with precision, recall, f1 and accuracy (macro-averaged)
    """
    metrics = {
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(   y_true, y_pred, average="macro", zero_division=0),
        "f1":        f1_score(       y_true, y_pred, average="macro", zero_division=0),
        "accuracy":  accuracy_score( y_true, y_pred),
    }
    print(f"\nMetrics — {label}:")
    print(f"  {'Metric':<12} {'Value':>6}")
    print("  " + "-" * 20)
    for k, v in metrics.items():
        print(f"  {k.capitalize():<12} {v:>6.4f}")
    return metrics

