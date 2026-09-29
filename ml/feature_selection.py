"""
Feature selection via SelectKBest sweep.

Train and test each model using only
the top 1, 2, 3, ... up to max_k features ranked by ANOVA F-score,
to find the optimal feature subset size.

Available Functions
-------------------
[Public]
select_k_best              - Select the top-k features by ANOVA F-score, fit on X_train and applied to both sets
sweep_feature_selection    - Full sweep: impute, then for each k=1..max_k run CorrelationFilter + SelectKBest + GridSearchCV inside a CV pipeline, tracking the best k per model by CV F1
results_to_dataframe_bestk - Build a macro-metrics results DataFrame (and print a per-class breakdown) for the best-k fitted models on the test set

------------------
[Private]
CorrelationFilter (class)  - sklearn-compatible transformer that drops one feature from each pair whose |Pearson correlation| exceeds a threshold

------------------
"""
# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# internal imports
from .gridsearch import _build_model

# ------------------------------------------------------------------------------------------------------------------- #
# CorrelationFilter transformer
# ------------------------------------------------------------------------------------------------------------------- #

class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer that removes one feature from each pair
    whose absolute Pearson correlation exceeds a given threshold.

    When placed inside a Pipeline before SelectKBest, it is re-fitted
    independently on each training fold of the cross-validation — eliminating
    the leakage that occurs when the correlation filter is applied to the full
    training set before cross-validation.

    The feature to keep from each correlated pair is determined by their
    order in the input array — the first feature (lower column index) is
    kept and the second is dropped. When fitted on a pandas DataFrame,
    feature names are preserved via feature_names_in_.

    :param threshold: pairs with |r| > threshold are considered redundant (default 0.85)

    """

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold

    def fit(self, X, y=None):
        """
                Compute the pairwise correlation matrix on X and determine which
                columns to keep (cols_to_keep_).

                :param X: training features (np.ndarray or pd.DataFrame)
                :param y: ignored, present for sklearn API compatibility
                :return: self
        """
        # Accept both numpy arrays and DataFrames
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns.tolist())
            X_arr = X.values.astype(float)
        else:
            self.feature_names_in_ = None
            X_arr = np.asarray(X, dtype=float)

        n_features = X_arr.shape[1]

        # Compute correlation matrix
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = np.corrcoef(X_arr, rowvar=False)
            corr = np.abs(corr)

        # Greedily drop the second feature of each correlated pair
        # (upper triangle only, to avoid double-counting)
        to_drop = set()
        for i in range(n_features):
            if i in to_drop:
                continue
            for j in range(i + 1, n_features):
                if j in to_drop:
                    continue
                if corr[i, j] > self.threshold:
                    to_drop.add(j)

        self.cols_to_keep_ = [i for i in range(n_features) if i not in to_drop]
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.cols_to_keep_].values
        return np.asarray(X, dtype=float)[:, self.cols_to_keep_]

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_in_ is not None:
            return self.feature_names_in_[self.cols_to_keep_]
        return np.array([f"x{i}" for i in self.cols_to_keep_])


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #


def select_k_best(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: np.ndarray,
    k:       int,
) -> Tuple[np.ndarray, np.ndarray, SelectKBest]:
    """
    Select top-k features by ANOVA F-score.

    Fit on X_train only, transform both sets.

    :param X_train: pd.DataFrame  training features (after prepare_features)
    :param X_test:  pd.DataFrame  test features
    :param y_train: np.ndarray    training labels
    :param k:       int           number of features to select

    :return: (X_train_k, X_test_k) as np.ndarrays
    """
    k = min(k, X_train.shape[1])
    selector  = SelectKBest(f_classif, k=k)
    X_train_k = selector.fit_transform(X_train, y_train)
    X_test_k  = selector.transform(X_test)
    return X_train_k, X_test_k, selector


def sweep_feature_selection(
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        subject_ids_train: np.ndarray,
        models_dict: Dict[str, dict],
        max_k: int = 20,
        n_cv_splits: int = 5,
        scoring: str = "f1_macro",
        correlation_threshold: float = 0.85,
) -> Tuple[pd.DataFrame,pd.DataFrame, Dict, Dict, Dict, Dict, list]:
    """
    Full feature selection pipeline:
      1. prepare_features (variance + correlation filter) — done once
      2. For each k=1..max_k: SelectKBest top-k -> GridSearchCV -> evaluate on test

    :param X_train:               pd.DataFrame  full training feature DataFrame
    :param y_train:               np.ndarray    training labels
    :param X_test:                pd.DataFrame  full test feature DataFrame
    :param y_test:                np.ndarray    test labels
    :param subject_ids_train:     np.ndarray    subject IDs for GroupKFold
    :param models_dict:           dict          model configs from model_configs.py
    :param max_k:                 int           maximum k to evaluate. Default 20.
    :param n_cv_splits:           int           GroupKFold splits. Default 5.
    :param scoring:               str           GridSearch scoring. Default 'f1_macro'.
    :param correlation_threshold: float         passed to prepare_features. Default 0.85.

    :return:
        df_results              : pd.DataFrame  index=k, columns=model names, values=test F1
        df_cv_results           : pd.DataFrame  index=k, columns=model names, values=CV F1
        best_gs_per_model       : dict  model_name -> best GridSearchCV object
        best_Xtr_per_model      : dict  model_name -> X_train with best-k features
        best_Xte_per_model      : dict  model_name -> X_test  with best-k features
        best_features_per_model : dict  model_name -> best k features (DataFrame column order)
        f_score_ranking         : list  of (feature, f_score) tuples sorted by F-score
                                  descending — the true importance order used by SelectKBest.
                                  Saved to saved_models/f_score_ranking.joblib by main.py
                                  and loaded by shap_analysis.py for the SelectKBest KDE plot.
    """
    best_features_per_model = {name: None for name in models_dict}

    # Step 1: variance filter only — applied once before the CV loop.
    # The correlation filter is now inside the pipeline (CorrelationFilter
    # transformer) and is re-fitted independently on each CV training fold,
    # eliminating leakage from the validation fold into feature selection.
    print("\nPreparing features (variance filter only)...")


    imputer = SimpleImputer(strategy="median")
    X_train_clean = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_clean = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    nan_count = X_train.isna().sum().sum() + X_test.isna().sum().sum()
    if nan_count > 0:
        print(f"  Imputer (median): filled {nan_count} NaN values")


    max_k = min(max_k, X_train_clean.shape[1])
    k_values = list(range(1, max_k + 1))
    results = {name: [] for name in models_dict}
    results_cv = {name: [] for name in models_dict}

    # track best GridSearchCV object per model — selected by CV F1, NOT test F1.
    # Using test F1 to pick best k is data leakage: the test set must never
    # influence any modelling decision, including feature count selection.
    best_gs_per_model    = {name: None   for name in models_dict}
    best_cv_f1_per_model = {name: -1.0   for name in models_dict}
    best_k_per_model     = {name: None   for name in models_dict}
    best_Xtr_per_model   = {name: None   for name in models_dict}
    best_Xte_per_model   = {name: None   for name in models_dict}

    # ── Build CV splits once — shared across all k and all models ─────────
    # SelectKBest is now inside the pipeline so it is re-fitted on each
    # training fold independently, eliminating leakage from the validation
    # fold into the feature ranking.
    cv_splits = []
    base_seed = 42
    n_repeats = 5
    for rep in range(n_repeats):
        gkf = StratifiedGroupKFold(
            n_splits=n_cv_splits, shuffle=True, random_state=base_seed + rep
        )
        cv_splits.extend(gkf.split(X_train_clean, y_train, subject_ids_train))

    total_folds = n_cv_splits * n_repeats
    print(f"\nSweep: k={k_values[0]} to {max_k}  |  "
          f"CV: {n_cv_splits} splits × {n_repeats} repeats = {total_folds} folds\n")

    # ── CorrelationFilter — variability across folds ──────────────────────
    # Shows which features are consistently removed vs occasionally removed
    # across different subject assignments in each CV fold.
    from collections import Counter as _Counter
    dropped_per_fold = []
    correlated_with = {}  # dropped_feat → Counter of kept features it correlated with

    for train_idx, _ in cv_splits:
        X_fold = X_train_clean.values[train_idx]
        df_fold = pd.DataFrame(X_fold, columns=X_train_clean.columns)
        cf = CorrelationFilter(threshold=correlation_threshold)
        cf.fit(df_fold)

        kept_cols = list(X_train_clean.columns[cf.cols_to_keep_])

        # Compute correlation matrix for this fold
        corr = df_fold.corr().abs()

        for i, c in enumerate(X_train_clean.columns):
            if i not in cf.cols_to_keep_:
                dropped_per_fold.append(c)
                # Find which kept feature it correlates with most
                corr_with_kept = corr[c][kept_cols]
                top_feat = corr_with_kept.idxmax()
                top_val = corr_with_kept.max()
                if c not in correlated_with:
                    correlated_with[c] = _Counter()
                correlated_with[c][f"{top_feat} (r={top_val:.3f})"] += 1

    counts = _Counter(dropped_per_fold)
    total = len(cv_splits)
    if counts:
        print(f"\n  Features dropped by CorrelationFilter across {total} folds:")
        for feat, count in sorted(counts.items(), key=lambda x: -x[1]):
            bar = "█" * int(count / total * 20)
            # Most common correlated feature across folds
            top_corr = correlated_with[feat].most_common(1)[0][0]
            print(f"    {feat:<50} {count:>2}/{total}  {bar}")
            print(f"      → correlated with {top_corr}")
    else:
        print(f"\n  CorrelationFilter: no features dropped in any fold")

    for k in k_values:
        print(f"\n{'=' * 55}")
        print(f"  k = {k}")
        print(f"{'=' * 55}")

        for model_name, config in models_dict.items():
            # Build pipeline with CorrelationFilter + SelectKBest INSIDE —
            # both are re-fitted on each CV training fold to avoid leakage.
            # CorrelationFilter removes redundant features first, then
            # SelectKBest ranks the remaining by ANOVA F-score.
            clf  = _build_model(config)
            pipe = Pipeline([
                ("scaler",   StandardScaler()),
                ("corr",     CorrelationFilter(threshold=correlation_threshold)),
                ("selector", SelectKBest(f_classif, k=min(k, X_train_clean.shape[1]))),
                ("clf",      clf),
            ])
            param_grid = {f"clf__{key}": val for key, val in config["_grid"].items()}

            # KNN: cap n_neighbors to smallest fold size
            if "clf__n_neighbors" in param_grid:
                min_fold_size = min(len(tr) for tr, _ in cv_splits)
                safe_k = [nk for nk in param_grid["clf__n_neighbors"] if nk < min_fold_size]
                param_grid["clf__n_neighbors"] = safe_k or [1]

            gs = GridSearchCV(
                estimator  = pipe,
                param_grid = param_grid if param_grid else {},
                cv         = cv_splits,
                scoring    = scoring,
                n_jobs     = -1,
                refit      = True,
                error_score= "raise",
            )
            gs.fit(X_train_clean.values, y_train)

            cv_f1   = gs.best_score_
            y_pred  = gs.predict(X_test_clean.values)
            test_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

            results[model_name].append(test_f1)
            results_cv[model_name].append(cv_f1)

            print(f"  {model_name:<25} k={k:>2}  CV={cv_f1:.4f}  Test={test_f1:.4f}")

            if cv_f1 > best_cv_f1_per_model[model_name]:
                best_cv_f1_per_model[model_name] = cv_f1
                best_k_per_model[model_name]     = k
                best_gs_per_model[model_name]    = gs

                # Recover selected feature names from the fitted pipeline.
                # With CorrelationFilter in the pipeline, the selector sees
                # only the features that survived the correlation filter —
                # map back to original feature names via cols_to_keep_.
                corr_filter     = gs.best_estimator_.named_steps["corr"]
                selector        = gs.best_estimator_.named_steps["selector"]
                cols_after_corr = X_train_clean.columns[corr_filter.cols_to_keep_]
                selected_feats  = list(cols_after_corr[selector.get_support()])
                best_features_per_model[model_name] = selected_feats

                # Store original (pre-pipeline) arrays — the full pipeline
                # applies scaler + corr + selector internally on predict/transform
                best_Xtr_per_model[model_name] = X_train_clean.values
                best_Xte_per_model[model_name] = X_test_clean.values

    df_results = pd.DataFrame(results, index=k_values)
    df_results.index.name = "n_features"

    df_cv_results = pd.DataFrame(results_cv, index=k_values)
    df_cv_results.index.name = "n_features"

    print(f"\n{'=' * 55}")
    print("  Best k per model  (selected by CV F1 — test F1 shown for reference)")
    print(f"{'=' * 55}")
    for name in models_dict:
        best_k = best_k_per_model[name]
        best_cv = best_cv_f1_per_model[name]
        if best_k is not None and best_k in df_results.index:
            test_at_k = df_results.loc[best_k, name]
        else:
            test_at_k = float('nan')
        print(f"  {name:<25} best k={best_k:>2}  CV F1={best_cv:.4f}  test F1={test_at_k:.4f}")

    # ── ANOVA F-score ranking — averaged across CV folds ─────────────────
    # F-scores are computed independently on each CV training fold and
    # averaged, consistent with how SelectKBest operates inside the pipeline.
    # This is more rigorous than computing on the full training set, as it
    # reflects the discriminative power of each feature across different
    # subject assignments rather than a single global estimate.
    from collections import defaultdict

    scores_by_feature = defaultdict(list)

    for train_idx, _ in cv_splits:
        X_fold = X_train_clean.values[train_idx]
        y_fold = y_train[train_idx]

        df_fold = pd.DataFrame(X_fold, columns=X_train_clean.columns)
        cf = CorrelationFilter(threshold=correlation_threshold)
        cf.fit(df_fold)
        kept_cols = X_train_clean.columns[cf.cols_to_keep_]
        X_fold_filtered = df_fold[kept_cols].values

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores, _ = f_classif(X_fold_filtered, y_fold)

        for feat, score in zip(kept_cols, scores):
            scores_by_feature[feat].append(np.nan_to_num(score, nan=0.0))

    f_score_ranking = sorted(
        [
            (feat, float(np.mean(scores)), len(scores))
            for feat, scores in scores_by_feature.items()
        ],
        key=lambda x: -x[1]
    )
    # f_score_ranking entries are now (feature, mean_f_score, n_folds_present)

    max_k_needed = max(v for v in best_k_per_model.values() if v is not None)
    print(f"\n{'=' * 55}")
    print(f"  Top {max_k_needed} features by mean ANOVA F-score "
          f"after correlation filtering, across {len(cv_splits)} CV folds")
    print(f"{'=' * 55}")
    for rank, (feat, score, n_folds) in enumerate(f_score_ranking[:max_k_needed], 1):
        print(f"  {rank:>3}. F={score:>8.2f}  ({n_folds:>2}/{len(cv_splits)} folds)  {feat}")

    return df_results, df_cv_results, best_gs_per_model, best_Xtr_per_model, best_Xte_per_model, best_features_per_model, f_score_ranking


def results_to_dataframe_bestk(
    fitted_models_bestk: Dict[str, object],
    X_test_per_model:    Dict[str, np.ndarray],
    best_gs_per_model:   Dict[str, GridSearchCV],
    best_k_per_model:    Dict[str, int],
    y_test:              np.ndarray,
) -> pd.DataFrame:
    """
    Build a results DataFrame with macro-averaged metrics and a separate
    per-class breakdown DataFrame.

    Returns the macro DataFrame (as before) and also prints the per-class
    breakdown table showing Precision, Recall and F1 for each class
    separately — making it transparent where each model succeeds or fails
    (e.g. high recall on controls but low recall on pathological).

    :return: pd.DataFrame with macro metrics indexed by model name
    """
    rows       = []
    rows_class = []   # per-class breakdown

    for model_name, model in fitted_models_bestk.items():
        y_pred = model.predict(X_test_per_model[model_name])

        # ── Macro metrics (original) ──────────────────────────────────────
        row = {
            "model":       model_name,
            "best_k":      best_k_per_model[model_name],
            "best_params": best_gs_per_model[model_name].best_params_,
            "cv_score":    best_gs_per_model[model_name].best_score_,
            "accuracy":    accuracy_score(y_test, y_pred),
            "precision":   precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall":      recall_score(y_test, y_pred, average="macro", zero_division=0),
            "f1":          f1_score(y_test, y_pred, average="macro", zero_division=0),
        }
        rows.append(row)

        # ── Per-class metrics ─────────────────────────────────────────────
        prec_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
        rec_per_class  = recall_score(y_test, y_pred, average=None, zero_division=0)
        f1_per_class   = f1_score(y_test, y_pred, average=None, zero_division=0)

        row_class = {"model": model_name}
        for cls_idx, cls_name in enumerate(["Control (0)", "Pathological (1)"]):
            if cls_idx < len(prec_per_class):
                row_class[f"Prec {cls_name}"]   = round(float(prec_per_class[cls_idx]), 4)
                row_class[f"Recall {cls_name}"] = round(float(rec_per_class[cls_idx]),  4)
                row_class[f"F1 {cls_name}"]     = round(float(f1_per_class[cls_idx]),   4)
        rows_class.append(row_class)

    df_macro = pd.DataFrame(rows).set_index("model").round(4)
    df_class = pd.DataFrame(rows_class).set_index("model")

    # ── Print per-class breakdown table ───────────────────────────────────
    print(f"\n{'=' * 90}")
    print("  Per-class metrics breakdown")
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

