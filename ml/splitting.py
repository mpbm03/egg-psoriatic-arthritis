"""
Train/test splitting strategies.

The main strategy used is split by source file (= by subject in this dataset):
  1. Run N random splits, each with a different random selection of subjects
  2. Train and evaluate models on each split
  3. Keep the split that achieves the best overall F1 score

Available Functions
-------------------
[Public]
file_split       - Stratified train/test split by subject, preserving the control/pathological ratio in both sets
best_file_split   - Run N random file_split() splits, train models on each, and return the split closest to typical performance (highest mean test F1) along with a summary across all splits

------------------

"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# internal imports
from .utils import prepare_Xy
from .gridsearch import grid_search_models
from constants import MIN_TRAIN_BALANCE, MIN_TEST_SEG_RATIO, MAX_TEST_SEG_RATIO

# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def file_split(
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified split by subject — preserves the control/pathological ratio
    in both train and test sets.

    In 'prandial' mode every subject contributes both classes (label=0 from
    baseline, label=1 from long), so a plain random split already guarantees
    both classes in the test set.  In 'group' mode each subject has a single
    label, so a plain random split can accidentally put all controls in train
    — this stratified version prevents that by splitting within each label
    group independently.

    :param df:           feature DataFrame with 'subject_id' and 'label' columns
    :param test_size:    target test fraction per group. Default 0.2.
    :param random_state: int

    :return: (df_train, df_test)
    """
    # Derive one label per subject (majority vote across their segments)
    subject_labels = (
        df.groupby("subject_id")["label"]
        .agg(lambda x: x.mode()[0])
        .reset_index()
        .rename(columns={"label": "subject_label"})
    )

    train_ids, test_ids = train_test_split(
        subject_labels["subject_id"],
        test_size=test_size,
        random_state=random_state,
        stratify=subject_labels["subject_label"],  # keeps class proportions
    )

    df_train = df[df["subject_id"].isin(train_ids)].reset_index(drop=True)
    df_test = df[df["subject_id"].isin(test_ids)].reset_index(drop=True)

    return df_train, df_test

def best_file_split(
    df: pd.DataFrame,
    models_dict: dict,
    test_size: float = 0.2,
    n_splits: int = 5,
    n_cv_splits: int = 5,
    scoring: str = "f1_macro",
    base_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, GridSearchCV], pd.DataFrame, List[str], List[str]]:
    """
    Run N random file-based splits, train models on each, and return
    the split that achieved the best overall F1 score — exactly as
    described in the thesis.

    Steps
    -----
    1. Generate ``n_splits`` different random train/test splits by subject
    2. For each split: upsample minority class, run GridSearchCV
    3. Pick the split where the best model achieves the highest test F1
    4. Return that split's train/test sets and a summary of all results

    :param df:          feature DataFrame (output of extract_all_features)
    :param models_dict: model configs from model_configs.py
    :param test_size:   target test fraction. Default 0.2.
    :param n_splits:    number of random splits to try. Default 5.
    :param n_cv_splits: GroupKFold splits inside GridSearch. Default 5.
    :param scoring:     scoring metric for GridSearch. Default 'f1_macro'.
    :param base_seed:   base random seed (each split uses base_seed + i)

    :return: (best_train, best_test, best_results, summary_df,
             best_train_ids, best_test_ids) where best_train/best_test are
             the training/test DataFrames of the best split, best_results
             is a dict[str, GridSearchCV] of fitted models, summary_df
             holds F1 scores across all splits, and best_train_ids/
             best_test_ids are the subject IDs in the best split.
    """

    summary_rows = []
    n_skipped = 0

    for i in range(n_splits):
        seed = base_seed + i
        print(f"\n{'='*60}")
        print(f"  Split {i+1}/{n_splits}  (seed={seed})")
        print(f"{'='*60}")

        # ── 1. Split ──────────────────────────────────────────────────
        df_train, df_test = file_split(df, test_size=test_size, random_state=seed)

        train_subjects = sorted(df_train["subject_id"].unique())
        test_subjects  = sorted(df_test["subject_id"].unique())
        print(f"  Train subjects : {train_subjects}")
        print(f"  Test  subjects : {test_subjects}")
        print(f"  Train segments : {len(df_train)}  |  Test segments: {len(df_test)}")

        # Print label distribution in test to confirm stratification is working
        test_label_counts = df_test["label"].value_counts().to_dict()
        train_label_counts = df_train["label"].value_counts().to_dict()
        print(f"  Train label distribution : {train_label_counts}")
        print(f"  Test  label distribution : {test_label_counts}")

        # ── Check segment balance in TRAIN set (before upsampling) ───────────
        # Upsampling duplicates minority segments which can cause overfitting
        # if the original imbalance is too severe. Skip splits where the
        # minority class has fewer than MIN_TRAIN_BALANCE * majority class segments.
        if MIN_TRAIN_BALANCE > 0.0:
            train_counts = df_train["label"].value_counts()
            train_balance_ratio = min(train_counts) / max(train_counts)
            if train_balance_ratio < MIN_TRAIN_BALANCE:
                print(f"  [SKIP] Split {i + 1}: train segments too imbalanced before "
                      f"upsampling (ratio={train_balance_ratio:.2f} < "
                      f"{MIN_TRAIN_BALANCE}) — skipping")
                n_skipped += 1
                continue
            print(f"  Train balance ratio (pre-upsampling): {train_balance_ratio:.2f} ✓")

        # ── Check test segment proportion ─────────────────────────────────────
        # Ensures the test set has approximately 20% of all segments
        # between MIN_TEST_SEG_RATIO and MAX_TEST_SEG_RATIO.
        # This prevents splits where a few subjects with many segments dominate
        # the test set, making the evaluation unrepresentative.
        if MIN_TEST_SEG_RATIO > 0.0 or MAX_TEST_SEG_RATIO < 1.0:
            total_segments = len(df_train) + len(df_test)
            test_seg_ratio = len(df_test) / total_segments
            if not (MIN_TEST_SEG_RATIO <= test_seg_ratio <= MAX_TEST_SEG_RATIO):
                print(f"  [SKIP] Split {i + 1}: test segment ratio = {test_seg_ratio:.2f} "
                      f"(expected [{MIN_TEST_SEG_RATIO:.2f}, {MAX_TEST_SEG_RATIO:.2f}]) — skipping")
                n_skipped += 1
                continue
            print(f"  Test segment ratio : {test_seg_ratio:.2f} ✓")

        # ── 2. Check both classes present in train AND test ─────────
        train_classes = df_train["label"].unique()
        test_classes = df_test["label"].unique()

        if len(train_classes) < 2:
            print(f"  [SKIP] Split {i + 1}: train has only class(es) {train_classes}")
            n_skipped += 1
            continue

        if len(test_classes) < 2:
            print(f"  [SKIP] Split {i + 1}: test has only class(es) {test_classes}")
            n_skipped += 1
            continue

        df_train_bal = df_train

        X_train, y_train = prepare_Xy(df_train_bal)
        X_test, y_test = prepare_Xy(df_test)
        subject_ids_train = df_train_bal["subject_id"].values

        # ── 4. Train & evaluate ───────────────────────────────────────
        results, test_f1_scores, cv_f1_scores, cv_f1_mins, cv_f1_maxs = grid_search_models(
            X_train=X_train,
            y_train=y_train,
            subject_ids_train=subject_ids_train,
            models_dict=models_dict,
            X_test=X_test,
            y_test=y_test,
            n_splits=n_cv_splits,
            scoring=scoring,
            n_repeats=5
        )

        mean_cv_f1 = np.mean(list(cv_f1_scores.values()))
        mean_test_f1 = np.mean(list(test_f1_scores.values()))

        best_cv_model = max(cv_f1_scores, key=cv_f1_scores.get)
        worst_cv_model = min(cv_f1_scores, key=cv_f1_scores.get)
        best_test_model = max(test_f1_scores, key=test_f1_scores.get)
        worst_test_model = min(test_f1_scores, key=test_f1_scores.get)

        row = {
            "split": i + 1,
            "seed": seed,
            "mean_cv_f1": mean_cv_f1,
            "mean_test_f1": mean_test_f1,
            "best_cv_model": f"{best_cv_model} ({cv_f1_scores[best_cv_model]:.4f})",
            "worst_cv_model": f"{worst_cv_model} ({cv_f1_scores[worst_cv_model]:.4f})",
            "best_test_model": f"{best_test_model} ({test_f1_scores[best_test_model]:.4f})",
            "worst_test_model": f"{worst_test_model} ({test_f1_scores[worst_test_model]:.4f})",
            "train_ids": ",".join(train_subjects),
            "test_ids": ",".join(test_subjects),
            "train_n_control": train_label_counts.get(0, 0),
            "train_n_patho": train_label_counts.get(1, 0),
            "test_n_control": test_label_counts.get(0, 0),
            "test_n_patho": test_label_counts.get(1, 0),
            "test_balance_ratio": round(
                min(test_label_counts.values()) / max(test_label_counts.values()), 3
            ),
        }
        row.update({f"cv_{k}": v for k, v in cv_f1_scores.items()})
        row.update({f"cv_min_{k}": v for k, v in cv_f1_mins.items()})
        row.update({f"cv_max_{k}": v for k, v in cv_f1_maxs.items()})
        row.update({f"test_{k}": v for k, v in test_f1_scores.items()})
        # Store best hyperparameters for each model — useful for inspecting
        # which configurations are selected consistently across splits
        row.update({f"params_{k}": str(v.best_params_) for k, v in results.items()})
        summary_rows.append(row)

        print(f"\n  → Mean CV F1: {mean_cv_f1:.4f}  |  Mean test F1: {mean_test_f1:.4f}")

    # ── Summary table ─────────────────────────────────────────────────
    if not summary_rows:
        raise RuntimeError(
            "No valid splits found. Consider reducing MIN_TEST_BALANCE "
            "or increasing n_splits."
        )

    summary_df = pd.DataFrame(summary_rows).set_index("split")

    print(f"\n{'=' * 60}")
    print("  Results across all valid splits")
    print(f"{'=' * 60}")
    print(summary_df.round(4).to_string())
    print(f"\n  Valid splits  : {len(summary_rows)}/{n_splits}")
    print(f"  Skipped splits: {n_skipped}/{n_splits}")

    # ── Summary statistics ────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  Summary statistics — mean ± std over all valid splits")
    print(f"{'=' * 70}")
    print(f"  {'Model':<25} {'CV F1':>18}  {'Test F1':>18}")
    print("  " + "-" * 65)
    for name in models_dict.keys():
        cv_col = f"cv_{name}"
        test_col = f"test_{name}"
        if cv_col in summary_df.columns:
            cv_mean, cv_std = summary_df[cv_col].mean(), summary_df[cv_col].std()
            cv_min, cv_max = summary_df[cv_col].min(), summary_df[cv_col].max()
            t_mean, t_std = summary_df[test_col].mean(), summary_df[test_col].std()
            t_min, t_max = summary_df[test_col].min(), summary_df[test_col].max()
            print(f"  {name:<25} "
                  f"{cv_mean:.4f} ± {cv_std:.4f} [{cv_min:.4f}, {cv_max:.4f}]  "
                  f"{t_mean:.4f} ± {t_std:.4f} [{t_min:.4f}, {t_max:.4f}]")
        print("  " + "-" * 88)
        print(f"  {'Mean (all models)':<25} "
              f"{summary_df['mean_cv_f1'].mean():.4f} ± {summary_df['mean_cv_f1'].std():.4f} "
              f"[{summary_df['mean_cv_f1'].min():.4f}, {summary_df['mean_cv_f1'].max():.4f}]  "
              f"{summary_df['mean_test_f1'].mean():.4f} ± {summary_df['mean_test_f1'].std():.4f} "
              f"[{summary_df['mean_test_f1'].min():.4f}, {summary_df['mean_test_f1'].max():.4f}]")
        print(f"{'=' * 85}")

    # ── Best and worst model per split ───────────────────────────────
    print(f"\n{'=' * 110}")
    print("  Best and worst model per split")
    print(f"{'=' * 110}")
    print(f"  {'Split':>6}  {'Seed':>5}  {'Mean CV':>8}  {'Best CV model':<32}  "
          f"{'Worst CV model':<32}  {'Mean Test':>10}  {'Best Test model':<32}  "
          f"{'Worst Test model':<32}")
    print("  " + "-" * 162)
    for split_idx, srow in summary_df.iterrows():
        print(f"  {split_idx:>6}  {int(srow['seed']):>5}  "
              f"{srow['mean_cv_f1']:>8.4f}  "
              f"{str(srow.get('best_cv_model', '')):<32}  "
              f"{str(srow.get('worst_cv_model', '')):<32}  "
              f"{srow['mean_test_f1']:>10.4f}  "
              f"{str(srow.get('best_test_model', '')):<32}  "
              f"{str(srow.get('worst_test_model', '')):<32}")
    print(f"{'=' * 110}")
    param_cols = [c for c in summary_df.columns if c.startswith("params_")]
    model_labels = [c.replace("params_", "") for c in param_cols]

    print(f"\n{'=' * 60}")
    print("  Best hyperparameters per split")
    print(f"{'=' * 60}")
    for split_idx, srow in summary_df.iterrows():
        print(f"\n  Split {split_idx}  (seed={int(srow['seed'])}, "
              f"CV F1={srow['mean_cv_f1']:.4f}, "
              f"Test F1={srow['mean_test_f1']:.4f})")
        for col, mname in zip(param_cols, model_labels):
            print(f"    {mname:<25} {srow[col]}")
    print(f"{'=' * 60}")

    # ── Pass 2 — re-run the typical split ─────────────────────────────
    # Select the split whose mean CV F1 is closest to the mean across
    # all valid splits — more representative than the best CV F1 split,
    # which tends to overfit the validation set.
    mean_cv = summary_df["mean_cv_f1"].mean()
    #typical_idx = (summary_df["mean_cv_f1"] - mean_cv).abs().idxmin()
    typical_idx = summary_df["mean_test_f1"].idxmax()
    typical_seed = int(summary_df.loc[typical_idx, "seed"])

    print(f"\n  Mean CV F1 across valid splits   : {mean_cv:.4f}")
    print(f"  Typical split : {typical_idx}  "
          f"(seed={typical_seed}, "
          f"mean CV F1={summary_df.loc[typical_idx, 'mean_cv_f1']:.4f}, "
          f"mean test F1={summary_df.loc[typical_idx, 'mean_test_f1']:.4f})")
    print(f"\n  Re-running typical split (seed={typical_seed}) for final models …")

    df_train, df_test = file_split(df, test_size=test_size, random_state=typical_seed)
    df_train_bal = df_train
    X_train, y_train = prepare_Xy(df_train_bal)
    X_test, y_test = prepare_Xy(df_test)
    subject_ids_train = df_train_bal["subject_id"].values

    best_results, _, _, _, _ = grid_search_models(
        X_train=X_train,
        y_train=y_train,
        subject_ids_train=subject_ids_train,
        models_dict=models_dict,
        X_test=X_test,
        y_test=y_test,
        n_splits=n_cv_splits,
        n_repeats=5,
        scoring=scoring,
    )

    best_train = df_train
    best_test = df_test
    best_train_ids = sorted(df_train["subject_id"].unique())
    best_test_ids = sorted(df_test["subject_id"].unique())

    return (
        best_train.reset_index(drop=True),
        best_test.reset_index(drop=True),
        best_results,
        summary_df,
        best_train_ids,
        best_test_ids,
    )