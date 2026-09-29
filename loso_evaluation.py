"""
Leave-One-Subject-Out (LOSO) evaluation.

For each of the 25 subjects, trains on the remaining 24 and evaluates
on the left-out subject. This produces one prediction per subject,
yielding a robust estimate of generalisation to unseen subjects that
uses all available data for both training and evaluation.

Procedure
---------
For each subject i (i = 1 ... N_subjects):
  1. Train set = all segments from the other 24 subjects
  2. Test set  = all segments from subject i
  3. [bestk_sweep only] prepare_features (variance + correlation filter)
     on train set only
  4. [bestk_sweep only] Sweep k=2..MAX_K_SWEEP to find optimal k via CV
  5. GridSearchCV with StratifiedGroupKFold (N_CV_SPLITS × N_CV_REPEATS)
     to select best hyperparameters — evaluated on train set only
  6. Re-train model with best hyperparameters on full train set
  7. Predict on subject i's segments
  8. Majority vote across segments → subject-level prediction

Both modes (full features and SelectKBest sweep) are run automatically.
Final metrics are computed at subject level and segment level.

Usage
-----
Run from the project root after main.py has produced
saved_models/df_features.joblib:

    python loso_evaluation.py

Outputs (per mode, in loso_results/full/ and loso_results/bestk_sweep/)
------------------------------------------------------------------------
- Console: per-subject predictions and aggregate metrics
- loso_<model>.csv           — per-subject results
- features_<model>.csv       — feature selection frequency (bestk_sweep only)
- cm_subj_<model>.png        — confusion matrix at subject level
- cm_seg_<model>.png         — confusion matrix at segment level
- loso_summary.csv           — aggregate metrics for all models (both levels)
loso_results/
- loso_summary_all_modes.csv — combined summary for both modes

Available Functions
-------------------
[Public]
run
    Run LOSO evaluation for all models in MODELS_DICT, for both the full
    feature set and the SelectKBest sweep modes, and save the combined
    summary CSV across both modes.

------------------
[Private]
_metrics_dict
    Compute accuracy, macro F1, and per-class precision/recall/F1 for a
    set of predictions and return them as a flat dict for CSV export.
_run_loso
    Run LOSO evaluation for all models in MODELS_DICT for one feature
    mode ('full' or 'bestk_sweep'), saving per-subject results, feature
    rankings, confusion matrices, and optionally the fitted models.
_duration_analysis
    Analyse how segment duration affects classification performance by
    binning segments by duration and computing macro F1 per bin per model.

------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              accuracy_score, confusion_matrix,
                              ConfusionMatrixDisplay)
from collections import Counter
from typing import List

# internal imports
from constants import MODELS_DICT
from ml.feature_selection import CorrelationFilter
from sklearn.impute import SimpleImputer
from ml.gridsearch import _build_model

# ------------------------------------------------------------------------------------------------------------------- #
# configuration
# ------------------------------------------------------------------------------------------------------------------- #

MODELS_DIR   = "saved_models"
SAVE_DIR     = "loso_results"

# Minimum and Maximum k to sweep in SelectKBest mode (per LOSO fold)
MIN_K_SWEEP = 1
MAX_K_SWEEP   = 20

# CV parameters (inner loop — hyperparameter selection within each LOSO fold)
N_CV_SPLITS   = 3
N_CV_REPEATS  = 5

# Majority vote weighting
# "uniform"  → each segment has equal weight (count-based)
# "duration" → weight by segment duration (requires "duration_sec" column)
VOTE_WEIGHT   = "uniform"

# Prepare features parameters (bestk_sweep mode only)
VARIANCE_THRESHOLD    = 0.0
CORRELATION_THRESHOLD = 0.85

# Both modes are always run automatically
MODES = [
    {"use_select_k": False, "slug": "full"},
    {"use_select_k": True,  "slug": "bestk_sweep"},
]

# ------------------------------------------------------------------------------------------------------------------- #
# helpers
# ------------------------------------------------------------------------------------------------------------------- #

def _metrics_dict(y_true, y_pred, model_name: str, level: str,
                  mode_slug: str, n_correct: int = None) -> dict:
    """Compute all metrics and return as a flat dict for CSV export."""
    return {
        "mode": mode_slug,
        "model": model_name,
        "level": level,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "prec_ctrl": round(precision_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0), 4),
        "rec_ctrl": round(recall_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0), 4),
        "f1_ctrl": round(f1_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0), 4),
        "prec_patho": round(precision_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0), 4),
        "rec_patho": round(recall_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0), 4),
        "f1_patho": round(f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0), 4),
        "correct": n_correct if n_correct is not None else "",
        "n_total": len(y_true),
    }


# ------------------------------------------------------------------------------------------------------------------- #
# main LOSO function
# ------------------------------------------------------------------------------------------------------------------- #

def _run_loso(
        df_features: pd.DataFrame,
        subjects: list,
        use_select_k: bool,
        mode_slug: str,
        save_models: bool = True
) -> List[dict]:
    """
    Run LOSO evaluation for all models in MODELS_DICT for one feature mode.

    :param df_features:  full feature DataFrame
    :param subjects:     sorted list of all subject IDs
    :param use_select_k: whether to use SelectKBest sweep inside pipeline
    :param mode_slug:    string label for output files ('full' or 'bestk_sweep')
    :param save_models: boolean to decide if we want to save the models
    :return:             list of summary metric dicts (for combined CSV)
    """
    save_dir = os.path.join(SAVE_DIR, mode_slug)
    os.makedirs(save_dir, exist_ok=True)

    models_save_dir = os.path.join(save_dir, "saved_models")
    os.makedirs(models_save_dir, exist_ok=True)

    n_subjects = len(subjects)

    print(f"\n{'=' * 65}")
    print(f"  LOSO — all models  [{mode_slug}]")
    print(f"  Subjects : {n_subjects}  |  Segments : {len(df_features)}")
    print(f"  Feature set: {'SelectKBest sweep k=2..' + str(MAX_K_SWEEP) if use_select_k else 'full'}")
    print(f"  CV: {N_CV_SPLITS} × {N_CV_REPEATS} = {N_CV_SPLITS * N_CV_REPEATS} folds")
    print(f"{'=' * 65}\n")

    results_per_model = {name: [] for name in MODELS_DICT}
    fold_fscores_all = []  # accumulates (feature_names, f_scores) per LOSO fold

    # ── LOSO loop ─────────────────────────────────────────────────────────
    for fold_idx, test_subject in enumerate(subjects, 1):
        print(f"\n{'=' * 65}")
        print(f"[fold {fold_idx:>2}/{n_subjects}]  Test subject: {test_subject}")
        print(f"{'=' * 65}")

        df_train = df_features[
            df_features["subject_id"] != test_subject
            ].reset_index(drop=True)
        df_test = df_features[
            df_features["subject_id"] == test_subject
            ].reset_index(drop=True)

        true_label = int(df_test["label"].mode()[0])
        group_name = "Control" if true_label == 0 else "Pathological"

        # Subject info
        n_segs = len(df_test)
        if "duration_sec" in df_test.columns:
            total_dur = df_test["duration_sec"].sum()
            print(f"  Subject: {test_subject} ({group_name})  |  "
                  f"Segments: {n_segs}  |  "
                  f"Total duration: {total_dur:.1f}s ({total_dur / 60:.1f} min)")
        else:
            print(f"  Subject: {test_subject} ({group_name})  |  Segments: {n_segs}")

        # Feature columns
        exclude = {"label", "subject_id", "_is_clean", "start_index",
                   "duration_sec", "n_segments_total"}
        feat_cols = [c for c in df_train.columns if c not in exclude]

        if use_select_k:
            imputer = SimpleImputer(strategy="median")
            X_train_clean = pd.DataFrame(
                imputer.fit_transform(df_train[feat_cols]), columns=feat_cols
            )
            X_test_clean = pd.DataFrame(
                imputer.transform(df_test[feat_cols]), columns=feat_cols
            )
        else:
            X_train_clean = df_train[feat_cols]
            X_test_clean = df_test[feat_cols]

        if fold_idx == 1:
            print(f"\n  Features entering model ({X_train_clean.shape[1]}):")
            for f in X_train_clean.columns:
                print(f"    {f}")

        y_train = df_train["label"].values
        y_test = df_test["label"].values
        subject_ids_train = df_train["subject_id"].values

        # ── Data shared across all models in this fold ────────────────────
        # X_train_clean/X_test_clean are the same for all 5 models within
        # the same fold, so they are saved ONCE per fold, outside the model
        # loop — this avoids 5 identical copies of the same files.
        if save_models:
            fold_shared_dir = os.path.join(models_save_dir, "_shared", f"fold_{test_subject}")
            os.makedirs(fold_shared_dir, exist_ok=True)
            joblib.dump(list(X_train_clean.columns), os.path.join(fold_shared_dir, "columns.joblib"))
            joblib.dump({"X_test": X_test_clean, "y_test": y_test}, os.path.join(fold_shared_dir, "testdata.joblib"))

        # Build CV splits — shared across all models for this fold
        cv_splits = []
        for rep in range(N_CV_REPEATS):
            gkf = StratifiedGroupKFold(
                n_splits=N_CV_SPLITS, shuffle=True, random_state=rep
            )
            try:
                cv_splits.extend(
                    list(gkf.split(X_train_clean.values, y_train, subject_ids_train))
                )
            except ValueError:
                pass

        if not cv_splits:
            print(f"  [SKIP] No valid CV folds for subject {test_subject}")
            continue

        # ── F-score per CV fold (for global ranking) ──────────────────────
        fold_fscores = []
        for train_idx, _ in cv_splits:
            X_fold = X_train_clean.values[train_idx]
            y_fold = y_train[train_idx]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores, _ = f_classif(X_fold, y_fold)
            fold_fscores.append(np.nan_to_num(scores, nan=0.0))
        fold_fscores_all.append(
            (list(X_train_clean.columns), np.mean(fold_fscores, axis=0))
        )

        # ── Model loop ────────────────────────────────────────────────────
        for model_name, model_config in MODELS_DICT.items():

            if use_select_k:
                # Sweep over k to find optimal number of features
                k_values = list(range(2, min(MAX_K_SWEEP, X_train_clean.shape[1]) + 1))
                best_cv_k = -1.0
                best_k = k_values[0]
                best_gs = None

                for k_candidate in k_values:
                    clf_k = _build_model(model_config)
                    pipe_k = Pipeline([
                        ("scaler", StandardScaler()),
                        ("corr", CorrelationFilter(threshold=CORRELATION_THRESHOLD)),
                        ("selector", SelectKBest(f_classif, k=k_candidate)),
                        ("clf", clf_k),
                    ])
                    pg_k = {f"clf__{key}": val
                            for key, val in model_config["_grid"].items()}
                    if "clf__n_neighbors" in pg_k:
                        min_fold = min(len(tr) for tr, _ in cv_splits)
                        safe_nk = [nk for nk in pg_k["clf__n_neighbors"]
                                   if nk < min_fold]
                        pg_k["clf__n_neighbors"] = safe_nk or [1]

                    gs_k = GridSearchCV(
                        estimator=pipe_k,
                        param_grid=pg_k if pg_k else {},
                        cv=cv_splits,
                        scoring="f1_macro",
                        n_jobs=-1,
                        refit=True,
                        error_score=np.nan,
                    )
                    gs_k.fit(X_train_clean.values, y_train)

                    if gs_k.best_score_ > best_cv_k:
                        best_cv_k = gs_k.best_score_
                        best_k = k_candidate
                        best_gs = gs_k

                gs = best_gs
                print(f"  {model_name:<25} best k={best_k}  cv F1={best_cv_k:.3f}")

            else:
                clf = _build_model(model_config)
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", clf),
                ])
                param_grid = {f"clf__{key}": val
                              for key, val in model_config["_grid"].items()}
                if "clf__n_neighbors" in param_grid:
                    min_fold = min(len(tr) for tr, _ in cv_splits)
                    safe_k = [nk for nk in param_grid["clf__n_neighbors"]
                              if nk < min_fold]
                    param_grid["clf__n_neighbors"] = safe_k or [1]

                gs = GridSearchCV(
                    estimator=pipe,
                    param_grid=param_grid if param_grid else {},
                    cv=cv_splits,
                    scoring="f1_macro",
                    n_jobs=-1,
                    refit=True,
                    error_score=np.nan,
                )
                gs.fit(X_train_clean.values, y_train)

            # Predict
            y_pred = gs.predict(X_test_clean.values)
            n_segments = len(y_pred)

            # Majority vote — uniform and duration
            cnt = Counter(y_pred)
            uniform_pred = cnt.most_common(1)[0][0]

            if "duration_sec" in df_test.columns:
                dur = {0: 0.0, 1: 0.0}
                for pred, w in zip(y_pred, df_test["duration_sec"].values):
                    dur[pred] += w
                duration_pred = max(dur, key=dur.get)
                vote_counts_duration = {k: round(v, 1) for k, v in dur.items()}
            else:
                duration_pred = uniform_pred
                vote_counts_duration = dict(cnt)

            subject_pred = duration_pred if VOTE_WEIGHT == "duration" else uniform_pred
            correct = int(subject_pred == true_label)
            seg_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

            if use_select_k and "selector" in gs.best_estimator_.named_steps:
                corr_filter = gs.best_estimator_.named_steps["corr"]
                selector = gs.best_estimator_.named_steps["selector"]
                cols_after_corr = X_train_clean.columns[corr_filter.cols_to_keep_]
                selected_feats = list(cols_after_corr[selector.get_support()])
            else:
                selected_feats = list(X_train_clean.columns)

            print(f"  {model_name:<25} "
                  f"True: {'Ctrl' if true_label == 0 else 'Path'}  "
                  f"Pred: {'Ctrl' if subject_pred == 0 else 'Path'}  "
                  f"{'✓' if correct else '✗'}  "
                  f"seg F1={seg_f1:.3f}  cv F1={gs.best_score_:.3f}")

            results_per_model[model_name].append({
                "subject_id": test_subject,
                "true_label": true_label,
                "pred_label": subject_pred,
                "correct": correct,
                "n_segments": n_segments,
                "segment_f1": round(seg_f1, 4),
                "cv_f1": round(gs.best_score_, 4),
                "best_params": str(gs.best_params_),
                "vote_uniform": str(dict(cnt)),
                "vote_duration": str(vote_counts_duration),
                "vote_method": VOTE_WEIGHT,
                "selected_feats": selected_feats,
                "y_test_segs": list(y_test),
                "y_pred_segs": list(y_pred),
                "dur_segs": list(df_test["duration_sec"].values)
                if "duration_sec" in df_test.columns else [],
            })

            if save_models:

                model_dir = os.path.join(models_save_dir, model_name.replace(' ', '_'))
                os.makedirs(model_dir, exist_ok=True)

                fold_tag = f"fold_{test_subject}"

                joblib.dump(
                    gs.best_estimator_,
                    os.path.join(model_dir, f"{fold_tag}_pipeline.joblib")
                )
                joblib.dump(
                    {
                        "test_subject": test_subject,
                        "best_params": gs.best_params_,
                        "cv_f1": gs.best_score_,
                        "selected_feats": selected_feats,
                        "mode_slug": mode_slug,
                    },
                    os.path.join(model_dir, f"{fold_tag}_meta.joblib")
                )

    # ── Aggregate metrics ─────────────────────────────────────────────────
    print(f"\n{'=' * 95}")
    print(f"  LOSO Summary [{mode_slug}] — all models  (subject-level, N={n_subjects})")
    print(f"{'=' * 95}")
    print(f"  {'Model':<25} {'Acc':>6}  {'F1 macro':>9}  "
          f"{'Prec(0)':>8}  {'Rec(0)':>7}  {'F1(0)':>6}  "
          f"{'Prec(1)':>8}  {'Rec(1)':>7}  {'F1(1)':>6}  {'Correct':>8}")
    print("  " + "-" * 95)

    summary_rows = []

    for model_name, rows in results_per_model.items():
        if not rows:
            continue

        df_res = pd.DataFrame(rows)
        y_true = df_res["true_label"].values
        y_pred_sub = df_res["pred_label"].values
        y_true_segs = np.concatenate([r["y_test_segs"] for r in rows])
        y_pred_segs = np.concatenate([r["y_pred_segs"] for r in rows])
        correct = int(df_res["correct"].sum())

        # Print subject-level
        m = _metrics_dict(y_true, y_pred_sub, model_name, "subject", mode_slug, correct)
        print(f"  {model_name:<25} {m['accuracy']:>6.4f}  {m['f1_macro']:>9.4f}  "
              f"{m['prec_ctrl']:>8.4f}  {m['rec_ctrl']:>7.4f}  {m['f1_ctrl']:>6.4f}  "
              f"{m['prec_patho']:>8.4f}  {m['rec_patho']:>7.4f}  {m['f1_patho']:>6.4f}  "
              f"{correct:>4}/{len(df_res)}")
        summary_rows.append(m)

        # Print segment-level
        ms = _metrics_dict(y_true_segs, y_pred_segs, model_name, "segment", mode_slug)
        print(f"  {'  (segment-level)':<25} {ms['accuracy']:>6.4f}  {ms['f1_macro']:>9.4f}  "
              f"{ms['prec_ctrl']:>8.4f}  {ms['rec_ctrl']:>7.4f}  {ms['f1_ctrl']:>6.4f}  "
              f"{ms['prec_patho']:>8.4f}  {ms['rec_patho']:>7.4f}  {ms['f1_patho']:>6.4f}")
        summary_rows.append(ms)
        print()

        # Feature ranking (bestk_sweep only)
        if use_select_k:
            feat_counts = Counter(feat for r in rows for feat in r["selected_feats"])
            n_folds = len(rows)
            print(f"  Feature selection frequency ({model_name}):")
            for feat, count in feat_counts.most_common(20):
                bar = "█" * int(count / n_folds * 20)
                print(f"    {feat:<50} {count:>2}/{n_folds}  {bar}")
            pd.DataFrame(
                feat_counts.most_common(), columns=["feature", "n_folds_selected"]
            ).assign(proportion=lambda d: d["n_folds_selected"] / n_folds).to_csv(
                os.path.join(save_dir, f"features_{model_name.replace(' ', '_')}.csv"),
                index=False
            )

        # Per-subject table
        print(f"\n  Per-subject ({model_name}):")
        print(f"  {'Subject':<12} {'True':<15} {'Pred':<15} "
              f"{'✓/✗':>4}  {'Seg F1':>8}  {'CV F1':>8}")
        print("  " + "-" * 65)
        for r in rows:
            true_name = "Control" if r["true_label"] == 0 else "Pathological"
            pred_name = "Control" if r["pred_label"] == 0 else "Pathological"
            mark = "✓" if r["correct"] else "✗"
            print(f"  {r['subject_id']:<12} {true_name:<15} {pred_name:<15} "
                  f"{mark:>4}  {r['segment_f1']:>8.4f}  {r['cv_f1']:>8.4f}")
        print()

        # Save per-subject CSV
        df_res.drop(columns=["selected_feats", "y_test_segs", "y_pred_segs"]).to_csv(
            os.path.join(save_dir, f"loso_{model_name.replace(' ', '_')}.csv"),
            index=False
        )

        # Confusion matrices
        for level, yt, yp, suffix, f1_val, acc_val in [
            ("Subject level", y_true, y_pred_sub, "subj",
             m["f1_macro"], m["accuracy"]),
            ("Segment level", y_true_segs, y_pred_segs, "seg",
             ms["f1_macro"], ms["accuracy"]),
        ]:
            cm = confusion_matrix(yt, yp)
            disp = ConfusionMatrixDisplay(cm, display_labels=["Control", "Pathological"])
            fig, ax = plt.subplots(figsize=(6, 5))
            disp.plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(
                f"LOSO {level} [{mode_slug}] — {model_name}\n"
                f"F1 = {f1_val:.4f}  |  Acc = {acc_val:.4f}",
                fontsize=18
            )
            ax.set_xlabel(ax.get_xlabel(), fontsize=16)
            ax.set_ylabel(ax.get_ylabel(), fontsize=16)
            ax.tick_params(axis="both", labelsize=16)
            for text in ax.texts:
                text.set_fontsize(16)
            plt.tight_layout()
            plt.savefig(
                os.path.join(save_dir, f"cm_{suffix}_{model_name.replace(' ', '_')}.png"),
                dpi=150, bbox_inches="tight"
            )
            plt.close()

    # ── Aggregated ANOVA F-score ranking across all LOSO folds ───────────
    # F-scores are averaged across all 25 LOSO folds — each fold uses a
    # different set of 24 training subjects, so the ranking reflects the
    # discriminative power of each feature across different subject
    # assignments, making it more robust than a single-split ranking.
    if fold_fscores_all:
        all_feat_names = fold_fscores_all[0][0]
        mean_fscores = np.mean([s for _, s in fold_fscores_all], axis=0)

        f_score_loso = sorted(
            zip(all_feat_names, mean_fscores),
            key=lambda x: -x[1]
        )

        print(f"\n{'=' * 65}")
        print(f"  Top 20 features by mean ANOVA F-score across all LOSO folds [{mode_slug}]")
        print(f"{'=' * 65}")
        for rank, (feat, score) in enumerate(f_score_loso[:20], 1):
            print(f"  {rank:>3}. F={score:>8.2f}  {feat}")

        pd.DataFrame(f_score_loso, columns=["feature", "mean_f_score"]).to_csv(
            os.path.join(save_dir, "loso_feature_ranking.csv"), index=False
        )
        print(f"[save] {os.path.join(save_dir, 'loso_feature_ranking.csv')}")

    # Duration-dependent performance analysis
    _duration_analysis(results_per_model, mode_slug, save_dir)

    # Save per-mode summary CSV
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(save_dir, "loso_summary.csv"), index=False
    )
    print(f"[save] {os.path.join(save_dir, 'loso_summary.csv')}")
    print(f"{'=' * 95}")

    return summary_rows


def _duration_analysis(
        results_per_model: dict,
        mode_slug: str,
        save_dir: str,
        n_bins: int = 5,
) -> None:
    """
    Analyse how segment duration affects classification performance.

    Segments are grouped into duration bins and the macro F1 is computed
    per bin for each model. This tests the hypothesis that longer segments
    (more gastric cycles) produce more reliable classifications.

    :param results_per_model: dict of per-subject result lists from LOSO loop
    :param mode_slug:         string label for output files
    :param save_dir:          output directory
    :param n_bins:            number of duration bins
    """
    # Collect all segment-level data with durations
    all_rows = []
    for model_name, rows in results_per_model.items():
        for r in rows:
            if not r["dur_segs"]:
                continue
            for y_true, y_pred, dur in zip(
                    r["y_test_segs"], r["y_pred_segs"], r["dur_segs"]
            ):
                all_rows.append({
                    "model": model_name,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "duration": dur,
                })

    if not all_rows:
        print("  [duration analysis] No duration data available — skipped")
        return

    df_seg = pd.DataFrame(all_rows)

    # Create duration bins (equal-width by default)
    df_seg["dur_bin"] = pd.cut(df_seg["duration"], bins=n_bins)
    bin_labels = sorted(df_seg["dur_bin"].unique())

    # Compute F1 per bin per model
    rows_out = []
    for model_name in results_per_model:
        df_m = df_seg[df_seg["model"] == model_name]
        for b in bin_labels:
            df_b = df_m[df_m["dur_bin"] == b]
            if len(df_b) < 2:
                continue
            f1 = f1_score(df_b["y_true"], df_b["y_pred"],
                          average="macro", zero_division=0)
            rows_out.append({
                "model": model_name,
                "dur_bin": str(b),
                "dur_mid_sec": round(b.mid, 1),
                "dur_mid_min": round(b.mid / 60, 2),
                "n_segments": len(df_b),
                "f1_macro": round(f1, 4),
            })

    df_dur = pd.DataFrame(rows_out)

    # Save CSV
    dur_csv = os.path.join(save_dir, "loso_duration_performance.csv")
    df_dur.to_csv(dur_csv, index=False)
    print(f"[save] {dur_csv}")

    # Print table
    print(f"\n  Duration-dependent performance [{mode_slug}]:")
    print(f"  {'Model':<25} " +
          "  ".join(f"{str(b)[:10]:>12}" for b in bin_labels))
    print("  " + "-" * 90)
    for model_name in results_per_model:
        row_vals = []
        for b in bin_labels:
            match = df_dur[
                (df_dur["model"] == model_name) & (df_dur["dur_bin"] == str(b))
                ]
            row_vals.append(f"{match['f1_macro'].values[0]:.4f}" if len(match) else "  N/A")
        print(f"  {model_name:<25} " + "  ".join(f"{v:>12}" for v in row_vals))

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name in results_per_model:
        df_m = df_dur[df_dur["model"] == model_name].sort_values("dur_mid_sec")
        if df_m.empty:
            continue
        ax.plot(df_m["dur_mid_min"], df_m["f1_macro"],
                marker="o", label=model_name)

    ax.set_xlabel("Segment duration (minutes)")
    ax.set_ylabel("Macro F1")
    ax.set_title(f"Duration-dependent classification performance [{mode_slug}]")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "loso_duration_performance.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"[save] {os.path.join(save_dir, 'loso_duration_performance.png')}")


# ------------------------------------------------------------------------------------------------------------------- #
# entry point
# ------------------------------------------------------------------------------------------------------------------- #

def run() -> None:
    """
    Run LOSO evaluation for all models in MODELS_DICT,
    for both full feature set and SelectKBest sweep modes.
    """
    df_features = joblib.load(os.path.join(MODELS_DIR, "df_features.joblib"))

    subjects = sorted(df_features["subject_id"].unique())
    all_summaries = []

    for mode in MODES:
        rows = _run_loso(
            df_features=df_features,
            subjects=subjects,
            use_select_k=mode["use_select_k"],
            mode_slug=mode["slug"],
        )
        all_summaries.extend(rows)

    # Save combined summary with both modes
    os.makedirs(SAVE_DIR, exist_ok=True)
    combined_path = os.path.join(SAVE_DIR, "loso_summary_all_modes.csv")
    pd.DataFrame(all_summaries).to_csv(combined_path, index=False)
    print(f"\n[save] {combined_path}")


if __name__ == "__main__":
    run()
