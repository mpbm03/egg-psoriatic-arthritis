"""
Full EGG processing and ML pipeline.

For each subject/recording found in the data directory:
  1. Load raw signal
  2. Detect and replace motion artefacts
  3. Centre and normalise EGG channels
  4. Low-pass filter and downsample
  5. Segment into x-minute windows
  6. Apply EMD and estimate IMF frequencies
  7. Reconstruct gastric + intestinal signals, assign Pathological/Control labels
  8. Feature extraction
  9. Train and test ML models

Usage
-----
Run from the project root:

    python main.py

Output
------
- Console: per-recording processing status, feature/NaN summary, model
  results tables (full feature set and SelectKBest sweep), tree-depth
  stats, majority-vote and per-segment comparison tables
- saved_models/df_features.joblib          — full extracted feature table
- saved_models/split_summary.csv           — summary of candidate train/test splits
- saved_models/df_all_meta.joblib          — metadata columns for all segments
- saved_models/best_test_ids.joblib        — subject IDs held out for testing
- saved_models/best_test.joblib            — held-out test DataFrame
- saved_models/<model>_full.joblib         — trained models (full feature set)
- saved_models/<model>_bestk<k>.joblib     — trained models (SelectKBest sweep)
- saved_models/best_features_per_model.joblib — selected features per model (bestk)
- saved_models/best_k_per_model.joblib
- saved_models/f_score_ranking.joblib      — ANOVA F-score feature ranking
- learning_curves.png / learning_curves_bestk.png
- confusion_matrices.png / confusion_matrices_bestk.png
- feature_selection_sweep.png

Available Functions
-------------------
This is a top-level script with no reusable functions — all logic runs
inside the `if __name__ == "__main__":` block below.
"""


# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd
import joblib, os

# internal imports
from data_io import load_all_data
from pipeline  import process_recording
from features import extract_all_features
from ml import (best_file_split, prepare_Xy, results_to_dataframe, plot_learning_curves,
                plot_confusion_matrices, plot_feature_selection, sweep_feature_selection, plot_confusion_matrices_bestk,
                plot_learning_curves_bestk, results_to_dataframe_bestk, get_display_labels)
from sklearn.metrics import f1_score as _f1
from utils import _majority_vote_subjects, _segment_level_comparison
# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
from constants import (DATA_DIR, EGG_CHANNELS, FS_DOWNSAMPLE, FEATURES_TO_NORMALIZE, MODELS_DICT, LABEL_MODE,
                       FEATURES_TO_DROP_SUBSTRINGS)

# ------------------------------------------------------------------------------------------------------------------- #
# suppress harmless sklearn warnings (single-class CV folds that are ignored)
# ------------------------------------------------------------------------------------------------------------------- #
import warnings
from sklearn.exceptions import FitFailedWarning

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.feature_selection._univariate_selection")
warnings.filterwarnings("ignore", category=FitFailedWarning)

if __name__ == '__main__':

    # 1. Load all recordings
    records = load_all_data(DATA_DIR)

    # 2. Process each recording
    all_processed = []

    all_emd_data = []  # list of dicts with df_segments, emd_results, meta
    all_processed = []  # list of processed DataFrames

    for rec in records:
        subject_id = rec["subject_id"]
        subject_dir = rec["subject_dir"]
        recording = rec["recording"]
        label = rec["label"]
        group = rec["group"]
        df_raw = rec["df"]

        print(f"\n{'=' * 50}")
        print(f"  {group} | {subject_id} | {recording}  (label={label})")
        print(f"{'=' * 50}")

        df_processed = process_recording(
            recording_name=recording,
            df_raw=df_raw,
            subject_id=subject_id,
            label=label,
            subject_dir=subject_dir,
        )
        if df_processed is not None and not df_processed.empty:
            all_processed.append(df_processed)
            print(f"  [OK] {len(df_processed)} segments")

    if not all_processed:
        raise RuntimeError("No recordings were successfully processed.")

    df_all = pd.concat(all_processed, ignore_index=True)

    print(f"\n{'=' * 50}")
    print(f"Pipeline complete.")
    print(f"Total segments : {len(df_all)}")
    print(f"Subjects       : {df_all['subject_id'].nunique()}")
    print(f"Label counts   : {df_all['label'].value_counts(dropna=False).to_dict()}")
    print(f"{'=' * 50}")


    # ── 3. Feature extraction ─────────────────────────────────────────────
    feature_channels = list(EGG_CHANNELS)
    df_features = extract_all_features(
        df_processed=df_all,
        egg_channels=feature_channels,
        fs=FS_DOWNSAMPLE,
        normalize_features=FEATURES_TO_NORMALIZE,
    )

    meta_cols = {"label", "subject_id", "_is_clean", "duration_sec",
                 "segment_order_global", "start_time_sec_feature", "is_long"}

    cols_to_drop = [
        c for c in df_features.columns
        if c not in meta_cols
           and any(sub in c for sub in FEATURES_TO_DROP_SUBSTRINGS)
    ]

    print(f"[drop] Removing {len(cols_to_drop)} features affected by normalisation:")
    for c in cols_to_drop:
        print(f"  {c}")

    df_features = df_features.drop(columns=cols_to_drop)
    print(f"[drop] Features remaining: {len([c for c in df_features.columns if c not in meta_cols])}")

    df_features["duration_sec"] = df_all["duration_sec"].values

    # Add temporal position features
    # In "signal" mode, segments have
    # variable durations so segment_order_global loses its meaning.
    # Instead, two alternative temporal features are added:
    #   - start_time_sec_feature : seconds from the start of the recording
    #                              file — comparable within the same recording
    #                              type across subjects (e.g. a segment starting
    #                              at 600s is always 10 minutes into the recording)
    #   - is_long                : 0 if baseline recording, 1 if long recording —
    #                              allows the model to distinguish pre/post-prandial
    #                              signal independently of other features

    if "start_time_sec" in df_all.columns:
        df_features["start_time_sec_feature"] = df_all["start_time_sec"].values
        print(f"[temporal] Added start_time_sec_feature "
              f"(seconds from start of recording, "
              f"range: {df_all['start_time_sec'].min():.0f}–"
              f"{df_all['start_time_sec'].max():.0f}s)")
    else:
        print(f"[temporal] start_time_sec not found in df_all — skipped")

    df_features["is_long"] = (df_all["recording"] == "long").astype(int).values

    # Print breakdown by recording type and class — only meaningful
    # when both baseline and long segments are present
    breakdown = df_all.groupby(["recording", "label"]).size().unstack(fill_value=0)
    n_long = (df_features["is_long"] == 1).sum()
    n_baseline = (df_features["is_long"] == 0).sum()
    print(f"[temporal] Added is_long feature (0=baseline: {n_baseline}, 1=long: {n_long})")
    print(f"[temporal] Segment breakdown by recording and class:")
    print(f"  {'Recording':<12} {'Control (0)':>12}  {'Pathological (1)':>17}  {'Total':>7}")
    print("  " + "-" * 55)
    for rec in ["baseline", "long"]:
        n_ctrl = int(breakdown.loc[rec, 0]) if rec in breakdown.index and 0 in breakdown.columns else 0
        n_patho = int(breakdown.loc[rec, 1]) if rec in breakdown.index and 1 in breakdown.columns else 0
        print(f"  {rec:<12} {n_ctrl:>12}  {n_patho:>17}  {n_ctrl + n_patho:>7}")
    print("  " + "-" * 55)
    total_ctrl = int(breakdown[0].sum()) if 0 in breakdown.columns else 0
    total_patho = int(breakdown[1].sum()) if 1 in breakdown.columns else 0
    print(f"  {'Total':<12} {total_ctrl:>12}  {total_patho:>17}  {total_ctrl + total_patho:>7}")

    print(f"Extracted features: {df_features.columns}")
    nan_cols = df_features.columns[df_features.isna().any()].tolist()
    print(f"Features with NaN ({len(nan_cols)}): {nan_cols}")
    print(df_features[nan_cols].isna().sum().sort_values(ascending=False))

    os.makedirs("saved_models", exist_ok=True)

    print("Features in df_features:")
    for feature in df_features.columns:
        print(feature)

    joblib.dump(df_features, "saved_models/df_features.joblib")
    print("[SAVE] saved_models/df_features.joblib")

    # ── 4. Random splits ──────────────────────────────────────────────────────
    df_features_for_split = df_features

    best_train, best_test, results, summary_df, best_train_ids, best_test_ids = best_file_split(
        df=df_features_for_split,
        models_dict=MODELS_DICT,
        test_size=0.2,
        n_splits=500,
        n_cv_splits=3,
        scoring="f1_macro"
    )
    # ── Save split summary ────────────────────────────────────────────────
    os.makedirs("saved_models", exist_ok=True)
    summary_path = "saved_models/split_summary.csv"
    summary_df.to_csv(summary_path, index=True)
    print(f"[SAVE] {summary_path}")

    # ── 5. Best train/test sets ──────────────────────────────────────────────────────

    best_train_bal_raw = best_train
    X_test, y_test = prepare_Xy(best_test)

    best_train_bal = best_train_bal_raw
    X_train, y_train = prepare_Xy(best_train_bal)
    subject_ids_train = best_train_bal["subject_id"].values

    # ── 6. Results - full feature set ──────────────────────────────────────────────────────

    results_df_full = results_to_dataframe(results, X_test, y_test)

    fitted_models_full = {name: gs.best_estimator_ for name, gs in results.items()}

    print(results_df_full.to_string())

    rf_model = fitted_models_full["Random Forest"].named_steps["clf"]
    depths = [tree.get_depth() for tree in rf_model.estimators_]
    print(f"Actual tree depths:")
    print(f"  Min  : {min(depths)}")
    print(f"  Max  : {max(depths)}")
    print(f"  Mean : {sum(depths) / len(depths):.1f}")

    subject_ids_test = best_test["subject_id"].values
    models_preds_full = {
        name: fitted_models_full[name].predict(X_test)
        for name in MODELS_DICT
    }

    for name, y_pred_full in models_preds_full.items():
        _majority_vote_subjects(
            model_name=name,
            y_pred=y_pred_full,
            y_true=y_test,
            subject_ids=subject_ids_test,
            segment_durations=best_test["duration_sec"].values
            if "duration_sec" in best_test.columns else None,
        )

    _segment_level_comparison(
        models_preds=models_preds_full,
        y_true=y_test,
        subject_ids=subject_ids_test,
        display_labels=get_display_labels(LABEL_MODE),
        segment_order_globals=best_test["start_time_sec_feature"].values,
        segment_durations=best_test["duration_sec"].values,
    )

    # Curves and confusion matrices — full feature set
    plot_learning_curves(fitted_models_full, X_train, y_train, subject_ids_train = subject_ids_train, save_path="learning_curves.png")
    plot_confusion_matrices(fitted_models_full, X_test, y_test, save_path="confusion_matrices.png")

    # ── 7. Feature selection sweep (k=1..max_k, GridSearchCV per k) ──────────────────────────────────────────────────────
    feature_cols = [c for c in best_train_bal.columns if c not in {"label", "subject_id", "_is_clean", "n_segments_total", "duration_sec"}]
    X_train_df = best_train_bal[feature_cols]
    X_test_df = best_test[feature_cols]

    print(f"\n[feature selection] Using SelectKBest (ANOVA F-score)")
    (df_sweep, df_cv_results, best_gs_per_model,
     X_train_per_model, X_test_per_model,
     best_features_per_model, f_score_ranking,
     ) = sweep_feature_selection(
        X_train=X_train_df,
        y_train=y_train,
        X_test=X_test_df,
        y_test=y_test,
        subject_ids_train=subject_ids_train,
        models_dict=MODELS_DICT,
        max_k=20,
        n_cv_splits=3,
        scoring="f1_macro",
        correlation_threshold=0.85,
    )

    for model, feats in best_features_per_model.items():
        print(f"\n{model} — {len(feats)} features (best k by CV):")
        for i, f in enumerate(feats, 1):
            print(f"  {i:>2}. {f}")

    plot_feature_selection(df_sweep, df_cv_results, save_path="feature_selection_sweep.png")

    # Best k per model — from CV F1 tracked inside sweep (no leakage)
    # df_sweep stores test F1 per k (for plotting only); best_gs_per_model
    # was already selected by CV F1 inside sweep_feature_selection().
    best_k_per_model = {
        name: gs.best_estimator_.named_steps["selector"].k
        for name, gs in best_gs_per_model.items()
    }
    fitted_models_bestk = {name: gs.best_estimator_ for name, gs in best_gs_per_model.items()}

    # ── Random Forest tree depths — best-k feature set ────────────────────────
    if "Random Forest" in fitted_models_bestk:
        rf_model_bestk = fitted_models_bestk["Random Forest"].named_steps["clf"]
        depths = [tree.get_depth() for tree in rf_model_bestk.estimators_]
        print(f"\nRandom Forest tree depths (best k={best_k_per_model['Random Forest']}):")
        print(f"  Min  : {min(depths)}")
        print(f"  Max  : {max(depths)}")
        print(f"  Mean : {sum(depths) / len(depths):.1f}")

    results_df_bestk = results_to_dataframe_bestk(
        fitted_models_bestk=fitted_models_bestk,
        X_test_per_model=X_test_per_model,
        best_gs_per_model=best_gs_per_model,
        best_k_per_model=best_k_per_model,
        y_test=y_test,
    )
    print(results_df_bestk.to_string())

    # Comparison table
    comparison_rows = []
    models_preds_bestk = {
        name: fitted_models_bestk[name].predict(X_test_per_model[name])
        for name in MODELS_DICT
    }

    for name in MODELS_DICT:
        best_k = best_k_per_model[name]
        y_pred_k = models_preds_bestk[name]
        f1_k = _f1(y_test, y_pred_k, average="macro", zero_division=0)
        f1_full = results_df_full.loc[name, "f1"]
        comparison_rows.append({
            "model": name,
            "best_k": best_k,
            "F1 (full)": round(f1_full, 4),
            "F1 (best k)": round(f1_k, 4),
            "Δ F1": round(f1_k - f1_full, 4),
        })
        _majority_vote_subjects(
            model_name=f"{name} (best k={best_k_per_model[name]})",
            y_pred=models_preds_bestk[name],
            y_true=y_test,
            subject_ids=subject_ids_test,
            segment_durations=best_test["duration_sec"].values
            if "duration_sec" in best_test.columns else None,
        )

    _segment_level_comparison(
        models_preds=models_preds_bestk,
        y_true=y_test,
        subject_ids=subject_ids_test,
        display_labels=get_display_labels(LABEL_MODE),
        segment_order_globals=best_test["start_time_sec_feature"].values,
        segment_durations=best_test["duration_sec"].values,
    )

    # Learning curves and confusion matrices — best k per model
    plot_learning_curves_bestk(
        fitted_models=fitted_models_bestk,
        X_train_per_model=X_train_per_model,
        y_train=y_train,
        subject_ids_train=subject_ids_train,
        best_k_per_model=best_k_per_model,
        save_path="learning_curves_bestk.png",
    )
    plot_confusion_matrices_bestk(
        fitted_models=fitted_models_bestk,
        X_test_per_model=X_test_per_model,
        y_test=y_test,
        best_k_per_model=best_k_per_model,
        save_path="confusion_matrices_bestk.png",
    )

    # ── 8. Save artefacts for post-hoc analysis ───────────────────────────

    meta_cols = [c for c in df_all.columns if c not in feature_channels]
    joblib.dump(df_all[meta_cols].copy(), "saved_models/df_all_meta.joblib")
    print("[SAVE] saved_models/df_all_meta.joblib")

    # Test IDs and DataFrame with features
    joblib.dump(best_test_ids, "saved_models/best_test_ids.joblib")
    joblib.dump(best_test, "saved_models/best_test.joblib")
    print("[SAVE] saved_models/best_test_ids.joblib")
    print("[SAVE] saved_models/best_test.joblib")

    # Trained Models - full set of features
    for name, model in fitted_models_full.items():
        path = f"saved_models/{name.replace(' ', '_')}_full.joblib"
        joblib.dump(model, path)
        print(f"[SAVE] {path}")

    # Trained Models — best-k features
    for name, model in fitted_models_bestk.items():
        k = best_k_per_model[name]
        path = f"saved_models/{name.replace(' ', '_')}_bestk{k}.joblib"
        joblib.dump(model, path)
        print(f"[SAVE] {path}")

    # Features
    joblib.dump(best_features_per_model, "saved_models/best_features_per_model.joblib")
    joblib.dump(best_k_per_model, "saved_models/best_k_per_model.joblib")

    joblib.dump(f_score_ranking, "saved_models/f_score_ranking.joblib")
    print("[SAVE] saved_models/f_score_ranking.joblib")
    print("[SAVE] saved_models/best_features_per_model.joblib")
    print("[SAVE] saved_models/best_k_per_model.joblib")

