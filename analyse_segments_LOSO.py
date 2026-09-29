"""
Post-hoc per-segment temporal error analysis, aggregated across LOSO folds.

Loads the artefacts saved by loso_evaluation.py for each of the 25 LOSO
folds, predicts the segments of the held-out test subject in each fold,
and aggregates everything into a single combined per-segment prediction
table — the same structure produced by the representative-split version
(analyse_segments.py) — so that each subject's segment-level timeline
figure can be produced exactly as before, plus an error-rate summary
broken down by recording type (baseline vs long), aggregated across all
subjects.

Usage
-----
Run from the project root after loso_evaluation.py has produced
loso_results/<mode>/saved_models/:

    python analyse_segments_LOSO.py

Output
------
- Console: per-recording error-rate summary and overall performance
- error_rate_by_recording.csv — error rate and accuracy per recording type
- segment_errors_<ID>.png     — per-subject segment-error timeline figures

Available Functions
-------------------
[Public]
run
    Load the 25 saved LOSO fold artefacts and produce per-subject segment
    error figures for the model specified by ANALYSIS_MODEL, aggregated
    across all folds (each subject predicted by the fold where it was
    the held-out test subject).
summarize_errors_by_recording
    Compute the classification error rate, separately for baseline and
    long segments, aggregated across all subjects.

------------------
[Private]
_build_id_display_map
    Build a mapping from the original subject_id values to the
    thesis-facing display IDs (pathological -> ID1-15, control -> ID16-30).
_build_predictions_loso
    Walk through the 25 LOSO folds, predict the held-out subject's
    segments in each fold, and aggregate everything into a single
    DataFrame with the same structure as the representative-split
    version, so _plot_subject can be reused unchanged.
_plot_subject
    Plot a timeline of correct/incorrect segment classifications for one
    subject, split by recording (baseline / long).

------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import f1_score

# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
LOSO_SAVE_DIR = "loso_results"          # SAVE_DIR from loso_evaluation.py
SAVE_DIR = "segment_errors_LOSO"        # output folder for the per-subject figures

MODE_SLUG = "bestk_sweep"       # or "full"
ANALYSIS_MODEL = "KNN"          # must match the folder name in saved_models/

# adjust if necessary — list of class labels, in the same order as the
# 0/1 values used in the dataset
DISPLAY_LABELS = ["control", "pathologic"]

_REC_COLORS = {
    "baseline": "#eaf3fb",  # light blue
    "long": "#fef9ec",  # light amber
}
_REC_DEFAULT_COLOR = "#f5f5f5"  # fallback for other recording names

# ------------------------------------------------------------------------------------------------------------------- #
# helpers
# ------------------------------------------------------------------------------------------------------------------- #

# ── thesis-facing ID mapping ──────────────────────────────────────────────────
def _build_id_display_map(pathological_ids: list, control_ids: list) -> dict:
    """
    Build a mapping from the original subject_id values to the thesis-facing
    display IDs, following the convention pathological -> 1-15, control -> 16-30.
    """
    mapping = {}
    for display_idx, original_id in enumerate(pathological_ids, start=1):
        mapping[original_id] = f"ID{display_idx}"
    for display_idx, original_id in enumerate(control_ids, start=len(pathological_ids) + 1):
        mapping[original_id] = f"ID{display_idx}"
    return mapping


_PATHOLOGICAL_ORIGINAL_IDS = [f"ID{i}" for i in
                              [1, 2, 3, 4, 5, 6, 7, 8, 9, 21, 22, 23, 24, 25, 26]]
_CONTROL_ORIGINAL_IDS = [f"ID{i}" for i in
                         [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]]

DEFAULT_ID_DISPLAY_MAP = _build_id_display_map(
    _PATHOLOGICAL_ORIGINAL_IDS, _CONTROL_ORIGINAL_IDS
)


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def summarize_errors_by_recording(df_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the classification error rate, separately for baseline and
    long segments, aggregated across all subjects.

    :param df_pred: DataFrame produced by _build_predictions_loso(), with
                     the "recording" and "correct" columns.
    :return: summary DataFrame, one row per recording type.
    """
    summary = (
        df_pred.groupby("recording")
        .agg(
            n_segments=("correct", "size"),
            n_correct=("correct", "sum"),
        )
    )
    summary["n_errors"] = summary["n_segments"] - summary["n_correct"]
    summary["error_rate_pct"] = (summary["n_errors"] / summary["n_segments"] * 100).round(1)
    summary["accuracy_pct"] = (summary["n_correct"] / summary["n_segments"] * 100).round(1)

    # also an "overall" row, with the combined total
    overall = pd.DataFrame({
        "n_segments": [len(df_pred)],
        "n_correct": [df_pred["correct"].sum()],
    }, index=["overall"])
    overall["n_errors"] = overall["n_segments"] - overall["n_correct"]
    overall["error_rate_pct"] = (overall["n_errors"] / overall["n_segments"] * 100).round(1)
    overall["accuracy_pct"] = (overall["n_correct"] / overall["n_segments"] * 100).round(1)

    summary = pd.concat([summary, overall])

    return summary


def run() -> None:
    """
    Load the 25 saved LOSO fold artefacts and produce per-subject segment
    error figures for the model specified by ANALYSIS_MODEL, aggregated
    across all folds (each subject predicted by the fold where it was
    the held-out test subject).
    """
    print(f"\n{'=' * 55}")
    print("  Segment-level temporal error analysis — LOSO aggregated")
    print(f"  Model : {ANALYSIS_MODEL}  |  Mode: {MODE_SLUG}")
    print(f"{'=' * 55}")

    os.makedirs(SAVE_DIR, exist_ok=True)   # create the output folder before any file is saved

    df_pred = _build_predictions_loso(ANALYSIS_MODEL, MODE_SLUG)

    summary = summarize_errors_by_recording(df_pred)
    print(summary)
    summary.to_csv(os.path.join(SAVE_DIR, "error_rate_by_recording.csv"))
    baseline_segs = df_pred[df_pred["recording"] == "baseline"]
    print(baseline_segs.groupby("subject_id")["correct"].agg(["size", "sum"]))

    # ── Plot per subject ──────────────────────────────────────────────────
    display_labels = DISPLAY_LABELS
    subjects = sorted(df_pred["subject_id"].unique())

    print(f"\n[analysis] Plotting {len(subjects)} test subjects → {SAVE_DIR}/\n")
    for subject in subjects:
        df_subj = df_pred[df_pred["subject_id"] == subject].copy()
        _plot_subject(df_subj, subject, ANALYSIS_MODEL, SAVE_DIR, display_labels)

    # ── Overall summary ───────────────────────────────────────────────────
    total = len(df_pred)
    n_errors = (~df_pred["correct"]).sum()
    overall_f1 = f1_score(df_pred["label"], df_pred["y_pred"],
                          average="macro", zero_division=0)
    print(f"\n{'=' * 55}")
    print(f"  Summary — {ANALYSIS_MODEL} (LOSO, {len(subjects)} subjects)")
    print(f"  Total segments : {total}")
    print(f"  Errors         : {n_errors} ({n_errors / total * 100:.1f}%)")
    print(f"  Overall F1     : {overall_f1:.4f}")
    print(f"{'=' * 55}")


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _build_predictions_loso(model_name: str, mode_slug: str) -> pd.DataFrame:
    """
    Walk through the 25 LOSO folds, predict the held-out subject's
    segments in each fold, and aggregate everything into a single
    DataFrame with the same structure as the representative-split
    version (_build_predictions), so that _plot_subject can be reused
    without changes.

    :param model_name: model name (must match the folder name in saved_models/)
    :param mode_slug:  feature mode used during training ('full' or 'bestk_sweep')
    :return: DataFrame with columns:
               subject_id | label | y_pred | correct |
               recording | start_time_sec | rel_time_min | segment_idx
    """
    models_root = os.path.join(LOSO_SAVE_DIR, mode_slug, "saved_models")
    shared_root = os.path.join(models_root, "_shared")
    model_dir   = os.path.join(models_root, model_name)

    fold_dirs = sorted(
        d for d in os.listdir(shared_root)
        if os.path.isdir(os.path.join(shared_root, d))
    )

    all_rows = []

    for fold_dir in fold_dirs:
        fold_tag = fold_dir  # e.g. "fold_ID3"
        test_subject = fold_tag.replace("fold_", "")

        pipeline_path = os.path.join(model_dir, f"{fold_tag}_pipeline.joblib")
        if not os.path.exists(pipeline_path):
            print(f"  [SKIP] {fold_tag}: pipeline not found")
            continue

        pipeline = joblib.load(pipeline_path)
        columns  = joblib.load(os.path.join(shared_root, fold_dir, "columns.joblib"))
        testdata = joblib.load(os.path.join(shared_root, fold_dir, "testdata.joblib"))
        X_test_fold, y_test_fold = testdata["X_test"], testdata["y_test"]

        X_test_raw = X_test_fold[columns].values.astype(float)
        y_pred = pipeline.predict(X_test_raw)

        df_fold = pd.DataFrame({
            "subject_id": test_subject,
            "label": y_test_fold,
            "y_pred": y_pred,
        })
        df_fold["correct"] = (df_fold["y_pred"] == df_fold["label"])

        # temporal metadata / recording type — present as real features, just
        # like in the original script
        for col in ["start_time_sec_feature", "is_long"]:
            if col in X_test_fold.columns:
                df_fold[col] = X_test_fold[col].values

        all_rows.append(df_fold)

    df_pred = pd.concat(all_rows, axis=0, ignore_index=True)

    # ── same post-processing as the original script ────────────────────────
    if "start_time_sec_feature" in df_pred.columns:
        df_pred["start_time_sec"] = df_pred["start_time_sec_feature"]

    if "recording" not in df_pred.columns and "is_long" in df_pred.columns:
        df_pred["recording"] = df_pred["is_long"].map({0: "baseline", 1: "long"})

    if "start_time_sec" in df_pred.columns and "recording" in df_pred.columns:
        df_pred["rel_time_min"] = df_pred["start_time_sec"] / 60
    else:
        df_pred["rel_time_min"] = None

    df_pred["segment_idx"] = df_pred.groupby("subject_id").cumcount()

    return df_pred.reset_index(drop=True)


def _plot_subject(
        df_subj: pd.DataFrame,
        subject: str,
        model_name: str,
        save_dir: str,
        display_labels: List[str],
        id_display_map: Dict[str, str] = None,
) -> None:
    """
    Plot a timeline of correct/incorrect segment classifications for one subject.

    Each recording (baseline / long) is drawn as a separate row (subplot) with
    a coloured background that identifies the file type.  Within each row the
    x-axis shows time in minutes relative to the start of that recording
    (always starts at 0), so gaps in the original timeline are removed and
    recordings are directly comparable in length.

    Bars:
      green → correctly classified
      red   → misclassified
      label inside bar → true class
      label below bar  → predicted class (only when wrong)

    :param df_subj:        rows from df_pred for this subject
    :param subject:        subject ID string (used in title and filename)
    :param model_name:     model name (used in title)
    :param save_dir:       directory where the figure is saved
    :param display_labels: list of class label strings e.g. ['control', 'pathologic']
    :param id_display_map: optional dict mapping the original subject_id
                           values to the thesis-facing IDs. If not provided,
                           DEFAULT_ID_DISPLAY_MAP is used.
    """
    if id_display_map is None:
        id_display_map = DEFAULT_ID_DISPLAY_MAP

    subject_display_id = id_display_map.get(subject, subject)

    has_recording = (
            "recording" in df_subj.columns
            and df_subj["recording"].notna().all()
    )
    has_rel_time = (
            "rel_time_min" in df_subj.columns
            and df_subj["rel_time_min"].notna().all()
    )

    n_segs = len(df_subj)
    n_errors = (~df_subj["correct"]).sum()
    true_label = int(df_subj["label"].mode()[0])
    true_class = display_labels[true_label]

    if has_recording:
        recordings = sorted(df_subj["recording"].unique())
    else:
        recordings = ["all"]

    n_rows = len(recordings)

    # ── figure width based on the actual time range, not segment count ────
    if has_rel_time:
        max_time_min = df_subj["rel_time_min"].max()
        fig_width = float(np.clip(max_time_min * 0.35, 10, 24))
    else:
        fig_width = float(np.clip(n_segs * 0.5, 10, 24))

    # ── font sizes, scaled up so they survive being shrunk to \textwidth ──
    FS_TITLE = 24
    FS_AXIS_LABEL = 22
    FS_ANNOTATION = 22
    FS_TICK = 22
    FS_LEGEND = 22

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(fig_width, 2.2 * n_rows + 1.2),
        squeeze=False,
    )

    bar_width = 9.0 if has_rel_time else 0.8

    for row_idx, rec_name in enumerate(recordings):
        ax = axes[row_idx, 0]

        if has_recording:
            df_rec = df_subj[df_subj["recording"] == rec_name].copy()
        else:
            df_rec = df_subj.copy()

        sort_col = "rel_time_min" if has_rel_time else "segment_idx"
        df_rec = df_rec.sort_values(sort_col).reset_index(drop=True)

        # Recording background
        rec_key = rec_name.split("_")[0] if "_" in rec_name else rec_name
        bg_color = _REC_COLORS.get(rec_key, _REC_DEFAULT_COLOR)
        ax.set_facecolor(bg_color)

        n_rec = len(df_rec)
        n_err_rec = (~df_rec["correct"]).sum()

        for _, row in df_rec.iterrows():
            x = row["rel_time_min"] if has_rel_time else row["segment_idx"]
            color = "#2ecc71" if row["correct"] else "#e74c3c"

            ax.bar(x, 1, width=bar_width, color=color,
                   edgecolor="white", linewidth=0.6, align="edge")

        if has_rel_time and len(df_rec) > 0:
            x_max = df_rec["rel_time_min"].max() + bar_width + 2
        else:
            x_max = n_rec * bar_width + 1

        ax.set_xlim(0, x_max)
        ax.set_ylim(-0.1, 1.25)
        ax.set_yticks([])

        # Recording label on the left
        ax.set_ylabel(rec_name, fontsize=FS_AXIS_LABEL, rotation=0,
                      ha="right", va="center", labelpad=10)

        ax.set_xlabel(
            "Time (min)" if has_rel_time else "Segment index",
            fontsize=FS_AXIS_LABEL,
        )
        ax.tick_params(axis="both", labelsize=FS_TICK)

        # Per-recording annotation — without F1
        ax.text(
            0.01, 1.05,
            f"{n_rec} seg  |  {n_err_rec} error(s)",
            transform=ax.transAxes,
            fontsize=FS_ANNOTATION, va="bottom", color="#555555",
        )

        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)

    # Legend
    bg_patches = [
        mpatches.Patch(facecolor=v, edgecolor="#aaaaaa", label=k)
        for k, v in _REC_COLORS.items()
    ]
    fig.legend(
        handles=[
                    mpatches.Patch(color="#2ecc71", label="Correct"),
                    mpatches.Patch(color="#e74c3c", label="Error"),
                ],
        loc="upper right", bbox_to_anchor=(0.99, 0.995),
        fontsize=FS_LEGEND, framealpha=0.8, ncol=2,
    )

    # Title — class in title, no F1
    fig.suptitle(
        f"Participant {subject_display_id} ({true_class})  —  {model_name} (LOSO)\n"
        f"{n_segs} total of segments  |  {n_errors} error(s)",
        fontsize=FS_TITLE, y=1.02, x=0.42, ha="center",
    )

    plt.tight_layout()
    path = os.path.join(save_dir, f"segment_errors_{subject_display_id}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {path}  ({n_segs} segs, {n_errors} err)")


# ------------------------------------------------------------------------------------------------------------------- #
# entry point
# ------------------------------------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    run()