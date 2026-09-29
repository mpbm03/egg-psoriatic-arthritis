"""
Post-hoc per-segment temporal error analysis.

Loads the artefacts saved by main.py (models, test set, temporal metadata)
and produces, for each test subject, a figure showing which segments were
correctly or incorrectly classified along the temporal axis of the recording.

This allows visual inspection of whether imf_selection errors cluster at the
beginning, middle, or end of a recording — which may suggest signal
instability at the start of a session, systematic artefacts at a particular
time, or physiological transitions that the model does not capture.

Available Functions
-------------------
[Public]
run
    Load saved artefacts and produce per-subject segment error figures for
    the model specified by ANALYSIS_MODEL.

------------------
[Private]
_build_id_display_map
    Build a mapping from original subject_id values to thesis-facing
    display IDs (pathological -> ID1-15, control -> ID16-30).
_build_predictions
    Generate per-segment predictions and join temporal and recording
    metadata for the chosen model.
_plot_subject
    Plot a timeline of correct/incorrect segment classifications for one
    subject, split by recording (baseline / long).

------------------
"""
# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import f1_score
from typing import Dict, List

# internal imports
from constants import LABEL_MODE
from ml.plots import get_display_labels

# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = "top_splits\\seed_44"  # folder written by main.py (best_test, best_k_per_model, fitted models)
SAVED_MODELS_DIR = os.path.join(_HERE, "saved_models")  # folder written by main.py (df_features, df_all_meta)
SAVE_DIR = "segment_errors_SVM_best"  # output folder for the per-subject figures

# ── Segment error analysis ────────────────────────────────────────────────────
# Model to use in the per-segment temporal error analysis (analyse_segments.py).
# Must match exactly one key in MODELS_DICT.
# Options: "Logistic Regression", "Random Forest", "SVM", "KNN"
ANALYSIS_MODEL = "SVM"

# 'full'   → model trained with all features
# 'best_k' → model trained with the k best features
ANALYSIS_FEATURE_SET = "best_k"

_REC_COLORS = {
    "baseline": "#eaf3fb",  # light blue
    "long": "#fef9ec",  # light amber
}
_REC_DEFAULT_COLOR = "#f5f5f5"  # fallback for other recording names

# ── thesis-facing ID mapping ──────────────────────────────────────────────────
# original subject_id -> ID1-30 (pathological -> 1-15, control -> 16-30),
# matching the convention used throughout the thesis text and figures.
# NOTE: these must match the exact format of subject_id in best_test/df_pred,
# which stores IDs as strings like "ID2", "ID22", "ID37", etc.

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

def run() -> None:
    """
    Load saved artefacts and produce per-subject segment error figures
    for the model specified by ANALYSIS_MODEL in constants.py.
    """
    print(f"\n{'=' * 55}")
    print("  Segment-level temporal error analysis")
    print(f"  Model : {ANALYSIS_MODEL}  |  Features: {ANALYSIS_FEATURE_SET}")
    print(f"{'=' * 55}")

    # ── Load artefacts ────────────────────────────────────────────────────────
    print("\n[load] Reading saved artefacts …")

    # df_all_meta — fallback to df_features if not available
    meta_path = os.path.join(SAVED_MODELS_DIR, "df_all_meta.joblib")
    feat_path = os.path.join(SAVED_MODELS_DIR, "df_features.joblib")
    if os.path.exists(meta_path):
        df_all_meta = joblib.load(meta_path)
    elif os.path.exists(feat_path):
        df_features_full = joblib.load(feat_path)
        meta_cols = ["subject_id", "label", "start_time_sec", "end_time_sec",
                     "duration_sec", "recording", "segment_order_global", "_is_clean"]
        df_all_meta = df_features_full[[c for c in meta_cols
                                        if c in df_features_full.columns]]
        print("[load] df_all_meta not found — using df_features as fallback")
    else:
        raise FileNotFoundError(
            f"Neither df_all_meta.joblib nor df_features.joblib found in {SAVED_MODELS_DIR}"
        )

    best_test = joblib.load(os.path.join(MODELS_DIR, "best_test.joblib"))
    best_k_per_model = joblib.load(os.path.join(MODELS_DIR, "best_k_per_model.joblib"))

    # Validate that the chosen model exists
    if ANALYSIS_MODEL not in best_k_per_model:
        raise ValueError(
            f"ANALYSIS_MODEL='{ANALYSIS_MODEL}' not found in saved models. "
            f"Available: {list(best_k_per_model.keys())}"
        )

    # Load model according to feature set
    exclude = {"label", "subject_id", "_is_clean", "start_index",
               "duration_sec", "n_segments_total"}
    feat_cols = [c for c in best_test.columns if c not in exclude]

    if ANALYSIS_FEATURE_SET == "best_k":
        k = best_k_per_model[ANALYSIS_MODEL]
        path = os.path.join(MODELS_DIR,
                            f"{ANALYSIS_MODEL.replace(' ', '_')}_bestk{k}.joblib")
        print(f"[load] {ANALYSIS_MODEL} (best_k, k={k})  ←  {path}")
    else:
        path = os.path.join(MODELS_DIR,
                            f"{ANALYSIS_MODEL.replace(' ', '_')}_full.joblib")
        print(f"[load] {ANALYSIS_MODEL} (full features)  ←  {path}")

    fitted_model = joblib.load(path)
    fitted_models_bestk = {ANALYSIS_MODEL: fitted_model}

    # ── Build per-segment predictions ─────────────────────────────────────────
    df_pred = _build_predictions(
        best_test=best_test,
        model_name=ANALYSIS_MODEL,
        fitted_models_bestk=fitted_models_bestk,
        feat_cols=feat_cols,
    )

    # ── Plot per subject ──────────────────────────────────────────────────────
    os.makedirs(SAVE_DIR, exist_ok=True)
    display_labels = get_display_labels(LABEL_MODE)
    subjects = sorted(df_pred["subject_id"].unique())

    print(f"\n[analysis] Plotting {len(subjects)} test subjects → {SAVE_DIR}/\n")
    for subject in subjects:
        df_subj = df_pred[df_pred["subject_id"] == subject].copy()
        _plot_subject(df_subj, subject, ANALYSIS_MODEL, SAVE_DIR, display_labels)

    # ── Overall summary ───────────────────────────────────────────────────────
    total = len(df_pred)
    n_errors = (~df_pred["correct"]).sum()
    overall_f1 = f1_score(df_pred["label"], df_pred["y_pred"],
                          average="macro", zero_division=0)
    print(f"\n{'=' * 55}")
    print(f"  Summary — {ANALYSIS_MODEL}")
    print(f"  Total segments : {total}")
    print(f"  Errors         : {n_errors} ({n_errors / total * 100:.1f}%)")
    print(f"  Overall F1     : {overall_f1:.4f}")
    print(f"{'=' * 55}")


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _build_predictions(
        best_test: pd.DataFrame,
        model_name: str,
        fitted_models_bestk: Dict,
        feat_cols: List[str],
) -> pd.DataFrame:
    """
    Generate per-segment predictions and join temporal + recording metadata.

    All metadata (start_time_sec_feature, recording, duration_sec, is_long)
    is read directly from best_test — no separate df_all_meta join needed.

    :return: DataFrame with columns:
               subject_id | label | y_pred | correct |
               recording | start_time_sec | rel_time_min | segment_idx
    """
    model = fitted_models_bestk[model_name]
    y_pred = model.predict(best_test[feat_cols].values)
    y_true = best_test["label"].values

    df_pred = best_test[["subject_id", "label"]].copy().reset_index(drop=True)
    df_pred["y_pred"] = y_pred
    df_pred["correct"] = (y_pred == y_true)

    # Use start_time_sec_feature directly from best_test — no join needed
    for col in ["start_time_sec_feature", "recording", "duration_sec",
                "is_long", "segment_order_global"]:
        if col in best_test.columns:
            df_pred[col] = best_test[col].values

    # Rename start_time_sec_feature → start_time_sec for plotting consistency
    if "start_time_sec_feature" in df_pred.columns:
        df_pred["start_time_sec"] = df_pred["start_time_sec_feature"]

    # Infer recording type from is_long if recording column is missing
    if "recording" not in df_pred.columns and "is_long" in df_pred.columns:
        df_pred["recording"] = df_pred["is_long"].map({0: "baseline", 1: "long"})

    # Relative time within each recording (minutes from first segment of that file)
    if "start_time_sec" in df_pred.columns and "recording" in df_pred.columns:
        df_pred["rel_time_min"] = df_pred["start_time_sec"] / 60
    else:
        df_pred["rel_time_min"] = None

    # Segment index within subject (across all recordings)
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
    # Using n_segs to size the figure stretches recordings with few segments
    # (e.g. a short baseline) across the same width as a long recording with
    # many segments, and produces figures so wide that all text becomes
    # illegible once shrunk to fit the thesis page width. Base the width on
    # the longest time range (in minutes) across recordings instead, with a
    # sensible minimum and maximum so the figure stays a reasonable size.
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
        f"Participant {subject_display_id} ({true_class})  —  {model_name}\n"
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