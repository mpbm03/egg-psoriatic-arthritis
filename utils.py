"""
Shared utility functions used across the project's scripts.

This is a utils module: it is not run directly. Its functions are imported
and called from other .py scripts in the project (Welch PSD parameter
selection, artifact-ratio matching between groups, and result-printing
helpers for subject-level and segment-level model comparisons).

Available Functions
-------------------
[Public]
get_welch_params
    Compute the best Welch window_sec and overlap_sec for a given signal
    length, balancing frequency resolution against the number of averages.
match_artifact_ratios
    Balance artifact ratio distributions between control and pathological
    groups via nearest-neighbour matching.

------------------
[Private]
_print_comparison
    Print a side-by-side comparison table of ML vs CNN results.
_majority_vote_subjects
    Aggregate segment-level predictions to subject level via majority
    voting and print a per-subject summary.
_segment_level_comparison
    Print a per-segment comparison of predictions across all models,
    highlighting segments where models disagree or all models are wrong.

------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
from typing import Dict

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import f1_score as _f1_score

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def get_welch_params(
        n_samples: int,
        fs: float,
        target_resolution_cpm: float = 0.2,
        overlap_frac: float = 0.5,
        min_windows: int = 4,
) -> tuple:
    """
    Compute the best Welch window_sec and overlap_sec for a given signal
    length, balancing frequency resolution against the number of averages.

    Strategy
    --------
    1. Start from the window needed to achieve target_resolution_cpm.
    2. If that window is too long to fit at least min_windows (with overlap),
       shorten it progressively until it fits — degrading resolution gracefully.
    3. Return (window_sec, overlap_sec) ready to pass to welch().

    Frequency resolution = fs / nperseg (Hz) = 60 * fs / nperseg (CPM)
    → nperseg = 60 * fs / target_resolution_cpm

    :param n_samples:             number of samples in the signal
    :param fs:                    sampling frequency (Hz)
    :param target_resolution_cpm: desired frequency resolution in CPM (default 0.2)
    :param overlap_frac:          fraction of window used as overlap (default 0.5)
    :param min_windows:           minimum number of Welch averages (default 4)

    :return: (window_sec, overlap_sec) as floats
    """
    duration_sec = n_samples / fs

    # Ideal window for target resolution
    res_hz = target_resolution_cpm / 60.0
    ideal_window_sec = 1.0 / res_hz

    # Maximum window that still gives min_windows averages
    max_window_for_min_avg = duration_sec / (
            1 + (min_windows - 1) * (1 - overlap_frac)
    )

    window_sec = min(ideal_window_sec, max_window_for_min_avg)

    # Hard floor: need at least 2 cycles of the lowest frequency of interest
    # gastric lower bound ~1 CPM = 1/60 Hz → period = 60 s → need ≥ 2 cycles
    floor_sec = 2 * 60.0
    window_sec = max(window_sec, floor_sec)

    # If even the floor doesn't fit, use a fraction of the segment
    window_sec = min(window_sec, duration_sec * 0.5)
    window_sec = max(window_sec, duration_sec / min_windows)

    overlap_sec = window_sec * overlap_frac

    return window_sec, overlap_sec

def match_artifact_ratios(
        df_all: pd.DataFrame,
        tolerance: float = 0.02,
) -> pd.DataFrame:
    """
    Balance artifact ratio distributions between control and pathological
    groups via nearest-neighbour matching.

    For each pathological segment, the closest unmatched control segment
    in terms of artifact_ratio_max is found (within a tolerance). Control
    segments without a pathological match are discarded, equalising the
    signal quality distribution between groups.

    This ensures that any imf_selection difference observed after matching
    cannot be attributed to systematic differences in signal quality between
    groups — a potential methodological confound identified by the
    Mann-Whitney U test in analyse_artifacts.py.

    :param df_all:    segmented DataFrame with artifact_ratio_max and label columns
    :param tolerance: maximum allowed difference in artifact_ratio_max between
                      a pathological and a matched control segment (default 0.02)
    :return: DataFrame with matched segments only
    """
    df_control = df_all[df_all["label"] == 0].copy().reset_index(drop=True)
    df_patho = df_all[df_all["label"] == 1].copy().reset_index(drop=True)

    control_ratios = df_control["artifact_ratio_max"].values.reshape(-1, 1)
    patho_ratios = df_patho["artifact_ratio_max"].values.reshape(-1, 1)

    distances = cdist(patho_ratios, control_ratios, metric="cityblock")

    matched_patho_indices = set()
    matched_control_indices = set()

    # Match each pathological to its nearest unmatched control
    for patho_idx in range(len(df_patho)):
        for ctrl_idx in np.argsort(distances[patho_idx]):
            if ctrl_idx not in matched_control_indices and \
                    distances[patho_idx, ctrl_idx] <= tolerance:
                matched_patho_indices.add(patho_idx)
                matched_control_indices.add(ctrl_idx)
                break

    df_patho_matched = df_patho.iloc[sorted(matched_patho_indices)].reset_index(drop=True)
    df_control_matched = df_control.iloc[sorted(matched_control_indices)].reset_index(drop=True)
    df_matched = pd.concat([df_patho_matched, df_control_matched], ignore_index=True)

    print(f"\n[matching] Artifact ratio matching (tolerance={tolerance}):")
    print(f"  Pathological (before) : {len(df_patho)}  →  (after) : {len(df_patho_matched)}")
    print(f"  Control      (before) : {len(df_control)}  →  (after) : {len(df_control_matched)}")
    print(f"  Total matched         : {len(df_matched)}")

    return df_matched

def _print_comparison(ml_results_df: pd.DataFrame, cnn_metrics: dict) -> None:
    """Print a side-by-side comparison table of ML vs CNN results."""
    print(f"\n{'='*65}")
    print("  ML vs CNN — Test-set comparison")
    print(f"{'='*65}")
    print(f"  {'Model':<25} {'F1':>6}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}")
    print("  " + "-" * 55)
    for model_name, row in ml_results_df.iterrows():
        print(f"  {model_name:<25} {row['f1']:>6.4f}  {row['accuracy']:>6.4f}"
              f"  {row['precision']:>6.4f}  {row['recall']:>6.4f}")
    m = cnn_metrics
    print(f"  {'CNN 1D':<25} {m['f1']:>6.4f}  {m['accuracy']:>6.4f}"
          f"  {m['precision']:>6.4f}  {m['recall']:>6.4f}")
    print(f"{'='*65}")

def _majority_vote_subjects(
        model_name:         str,
        y_pred:             np.ndarray,
        y_true:             np.ndarray,
        subject_ids:        np.ndarray,
        segment_durations:  np.ndarray = None,
) -> None:
    """
    Aggregate segment-level predictions to subject level via majority voting
    and print a per-subject summary.

    Each subject is assigned a single predicted label based on the proportion
    of its segments (or signal duration) predicted as pathological (label=1):
      - pred_label = 1  if  patho_ratio > 0.5
      - pred_label = 0  otherwise

    When segment_durations is provided, the ratio is weighted by duration —
    a 4-minute sub-segment contributes twice as much as a 2-minute one.
    This is particularly relevant when SEGMENTATION_MODE='signal' produces
    variable-length sub-segments. When segment_durations is None, all
    segments are weighted equally (original behaviour).

    :param model_name:         model name shown in the printed header
    :param y_pred:             segment-level predicted labels (0 or 1)
    :param y_true:             segment-level true labels (0 or 1)
    :param subject_ids:        subject ID per segment
    :param segment_durations:  optional duration in seconds per segment.
                               If None, equal weighting is used.
    """
    weighted = segment_durations is not None

    rows = []
    for subj in sorted(set(subject_ids)):
        mask       = subject_ids == subj
        true_label = int(y_true[mask][0])
        n_segments = int(mask.sum())

        if weighted:
            durations      = segment_durations[mask]
            total_duration = durations.sum()
            patho_duration = durations[y_pred[mask] == 1].sum()
            patho_ratio    = float(patho_duration / total_duration) if total_duration > 0 else 0.0
            # n_patho_pred shown as weighted seconds for transparency
            n_patho_pred   = round(float(patho_duration), 1)
        else:
            n_patho_pred = int((y_pred[mask] == 1).sum())
            patho_ratio  = n_patho_pred / n_segments
            total_duration = None

        pred_label = 1 if patho_ratio > 0.5 else 0
        correct    = pred_label == true_label

        rows.append({
            "subject_id":   subj,
            "true_label":   true_label,
            "pred_label":   pred_label,
            "correct":      correct,
            "n_segments":   n_segments,
            "total_duration": round(float(total_duration), 1) if total_duration is not None else None,
            "n_patho_pred": n_patho_pred,
            "patho_ratio":  round(patho_ratio, 3),
        })

    df_subj   = pd.DataFrame(rows)
    n_correct = df_subj["correct"].sum()
    n_total   = len(df_subj)
    subj_f1   = _f1_score(df_subj["true_label"], df_subj["pred_label"],
                          average="macro", zero_division=0)

    weight_note = " (duration-weighted)" if weighted else " (majority voting)"

    print(f"\n  Subject-level — {model_name}{weight_note}:")
    if weighted:
        print(f"  {'Subject':<12} {'True':>5}  {'Pred':>5}  {'Correct':>8}  "
              f"{'Segs':>5}  {'Total dur(s)':>13}  {'Patho dur(s)':>13}  {'Patho ratio':>12}")
        print("  " + "-" * 78)
        for _, row in df_subj.iterrows():
            mark = "✓" if row["correct"] else "✗"
            print(f"  {row['subject_id']:<12} {row['true_label']:>5}  {row['pred_label']:>5}  "
                  f"{mark:>8}  {row['n_segments']:>5}  {row['total_duration']:>13.1f}  "
                  f"{row['n_patho_pred']:>13}  {row['patho_ratio']:>12.3f}")
        print("  " + "-" * 78)
    else:
        print(f"  {'Subject':<12} {'True':>5}  {'Pred':>5}  {'Correct':>8}  "
              f"{'Segs':>5}  {'Patho pred':>12}  {'Patho ratio':>12}")
        print("  " + "-" * 68)
        for _, row in df_subj.iterrows():
            mark = "✓" if row["correct"] else "✗"
            print(f"  {row['subject_id']:<12} {row['true_label']:>5}  {row['pred_label']:>5}  "
                  f"{mark:>8}  {row['n_segments']:>5}  {row['n_patho_pred']:>12}  "
                  f"{row['patho_ratio']:>12.3f}")
        print("  " + "-" * 68)
    print(f"  Subject accuracy : {n_correct}/{n_total} ({n_correct/n_total*100:.1f}%)")
    print(f"  Subject F1 macro : {subj_f1:.4f}")

    print(f"\n  Per-group summary:")
    print(f"  {'Group':<15} {'Total':>6}  {'Correct':>8}  {'Accuracy':>9}")
    print("  " + "-" * 45)
    for label_val, group_df in df_subj.groupby("true_label"):
        n_total_g   = len(group_df)
        n_correct_g = group_df["correct"].sum()
        print(f"  {label_val:<15} {n_total_g:>6}  {n_correct_g:>8}  "
              f"{n_correct_g/n_total_g:>9.1%}")

def _segment_level_comparison(
        models_preds:          Dict[str, np.ndarray],
        y_true:                np.ndarray,
        subject_ids:           np.ndarray,
        display_labels:        list = None,
        segment_order_globals: np.ndarray = None,
        segment_durations:     np.ndarray = None,
) -> None:
    """
    Print a per-segment comparison of predictions across all models.
    Highlights segments where models disagree or all models are wrong.

    When segment_durations is provided, the duration of each segment is
    shown in the table and a summary analysis at the end compares the mean
    duration of correctly vs incorrectly classified segments — useful for
    detecting whether shorter sub-segments (less signal) are harder to
    classify correctly.

    :param models_preds:          dict {model_name: y_pred array}
    :param y_true:                true labels per segment
    :param subject_ids:           subject ID per segment
    :param display_labels:        class label strings
    :param segment_order_globals: temporal index per segment (optional)
    :param segment_durations:     duration in seconds per segment (optional)
    """
    model_names = list(models_preds.keys())
    show_dur    = segment_durations is not None

    print(f"\n{'=' * 75}")
    print("  Per-segment prediction comparison across models")
    print(f"{'=' * 75}")

    header  = f"  {'Subject':<10} {'Seg':>4}  {'True':>5}  "
    if show_dur:
        header += f"{'Dur(s)':>7}  "
    header += "  ".join(f"{name[:6]:>6}" for name in model_names)
    header += "  Agreement"
    print(header)
    print("  " + "-" * (80 if show_dur else 73))

    # Track per-segment correctness for the duration analysis at the end
    # A segment is considered "correct" if ALL models predict it correctly
    all_correct_mask = []

    for subj in sorted(set(subject_ids)):
        mask    = subject_ids == subj
        indices = np.where(mask)[0]

        for seg_num, idx in enumerate(indices):
            true_lbl  = int(y_true[idx])
            preds     = {name: int(models_preds[name][idx]) for name in model_names}
            n_agree   = sum(1 for p in preds.values() if p == true_lbl)
            all_same  = len(set(preds.values())) == 1
            all_right = n_agree == len(model_names)
            all_wrong = n_agree == 0

            all_correct_mask.append(all_right)

            true_str  = display_labels[true_lbl] if display_labels else str(true_lbl)
            pred_strs = "  ".join(
                f"{'✓' if p == true_lbl else '✗':>6}"
                for p in preds.values()
            )
            agree_str = f"{n_agree}/{len(model_names)}"
            flag      = " ← all wrong" if all_wrong else \
                        " ← disagree"  if not all_same else ""
            seg_label = int(segment_order_globals[idx]) \
                        if segment_order_globals is not None else seg_num + 1

            dur_str = f"{segment_durations[idx]:>7.0f}  " if show_dur else ""

            print(f"  {subj:<10} {seg_label:>4}  {true_str:>5}  "
                  f"{dur_str}{pred_strs}  {agree_str:>10}{flag}")