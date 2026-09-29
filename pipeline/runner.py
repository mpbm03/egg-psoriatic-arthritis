"""
Orchestrates the full EGG processing pipeline for a single recording.

This module is the glue between all sub-packages. It calls each step in order
and returns a labelled DataFrame ready for feature extraction.

Steps
-----
1. Detect and replace motion artefacts
2. Centre → normalise → low-pass filter → downsample
3. Convert artefact/eating indices to downsampled scale
4. Segment into fixed-length windows
5. Apply EMD
6. Build final DataFrame with label

Available Functions
-------------------
[Public]
process_recording          - Run the full pipeline for a single recording (convenience wrapper: process_recording_to_emd + process_recording_from_emd)
process_recording_to_emd   - Steps 1-6: artefact removal, preprocessing, eating-window/segmentation, and EMD decomposition
process_recording_from_emd - Steps 7-8: build the final feature-ready DataFrame from EMD results and assign labels per LABEL_MODE

------------------
[Private]
_load_prandial_offsets   - Load per-subject meal offsets from PRANDIAL_OFFSETS_FILE
_get_prandial_offsets    - Return the cached prandial offsets dict, loading it once on first call

------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd

# internal imports
from constants import (
    FS, FS_DOWNSAMPLE,
    EGG_CHANNELS,
    BANDS_CPM, SEGMENT_MINUTES,
    LABEL_MODE, PRANDIAL_OFFSETS_FILE,
    PRANDIAL_END_MIN, PRANDIAL_START_MIN,
    SEGMENTATION_MODE, MIN_SIGNAL_SUBSEG_SEC, MAX_SIGNAL_SUBSEG_SEC
)
from data_io.artifact_loader import load_manual_artifact_indices_per_channel, load_manual_questionable_indices_per_channel

from preprocessing   import ( replace_artifacts_per_channel,
    center_egg_signals,
    normalize_egg_signals_centered,
    lowpass_filter_df,
    downsampling
)
from segmentation    import convert_artifact_indices, convert_artifact_indices_per_channel, segment_by_clean_signal
from decomposition   import apply_emd_to_segments, compute_segment_imf_frequencies
from imf_selection  import build_final_dataframe


# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def process_recording(
        recording_name: str,
        df_raw: pd.DataFrame,
        subject_id: str,
        label: int,
        subject_dir: "Path | None" = None,
        normalization_method: str = "clipping",
        threshold_factor: float = 3.0,
        min_duration_sec: int = 30,
        max_gap_sec: int = 10,
        segment_minutes: int = SEGMENT_MINUTES,
        kmeans_model=None,
        scaler_model=None,
        gastric_cluster: int = None,
        intestinal_cluster: int = None,
        feature_cols: list = None,
) -> pd.DataFrame | None:
    """
    Run the full pipeline for a single recording (steps 1-8).

    This is a convenience wrapper that calls process_recording_to_emd()
    followed by process_recording_from_emd(). Use the two-step approach
    directly in main.py when IMF_SELECTION_MODE='centroid' requires
    global K-means clustering across all subjects before building the
    final DataFrame.

    :param recording_name:       'baseline' or 'long'
    :param df_raw:                raw recording DataFrame
    :param subject_id:            subject ID string
    :param label:                 group label (used in 'group' mode)
    :param subject_dir:           Path to the subject's recording directory (optional)
    :param normalization_method:  method passed to normalize_egg_signals_centered
    :param threshold_factor:      artefact-detection threshold factor
    :param min_duration_sec:      minimum clean segment duration (seconds)
    :param max_gap_sec:           maximum gap to interpolate over (seconds)
    :param segment_minutes:       target segment length in minutes
    :param kmeans_model:          fitted KMeans — required for 'centroid' mode
    :param scaler_model:          fitted StandardScaler — required for 'centroid' mode
    :param gastric_cluster:       cluster ID for gastric band — required for 'centroid'
    :param intestinal_cluster:    cluster ID for intestinal band — required for 'centroid'
    :param feature_cols:          feature names from cluster_imfs() — required for 'centroid'
    :return: processed DataFrame ready for feature extraction, or None if
             no valid segments were produced
    """
    result = process_recording_to_emd(
        recording_name=recording_name,
        df_raw=df_raw,
        subject_id=subject_id,
        label=label,
        subject_dir=subject_dir,
        normalization_method=normalization_method,
        threshold_factor=threshold_factor,
        min_duration_sec=min_duration_sec,
        max_gap_sec=max_gap_sec,
        segment_minutes=segment_minutes,
    )
    if result is None:
        return None

    df_segments, emd_results, _recording_name, _label = result

    return process_recording_from_emd(
        df_segments        = df_segments,
        emd_results        = emd_results,
        recording_name     = _recording_name,
        subject_id         = subject_id,
        label              = _label,
        kmeans_model       = kmeans_model,
        scaler_model       = scaler_model,
        gastric_cluster    = gastric_cluster,
        intestinal_cluster = intestinal_cluster,
        feature_cols       = feature_cols,
    )


def process_recording_to_emd(
        recording_name: str,
        df_raw: pd.DataFrame,
        subject_id: str,
        label: int,
        subject_dir: "Path | None" = None,
        normalization_method: str = "clipping",
        threshold_factor: float = 3.0,
        min_duration_sec: int = 30,
        max_gap_sec: int = 10,
        segment_minutes: int = SEGMENT_MINUTES,
) -> tuple | None:
    """
    Steps 1–6 of the pipeline: preprocessing, segmentation, and EMD.

    Returns (df_segments, emd_results, recording_name, label) or None if
    no valid segments were produced or the subject is excluded in prandial
    mode.

    The returned emd_results can be collected across all subjects before
    running global K-means clustering (IMF_SELECTION_MODE='centroid'), then
    passed to process_recording_from_emd() in a second pass.

    :param recording_name:       'baseline' or 'long'
    :param df_raw:                raw recording DataFrame
    :param subject_id:            subject ID string
    :param label:                 group label (used in 'group' mode)
    :param subject_dir:           Path to the subject's recording directory (optional)
    :param normalization_method:  method passed to normalize_egg_signals_centered
    :param threshold_factor:      artefact-detection threshold factor
    :param min_duration_sec:      minimum clean segment duration (seconds)
    :param max_gap_sec:           maximum gap to interpolate over (seconds)
    :param segment_minutes:       target segment length in minutes
    :return: (df_segments, emd_results, recording_name, label), or None if
             no valid segments were produced or the subject is excluded in
             prandial mode
    """

    # ── 1. Artefact detection (original fs) ───────────────────────────────
    # Manual annotations (pathological subjects): per-channel intervals from
    # <subject_id>_[baseline_]noisy_intervals.txt.  Each channel is cleaned
    # independently — a noisy interval on CH2 does not affect CH1.
    #
    # Automatic detection (controls, or pathological without annotation file):
    # accelerometer-based detector produces a single list applied to all channels.
    questionable_indices_per_ch = {ch: [] for ch in EGG_CHANNELS}
    questionable_down = {ch: [] for ch in EGG_CHANNELS}

    artifact_indices_per_ch = load_manual_artifact_indices_per_channel(
        subject_dir, subject_id, recording_name
    )
    questionable_indices_per_ch = load_manual_questionable_indices_per_channel(
        subject_dir, subject_id, recording_name
    )
    n_total = sum(len(v) for v in artifact_indices_per_ch.values())
    print(f"  [artifacts] {subject_id}/{recording_name}: using manual annotations "
          f"({n_total} noisy interval(s) across channels)")
    df_clean = replace_artifacts_per_channel(df_raw, artifact_indices_per_ch)
    # Pass the dict directly — segment_egg_dataframe handles per-channel
    # artifact ratios and will discard segments where any channel exceeds
    # ARTIFACT_RATIO_THRESHOLD.
    artifact_indices = artifact_indices_per_ch

    # ── 2. Centre → normalise → filter → downsample ───────────────────────
    df_centered = center_egg_signals(df_clean, EGG_CHANNELS)
    df_normalised = normalize_egg_signals_centered(df_centered, EGG_CHANNELS,
                                                   method=normalization_method)
    df_filtered = lowpass_filter_df(df_normalised, EGG_CHANNELS, FS, FS_DOWNSAMPLE)
    df_downsampled = downsampling(df_filtered, FS, FS_DOWNSAMPLE)

    # ── 3. Convert artefact indices to downsampled scale ─────────────────
    # Manual path: convert per-channel dict; automatic path: convert flat list.
    # Both are passed directly to segment_egg_dataframe which handles both types.
    if isinstance(artifact_indices, dict):
        artifact_indices_down = convert_artifact_indices_per_channel(
            artifact_indices, FS, FS_DOWNSAMPLE
        )
        questionable_down = convert_artifact_indices_per_channel(
            questionable_indices_per_ch, FS, FS_DOWNSAMPLE
        )
    else:
        artifact_indices_down = convert_artifact_indices(artifact_indices, FS, FS_DOWNSAMPLE)

    # ── 3b. Merge questionable into noisy ─────────────────────────────────
    # Questionable intervals are always added to the noisy intervals before
    # segmentation so they count towards the artifact_ratio threshold used
    # to discard segments.
    if isinstance(artifact_indices_down, dict):
        for ch in EGG_CHANNELS:
            combined = artifact_indices_down.get(ch, []) + questionable_down.get(ch, [])
            combined = sorted(combined, key=lambda x: x[0])
            merged = []
            for start, end in combined:
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            artifact_indices_down[ch] = merged
        n_quest_total = sum(len(v) for v in questionable_down.values())
        print(f"  [artifacts] merged {n_quest_total} questionable interval(s) into noisy")

    # ── 4. Eating interval (group mode only) ─────────────────────────────
    # In 'group' mode the label is control vs. pathological, so all segments
    # from both baseline and long recordings are used.  However, the ~10 min
    # during which the subject is eating should be excluded: the signal is
    # contaminated by movement/chewing and does not represent a stable
    # gastric state.  The meal window [meal_offset, meal_offset + PRANDIAL_START_MIN]
    # is computed in original-scale samples (1000 Hz) and converted to the
    # downsampled scale via convert_artifact_indices — exactly the same
    # pipeline used for artefact indices — before being passed to
    # segment_egg_dataframe, which removes those samples from each window.
    #
    # In 'prandial' mode this is unnecessary: the post-prandial window
    # already starts PRANDIAL_START_MIN minutes after the meal, so the
    # eating period is never included.
    eating_indices_down = []

    if recording_name == "long":
        offsets = _get_prandial_offsets()
        sid_key = subject_id.strip().upper()
        if sid_key in offsets and not offsets[sid_key]["exclude"]:
            meal_min = offsets[sid_key]["meal_offset_min"]
            # ── Remove pre-meal window (everything before the meal) ──────
            # Segments recorded before the meal are equivalent to baseline
            # (resting state) and should not be used for group classification.
            # Remove [0, meal_offset_min] from the signal.
            pre_meal_end_orig = int(meal_min * 60 * FS)
            pre_meal_indices = [(0, pre_meal_end_orig)]
            pre_meal_down = convert_artifact_indices(pre_meal_indices, FS, FS_DOWNSAMPLE)

            # ── Remove eating window ──────────────────────────────────────
            eat_start_orig = int(meal_min * 60 * FS)
            eat_end_orig = int((meal_min + PRANDIAL_START_MIN) * 60 * FS)
            eating_indices_orig = [(eat_start_orig, eat_end_orig)]
            eating_indices_down = convert_artifact_indices(eating_indices_orig, FS, FS_DOWNSAMPLE)

            # Combine both into eating_indices_down
            eating_indices_down = pre_meal_down + eating_indices_down

            print(f"  [eating] {subject_id}/{recording_name}: removing pre-meal "
                  f"[0–{meal_min:.1f} min] and meal window "
                  f"[{meal_min:.1f}–{meal_min + PRANDIAL_START_MIN:.1f} min]")

    # ── 4. Segment ────────────────────────────────────────────────────────
    #   "signal" — signal-level splitting on the full recording.
    #      Artifact + questionable intervals are unioned across all channels,
    #      short isolated gaps are repaired by cubic interpolation, and the
    #      remaining clean signal is split into variable-length sub-segments
    #      of [MIN_SIGNAL_SUBSEG_SEC, MAX_SIGNAL_SUBSEG_SEC].
    #      Recovers more usable signal at the cost of variable durations.
    #      SPLIT_CLEAN_SUBSEGMENTS is not applied in this mode (already handled).
    if SEGMENTATION_MODE == "signal":
        df_segments = segment_by_clean_signal(
            df_downsampled=df_downsampled,
            artifact_indices_down=artifact_indices_down,
            questionable_down={},
            fs=FS_DOWNSAMPLE,
            egg_channels=EGG_CHANNELS,
            min_subseg_sec=MIN_SIGNAL_SUBSEG_SEC,
            max_subseg_sec=MAX_SIGNAL_SUBSEG_SEC,
        )
    else:
        raise ValueError(
            f"Unknown SEGMENTATION_MODE '{SEGMENTATION_MODE}'. "
            f"Expected 'signal'."
        )

    if df_segments.empty:
        print(f"  [WARN] No valid segments for {subject_id}/{recording_name}")
        return None

    # ── 5. Store artifact indices — needed before optional split ─────────
    # Both stored regardless of SPLIT_CLEAN_SUBSEGMENTS so that downstream
    # analysis tools can always access them from df_segments.
    df_segments["artifact_indices"] = [artifact_indices_down] * len(df_segments)
    df_segments["questionable_artifact_indices"] = [questionable_down] * len(df_segments)

    for orig_idx, orig_row in df_segments.iterrows():
        artifact_indices = orig_row.get("artifact_indices", {})
        questionable_indices = orig_row.get("questionable_artifact_indices", {})

        n_art = sum(len(v) for v in artifact_indices.values()) if isinstance(artifact_indices, dict) else len(
            artifact_indices)
        n_quest = sum(len(v) for v in questionable_indices.values()) if isinstance(questionable_indices, dict) else 0

        print(f"  seg {orig_idx}: {n_art} artifact intervals, {n_quest} questionable intervals")

    # ── 6. EMD ────────────────────────────────────────────────────────────
    emd_results = apply_emd_to_segments(df_segments, EGG_CHANNELS)



    for i, seg_result in enumerate(emd_results):
        for ch in EGG_CHANNELS:
            imfs = seg_result.get(f"{ch}_imfs")
            if imfs is None:
                print(f"  [DEBUG] seg {i} {ch}: no IMFs")
            else:
                print(f"  [DEBUG] seg {i} {ch}: {imfs.shape[1]} IMFs")

    return df_segments, emd_results, recording_name, label


def process_recording_from_emd(
        df_segments: pd.DataFrame,
        emd_results: list,
        recording_name: str,
        subject_id: str,
        label: int,
        kmeans_model=None,
        scaler_model=None,
        gastric_cluster: int = None,
        intestinal_cluster: int = None,
        feature_cols: list = None,
) -> pd.DataFrame | None:
    """
    Steps 7–8 of the pipeline: build final DataFrame and assign labels.

    Takes the output of process_recording_to_emd() and optionally a
    pre-trained K-means model for centroid-based IMF selection.

    :param df_segments:        output of segment_egg_dataframe
    :param emd_results:        output of apply_emd_to_segments
    :param recording_name:     'baseline' or 'long'
    :param subject_id:         subject ID string
    :param label:              group label (used in 'group' mode)
    :param kmeans_model:       fitted KMeans — required for 'centroid' mode
    :param scaler_model:       fitted StandardScaler — required for 'centroid' mode
    :param gastric_cluster:    cluster ID for gastric band — required for 'centroid'
    :param intestinal_cluster: cluster ID for intestinal band — required for 'centroid'
    :param feature_cols:       feature names from cluster_imfs() — required for 'centroid'

    :return: pd.DataFrame or None
    """
    emd_frequencies = compute_segment_imf_frequencies(
        emd_results=emd_results,
        egg_channels=EGG_CHANNELS,
        fs=FS_DOWNSAMPLE,
    )

    # ── 7. Build final DataFrame ──────────────────────────────────────────
    df_processed = build_final_dataframe(
        df_segments=df_segments,
        emd_results=emd_results,
        fs_downsample=FS_DOWNSAMPLE,
        bands_cpm=BANDS_CPM,
        egg_channels=EGG_CHANNELS,
        subject_id=subject_id,
        emd_frequencies=emd_frequencies
    )

    if df_processed is None or df_processed.empty:
        print(f"  [WARN] No valid segments after IMF selection for "
              f"{subject_id}/{recording_name}")
        return None

    df_processed["recording"] = recording_name

    # ── 8. Assign label depending on LABEL_MODE ───────────────────────────
    if LABEL_MODE == "group":
        df_processed["label"] = label
    else:
        raise ValueError(
            f"Unknown LABEL_MODE '{LABEL_MODE}'. "
            f"Expected 'group'."
        )

    return df_processed
# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #


def _load_prandial_offsets() -> dict:
    """
    Load per-subject meal offsets from PRANDIAL_OFFSETS_FILE.

    Returns a dict keyed by subject_id (case-insensitive):
        {
            "ID1":  {"meal_offset_min": 220, "exclude": False},
            "ID4":  {"meal_offset_min": -120, "exclude": True},
            ...
        }

    Subjects with exclude=true are flagged so process_recording can skip
    their 'long' recording in prandial mode.

    Raises FileNotFoundError if the CSV is missing.
    """
    if not PRANDIAL_OFFSETS_FILE.exists():
        raise FileNotFoundError(
            f"Prandial offsets file not found: {PRANDIAL_OFFSETS_FILE}\n"
            f"Create it or set PRANDIAL_OFFSETS_FILE in constants.py."
        )

    df = pd.read_csv(PRANDIAL_OFFSETS_FILE)
    required = {"subject_id", "meal_offset_min", "exclude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"prandial_offsets.csv is missing columns: {missing}")

    offsets = {}
    for _, row in df.iterrows():
        sid = str(row["subject_id"]).strip().upper()
        offsets[sid] = {
            "meal_offset_min": float(row["meal_offset_min"]),
            "exclude": str(row["exclude"]).strip().lower() == "true",
        }
    return offsets


# Cache so the CSV is only read once per process
_PRANDIAL_OFFSETS: dict | None = None


def _get_prandial_offsets() -> dict:
    """
        Return the cached prandial offsets dict, loading it from
        PRANDIAL_OFFSETS_FILE via _load_prandial_offsets() once on first call.

        :return: dict keyed by subject_id, see _load_prandial_offsets()
    """
    global _PRANDIAL_OFFSETS
    if _PRANDIAL_OFFSETS is None:
        _PRANDIAL_OFFSETS = _load_prandial_offsets()
    return _PRANDIAL_OFFSETS