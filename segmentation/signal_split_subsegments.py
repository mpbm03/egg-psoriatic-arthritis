"""
Signal-level segmentation for EGG recordings.

Instead of dividing the signal into fixed windows and discarding those with too many artifacts,
this module operates on the full recording signal:

  1. Union of artifact + questionable intervals across all EGG channels.
     Any sample that is noisy in ANY channel is excluded from ALL channels,
     ensuring all channels cover the same time intervals.

  2. Splitting of the remaining clean signal into sub-segments of duration
     in [MIN_SIGNAL_SUBSEG_SEC, MAX_SIGNAL_SUBSEG_SEC]. Clean blocks shorter
     than MIN_SIGNAL_SUBSEG_SEC are discarded. Blocks longer than
     MAX_SIGNAL_SUBSEG_SEC are divided into multiple sub-segments of up to
     MAX_SIGNAL_SUBSEG_SEC each (the last may be shorter but >= MIN).

The output DataFrame has the same column structure as segment_egg_dataframe()
so the rest of the pipeline (EMD, feature extraction, ML) is unaffected.

Available Functions
-------------------
[Public]
convert_artifact_indices             - Convert a flat list of artefact intervals from the original to the downsampled sample scale
convert_artifact_indices_per_channel - Convert per-channel artefact interval dicts from the original to the downsampled sample scale
segment_by_clean_signal               - Union artefact/questionable intervals across channels, then split the remaining clean signal into variable-length sub-segments

------------------
[Private]
_union_intervals      - Merge overlapping/adjacent (start, end) intervals into a sorted, non-overlapping list, clipped to the signal length
_split_clean_blocks   - Split each contiguous clean block into sub-segments within [min_samples, max_samples], discarding blocks shorter than the minimum

------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from constants import (EGG_CHANNELS, FS_DOWNSAMPLE,
                       MIN_SIGNAL_SUBSEG_SEC, MAX_SIGNAL_SUBSEG_SEC)

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def convert_artifact_indices(
        artifact_indices: List[Tuple[int, int]],
        fs_orig: float,
        fs_down: float
) -> List[Tuple[int, int]]:
    """
    Convert artifact indices from the original signal to the downsampled signal scale.

    :param artifact_indices: List of tuples [(start, end), ...] in the original signal
    :param fs_orig: Original sampling frequency in Hz
    :param fs_down: Sampling frequency after downsampling in Hz
    :return: List of tuples [(start_down, end_down), ...] in the downsampled signal scale
    """
    ratio = fs_down / fs_orig
    return [(int(start * ratio), int(end * ratio)) for start, end in artifact_indices]


def convert_artifact_indices_per_channel(
        artifact_indices_per_channel: Dict[str, List[Tuple[int, int]]],
        fs_orig: float,
        fs_down: float,
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Convert per-channel artifact intervals from the original to the
    downsampled signal scale.

    :param artifact_indices_per_channel: {channel: List[(start, end)]} at fs_orig
    :param fs_orig: original sampling frequency (Hz)
    :param fs_down: target sampling frequency (Hz)
    :return: same structure with indices scaled to fs_down
    """
    ratio = fs_down / fs_orig
    return {
        ch: [(int(s * ratio), int(e * ratio)) for s, e in intervals]
        for ch, intervals in artifact_indices_per_channel.items()
    }

def segment_by_clean_signal(
        df_downsampled: pd.DataFrame,
        artifact_indices_down: Dict[str, List[Tuple[int, int]]],
        questionable_down: Dict[str, List[Tuple[int, int]]],
        fs: float = FS_DOWNSAMPLE,
        egg_channels: List[str] = EGG_CHANNELS,
        min_subseg_sec: float = MIN_SIGNAL_SUBSEG_SEC,
        max_subseg_sec: float = MAX_SIGNAL_SUBSEG_SEC,
) -> pd.DataFrame:
    """
    Segment the full EGG recording into clean sub-segments of variable length.

    Unlike segment_egg_dataframe(), which divides the signal into fixed windows
    and discards those with too many artifacts, this function works at the signal
    level:
      1. Unions artifact + questionable intervals across all channels
      2. Splits the remaining clean signal into sub-segments of duration
         in [min_subseg_sec, max_subseg_sec]

    The output has the same column structure as segment_egg_dataframe() so
    the rest of the pipeline (EMD, feature extraction, ML) is unaffected.
    Since no interpolation is performed, every retained sample is genuinely
    clean, so artifact_ratio_max is always 0.0 — kept in the output only for
    schema compatibility with segment_egg_dataframe() and downstream code.

    :param df_downsampled:          downsampled EGG DataFrame with TIME_COL
    :param artifact_indices_down:   dict {ch: [(start, end), ...]} in downsampled samples
    :param questionable_down:       dict {ch: [(start, end), ...]} in downsampled samples
    :param fs:                      sampling frequency in Hz
    :param egg_channels:            list of EGG channel names
    :param min_subseg_sec:          minimum sub-segment duration in seconds
    :param max_subseg_sec:          maximum sub-segment duration in seconds

    :return: DataFrame with one row per sub-segment, columns:
             segment_order | start_index | start_time_sec | end_time_sec |
             duration_sec | artifact_ratio_max | n_segments_total | <channel columns>
    """
    n_samples = len(df_downsampled)
    min_samples = int(min_subseg_sec * fs)
    max_samples = int(max_subseg_sec * fs)

    # ── Step 1: union artifact + questionable across all channels ─────────
    # Any sample noisy in any channel is excluded from all channels to ensure
    # all channels cover exactly the same time intervals in each sub-segment.
    all_intervals: List[Tuple[int, int]] = []
    for ch in egg_channels:
        all_intervals.extend(artifact_indices_down.get(ch, []))
        all_intervals.extend(questionable_down.get(ch, []))

    noise_intervals = _union_intervals(all_intervals, n_samples)

    # Extract raw signals
    signals: Dict[str, np.ndarray] = {
        ch: df_downsampled[ch].values.copy().astype(float)
        for ch in egg_channels
        if ch in df_downsampled.columns
    }

    # Build clean mask from the unioned noise intervals
    clean_mask = np.ones(n_samples, dtype=bool)
    for s, e in noise_intervals:
        s_clip = max(0, s)
        e_clip = min(n_samples, e)
        clean_mask[s_clip:e_clip] = False

    # ── Step 2: split clean blocks into sub-segments ───────────────────────
    blocks = _split_clean_blocks(clean_mask, min_samples, max_samples)

    if not blocks:
        print(f"  [signal_split] No valid sub-segments found "
              f"(min_duration={min_subseg_sec}s)")
        return pd.DataFrame()

    # ── Build output DataFrame ────────────────────────────────────────────
    # Attempt to use a time column for start/end times
    time_col = None
    for candidate in ["time", "Time", "timestamp", "t"]:
        if candidate in df_downsampled.columns:
            time_col = candidate
            break
    time_values = df_downsampled[time_col].values if time_col else None

    rows = []
    n_subsegments_total = len(blocks)

    for seg_idx, (start, end) in enumerate(blocks):
        seg_signals = {ch: signals[ch][start:end] for ch in egg_channels
                       if ch in signals}

        start_time = float(time_values[start]) if time_values is not None else start / fs
        end_time = float(time_values[end - 1]) if time_values is not None else (end - 1) / fs

        row = {
            "segment_order": seg_idx,
            "start_index": start,
            "start_time_sec": round(start_time, 3),
            "end_time_sec": round(end_time, 3),
            "duration_sec": round((end - start) / fs, 3),
            "artifact_ratio_max": 0.0,  # no interpolation is performed — every
                                        # retained sample is genuinely clean
            "n_segments_total": n_subsegments_total,
        }
        row.update(seg_signals)
        rows.append(row)

    df_out = pd.DataFrame(rows)

    print(f"  [signal_split] {n_subsegments_total} sub-segments produced "
          f"(durations: {df_out['duration_sec'].min():.0f}–"
          f"{df_out['duration_sec'].max():.0f}s)")

    return df_out


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _union_intervals(
        intervals: List[Tuple[int, int]],
        n_samples: int,
) -> List[Tuple[int, int]]:
    """
    Compute the union of a list of (start, end) intervals, clipping to
    [0, n_samples] and merging overlapping or adjacent intervals.

    :param intervals:  list of (start, end) sample index pairs
    :param n_samples:  total signal length (for clipping)
    :return:           sorted, merged list of non-overlapping intervals
    """
    if not intervals:
        return []

    clipped = []
    for s, e in intervals:
        s = max(0, s)
        e = min(n_samples, e)
        if e > s:
            clipped.append((s, e))

    if not clipped:
        return []

    clipped.sort(key=lambda x: x[0])
    merged = [clipped[0]]
    for s, e in clipped[1:]:
        prev_s, prev_e = merged[-1]
        if s <= prev_e:
            # overlapping or adjacent — extend the current interval
            merged[-1] = (prev_s, max(prev_e, e))
        else:
            merged.append((s, e))

    return merged


def _split_clean_blocks(
        clean_mask: np.ndarray,
        min_samples: int,
        max_samples: int,
) -> List[Tuple[int, int]]:
    """
    Find contiguous clean blocks in clean_mask and split each into
    sub-segments of length in [min_samples, max_samples].

    For each contiguous True block:
      - Block < min_samples → discard
      - min_samples <= block <= max_samples → one sub-segment
      - Block > max_samples → divide greedily into sub-segments of max_samples,
        keeping the last one only if it is >= min_samples

    :param clean_mask:   boolean array, True = clean sample
    :param min_samples:  minimum sub-segment length in samples
    :param max_samples:  maximum sub-segment length in samples
    :return:             list of (start, end) sample index pairs
    """
    # Find contiguous True blocks using edge detection
    padded = np.concatenate([[False], clean_mask, [False]])
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]  # rising edges  → block starts
    ends = np.where(changes == -1)[0]  # falling edges → block ends

    sub_segments = []

    for block_start, block_end in zip(starts, ends):
        block_len = block_end - block_start

        if block_len < min_samples:
            # Block too short — discard entirely
            continue

        # Divide the block into sub-segments of up to max_samples each
        pos = block_start
        while pos < block_end:
            seg_end = min(pos + max_samples, block_end)
            seg_len = seg_end - pos

            if seg_len < min_samples:
                # Merge tail into the previous sub-segment instead of discarding
                if sub_segments:
                    prev_start, _ = sub_segments[-1]
                    sub_segments[-1] = (prev_start, seg_end)
                break

            sub_segments.append((pos, seg_end))
            pos = seg_end

    return sub_segments