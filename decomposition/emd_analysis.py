"""
EMD decomposition and dominant-frequency estimation per segment.

Available Functions
-------------------
[Public]
apply_emd_to_segments         - Apply EMD to every segment/channel, storing the resulting IMFs per segment
compute_segment_imf_frequencies - Estimate each IMF's dominant frequency (CPM) via median inter-peak interval, more robust to non-stationary IMFs than Welch

------------------
[Private]


------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import pandas as pd
import emd
from typing import List

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def apply_emd_to_segments(df_segments: pd.DataFrame, egg_channels: List[str]):
    """
    Apply Empirical Mode Decomposition (EMD) to each segment of EGG signals.

    Each row in df_segments is assumed to represent a segment of the signal,
    containing start/end times and the EGG channels as arrays.
    This function computes the Intrinsic Mode Functions (IMFs) for each channel
    using EMD.

    :param df_segments: Pandas DataFrame where each row is a signal segment.
                        Columns must include 'start_time_sec', 'end_time_sec'
                        and the EGG channels as arrays.
    :param egg_channels: List of strings with the names of EGG channels to process.
    :return: List of dictionaries, one per segment, containing:
             - segment_id
             - start_time_sec
             - end_time_sec
             - each channel's IMFs as "{channel}_imfs"
    """

    results = []

    # Iterate over each segment in the DataFrame
    for idx, row in df_segments.iterrows():

        # Initialize the dictionary to store segment results
        segment_result = {
            "segment_id": idx,
            "start_time_sec": row["start_time_sec"],
            "end_time_sec": row["end_time_sec"],
        }

        # Apply EMD to each specified EGG channel
        for ch in egg_channels:

            signal = row[ch]  # Extract the signal array for this channel

            # Skip signals that would cause EMD to fail
            if signal is None:
                print(f"  [SKIP EMD] Segment {idx} ch {ch} — nulled (noisy channel)")
                segment_result[f"{ch}_imfs"] = None
                continue
            if len(signal) < 100:
                print(f"  [SKIP EMD] Segment {idx} ch {ch} — too short ({len(signal)} samples)")
                segment_result[f"{ch}_imfs"] = None
                continue
            if not np.isfinite(signal).all():
                print(f"  [SKIP EMD] Segment {idx} ch {ch} — contains NaN or Inf")
                segment_result[f"{ch}_imfs"] = None
                continue
            if np.std(signal) == 0:
                print(f"  [SKIP EMD] Segment {idx} ch {ch} — constant signal")
                segment_result[f"{ch}_imfs"] = None
                continue

            try:
                imfs = emd.sift.sift(signal)  # Compute IMFs using EMD

                segment_result[f"{ch}_imfs"] = imfs  # Store IMFs in the result dictionary
            except Exception as e:
                print(f"  [SKIP EMD] Segment {idx} ch {ch} — EMD failed: {e}")
                segment_result[f"{ch}_imfs"] = None
        results.append(segment_result)  # Append the result for this segment

    return results


def compute_segment_imf_frequencies(
        emd_results: List,
        fs: float,
        egg_channels: List[str],
):
    """
    Compute dominant frequencies for all IMFs using peak-derived CPM
    (median inter-peak interval) instead of Welch PSD.

    More robust than Welch for non-stationary IMFs where the instantaneous
    frequency varies along the segment — the Welch estimate reflects the
    average spectral peak, which can be smeared if the frequency drifts.

    Uses the same adaptive prominence threshold as _passes_peak_filter
    (PEAK_FILTER_PROMINENCE_FRACTION × amplitude range).

    :param emd_results:  output of apply_emd_to_segments
    :param fs:           sampling frequency (Hz)
    :param egg_channels: list of EGG channel names
    :return:             same structure as compute_segment_imf_frequencies —
                         list of {'segment_id': int, '<ch>_freqs': list[float]}
                         where each float is the peak-derived CPM (or np.nan
                         if fewer than 2 peaks were detected)
    """
    from scipy.signal import find_peaks as _find_peaks
    from imf_selection.selector import PEAK_FILTER_PROMINENCE_FRACTION

    all_freqs = []

    for segment in emd_results:
        seg_freqs = {"segment_id": segment["segment_id"]}

        for ch in egg_channels:
            imfs = segment.get(f"{ch}_imfs")
            if imfs is None:
                seg_freqs[f"{ch}_freqs"] = []
                continue

            ch_freqs = []
            for i in range(imfs.shape[1]):
                imf       = imfs[:, i]
                amp_range = imf.max() - imf.min()

                if amp_range < 1e-10:
                    ch_freqs.append(np.nan)
                    continue

                min_prom     = amp_range * PEAK_FILTER_PROMINENCE_FRACTION
                peaks, _     = _find_peaks(imf, prominence=min_prom)

                if len(peaks) >= 2:
                    intervals_sec = np.diff(peaks) / fs
                    median_cpm    = float(60.0 / np.nanmedian(intervals_sec))
                else:
                    median_cpm = np.nan

                ch_freqs.append(median_cpm)

            seg_freqs[f"{ch}_freqs"] = ch_freqs

        all_freqs.append(seg_freqs)

    return all_freqs
