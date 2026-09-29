"""
Reconstruct IMF signals and build the final processed segment DataFrame.

For each segment and EGG channel, selects the best gastric and intestinal
IMFs (via imf_selection.selector) and combines them into the final signal
columns, keeping only segments where at least one channel was reconstructed.

Available Functions
-------------------
[Public]
build_final_dataframe   - Select and reconstruct gastric/intestinal IMFs per segment/channel, assemble the final DataFrame (signals, metadata, frequencies)

------------------
"""
# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd
from typing import List, Dict
from .selector import select_imf
# internal imports
from constants import EGG_CHANNELS, BANDS_CPM, IMF_SELECTION_MODE

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def build_final_dataframe(
        df_segments: pd.DataFrame,
        emd_results: List[Dict],
        fs_downsample: float,
        bands_cpm: dict = BANDS_CPM,
        egg_channels: List[str] = EGG_CHANNELS,
        subject_id: str = None,
        imf_selection_mode: str = IMF_SELECTION_MODE,
        emd_frequencies: List[Dict] = None,
) -> pd.DataFrame:
    """
     Reconstruct IMF signals and build the processed segment DataFrame.

    For each segment and each EGG channel, selects the best gastric IMF
    and the best intestinal IMF according to imf_selection_mode.

    How the two components are stored depends on reconstruction_mode:

      'sum'      — gastric + intestinal summed into a single signal per
                   channel (original behaviour).  Output columns: CH2, CH3, ...


    The 'label' column is left as None — it must be set by the caller
    (e.g. runner.py) based on the experimental design.

    :param df_segments:        pd.DataFrame  output of segment_egg_dataframe
    :param emd_results:        list of dicts  output of apply_emd_to_segments
    :param fs_downsample:      float  downsampled sampling frequency (Hz)
    :param bands_cpm:          dict   frequency bands in CPM
    :param egg_channels:       list of str  base channel names (e.g. ['CH2', 'CH3'])
    :param subject_id:         str   added as first column if provided
    :param imf_selection_mode:  'band_power', 'band_power_classified',
                                'snr_db', 'snr_db_unclassified', or 'autocorrelation'
    :param emd_frequencies:     list of dicts from compute_segment_imf_frequencies
    :return: pd.DataFrame with columns:
        [subject_id,] duration_sec, <signal_columns>, label
        In 'sum' mode signal_columns = egg_channels.
    """


    gastric_band_cpm = bands_cpm["gastric"]
    intestinal_band_cpm = bands_cpm["intestinal"]

    signal_cols = list(egg_channels)

    df_final = df_segments.copy()
    # Pre-create the new signal columns so df_final.at[] assignments work
    for col in signal_cols:
        if col not in df_final.columns:
            df_final[col] = None

    valid_mask = []

    for idx, segment in enumerate(emd_results):
        seg_id = segment.get("segment_id", idx)
        all_channels_valid = True

        for ch in egg_channels:
            imfs = segment.get(f"{ch}_imfs")
            if imfs is None:
                all_channels_valid = False
                continue

            # ── retrieve pre-computed dominant frequencies for this segment/channel
            seg_freqs_entry = next(
                (s for s in emd_frequencies if s["segment_id"] == seg_id), None
            ) if emd_frequencies is not None else None
            ch_freqs = seg_freqs_entry.get(f"{ch}_freqs", []) if seg_freqs_entry is not None else []

            # ── all other modes — select gastric and intestinal separately
            gastric_best, gastric_best_idx, _ = select_imf(
                imfs=imfs,
                imf_freqs=ch_freqs,
                band_cpm=gastric_band_cpm,
                fs=fs_downsample,
                mode=imf_selection_mode
            )
            intestinal_best, intestinal_best_idx, _ = select_imf(
                imfs=imfs,
                imf_freqs=ch_freqs,
                band_cpm=intestinal_band_cpm,
                fs=fs_downsample,
                mode=imf_selection_mode
            )

            # ── store signals ─────────────────────────────────────────

            if gastric_best is not None and intestinal_best is not None:
                df_final.at[idx, ch] = gastric_best + intestinal_best
            elif gastric_best is not None:
                df_final.at[idx, ch] = gastric_best
            elif intestinal_best is not None:
                df_final.at[idx, ch] = intestinal_best
            else:
                all_channels_valid = False

            if gastric_best_idx is None and intestinal_best_idx is None:
                print(f"  [DEBUG Neither] seg={idx} ch={ch} "
                      f"seg_freqs_entry={'exists' if seg_freqs_entry else 'None'} "
                      f"ch_freqs={ch_freqs[:5] if ch_freqs else '[]'} "
                      f"n_imfs={imfs.shape[1]}")
            elif gastric_best_idx is not None:
                if seg_freqs_entry is None:
                    print(f"  [DEBUG] seg={idx} ch={ch} gastric_idx={gastric_best_idx} "
                          f"BUT seg_freqs_entry=None → freq NaN")
                elif gastric_best_idx >= len(ch_freqs):
                    print(f"  [DEBUG] seg={idx} ch={ch} gastric_idx={gastric_best_idx} "
                          f"BUT ch_freqs only has {len(ch_freqs)} entries → freq NaN")

        valid_mask.append(all_channels_valid)

    # Keep only segments where at least one channel was reconstructed
    df_final = df_final[valid_mask].reset_index(drop=True)
    df_segments_valid = df_segments[valid_mask].reset_index(drop=True)

    df_final["label"] = None  # caller assigns the actual label

    # Preserve metadata columns from segment_egg_dataframe
    meta_cols = [c for c in ["start_index", "segment_order",
                             "artifact_ratio_max", "noisy_channels", "n_segments_total"]
                 if c in df_final.columns]

    freq_cols = [f"{ch}_{band}_freq"
                 for ch in egg_channels
                 for band in ("gastric", "intestinal")
                 if f"{ch}_{band}_freq" in df_final.columns]

    final_cols = meta_cols + ["duration_sec"] + signal_cols + ["label"] + freq_cols
    df_final = df_final[final_cols]

    if subject_id is not None:
        df_final.insert(0, "subject_id", subject_id)

    df_final["start_time_sec"] = df_segments_valid["start_time_sec"].values
    df_final["end_time_sec"] = df_segments_valid["end_time_sec"].values

    return df_final