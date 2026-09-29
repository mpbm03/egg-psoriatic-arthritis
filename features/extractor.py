"""
Top-level feature extraction: combines TSFEL (temporal + statistical)
and spectral features into a single DataFrame.

This is the main entry point for the feature extraction stage.
The output is ready for train/test splitting and model training.

Available Functions
-------------------
[Public]
extract_all_features   - Extract TSFEL, spectral, and Morlet wavelet features, combine them, and drop any segment left with NaN values

------------------
[Private]


------------------
"""
# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd
from typing import List

# internal imports
from .tsfel import extract_tsfel_features
from .spectral import extract_spectral_features
from .morlet_features import extract_morlet_features

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #


def extract_all_features(
        df_processed: pd.DataFrame,
        egg_channels: List[str],
        normalize_features: List[str],
        fs: float = 1.0,
) -> pd.DataFrame:
    """
    Extract both TSFEL temporal/statistical features and spectral features,
    combine them into a single DataFrame, and drop any segment that still
    contains NaN values after extraction.

    NaN values can occur for a few reasons regardless of CHANNEL_NULLING:
      - When CHANNEL_NULLING=True, noisy channels were set to None during
        segmentation, producing NaN feature values for that channel.
      - TSFEL features such as Kurtosis and Skewness are undefined for
        near-constant signals (e.g. after artifact replacement with channel
        mean) or for very short sub-segments produced by SPLIT_CLEAN_SUBSEGMENTS.

    Any segment with at least one NaN feature value is dropped entirely —
    no imputation is applied.

    :param df_processed:       segmented DataFrame from the pipeline
    :param egg_channels:       list of EGG channel names to extract features from
    :param normalize_features: list of feature names to normalise by signal length
    :param fs:                 sampling frequency in Hz

    :return: DataFrame with one row per segment and one column per feature,
             plus 'label' and 'subject_id' columns
    """
    # TSFEL features
    tsfel_df = extract_tsfel_features(df_processed, egg_channels, normalize_features, fs)

    # Spectral features
    spectral_df = extract_spectral_features(df_processed, egg_channels, fs)

    # Morlet wavelet similarity features — one per channel. The frequency
    # band is selected automatically based on the channel name. See morlet_features.py
    # for the physiological justification (Komorowski & Pietraszek 2016).
    morlet_df = extract_morlet_features(df_processed, egg_channels, fs)

    # Merge — keep label and subject_id from tsfel_df
    combined_df = pd.concat(
        [tsfel_df.drop(columns="label"), spectral_df, morlet_df], axis=1
    )

    # ── Drop segments with NaN values ──────────────────────────────────────────
    # No imputation is applied — any segment with at least one NaN feature
    # value (e.g. from a nulled noisy channel when CHANNEL_NULLING=True, or
    # from an undefined TSFEL statistic such as Kurtosis/Skewness on a
    # near-constant signal) is dropped entirely.
    feature_cols = [c for c in combined_df.columns if c not in {"label", "subject_id"}]
    n_nan_before = combined_df[feature_cols].isna().sum().sum()

    if n_nan_before > 0:
        nan_segment_mask = combined_df[feature_cols].isna().any(axis=1)
        n_dropped_segs = nan_segment_mask.sum()
        n_total = len(combined_df)
        combined_df = combined_df[~nan_segment_mask].reset_index(drop=True)
        print(f"[impute] Dropping {n_dropped_segs} segment(s) with NaN out of "
              f"{n_total} total ({n_dropped_segs / n_total * 100:.1f}%)")

    return combined_df