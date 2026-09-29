"""
Temporal and statistical feature extraction via TSFEL.

Features that depend on signal duration are normalised by ``duration_sec``
to ensure comparability across sub-segments of different lengths.

Available Functions
-------------------
[Public]
extract_tsfel_features   - Extract TSFEL temporal/statistical features per channel,
normalising duration-dependent features and prefixing columns by channel name

------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import pandas as pd
import tsfel
from typing import List

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def extract_tsfel_features(
    df_processed: pd.DataFrame,
    egg_channels: list[str],
    normalize_features: List[str],
    fs: float = 1.0
) -> pd.DataFrame:
    """
    Extract temporal and statistical features from multi-channel IMF signals.

    For each row in the dataframe, features are extracted from each channel
    using TSFEL (temporal and statistical domains only). Features that depend
    on signal duration can be normalized by the sub-segment duration to ensure
    comparability across sub-segments of different lengths.

    :param df_processed: pd.DataFrame
        DataFrame with columns:
            - 'duration_sec': duration of the sub-segment in seconds.
            - EGG channels (each containing np.ndarray or list of IMF samples)
            - 'label': class label (0=pre-prandial, 1=post-prandial)

    :param egg_channels: list[str]
        List of channel names to extract features from (e.g., ['CH1', 'CH2']).

    :param normalize_features: list[str]
        List of features (as strings) to normalize by duration.

    :param fs: float, optional
        Sampling frequency of the signals in Hz. Default is 1.0.

    :return: pd.DataFrame
        DataFrame where each row contains the extracted features for all channels,
        prefixed by channel name (e.g., 'CH1_Mean'), plus the 'label' column.
    """

    # TSFEL config for temporal + statistical features
    cfg = {**tsfel.get_features_by_domain("temporal"),
           **tsfel.get_features_by_domain("statistical")}

    feature_rows = []

    print("Extracting features from all segments...")

    # Iterate over segments
    for idx, row in df_processed.iterrows():
        duration = row["duration_sec"]
        all_features = {}

        for ch in egg_channels:
            raw = np.array(row[ch])

            if raw is None:
                all_features.update({f"{ch}_{f}": np.nan for f in cfg.keys()})
                continue

            signal = np.array(raw)

            if signal.ndim == 0 or len(signal) == 0:
                # Fill NaNs if no signal
                all_features.update({f"{ch}_{f}": np.nan for f in cfg.keys()})
                continue

            feats = tsfel.time_series_features_extractor(cfg, signal, fs=fs, verbose=0)
            feats.columns = [f"{ch}_{col}" for col in feats.columns]

            # Normalize only the selected features by duration
            for col in feats.columns:
                if any(f.lower() in col.lower() for f in normalize_features):
                    feats[col] = feats[col] / duration

            all_features.update(feats.iloc[0].to_dict())

        # Add label
        all_features["label"] = row["label"]
        feature_rows.append(all_features)

    features_df = pd.DataFrame(feature_rows)
    return features_df


