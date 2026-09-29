"""
Motion-artefact detection and replacement for EGG recordings.

Available Functions
-------------------
[Public]
replace_artifacts_per_channel   - Replace each channel's artefact intervals with that channel's own clean mean, using per-channel interval lists

------------------
[Private]


------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import pandas as pd


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def replace_artifacts_per_channel(
        df: pd.DataFrame,
        artifact_indices_per_channel: dict,
) -> pd.DataFrame:
    """
    Replace artefact segments with each channel's own clean mean, using
    per-channel interval lists.

    This is the correct replacement to use with manually annotated artefacts,
    where different channels may have different noisy intervals.  A noisy
    interval on CH2 does not affect CH1.

    :param df: DataFrame containing the signals.
    :param artifact_indices_per_channel: dict
        {channel_name: List[Tuple[int, int]]}
        e.g. {"CH1": [(0, 500)], "CH2": [(200, 800), (1000, 1200)], ...}
        Channels absent from the dict are left untouched.

    :return: DataFrame with artefact segments replaced per channel.
    """
    cleaned_df = df.copy()
    n = len(df)

    for ch, intervals in artifact_indices_per_channel.items():
        if ch not in cleaned_df.columns or not intervals:
            continue

        # Build a boolean mask for this channel only
        mask = np.zeros(n, dtype=bool)
        for start, end in intervals:
            mask[start:min(end, n)] = True

        if not mask.any():
            continue

        clean_mean = cleaned_df.loc[~mask, ch].mean()
        cleaned_df.loc[mask, ch] = clean_mean

    return cleaned_df