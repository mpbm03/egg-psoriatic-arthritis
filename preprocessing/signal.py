"""
Signal-level preprocessing: centering, normalisation, filtering, downsampling.

Available Functions
-------------------
[Public]
center_egg_signals            - Subtract each channel's mean to center the EGG signals
normalize_egg_signals_centered - Normalise already-centered EGG signals (currently supports "clipping": percentile clip + max-abs scaling)
lowpass_filter_df              - Apply a 4th-order Butterworth lowpass filter to selected channels before downsampling, with cutoff at the new Nyquist frequency
downsampling                    - Downsample a DataFrame's signals by an integer decimation factor (fs / fs_new)

------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from typing import List

# internal imports
from constants import EGG_CHANNELS

# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def center_egg_signals(df: pd.DataFrame, egg_channels: List[str] = EGG_CHANNELS) -> pd.DataFrame:
    """
    Centers EGG signals by subtracting the mean of each channel.

    :param df: pd.DataFrame containing the signals.
    :param egg_channels: List[str], names of the EGG channels to be centered.
    :return: pd.DataFrame with centered EGG signals.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_centered = df.copy()

    # Subtract the mean from each EGG channel
    for ch in egg_channels:
        df_centered[ch] = df_centered[ch] - df_centered[ch].mean()

    return df_centered

def normalize_egg_signals_centered(
    df: pd.DataFrame,
    egg_channels: list,
    method: str = "clipping"
) -> pd.DataFrame:
    """
    Apply normalization methods assuming signals are already centered.

    :param df: pd.DataFrame containing the signals.
    :param egg_channels: List[str], names of the EGG channels to be normalized.
    :param method: str, options: "clipping", "weighted", "maxabs", "iterative", "standard"

    :return: pd.DataFrame with normalized EGG signals.
    """

    df_norm = df.copy()

    for ch in egg_channels:

        x = df_norm[ch].values.astype(float)

        #Clipping
        if method == "clipping":

            clip_percentile = 99.5

            clip_val = np.percentile(np.abs(x), clip_percentile)
            x_clipped = np.clip(x, -clip_val, clip_val)

            max_val = np.max(np.abs(x_clipped))
            x_norm = x_clipped / max_val if max_val != 0 else x_clipped

        else:
            raise ValueError("Unknown normalization method")

        df_norm[ch] = x_norm

    return df_norm

def lowpass_filter_df(df: pd.DataFrame, channels: list, fs: float, fs_new: float) -> pd.DataFrame:
    """
    Apply a lowpass filter to specified channels in a DataFrame before downsampling.

    :param df: Original DataFrame containing all signals.
    :param channels: List of channel names to apply the filter (e.g., EGG_CHANNELS)
    :param fs: Original sampling frequency (Hz)
    :param fs_new: Target sampling frequency (Hz)
    :return: New DataFrame with filtered channels.
    """
    df_filtered = df.copy()

    # Cutoff frequency is half of the new sampling frequency (Nyquist for downsampling)
    cutoff = 0.5 * fs_new
    nyq = 0.5 * fs
    b, a = butter(4, cutoff / nyq, btype='low')  # 4th order Butterworth

    for ch in channels:
        if ch in df_filtered.columns:
            df_filtered[ch] = filtfilt(b, a, df_filtered[ch].values)

    return df_filtered



def downsampling(df: pd.DataFrame, fs: int, fs_new:int):
    """
    :param df: pd.DataFrame containing the signals.
    :param fs: original sampling frequency
    :param fs_new: new sampling frequency
    :return: pd.DataFrame containing the downsampled signals. """
    factor = fs// fs_new
    df_downsampled = df.iloc[::factor].reset_index(drop=True)

    return df_downsampled