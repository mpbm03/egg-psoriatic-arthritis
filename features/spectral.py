"""
All frequency units are in cycles per minute (CPM).
Features are computed via Welch's PSD and the STFT spectrogram.

Available Functions
-------------------
[Public]
coeff_of_variation      - Compute the coefficient of variation (std/mean) of a set of values
compute_spectral_features - Compute spectral features (dominant frequency/power, band power, instability, STFT energy) from a single-channel signal
extract_spectral_features - Apply compute_spectral_features to every segment and channel of a segmented DataFrame, prefixing columns by channel name

------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import pandas as pd
from scipy.signal import welch
from typing import Dict, List
from utils import get_welch_params

# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #
GASTRIC_BAND = (2.0,  4.0)   # normal gastric rhythm
LOW_BAND     = (0.0,  2.0)   # sub-gastric / bradygastria

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def coeff_of_variation(values) -> float:
    """
    Coefficient of variation: std / mean.

    Returns 0.0 if the mean is zero to avoid division by zero.

    :param values: array-like of numeric values
    :return: coefficient of variation (std / mean)
    """
    values = np.array(values, dtype=float)
    mean   = values.mean()
    return float(values.std() / mean) if mean != 0 else 0.0


# ── Per-channel spectral features ────────────────────────────────────────────

def compute_spectral_features(
    signal: np.ndarray,
    fs: float,
    duration_sec: float,
) -> Dict[str, float]:
    """
    Compute spectral features from a single-channel signal.

    Features are derived from the Power Spectral Density (PSD) via Welch's method.
    Frequency units are in cycles per minute (CPM). Only scale-invariant features
    are returned (frequencies, coefficients of variation, and relative power
    ratios) — raw power/energy features were removed because they depend on the
    signal's normalisation/amplitude scale and are therefore not reliably
    comparable across segments or subjects.

    :param signal: np.ndarray
        1-D array of signal samples.
    :param fs: float
        Sampling frequency of the signal in Hz.
    :param duration_sec: float
        Duration of the sub-segment in seconds, used to normalize features
        dependent on duration.

    :return: dict[str, float]
        Dictionary containing the following spectral features:
            - "Dominant Frequency (CPM)": Dominant frequency of the PSD in CPM
            - "Frequency Instability Coefficient": Coefficient of variation of dominant frequency in sliding windows
            - "Power Instability Coefficient": Coefficient of variation of dominant power in sliding windows
            - "Percent Power [2-4 CPM]": Percent of total power in the gastric band
            - "Power Density Ratio": Ratio of low band power (0–2 CPM) to gastric band power
    """

    window_sec, overlap_sec = get_welch_params(
        n_samples=len(signal),
        fs=fs,
    )
    nperseg = int(window_sec * fs)
    noverlap = int(overlap_sec * fs)
    # Safety clamps
    nperseg = min(nperseg, len(signal))
    noverlap = min(noverlap, nperseg - 1)

    freqs_hz, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    freqs_cpm = freqs_hz * 60.0

    # Band masks
    gastric_mask = (freqs_cpm >= GASTRIC_BAND[0]) & (freqs_cpm <= GASTRIC_BAND[1])
    low_mask     = (freqs_cpm >= LOW_BAND[0]) & (freqs_cpm <= LOW_BAND[1])

    # 1. Dominant frequency
    idx_dom       = np.argmax(psd)
    dominant_freq = freqs_cpm[idx_dom]

    # 2. Band power — kept internally (not returned) to compute percent_power
    #    and power_density_ratio below.
    band_power = np.trapezoid(psd[gastric_mask], freqs_cpm[gastric_mask]) / duration_sec

    # 3-4. Instability coefficients
    step       = nperseg - noverlap
    dom_freqs  = []
    dom_powers = []

    for start in range(0, len(signal) - nperseg + 1, step):
        window       = signal[start:start+nperseg]
        f_w, p_w     = welch(window, fs=fs, nperseg=min(nperseg,len(window)))
        f_w_cpm      = f_w * 60.0
        idx_window   = np.argmax(p_w)
        dom_freqs.append(f_w_cpm[idx_window])
        dom_powers.append(p_w[idx_window])

    freq_instability  = coeff_of_variation(dom_freqs)
    power_instability = coeff_of_variation(dom_powers)

    # 5-6. Percent power & power density ratio
    total_power        = np.trapezoid(psd, freqs_cpm)
    percent_power      = (band_power * duration_sec / total_power * 100) if total_power != 0 else 0.0
    low_band_power     = np.trapezoid(psd[low_mask], freqs_cpm[low_mask])
    power_density_ratio = (low_band_power / band_power) if band_power != 0 else 0.0

    return {
        "Dominant Frequency (CPM)":           dominant_freq,
        "Frequency Instability Coefficient":  freq_instability,
        "Power Instability Coefficient":      power_instability,
        "Percent Power [2-4 CPM]":            percent_power,
        "Power Density Ratio":                 power_density_ratio,
    }

# ── DataFrame-level extraction ────────────────────────────────────────────────

def extract_spectral_features(
    df_processed: pd.DataFrame,
    egg_channels: List[str],
    fs: float = 1.0
) -> pd.DataFrame:
    """
    Extract spectral features from multiple EGG channels for all segments in a dataframe.

    For each row (sub-segment) in the dataframe, this function computes spectral features
    for each specified channel using the `compute_spectral_features` function. The features
    include dominant frequency and instability/relative-power measures. All
    features are prefixed by the channel name to distinguish between channels.

    :param df_processed: DataFrame containing sub-segments of EGG signals. Must include:
            - 'duration_sec': duration of each sub-segment in seconds
            - EGG channel columns containing 1D signals (e.g., 'CH1', 'CH2', ...)
            - 'label': class label for each sub-segment
    :param egg_channels: List of column names in `df_processed` representing the EGG channels to analyze.
    :param fs: Sampling frequency of the signals in Hz. Default is 1.0.

    :return: A dataframe with one row per sub-segment, containing spectral features for all
        channels, with column names prefixed by the channel (e.g., 'CH1_Dominant Frequency (CPM)').
        The dataframe also contains the 'label' column for each segment.
    """
    # List to store features for all segments
    feature_rows = []

    # Iterate over each sub-segment (row) in the dataframe
    for idx, row in df_processed.iterrows():
        all_features = {}
        duration = row["duration_sec"]  # segment duration for normalization

        # Extract spectral features for each channel
        for ch in egg_channels:
            raw = row[ch]

            # Guard against None (e.g. missing gastric/intestinal component
            # in 'separate' reconstruction mode) — fill with NaN features.
            if raw is None:
                all_features.update({f"{ch}_{k}": np.nan for k in [
                    "Dominant Frequency (CPM)",
                    "Frequency Instability Coefficient", "Power Instability Coefficient",
                    "Percent Power [2-4 CPM]", "Power Density Ratio",
                ]})
                continue

            signal = np.array(raw)  # ensure signal is a numpy array
            feats = compute_spectral_features(signal, fs=fs, duration_sec=duration)

            # Prefix feature names with channel name
            all_features.update({f"{ch}_{k}": v for k, v in feats.items()})

        # Add label and subject_id column
        all_features["label"] = row["label"]
        all_features["subject_id"] = row["subject_id"]

        # Append the features for this segment to the list
        feature_rows.append(all_features)

    # Convert the list of dictionaries into a DataFrame
    return pd.DataFrame(feature_rows)
