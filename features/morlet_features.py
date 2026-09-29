"""
Morlet wavelet-based features for EGG signals.

Implements the non-analytic Morlet wavelet with omega0=6, following
Komorowski & Pietraszek (2016) "The Use of Continuous Wavelet Transform
Based on the Fast Fourier Transform in the Analysis of Multi-channel
Electrogastrography Recordings" (DOI 10.1007/s10916-015-0358-4).

The authors tested multiple wavelets (Gaussian derivatives, analytic and
non-analytic Morlet, Mexican hat) for EGG analysis via CWTFT and found that
the non-analytic Morlet wavelet with omega0=6 produced scalograms consistent
with known EGG properties, a dominant peak around 3 CPM and the expected
postprandial power changes. omega0=6 also satisfies the admissibility
condition required for a valid wavelet.

This module computes a "gastric similarity" feature: the CWT coefficient
magnitude in a given frequency band, averaged over time and scales.
A high value indicates a strong, well-localized oscillation matching the
Morlet waveform at that frequency. A low value indicates the signal lacks this characteristic
oscillation (irregular / dysrhythmic activity).

extract_morlet_features() applies this to every row of a segmented
DataFrame (one feature per channel). The frequency band is selected
automatically per channel based on the column name: plain channel names
(e.g. 'CH1') use UNION_BAND_CPM — the combined
gastric + intestinal band, since both rhythms are merged into a single
signal.

Available Functions
--------------------
[Public]
morlet_gastric_similarity
    Compute a Morlet-wavelet similarity feature for one signal and frequency band.
extract_morlet_features
    Apply morlet_gastric_similarity() to every segment and channel of a
    segmented DataFrame, producing one feature column per channel.
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import pandas as pd
import pycwt
from typing import List, Tuple

from constants import MORLET_OMEGA0, BANDS_CPM


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def morlet_gastric_similarity(
        signal:   np.ndarray,
        fs:       float,
        band_cpm: Tuple[float, float] = (BANDS_CPM["gastric"][0], BANDS_CPM["intestinal"][1]),
        omega0:   float = MORLET_OMEGA0,
        agg:      str = "mean",
) -> float:
    """
    Compute a Morlet-wavelet similarity feature for a given frequency band.

    The CWT is computed with the non-analytic Morlet wavelet (omega0=6, as
    in Komorowski & Pietraszek 2016) across the scales supported by pycwt.
    Each scale corresponds to a pseudo-frequency. The feature is the
    magnitude of the CWT coefficients restricted to scales whose
    pseudo-frequency falls within band_cpm, aggregated over time and scales.

    A high value indicates the signal contains a strong, well-localized
    oscillation matching the Morlet waveform at that frequency — i.e. a
    regular, sinusoid-like slow wave. A low value indicates the signal
    lacks this characteristic oscillation (irregular/dysrhythmic activity).

    :param signal:   1D array, EGG signal (downsampled, single channel or IMF)
    :param fs:       sampling frequency (Hz)
    :param band_cpm: (min, max) frequency band in cycles per minute.
                     Default UNION_BAND_CPM — combined gastric + intestinal band.
    :param omega0:   non-dimensional frequency parameter of the Morlet
                     wavelet. Default 6.0 (Komorowski & Pietraszek 2016;
                     satisfies the admissibility condition).
    :param agg:      aggregation over the |CWT| matrix in the selected band:
                     "mean" (average activity, default) or
                     "max"  (peak activity).

    :return: scalar feature — aggregated |CWT| magnitude in band_cpm.
             Returns 0.0 if the signal is too short or no scale falls
             within the requested band.
    """
    signal = np.asarray(signal, dtype=float)

    mother = pycwt.Morlet(omega0)
    dt = 1.0 / fs

    try:
        wave, scales, freqs, coi, fft, fftfreqs = pycwt.cwt(signal, dt, wavelet=mother)
    except Exception:
        return 0.0

    # Convert frequencies (Hz) to cycles per minute
    freqs_cpm = freqs * 60.0

    # Select scales whose pseudo-frequency falls within the requested band
    band_mask = (freqs_cpm >= band_cpm[0]) & (freqs_cpm <= band_cpm[1])

    if not band_mask.any():
        return 0.0

    magnitude = np.abs(wave[band_mask, :])

    if agg == "max":
        return float(magnitude.max())
    return float(magnitude.mean())


def extract_morlet_features(
        df_processed: pd.DataFrame,
        egg_channels: List[str],
        fs: float,
        band_cpm: Tuple[float, float] = None,
        omega0: float = MORLET_OMEGA0,
        agg: str = "mean",
) -> pd.DataFrame:
    """
    Apply morlet_gastric_similarity() to every segment and channel of a
    segmented DataFrame, producing one feature column per channel.

    Channel columns hold the summed gastric + intestinal signal,
    so every channel uses the union of
    BANDS_CPM["gastric"] and BANDS_CPM["intestinal"] by default, capturing
    waveform regularity across both rhythms in a single value per channel.

    An explicit band_cpm overrides this default for all channels.

    :param df_processed: segmented DataFrame — one row per segment, columns
                         include the channels in egg_channels (each holding
                         a 1D array of signal samples)
    :param egg_channels: list of channel column names to compute the
                         feature for (e.g. ['CH1', 'CH2', 'CH3', 'CH4'])
    :param fs:           sampling frequency (Hz)
    :param band_cpm:     (min, max) frequency band in cycles per minute.
                         If None (default), the union of the gastric and
                         intestinal bands is used for every channel.
    :param omega0:       Morlet wavelet frequency parameter. Default 6.0.
    :param agg:          aggregation mode, "mean" or "max". Default "mean".

    :return: pd.DataFrame with one column per channel:
             '<channel>_morlet_sim', same row index as df_processed.
    """
    resolved_band = band_cpm or (BANDS_CPM["gastric"][0], BANDS_CPM["intestinal"][1])

    feature_cols = {}

    for ch in egg_channels:
        if ch not in df_processed.columns:
            continue

        col_name = f"{ch}_morlet_sim"
        feature_cols[col_name] = df_processed[ch].apply(
            lambda sig: morlet_gastric_similarity(
                sig, fs=fs, band_cpm=resolved_band, omega0=omega0, agg=agg
            )
        )

    return pd.DataFrame(feature_cols, index=df_processed.index)