"""
IMF selection strategy for EGG signal reconstruction.

Each function receives the IMF matrix for a single segment and channel,
along with pre-computed dominant frequencies, and returns the index and
signal of the best IMF for each physiological band.

Available Functions
-------------------
[Public]
select_imf   - Unified entry point: select the best IMF for a target band via the "band_power_peaks" strategy (band power ranking, restricted to IMFs whose dominant frequency and physiologically plausible peak count match the band)

------------------
[Private]
_expected_peak_range              - Compute the expected number of peaks for a band/duration, used to validate an IMF's peak count
_passes_peak_filter                - Check whether an IMF has a physiologically plausible peak count for the band, both globally and via sliding-window local coverage
_select_band_power                 - Select the IMF with the highest integrated band power, optionally restricted to frequency-classified and/or peak-filtered IMFs
_sliding_window_rhythm_coverage    - Compute the fraction of an IMF's duration covered by windows whose local peak count matches the target band (detects non-stationary rhythms)
_band_power                        - Compute a signal's integrated power within a frequency band via Welch's PSD

------------------
"""

# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np
from typing import Optional, Tuple, List

from scipy.signal import welch

from utils import get_welch_params

# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #

VALID_IMF_SELECTION_MODES = (
    "band_power_peaks",
)


# Tolerance around theoretical peak count for peak-filter modes
# e.g. 0.20 means ±20% of the theoretical [2*f_low*T, 2*f_high*T] range
PEAK_FILTER_TOLERANCE = 0
PEAK_FILTER_PROMINENCE_FRACTION = 0.05

# ── sliding-window local-rhythm coverage (used by both peak_cpm and
# band_power_peaks) ──────────────────────────────────────────────────────
# Detects non-stationary IMFs where periodicity is only present in part of
# the segment (e.g. rhythm at the start and end but irregular in the middle)
# — a global CPM/peak-count check over the whole segment can mask this.
# A sliding window scans the IMF; in each window the local CPM (from local
# peak intervals) is checked against the target band (with tolerance). The
# fraction of total duration covered by valid windows must meet a minimum
# threshold for the IMF to be accepted.
SLIDING_WINDOW_SEC          = 60.0   # window length
SLIDING_WINDOW_STEP_SEC     = 10.0   # step between consecutive windows
SLIDING_WINDOW_MIN_COVERAGE = 0.9    # min fraction of segment duration that
                                      # must be covered by valid windows

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def select_imf(
        imfs: np.ndarray,
        imf_freqs: List[float],
        band_cpm: Tuple[float, float],
        fs: float,
        mode: str,
) -> Tuple[Optional[np.ndarray], Optional[int], Optional[str]]:
    """
    Unified IMF selection entry point — dispatches to the appropriate strategy.

    :param imfs:               (n_samples, n_imfs) matrix of IMFs for one channel
    :param imf_freqs:          dominant frequencies in CPM, one per IMF column.
    :param band_cpm:           (low_cpm, high_cpm) target physiological band.
                               Not used in 'centroid' mode.
    :param fs:                 sampling frequency (Hz)
    :param mode:               "band_power_peaks"

    :return: (best_imf, best_idx, best_label) — all None if no IMF found.
    """
    band_hz = (band_cpm[0] / 60.0, band_cpm[1] / 60.0)

    if mode == "band_power_peaks":
        # same as band_power+ physiological peak count filter
        imf, idx = _select_band_power(imfs, imf_freqs, band_cpm, band_hz, fs,
                                      classified=True, peak_filter=True)
        return imf, idx, None

    else:
        raise ValueError(
            f"imf_selection_mode must be one of {VALID_IMF_SELECTION_MODES}. "
            f"Got: {mode!r}"
        )

# ------------------------------------------------------------------------------------------------------------------- #
# private — selection strategies
# ------------------------------------------------------------------------------------------------------------------- #


def _expected_peak_range(
        band_cpm: Tuple[float, float],
        duration_sec: float,
        tolerance: float = PEAK_FILTER_TOLERANCE,
) -> Tuple[int, int]:
    """
    Compute the expected number of significant positive peaks (maxima only)
    for an IMF in a given frequency band and signal duration.

    For a sinusoidal signal at frequency f CPM and duration T seconds:
        n_peaks = f * T / 60   (maxima only)

    The range is computed using the band limits with a tolerance factor
    to account for non-perfectly sinusoidal IMFs.

    :param band_cpm:     (low_cpm, high_cpm)
    :param duration_sec: signal duration in seconds
    :param tolerance:    fractional tolerance around theoretical range (default 0.20)
    :return:             (min_peaks, max_peaks) inclusive range
    """
    min_peaks = int((band_cpm[0] * duration_sec / 60) * (1 - tolerance))
    max_peaks = int((band_cpm[1] * duration_sec / 60) * (1 + tolerance))
    return max(1, min_peaks), max(2, max_peaks)

def _passes_peak_filter(
        imf: np.ndarray,
        band_cpm: Tuple[float, float],
        fs: float,
        tolerance: float = PEAK_FILTER_TOLERANCE,
        return_indices: bool = False,
):
    """
    Check whether the IMF has a physiologically plausible number of significant
    peaks for the given band and signal duration.

    Uses prominence threshold of 10% of the amplitude range to filter out
    small noise oscillations before counting peaks.

    In addition to the global peak count over the whole segment, requires
    that the rhythm be locally consistent: a sliding-window scan
    (_sliding_window_rhythm_coverage) must cover at least
    SLIDING_WINDOW_MIN_COVERAGE of the segment duration with windows whose
    local peak count also matches band_cpm. This rejects IMFs that satisfy
    the global peak count only because of a non-stationary mix — e.g. a
    rhythm present at the start and end of the segment but absent/irregular
    in the middle, which a single whole-segment peak count can hide.

    :param imf:            1D IMF signal
    :param band_cpm:       (low_cpm, high_cpm) target band
    :param fs:             sampling frequency (Hz)
    :param tolerance:      fractional tolerance (default 0.20 = ±20%)
    :param return_indices: if False (default) returns bool.
                           if True, returns dict with peak_indices (sorted
                           array of all peak positions), peak_count, passes,
                           min_p, max_p — useful for visualisation.
    :return: bool if return_indices=False, dict if return_indices=True
    """
    from scipy.signal import find_peaks as _find_peaks

    duration_sec = len(imf) / fs
    min_p, max_p = _expected_peak_range(band_cpm, duration_sec, tolerance)
    amp_range = imf.max() - imf.min()

    if amp_range < 1e-10:
        if return_indices:
            return {"peak_indices": np.array([], dtype=int),
                    "peak_count": 0, "passes": False,
                    "min_p": min_p, "max_p": max_p}
        return False

    min_prom = amp_range * PEAK_FILTER_PROMINENCE_FRACTION
    peaks_pos, _ = _find_peaks(imf, prominence=min_prom)
    n_peaks = len(peaks_pos)

    coverage = _sliding_window_rhythm_coverage(
        imf, band_cpm, fs,
        prominence_fraction=PEAK_FILTER_PROMINENCE_FRACTION,
        tolerance=tolerance,
    )
    passes = (coverage >= SLIDING_WINDOW_MIN_COVERAGE)

    if return_indices:
        return {
            "peak_indices": peaks_pos,
            "peak_count": n_peaks,
            "passes": passes,
            "min_p": min_p,
            "max_p": max_p,
            "coverage": coverage,
        }
    return passes


def _select_band_power(
        imfs: np.ndarray,
        imf_freqs: List[float],
        band_cpm: Tuple[float, float],
        band_hz: Tuple[float, float],
        fs: float,
        classified: bool = False,
        peak_filter: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Select the IMF with the highest integrated band power.

    :param classified:  if True, only IMFs whose pre-computed dominant
                        frequency falls within band_cpm are evaluated.
    :param peak_filter: if True, additionally requires the IMF to have a
                        physiologically plausible number of peaks for the
                        target band — uses _passes_peak_filter (±20% count
                        tolerance, 10% amplitude range prominence threshold).
    :return:            (best_imf, best_idx)
    """
    best_imf   = None
    best_score = -np.inf
    best_idx   = None

    for i in range(imfs.shape[1]):
        if classified:
            freq = imf_freqs[i] if i < len(imf_freqs) else np.nan
            if np.isnan(freq) or not (band_cpm[0] <= freq <= band_cpm[1]):
                continue

        if peak_filter and not _passes_peak_filter(imfs[:, i], band_cpm, fs):
            continue

        score = _band_power(imfs[:, i], fs, band_hz)
        if score > best_score:
            best_score = score
            best_imf   = imfs[:, i]
            best_idx   = i

    return best_imf, best_idx



# ------------------------------------------------------------------------------------------------------------------- #
# private — helpers
# ------------------------------------------------------------------------------------------------------------------- #


def _sliding_window_rhythm_coverage(
        imf: np.ndarray,
        band_cpm: Tuple[float, float],
        fs: float,
        prominence_fraction: float,
        tolerance: float,
        window_sec: float = SLIDING_WINDOW_SEC,
        step_sec: float = SLIDING_WINDOW_STEP_SEC,
) -> float:
    """
    Compute the fraction of the IMF's duration covered by sliding windows
    where the local peak count is consistent with band_cpm.

    Detects non-stationary IMFs where a global (whole-segment) peak count or
    median-CPM check would be satisfied even if the periodicity is only
    present in part of the signal — e.g. a rhythm visible at the start and
    end of the segment but absent or irregular in the middle. A single
    aggregate statistic over the whole segment can mask this; this function
    instead scores local consistency directly.

    For each sliding window (length window_sec, step step_sec), peaks are
    detected within that window using a prominence threshold of
    prominence_fraction * (local window amplitude range), and the window is
    marked "valid" if the resulting local peak count falls within the
    expected range for band_cpm at that window's duration (computed via
    _expected_peak_range with the given tolerance). The function returns the
    fraction of total signal duration spanned by valid windows (with overlap
    counted once, not double-counted).

    :param imf:                 1D IMF signal
    :param band_cpm:            (low_cpm, high_cpm) target band
    :param fs:                  sampling frequency (Hz)
    :param prominence_fraction: fraction of the window's amplitude range used
                                as the peak-detection prominence threshold —
                                pass PEAK_FILTER_PROMINENCE_FRACTION for
                                consistency with _passes_peak_filter
    :param tolerance:           fractional tolerance passed to
                                _expected_peak_range for each window
    :param window_sec:          sliding window length in seconds
    :param step_sec:            step between consecutive window starts

    :return: coverage fraction in [0, 1] — fraction of the signal duration
             covered by windows classified as locally rhythmic.
             Returns 0.0 if the signal is shorter than window_sec.
    """
    from scipy.signal import find_peaks as _find_peaks

    n = len(imf)
    win_n = int(round(window_sec * fs))
    step_n = max(1, int(round(step_sec * fs)))

    if n < win_n:
        return 0.0

    valid_mask = np.zeros(n, dtype=bool)

    for start in range(0, n - win_n + 1, step_n):
        end = start + win_n
        section = imf[start:end]

        amp_range = section.max() - section.min()
        if amp_range < 1e-10:
            continue

        min_prom = amp_range * prominence_fraction
        peaks, _ = _find_peaks(section, prominence=min_prom)
        n_peaks = len(peaks)

        min_p, max_p = _expected_peak_range(band_cpm, window_sec, tolerance)

        if min_p <= n_peaks <= max_p:
            valid_mask[start:end] = True

    return float(valid_mask.sum()) / n



def _band_power(signal: np.ndarray, fs: float, band: tuple[float, float]) -> float:
    """
    Compute the power of a signal within a specific frequency band.

    Welch parameters are computed adaptively via get_welch_params() based
    on the actual signal length, balancing frequency resolution against
    the number of averages available.

    :param signal: 1D signal array
    :param fs:     sampling frequency (Hz)
    :param band:   (low_hz, high_hz) frequency band in Hz

    :return: power within the band (trapezoidal integration of PSD)
    """
    window_sec, overlap_sec = get_welch_params(
        n_samples=len(signal),
        fs=fs,
    )
    nperseg = int(window_sec * fs)
    noverlap = int(overlap_sec * fs)

    f, Pxx = welch(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    mask = (f >= band[0]) & (f <= band[1])
    return float(np.trapezoid(Pxx[mask], f[mask]))
