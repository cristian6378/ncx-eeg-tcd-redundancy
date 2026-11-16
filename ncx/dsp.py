"""Signal-processing primitives shared by both modalities.

Implementations are deliberately dependency-light (SciPy + NumPy only) so the
pipeline runs on a clinical workstation without a conda environment. Where a
published estimator has several conventions in circulation -- sample entropy's
distance metric, Higuchi's k_max, the WPLI denominator -- the convention used
is stated in the docstring, because these choices change the numbers a
reviewer will try to reproduce.
"""
from __future__ import annotations
import numpy as np
from scipy import signal as sps
from scipy.stats import skew, kurtosis


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------

def bandlimit(x: np.ndarray, fs: float, hp: float | None, lp: float | None,
              notch: float | None = None, notch_q: float = 30.0,
              order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-limiting with optional mains notch.

    Applied per channel along the last axis. Zero-phase (filtfilt) is used so
    that no group delay is introduced ahead of the latency-sensitive
    morphological features (upstroke slope, time-to-peak).
    """
    y = np.asarray(x, dtype=float)
    nyq = fs / 2.0
    if notch is not None and 0 < notch < nyq:
        b, a = sps.iirnotch(notch / nyq, notch_q)
        y = sps.filtfilt(b, a, y, axis=-1)
    if hp is not None and hp > 0:
        b, a = sps.butter(order, hp / nyq, btype="highpass")
        y = sps.filtfilt(b, a, y, axis=-1)
    if lp is not None and lp < nyq:
        b, a = sps.butter(order, lp / nyq, btype="lowpass")
        y = sps.filtfilt(b, a, y, axis=-1)
    return y


# --------------------------------------------------------------------------
# Spectral
# --------------------------------------------------------------------------

def band_powers(x: np.ndarray, fs: float, bands: dict) -> dict:
    """Absolute band power per channel via Welch's method.

    Welch is preferred over a single periodogram of the whole epoch because
    the stroke EEG here is non-stationary across an 8 s trial; segment
    averaging trades frequency resolution for a lower-variance estimate, which
    matters more when the downstream feature is a ratio of two bands.
    """
    x = np.atleast_2d(x)
    nper = int(min(x.shape[-1], round(2.0 * fs)))
    f, pxx = sps.welch(x, fs=fs, nperseg=nper, noverlap=nper // 2,
                       window="hann", axis=-1, detrend="constant")
    out = {}
    for name, (lo, hi) in bands.items():
        m = (f >= lo) & (f < hi)
        out[name] = np.trapezoid(pxx[..., m], f[m], axis=-1) if m.any() \
            else np.zeros(pxx.shape[0])
    return out


# --------------------------------------------------------------------------
# Complexity
# --------------------------------------------------------------------------

def sample_entropy(x: np.ndarray, m: int = 2, r: float | None = None) -> float:
    """Sample entropy with Chebyshev distance and r = 0.2 * SD by default.

    Returns NaN rather than +inf when no template match is found, so that a
    degenerate channel propagates as missing instead of poisoning the median.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < m + 2:
        return np.nan
    sd = np.std(x)
    if sd == 0:
        return np.nan
    r = 0.2 * sd if r is None else r

    # The Chebyshev distance between two m-length templates is the running
    # maximum of the single-sample distance matrix along a diagonal shift:
    # d_m(i,j) = max_k |x[i+k] - x[j+k]| for k < m. Building |x_i - x_j| once
    # and folding m shifts into it replaces the per-template Python loop with
    # a handful of array operations, which is what makes a sensitivity
    # analysis over five pooling variants and fifty subjects tractable.
    D = np.abs(x[:, None] - x[None, :])

    def _phi(mm: int) -> int:
        M = n - mm + 1
        A = D[:M, :M].copy()
        for k in range(1, mm):
            np.maximum(A, D[k:k + M, k:k + M], out=A)
        # A is symmetric with a zero diagonal, so the number of unordered
        # pairs within r is (all matches - the M diagonal ones) / 2.
        return (int(np.count_nonzero(A <= r)) - M) // 2

    b, a = _phi(m), _phi(m + 1)
    if b == 0 or a == 0:
        return np.nan
    return float(-np.log(a / b))


def higuchi_fd(x: np.ndarray, k_max: int = 10) -> float:
    """Higuchi fractal dimension estimated over k = 1..k_max.

    k_max = 10 is used throughout; the estimate is mildly sensitive to this
    choice, so it is fixed rather than tuned per modality.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 2 * k_max:
        return np.nan
    lk = []
    ks = np.arange(1, k_max + 1)
    for k in ks:
        lm = []
        for mm in range(k):
            idx = np.arange(mm, n, k)
            if idx.size < 2:
                continue
            length = np.sum(np.abs(np.diff(x[idx])))
            # Higuchi (1988): the curve length is normalised by (N-1), by the
            # number of differences actually taken, and by k twice -- once for
            # the sampling interval and once in the definition of L(k). The
            # second factor is easy to drop, and dropping it biases every
            # estimate down by exactly 1.0.
            norm = (n - 1) / ((idx.size - 1) * k * k)
            lm.append(length * norm)
        if lm:
            lk.append(np.mean(lm))
    if len(lk) < 2:
        return np.nan
    lk = np.asarray(lk)
    good = lk > 0
    if good.sum() < 2:
        return np.nan
    slope = np.polyfit(np.log(1.0 / ks[good]), np.log(lk[good]), 1)[0]
    return float(slope)


# --------------------------------------------------------------------------
# Connectivity
# --------------------------------------------------------------------------

def wpli_median(x: np.ndarray, fs: float, fmin: float = 1.0,
                fmax: float = 45.0, max_pairs: int = 400) -> float:
    """Median weighted phase-lag index across channel pairs.

    WPLI is used in preference to plain coherence or PLV because it is
    insensitive to zero-lag components, which in scalp EEG are dominated by
    volume conduction rather than genuine interaction. Pairs are subsampled
    deterministically when the montage is large, to bound run time.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    nch = x.shape[0]
    if nch < 2:
        return np.nan
    nper = int(min(x.shape[-1], round(2.0 * fs)))
    pairs = [(i, j) for i in range(nch) for j in range(i + 1, nch)]
    if len(pairs) > max_pairs:
        step = int(np.ceil(len(pairs) / max_pairs))
        pairs = pairs[::step]
    vals = []
    for i, j in pairs:
        f, cs = sps.csd(x[i], x[j], fs=fs, nperseg=nper, noverlap=nper // 2,
                        window="hann")
        m = (f >= fmin) & (f <= fmax)
        im = np.imag(cs[m])
        denom = np.mean(np.abs(im))
        if denom > 0:
            vals.append(np.abs(np.mean(im)) / denom)
    return float(np.median(vals)) if vals else np.nan


def coherence_median(x: np.ndarray, fs: float, fmin: float = 1.0,
                     fmax: float = 45.0, max_pairs: int = 400) -> float:
    """Median magnitude-squared coherence across channel pairs."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    nch = x.shape[0]
    if nch < 2:
        return np.nan
    nper = int(min(x.shape[-1], round(2.0 * fs)))
    pairs = [(i, j) for i in range(nch) for j in range(i + 1, nch)]
    if len(pairs) > max_pairs:
        step = int(np.ceil(len(pairs) / max_pairs))
        pairs = pairs[::step]
    vals = []
    for i, j in pairs:
        f, cxy = sps.coherence(x[i], x[j], fs=fs, nperseg=nper,
                               noverlap=nper // 2, window="hann")
        m = (f >= fmin) & (f <= fmax)
        if m.any():
            vals.append(np.mean(cxy[m]))
    return float(np.median(vals)) if vals else np.nan


# --------------------------------------------------------------------------
# Distributional helpers
# --------------------------------------------------------------------------

def dist_stats(x: np.ndarray) -> tuple[float, float]:
    """Median across channels of per-channel skewness and excess kurtosis."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    return (float(np.median(skew(x, axis=-1, bias=False))),
            float(np.median(kurtosis(x, axis=-1, bias=False))))


def safe_ratio(num, den, eps: float = 1e-12):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    return num / np.where(np.abs(den) < eps, np.nan, den)


def asymmetry_index(left, right, eps: float = 1e-12) -> float:
    """Normalised left-right asymmetry, |L-R| / (L+R), in [0, 1].

    The same functional form is used for the EEG pDBSI and the TCD
    hemispheric indices; keeping it identical across modalities is what makes
    the asymmetry families comparable rather than merely analogous.
    """
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    den = left + right
    v = np.abs(left - right) / np.where(np.abs(den) < eps, np.nan, den)
    return float(np.nanmedian(v))
