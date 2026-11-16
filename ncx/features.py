"""Feature extraction for both modalities.

The two candidate spaces are built to the same shape on purpose: 17 features
each, organised into the same six families (magnitude, ratio, distribution,
complexity, coupling, asymmetry). Matched cardinality and matched taxonomy are
what allow the redundancy structures to be compared without confounding the
comparison with differences in how generously each modality was parameterised.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import signal as sps

from .config import Config, EEG_BANDS, feature_order
from . import dsp


# ==========================================================================
# EEG
# ==========================================================================

def eeg_trial_features(x: np.ndarray, fs: float, ch_names: list[str],
                       cfg: Config, complexity_max_ch: int = 12) -> dict:
    """Seventeen qEEG descriptors for one trial (channels x samples).

    Per-channel quantities are pooled with the median rather than the mean.
    In prehospital and acute-ward recordings a single electrode with poor
    contact is common, and the median keeps one bad channel from dragging a
    trial-level descriptor with it.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    xf = dsp.bandlimit(x, fs, cfg.hp_hz, cfg.lp_hz, cfg.notch_hz,
                       cfg.notch_q, cfg.filter_order)

    bp = dsp.band_powers(xf, fs, EEG_BANDS)
    d, t, a, b = (np.median(bp[k]) for k in ("delta", "theta", "alpha", "beta"))

    feats: dict = {"delta": d, "theta": t, "alpha": a, "beta": b}
    feats["delta_alpha"] = float(dsp.safe_ratio(d, a))
    feats["theta_alpha"] = float(dsp.safe_ratio(t, a))
    feats["dt_ab"] = float(dsp.safe_ratio(d + t, a + b))

    sk, ku = dsp.dist_stats(xf)
    feats["skewness"], feats["kurtosis"] = sk, ku

    # Complexity is the expensive part; a deterministic channel subset keeps
    # extraction tractable over 50 subjects x 40 trials without biasing the
    # estimate toward any one scalp region.
    idx = np.linspace(0, xf.shape[0] - 1, min(complexity_max_ch, xf.shape[0])).astype(int)
    feats["sample_entropy"] = float(np.nanmedian([dsp.sample_entropy(xf[i]) for i in idx]))
    feats["higuchi_fd"] = float(np.nanmedian([dsp.higuchi_fd(xf[i]) for i in idx]))

    feats["wpli"] = dsp.wpli_median(xf, fs, cfg.hp_hz, cfg.lp_hz)
    feats["ms_coherence"] = dsp.coherence_median(xf, fs, cfg.hp_hz, cfg.lp_hz)

    # pDBSI per band over homologous pairs present in this montage.
    upper = [c.upper() for c in ch_names]
    pos = {c: i for i, c in enumerate(upper)}
    pairs = [(pos[l], pos[r]) for l, r in cfg.homologous_pairs
             if l.upper() in pos and r.upper() in pos]
    for band in ("delta", "theta", "alpha", "beta"):
        if pairs:
            pw = bp[band]
            feats[f"pdbsi_{band}"] = dsp.asymmetry_index(
                np.array([pw[i] for i, _ in pairs]),
                np.array([pw[j] for _, j in pairs]))
        else:
            feats[f"pdbsi_{band}"] = np.nan
    return feats


def _one_eeg(rec, cfg, k):
    f = eeg_trial_features(rec["data"], rec.get("fs", cfg.eeg_fs),
                           rec.get("ch_names", []), cfg)
    f["subject"] = rec["subject"]
    f["label"] = rec["label"]
    f["trial"] = rec.get("trial", k)
    return f


def build_eeg_table(records, cfg: Config, verbose: bool = True) -> pd.DataFrame:
    """Extract the qEEG table, one row per trial.

    Extraction is embarrassingly parallel -- each trial is independent and the
    estimators are deterministic -- so workers change wall-clock time and
    nothing else. Entropy and fractal dimension dominate the cost; on 2000
    trials this is the difference between an hour and a few minutes.
    """
    records = list(records)
    jobs = getattr(cfg, "extract_jobs", 1)
    if jobs != 1 and len(records) > 8:
        try:
            from joblib import Parallel, delayed
            rows = Parallel(n_jobs=jobs, verbose=5 if verbose else 0)(
                delayed(_one_eeg)(r, cfg, k) for k, r in enumerate(records))
        except ImportError:
            rows = [_one_eeg(r, cfg, k) for k, r in enumerate(records)]
    else:
        rows = [_one_eeg(r, cfg, k) for k, r in enumerate(records)]
    df = pd.DataFrame(rows)
    cols = feature_order("eeg") + ["subject", "label", "trial"]
    return df[[c for c in cols if c in df.columns]]


# ==========================================================================
# TCD
# ==========================================================================

def _beat_indices(env: np.ndarray, fs: float) -> np.ndarray:
    """Systolic peak detection on the CBFV envelope.

    A refractory period of 300 ms is enforced (physiological ceiling ~200 bpm)
    and the prominence threshold is set from the envelope's own dispersion, so
    the detector does not need retuning between recordings with different
    insonation depths and gain settings.
    """
    env = np.asarray(env, dtype=float)
    if env.size < int(2 * fs):
        return np.array([], dtype=int)
    prom = max(np.nanstd(env) * 0.3, 1e-9)
    peaks, _ = sps.find_peaks(env, distance=max(int(0.3 * fs), 1), prominence=prom)
    return peaks


def tcd_envelope_features(env: np.ndarray, fs: float) -> dict:
    """Per-record hemodynamic descriptors from one CBFV envelope."""
    env = np.asarray(env, dtype=float)
    env = env[np.isfinite(env)]
    out = {k: np.nan for k in
           ["psv", "edv", "mfv", "pulse_amplitude", "pulsatility_index",
            "resistance_index", "sd_ratio", "env_skewness", "env_kurtosis",
            "env_sample_entropy", "env_higuchi_fd", "upstroke_slope",
            "time_to_peak_frac"]}
    if env.size < int(2 * fs):
        return out

    pk = _beat_indices(env, fs)
    psv_l, edv_l, mfv_l, up_l, ttp_l = [], [], [], [], []
    for i in range(len(pk) - 1):
        s, e = pk[i], pk[i + 1]
        beat = env[s:e]
        if beat.size < 3:
            continue
        psv_l.append(float(np.max(beat)))
        edv_l.append(float(np.min(beat)))
        mfv_l.append(float(np.mean(beat)))
        # Upstroke is measured on the following beat's rising limb.
        trough = int(np.argmin(beat))
        rise = beat[trough:]
        if rise.size >= 3:
            pk_rel = int(np.argmax(rise))
            if pk_rel > 0:
                up_l.append(float((rise[pk_rel] - rise[0]) / (pk_rel / fs)))
                ttp_l.append(float(pk_rel / rise.size))

    if not psv_l:
        return out
    psv, edv, mfv = np.median(psv_l), np.median(edv_l), np.median(mfv_l)
    out["psv"], out["edv"], out["mfv"] = float(psv), float(edv), float(mfv)
    out["pulse_amplitude"] = float(psv - edv)
    out["pulsatility_index"] = float(dsp.safe_ratio(psv - edv, mfv))
    out["resistance_index"] = float(dsp.safe_ratio(psv - edv, psv))
    out["sd_ratio"] = float(dsp.safe_ratio(psv, edv))
    sk, ku = dsp.dist_stats(env[None, :])
    out["env_skewness"], out["env_kurtosis"] = sk, ku
    out["env_sample_entropy"] = dsp.sample_entropy(env)
    out["env_higuchi_fd"] = dsp.higuchi_fd(env)
    out["upstroke_slope"] = float(np.median(up_l)) if up_l else np.nan
    out["time_to_peak_frac"] = float(np.median(ttp_l)) if ttp_l else np.nan
    return out


def tcd_record_features(left: np.ndarray | None, right: np.ndarray | None,
                        fs: float) -> dict:
    """Combine bilateral envelopes into the 17-feature TCD vector.

    When only one side was insonated the four asymmetry descriptors are
    returned as NaN and the caller is expected to report the reduced
    cardinality rather than impute; an imputed asymmetry index would
    manufacture exactly the kind of spurious structure this study is about.
    """
    sides = [s for s in (left, right) if s is not None and np.size(s) > 0]
    if not sides:
        return {k: np.nan for k in feature_order("tcd")}

    per = [tcd_envelope_features(s, fs) for s in sides]
    base = {k: float(np.nanmean([p[k] for p in per])) for k in per[0]}

    if left is not None and right is not None and np.size(left) and np.size(right):
        l, r = per[0], per[1]
        base["asym_mfv"] = dsp.asymmetry_index(l["mfv"], r["mfv"])
        base["asym_psv"] = dsp.asymmetry_index(l["psv"], r["psv"])
        base["asym_edv"] = dsp.asymmetry_index(l["edv"], r["edv"])
        base["asym_pi"] = dsp.asymmetry_index(l["pulsatility_index"],
                                              r["pulsatility_index"])
    else:
        for k in ("asym_mfv", "asym_psv", "asym_edv", "asym_pi"):
            base[k] = np.nan
    return base


def build_tcd_table(records, cfg: Config) -> pd.DataFrame:
    """records: dicts with left/right envelopes (or a prebuilt feature dict)."""
    rows = []
    for k, rec in enumerate(records):
        if "features" in rec:                 # already-tabulated TCD source
            f = dict(rec["features"])
        else:
            f = tcd_record_features(rec.get("left"), rec.get("right"),
                                    rec.get("fs", cfg.tcd_fs))
        f["subject"] = rec["subject"]
        f["label"] = rec["label"]
        f["trial"] = rec.get("trial", k)
        rows.append(f)
    df = pd.DataFrame(rows)
    cols = feature_order("tcd") + ["subject", "label", "trial"]
    return df[[c for c in cols if c in df.columns]]
