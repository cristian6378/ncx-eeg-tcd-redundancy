"""Channel-pooling sensitivity analysis.

The trial-level qEEG descriptors in the main analysis are obtained by pooling
per-channel quantities with the median. That choice is defensible on
robustness grounds, but it is not neutral: the median is a spatially
symmetric operator, and motor imagery expresses itself as a *lateralised*
desynchronisation over sensorimotor cortex. Pooling symmetrically across the
montage can therefore cancel precisely the contrast the labels encode.

This module separates the two things the main pipeline computes together. A
per-channel cache is built once -- band powers, distribution moments and
complexity indices for every channel of every trial, plus the two connectivity
descriptors, which are properties of the montage as a whole and do not depend
on the pooling operator. Every pooling variant is then a re-aggregation of
that cache, so the variants differ *only* in the spatial operator and not in
any filtering, epoching or spectral-estimation detail.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

from . import dsp
from .config import Config, EEG_BANDS

# The sensorimotor region of interest. Motor-imagery rhythms are generated in
# and around the pre- and post-central gyri; these are the electrodes of this
# montage that overlie them.
SENSORIMOTOR = ["FC3", "FC4", "C3", "C4", "CZ", "CP3", "CP4"]

# The lateralisation contrast. C3/C4 is the canonical pair for hand motor
# imagery: imagined left-hand movement desynchronises the right hemisphere and
# vice versa, so a contrast between them carries the class signal that a
# symmetric pooling operator destroys.
LATERAL_PAIR = ("C3", "C4")

BANDS = ("delta", "theta", "alpha", "beta")
PER_CHANNEL_COLS = list(BANDS) + ["skewness", "kurtosis",
                                  "sample_entropy", "higuchi_fd"]


def per_channel_cache(rec: dict, cfg: Config, complexity_max_ch: int = 12,
                      connectivity: dict | None = None) -> pd.DataFrame:
    """Per-channel descriptors for one trial, in long form.

    Complexity indices are computed on a deterministic evenly spaced channel
    subset, exactly as in the main pipeline, and are left missing elsewhere;
    the pooling variants use a nan-aware aggregation so the subset is handled
    identically across variants.
    """
    x = np.atleast_2d(np.asarray(rec["data"], dtype=float))
    fs = rec.get("fs", cfg.eeg_fs)
    names = [c.upper() for c in rec.get("ch_names", [])]
    if len(names) != x.shape[0]:
        names = [f"CH{i}" for i in range(x.shape[0])]

    xf = dsp.bandlimit(x, fs, cfg.hp_hz, cfg.lp_hz, cfg.notch_hz,
                       cfg.notch_q, cfg.filter_order)
    bp = dsp.band_powers(xf, fs, EEG_BANDS)

    n_ch = xf.shape[0]
    idx = np.linspace(0, n_ch - 1, min(complexity_max_ch, n_ch)).astype(int)
    sampen = np.full(n_ch, np.nan)
    hfd = np.full(n_ch, np.nan)
    for i in idx:
        sampen[i] = dsp.sample_entropy(xf[i])
        hfd[i] = dsp.higuchi_fd(xf[i])

    with np.errstate(all="ignore"):
        mu = xf.mean(axis=1, keepdims=True)
        sd = xf.std(axis=1, keepdims=True)
        z = (xf - mu) / np.where(sd > 0, sd, np.nan)
        skew = np.nanmean(z ** 3, axis=1)
        kurt = np.nanmean(z ** 4, axis=1) - 3.0

    out = pd.DataFrame({"channel": names, **{b: np.asarray(bp[b], dtype=float)
                                             for b in BANDS},
                        "skewness": skew, "kurtosis": kurt,
                        "sample_entropy": sampen, "higuchi_fd": hfd})
    out["subject"] = rec["subject"]
    out["trial"] = rec.get("trial", 0)
    out["label"] = rec["label"]
    # Connectivity is a montage-level property computed over all channel
    # pairs: it is identical under every pooling operator, so it is joined
    # from the main feature table rather than recomputed here.
    if connectivity is not None:
        key = (rec["subject"], rec.get("trial", 0))
        w, c = connectivity.get(key, (np.nan, np.nan))
        out["wpli"], out["ms_coherence"] = w, c
    else:
        out["wpli"] = dsp.wpli_median(xf, fs, cfg.hp_hz, cfg.lp_hz)
        out["ms_coherence"] = dsp.coherence_median(xf, fs, cfg.hp_hz, cfg.lp_hz)
    return out


def connectivity_lookup(path: Path) -> dict:
    """(subject, trial) -> (wPLI, magnitude-squared coherence)."""
    # usecols returns columns in file order, not the order requested, so the
    # frame is reindexed explicitly before it is unpacked positionally.
    t = pd.read_csv(path, usecols=["subject", "trial", "wpli", "ms_coherence"])
    t = t[["subject", "trial", "wpli", "ms_coherence"]]
    return {(str(s), int(tr)): (float(w), float(c))
            for s, tr, w, c in t.itertuples(index=False)}


def build_cache(records, cfg: Config, verbose: bool = True,
                connectivity: dict | None = None) -> pd.DataFrame:
    frames, n = [], 0
    for rec in records:
        frames.append(per_channel_cache(rec, cfg, connectivity=connectivity))
        n += 1
        if verbose and n % 200 == 0:
            print(f"      cached {n} trials", flush=True)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------
# Pooling operators
# --------------------------------------------------------------------------

def _agg(values: np.ndarray, how: str) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    if how == "median":
        return float(np.median(v))
    if how == "mean":
        return float(np.mean(v))
    if how == "trimmed":
        # 20 % symmetric trim: retains the median's resistance to a single bad
        # electrode while using more of the montage than the median alone.
        return float(trim_mean(v, 0.2)) if v.size >= 5 else float(np.median(v))
    raise ValueError(how)


VARIANTS = {
    "median_all":      dict(how="median", roi=None,         lateral=False),
    "mean_all":        dict(how="mean",   roi=None,         lateral=False),
    "trimmed_all":     dict(how="trimmed", roi=None,        lateral=False),
    "median_roi":      dict(how="median", roi=SENSORIMOTOR, lateral=False),
    "lateralised_roi": dict(how="median", roi=SENSORIMOTOR, lateral=True),
}


def pool_trial(g: pd.DataFrame, spec: dict, pdbsi: dict) -> dict:
    """Collapse one trial's per-channel cache under a single pooling spec."""
    how, roi, lateral = spec["how"], spec["roi"], spec["lateral"]
    sub = g[g.channel.isin(roi)] if roi else g
    if sub.empty:
        sub = g

    f = {c: _agg(sub[c].to_numpy(), how) for c in PER_CHANNEL_COLS}

    if lateral:
        # Replace the four magnitude descriptors with hemispheric log-ratios.
        # The log keeps the contrast symmetric about zero and stabilises the
        # variance of a ratio of two positive, heavy-tailed power estimates.
        left = g[g.channel == LATERAL_PAIR[0]]
        right = g[g.channel == LATERAL_PAIR[1]]
        for b in BANDS:
            if len(left) and len(right):
                lv, rv = float(left[b].iloc[0]), float(right[b].iloc[0])
                f[b] = (float(np.log((lv + 1e-20) / (rv + 1e-20)))
                        if lv > 0 and rv > 0 else np.nan)
            else:
                f[b] = np.nan

    # Ratios are formed from the pooled band powers of this variant, except
    # under the lateralised variant where the magnitudes are contrasts and a
    # ratio of contrasts is not interpretable; there the ROI-pooled powers are
    # used so that the ratio family keeps its usual meaning.
    if lateral:
        src = {b: _agg(sub[b].to_numpy(), how) for b in BANDS}
    else:
        src = {b: f[b] for b in BANDS}
    d, t, a, bt = (src["delta"], src["theta"], src["alpha"], src["beta"])
    f["delta_alpha"] = float(dsp.safe_ratio(d, a))
    f["theta_alpha"] = float(dsp.safe_ratio(t, a))
    f["dt_ab"] = float(dsp.safe_ratio(d + t, a + bt))

    f["wpli"] = float(g["wpli"].iloc[0])
    f["ms_coherence"] = float(g["ms_coherence"].iloc[0])
    f.update(pdbsi)
    return f


def pdbsi_from_cache(g: pd.DataFrame, cfg: Config) -> dict:
    """pDBSI is pooling-invariant: it is defined over homologous pairs."""
    pos = {c: i for i, c in enumerate(g.channel.tolist())}
    pairs = [(pos[l.upper()], pos[r.upper()]) for l, r in cfg.homologous_pairs
             if l.upper() in pos and r.upper() in pos]
    out = {}
    for b in BANDS:
        if pairs:
            pw = g[b].to_numpy(dtype=float)
            out[f"pdbsi_{b}"] = dsp.asymmetry_index(
                np.array([pw[i] for i, _ in pairs]),
                np.array([pw[j] for _, j in pairs]))
        else:
            out[f"pdbsi_{b}"] = np.nan
    return out


def build_variant(cache: pd.DataFrame, variant: str, cfg: Config) -> pd.DataFrame:
    spec = VARIANTS[variant]
    rows = []
    for (subj, trial), g in cache.groupby(["subject", "trial"], sort=False):
        pd_ = pdbsi_from_cache(g, cfg)
        f = pool_trial(g, spec, pd_)
        f.update(subject=subj, trial=trial, label=g["label"].iloc[0])
        rows.append(f)
    return pd.DataFrame(rows)


def build_all_variants(cache: pd.DataFrame, cfg: Config,
                       outdir: Path) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for v in VARIANTS:
        df = build_variant(cache, v, cfg)
        p = outdir / f"features_eeg_{v}.csv"
        df.to_csv(p, index=False)
        paths[v] = p
        print(f"    {v:16s} -> {df.shape[0]} trials x {df.shape[1]} cols")
    return paths
