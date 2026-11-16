"""Sensitivity of the redundancy result to its own analytical choices.

The central claim of this study is comparative: that the TCD candidate space
is more redundant than the qEEG one, at matched taxonomy and matched
cardinality. Two things could undermine that claim without either being
visible in the headline numbers. The screening thresholds ($|r| \\ge 0.6$,
VIF $> 10$) are conventional rather than derived, so the retained-set sizes
might be an artefact of where those lines were drawn; and the correlation
summaries are point estimates from finite cohorts, so the gap between
modalities might not survive resampling.

This module addresses both. `threshold_sweep` recomputes the whole reduction
chain across a grid of screening thresholds and reports whether the ordering
between modalities ever reverses. `bootstrap_redundancy_gap` resamples
subjects within each cohort to put a confidence interval on the difference
itself, which is the inferential test the comparative claim needs and which
the original analysis did not provide.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .redundancy import RedundancyReducer


# --------------------------------------------------------------------------
# Threshold sensitivity
# --------------------------------------------------------------------------

def _chain(X, names, corr_t, vif_t):
    r = RedundancyReducer(corr_t, vif_t, names).fit(X)
    return {"after_corr": int(r.n_after_corr_),
            "retained": int(len(r.support_)),
            "kept": list(r.kept_names_)}


def threshold_sweep(frames: dict, corr_grid=None, vif_grid=None) -> pd.DataFrame:
    """Reduction chain over a grid of screening thresholds.

    `frames` maps a label such as 'eeg/core' to (dataframe, feature names).
    Every combination is evaluated for every label, so the resulting table
    answers the question the single-threshold result cannot: whether the
    modality ordering is a property of the data or of the cut-off.
    """
    corr_grid = (np.round(np.arange(0.30, 0.96, 0.05), 2)
                 if corr_grid is None else np.asarray(corr_grid))
    vif_grid = (np.array([2.5, 5.0, 10.0, 20.0])
                if vif_grid is None else np.asarray(vif_grid))
    rows = []
    for label, (df, names) in frames.items():
        X = np.nan_to_num(df[names].to_numpy(dtype=float))
        p = len(names)
        for ct in corr_grid:
            for vt in vif_grid:
                c = _chain(X, names, float(ct), float(vt))
                rows.append({"set": label, "corr_threshold": float(ct),
                             "vif_threshold": float(vt), "n_candidates": p,
                             "after_corr": c["after_corr"],
                             "retained": c["retained"],
                             "retained_frac": c["retained"] / p,
                             "removed_frac": 1.0 - c["retained"] / p})
    return pd.DataFrame(rows)


def sweep_agreement(sweep: pd.DataFrame, a: str, b: str) -> dict:
    """Does `a` retain a larger fraction than `b` at every threshold pair?

    Reported as the fraction of the grid on which the ordering holds, plus the
    grid cells where it does not, so a partial result is visible rather than
    averaged away.
    """
    pa = sweep[sweep.set == a].set_index(["corr_threshold", "vif_threshold"])
    pb = sweep[sweep.set == b].set_index(["corr_threshold", "vif_threshold"])
    j = pa[["retained_frac"]].join(pb[["retained_frac"]], lsuffix="_a",
                                   rsuffix="_b", how="inner").dropna()
    holds = j.retained_frac_a > j.retained_frac_b
    ties = np.isclose(j.retained_frac_a, j.retained_frac_b)
    return {"n_cells": int(len(j)),
            "frac_ordering_holds": float(holds.mean()),
            "frac_ties": float(ties.mean()),
            "frac_reversed": float((~holds & ~ties).mean()),
            "median_gap": float((j.retained_frac_a - j.retained_frac_b).median()),
            "min_gap": float((j.retained_frac_a - j.retained_frac_b).min()),
            "max_gap": float((j.retained_frac_a - j.retained_frac_b).max())}


# --------------------------------------------------------------------------
# Inferential test of the comparative claim
# --------------------------------------------------------------------------

def _mean_abs_offdiag(X):
    R = np.corrcoef(X, rowvar=False)
    R = np.nan_to_num(np.atleast_2d(R))
    m = ~np.eye(R.shape[0], dtype=bool)
    return float(np.abs(R[m]).mean())


def bootstrap_redundancy_gap(df_a, names_a, groups_a,
                             df_b, names_b, groups_b,
                             n_boot=2000, random_state=0) -> dict:
    """Confidence interval on the between-modality difference in mean |r|.

    Resampling is at the level of the *subject*, not the record, because
    records from one patient are not independent draws; resampling rows would
    give an interval far too narrow. The two cohorts are disjoint, so they are
    resampled independently and the difference is formed per replicate.
    """
    rng = np.random.default_rng(random_state)
    Xa = np.nan_to_num(df_a[names_a].to_numpy(dtype=float))
    Xb = np.nan_to_num(df_b[names_b].to_numpy(dtype=float))
    ga = np.asarray(groups_a)
    gb = np.asarray(groups_b)
    ua, ub = np.unique(ga), np.unique(gb)
    idx_a = {g: np.flatnonzero(ga == g) for g in ua}
    idx_b = {g: np.flatnonzero(gb == g) for g in ub}

    obs_a, obs_b = _mean_abs_offdiag(Xa), _mean_abs_offdiag(Xb)
    diffs = np.empty(n_boot)
    va = np.empty(n_boot)
    vb = np.empty(n_boot)
    for i in range(n_boot):
        sa = np.concatenate([idx_a[g] for g in rng.choice(ua, ua.size, replace=True)])
        sb = np.concatenate([idx_b[g] for g in rng.choice(ub, ub.size, replace=True)])
        va[i] = _mean_abs_offdiag(Xa[sa])
        vb[i] = _mean_abs_offdiag(Xb[sb])
        diffs[i] = vb[i] - va[i]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean_abs_r_a": obs_a, "mean_abs_r_b": obs_b,
            "observed_gap": obs_b - obs_a,
            "boot_gap_mean": float(diffs.mean()),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "p_gap_le_0": float((diffs <= 0).mean()),
            "ci_a": [float(x) for x in np.percentile(va, [2.5, 97.5])],
            "ci_b": [float(x) for x in np.percentile(vb, [2.5, 97.5])],
            "n_boot": n_boot,
            "n_subjects_a": int(ua.size), "n_subjects_b": int(ub.size)}


def permutation_redundancy_gap(df_a, names_a, df_b, names_b,
                               n_perm=2000, random_state=0) -> dict:
    """Null for the modality gap built by shuffling the modality label.

    The two cohorts are unpaired and differ in size, so a paired permutation
    is unavailable. Instead the descriptor *columns* of the two matched cores
    are pooled and reassigned at random to two groups of the observed sizes,
    which asks whether a gap this large arises from having drawn two arbitrary
    sets of descriptors of these cardinalities. It is a weaker null than a
    subject-level one, and is reported as a supporting check rather than as
    the primary evidence.
    """
    rng = np.random.default_rng(random_state)
    Ra = np.abs(np.nan_to_num(np.corrcoef(
        np.nan_to_num(df_a[names_a].to_numpy(float)), rowvar=False)))
    Rb = np.abs(np.nan_to_num(np.corrcoef(
        np.nan_to_num(df_b[names_b].to_numpy(float)), rowvar=False)))
    ma = Ra[~np.eye(len(names_a), dtype=bool)]
    mb = Rb[~np.eye(len(names_b), dtype=bool)]
    obs = mb.mean() - ma.mean()
    pool = np.concatenate([ma, mb])
    n_a = ma.size
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(pool)
        null[i] = p[n_a:].mean() - p[:n_a].mean()
    return {"observed_gap": float(obs),
            "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
            "p": float(((null >= obs).sum() + 1) / (n_perm + 1)),
            "n_perm": n_perm}
