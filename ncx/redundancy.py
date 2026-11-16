"""Redundancy characterisation and reduction.

The reducer is written as a scikit-learn transformer rather than as a script
step. That is not a stylistic preference: it is the only way to guarantee that
correlation screening and VIF elimination are re-estimated inside every
training fold, which is what keeps the reported cross-validated performance
free of selection leakage.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def variance_inflation(X: np.ndarray) -> np.ndarray:
    """VIF per column via the R^2 of each column on the others.

    Computed from the correlation matrix inverse where it is well
    conditioned, falling back to per-column least squares otherwise; the
    fallback matters because a near-singular design is exactly the situation
    VIF is being used to detect.
    """
    X = np.asarray(X, dtype=float)
    p = X.shape[1]
    if p < 2:
        return np.ones(p)
    Xs = StandardScaler().fit_transform(X)
    R = np.corrcoef(Xs, rowvar=False)
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 1.0)
    try:
        cond = np.linalg.cond(R)
        if np.isfinite(cond) and cond < 1e12:
            return np.clip(np.diag(np.linalg.inv(R)), 1.0, np.inf)
    except np.linalg.LinAlgError:
        pass
    vifs = np.ones(p)
    for j in range(p):
        others = np.delete(Xs, j, axis=1)
        beta, *_ = np.linalg.lstsq(others, Xs[:, j], rcond=None)
        resid = Xs[:, j] - others @ beta
        ss_res = float(resid @ resid)
        ss_tot = float(Xs[:, j] @ Xs[:, j])
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vifs[j] = 1.0 / max(1e-12, 1.0 - min(r2, 1 - 1e-12))
    return vifs


class RedundancyReducer(BaseEstimator, TransformerMixin):
    """Two-stage, label-agnostic reduction: |r| screening, then VIF pruning.

    Label-agnostic by design. Selecting on the target would tune the retained
    set to whatever the downstream classifier can already exploit, and the
    question here is how much overlap the *representation* carries, which has
    to be answerable without reference to any one model.
    """

    def __init__(self, corr_threshold: float = 0.60, vif_threshold: float = 10.0,
                 feature_names: list[str] | None = None):
        self.corr_threshold = corr_threshold
        self.vif_threshold = vif_threshold
        self.feature_names = feature_names

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        names = (list(self.feature_names) if self.feature_names is not None
                 else [f"f{i}" for i in range(X.shape[1])])
        keep = list(range(X.shape[1]))

        # ---- Stage 1: pairwise correlation screening ----------------------
        R = np.corrcoef(np.nan_to_num(X[:, keep]), rowvar=False)
        R = np.nan_to_num(np.atleast_2d(R))
        dropped_corr = []
        changed = True
        while changed and len(keep) > 2:
            changed = False
            sub = np.abs(R[np.ix_(range(len(keep)), range(len(keep)))])
            np.fill_diagonal(sub, 0.0)
            i, j = np.unravel_index(np.argmax(sub), sub.shape)
            if sub[i, j] >= self.corr_threshold:
                # Drop whichever of the pair is more entangled with the rest;
                # ties resolve to the later column so the rule is deterministic.
                mi, mj = np.mean(sub[i]), np.mean(sub[j])
                drop_local = i if mi > mj else j
                dropped_corr.append(names[keep[drop_local]])
                keep.pop(drop_local)
                R = np.corrcoef(np.nan_to_num(X[:, keep]), rowvar=False)
                R = np.nan_to_num(np.atleast_2d(R))
                changed = True

        self.n_after_corr_ = len(keep)
        self.dropped_correlation_ = dropped_corr

        # ---- Stage 2: iterative VIF pruning -------------------------------
        dropped_vif = []
        while len(keep) > 2:
            v = variance_inflation(X[:, keep])
            k = int(np.argmax(v))
            if v[k] > self.vif_threshold:
                dropped_vif.append((names[keep[k]], float(v[k])))
                keep.pop(k)
            else:
                break

        self.support_ = np.array(sorted(keep))
        self.dropped_vif_ = dropped_vif
        self.kept_names_ = [names[i] for i in self.support_]
        self.final_vif_ = (variance_inflation(X[:, self.support_])
                           if len(self.support_) > 1 else np.ones(len(self.support_)))
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.support_]

    def get_support(self, indices: bool = False):
        if indices:
            return self.support_
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.support_] = True
        return mask


# --------------------------------------------------------------------------
# Descriptive diagnostics (whole-cohort; reported as description, not as a
# basis for any performance estimate)
# --------------------------------------------------------------------------

def correlation_summary(df: pd.DataFrame, names: list[str]) -> dict:
    X = df[names].to_numpy(dtype=float)
    R = np.nan_to_num(np.corrcoef(X, rowvar=False))
    off = R[~np.eye(len(names), dtype=bool)]
    return {"matrix": pd.DataFrame(R, index=names, columns=names),
            "mean_abs_offdiag": float(np.mean(np.abs(off))),
            "median_abs_offdiag": float(np.median(np.abs(off))),
            "frac_pairs_above_06": float(np.mean(np.abs(off) >= 0.60)),
            "frac_pairs_above_08": float(np.mean(np.abs(off) >= 0.80)),
            "max_abs_offdiag": float(np.max(np.abs(off)))}


def pca_profile(df: pd.DataFrame, names: list[str],
                targets=(0.80, 0.90, 0.95)) -> dict:
    X = StandardScaler().fit_transform(df[names].to_numpy(dtype=float))
    p = PCA().fit(X)
    cum = np.cumsum(p.explained_variance_ratio_)
    out = {"explained_variance_ratio": p.explained_variance_ratio_.tolist(),
           "cumulative": cum.tolist(),
           "pc1": float(p.explained_variance_ratio_[0]),
           "eigenvalues": p.explained_variance_.tolist()}
    for t in targets:
        out[f"n_components_for_{int(t*100)}"] = int(np.searchsorted(cum, t) + 1)
    # Participation ratio: a scalar summary of how many directions actually
    # carry variance, less arbitrary than a fixed variance cut-off.
    ev = p.explained_variance_
    out["participation_ratio"] = float((ev.sum() ** 2) / np.sum(ev ** 2))
    return out


def family_redundancy(df: pd.DataFrame, names: list[str],
                      family_fn) -> pd.DataFrame:
    """Mean |r| within and between feature families.

    This is the table that answers whether redundancy concentrates inside a
    family (band powers with band powers) or leaks across families, which is
    the part of the structure that differs most between the two modalities.
    """
    X = df[names].to_numpy(dtype=float)
    R = np.abs(np.nan_to_num(np.corrcoef(X, rowvar=False)))
    fams = sorted({family_fn(n) for n in names})
    idx = {f: [i for i, n in enumerate(names) if family_fn(n) == f] for f in fams}
    out = pd.DataFrame(index=fams, columns=fams, dtype=float)
    for a in fams:
        for b in fams:
            ia, ib = idx[a], idx[b]
            if not ia or not ib:
                continue
            block = R[np.ix_(ia, ib)]
            if a == b:
                if len(ia) < 2:
                    out.loc[a, b] = np.nan
                    continue
                m = ~np.eye(len(ia), dtype=bool)
                out.loc[a, b] = float(block[m].mean())
            else:
                out.loc[a, b] = float(block.mean())
    return out


def partial_correlation_summary(df: pd.DataFrame, names: list[str],
                                generators: list[str]) -> dict:
    """Redundancy remaining after partialling out the generating variables.

    Some descriptors are deterministic transforms of others -- the pulsatility
    and resistance indices are closed-form functions of the velocity triple,
    and the slowing ratios are functions of band power. Raw correlations
    therefore conflate redundancy that is algebraic by construction with
    redundancy that reflects a shared physiological driver. Partialling out
    the generating set isolates the second kind, which is the kind that can
    differ between modalities for interesting reasons.
    """
    X = StandardScaler().fit_transform(df[names].to_numpy(dtype=float))
    gen_idx = [names.index(g) for g in generators if g in names]
    rest = [i for i in range(len(names)) if i not in gen_idx]
    if not gen_idx or len(rest) < 2:
        return {}
    G = X[:, gen_idx]
    # residualise every remaining descriptor on the generating set
    R = np.empty((X.shape[0], len(rest)))
    for k, j in enumerate(rest):
        beta, *_ = np.linalg.lstsq(G, X[:, j], rcond=None)
        R[:, k] = X[:, j] - G @ beta
    C = np.nan_to_num(np.corrcoef(R, rowvar=False))
    m = ~np.eye(len(rest), dtype=bool)
    raw = np.abs(np.nan_to_num(np.corrcoef(X[:, rest], rowvar=False)))[m]
    return {"features": [names[i] for i in rest],
            "generators": [names[i] for i in gen_idx],
            "mean_abs_partial": float(np.abs(C[m]).mean()),
            "mean_abs_raw": float(raw.mean()),
            "frac_partial_ge_06": float((np.abs(C[m]) >= 0.60).mean()),
            "frac_raw_ge_06": float((raw >= 0.60).mean()),
            "matrix": pd.DataFrame(C, index=[names[i] for i in rest],
                                   columns=[names[i] for i in rest])}
