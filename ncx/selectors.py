"""Supervised selection baselines, at matched cardinality.

Reviewers of redundancy work reasonably ask whether a label-aware selector
would simply do better. Answering that fairly requires two things that are
easy to get wrong: the selector must be re-fitted inside each training fold,
and it must be asked for the *same number* of features the unsupervised
pipeline retained, so that what is being compared is selector quality rather
than subset size. Both are enforced here.
"""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class MRMRSelector(BaseEstimator, TransformerMixin):
    """Minimum redundancy, maximum relevance (MID variant).

    Relevance is mutual information with the label; redundancy is mean
    absolute Pearson correlation with the already-selected set.
    """

    def __init__(self, k: int = 9, random_state: int = 0):
        self.k = k
        self.random_state = random_state

    def fit(self, X, y):
        X = np.nan_to_num(np.asarray(X, dtype=float))
        p = X.shape[1]
        k = min(self.k, p)
        rel = mutual_info_classif(X, y, random_state=self.random_state)
        R = np.abs(np.nan_to_num(np.corrcoef(X, rowvar=False)))
        if R.ndim == 0:
            R = np.ones((1, 1))
        sel = [int(np.argmax(rel))]
        while len(sel) < k:
            best, best_score = None, -np.inf
            for j in range(p):
                if j in sel:
                    continue
                red = float(np.mean(R[j, sel]))
                score = rel[j] - red
                if score > best_score:
                    best, best_score = j, score
            if best is None:
                break
            sel.append(best)
        self.support_ = np.array(sorted(sel))
        self.n_features_in_ = p
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.support_]


class ReliefFSelector(BaseEstimator, TransformerMixin):
    """ReliefF with k nearest hits/misses, standardised feature space."""

    def __init__(self, k: int = 9, n_neighbors: int = 10,
                 n_samples: int = 300, random_state: int = 0):
        self.k = k
        self.n_neighbors = n_neighbors
        self.n_samples = n_samples
        self.random_state = random_state

    def fit(self, X, y):
        X = np.nan_to_num(np.asarray(X, dtype=float))
        y = np.asarray(y)
        Xs = StandardScaler().fit_transform(X)
        n, p = Xs.shape
        rng = np.random.default_rng(self.random_state)
        idx = (rng.choice(n, size=min(self.n_samples, n), replace=False)
               if n > self.n_samples else np.arange(n))
        classes, counts = np.unique(y, return_counts=True)
        prior = dict(zip(classes, counts / n))
        rng_span = np.ptp(Xs, axis=0)
        rng_span[rng_span == 0] = 1.0
        W = np.zeros(p)
        for i in idx:
            d = np.abs(Xs - Xs[i]) / rng_span
            dist = d.sum(axis=1)
            dist[i] = np.inf
            same = y == y[i]
            hit = np.argsort(np.where(same, dist, np.inf))[:self.n_neighbors]
            W -= d[hit].mean(axis=0) / len(idx)
            for c in classes:
                if c == y[i]:
                    continue
                mask = y == c
                if not mask.any():
                    continue
                miss = np.argsort(np.where(mask, dist, np.inf))[:self.n_neighbors]
                w = prior[c] / max(1e-12, 1.0 - prior[y[i]])
                W += w * d[miss].mean(axis=0) / len(idx)
        self.scores_ = W
        self.support_ = np.array(sorted(np.argsort(W)[::-1][:min(self.k, p)]))
        self.n_features_in_ = p
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.support_]


class LassoSelector(BaseEstimator, TransformerMixin):
    """L1 multinomial logistic regression, C tuned down until k survive.

    Rather than fixing C and accepting whatever cardinality falls out, the
    penalty is walked until the support reaches the target size; this is what
    makes the comparison with the unsupervised pipeline like-for-like.
    """

    def __init__(self, k: int = 9, random_state: int = 0):
        self.k = k
        self.random_state = random_state

    def fit(self, X, y):
        X = np.nan_to_num(np.asarray(X, dtype=float))
        Xs = StandardScaler().fit_transform(X)
        p = Xs.shape[1]
        k = min(self.k, p)
        best = np.arange(p)
        # A coarser penalty ladder: the selector only has to reach the
        # target cardinality, so resolving the full L1 path inside every
        # fold is wasted work.
        for C in np.logspace(1.0, -2.5, 12):
            # `multi_class` was removed in scikit-learn 1.8; saga defaults
            # to multinomial for multiclass targets, so it is no longer passed.
            m = LogisticRegression(penalty="l1", solver="saga", C=C,
                                   max_iter=1200, tol=1e-3,
                                   random_state=self.random_state)
            try:
                m.fit(Xs, y)
            except Exception:                   # noqa: BLE001
                continue
            imp = np.abs(m.coef_).sum(axis=0)
            nz = np.flatnonzero(imp > 1e-10)
            if len(nz) <= k:
                best = nz if len(nz) else np.argsort(imp)[::-1][:k]
                break
            best = np.argsort(imp)[::-1][:k]
        self.support_ = np.array(sorted(np.asarray(best)[:k]))
        self.n_features_in_ = p
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.support_]


def make_selector(name: str, k: int, random_state: int = 0):
    name = name.lower()
    if name == "mrmr":
        return MRMRSelector(k=k, random_state=random_state)
    if name == "relieff":
        return ReliefFSelector(k=k, random_state=random_state)
    if name == "lasso":
        return LassoSelector(k=k, random_state=random_state)
    raise ValueError(f"unknown selector: {name}")
