"""Regression tests for two failures that once passed silently.

1. The inner cross-validation was handed to RandomizedSearchCV as the generator
   that `split()` returns. scikit-learn >= 1.9 ships `cv` to worker processes,
   a generator cannot be pickled, and every fit failed. Each failure was caught
   per fold, so the run finished, wrote nothing, and exited with status 0.
2. Nothing distinguished "a few folds failed" from "every fold failed".

Both are asserted here so neither can recur unnoticed.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from ncx import evaluate as ev
from ncx.config import Config


def _toy(n_subj=24, per_subj=6, p=5, seed=0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_subj), per_subj)
    y = np.repeat(np.arange(n_subj) % 2, per_subj)
    X = rng.standard_normal((g.size, p)) + 0.8 * y[:, None]
    return X, y.astype(str), g.astype(str), [f"f{i}" for i in range(p)]


def _small_cfg(search_jobs: int) -> Config:
    cfg = Config()
    cfg.n_repeats, cfg.n_splits, cfg.inner_splits = 1, 3, 2
    cfg.n_search_iter = 2
    cfg.search_jobs = search_jobs
    return cfg


def test_parallel_search_produces_results():
    """With worker processes the search must fit, not fail on pickling."""
    X, y, g, names = _toy()
    cfg = _small_cfg(search_jobs=2)
    models = {"Logistic Regression": ev.build_models(cfg)["Logistic Regression"]}
    out = ev.run_condition(X, y, g, names, cfg, "toy", mode="full", models=models)
    assert len(out) == cfg.n_repeats * cfg.n_splits
    assert out["macro_f1"].notna().all()


class _AlwaysFails(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y):
        raise ValueError("deliberate failure")


def test_total_failure_raises_instead_of_returning_empty():
    X, y, g, names = _toy()
    cfg = _small_cfg(search_jobs=1)
    models = {"broken": (_AlwaysFails(), {"clf__C": [0.1, 1.0, 10.0]})}
    with pytest.raises(RuntimeError, match="all 3 fits failed"):
        ev.run_condition(X, y, g, names, cfg, "toy", mode="full", models=models)


def test_partial_failure_is_tolerated():
    """A model that fails is skipped; the others still report."""
    X, y, g, names = _toy()
    cfg = _small_cfg(search_jobs=1)
    models = {"broken": (_AlwaysFails(), {"clf__C": [1.0]}),
              "ok": (LogisticRegression(max_iter=2000), {"clf__C": [0.1, 1.0]})}
    with pytest.warns(UserWarning):
        out = ev.run_condition(X, y, g, names, cfg, "toy", mode="full",
                               models=models)
    assert set(out["model"]) == {"ok"}
    assert len(out) == cfg.n_splits
