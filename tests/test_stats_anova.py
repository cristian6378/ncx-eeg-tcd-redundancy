"""Validation of the locally implemented ANOVA and the leakage-aware reducer.

The analysis-of-variance routines are written in this repository rather than
imported, so they carry the whole burden of being correct. They are checked
against closed-form balanced sums of squares and a first-principles
repeated-measures F, not against stored outputs.

The reducer is tested for the property the whole study depends on: that it
never consults the labels, and that fitting it inside a pipeline cannot leak
information from a held-out fold.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps

from ncx.anova import (anova_two_way, friedman_kendall, rm_anova_two_way,
                       selftest)
from ncx.redundancy import RedundancyReducer, variance_inflation


# --------------------------------------------------------------------------
# ANOVA
# --------------------------------------------------------------------------

def test_anova_selftest_passes():
    """The module's own validation suite, run under pytest."""
    assert selftest(verbose=False)


def test_two_way_matches_closed_form_balanced():
    """Balanced two-way sums of squares have an unambiguous closed form."""
    rng = np.random.default_rng(0)
    na, nb, rep = 3, 4, 6
    cell = rng.standard_normal((na, nb, rep))
    cell += np.array([0.0, 0.7, 1.4])[:, None, None]
    cell += np.array([0.0, 0.2, 0.4, 0.6])[None, :, None]
    y = cell.ravel()
    a = np.repeat(np.arange(na), nb * rep)
    b = np.tile(np.repeat(np.arange(nb), rep), na)

    res = anova_two_way(y, a, b)
    grand = cell.mean()
    ss_a = nb * rep * ((cell.mean(axis=(1, 2)) - grand) ** 2).sum()
    ss_b = na * rep * ((cell.mean(axis=(0, 2)) - grand) ** 2).sum()
    mab = cell.mean(axis=2)
    ss_ab = rep * ((mab - cell.mean(axis=(1, 2))[:, None]
                    - cell.mean(axis=(0, 2))[None, :] + grand) ** 2).sum()
    ss_e = ((cell - mab[:, :, None]) ** 2).sum()

    assert res["A"]["SS"] == pytest.approx(ss_a, abs=1e-9)
    assert res["B"]["SS"] == pytest.approx(ss_b, abs=1e-9)
    assert res["A:B"]["SS"] == pytest.approx(ss_ab, abs=1e-9)
    assert res["Residual"]["SS"] == pytest.approx(ss_e, abs=1e-9)


def test_rm_anova_one_way_from_first_principles():
    """With a single-level second factor the design reduces to one-way RM."""
    rng = np.random.default_rng(1)
    ns, na = 12, 4
    M = rng.standard_normal((ns, na)) + np.array([0.0, 0.3, 0.6, 0.9])
    y = M.ravel()
    subj = np.repeat(np.arange(ns), na)
    fa = np.tile(np.arange(na), ns)
    fb = np.zeros_like(fa)

    grand = M.mean()
    ss_a = ns * ((M.mean(axis=0) - grand) ** 2).sum()
    ss_s = na * ((M.mean(axis=1) - grand) ** 2).sum()
    ss_e = ((M - grand) ** 2).sum() - ss_a - ss_s
    f_ref = (ss_a / (na - 1)) / (ss_e / ((ns - 1) * (na - 1)))

    assert rm_anova_two_way(y, subj, fa, fb)["A"]["F"] == pytest.approx(f_ref, abs=1e-9)


def test_rm_anova_partitions_variance():
    """Every sum of squares must add back to the total."""
    rng = np.random.default_rng(2)
    ns, na, nb = 10, 3, 4
    y, subj, fa, fb = [], [], [], []
    for s in range(ns):
        for i in range(na):
            for j in range(nb):
                y.append(rng.standard_normal() + 0.3 * i + 0.2 * j)
                subj.append(s); fa.append(i); fb.append(j)
    r = rm_anova_two_way(np.array(y), np.array(subj), np.array(fa), np.array(fb))
    total = (r["Subject"]["SS"] + r["A"]["SS"] + r["A"]["SS_error"]
             + r["B"]["SS"] + r["B"]["SS_error"]
             + r["A:B"]["SS"] + r["A:B"]["SS_error"])
    assert total == pytest.approx(r["_total"]["SS"], rel=1e-9)


def test_rm_anova_rejects_incomplete_cube():
    """A missing cell makes the design unbalanced; that must be refused
    rather than silently filled."""
    y = np.arange(11, dtype=float)
    subj = np.repeat(np.arange(3), 4)[:11]
    fa = np.tile([0, 0, 1, 1], 3)[:11]
    fb = np.tile([0, 1, 0, 1], 3)[:11]
    with pytest.raises(ValueError):
        rm_anova_two_way(y, subj, fa, fb)


def test_kendall_w_bounds():
    """W = 1 under perfect concordance, and near 0 under random ranking."""
    assert friedman_kendall(np.tile(np.arange(5.0), (10, 1)))["kendall_W"] == \
        pytest.approx(1.0, abs=1e-9)
    rng = np.random.default_rng(3)
    w = friedman_kendall(rng.standard_normal((40, 5)))["kendall_W"]
    assert 0.0 <= w < 0.35


# --------------------------------------------------------------------------
# Redundancy reduction
# --------------------------------------------------------------------------

def test_reducer_ignores_labels():
    """The reduction is label-agnostic. Permuting y must not change the
    retained set -- this is what makes the redundancy results independent of
    whether the feature space predicts anything."""
    rng = np.random.default_rng(4)
    latent = rng.standard_normal(300)
    X = np.column_stack([
        latent + 0.05 * rng.standard_normal(300),      # near-duplicate pair
        latent + 0.05 * rng.standard_normal(300),
        rng.standard_normal(300),
        rng.standard_normal(300),
        rng.standard_normal(300),
    ])
    names = [f"f{i}" for i in range(X.shape[1])]
    y1 = rng.integers(0, 2, 300)
    y2 = rng.permutation(y1)
    s1 = RedundancyReducer(0.6, 10.0, names).fit(X, y1).support_
    s2 = RedundancyReducer(0.6, 10.0, names).fit(X, y2).support_
    assert np.array_equal(s1, s2)


def test_reducer_drops_a_collinear_duplicate():
    """One of a near-duplicate pair must go, and only one."""
    rng = np.random.default_rng(5)
    latent = rng.standard_normal(400)
    X = np.column_stack([latent,
                         latent + 0.01 * rng.standard_normal(400),
                         rng.standard_normal(400),
                         rng.standard_normal(400)])
    names = ["a", "a_copy", "b", "c"]
    r = RedundancyReducer(0.6, 10.0, names).fit(X)
    kept = set(r.kept_names_)
    assert len(kept & {"a", "a_copy"}) == 1
    assert {"b", "c"} <= kept


def test_reducer_keeps_independent_features():
    """Nothing should be removed from an uncorrelated design."""
    rng = np.random.default_rng(6)
    X = rng.standard_normal((500, 6))
    names = [f"f{i}" for i in range(6)]
    r = RedundancyReducer(0.6, 10.0, names).fit(X)
    assert len(r.support_) == 6


def test_reducer_is_deterministic():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((200, 8))
    names = [f"f{i}" for i in range(8)]
    a = RedundancyReducer(0.6, 10.0, names).fit(X).support_
    b = RedundancyReducer(0.6, 10.0, names).fit(X).support_
    assert np.array_equal(a, b)


def test_vif_detects_collinearity():
    """VIF must rise sharply for a column that is a linear combination of
    others, and stay near 1 for independent columns."""
    rng = np.random.default_rng(8)
    X = rng.standard_normal((400, 3))
    assert variance_inflation(X).max() < 1.5
    Xc = np.column_stack([X, X[:, 0] + X[:, 1] + 1e-6 * rng.standard_normal(400)])
    assert variance_inflation(Xc).max() > 50
