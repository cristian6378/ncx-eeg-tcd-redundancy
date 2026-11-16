"""The Nadeau-Bengio correction, as applied to paired and equivalence tests.

An earlier version computed n_train as (J - 1) * n_test with J the number of
folds across all repeats. For 5 x 5 folds that is 24 * n_test instead of
4 * n_test, which under-corrected every paired test, and the equivalence
tests applied no correction at all. Both are pinned here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ncx.stats import (_mean_split_sizes, compare_conditions,
                       nadeau_bengio_ratio, tost_equivalence, tost_power)


def _design(n_rep=5, n_fold=5, n=951, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_rep):
        sizes = np.array_split(np.arange(n), n_fold)
        for f, idx in enumerate(sizes):
            for cond, shift in (("full", 0.0), ("reduced", -0.01)):
                rows.append({"modality": "m", "model": "x", "condition": cond,
                             "repeat": r, "fold": f, "n_test": len(idx),
                             "macro_f1": 0.6 + shift + rng.normal(0, 0.05)})
    return pd.DataFrame(rows)


def test_split_sizes_give_one_quarter_for_five_folds():
    pf = _design()
    s = pf[pf.condition == "full"].set_index(
        ["modality", "model", "repeat", "fold"])["n_test"]
    n_test, n_train = _mean_split_sizes(s)
    assert n_test == pytest.approx(951 / 5)
    assert n_train == pytest.approx(951 - 951 / 5)
    assert nadeau_bengio_ratio(pf)["m"] == pytest.approx(0.25)


def test_compare_conditions_uses_the_correct_ratio():
    pf = _design()
    out = compare_conditions(pf, "macro_f1", "full", "reduced")
    key = ["modality", "model", "repeat", "fold"]
    a = pf[pf.condition == "full"].set_index(key).macro_f1
    b = pf[pf.condition == "reduced"].set_index(key).macro_f1
    d = (a - b).to_numpy()
    J = d.size
    t = d.mean() / np.sqrt(d.var(ddof=1) * (1 / J + 0.25))
    p = 2 * (1 - stats.t.cdf(abs(t), J - 1))
    assert out.loc[0, "p_nadeau_bengio"] == pytest.approx(p, rel=1e-9)


def test_tost_without_correction_is_unchanged():
    d = np.random.default_rng(1).normal(0.01, 0.02, 25)
    se = d.std(ddof=1) / np.sqrt(25)
    res = tost_equivalence(d, 0.05)
    half = res["ci90_hi"] - res["mean"]
    assert half == pytest.approx(stats.t.ppf(0.95, 24) * se)


def test_tost_correction_widens_the_interval_by_the_expected_factor():
    d = np.random.default_rng(2).normal(0.01, 0.02, 25)
    plain = tost_equivalence(d, 0.05)
    corr = tost_equivalence(d, 0.05, test_train_ratio=0.25)
    ratio = (corr["ci90_hi"] - corr["mean"]) / (plain["ci90_hi"] - plain["mean"])
    assert ratio == pytest.approx(np.sqrt((1 / 25 + 0.25) / (1 / 25)))
    assert corr["p_tost"] >= plain["p_tost"]


def test_corrected_power_is_never_higher():
    for sd in (0.02, 0.05, 0.09):
        assert tost_power(sd, 25, 0.05, test_train_ratio=0.25) \
            <= tost_power(sd, 25, 0.05) + 1e-12
