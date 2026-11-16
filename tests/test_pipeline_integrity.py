"""Tests of the properties the published claims actually rest on.

Three things would invalidate the study if they were untrue, and none of them
is visible by inspecting a results table: that no patient's data can appear on
both sides of a split; that selection is re-fitted inside each training fold
rather than once on everything; and that the statistical helpers behave
correctly in the null case. Each is asserted here directly.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ncx.config import Config, feature_order, matched_core
from ncx.redundancy import RedundancyReducer
from ncx.selectors import make_selector
from ncx.stats import (bootstrap_ci, cliffs_delta, holm_bonferroni,
                       nadeau_bengio_t, tost_equivalence)


# --------------------------------------------------------------------------
# Grouping: the leakage guarantee
# --------------------------------------------------------------------------

def test_no_subject_spans_a_split():
    """Grouping by patient must keep every record of a patient in one side.

    This is the guarantee the whole validation design rests on: records from
    one patient resemble each other far more than they resemble anyone
    else's, so a subject straddling a split inflates every score.
    """
    rng = np.random.default_rng(0)
    n_subj, per_subj = 40, 25
    groups = np.repeat(np.arange(n_subj), per_subj)
    y = np.repeat(rng.integers(0, 3, n_subj), per_subj)     # label is per patient
    X = rng.standard_normal((n_subj * per_subj, 6))

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for train, test in cv.split(X, y, groups):
        assert not (set(groups[train]) & set(groups[test])), \
            "a subject appeared in both the training and the test partition"


def test_sessions_of_a_patient_stay_together():
    """Longitudinal records: grouping is on patient, not on session, so
    several sessions of one patient cannot be split across folds."""
    rng = np.random.default_rng(1)
    patients = np.repeat(np.arange(30), 9)                  # 3 sessions x 3 records
    sessions = np.tile(np.repeat(np.arange(3), 3), 30)
    y = np.repeat(rng.integers(0, 2, 30), 9)
    X = rng.standard_normal((270, 5))

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for train, test in cv.split(X, y, groups=patients):
        for p in np.unique(patients[test]):
            assert p not in patients[train]
            # and every session of that patient is on the test side
            assert set(sessions[patients == p]) == set(sessions[test][patients[test] == p])


# --------------------------------------------------------------------------
# Fold-internal fitting
# --------------------------------------------------------------------------

def test_selection_is_refitted_per_fold():
    """A reducer inside a Pipeline must see only the training rows.

    If selection were fitted once on the full matrix, the support would be
    identical across folds by construction. Here the folds differ, which is
    the observable signature of in-fold fitting.
    """
    rng = np.random.default_rng(2)
    n = 400
    groups = np.repeat(np.arange(40), 10)
    y = np.repeat(rng.integers(0, 2, 40), 10)
    latent = rng.standard_normal(n)
    X = np.column_stack([latent + 0.3 * rng.standard_normal(n),
                         latent + 0.3 * rng.standard_normal(n),
                         latent + 0.3 * rng.standard_normal(n),
                         rng.standard_normal(n),
                         rng.standard_normal(n)])
    names = [f"f{i}" for i in range(X.shape[1])]

    supports = []
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for train, _ in cv.split(X, y, groups):
        pipe = Pipeline([("s", StandardScaler()),
                         ("sel", RedundancyReducer(0.6, 10.0, names))])
        pipe.fit(X[train], y[train])
        supports.append(tuple(pipe.named_steps["sel"].support_))
    assert len(set(supports)) > 1, \
        "every fold selected the same subset; selection may not be fold-internal"


@pytest.mark.parametrize("name", ["mrmr", "relieff", "lasso"])
def test_supervised_selectors_honour_requested_cardinality(name):
    """Baselines must return exactly k features, or the comparison with the
    unsupervised pipeline is not like-for-like."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((250, 12))
    y = rng.integers(0, 3, 250)
    k = 7
    sel = make_selector(name, k, random_state=0).fit(X, y)
    assert len(sel.support_) == k
    assert sel.transform(X).shape == (250, k)


# --------------------------------------------------------------------------
# Taxonomy bookkeeping
# --------------------------------------------------------------------------

def test_matched_core_is_cardinality_matched():
    """The comparative claim requires the two cores to be the same size and
    to share their family structure."""
    ce, ct = matched_core("eeg"), matched_core("tcd")
    assert len(ce) == len(ct), "matched core must have equal cardinality"
    assert len(ce) == 11


def test_native_sets_have_expected_sizes():
    assert len(feature_order("eeg")) == 17
    assert len(feature_order("tcd")) == 15


def test_homologous_pairs_are_bilateral():
    """pDBSI pairs must be genuinely left/right and never repeat an electrode."""
    cfg = Config()
    seen = set()
    for left, right in cfg.homologous_pairs:
        assert left.upper() != right.upper()
        assert left.upper() not in seen and right.upper() not in seen
        seen |= {left.upper(), right.upper()}


# --------------------------------------------------------------------------
# Statistical helpers under the null
# --------------------------------------------------------------------------

def test_cliffs_delta_bounds():
    rng = np.random.default_rng(4)
    a = rng.standard_normal(200)
    assert abs(cliffs_delta(a, a.copy())) < 1e-9
    assert cliffs_delta(a + 10, a) == pytest.approx(1.0, abs=1e-9)
    assert cliffs_delta(a - 10, a) == pytest.approx(-1.0, abs=1e-9)


def test_nadeau_bengio_is_conservative():
    """The correction inflates variance, so its p-value must never be
    smaller than the uncorrected paired t-test on the same differences."""
    from scipy import stats as sps
    rng = np.random.default_rng(5)
    d = rng.standard_normal(25) * 0.05 + 0.02
    _, p_corrected = nadeau_bengio_t(d, n_train=1600, n_test=400)
    _, p_plain = sps.ttest_1samp(d, 0.0)
    assert p_corrected >= p_plain


def test_holm_is_conservative_and_ordered():
    p = [0.001, 0.02, 0.04, 0.5]
    adj, _ = holm_bonferroni(p)
    assert all(a >= b for a, b in zip(adj, p))
    assert all(adj[i] <= adj[i + 1] for i in range(len(adj) - 1))


def test_tost_declares_equivalence_for_negligible_differences():
    """Fold differences far inside the margin must be declared equivalent."""
    rng = np.random.default_rng(6)
    diff = rng.standard_normal(60) * 0.002           # well within +/- 0.05
    out = tost_equivalence(diff, bound=0.05)
    assert out["p_tost"] < 0.05 and out["equivalent"] is True


def test_tost_refuses_equivalence_for_a_real_difference():
    """A difference larger than the margin must not be declared equivalent."""
    rng = np.random.default_rng(7)
    diff = rng.standard_normal(60) * 0.01 + 0.12     # far outside +/- 0.05
    out = tost_equivalence(diff, bound=0.05)
    assert out["p_tost"] > 0.05 and out["equivalent"] is False


def test_bootstrap_ci_contains_the_sample_mean_and_narrows_with_n():
    """Deterministic properties of the interval, rather than whether one
    particular draw happens to bracket zero."""
    rng = np.random.default_rng(8)
    small = rng.standard_normal(50) * 0.01
    large = rng.standard_normal(2000) * 0.01
    lo_s, hi_s = bootstrap_ci(small, random_state=0)
    lo_l, hi_l = bootstrap_ci(large, random_state=0)
    assert lo_s < small.mean() < hi_s
    assert lo_l < large.mean() < hi_l
    assert (hi_l - lo_l) < (hi_s - lo_s), "interval should narrow as n grows"
