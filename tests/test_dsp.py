"""Numerical validation of the signal-processing primitives.

These three are tested against reference implementations and known limits
rather than against stored outputs, because each has a failure mode that is
silent: a wrong result that still looks like a plausible number. Sample
entropy is computed by a vectorised route that must agree with the textbook
definition exactly; the Higuchi normalisation contains a factor whose omission
shifts every estimate by precisely 1.0; and the asymmetry index must be bounded
and sign-symmetric or the pDBSI family means nothing.
"""
from __future__ import annotations

import numpy as np
import pytest

from ncx import dsp


# --------------------------------------------------------------------------
# Sample entropy
# --------------------------------------------------------------------------

def _sampen_reference(x, m=2, r=None):
    """Textbook sample entropy: explicit loop over template pairs."""
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    sd = np.std(x)
    if sd == 0:
        return np.nan
    r = 0.2 * sd if r is None else r

    def phi(mm):
        idx = np.arange(n - mm + 1)
        tpl = x[idx[:, None] + np.arange(mm)]
        count = 0
        for i in range(tpl.shape[0] - 1):
            d = np.max(np.abs(tpl[i + 1:] - tpl[i]), axis=1)
            count += int(np.sum(d <= r))
        return count

    b, a = phi(m), phi(m + 1)
    if b == 0 or a == 0:
        return np.nan
    return float(-np.log(a / b))


@pytest.mark.parametrize("name", ["white", "brown", "sine", "sine_noise"])
def test_sample_entropy_matches_reference(name):
    """The vectorised form must be bit-identical to the explicit loop."""
    rng = np.random.default_rng(0)
    n = 600
    sig = {
        "white": rng.standard_normal(n),
        "brown": np.cumsum(rng.standard_normal(n)),
        "sine": np.sin(2 * np.pi * 10 * np.arange(n) / 500),
        "sine_noise": (np.sin(2 * np.pi * 10 * np.arange(n) / 500)
                       + 0.3 * rng.standard_normal(n)),
    }[name]
    fast, ref = dsp.sample_entropy(sig), _sampen_reference(sig)
    if np.isnan(ref):
        assert np.isnan(fast)
    else:
        assert fast == pytest.approx(ref, abs=1e-12)


def test_sample_entropy_orders_by_regularity():
    """A regular signal must score below a random one."""
    rng = np.random.default_rng(1)
    t = np.arange(800)
    periodic = np.sin(2 * np.pi * 8 * t / 500)
    noise = rng.standard_normal(800)
    assert dsp.sample_entropy(periodic) < dsp.sample_entropy(noise)


def test_sample_entropy_degenerate_is_nan():
    """A flat channel propagates as missing, never as zero or infinity."""
    assert np.isnan(dsp.sample_entropy(np.zeros(500)))


# --------------------------------------------------------------------------
# Higuchi fractal dimension
# --------------------------------------------------------------------------

def test_higuchi_in_theoretical_range():
    """HFD of a 1-D series lies in [1, 2] in the limit.

    Finite-sample estimates overshoot the upper bound slightly -- white noise
    of 2000 points returns about 2.008 -- so the assertion allows a small
    margin above 2. The failure this guards against is not a 1% overshoot but
    the normalisation bug that omits one factor of k and shifts every estimate
    down by exactly 1.0, landing white noise near 1.0 and smooth signals near
    0.03.
    """
    rng = np.random.default_rng(2)
    for sig in (rng.standard_normal(2000),
                np.cumsum(rng.standard_normal(2000)),
                np.sin(2 * np.pi * 10 * np.arange(2000) / 500)):
        fd = dsp.higuchi_fd(sig)
        assert 1.0 <= fd <= 2.05, f"HFD out of range: {fd}"


def test_higuchi_orders_white_above_brown():
    """White noise fills the plane more densely than a Brownian path."""
    rng = np.random.default_rng(3)
    white = dsp.higuchi_fd(rng.standard_normal(2000))
    brown = dsp.higuchi_fd(np.cumsum(rng.standard_normal(2000)))
    assert white > brown


def test_higuchi_white_noise_near_two():
    """White noise should approach the upper limit, not sit near 1."""
    rng = np.random.default_rng(4)
    assert dsp.higuchi_fd(rng.standard_normal(4000)) > 1.8


# --------------------------------------------------------------------------
# Asymmetry index
# --------------------------------------------------------------------------

def test_asymmetry_zero_when_balanced():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    assert dsp.asymmetry_index(v, v) == pytest.approx(0.0, abs=1e-12)


def test_asymmetry_bounded_and_symmetric():
    rng = np.random.default_rng(5)
    left = rng.random(12) + 0.1
    right = rng.random(12) + 0.1
    a = dsp.asymmetry_index(left, right)
    b = dsp.asymmetry_index(right, left)
    assert 0.0 <= a <= 1.0
    assert a == pytest.approx(b, abs=1e-12), "index must not depend on side order"
