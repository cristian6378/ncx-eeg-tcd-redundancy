"""The numbers the manuscript quotes, pinned against reference_results/.

Every value asserted here appears in the text or a table of the paper. If a
reference file changes, or the manuscript is edited to a number these files
do not support, one of these tests says which. Several assertions exist
because the corresponding manuscript statement was once wrong:

* the maximum |r| of the qEEG set is 0.997, not the TCD value of 0.963;
* six of the 40 equivalence tests fail, not two -- the six belong to two
  models;
* the threshold grid has 56 cells; 224 counts reduction chains;
* the TCD between-family correlations reach 0.204, not 0.20;
* the clinical specificity range for the linear and kernel models starts at
  0.8185, not 0.8205.

The last test re-runs the reconstructed post-review analysis on the
published per-fold metrics and requires every deterministic value to match.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference_results"


def _json(name):
    return json.loads((REF / name).read_text())


def _csv(name):
    return pd.read_csv(REF / name)


# --------------------------------------------------------------- structure

def test_matched_core_redundancy_gap():
    s = _json("structural.json")
    assert s["eeg"]["core"]["mean_r"] == pytest.approx(0.229, abs=5e-4)
    assert s["tcd"]["core"]["mean_r"] == pytest.approx(0.316, abs=5e-4)
    assert (s["eeg"]["core"]["n"], s["eeg"]["core"]["retained"]) == (11, 7)
    assert (s["tcd"]["core"]["n"], s["tcd"]["core"]["retained"]) == (11, 4)


def test_native_chains_and_vif_removes_nothing():
    s = _json("structural.json")
    assert (s["eeg"]["native"]["n"], s["eeg"]["native"]["retained"]) == (17, 12)
    assert (s["tcd"]["native"]["n"], s["tcd"]["native"]["retained"]) == (15, 7)
    for m in ("eeg", "tcd"):
        for space in ("native", "core"):
            assert not s[m][space]["dropped_vif"]


def test_maximum_correlation_differs_by_modality():
    s = _json("structural.json")
    assert s["eeg"]["native"]["max_r"] == pytest.approx(0.997, abs=5e-4)
    assert s["eeg"]["core"]["max_r"] == pytest.approx(0.997, abs=5e-4)
    assert s["tcd"]["native"]["max_r"] == pytest.approx(0.963, abs=5e-4)
    assert s["tcd"]["core"]["max_r"] == pytest.approx(0.963, abs=5e-4)


def _blocks(modality):
    fb = _json("structural.json")[modality]["family_blocks"]
    return pd.DataFrame(fb["data"], index=fb["index"], columns=fb["columns"])


def test_family_count_is_seven_with_three_shared():
    e, t = set(_blocks("eeg").index), set(_blocks("tcd").index)
    assert len(e) == 6 and len(t) == 4
    assert len(e | t) == 7
    assert e & t == {"magnitude", "ratio", "asymmetry"}


def test_tcd_is_block_diagonal_with_one_algebraic_exception():
    b = _blocks("tcd")
    off = {(i, j): b.loc[i, j] for i in b.index for j in b.columns if i < j}
    exception = off.pop(("magnitude", "ratio"))
    assert exception == pytest.approx(0.344, abs=5e-4)
    assert 0.058 < min(off.values()) and max(off.values()) < 0.21


def test_asymmetry_inversion():
    assert _blocks("eeg").loc["asymmetry", "asymmetry"] == pytest.approx(0.170, abs=5e-4)
    assert _blocks("tcd").loc["asymmetry", "asymmetry"] == pytest.approx(0.681, abs=5e-4)


def test_partial_correlation_leaves_most_of_the_gap():
    pc = _json("reviewer_extra.json")["partial_correlation"]
    raw = pc["tcd"]["mean_abs_raw"] - pc["eeg"]["mean_abs_raw"]
    part = pc["tcd"]["mean_abs_partial"] - pc["eeg"]["mean_abs_partial"]
    assert raw == pytest.approx(0.113, abs=5e-4)
    assert part == pytest.approx(0.064, abs=5e-4)
    assert 0.5 < part / raw < 0.6        # "more than half" remains


# --------------------------------------------------------------- inference

def test_bootstrap_gap():
    b = _json("reviewer_extra.json")["redundancy_gap_bootstrap"]
    assert b["observed_gap"] == pytest.approx(0.0865, abs=5e-5)
    assert (b["ci_lo"], b["ci_hi"]) == pytest.approx((0.0445, 0.1571), abs=5e-5)
    assert b["p_gap_le_0"] == 0.0 and b["n_boot"] == 2000


def test_threshold_grid_has_56_cells_per_space_and_never_reverses():
    sw = _csv("threshold_sweep.csv")
    for s in ("eeg/native", "tcd/native", "eeg/core", "tcd/core"):
        assert (sw.set == s).sum() == 56
    assert len(sw) == 224                          # chains, not grid cells
    ag = _json("reviewer_extra.json")["threshold_sweep_agreement"]
    for space in ("native", "matched core"):
        assert ag[space]["n_cells"] == 56
        assert ag[space]["frac_ordering_holds"] == 1.0


def test_equivalence_34_of_40_six_failures_two_models():
    eq = _csv("equivalence_tost.csv")
    assert len(eq) == 40 and eq.equivalent.sum() == 34
    fail = eq[~eq.equivalent]
    assert len(fail) == 6
    assert set(fail.modality) == {"tcd"}
    assert set(fail.model) == {"Decision Tree", "Random Forest"}
    assert "specificity_macro" not in set(fail.metric)


def test_tost_power_explains_both_failures():
    pw = _csv("tost_power.csv").set_index(["modality", "model"])
    assert pw.loc[("tcd", "Decision Tree"), "power_at_0p05"] == pytest.approx(0.754)
    assert pw.loc[("tcd", "Random Forest"), "power_at_0p05"] == pytest.approx(0.803)
    assert pw.loc[("tcd", "Decision Tree"), "min_margin_80pct"] >= 0.0498
    assert (pw.loc["eeg", "power_at_0p05"] == 1.0).all()


def test_permutation_nulls():
    p = _json("reviewer_extra.json")["permutation_1000"]
    assert p["eeg"]["p_permutation"] > 0.5                 # qEEG at chance
    assert p["tcd"]["p_permutation"] == pytest.approx(1 / 1001)
    z = (p["tcd"]["observed"] - p["tcd"]["null_mean"]) / p["tcd"]["null_sd"]
    assert z == pytest.approx(5.85, abs=0.01)


def test_clinical_ranges_for_linear_and_kernel_models():
    sm = _csv("summary_metrics.csv")
    lk = sm[(sm.modality == "tcd") & (sm.condition == "reduced")
            & sm.model.isin(["Logistic Regression", "SVM (Linear)", "SVM (RBF)"])]
    assert (lk.sensitivity_macro_mean.min(), lk.sensitivity_macro_mean.max()) \
        == pytest.approx((0.6114, 0.6147), abs=1e-4)
    assert (lk.specificity_macro_mean.min(), lk.specificity_macro_mean.max()) \
        == pytest.approx((0.8185, 0.8269), abs=1e-4)


def test_fold_selection_stability_counts():
    for m, always in (("eeg", 10), ("tcd", 4)):
        st = _csv(f"stability_{m}.csv")
        assert (st.selection_frequency == 1.0).sum() == always


def test_nogueira_table_is_from_fifty_partitions():
    st = _json("reviewer_extra.json")["selector_stability"]
    assert all(v["M"] == 50 for v in st.values())
    assert st["eeg/reduced"]["stability"] == pytest.approx(0.826, abs=5e-4)
    assert st["tcd/reduced"]["stability"] == pytest.approx(0.633, abs=5e-4)


# ---------------------------------------------- reconstruction reproduces

def test_reconstructed_perfold_stage_matches_published(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_extras
    from compare_to_reference import compare_json

    pf = pd.read_csv(REF / "per_fold_metrics.csv")
    seed = 20260927
    fresh = {"anova": run_extras.anova_block(pf, "macro_f1", 50, seed),
             "anova_posthoc": run_extras.posthoc_block(pf, "macro_f1", 50, seed)}
    fresh = json.loads(json.dumps(run_extras._jsonable(fresh)))
    res = compare_json(_json("reviewer_extra.json"), fresh, 1e-3, 1.0)
    det = res[res.kind == "deterministic"]
    assert len(det) >= 130
    assert det.ok.all(), det[~det.ok].to_string()

    power = run_extras.tost_power_table(pf, "macro_f1")
    ref = _csv("tost_power.csv")
    m = ref.merge(power, on=["modality", "model"], suffixes=("_r", "_f"))
    for c in ("sd_of_differences", "power_at_0p05", "min_margin_80pct"):
        assert np.allclose(m[f"{c}_r"], m[f"{c}_f"], atol=1e-4), c
