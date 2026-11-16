# Two origins of feature redundancy: qEEG and transcranial Doppler

[![tests](https://github.com/OWNER/ncx-redundancy/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/ncx-redundancy/actions/workflows/ci.yml)

Analysis code for *“Two origins of feature redundancy in stroke biosignals:
algebraic construction in transcranial Doppler, shared physiology in
quantitative EEG”* (Rodriguez Rivero et al., 2026), under submission to
*Neuroscience Informatics*.

Redundancy among biosignal descriptors is usually treated as a nuisance to be
pruned before modelling. This study treats it as the thing being measured and
asks where it comes from: from the physics of the recording, or from the way
clinically motivated descriptor sets are habitually assembled. That cannot be
answered inside one modality, so two public stroke-related cohorts are mapped
onto one family taxonomy, cut down to an 11-descriptor **matched core** that is
identical across modalities in family structure and cardinality, and reduced
by one label-agnostic pipeline in which every data-dependent step is re-fitted
inside each subject-grouped training fold.

## Findings at a glance

| | qEEG | TCD |
|---|---|---|
| Mean \|r\|, matched core | 0.229 | 0.316 |
| Descriptors retained, matched core | 7 of 11 | 4 of 11 |
| Descriptors retained, native set | 12 of 17 | 7 of 15 |
| Structure of the redundancy | crosses family boundaries | near block-diagonal |
| Asymmetry family, within-family \|r\| | 0.170 (least redundant) | 0.681 (among the most) |

- **The gap is inferential, not descriptive.** TCD − qEEG = +0.0865, 95 % CI
  [+0.0445, +0.1571], subject-level bootstrap, 2,000 replicates, none at or
  below zero. The ordering holds at all 56 cells of a 14 × 4 grid of
  screening thresholds in both feature spaces.
- **Two origins.** TCD redundancy is *algebraic*: ratio and asymmetry indices
  are closed-form functions of a few envelope landmarks, so overlap stays
  inside the blocks the definitions predict. qEEG redundancy is
  *physiological*: complexity, magnitude and coupling descriptors share
  variance through signal slowing and volume conduction, so it crosses
  families the definitions say are separate. Partialling out the generating
  descriptors leaves more than half of the TCD excess (0.113 → 0.064).
- **Pruning is cheap where it can be tested.** 34 of 40 full-versus-reduced
  comparisons are equivalent within a prespecified ±0.05 Macro-F1. All six
  failures belong to the two tree-based TCD models, which were also the only
  two comparisons with less than 80 % power at that margin.
- **The qEEG null was tested, not inferred.** The qEEG space does not
  discriminate the available motor-imagery label (permutation p = 0.84), and
  that result survives five channel-pooling operators and within-subject
  decoding. It bounds the qEEG conclusions to redundancy structure, which is
  label-agnostic by construction.

The two cohorts share no patients. No fusion model is fitted, and none should
be read into the comparison.

---

## Datasets

Neither dataset is redistributed here. Both are public; download them yourself
and cite them in any work that uses this code. See [`data/README.md`](data/README.md)
for the expected layout.

### 1. EEG: motor imagery in acute stroke

> **[dataset]** Liu, H., & Lv, X. (2022). *EEG datasets of stroke patients.*
> figshare. <https://doi.org/10.6084/m9.figshare.21679035.v5>

This is the depositors' own citation, as given in the data descriptor:

> Liu, H., Wei, P., Wang, H., Lv, X., Duan, W., Li, M., Zhao, Y., Wang, Q.,
> Chen, X., Shi, G., Han, B., & Hao, J. (2024). An EEG motor imagery dataset
> for brain computer interface in acute stroke patients. *Scientific Data*
> **11**, 131. <https://doi.org/10.1038/s41597-023-02787-8>

50 acute stroke patients, 40 motor-imagery trials each (2,000 in total),
500 Hz. The deposit stores 33 traces per trial. The loader identifies the 29
scalp channels, the recording reference, the two EOG channels and the event
marker **from the signals themselves**, not by index, so a revised deposit
cannot silently mis-slice the montage.

### 2. Transcranial Doppler: waveform morphology

> **[dataset]** Thorpe, S. G. (2019). *TCD Clustering Data.* figshare.
> <https://doi.org/10.6084/m9.figshare.9037388>

The figshare record credits Samuel Thorpe as its sole author. The study it
accompanies:

> Thorpe, S. G., Thibeault, C. M., Canac, N., Jalaleddini, K., Dorn, A.,
> Wilk, S. J., Devlin, T., Scalzo, F., & Hamilton, R. B. (2020). Toward
> automated classification of pathological transcranial Doppler waveform
> morphology via spectral clustering. *PLOS ONE* **15**(2), e0228642.
> <https://doi.org/10.1371/journal.pone.0228642>

996 middle cerebral artery insonations from 105 patients across 120 sessions,
labelled large vessel occlusion, in-hospital control or out-of-hospital
control. 951 records from 104 patients survive the requirement that a
bilateral counterpart exist. The deposit ships **derived per-insonation
descriptors, not envelope time series**, which is why the TCD taxonomy has no
complexity or coupling family.

> **Note on `curvature`.** The source publication names this descriptor by
> reference to a velocity curvature index but gives no formula, and the deposit
> ships computed values without a specification. The released values are used
> as distributed.

---

## Installation

```bash
git clone https://github.com/OWNER/ncx-redundancy.git
cd ncx-redundancy
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python ≥ 3.10. Core dependencies are NumPy, SciPy, pandas, scikit-learn and
matplotlib. The analysis of variance is implemented in `ncx/anova.py` rather
than imported, so no statistics package is needed. `requirements-frozen.txt`
pins the exact versions behind the published numbers.

Verify the install before pointing anything at real data:

```bash
pytest -q                      # 56 tests, no data needed
python scripts/run.py selftest # end-to-end on synthetic fixtures
```

`selftest` exercises extraction, fold-internal selection, nested tuning,
metrics, the corrected tests and every figure on synthetic fixtures, and exits
non-zero if any stage produces nothing. **No number it produces belongs
anywhere near a manuscript**: it proves the code runs, not that it is right.

---

## Reproducing the analysis

```bash
# 1. inspect the deposits as distributed (no assumptions about layout)
python scripts/run.py inspect --eeg data/21679035 --tcd data/cluster_data.csv

# 2. extract features -> results/cache/features_{eeg,tcd}.csv
python scripts/run.py features --eeg data/21679035/sourcedata \
                               --tcd data/cluster_data.csv --out results

# 3. structure, performance, equivalence, figures and tables
python scripts/run.py analyse --out results

# 4. analyses added in peer review: ANOVA, power, sensitivity, stability,
#    permutation nulls
python scripts/run_extras.py all --out results

# 5. check the run against the published numbers
python scripts/compare_to_reference.py --fresh results/tables
```

`run.py all` chains stages 2 and 3. On a machine with a wall-clock limit, use
`scripts/run_chunk.py` to evaluate one (modality, condition) slice at a time;
results append to `results/per_fold_metrics.csv` and completed slices are
skipped on re-run.

### Channel-pooling sensitivity analysis

Tests whether the chance-level qEEG result is an artefact of pooling across
the montage. Build the per-channel cache once, then evaluate the variants:

```bash
python scripts/build_pooling_cache.py \
    --source data/21679035/sourcedata --bids data/21679035 \
    --features results/cache/features_eeg.csv \
    --out results/cache/pooling_cache.csv --purge

python scripts/pooling_eval.py --conds reduced --repeats 5 --splits 5 \
                               --iters 10 --rf-trees 150
```

`--purge` deletes each raw recording after processing, so the full 1.8 GB
deposit never has to be resident. The cache is resumable.

### Figures

```bash
python scripts/make_overview.py            # Fig. 1, the design schematic
python scripts/make_graphical_abstract.py  # graphical abstract, 13 x 5 cm
```

---

## Reference results

[`reference_results/`](reference_results/) holds the output files every number
in the manuscript was checked against. They are aggregate statistics only:
no record-level or patient-level data. `tests/test_reference_results.py` pins
the values the paper quotes against these files, so an edit to either side
that breaks agreement fails the test suite.

| Reported result | Produced by | Reproduced from the repository |
|---|---|---|
| Redundancy structure, PCA, family blocks, reduction chains | `run.py analyse` | yes |
| Per-fold metrics, summaries, chance-level tests | `run.py analyse` | yes |
| TOST equivalence (40 comparisons) | `run.py analyse` | yes |
| Fold-wise selection frequency | `run.py analyse` | yes |
| Repeated-measures ANOVA, post-hoc decomposition, TOST power | `run_extras.py perfold` | **yes, verified**: 130 of 130 deterministic values match, permutation p-values within Monte Carlo error |
| Partial correlation, threshold sweep, bootstrap and permutation tests on the gap | `run_extras.py structure` | reconstructed (see below) |
| Nogueira selection stability, 50 partitions | `run_extras.py structure` | reconstructed (see below) |
| Label-permutation nulls, 1,000 shuffles | `run_extras.py permutation` | reconstructed (see below) |
| Channel-pooling variants | `pooling_eval.py` | yes |
| Within-subject decoding (pooling table, lower block) | — | **not scripted** |
| TCD calibration: AUC, Brier score, ECE (Table 12, Fig. 9) | — | **not scripted** |

**About the reconstructed rows.** The analyses added in peer review were first
run by a script that was never committed. `run_extras.py` rebuilds them from
the library functions in `ncx/`, which are the ones originally used and are
unchanged and tested. Where reconstruction could be checked against published
values without the raw data, it reproduces them exactly. The stages that need
the feature caches have been run only on synthetic fixtures here. Run them on
the real data and confirm agreement with `compare_to_reference.py` before
relying on them. The two unscripted analyses keep their published values in
`reference_results/` and are listed as open items in [`CHANGELOG.md`](CHANGELOG.md).

---

## What is in `ncx/`

| Module | Responsibility |
|---|---|
| `config.py` | Every tunable that affects a reported number: bands, thresholds, the feature taxonomies, the matched core, homologous electrode pairs. Methods text and code cannot drift apart. |
| `io.py` | Loaders written against the deposits **as actually distributed**. Where a deposit and its paper disagree, the file wins and the discrepancy is reported. |
| `dsp.py` | Band powers, sample entropy, Higuchi fractal dimension, wPLI, coherence, asymmetry index. |
| `features.py` | Trial- and record-level descriptor construction for both modalities. |
| `redundancy.py` | `RedundancyReducer` (a scikit-learn transformer: correlation screening, then iterative VIF), plus correlation, PCA, family-block and partial-correlation diagnostics. |
| `selectors.py` | mRMR, ReliefF and LASSO baselines, each returning exactly the requested cardinality so the comparison is like for like. |
| `evaluate.py` | Repeated subject-grouped nested cross-validation; per-fold clinical metrics. |
| `stats.py` | Cliff's delta, Nadeau–Bengio corrected resampled *t*, Holm–Bonferroni, TOST with power and minimum detectable margin, permutation nulls, Nogueira stability index. |
| `anova.py` | Factorial and two-way repeated-measures ANOVA with Greenhouse–Geisser correction, within-fold permutation nulls, assumption diagnostics, Friedman and Kendall's *W*. Validated against closed-form references. |
| `sensitivity.py` | Screening-threshold sweep, subject-level bootstrap and column-permutation test on the between-modality redundancy gap. |
| `pooling.py` | Per-channel cache and the five channel-pooling operators. |
| `report.py` | Figures and tables. Colour-vision-safe palette with hatching as a secondary encoding. |

---

## Design decisions worth knowing

**Selection is label-agnostic and fold-internal.** `RedundancyReducer` is a
transformer, not a script step, so correlation screening and VIF elimination
are re-estimated inside every training fold. Leakage is prevented structurally
rather than by intention. `tests/test_pipeline_integrity.py` asserts both
properties directly, including that permuting the labels does not change the
retained set.

**Grouping is by patient, never by record or session.** A patient contributing
several sessions has all of them in the same fold. The effective sample size is
50 and 104 subjects, not 2,000 and 951 records, and that is what sets the width
of every reported interval.

**Total failure is an error, not a gap.** A fold that fails to fit is reported
and skipped. A condition in which *every* fit fails raises, and the driver
exits non-zero. Before version 1.1 this was not so: under scikit-learn ≥ 1.9
every fit failed on a pickling error and the run still exited cleanly with no
results (see [`CHANGELOG.md`](CHANGELOG.md)).

**Two numerically delicate implementations.** The Higuchi normalisation is
`(N-1) / (⌊(N-1)/k⌋ · k²)`. The factor `k` appears twice, once as the sampling
interval and once in the definition of the curve length, and omitting the
second occurrence biases every estimate down by exactly 1.0. Sample entropy
folds `m` diagonal shifts of the single-sample distance matrix instead of
looping over templates. It is algebraically identical to the textbook
formulation and is tested to be bit-identical against it.

**Equivalence, not absence of difference.** Most comparisons here are expected
to be null, so a non-significant difference is not treated as evidence of no
difference. TOST against a prespecified margin is reported with its power and
the smallest margin the design could have resolved at 80 % power.

---

## Citing

If you use this code, please cite the paper and **both datasets above**. See
[`CITATION.cff`](CITATION.cff).

## License

Code released under the MIT License (see [`LICENSE`](LICENSE)). The datasets
carry their own licences at the figshare records linked above; this repository
neither redistributes nor relicenses them.
