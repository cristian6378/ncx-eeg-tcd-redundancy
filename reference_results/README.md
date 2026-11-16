# Reference results

The output files every number in the manuscript was checked against. They are
aggregate statistics: correlation summaries, fold-level scores, test results
and selection frequencies. No record-level or patient-level data is included,
and nothing here identifies a participant.

`tests/test_reference_results.py` pins the values the paper quotes against
these files. `scripts/compare_to_reference.py` compares a fresh run against
them.

| File | Contents | Produced by |
|---|---|---|
| `structural.json` | Per modality and feature space: mean, median and max \|r\|, Spearman check, fraction of pairs ≥ 0.6 and ≥ 0.8, PCA profile, participation ratio, reduction chain, retained and dropped descriptors, final VIF, family-block correlation matrix | `run.py analyse` |
| `per_fold_metrics.csv` | Every metric for every modality × condition × model × repeat × fold (1,250 rows) | `run.py analyse` |
| `summary_metrics.csv` | Means and standard deviations of the above | `run.py analyse` |
| `statistical_comparisons.csv` | Pairwise condition comparisons: Wilcoxon, Nadeau–Bengio corrected *t* (n_test/n_train = 1/4), Cliff's delta, Holm-adjusted *p* | `run.py analyse` |
| `equivalence_tost.csv` | TOST of full versus reduced: 5 models × 2 modalities × 4 metrics = 40 tests, Nadeau–Bengio corrected (primary) | `run.py analyse` |
| `equivalence_tost_uncorrected.csv` | The same 40 tests without the correction, as first computed | `run.py analyse` |
| `vs_chance.csv` | Each model and condition against analytic chance | `run.py analyse` |
| `stability_eeg.csv`, `stability_tcd.csv` | Per-descriptor selection frequency across the 25 evaluation folds | `run.py analyse` |
| `tost_power.csv` | Power of each TOST at ±0.05 and the minimum margin detectable at 80 % power, corrected | `run_extras.py perfold` |
| `tost_power_uncorrected.csv` | The same, without the correction, as first computed | `run_extras.py perfold` |
| `threshold_sweep.csv` | Reduction chain at every cell of the 14 × 4 threshold grid, 4 feature sets (224 chains) | `run_extras.py structure` |
| `selector_stability.csv` | Per-descriptor selection frequency across the 50 partitions of the stability analysis | `run_extras.py structure` |
| `pooling_sensitivity.csv` | Five channel-pooling operators on qEEG | `pooling_eval.py` |
| `within_subject.csv` | Within-subject decoding per pooling variant | not scripted |
| `tcd_calibration.csv` | TCD AUC, Brier score and expected calibration error per model | not scripted |
| `hyperparameters.csv` | Search spaces and fixed settings per classifier | documentation |
| `reviewer_extra.json` | Everything added in peer review: ANOVA and post-hoc, permutation nulls, partial correlation, gap bootstrap and permutation, sweep agreement, Nogueira stability, pooling and within-subject summaries, calibration | see the README's reproducibility map |

## Two selection-stability sources

Two different files report how often each descriptor is selected, and they
answer slightly different questions:

- `stability_{eeg,tcd}.csv` comes from the 25 folds of the main evaluation.
  It supports the statement that 10 of 17 qEEG and 4 of 15 TCD descriptors
  are retained in every fold.
- `selector_stability.csv` and `reviewer_extra.json → selector_stability`
  come from a separate run over 50 partitions, used for the Nogueira index of
  all four selectors (Table 14). Over 50 partitions, 9 qEEG descriptors are
  retained every time, not 10.

Both counts are correct for their own resampling scheme.

## Permutation p-values vary between runs

The qEEG condition effect's within-fold permutation *p* is 0.006 in
`reviewer_extra.json`. A re-run with a different random stream gives 0.003.
Both sit at the resolution limit of 1,000 permutations, and every conclusion
is unchanged. This is Monte Carlo error, not a discrepancy.
