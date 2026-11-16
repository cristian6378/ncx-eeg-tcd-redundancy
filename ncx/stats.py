"""Inference on paired cross-validation results.

Repeated k-fold folds are not independent: the same observations recur across
repeats, so the naive paired test understates variance and returns p-values
that look far more decisive than the evidence warrants. Both a conventional
Wilcoxon test and the Nadeau-Bengio corrected t-statistic are therefore
reported, alongside a distribution-free effect size, and the manuscript leans
on the effect size rather than the p-value.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def cliffs_delta(a, b) -> float:
    """Cliff's delta for paired-sample location shift, in [-1, 1]."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size == 0:
        return np.nan
    gt = np.sum(a[:, None] > b[None, :])
    lt = np.sum(a[:, None] < b[None, :])
    return float((gt - lt) / (a.size * b.size))


def nadeau_bengio_t(diff, n_train: int, n_test: int):
    """Corrected resampled t-test for repeated cross-validation.

    Inflates the variance of the mean difference by (1/J + n_test/n_train) to
    account for the overlap between training sets across folds. Without this
    correction, repeated-CV p-values are known to be anticonservative by
    orders of magnitude.
    """
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    J = d.size
    if J < 2:
        return np.nan, np.nan
    var = d.var(ddof=1)
    if var <= 0:
        return np.nan, np.nan
    corr = (1.0 / J) + (n_test / max(n_train, 1))
    t = d.mean() / np.sqrt(var * corr)
    p = 2.0 * (1.0 - stats.t.cdf(abs(t), df=J - 1))
    return float(t), float(p)


def bootstrap_ci(diff, n_boot: int = 10000, alpha: float = 0.05,
                 random_state: int = 0):
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(random_state)
    means = rng.choice(d, size=(n_boot, d.size), replace=True).mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def holm_bonferroni(pvals, alpha: float = 0.05):
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    adj = np.full(p.shape, np.nan)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return adj, np.zeros(p.shape, dtype=bool)
    order = idx[np.argsort(p[idx])]
    m = order.size
    running = 0.0
    for rank, j in enumerate(order):
        val = min(1.0, (m - rank) * p[j])
        running = max(running, val)
        adj[j] = running
    return adj, (adj < alpha) & ok


def compare_conditions(per_fold: pd.DataFrame, metric: str,
                       cond_a: str, cond_b: str, cfg=None,
                       alpha: float = 0.05) -> pd.DataFrame:
    """Per-model paired comparison of two feature-space conditions."""
    rows = []
    key = ["modality", "model", "repeat", "fold"]
    a = per_fold[per_fold.condition == cond_a].set_index(key)[metric]
    b = per_fold[per_fold.condition == cond_b].set_index(key)[metric]
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    sizes = (per_fold[per_fold.condition == cond_a]
             .set_index(key)[["n_test"]].loc[common])

    for (mod, model), grp in a.groupby(level=["modality", "model"]):
        av = grp.to_numpy(dtype=float)
        bv = b.loc[grp.index].to_numpy(dtype=float)
        d = av - bv
        n_test, n_train = _mean_split_sizes(sizes.loc[grp.index, "n_test"])
        try:
            w_p = stats.wilcoxon(av, bv, zero_method="zsplit").pvalue
        except Exception:                       # noqa: BLE001
            w_p = np.nan
        t_nb, p_nb = nadeau_bengio_t(d, n_train=int(n_train), n_test=int(n_test))
        lo, hi = bootstrap_ci(d, alpha=alpha,
                              random_state=getattr(cfg, "random_state", 0))
        rows.append({
            "modality": mod, "model": model, "metric": metric,
            f"{cond_a}_mean": float(np.nanmean(av)),
            f"{cond_a}_sd": float(np.nanstd(av, ddof=1)),
            f"{cond_b}_mean": float(np.nanmean(bv)),
            f"{cond_b}_sd": float(np.nanstd(bv, ddof=1)),
            "delta_mean": float(np.nanmean(d)),
            "delta_rel_pct": float(100 * np.nanmean(d) / np.nanmean(av))
            if np.nanmean(av) else np.nan,
            "ci95_lo": lo, "ci95_hi": hi,
            "cliffs_delta": cliffs_delta(av, bv),
            "p_wilcoxon": w_p, "t_nadeau_bengio": t_nb, "p_nadeau_bengio": p_nb,
            "n_folds": int(len(d)),
        })
    out = pd.DataFrame(rows)
    if len(out):
        for col, tgt in (("p_wilcoxon", "p_wilcoxon_holm"),
                         ("p_nadeau_bengio", "p_nadeau_bengio_holm")):
            adj, rej = holm_bonferroni(out[col].to_numpy(), alpha)
            out[tgt] = adj
            out[tgt.replace("p_", "sig_")] = rej
    return out


def summarise_per_fold(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- SD of every metric, by modality / condition / model."""
    metrics = ["macro_f1", "balanced_accuracy", "auc_macro_ovr",
               "sensitivity_macro", "specificity_macro", "ppv_macro",
               "npv_macro", "n_features"]
    g = per_fold.groupby(["modality", "condition", "model"])
    out = g[metrics].agg(["mean", "std"]).round(4)
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


# --------------------------------------------------------------------------
# Extended inference
# --------------------------------------------------------------------------

def _mean_split_sizes(n_test_per_fold: pd.Series) -> tuple[float, float]:
    """Mean test- and training-set sizes of a repeated k-fold design.

    `n_test_per_fold` is indexed by (..., repeat, fold). Within one repeat
    the test sets partition the data, so the dataset size N is their sum and
    each fold trains on N - n_test. (An earlier version took n_train as
    (J - 1) * n_test with J the number of folds across *all* repeats, which
    for 5 x 5 folds overstated n_train sixfold and under-corrected every
    Nadeau-Bengio test.)
    """
    s = n_test_per_fold.astype(float)
    reps = s.index.get_level_values("repeat")
    N = float(s.groupby(reps).sum().mean())
    n_test = float(s.mean())
    return n_test, max(1.0, N - n_test)


def nadeau_bengio_ratio(per_fold: pd.DataFrame) -> dict:
    """n_test / n_train per modality, for the corrected variance."""
    out = {}
    key = ["modality", "model", "condition", "repeat", "fold"]
    for mod, g in per_fold.groupby("modality"):
        one = g[(g.model == g.model.iloc[0]) & (g.condition == g.condition.iloc[0])]
        n_test, n_train = _mean_split_sizes(one.set_index(key)["n_test"])
        out[mod] = n_test / n_train
    return out


def tost_equivalence(diff, bound: float, alpha: float = 0.05,
                     test_train_ratio: float = 0.0):
    """Two one-sided tests for equivalence within +/- `bound`.

    A non-significant difference test is not evidence of equivalence: it fails
    to reject, which is not the same as supporting, the null. When the claim
    being made is that pruning costs nothing, TOST is the test that can
    actually support it. The equivalence margin must be set on substantive
    grounds and stated, not chosen after seeing the data.
    """
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 3:
        return {"p_tost": np.nan, "equivalent": False, "n": int(n)}
    # With test_train_ratio > 0 the variance of the mean difference is
    # inflated by (1/J + n_test/n_train), as in nadeau_bengio_t: repeated
    # folds share training data, and treating them as independent makes an
    # equivalence claim look better supported than it is.
    se = d.std(ddof=1) * np.sqrt(1.0 / n + test_train_ratio)
    if se == 0:
        eq = abs(d.mean()) < bound
        return {"p_tost": 0.0 if eq else 1.0, "equivalent": bool(eq), "n": int(n),
                "mean": float(d.mean()), "bound": float(bound)}
    df = n - 1
    t_lo = (d.mean() + bound) / se          # H0: mu <= -bound
    t_hi = (d.mean() - bound) / se          # H0: mu >=  bound
    p_lo = 1.0 - stats.t.cdf(t_lo, df)
    p_hi = stats.t.cdf(t_hi, df)
    p = max(p_lo, p_hi)
    return {"p_tost": float(p), "equivalent": bool(p < alpha), "n": int(n),
            "mean": float(d.mean()), "bound": float(bound),
            "ci90_lo": float(d.mean() - stats.t.ppf(1 - alpha, df) * se),
            "ci90_hi": float(d.mean() + stats.t.ppf(1 - alpha, df) * se)}


def one_sample_vs_reference(values, reference: float, n_boot: int = 10000,
                            random_state: int = 0):
    """Is a fold-wise score distribution distinguishable from a reference?

    Used to ask whether a classifier operates above its chance level. The
    Wilcoxon signed-rank test is paired against the constant reference, and a
    bootstrap interval on the mean is reported alongside, because with
    non-independent folds the interval is the more honest summary.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return {"mean": np.nan, "p": np.nan}
    try:
        p = stats.wilcoxon(v - reference, zero_method="zsplit").pvalue
    except Exception:                           # noqa: BLE001
        p = np.nan
    lo, hi = bootstrap_ci(v - reference, n_boot=n_boot, random_state=random_state)
    rng = np.random.default_rng(random_state)
    means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "reference": float(reference), "delta": float(v.mean() - reference),
            "p": float(p) if np.isfinite(p) else np.nan,
            "ci95_lo": float(np.percentile(means, 2.5)),
            "ci95_hi": float(np.percentile(means, 97.5)),
            "delta_ci95_lo": lo, "delta_ci95_hi": hi}


def permutation_null(fit_predict_fn, y, groups, n_perm: int = 200,
                     random_state: int = 0):
    """Empirical null for a grouped classification score.

    Labels are permuted *within the subject grouping* -- each subject keeps a
    single permuted label block -- so the null preserves the dependence
    structure that subject-grouped cross-validation is designed to respect.
    Permuting trial labels independently would break that and produce an
    optimistically tight null.
    """
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    # one label per subject, since the label is a subject-level property here
    sub_lab = {g: y[groups == g][0] for g in uniq}
    observed = fit_predict_fn(y)
    null = []
    for _ in range(n_perm):
        perm = rng.permutation([sub_lab[g] for g in uniq])
        mapping = dict(zip(uniq, perm))
        y_perm = np.array([mapping[g] for g in groups])
        null.append(fit_predict_fn(y_perm))
    null = np.asarray(null, dtype=float)
    p = (1.0 + np.sum(null >= observed)) / (1.0 + null.size)
    return {"observed": float(observed), "null_mean": float(null.mean()),
            "null_sd": float(null.std(ddof=1)),
            "null_p95": float(np.percentile(null, 95)),
            "p_permutation": float(p), "n_perm": int(n_perm)}


def nogueira_stability(selections, n_features: int) -> dict:
    """Nogueira's stability measure for a set of selected subsets.

    Defined as 1 minus the ratio of the average per-feature selection variance
    to the variance expected if the same number of features were chosen at
    random. It is corrected for subset size and for chance, which per-feature
    selection frequencies alone are not, and so allows selectors that retain
    different numbers of features to be compared on one scale. Unity is
    perfect stability; zero is chance.
    """
    Z = np.asarray(selections, dtype=float)      # M subsets x d features
    M, d = Z.shape
    if M < 2:
        return {"stability": np.nan, "M": int(M)}
    pf = Z.mean(axis=0)
    kbar = Z.sum(axis=1).mean()
    num = (M / (M - 1.0)) * np.sum(pf * (1.0 - pf))
    den = (kbar / d) * (1.0 - kbar / d) * d
    if den <= 0:
        return {"stability": np.nan, "M": int(M)}
    return {"stability": float(1.0 - num / den), "M": int(M),
            "mean_subset_size": float(kbar), "n_features": int(d)}


def tost_power(sd: float, n: int, bound: float, true_delta: float = 0.0,
               alpha: float = 0.05, test_train_ratio: float = 0.0) -> float:
    """Power of a paired TOST at a given margin, SD and sample size.

    Reported so that the equivalence margin is defensible on statistical
    grounds and not only on face validity: a margin the design has no power
    to interrogate would make an equivalence claim vacuous.
    """
    if sd <= 0 or n < 3:
        return np.nan
    se = sd * np.sqrt(1.0 / n + test_train_ratio)
    df = n - 1
    t_crit = stats.t.ppf(1 - alpha, df)
    # probability that both one-sided tests reject, via the shifted t
    lo = (true_delta + bound) / se
    hi = (true_delta - bound) / se
    p = stats.nct.cdf(-t_crit, df, hi) - stats.nct.cdf(t_crit, df, lo)
    return float(max(0.0, min(1.0, p)))


def min_detectable_margin(sd: float, n: int, power: float = 0.80,
                          alpha: float = 0.05,
                          test_train_ratio: float = 0.0) -> float:
    """Smallest equivalence margin achieving the target power."""
    lo, hi = 1e-4, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if tost_power(sd, n, mid, alpha=alpha,
                      test_train_ratio=test_train_ratio) < power:
            lo = mid
        else:
            hi = mid
    return float(hi)


EQUIVALENCE_METRICS = ("macro_f1", "balanced_accuracy",
                       "sensitivity_macro", "specificity_macro")


def equivalence_table(per_fold: pd.DataFrame, bound: float = 0.05,
                      corrected: bool = True, alpha: float = 0.05,
                      metrics=EQUIVALENCE_METRICS) -> pd.DataFrame:
    """TOST of full versus reduced for every model x modality x metric.

    Holm-adjusted across the whole table. With `corrected`, the standard
    error carries the Nadeau-Bengio inflation computed from the fold sizes.
    """
    ratio = nadeau_bengio_ratio(per_fold) if corrected else {}
    key = ["modality", "model", "repeat", "fold"]
    rows = []
    for metric in metrics:
        full = per_fold[per_fold.condition == "full"].set_index(key)[metric]
        red = per_fold[per_fold.condition == "reduced"].set_index(key)[metric]
        diff = (full - red).dropna().reset_index()
        for (mod, model), g in diff.groupby(["modality", "model"]):
            r = ratio.get(mod, 0.0)
            t = tost_equivalence(g[metric].to_numpy(), bound, alpha, r)
            rows.append({"modality": mod, "model": model, "metric": metric,
                         "delta": t.get("mean"), "bound": bound,
                         "p_tost": t["p_tost"], "equivalent": t["equivalent"],
                         "ci90_lo": t.get("ci90_lo"), "ci90_hi": t.get("ci90_hi"),
                         "test_train_ratio": r})
    out = pd.DataFrame(rows)
    out["p_tost_holm"] = holm_bonferroni(out["p_tost"].to_numpy(), alpha)[0]
    out["equivalent"] = out["p_tost_holm"] < alpha
    return out


COMPARISON_METRICS = ("macro_f1", "balanced_accuracy", "sensitivity_macro",
                      "specificity_macro", "ppv_macro", "npv_macro",
                      "auc_macro_ovr")


def comparison_table(per_fold: pd.DataFrame, cfg=None,
                     metrics=COMPARISON_METRICS) -> pd.DataFrame:
    """Full vs reduced, and reduced vs each supervised selector, per metric."""
    comps = []
    present = set(per_fold.condition)
    for metric in metrics:
        if metric not in per_fold:
            continue
        for cond in ("mrmr", "relieff", "lasso"):
            if cond in present:
                c = compare_conditions(per_fold, metric, "reduced", cond, cfg)
                if len(c):
                    c["comparison"] = f"reduced_vs_{cond}"
                    comps.append(c)
        c = compare_conditions(per_fold, metric, "full", "reduced", cfg)
        if len(c):
            c["comparison"] = "full_vs_reduced"
            comps.append(c)
    return pd.concat(comps, ignore_index=True) if comps else pd.DataFrame()
