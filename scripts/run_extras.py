"""Supplementary analyses reported in the manuscript beyond the main pipeline.

The main pipeline (`run.py analyse`) produces the per-fold metrics, the
structural redundancy summary, the equivalence tests and the chance-level
tests. The analyses here were added during peer review, and the script that
originally produced them was not committed. This file reconstructs them from
the library functions in `ncx/`, which are unchanged and tested.

Stages
------
perfold    Repeated-measures ANOVA, post-hoc decomposition and TOST power.
           Input: per_fold_metrics.csv only. Verified to reproduce
           reference_results/reviewer_extra.json exactly, apart from the
           Monte Carlo error of the permutation p-values.
structure  Partial correlation, threshold sweep, bootstrap and permutation
           tests on the modality gap, and Nogueira selection stability.
           Input: the feature caches written by `run.py features`.
permutation
           Label-permutation null for each feature space, fixed logistic
           regression, subject-grouped cross-validation.
           Input: the feature caches.

Run `python scripts/compare_to_reference.py` afterwards to check a fresh
result against reference_results/.

Usage
-----
    python scripts/run_extras.py perfold     --out results
    python scripts/run_extras.py structure   --out results
    python scripts/run_extras.py permutation --out results --n-perm 1000
    python scripts/run_extras.py all         --out results
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ncx.anova import (assumption_checks, friedman_kendall, rm_anova_two_way,
                       rm_permutation_p)
from ncx.config import Config, feature_order, matched_core
from ncx.redundancy import RedundancyReducer, partial_correlation_summary
from ncx.selectors import make_selector
from ncx.sensitivity import (bootstrap_redundancy_gap,
                             permutation_redundancy_gap, sweep_agreement,
                             threshold_sweep)
from ncx.stats import min_detectable_margin, nogueira_stability, tost_power

COND_ORDER = ["full", "reduced", "mrmr", "relieff", "lasso"]
SELECTORS = ["mrmr", "relieff", "lasso"]
GENERATORS = {"eeg": ["delta", "theta", "alpha", "beta"],
              "tcd": ["psv", "edv", "mfv"]}
EQUIV_MARGIN = 0.05


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()
                if not isinstance(v, pd.DataFrame)}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def _merge_into(path: Path, payload: dict) -> None:
    current = json.loads(path.read_text()) if path.exists() else {}
    current.update(_jsonable(payload))
    path.write_text(json.dumps(current, indent=2))


# --------------------------------------------------------------------------
# Stage 1: everything computable from the per-fold metrics
# --------------------------------------------------------------------------

def _fold_frame(pf: pd.DataFrame, modality: str, metric: str) -> pd.DataFrame:
    d = pf[pf.modality == modality].copy()
    d["subject"] = d["repeat"].astype(str) + "_" + d["fold"].astype(str)
    return d[["subject", "condition", "model", metric]].rename(columns={metric: "y"})


def _anova(d: pd.DataFrame) -> dict:
    return rm_anova_two_way(d.y.to_numpy(), d.subject.to_numpy(),
                            d.condition.to_numpy(), d.model.to_numpy())


def anova_block(pf: pd.DataFrame, metric: str, n_perm: int, seed: int) -> dict:
    out = {}
    for modality in sorted(pf.modality.unique()):
        d = _fold_frame(pf, modality, metric)
        res = _anova(d)
        cells = d.condition + "|" + d.model
        perm = {}
        for effect in ["A", "B", "A:B"]:
            perm[effect] = rm_permutation_p(
                d.y.to_numpy(), d.subject.to_numpy(), d.condition.to_numpy(),
                d.model.to_numpy(), effect=effect, n_perm=n_perm,
                random_state=seed)
        cube = (d.groupby(["subject", "condition"]).y.mean()
                 .unstack()[COND_ORDER].to_numpy())
        out[modality] = {"rm_anova": {k: v for k, v in res.items()
                                      if k != "_total"},
                         "assumptions": assumption_checks(d.y.to_numpy(),
                                                          cells.to_numpy()),
                         "permutation": perm,
                         "friedman": friedman_kendall(cube),
                         "cond_order": COND_ORDER}
    return out


def posthoc_block(pf: pd.DataFrame, metric: str, n_perm: int, seed: int) -> dict:
    out = {}
    for modality in sorted(pf.modality.unique()):
        d = _fold_frame(pf, modality, metric)
        mm = {c: {"mean": round(float(g.y.mean()), 4),
                  "std": round(float(g.y.std(ddof=1)), 4)}
              for c, g in d.groupby("condition")}

        fr = d[d.condition.isin(["full", "reduced"])]
        a = _anova(fr)["A"]
        p_perm = rm_permutation_p(fr.y.to_numpy(), fr.subject.to_numpy(),
                                  fr.condition.to_numpy(), fr.model.to_numpy(),
                                  effect="A", n_perm=n_perm, random_state=seed)
        full_vs_reduced = {"F": a["F"], "p": a["p"],
                           "eta2_generalised": a["eta2_generalised"],
                           "p_perm": p_perm["p_permutation"]}

        loso = {}
        for drop in SELECTORS:
            r = _anova(d[d.condition != drop])["A"]
            loso[drop] = {"F": r["F"], "p": r["p"], "p_gg": r["p_gg"],
                          "eta2G": r["eta2_generalised"]}
        out[modality] = {"marginal_means": mm,
                         "full_vs_reduced": full_vs_reduced,
                         "leave_one_selector_out": loso}
    return out


def tost_power_table(pf: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    key = ["modality", "model", "repeat", "fold"]
    full = pf[pf.condition == "full"].set_index(key)[metric]
    red = pf[pf.condition == "reduced"].set_index(key)[metric]
    diff = (full - red).dropna().reset_index()
    for (modality, model), g in diff.groupby(["modality", "model"]):
        sd = float(g[metric].std(ddof=1))
        n = int(len(g))
        rows.append({"modality": modality, "model": model,
                     "sd_of_differences": round(sd, 4), "n_folds": n,
                     "power_at_0p05": round(tost_power(sd, n, EQUIV_MARGIN), 3),
                     "min_margin_80pct": round(min_detectable_margin(sd, n), 4)})
    return pd.DataFrame(rows)


def stage_perfold(out: Path, per_fold: Path, n_perm: int, seed: int) -> None:
    pf = pd.read_csv(per_fold)
    log(f"per-fold metrics: {len(pf)} rows from {per_fold}")
    log("repeated-measures ANOVA (+ within-fold permutation) ...")
    anova = anova_block(pf, "macro_f1", n_perm, seed)
    log("post-hoc decomposition ...")
    posthoc = posthoc_block(pf, "macro_f1", n_perm, seed)
    power = tost_power_table(pf, "macro_f1")
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    power.to_csv(tables / "tost_power.csv", index=False)
    _merge_into(tables / "extras.json", {"anova": anova, "anova_posthoc": posthoc})
    log(f"wrote {tables / 'extras.json'} and {tables / 'tost_power.csv'}")


# --------------------------------------------------------------------------
# Stage 2: structural sensitivity analyses, from the feature caches
# --------------------------------------------------------------------------

def _load_features(cache: Path, modality: str):
    df = pd.read_csv(cache / f"features_{modality}.csv")
    names = [n for n in feature_order(modality) if n in df.columns]
    df = df.dropna(subset=names).reset_index(drop=True)
    core = [n for n in matched_core(modality) if n in names]
    return df, names, core


def selection_matrix(df, names, cfg, selector: str, k: int,
                     n_repeats: int, n_splits: int) -> np.ndarray:
    """One binary support row per training partition."""
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler
    X = np.nan_to_num(df[names].to_numpy(dtype=float))
    y = df["label"].astype(str).to_numpy()
    g = df["subject"].astype(str).to_numpy()
    rows = []
    for rep in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                  random_state=cfg.random_state + rep)
        for tr, _ in cv.split(X, y, g):
            Xs = StandardScaler().fit_transform(X[tr])
            sel = (RedundancyReducer(cfg.corr_threshold, cfg.vif_threshold, names)
                   if selector == "reduced"
                   else make_selector(selector, k, cfg.random_state))
            sel.fit(Xs, y[tr])
            mask = np.zeros(len(names), dtype=bool)
            mask[np.asarray(sel.support_, dtype=int)] = True
            rows.append(mask)
    return np.vstack(rows)


def stage_structure(out: Path, cache: Path, cfg: Config,
                    n_boot: int, n_perm: int, stab_repeats: int,
                    stab_splits: int) -> None:
    frames, data = {}, {}
    for m in ["eeg", "tcd"]:
        df, names, core = _load_features(cache, m)
        data[m] = (df, names, core)
        frames[f"{m}/native"] = (df, names)
        frames[f"{m}/core"] = (df, core)
        log(f"{m}: {len(df)} records, {len(names)} native, {len(core)} core")

    payload = {}

    log("partial correlation against the generating descriptors ...")
    payload["partial_correlation"] = {
        m: partial_correlation_summary(data[m][0], data[m][1], GENERATORS[m])
        for m in data}

    log("threshold sweep (14 correlation x 4 VIF thresholds) ...")
    sweep = threshold_sweep(frames)
    payload["threshold_sweep_agreement"] = {
        "native": sweep_agreement(sweep, "eeg/native", "tcd/native"),
        "matched core": sweep_agreement(sweep, "eeg/core", "tcd/core")}

    (de, _, ce), (dt, _, ct) = data["eeg"], data["tcd"]
    log(f"subject-level bootstrap on the modality gap ({n_boot} replicates) ...")
    payload["redundancy_gap_bootstrap"] = bootstrap_redundancy_gap(
        de, ce, de["subject"], dt, ct, dt["subject"], n_boot=n_boot,
        random_state=cfg.random_state)
    log(f"column-permutation check on the gap ({n_perm} permutations) ...")
    payload["redundancy_gap_permutation"] = permutation_redundancy_gap(
        de, ce, dt, ct, n_perm=n_perm, random_state=cfg.random_state)

    log(f"selection stability ({stab_repeats} x {stab_splits} partitions) ...")
    stab, freq_rows = {}, []
    for m in ["eeg", "tcd"]:
        df, names, _ = data[m]
        k = int(RedundancyReducer(cfg.corr_threshold, cfg.vif_threshold, names)
                .fit(np.nan_to_num(df[names].to_numpy(float))).support_.size)
        for sel in ["reduced"] + SELECTORS:
            Z = selection_matrix(df, names, cfg, sel, k, stab_repeats, stab_splits)
            stab[f"{m}/{sel}"] = nogueira_stability(Z, len(names))
            for name, f in zip(names, Z.mean(axis=0)):
                freq_rows.append({"modality": m, "selector": sel,
                                  "feature": name,
                                  "selection_frequency": round(float(f), 4)})
    payload["selector_stability"] = stab

    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(tables / "threshold_sweep.csv", index=False)
    pd.DataFrame(freq_rows).to_csv(tables / "selector_stability.csv", index=False)
    _merge_into(tables / "extras.json", payload)
    log(f"wrote {tables / 'extras.json'}, threshold_sweep.csv, "
        f"selector_stability.csv")


# --------------------------------------------------------------------------
# Stage 3: label-permutation null per feature space
# --------------------------------------------------------------------------

def _grouped_score(X, y, g, seed: int, n_splits: int = 5) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr, te in cv.split(X, y, g):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=5000, random_state=seed))
        clf.fit(X[tr], y[tr])
        scores.append(f1_score(y[te], clf.predict(X[te]), average="macro"))
    return float(np.mean(scores))


def _permute(y, g, rng, subject_level: bool):
    y_perm = y.copy()
    if subject_level:
        # one label per subject: shuffle the labels between subjects
        uniq = np.unique(g)
        lab = np.array([y[g == s][0] for s in uniq])
        mapping = dict(zip(uniq, rng.permutation(lab)))
        return np.array([mapping[s] for s in g])
    # label varies within subject: shuffle inside each subject, so every
    # subject keeps its own class balance and the grouping is untouched
    for s in np.unique(g):
        idx = np.flatnonzero(g == s)
        y_perm[idx] = rng.permutation(y[idx])
    return y_perm


def stage_permutation(out: Path, cache: Path, cfg: Config, n_perm: int) -> None:
    payload = {}
    for m in ["eeg", "tcd"]:
        df, names, _ = _load_features(cache, m)
        X = np.nan_to_num(df[names].to_numpy(dtype=float))
        y = df["label"].astype(str).to_numpy()
        g = df["subject"].astype(str).to_numpy()
        subject_level = bool(df.groupby("subject")["label"].nunique().max() == 1)
        rng = np.random.default_rng(cfg.random_state)
        t0 = time.time()
        observed = _grouped_score(X, y, g, cfg.random_state)
        null = np.array([_grouped_score(X, _permute(y, g, rng, subject_level), g,
                                        cfg.random_state)
                         for _ in range(n_perm)])
        payload[m] = {"label_is_subject_level": subject_level,
                      "observed": observed,
                      "null_mean": float(null.mean()),
                      "null_sd": float(null.std(ddof=1)),
                      "null_p95": float(np.percentile(null, 95)),
                      "null_p99": float(np.percentile(null, 99)),
                      "p_permutation": float((1 + np.sum(null >= observed))
                                             / (1 + n_perm)),
                      "n_perm": n_perm,
                      "seconds": round(time.time() - t0, 1)}
        log(f"{m}: observed {observed:.4f}, null {null.mean():.4f} "
            f"+/- {null.std(ddof=1):.4f}, p = {payload[m]['p_permutation']:.4f}")
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    _merge_into(tables / "extras.json", {f"permutation_{n_perm}": payload})


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=["perfold", "structure", "permutation", "all"])
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--per-fold", type=Path, default=None,
                    help="defaults to <out>/tables/per_fold_metrics.csv")
    ap.add_argument("--cache", type=Path, default=None,
                    help="defaults to <out>/cache")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--gap-perm", type=int, default=2000)
    ap.add_argument("--stab-repeats", type=int, default=10)
    ap.add_argument("--stab-splits", type=int, default=5)
    a = ap.parse_args()

    cfg = Config()
    per_fold = a.per_fold or a.out / "tables" / "per_fold_metrics.csv"
    cache = a.cache or a.out / "cache"

    if a.stage in ("perfold", "all"):
        stage_perfold(a.out, per_fold, a.n_perm, cfg.random_state)
    if a.stage in ("structure", "all"):
        stage_structure(a.out, cache, cfg, a.n_boot, a.gap_perm,
                        a.stab_repeats, a.stab_splits)
    if a.stage in ("permutation", "all"):
        stage_permutation(a.out, cache, cfg, a.n_perm)


if __name__ == "__main__":
    main()
