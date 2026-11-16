#!/usr/bin/env python3
"""Cross-modal redundancy study -- command line entry point.

Usage
-----
    python run.py inspect                 # Stage 0: report what is on disk
    python run.py features                # Stage 1: extract both feature tables
    python run.py analyse                 # Stage 2: redundancy + models + stats
    python run.py all                     # Stages 1-2
    python run.py selftest                # run everything on synthetic fixtures

Run `inspect` first and send the two JSON reports it writes; the loaders are
pinned from those before anything else is trusted.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ncx.config import (Config, feature_order, family_of,        # noqa: E402
                        matched_core)
from ncx import io as ncio                                        # noqa: E402
from ncx import features as feat                                  # noqa: E402
from ncx import redundancy as red                                 # noqa: E402
from ncx import evaluate as ev                                    # noqa: E402
from ncx import stats as st                                       # noqa: E402
from ncx import report as rep                                     # noqa: E402

CONDITIONS = ["full", "reduced", "mrmr", "relieff", "lasso"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------

def cmd_inspect(cfg: Config) -> None:
    d = cfg.dirs()
    for name, root in (("eeg", cfg.eeg_root), ("tcd", cfg.tcd_root)):
        r = ncio.inspect_tree(root)
        out = d["logs"] / f"inspect_{name}.json"
        ncio.save_report(r, out)
        log(f"{name}: exists={r.get('exists')} files={r.get('n_files_total')} "
            f"-> {out}")
        if r.get("files_by_extension"):
            log(f"     extensions: {r['files_by_extension']}")


def cmd_features(cfg: Config, tcd_mapping: dict | None = None) -> None:
    d = cfg.dirs()
    log("extracting EEG features ...")
    eeg = feat.build_eeg_table(ncio.load_eeg_records(cfg.eeg_root, cfg), cfg)
    eeg.to_csv(d["cache"] / "features_eeg.csv", index=False)
    log(f"  EEG: {eeg.shape[0]} trials x {eeg.shape[1]} cols "
        f"({eeg['subject'].nunique()} subjects)")

    log("extracting TCD features ...")
    tcd = feat.build_tcd_table(
        ncio.load_tcd_records(cfg.tcd_root, cfg, tcd_mapping), cfg)
    tcd.to_csv(d["cache"] / "features_tcd.csv", index=False)
    log(f"  TCD: {tcd.shape[0]} records x {tcd.shape[1]} cols "
        f"({tcd['subject'].nunique() if len(tcd) else 0} subjects)")


def _load_table(path: Path, modality: str):
    """Load a feature table and drop incomplete rows.

    Rows carrying a missing descriptor are removed rather than imputed. The
    only features that go missing here are the bilateral asymmetry indices of
    sessions insonated on one side, and imputing those would invent the very
    structure the study measures.
    """
    df = pd.read_csv(path)
    names = [c for c in feature_order(modality) if c in df.columns]
    df = df.dropna(subset=["label"])
    df = df[df["label"].astype(str).str.lower() != "unknown"]
    keep = [c for c in names if df[c].notna().mean() > 0.5]
    dropped = sorted(set(names) - set(keep))
    n0 = len(df)
    df = df.dropna(subset=keep)
    n_incomplete = n0 - len(df)
    return df, keep, dropped, n_incomplete


def cmd_analyse(cfg: Config) -> None:
    d = cfg.dirs()
    tables, figs, bundle = {}, d["figures"], {"config": str(cfg.out_root)}
    per_fold_all, profiles, blocks, stab = [], {}, {}, {}

    for modality in ("eeg", "tcd"):
        path = d["cache"] / f"features_{modality}.csv"
        if not path.exists():
            log(f"! {path} missing -- run `features` first; skipping {modality}")
            continue
        df, names, dropped, n_inc = _load_table(path, modality)
        if len(df) < 20 or len(names) < 4:
            log(f"! {modality}: too few usable rows/features; skipping")
            continue
        log(f"{modality}: {len(df)} rows ({n_inc} incomplete dropped), "
            f"{len(names)} features, {df['subject'].nunique()} subjects, "
            f"classes={dict(df['label'].value_counts())}"
            + (f" | unavailable: {dropped}" if dropped else ""))
        bundle[f"cohort_{modality}"] = {
            "n_rows": int(len(df)), "n_incomplete_dropped": int(n_inc),
            "n_subjects": int(df["subject"].nunique()),
            "n_features": len(names), "features": names,
            "class_counts": {str(k): int(v)
                             for k, v in df["label"].value_counts().items()},
            "subjects_per_class": {str(k): int(v) for k, v in
                                   df.groupby("label")["subject"].nunique().items()}}

        # ---- descriptive redundancy -----------------------------------
        cs = red.correlation_summary(df, names)
        pr = red.pca_profile(df, names, cfg.pca_variance_targets)
        fb = red.family_redundancy(df, names, lambda n: family_of(modality, n))
        profiles[modality], blocks[modality] = pr, fb
        tables[f"corr_matrix_{modality}"] = cs["matrix"].reset_index(names="feature")
        tables[f"family_blocks_{modality}"] = fb.reset_index(names="family")
        rep.fig_correlation(cs["matrix"],
                            f"{modality.upper()} candidate space",
                            figs / f"fig_corr_{modality}.png")

        # ---- whole-cohort reduction chain (descriptive only) ----------
        X = np.nan_to_num(df[names].to_numpy(dtype=float))
        rr = red.RedundancyReducer(cfg.corr_threshold, cfg.vif_threshold, names).fit(X)
        chain = {"n_candidates": len(names),
                 "n_after_correlation": int(rr.n_after_corr_),
                 "n_retained": int(len(rr.support_)),
                 "reduction_pct": round(100 * (1 - len(rr.support_) / len(names)), 1),
                 "dropped_correlation": rr.dropped_correlation_,
                 "dropped_vif": rr.dropped_vif_,
                 "retained": rr.kept_names_,
                 "final_vif": dict(zip(rr.kept_names_,
                                       np.round(rr.final_vif_, 3).tolist()))}
        bundle[f"chain_{modality}"] = chain
        bundle[f"correlation_{modality}"] = {k: v for k, v in cs.items()
                                             if k != "matrix"}
        bundle[f"pca_{modality}"] = {k: v for k, v in pr.items()
                                     if k not in ("explained_variance_ratio",
                                                  "cumulative", "eigenvalues")}
        log(f"  chain: {chain['n_candidates']} -> {chain['n_after_correlation']}"
            f" -> {chain['n_retained']} ({chain['reduction_pct']}% reduction)")

        # ---- matched core: the 11 descriptors present in BOTH modalities
        core = [c for c in matched_core(modality) if c in names]
        if len(core) >= 6:
            cs_c = red.correlation_summary(df, core)
            pr_c = red.pca_profile(df, core, cfg.pca_variance_targets)
            Xc = np.nan_to_num(df[core].to_numpy(dtype=float))
            rr_c = red.RedundancyReducer(cfg.corr_threshold,
                                         cfg.vif_threshold, core).fit(Xc)
            bundle[f"core_{modality}"] = {
                "n_core": len(core), "features": core,
                "correlation": {k: v for k, v in cs_c.items() if k != "matrix"},
                "pca": {k: v for k, v in pr_c.items()
                        if k not in ("explained_variance_ratio", "cumulative",
                                     "eigenvalues")},
                "n_retained": int(len(rr_c.support_)),
                "retained": rr_c.kept_names_,
                "dropped_correlation": rr_c.dropped_correlation_,
                "dropped_vif": rr_c.dropped_vif_}
            profiles[f"{modality} (core)"] = pr_c
            tables[f"corr_matrix_core_{modality}"] = \
                cs_c["matrix"].reset_index(names="feature")
            log(f"  matched core: {len(core)} -> {len(rr_c.support_)} retained, "
                f"mean|r|={cs_c['mean_abs_offdiag']:.3f}, "
                f"PR={pr_c['participation_ratio']:.2f}")

        # ---- evaluation ------------------------------------------------
        y = df["label"].astype(str).to_numpy()
        g = df["subject"].astype(str).to_numpy()
        k_match = int(len(rr.support_))
        for cond in CONDITIONS:
            log(f"  evaluating [{modality}/{cond}] ...")
            pf = ev.run_condition(X, y, g, names, cfg, modality, mode=cond,
                                  k=k_match)
            if len(pf):
                per_fold_all.append(pf)
        stab[modality] = ev.selection_stability(X, y, g, names, cfg, "reduced")
        tables[f"stability_{modality}"] = stab[modality]

    if not per_fold_all:
        log("no results produced; check the inspect report and loaders.")
        raise SystemExit(1)

    per_fold = pd.concat(per_fold_all, ignore_index=True)
    tables["per_fold_metrics"] = per_fold
    summary = st.summarise_per_fold(per_fold)
    tables["summary_metrics"] = summary

    comps = []
    for metric in ("macro_f1", "balanced_accuracy", "sensitivity_macro",
                   "specificity_macro", "ppv_macro", "npv_macro",
                   "auc_macro_ovr"):
        for cond in ("mrmr", "relieff", "lasso"):
            if cond in set(per_fold.condition):
                c = st.compare_conditions(per_fold, metric, "reduced", cond, cfg)
                if len(c):
                    c["comparison"] = f"reduced_vs_{cond}"
                    comps.append(c)
        c = st.compare_conditions(per_fold, metric, "full", "reduced", cfg)
        if len(c):
            c["comparison"] = "full_vs_reduced"
            comps.append(c)
    if comps:
        tables["statistical_comparisons"] = pd.concat(comps, ignore_index=True)

    if profiles:
        rep.fig_pca(profiles, figs / "fig_pca.png")
    if blocks:
        rep.fig_family_blocks(blocks, figs / "fig_family_blocks.png")
    if stab:
        rep.fig_stability(stab, figs / "fig_stability.png")
    for m in ("macro_f1", "sensitivity_macro", "specificity_macro"):
        rep.fig_performance(summary, m, figs / f"fig_perf_{m}.png")

    rep.write_tables(tables, d["tables"])
    bundle["summary"] = summary
    if "statistical_comparisons" in tables:
        bundle["comparisons"] = tables["statistical_comparisons"]
    rep.write_bundle(bundle, cfg.out_root / "results_bundle.json")
    cfg.save(cfg.out_root / "config_used.json")
    log(f"done -> {cfg.out_root}")
    log("Send results_bundle.json plus results/tables/*.csv to write the "
        "Results section.")


def cmd_selftest(cfg: Config) -> None:
    """End-to-end run on synthetic fixtures -- validates code paths only."""
    from tests.make_fixtures import make_fixtures
    make_fixtures(cfg)
    cfg.n_repeats, cfg.n_splits, cfg.inner_splits, cfg.n_search_iter = 2, 4, 2, 4
    cmd_analyse(cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["inspect", "features", "analyse", "all",
                                      "selftest"])
    ap.add_argument("--eeg", type=Path, default=None)
    ap.add_argument("--tcd", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--splits", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None,
                    help="hyperparameter search iterations per model")
    a = ap.parse_args()

    cfg = Config()
    if a.eeg: cfg.eeg_root = a.eeg
    if a.tcd: cfg.tcd_root = a.tcd
    if a.out: cfg.out_root = a.out
    if a.repeats: cfg.n_repeats = a.repeats
    if a.splits: cfg.n_splits = a.splits
    if a.iters: cfg.n_search_iter = a.iters

    if a.stage == "inspect":
        cmd_inspect(cfg)
    elif a.stage == "features":
        cmd_features(cfg)
    elif a.stage == "analyse":
        cmd_analyse(cfg)
    elif a.stage == "all":
        cmd_features(cfg); cmd_analyse(cfg)
    else:
        cmd_selftest(cfg)


if __name__ == "__main__":
    main()
