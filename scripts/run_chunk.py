#!/usr/bin/env python3
"""Run one modality/condition slice of the model comparison and append it.

Splitting the evaluation into resumable slices keeps each invocation short
enough to finish inside a command timeout, and makes an interrupted run cost
one slice rather than the whole panel. Results are appended to a single
per-fold CSV, which is the same artefact `run.py analyse` would have written.
"""
from __future__ import annotations
import argparse, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from ncx.config import Config, feature_order                      # noqa: E402
from ncx import evaluate as ev                                    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--modality", required=True, choices=["eeg", "tcd"])
ap.add_argument("--conditions", default="full,reduced,mrmr,relieff,lasso")
ap.add_argument("--out", type=Path, default=Path("results"))
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--splits", type=int, default=5)
ap.add_argument("--inner", type=int, default=3)
ap.add_argument("--iters", type=int, default=10)
ap.add_argument("--trees", type=int, default=150)
a = ap.parse_args()

cfg = Config()
cfg.n_repeats, cfg.n_splits = a.repeats, a.splits
cfg.inner_splits, cfg.n_search_iter = a.inner, a.iters

df = pd.read_csv(a.out / "cache" / f"features_{a.modality}.csv")
names = [c for c in feature_order(a.modality) if c in df.columns]
df = df.dropna(subset=names)
X = df[names].to_numpy(float)
y = df["label"].astype(str).to_numpy()
g = df["subject"].astype(str).to_numpy()

# The forest dominates run time; 150 trees is ample for a 15-descriptor space
# and keeps the panel balanced against the other four models.
models = ev.build_models(cfg)
if "Random Forest" in models:
    est, grid = models["Random Forest"]
    est.set_params(n_estimators=a.trees)
    models["Random Forest"] = (est, grid)

k_match = {"eeg": 12, "tcd": 7}[a.modality]
dest = a.out / "tables" / "per_fold_metrics.csv"
dest.parent.mkdir(parents=True, exist_ok=True)

for cond in a.conditions.split(","):
    t0 = time.time()
    pf = ev.run_condition(X, y, g, names, cfg, a.modality, mode=cond, k=k_match,
                          models=models)
    if len(pf):
        pf.to_csv(dest, mode="a", header=not dest.exists(), index=False)
    print(f"  [{a.modality}/{cond}] {len(pf)} rows in {time.time()-t0:.0f}s",
          flush=True)
print("done")
