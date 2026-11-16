"""Evaluate every channel-pooling variant under the paper's own protocol.

Resumable: each (variant, condition) slice appends to the same file and is
skipped if already present, so the analysis survives being run in pieces.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from ncx.config import Config, feature_order
from ncx.evaluate import build_models, run_condition
from ncx.pooling import VARIANTS

OUT = Path("results/pooling_per_fold.csv")
SRC = Path("results/cache/pooling")


def main() -> None:
    cfg = Config()
    # A deliberately lighter budget than the headline protocol: the question
    # here is whether the spatial operator changes the answer, and that is a
    # within-analysis comparison, so what matters is that every variant is
    # given exactly the same budget rather than that the budget is maximal.
    if "--iters" in sys.argv:
        cfg.n_search_iter = int(sys.argv[sys.argv.index("--iters") + 1])
    if "--repeats" in sys.argv:
        cfg.n_repeats = int(sys.argv[sys.argv.index("--repeats") + 1])
    if "--splits" in sys.argv:
        cfg.n_splits = int(sys.argv[sys.argv.index("--splits") + 1])
    conds = (sys.argv[sys.argv.index("--conds") + 1].split(",")
             if "--conds" in sys.argv else ["full", "reduced"])
    variants = (sys.argv[sys.argv.index("--variants") + 1].split(",")
                if "--variants" in sys.argv else list(VARIANTS))

    names = feature_order("eeg")
    # Slices are per (variant, condition, model) so that a run cut short by a
    # wall-clock limit keeps everything it finished.
    done = set()
    if OUT.exists():
        d = pd.read_csv(OUT, usecols=["variant", "condition", "model"])
        done = set(map(tuple, d.drop_duplicates().values))

    all_models = build_models(cfg)
    if "--rf-trees" in sys.argv:
        # The forest dominates run time and its ensemble size is not what is
        # under test here; it is trimmed identically for every variant so the
        # comparison across pooling operators remains like-for-like.
        n = int(sys.argv[sys.argv.index("--rf-trees") + 1])
        est, grid = all_models["Random Forest"]
        est.set_params(n_estimators=n)
        all_models["Random Forest"] = (est, grid)
    wanted = (sys.argv[sys.argv.index("--models") + 1].split(",")
              if "--models" in sys.argv else list(all_models))

    for v in variants:
        df = pd.read_csv(SRC / f"features_eeg_{v}.csv").dropna(subset=names)
        X = np.nan_to_num(df[names].to_numpy(float))
        y = df.label.astype(str).to_numpy()
        g = df.subject.astype(str).to_numpy()
        for cond in conds:
            for mname in wanted:
                if (v, cond, mname) in done:
                    print(f"  skip {v}/{cond}/{mname}")
                    continue
                print(f"  {v}/{cond}/{mname} ...", flush=True)
                r = run_condition(X, y, g, names, cfg, modality="eeg", mode=cond,
                                  models={mname: all_models[mname]})
                if r.empty:
                    continue
                r.insert(0, "variant", v)
                r.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
                print(f"    -> bal.acc {r.balanced_accuracy.mean():.4f} "
                      f"+/- {r.balanced_accuracy.std():.4f} (n={len(r)})", flush=True)


if __name__ == "__main__":
    main()
