"""Build the per-channel cache that the channel-pooling analysis re-aggregates.

Separated from the pooling analysis itself because this is the expensive half:
band powers, distribution moments and complexity indices are computed once per
channel per trial, after which every pooling variant is a cheap re-aggregation
of the same numbers. That separation is what guarantees the variants differ
only in the spatial operator and not in any filtering or spectral detail.

Resumable by design. Subjects already present in the cache are skipped, so a
run interrupted by a wall-clock limit continues where it stopped, and
``--purge`` deletes each raw recording once it has been processed so the whole
1.8 GB deposit never has to be resident at once.

Example
-------
    python scripts/build_pooling_cache.py \\
        --source data/21679035/sourcedata \\
        --bids   data/21679035 \\
        --features results/cache/features_eeg.csv \\
        --out    results/cache/pooling_cache.csv
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from ncx.config import Config
from ncx.io import load_eeg_records
from ncx.pooling import build_cache, connectivity_lookup


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, required=True,
                   help="sourcedata/ directory of the EEG deposit (one folder per subject)")
    p.add_argument("--bids", type=Path, default=None,
                   help="deposit root holding task-motor-imagery_channels.tsv "
                        "(defaults to the parent of --source)")
    p.add_argument("--features", type=Path, required=True,
                   help="features_eeg.csv from the main run; wPLI and coherence are "
                        "joined from it rather than recomputed, since both are "
                        "montage-level and invariant to pooling")
    p.add_argument("--out", type=Path, required=True,
                   help="per-channel cache to create or extend")
    p.add_argument("--purge", action="store_true",
                   help="delete each .mat after processing it (frees disk on "
                        "machines that cannot hold the whole deposit)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()
    bids = args.bids or args.source.parent
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if not args.features.exists():
        raise SystemExit(
            f"{args.features} not found. Run `python scripts/run.py features` "
            "first: the connectivity descriptors are joined from it.")
    conn = connectivity_lookup(args.features)
    print(f"connectivity joined for {len(conn)} trials")

    done: set[str] = set()
    if args.out.exists():
        done = set(pd.read_csv(args.out, usecols=["subject"])
                   .subject.astype(str).unique())
        print(f"cache already holds {len(done)} subjects")

    mats = sorted(args.source.rglob("*_eeg.mat"))
    todo = [m for m in mats if m.stem.split("_")[0] not in done]
    print(f"{len(mats)} recordings found, {len(todo)} to process")

    for m in todo:
        subj = m.stem.split("_")[0]
        print(f"  {subj} ...", flush=True)
        recs = list(load_eeg_records(m.parent, cfg, bids_root=bids))
        if not recs:
            print(f"    no trials recovered for {subj}")
            continue
        df = build_cache(recs, cfg, verbose=False, connectivity=conn)
        df.to_csv(args.out, mode="a", header=not args.out.exists(), index=False)
        print(f"    {subj}: {len(recs)} trials, {df.shape[0]} channel-rows",
              flush=True)
        if args.purge:
            m.unlink(missing_ok=True)

    if args.out.exists():
        n = pd.read_csv(args.out, usecols=["subject"]).subject.nunique()
        print(f"cache now covers {n} subjects -> {args.out}")


if __name__ == "__main__":
    main()
