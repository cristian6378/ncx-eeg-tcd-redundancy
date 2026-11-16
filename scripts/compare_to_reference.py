"""Compare a fresh run against reference_results/.

Walks every numeric leaf of reviewer_extra.json that the fresh extras.json
also contains, and every shared numeric cell of the CSV tables, and reports
anything outside tolerance. Deterministic quantities must agree to rounding;
quantities with Monte Carlo error -- permutation p-values, bootstrap
intervals -- are reported against a looser tolerance and flagged as such.

    python scripts/compare_to_reference.py --fresh results/tables
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REF = Path(__file__).resolve().parents[1] / "reference_results"

# keys whose values carry sampling noise from a random procedure
STOCHASTIC = ("p_perm", "p_permutation", "ci_lo", "ci_hi", "ci_a", "ci_b",
              "boot_gap_mean", "null_mean", "null_sd", "null_p95", "null_p99",
              "seconds", "/p", "p_gap_le_0")


def _leaves(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _leaves(v, f"{path}/{k}")
    elif isinstance(o, list):
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in o):
            for i, x in enumerate(o):
                yield f"{path}[{i}]", x
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield path, o


def _stochastic(path: str) -> bool:
    if "/permutation/" in path or "redundancy_gap_permutation" in path:
        return True
    return any(path.endswith(k) or f"{k}[" in path for k in STOCHASTIC)


def compare_json(ref: dict, fresh: dict, rtol: float, stoch_atol: float):
    ref_leaves = dict(_leaves(ref))
    rows = []
    for path, val in _leaves(fresh):
        if path not in ref_leaves:
            continue
        r = ref_leaves[path]
        stoch = _stochastic(path)
        ok = (abs(val - r) <= stoch_atol) if stoch else \
             np.isclose(val, r, rtol=rtol, atol=1e-12)
        rows.append({"key": path, "reference": r, "fresh": val,
                     "kind": "stochastic" if stoch else "deterministic",
                     "ok": bool(ok)})
    return pd.DataFrame(rows)


def compare_csv(name: str, fresh_dir: Path, keys: list[str], rtol: float):
    a, b = REF / name, fresh_dir / name
    if not b.exists():
        return None
    ra, fb = pd.read_csv(a), pd.read_csv(b)
    m = ra.merge(fb, on=keys, suffixes=("_ref", "_fresh"))
    rows = []
    for c in ra.columns:
        if c in keys or f"{c}_fresh" not in m:
            continue
        if not np.issubdtype(m[f"{c}_ref"].dtype, np.number):
            continue
        for _, r in m.iterrows():
            ok = np.isclose(r[f"{c}_fresh"], r[f"{c}_ref"], rtol=rtol, atol=1e-4)
            rows.append({"key": f"{name}:{'/'.join(str(r[k]) for k in keys)}:{c}",
                         "reference": r[f"{c}_ref"], "fresh": r[f"{c}_fresh"],
                         "kind": "deterministic", "ok": bool(ok)})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fresh", type=Path, default=Path("results/tables"))
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--stoch-atol", type=float, default=0.02)
    a = ap.parse_args()

    frames = []
    extras = a.fresh / "extras.json"
    if extras.exists():
        frames.append(compare_json(json.loads((REF / "reviewer_extra.json").read_text()),
                                   json.loads(extras.read_text()),
                                   a.rtol, a.stoch_atol))
    for name, keys in [("tost_power.csv", ["modality", "model"]),
                       ("equivalence_tost.csv", ["modality", "model", "metric"]),
                       ("summary_metrics.csv", ["modality", "condition", "model"]),
                       ("threshold_sweep.csv", ["set", "corr_threshold", "vif_threshold"])]:
        f = compare_csv(name, a.fresh, keys, a.rtol)
        if f is not None:
            frames.append(f)

    if not frames:
        print(f"nothing to compare in {a.fresh}")
        return 1
    res = pd.concat(frames, ignore_index=True)
    det = res[res.kind == "deterministic"]
    sto = res[res.kind == "stochastic"]
    print(f"deterministic: {det.ok.sum()}/{len(det)} within rtol={a.rtol}")
    print(f"stochastic:    {sto.ok.sum()}/{len(sto)} within atol={a.stoch_atol}")
    bad = res[~res.ok]
    if len(bad):
        print("\noutside tolerance:")
        with pd.option_context("display.width", 160, "display.max_colwidth", 80):
            print(bad.to_string(index=False))
    return 0 if det.ok.all() else 1


if __name__ == "__main__":
    sys.exit(main())
