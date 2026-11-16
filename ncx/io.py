"""Loaders pinned to the two deposits as they are actually distributed.

Both loaders were written against the real files, not against the
documentation. Where the deposit and its accompanying paper disagree -- the
channel count is the clearest case -- the file wins and the discrepancy is
reported rather than silently reconciled.
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .config import Config


# ==========================================================================
# Stage 0 -- inspection
# ==========================================================================

def _describe_mat(path: Path, max_items: int = 40) -> dict:
    info: dict = {"reader": None, "vars": {}}
    try:
        md = loadmat(path, squeeze_me=True, struct_as_record=False)
        info["reader"] = "scipy(<=v7.2)"
        for k, v in list(md.items())[:max_items]:
            if k.startswith("__"):
                continue
            if isinstance(v, np.ndarray):
                info["vars"][k] = {"type": "ndarray", "shape": list(v.shape),
                                   "dtype": str(v.dtype)}
            elif hasattr(v, "_fieldnames"):
                fields = {}
                for fn in v._fieldnames:
                    a = getattr(v, fn)
                    fields[fn] = ({"shape": list(a.shape), "dtype": str(a.dtype)}
                                  if isinstance(a, np.ndarray) else str(type(a).__name__))
                info["vars"][k] = {"type": "struct", "fields": fields}
            else:
                info["vars"][k] = {"type": type(v).__name__}
        return info
    except NotImplementedError:
        info["error"] = "MATLAB v7.3 (HDF5); install h5py"
    except Exception as e:                      # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def _describe_table(path: Path, nrows: int = 2000) -> dict:
    try:
        if path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(path, nrows=nrows)
        else:
            df = pd.read_csv(path, nrows=nrows, sep=None, engine="python")
    except Exception as e:                      # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    out = {"n_cols": int(df.shape[1]), "head_rows": int(df.shape[0]), "columns": []}
    for c in df.columns[:60]:
        s = df[c]
        col = {"name": str(c), "dtype": str(s.dtype),
               "n_unique": int(s.nunique(dropna=True)),
               "null_frac": round(float(s.isna().mean()), 4)}
        if pd.api.types.is_numeric_dtype(s) and s.notna().any():
            col.update(min=float(np.nanmin(s)), max=float(np.nanmax(s)))
        else:
            col["examples"] = [str(v) for v in s.dropna().unique()[:6]]
        out["columns"].append(col)
    return out


def inspect_tree(root: Path, max_files: int = 40) -> dict:
    root = Path(root)
    report = {"root": str(root), "exists": root.exists(), "files": []}
    if not root.exists():
        return report
    paths = [p for p in sorted(root.rglob("*")) if p.is_file()]
    report["n_files_total"] = len(paths)
    by_ext: dict = {}
    for p in paths:
        by_ext[p.suffix.lower()] = by_ext.get(p.suffix.lower(), 0) + 1
    report["files_by_extension"] = by_ext
    for p in paths[:max_files]:
        item = {"path": str(p.relative_to(root)),
                "size_mb": round(p.stat().st_size / 1e6, 3),
                "suffix": p.suffix.lower()}
        if p.suffix.lower() == ".mat":
            item["detail"] = _describe_mat(p)
        elif p.suffix.lower() in {".csv", ".tsv", ".txt", ".xlsx", ".xls"}:
            item["detail"] = _describe_table(p)
        report["files"].append(item)
    return report


# ==========================================================================
# EEG -- Liu et al. (2024), figshare 21679035, sourcedata/*.mat
# ==========================================================================

MI_WINDOW_S = (2.0, 6.0)      # motor-imagery epoch within the 8 s trial
TRIAL_LABELS = {1: "left", 2: "right"}


def read_bids_eeg_channels(bids_root: Path) -> list[str]:
    """EEG channel names, in recording order, from the BIDS sidecar."""
    f = Path(bids_root) / "task-motor-imagery_channels.tsv"
    if not f.exists():
        return []
    ch = pd.read_csv(f, sep="\t")
    return [str(n) for n, t in zip(ch["name"], ch["type"])
            if str(t).upper() == "EEG"]


def classify_channels(X: np.ndarray, n_eeg_expected: int) -> dict:
    """Separate true EEG channels from reference, EOG and marker traces.

    The deposit stores 33 traces per trial while the sidecar names 31 and the
    JSON declares 29 EEG channels. The extra traces are identified from the
    data rather than assumed: a channel with zero variance is the recording
    reference (it is measured against itself), a near-integer channel with a
    handful of distinct values is the event marker, and the remaining
    high-amplitude outliers at the end of the montage are the EOG pair.
    """
    n_ch = X.shape[1]
    flat = X.reshape(-1, n_ch, X.shape[-1]) if X.ndim == 3 else X[None]
    v = flat.transpose(1, 0, 2).reshape(n_ch, -1)
    sd = np.std(v, axis=1)
    nuniq = np.array([len(np.unique(v[c])) for c in range(n_ch)])

    reference = [int(c) for c in np.flatnonzero(sd == 0)]
    marker = [int(c) for c in np.flatnonzero((nuniq <= 16) & (sd > 0))]
    excluded = set(reference) | set(marker)

    candidates = [c for c in range(n_ch) if c not in excluded]
    eog: list[int] = []
    n_extra = len(candidates) - n_eeg_expected
    if n_extra > 0:
        # EOG traces carry far more variance than scalp EEG and sit at the end
        # of the montage; take the highest-variance trailing channels.
        tail = sorted(candidates[-max(n_extra * 2, 2):], key=lambda c: -sd[c])
        eog = sorted(tail[:n_extra])
    eeg = [c for c in candidates if c not in set(eog)]
    return {"eeg": eeg, "reference": reference, "eog": eog, "marker": marker,
            "sd": sd.tolist()}


def load_eeg_records(root: Path, cfg: Config, bids_root: Path | None = None,
                     limit: int | None = None, window_s=MI_WINDOW_S):
    """Yield one record per motor-imagery trial.

    Only the imagery window is used. The 8 s trial also contains a 2 s
    instruction cue and a 2 s break, and including them would dilute every
    spectral descriptor with periods in which the participant was, by design,
    not performing the task.
    """
    root = Path(root)
    bids_root = Path(bids_root) if bids_root else root.parent
    names = read_bids_eeg_channels(bids_root)

    files = sorted(root.rglob("*_eeg.mat")) or sorted(root.rglob("*.mat"))
    if limit:
        files = files[:limit]
    if not files:
        warnings.warn(f"No .mat under {root}; point --eeg at sourcedata/.")
        return

    reported = False
    for f in files:
        try:
            md = loadmat(f, squeeze_me=True, struct_as_record=False)
        except Exception as e:                  # noqa: BLE001
            warnings.warn(f"Skipping {f.name}: {e}")
            continue
        st = md.get("eeg")
        if st is None or not hasattr(st, "rawdata"):
            warnings.warn(f"Skipping {f.name}: no eeg.rawdata")
            continue

        X = np.asarray(st.rawdata, dtype=float)          # trials x ch x samples
        y = np.atleast_1d(np.asarray(getattr(st, "label", [])))
        if X.ndim != 3:
            warnings.warn(f"Skipping {f.name}: rawdata ndim={X.ndim}")
            continue

        roles = classify_channels(X, n_eeg_expected=len(names) or 29)
        keep = roles["eeg"]
        if not reported:
            print(f"    channel roles: {len(keep)} EEG, "
                  f"reference={roles['reference']}, EOG={roles['eog']}, "
                  f"marker={roles['marker']}")
            reported = True

        ch_names = names[:len(keep)] if names else [f"ch{c}" for c in keep]
        s0, s1 = (int(window_s[0] * cfg.eeg_fs), int(window_s[1] * cfg.eeg_fs))
        s1 = min(s1, X.shape[-1])
        subject = f.stem.split("_")[0]

        for t in range(X.shape[0]):
            lab = TRIAL_LABELS.get(int(y[t]), "unknown") if t < len(y) else "unknown"
            yield {"data": X[t][keep, s0:s1], "fs": cfg.eeg_fs,
                   "ch_names": ch_names, "subject": subject,
                   "trial": t, "label": lab}


# ==========================================================================
# TCD -- Thorpe et al. (2020), figshare 9037388, cluster_data.csv
# ==========================================================================

TCD_COLUMNS = {"systole": "psv", "diastole": "edv", "mean_velocity": "mfv",
               "pulsatility": "pulsatility_index", "onset_score": "onset_score",
               "peaks_score": "peaks_score", "canopy_score": "canopy_score",
               "curvature": "curvature"}


def _side_of(vessel: str) -> str | None:
    v = str(vessel).strip().lower()
    if v.startswith("l"):
        return "L"
    if v.startswith("r"):
        return "R"
    return None


def load_tcd_records(root: Path, cfg: Config, mapping: dict | None = None,
                     limit: int | None = None):
    """Yield one record per insonation.

    Each row of the deposit is one vessel recording. Bilateral asymmetry is
    computed within a session (subject + guid) by pooling each side with the
    median and taking |L-R|/(L+R); where a session was insonated on one side
    only, the subject's other recordings supply the missing side, and if that
    still fails the asymmetry descriptors are left missing rather than
    imputed. Insonation depth is carried through as a covariate, never as a
    feature.
    """
    root = Path(root)
    files = ([root] if root.is_file()
             else sorted(root.rglob("cluster_data.csv")) or
                  [p for p in sorted(root.rglob("*.csv"))])
    if not files:
        warnings.warn(f"No TCD csv under {root}")
        return

    for path in files[:limit] if limit else files:
        df = pd.read_csv(path, sep=None, engine="python")
        df.columns = [str(c).strip() for c in df.columns]
        missing = [c for c in TCD_COLUMNS if c not in df.columns]
        if missing:
            warnings.warn(f"{path.name}: missing expected columns {missing}")
            continue

        subj_col = mapping.get("subject", "subId") if mapping else "subId"
        lab_col = mapping.get("label", "cond") if mapping else "cond"
        sess_col = "guid" if "guid" in df.columns else subj_col
        df["_side"] = df["vesselId"].map(_side_of) if "vesselId" in df.columns else None

        # --- session- and subject-level asymmetry ------------------------
        def _asym_table(keys):
            g = (df.dropna(subset=["_side"])
                   .groupby(keys + ["_side"])[["mean_velocity", "systole",
                                               "diastole", "pulsatility"]]
                   .median().unstack("_side"))
            out = {}
            for idx, row in g.iterrows():
                vals = {}
                for src, dst in (("mean_velocity", "asym_mfv"),
                                 ("systole", "asym_psv"),
                                 ("diastole", "asym_edv"),
                                 ("pulsatility", "asym_pi")):
                    try:
                        l, r = row[(src, "L")], row[(src, "R")]
                    except KeyError:
                        l = r = np.nan
                    den = (l + r)
                    vals[dst] = (abs(l - r) / den
                                 if np.isfinite(den) and abs(den) > 1e-12 else np.nan)
                out[idx if isinstance(idx, tuple) else (idx,)] = vals
            return out

        by_session = _asym_table([subj_col, sess_col])
        by_subject = _asym_table([subj_col])

        for i, row in df.iterrows():
            sys_, dia = float(row["systole"]), float(row["diastole"])
            feats = {dst: float(row[src]) for src, dst in TCD_COLUMNS.items()}
            feats["pulse_amplitude"] = sys_ - dia
            feats["resistance_index"] = (sys_ - dia) / sys_ if sys_ else np.nan
            feats["sd_ratio"] = sys_ / dia if dia else np.nan

            key_s = (row[subj_col], row[sess_col])
            asym = by_session.get(key_s) or by_subject.get((row[subj_col],)) or {}
            for k in ("asym_mfv", "asym_psv", "asym_edv", "asym_pi"):
                feats[k] = asym.get(k, np.nan)

            yield {"features": feats,
                   "subject": str(row[subj_col]),
                   "label": str(row[lab_col]),
                   "trial": int(i),
                   "covariates": {"depth": float(row["depth"]) if "depth" in df.columns else np.nan,
                                  "side": row["_side"], "session": str(row[sess_col])}}


def save_report(obj: dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str))
