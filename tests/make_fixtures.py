"""Synthetic fixtures used only to exercise the code paths.

These are not data. They exist so that `python run.py selftest` proves the
pipeline runs end to end -- extraction, fold-internal selection, nested
tuning, the clinical metrics, the corrected tests and every figure -- before
it is pointed at the real deposits. No number produced from these fixtures
should appear anywhere near the manuscript.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ncx.config import Config, feature_order
from ncx import features as feat


def _correlated_block(rng, n, k, rho):
    """k columns sharing a latent factor at correlation ~rho."""
    latent = rng.standard_normal(n)
    return np.stack([np.sqrt(rho) * latent + np.sqrt(1 - rho) *
                     rng.standard_normal(n) for _ in range(k)], axis=1)


def _synth_table(rng, modality: str, n_subjects: int, per_subject: int,
                 classes, effect: float, rho_within: float):
    names = feature_order(modality)
    n = n_subjects * per_subject
    subj = np.repeat([f"{modality}{i:02d}" for i in range(n_subjects)], per_subject)
    y = rng.choice(classes, size=n_subjects)
    y = np.repeat(y, per_subject)

    # Family-structured redundancy: strong within family, weak across.
    from ncx.config import FEATURE_FAMILIES
    cols = {}
    for fam, members in FEATURE_FAMILIES[modality].items():
        block = _correlated_block(rng, n, len(members), rho_within)
        for j, m in enumerate(members):
            cols[m] = block[:, j]
    X = np.stack([cols[c] for c in names], axis=1)

    # A class effect on a handful of columns, plus subject-level offsets.
    signal_idx = [0, 1, len(names) - 2, len(names) - 1]
    for ci, c in enumerate(classes):
        m = y == c
        X[np.ix_(m, signal_idx)] += effect * (ci - (len(classes) - 1) / 2)
    off = rng.standard_normal((n_subjects, len(names))) * 0.3
    X += np.repeat(off, per_subject, axis=0)

    df = pd.DataFrame(X, columns=names)
    df["subject"], df["label"] = subj, y
    df["trial"] = np.tile(np.arange(per_subject), n_subjects)
    return df


def check_extractors(cfg: Config, rng) -> dict:
    """Smoke-test both extractors on synthetic signals."""
    fs = 250.0
    t = np.arange(0, 4.0, 1 / fs)
    ch = ["FP1", "FP2", "C3", "C4", "P3", "P4", "O1", "O2"]
    x = np.stack([np.sin(2 * np.pi * 10 * t) * (1 + 0.2 * i)
                  + 0.6 * rng.standard_normal(t.size) for i in range(len(ch))])
    ef = feat.eeg_trial_features(x, fs, ch, cfg, complexity_max_ch=4)

    fs_t = 125.0
    tt = np.arange(0, 12.0, 1 / fs_t)
    beat = (np.maximum(0, np.sin(2 * np.pi * 1.2 * tt)) ** 2)
    left = 40 + 60 * beat + 1.5 * rng.standard_normal(tt.size)
    right = 38 + 52 * beat + 1.5 * rng.standard_normal(tt.size)
    tf = feat.tcd_record_features(left, right, fs_t)

    return {"eeg_n": len(ef), "eeg_finite": int(np.sum(np.isfinite(list(ef.values())))),
            "tcd_n": len(tf), "tcd_finite": int(np.sum(np.isfinite(list(tf.values()))))}


def make_fixtures(cfg: Config, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    d = cfg.dirs()

    chk = check_extractors(cfg, rng)
    print(f"  extractor smoke test: EEG {chk['eeg_finite']}/{chk['eeg_n']} finite, "
          f"TCD {chk['tcd_finite']}/{chk['tcd_n']} finite")

    eeg = _synth_table(rng, "eeg", n_subjects=40, per_subject=8,
                       classes=["left", "right"], effect=0.9, rho_within=0.72)
    tcd = _synth_table(rng, "tcd", n_subjects=36, per_subject=4,
                       classes=["normal", "abnormal"], effect=0.8, rho_within=0.55)
    eeg.to_csv(d["cache"] / "features_eeg.csv", index=False)
    tcd.to_csv(d["cache"] / "features_tcd.csv", index=False)
    print(f"  fixtures: EEG {eeg.shape}, TCD {tcd.shape}")
    return chk
