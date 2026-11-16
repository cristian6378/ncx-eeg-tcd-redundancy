"""Central configuration for the cross-modal redundancy study.

Every tunable that affects a reported number lives here, so that the
manuscript's Methods section and the code cannot drift apart.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


# --------------------------------------------------------------------------
# Signal / feature parameters
# --------------------------------------------------------------------------

EEG_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 18.0),
}

# Homologous electrode pairs for the pairwise derived brain symmetry index.
# Names follow the 10-20 montage used by Liu et al. (2024). Pairs absent from
# a given recording are skipped and reported, never silently imputed.
# Verified against task-motor-imagery_channels.tsv of the deposit: the montage
# uses the older T3/T4/T5/T6 labels, not T7/T8/P7/P8. Midline channels
# (Fz, FCz, Cz, Pz, Oz) have no homologue and are excluded from pDBSI.
DEFAULT_HOMOLOGOUS_PAIRS = [
    ("FP1", "FP2"), ("F3", "F4"), ("F7", "F8"), ("FC3", "FC4"),
    ("FT7", "FT8"), ("C3", "C4"), ("T3", "T4"), ("CP3", "CP4"),
    ("TP7", "TP8"), ("P3", "P4"), ("T5", "T6"), ("O1", "O2"),
]

# The 17-feature qEEG candidate space, grouped by domain. The grouping is what
# lets us pose the cross-modal question, so it is declared explicitly rather
# than inferred from column names downstream.
EEG_FEATURE_FAMILIES = {
    "magnitude":   ["delta", "theta", "alpha", "beta"],
    "ratio":       ["delta_alpha", "theta_alpha", "dt_ab"],
    "distribution":["skewness", "kurtosis"],
    "complexity":  ["sample_entropy", "higuchi_fd"],
    "coupling":    ["wpli", "ms_coherence"],
    "asymmetry":   ["pdbsi_delta", "pdbsi_theta", "pdbsi_alpha", "pdbsi_beta"],
}

# The TCD candidate space is deliberately constructed to mirror the EEG
# taxonomy family-for-family and at matched cardinality (17), so that any
# difference in redundancy structure cannot be attributed to one modality
# simply having been given more, or more heterogeneous, candidates.
# The TCD deposit (Thorpe et al., 2020) distributes derived per-insonation
# descriptors rather than envelope time series, so entropy, fractal dimension
# and within-beat timing cannot be computed. The taxonomy therefore reflects
# what the source actually supports: the velocity triple and its ratios, the
# waveform-morphology scores released with the deposit, and bilateral
# asymmetry. Insonation depth is held out as an acquisition covariate, not a
# feature -- a model allowed to use it would be exploiting protocol rather
# than physiology.
TCD_FEATURE_FAMILIES = {
    "magnitude":   ["psv", "edv", "mfv", "pulse_amplitude"],
    "ratio":       ["pulsatility_index", "resistance_index", "sd_ratio"],
    "morphology":  ["onset_score", "peaks_score", "canopy_score", "curvature"],
    "asymmetry":   ["asym_mfv", "asym_psv", "asym_edv", "asym_pi"],
}


@dataclass
class Config:
    # --- paths -----------------------------------------------------------
    eeg_root: Path = Path("data/eeg")
    tcd_root: Path = Path("data/tcd")
    out_root: Path = Path("results")

    # --- EEG preprocessing ----------------------------------------------
    eeg_fs: float = 500.0            # Liu et al. (2024) native rate
    hp_hz: float = 0.5               # high-pass, drift removal
    lp_hz: float = 45.0              # low-pass, EMG attenuation
    notch_hz: float | None = 50.0    # mains
    notch_q: float = 30.0
    filter_order: int = 4
    epoch_s: float | None = None     # None = use full trial

    # --- TCD ------------------------------------------------------------
    tcd_fs: float = 125.0            # overridden by the loader when declared
    tcd_has_bilateral: bool = True   # set False if only one insonation side

    # --- redundancy pipeline --------------------------------------------
    corr_threshold: float = 0.60     # Stage 1, |r|
    vif_threshold: float = 10.0      # Stage 2
    pca_variance_targets: tuple = (0.80, 0.90, 0.95)

    # --- evaluation ------------------------------------------------------
    n_splits: int = 10
    n_repeats: int = 10
    inner_splits: int = 5
    n_search_iter: int = 50
    random_state: int = 20260927
    scoring: str = "f1_macro"

    # --- statistics ------------------------------------------------------
    alpha: float = 0.05
    n_bootstrap: int = 10000

    # --- misc ------------------------------------------------------------
    homologous_pairs: list = field(default_factory=lambda: list(DEFAULT_HOMOLOGOUS_PAIRS))
    n_jobs: int = 1                  # 1 keeps the run bit-for-bit reproducible
    # Hyperparameter search may be parallelised safely: every estimator is
    # deterministic and seeded, so workers change scheduling, not results.
    search_jobs: int = -1
    extract_jobs: int = -1           # feature extraction is pure per trial

    def dirs(self) -> dict:
        d = {
            "tables": self.out_root / "tables",
            "figures": self.out_root / "figures",
            "cache": self.out_root / "cache",
            "logs": self.out_root / "logs",
        }
        for p in d.values():
            p.mkdir(parents=True, exist_ok=True)
        return d

    def save(self, path: Path) -> None:
        payload = {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in asdict(self).items()}
        Path(path).write_text(json.dumps(payload, indent=2, default=str))


FEATURE_FAMILIES = {"eeg": EEG_FEATURE_FAMILIES, "tcd": TCD_FEATURE_FAMILIES}

FAMILY_ORDER = ["magnitude", "ratio", "distribution", "complexity",
                "coupling", "morphology", "asymmetry"]

# Families present in BOTH modalities with the same cardinality. The strict
# cross-modal comparison is made over this matched core (11 descriptors each);
# the native sets are larger and unequal, and are reported as such.
MATCHED_CORE_FAMILIES = ["magnitude", "ratio", "asymmetry"]


def feature_order(modality: str) -> list[str]:
    """Canonical column order for a modality, grouped by family."""
    out = []
    for fam in FAMILY_ORDER:
        out.extend(FEATURE_FAMILIES[modality].get(fam, []))
    return out


def matched_core(modality: str) -> list[str]:
    """The 11 descriptors that are family- and cardinality-matched across
    modalities: magnitude (4), ratio (3), asymmetry (4)."""
    out = []
    for fam in MATCHED_CORE_FAMILIES:
        out.extend(FEATURE_FAMILIES[modality].get(fam, []))
    return out


def family_of(modality: str, name: str) -> str:
    for fam, names in FEATURE_FAMILIES[modality].items():
        if name in names:
            return fam
    return "unknown"
