"""Graphical abstract, sized to the journal's specification.

BSPC asks for 531 x 1328 pixels (h x w) or proportionally more, readable at
5 x 13 cm. That is an unusually wide, short canvas, so the design carries one
claim only -- the same taxonomy yields different redundancy in the two
modalities -- and everything else is cut. The two correlation panels are real
matched-core matrices from the analysis rather than schematic stand-ins,
because the visual difference between them is the finding.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from ncx.config import matched_core
from ncx.report import CATEGORICAL, DIVERGING, INK, MUTED, SURFACE, _style

_style()
EEG_C, TCD_C = CATEGORICAL[0], CATEGORICAL[1]
CM = 1 / 2.54


def core_matrix(df, cols):
    d = df.dropna(subset=cols)
    R = np.abs(np.corrcoef(d[cols].to_numpy(dtype=float), rowvar=False))
    return np.nan_to_num(R)


def main() -> None:
    eeg = pd.read_csv("results/cache/features_eeg.csv")
    tcd = pd.read_csv("results/cache/features_tcd.csv")
    ce = [c for c in matched_core("eeg") if c in eeg.columns]
    ct = [c for c in matched_core("tcd") if c in tcd.columns]
    Re, Rt = core_matrix(eeg, ce), core_matrix(tcd, ct)

    # Exactly 13 x 5 cm, the size the journal says it must be readable at, at
    # 600 dpi -> 3071 x 1181 px, above the 1328 x 531 floor and on the same
    # 2.6:1 aspect so nothing is rescaled at production.
    # The shared style turns constrained layout on, which re-fits the canvas;
    # here the canvas size IS the specification, so it is switched off.
    # The shared style turns on constrained layout and a tight save box, both
    # of which re-fit the canvas to its content. Here the canvas size IS the
    # specification (13 x 5 cm), so both are switched off. Note that passing
    # bbox_inches=None to savefig does NOT do this -- None means "use the
    # rcParam", which is exactly the tight box we are trying to avoid.
    plt.rcParams["figure.constrained_layout.use"] = False
    plt.rcParams["savefig.bbox"] = None
    fig = plt.figure(figsize=(13 * CM, 5 * CM))
    fig.patch.set_facecolor(SURFACE)

    # ---- title and byline ------------------------------------------------
    fig.text(0.5, 0.985,
             "Two origins of feature redundancy in stroke biosignals:",
             ha="center", va="top", fontsize=7.0, weight="bold", color=INK)
    fig.text(0.5, 0.890,
             "algebraic construction in transcranial Doppler,\n"
             "shared physiology in quantitative EEG",
             ha="center", va="top", fontsize=6.0, color=INK, linespacing=1.35)
    fig.text(0.5, 0.762, "Valchev et al., 2026",
             ha="center", va="top", fontsize=5.8, style="italic", color=MUTED)

    # ---- the two correlation panels -------------------------------------
    for i, (R, name, colour, mean_r, kept) in enumerate([
            (Re, "qEEG", EEG_C, 0.229, 7), (Rt, "TCD", TCD_C, 0.316, 4)]):
        ax = fig.add_axes([0.055 + i * 0.175, 0.280, 0.138, 0.345])
        ax.imshow(R, cmap=DIVERGING, vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(colour); sp.set_linewidth(1.3)
        ax.set_title(name, fontsize=6.8, color=colour, weight="bold", pad=2.5)
        ax.text(0.5, -0.10, f"mean $|r|$ = {mean_r:.3f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=6.0,
                color=INK)
        ax.text(0.5, -0.33, f"retains {kept} of 11", transform=ax.transAxes,
                ha="center", va="top", fontsize=6.2, color=colour,
                weight="bold")

    # ---- colour key ------------------------------------------------------
    cax = fig.add_axes([0.075, 0.048, 0.255, 0.024])
    cax.imshow(np.linspace(0, 1, 256).reshape(1, -1), cmap=DIVERGING,
               aspect="auto")
    cax.set_yticks([]); cax.set_xticks([0, 255])
    cax.set_xticklabels(["$|r|$ = 0", "1"], fontsize=5.6)
    cax.tick_params(length=0, pad=1)
    for sp in cax.spines.values():
        sp.set_edgecolor(MUTED); sp.set_linewidth(0.5)

    # ---- arrow -----------------------------------------------------------
    fig.patches.append(FancyArrowPatch(
        (0.392, 0.390), (0.448, 0.390), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=8, linewidth=1.1, color=MUTED))

    # ---- result block ----------------------------------------------------
    fig.patches.append(FancyBboxPatch(
        (0.472, 0.075), 0.500, 0.575, transform=fig.transFigure,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        linewidth=1.1, edgecolor="#1E6B52", facecolor="#EAF4F0"))

    cxr = 0.722
    fig.text(cxr, 0.618, "TCD is the more redundant space",
             ha="center", va="top", fontsize=6.8, weight="bold",
             color="#14503D")
    fig.text(cxr, 0.520,
             "gap in mean $|r|$   $+0.087$   95% CI $[+0.045,\ +0.157]$",
             ha="center", va="top", fontsize=6.2, color=INK)
    fig.text(cxr, 0.432, "holds at all 56 screening-threshold pairs",
             ha="center", va="top", fontsize=6.0, color=MUTED)
    fig.text(cxr, 0.348,
             "no detectable cost of pruning; TCD equivalence untested",
             ha="center", va="top", fontsize=6.0, color=INK)
    fig.text(cxr, 0.245,
             "Algebraic vs. physiological redundancy:\n"
             "pruning recipes do not transfer",
             ha="center", va="top", fontsize=6.3, weight="bold",
             color="#14503D", linespacing=1.4)

    for ext in ("png", "pdf", "eps", "tiff"):
        fig.savefig(f"results/figures/graphical_abstract.{ext}",
                    dpi=600, facecolor=SURFACE)
    plt.close(fig)
    print("wrote graphical_abstract.{png,pdf,eps,tiff}")


if __name__ == "__main__":
    main()
