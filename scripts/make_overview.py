"""Schematic overview of the matched-taxonomy benchmark.

Drawn rather than plotted: the point of the figure is the *architecture* of
the comparison -- which stages differ between modalities and which are held
identical -- and that is a claim about the design, not a summary of the data.
Colour follows the manuscript's established palette, so the two modalities
read the same here as in every results figure, and the two hues are the
Okabe--Ito blue/vermillion pair that stays distinguishable under the common
forms of colour-vision deficiency.

The layout runs top to bottom because the argument does: two cohorts enter by
separate routes, the routes are forced to converge on a common taxonomy, and
everything below that point is shared. Stages are drawn side by side only
while they genuinely differ.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

from ncx.report import CATEGORICAL, INK, MUTED, SURFACE, _style

_style()

EEG_C, TCD_C = CATEGORICAL[0], CATEGORICAL[1]
EEG_F, TCD_F = "#EAF2F8", "#FBEFE7"
SHARED_E, SHARED_F = "#7A7A76", "#EFEFEC"
VALID_E, VALID_F = "#A03828", "#FBEEEB"
OUT_E, OUT_F = "#1E6B52", "#E9F3EF"

# lane geometry
LX, RX, BW = 0.045, 0.525, 0.430          # left / right lane origin, width
CX = 0.5                                   # centre


def box(ax, x, y, w, h, text, *, edge, fill="white", fs=6.5, lw=0.9,
        ls="-", z=3):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.010",
        linewidth=lw, edgecolor=edge, facecolor=fill, linestyle=ls, zorder=z))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, zorder=z + 1, linespacing=1.4)


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=0.9, ls="-", z=2):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, linewidth=lw,
        color=color, linestyle=ls, zorder=z, shrinkA=0, shrinkB=0))


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_facecolor(SURFACE)

    H = 0.098                                  # standard box height
    y1, y2, y3 = 0.880, 0.735, 0.590           # sources / preproc / native
    y4 = 0.455                                 # matched core
    y5 = 0.288                                 # shared pipeline row
    y6 = 0.145                                 # validation band
    y7 = 0.018                                 # outputs

    # ================= 1. cohorts ========================================
    box(ax, LX, y1, BW, H,
        "$\\bf{qEEG\\ cohort}$   Liu et al. (2024)\n"
        "50 stroke patients · 2000 motor-imagery trials\n"
        "29 channels · 500 Hz · left vs right imagery",
        edge=EEG_C, fill=EEG_F)
    box(ax, RX, y1, BW, H,
        "$\\bf{TCD\\ cohort}$   Thorpe et al. (2020)\n"
        "104 patients · 951 MCA insonations\n"
        "LVO vs in- and out-of-hospital controls",
        edge=TCD_C, fill=TCD_F)

    # ================= 2. modality-specific preprocessing ================
    box(ax, LX, y2, BW, H,
        "Band-pass and notch filtering\n"
        "channel roles inferred from signal statistics\n"
        "restricted to the 4 s imagery window",
        edge=EEG_C)
    box(ax, RX, y2, BW, H,
        "Per-insonation descriptors taken as released\n"
        "bilateral pairing within session\n"
        "insonation depth excluded from the features",
        edge=TCD_C)
    ax.text(CX, y2 - 0.016, "the only modality-specific stage", ha="center",
            va="top", fontsize=6.3, style="italic", color=MUTED)

    # ================= 3. native taxonomies ==============================
    box(ax, LX, y3, BW, H,
        "$\\bf{native\\ set:\\ 17\\ descriptors}$\n"
        "magnitude · ratio · distribution\n"
        "complexity · coupling · asymmetry",
        edge=EEG_C)
    box(ax, RX, y3, BW, H,
        "$\\bf{native\\ set:\\ 15\\ descriptors}$\n"
        "magnitude · ratio\n"
        "morphology · asymmetry",
        edge=TCD_C)

    # ================= 4. matched core ===================================
    box(ax, 0.185, y4, 0.630, 0.085,
        "$\\bf{matched\\ core:\\ 11\\ vs\\ 11\\ descriptors}$    "
        "magnitude 4 · ratio 3 · asymmetry 4\n"
        "identical family structure and cardinality, so any difference in "
        "redundancy\ncannot be attributed to one modality having been given "
        "more candidates",
        edge=SHARED_E, fill=SHARED_F, fs=6.4)

    # ================= 5. shared pipeline ================================
    ax.add_patch(Rectangle((0.028, y5 - 0.036), 0.944, 0.152,
                           linewidth=1.0, edgecolor=SHARED_E,
                           facecolor=SHARED_F, linestyle=(0, (4, 2)), zorder=1))
    ax.text(0.5, y5 + 0.122, "one pipeline, applied identically to both modalities",
            ha="center", va="bottom", fontsize=6.6, style="italic",
            color=SHARED_E, zorder=2)
    pw, gap = 0.292, 0.026
    px = [0.050, 0.050 + pw + gap, 0.050 + 2 * (pw + gap)]
    box(ax, px[0], y5, pw, H,
        "$\\bf{Redundancy\\ reduction}$\n"
        "pairwise $|r| \\geq 0.6$ screening\n"
        "then iterative VIF $> 10$",
        edge=SHARED_E, fs=6.4)
    box(ax, px[1], y5, pw, H,
        "$\\bf{Supervised\\ baselines}$\n"
        "mRMR · ReliefF · LASSO\n"
        "at matched cardinality",
        edge=SHARED_E, fs=6.4)
    box(ax, px[2], y5, pw, H,
        "$\\bf{Five\\ classifiers}$\n"
        "logistic regression · linear and RBF SVM\n"
        "decision tree · random forest",
        edge=SHARED_E, fs=6.4)

    # ================= 6. validation band ================================
    box(ax, 0.050, y6, 0.900, 0.078,
        "$\\bf{Leakage\\!-\\!aware\\ validation}$    "
        "StratifiedGroupKFold grouped by patient — every session of a patient "
        "in one fold\n"
        "5 × 5 outer folds · nested inner randomised search · "
        "scaling, screening and every selector re-fitted inside the training fold alone",
        edge=VALID_E, fill=VALID_F, fs=6.4, lw=1.0)

    # ================= 7. outputs ========================================
    ow, ogap = 0.292, 0.026
    ox = [0.050, 0.050 + ow + ogap, 0.050 + 2 * (ow + ogap)]
    box(ax, ox[0], y7, ow, 0.098,
        "$\\bf{Redundancy\\ structure}$\n"
        "mean $|r|$, PCA, family blocks\n"
        "bootstrap CI on the modality gap\n"
        "56-cell threshold sweep",
        edge=OUT_E, fill=OUT_F, fs=6.3)
    box(ax, ox[1], y7, ow, 0.098,
        "$\\bf{Cost\\ of\\ reduction}$\n"
        "repeated-measures ANOVA\n"
        "TOST equivalence with power\n"
        "selector stability index",
        edge=OUT_E, fill=OUT_F, fs=6.3)
    box(ax, ox[2], y7, ow, 0.098,
        "$\\bf{Validity\\ controls}$\n"
        "permutation null, 1000 shuffles\n"
        "five channel-pooling variants\n"
        "within-subject decoding",
        edge=OUT_E, fill=OUT_F, fs=6.3)

    # ================= arrows ============================================
    for x, c in ((LX + BW / 2, EEG_C), (RX + BW / 2, TCD_C)):
        arrow(ax, x, y1, x, y2 + H + 0.008, color=c)          # 1 -> 2
        arrow(ax, x, y2, x, y3 + H + 0.008, color=c)          # 2 -> 3
    # native -> matched core (converging)
    arrow(ax, LX + BW / 2, y3, 0.40, y4 + 0.085 + 0.008, color=EEG_C)
    arrow(ax, RX + BW / 2, y3, 0.60, y4 + 0.085 + 0.008, color=TCD_C)
    # core -> pipeline
    arrow(ax, CX, y4, CX, y5 + H + 0.044, color=SHARED_E)
    # pipeline internal
    for a, b in ((0, 1), (1, 2)):
        arrow(ax, px[a] + pw + 0.004, y5 + H / 2, px[b] - 0.004, y5 + H / 2,
              color=SHARED_E, lw=0.8)
    # pipeline -> validation
    arrow(ax, CX, y5 - 0.036, CX, y6 + 0.078 + 0.006, color=VALID_E)
    # validation -> outputs
    for x in (ox[0] + ow / 2, ox[1] + ow / 2, ox[2] + ow / 2):
        arrow(ax, x, y6, x, y7 + 0.098 + 0.006, color=OUT_E)

    # PNG for the working build; PDF because the journal prefers vector for
    # line drawings and a vector schematic sidesteps the 1000 dpi bitmap rule.
    fig.savefig("results/figures/fig_overview.png")
    fig.savefig("results/figures/fig_overview.pdf")
    plt.close(fig)
    print("wrote fig_overview.png and fig_overview.pdf")


if __name__ == "__main__":
    main()
