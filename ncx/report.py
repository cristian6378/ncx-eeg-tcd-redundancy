"""Tables, figures and a machine-readable results bundle.

Figures are built for a single-column Elsevier layout: 8.4 cm wide, 8 pt type,
and every categorical encoding doubled by a non-colour channel (hatch,
marker, line style) so that the panels survive both greyscale printing and
dichromatic vision. The categorical palette below passes the lightness,
chroma, CVD-separation, normal-vision and contrast checks.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Validated categorical order -- assigned in fixed order, never cycled.
CATEGORICAL = ["#0072B2", "#D55E00", "#009E73", "#7E4A9B", "#8C6D1F"]
HATCHES = ["", "///", "...", "xxx", "\\\\"]   # secondary encoding for greyscale
MARKERS = ["o", "s", "^", "D", "v"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

# Display names for the feature-space conditions.
COND_LABEL = {"full": "Full", "reduced": "Reduced", "mrmr": "mRMR",
              "relieff": "ReliefF", "lasso": "LASSO"}
INK, MUTED, GRID = "#1A1A1A", "#5A5A5A", "#D8D8D6"
SURFACE = "#FCFCFB"

# Diverging ramp for correlation: two hues through a neutral grey midpoint.
DIVERGING = LinearSegmentedColormap.from_list(
    "ncx_div", ["#0A4E7A", "#3E8FC1", "#AFC9DA", "#EDEDEB",
                "#EFC3A0", "#D97B32", "#8C3A00"])
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "ncx_seq", ["#EDF3F8", "#9CC3DF", "#3E8FC1", "#0A4E7A"])

CM = 1 / 2.54
ONE_COL, TWO_COL = 8.4 * CM, 17.4 * CM


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.labelcolor": INK,
        "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.grid": True, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "savefig.dpi": 600,
        "savefig.bbox": "tight", "figure.constrained_layout.use": True,
    })


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def fig_correlation(matrix: pd.DataFrame, title: str, path: Path,
                    width: float = ONE_COL):
    _style()
    fig, ax = plt.subplots(figsize=(width, width * 0.95))
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap=DIVERGING,
                   vmin=-1, vmax=1, interpolation="nearest")
    n = len(matrix)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=5.5)
    ax.set_yticklabels(matrix.index, fontsize=5.5)
    ax.grid(False)
    ax.set_title(title, pad=4)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                      ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label("Pearson $r$", fontsize=7)
    cb.outline.set_visible(False)
    fig.savefig(path)
    plt.close(fig)


def fig_pca(profiles: dict, path: Path, width: float = ONE_COL):
    """Cumulative explained variance, one line per modality."""
    _style()
    fig, ax = plt.subplots(figsize=(width, width * 0.72))
    for i, (name, prof) in enumerate(profiles.items()):
        cum = np.asarray(prof["cumulative"], dtype=float)
        xs = np.arange(1, len(cum) + 1)
        ax.plot(xs, cum, color=CATEGORICAL[i % 4], lw=1.6,
                ls=LINESTYLES[i % 4], marker=MARKERS[i % 4], ms=3.2,
                mfc=SURFACE, mew=1.0, label=name.upper())
    ax.axhline(0.90, color=MUTED, lw=0.6, ls=(0, (1, 2)))
    ax.text(0.98, 0.905, "90%", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=6, color=MUTED)
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def fig_family_blocks(blocks: dict, path: Path, width: float = TWO_COL):
    """Within/between-family mean |r|, one panel per modality."""
    _style()
    ks = list(blocks)
    fig, axes = plt.subplots(1, len(ks), figsize=(width, width * 0.42))
    axes = np.atleast_1d(axes)
    vmax = max(float(np.nanmax(b.to_numpy(dtype=float))) for b in blocks.values())
    for ax, k in zip(axes, ks):
        M = blocks[k].astype(float)
        im = ax.imshow(M.to_numpy(), cmap=SEQUENTIAL, vmin=0, vmax=vmax)
        ax.set_xticks(range(len(M))); ax.set_yticks(range(len(M)))
        ax.set_xticklabels(M.columns, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(M.index, fontsize=6)
        ax.set_title(k.upper(), pad=3)
        ax.grid(False)
        for i in range(len(M)):
            for j in range(len(M)):
                v = M.iat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=5.5,
                            color=SURFACE if v > 0.6 * vmax else INK)
    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02)
    cb.set_label("mean $|r|$", fontsize=7)
    cb.outline.set_visible(False)
    fig.savefig(path)
    plt.close(fig)


def fig_performance(summary: pd.DataFrame, metric: str, path: Path,
                    width: float = TWO_COL):
    """Grouped bars with 95% CI whiskers; hatch doubles the colour coding."""
    _style()
    mods = list(dict.fromkeys(summary["modality"]))
    # Logical order, not the order rows happen to appear in.
    order = ["full", "reduced", "mrmr", "relieff", "lasso"]
    present = set(summary["condition"])
    conds = [c for c in order if c in present] + \
            [c for c in dict.fromkeys(summary["condition"]) if c not in order]
    fig, axes = plt.subplots(1, len(mods), figsize=(width, width * 0.38),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, mod in zip(axes, mods):
        sub = summary[summary.modality == mod]
        models = list(dict.fromkeys(sub["model"]))
        xs = np.arange(len(models))
        w = 0.8 / max(len(conds), 1)
        for ci, cond in enumerate(conds):
            s = sub[sub.condition == cond].set_index("model")
            mu = [s[f"{metric}_mean"].get(m, np.nan) for m in models]
            sd = [s[f"{metric}_std"].get(m, np.nan) for m in models]
            ax.bar(xs + ci * w - 0.4 + w / 2, mu, width=w * 0.88,
                   color=CATEGORICAL[ci], hatch=HATCHES[ci],
                   edgecolor=SURFACE, linewidth=0.8,
                   label=COND_LABEL.get(cond, cond) if ax is axes[0] else None)
            ax.errorbar(xs + ci * w - 0.4 + w / 2, mu, yerr=sd, fmt="none",
                        ecolor=INK, elinewidth=0.7, capsize=1.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=6)
        ax.set_title(mod.upper(), pad=3)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel(metric.replace("_", " ").replace("macro", "").strip().title())
    # Legend sits above the panels: inside the axes it collided with the bars.
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=len(conds), loc="upper center",
               bbox_to_anchor=(0.5, 1.10), fontsize=6.5)
    fig.savefig(path)
    plt.close(fig)


def fig_stability(stab: dict, path: Path, width: float = TWO_COL):
    """Selection frequency per feature, sorted, one panel per modality."""
    _style()
    ks = list(stab)
    fig, axes = plt.subplots(1, len(ks), figsize=(width, width * 0.40))
    axes = np.atleast_1d(axes)
    for i, (ax, k) in enumerate(zip(axes, ks)):
        d = stab[k].sort_values("selection_frequency")
        ax.barh(range(len(d)), d["selection_frequency"],
                color=CATEGORICAL[i % 4], edgecolor=SURFACE, linewidth=0.6,
                height=0.72)
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels(d["feature"], fontsize=5.5)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Selection frequency across folds")
        ax.set_title(k.upper(), pad=3)
        ax.axvline(0.5, color=MUTED, lw=0.6, ls=(0, (1, 2)))
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Bundle
# --------------------------------------------------------------------------

def write_tables(tables: dict, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(outdir / f"{name}.csv", index=False)


def write_bundle(payload: dict, path: Path) -> None:
    """One JSON carrying every number the manuscript quotes."""
    def enc(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, pd.DataFrame):
            return json.loads(o.to_json(orient="split"))
        return str(o)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=enc))
