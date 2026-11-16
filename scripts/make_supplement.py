"""Build the supplementary material from reference_results/ (or a fresh run).

Writes, under supplement/build/:
  tables/S*.tex                       LaTeX tables input by supplementary_material.tex
  fig_S1_equivalence.png/.pdf         corrected vs uncorrected TOST intervals
  fig_S2_selection_frequency.png/.pdf selection frequency, four selectors
  Supplementary_Data_S2.xlsx          every numeric result, one sheet per table

Nothing in the supplement is typed by hand: every number is read from the
result files, so it cannot drift from the analysis.

    python scripts/make_supplement.py               # from reference_results/
    python scripts/make_supplement.py --src results/tables
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ncx.config import FEATURE_FAMILIES, feature_order  # noqa: E402

MODELS = ["Logistic Regression", "SVM (Linear)", "SVM (RBF)",
          "Decision Tree", "Random Forest"]
CONDS = ["full", "reduced", "mrmr", "relieff", "lasso"]
COND_LABEL = {"full": "Full", "reduced": "Reduced", "mrmr": "mRMR",
              "relieff": "ReliefF", "lasso": "LASSO"}
METRICS = ["macro_f1", "balanced_accuracy", "sensitivity_macro", "specificity_macro"]
METRIC_LABEL = {"macro_f1": "Macro-F1", "balanced_accuracy": "Balanced accuracy",
                "sensitivity_macro": "Macro sensitivity",
                "specificity_macro": "Macro specificity",
                "ppv_macro": "Macro PPV", "npv_macro": "Macro NPV",
                "auc_macro_ovr": "Macro AUC (OvR)"}
MOD_LABEL = {"eeg": "qEEG", "tcd": "TCD"}

TEX = {
    "delta": r"$\delta$ power", "theta": r"$\theta$ power",
    "alpha": r"$\alpha$ power", "beta": r"$\beta$ power",
    "delta_alpha": r"$\delta/\alpha$", "theta_alpha": r"$\theta/\alpha$",
    "dt_ab": r"$(\delta{+}\theta)/(\alpha{+}\beta)$",
    "skewness": "Skewness", "kurtosis": "Kurtosis",
    "sample_entropy": "Sample entropy", "higuchi_fd": "Higuchi FD",
    "wpli": "wPLI", "ms_coherence": "MS coherence",
    "pdbsi_delta": r"pDBSI $\delta$", "pdbsi_theta": r"pDBSI $\theta$",
    "pdbsi_alpha": r"pDBSI $\alpha$", "pdbsi_beta": r"pDBSI $\beta$",
    "psv": "PSV", "edv": "EDV", "mfv": "MFV", "pulse_amplitude": "Pulse amplitude",
    "pulsatility_index": "PI", "resistance_index": "RI", "sd_ratio": "S/D ratio",
    "onset_score": "Onset score", "peaks_score": "Peaks score",
    "canopy_score": "Canopy score", "curvature": "Curvature",
    "asym_mfv": "Asym.\\ MFV", "asym_psv": "Asym.\\ PSV",
    "asym_edv": "Asym.\\ EDV", "asym_pi": "Asym.\\ PI",
}
PLAIN = {k: (v.replace("$", "").replace("\\delta", "δ").replace("\\theta", "θ")
             .replace("\\alpha", "α").replace("\\beta", "β").replace("{+}", "+")
             .replace("\\ ", " "))
         for k, v in TEX.items()}


def family(mod: str, name: str) -> str:
    for fam, names in FEATURE_FAMILIES[mod].items():
        if name in names:
            return fam
    return ""


def f(x, nd=4, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "$-$", 1) if s.startswith("-") else s


def p_fmt(p):
    if p is None or not np.isfinite(p):
        return "--"
    if p < 1e-4:
        return r"$<10^{-4}$"
    return f"{p:.4f}" if p < 0.001 else f"{p:.3f}"


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def table(caption, label, colspec, header, body, size=r"\small", note=None):
    num = label.split(":")[1]
    out = [r"\begin{table}[!htbp]", r"\centering",
           rf"\renewcommand{{\thetable}}{{{num}}}",
           rf"\caption{{{caption}}}", rf"\label{{{label}}}", size,
           rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", header + r" \\",
           r"\midrule", *body, r"\bottomrule", r"\end{tabular}"]
    if note:
        out += [r"\par\smallskip", r"\begin{minipage}{0.95\linewidth}\footnotesize",
                note, r"\end{minipage}"]
    out.append(r"\end{table}")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def t_equivalence(eq: pd.DataFrame, label: str, caption: str) -> str:
    body = []
    for mod in ("eeg", "tcd"):
        for mi, metric in enumerate(METRICS):
            g = eq[(eq.modality == mod) & (eq.metric == metric)].set_index("model")
            for i, model in enumerate(MODELS):
                r = g.loc[model]
                mod_cell = MOD_LABEL[mod] if (mi == 0 and i == 0) else ""
                met_cell = METRIC_LABEL[metric] if i == 0 else ""
                body.append(
                    f"{mod_cell} & {met_cell} & {model} & {f(r.delta, sign=True)} & "
                    f"$[{r.ci90_lo:+.3f},\\,{r.ci90_hi:+.3f}]$ & "
                    f"{p_fmt(r.p_tost_holm)} & {'yes' if r.equivalent else 'no'} \\\\")
            if not (mod == "tcd" and metric == METRICS[-1]):
                body.append(r"\addlinespace[2pt]")
        if mod == "eeg":
            body.append(r"\midrule")
    hdr = (r"Modality & Metric & Model & $\Delta$ & 90\% CI & "
           r"$p_{\mathrm{TOST}}$ (Holm) & Equiv.")
    return table(caption, label, "lllcccc", hdr, body, size=r"\footnotesize")


def t_conditions(summary: pd.DataFrame) -> str:
    body = []
    for mod in ("eeg", "tcd"):
        for i, model in enumerate(MODELS):
            cells = []
            for c in CONDS:
                r = summary[(summary.modality == mod) & (summary.model == model)
                            & (summary.condition == c)].iloc[0]
                cells.append(f"{r.macro_f1_mean:.4f}\\,$\\pm$\\,{r.macro_f1_std:.4f}")
            body.append(f"{MOD_LABEL[mod] if i == 0 else ''} & {model} & "
                        + " & ".join(cells) + r" \\")
        if mod == "eeg":
            body.append(r"\midrule")
    hdr = "Modality & Model & " + " & ".join(COND_LABEL[c] for c in CONDS)
    return table(
        r"Macro-F1 (mean $\pm$ SD over 25 subject-grouped folds) under all five "
        r"feature-space conditions. Supervised selectors are run at the cardinality "
        r"of the label-agnostic reduced set ($k=12$ for qEEG, $k=7$ for TCD). All "
        r"other metrics are in Supplementary Data S2.",
        "tab:S3", "llccccc", hdr, body, size=r"\footnotesize")


def t_selectors(comp: pd.DataFrame) -> str:
    c = comp[(comp.metric == "macro_f1") & (comp.comparison != "full_vs_reduced")]
    body = []
    for mod in ("eeg", "tcd"):
        for si, sel in enumerate(("mrmr", "relieff", "lasso")):
            g = c[(c.modality == mod) & (c.comparison == f"reduced_vs_{sel}")].set_index("model")
            for i, model in enumerate(MODELS):
                r = g.loc[model]
                body.append(
                    f"{MOD_LABEL[mod] if (si == 0 and i == 0) else ''} & "
                    f"{COND_LABEL[sel] if i == 0 else ''} & {model} & "
                    f"{f(r.delta_mean, sign=True)} & {f(r.cliffs_delta, 3, True)} & "
                    f"{p_fmt(r.p_wilcoxon_holm)} & {p_fmt(r.p_nadeau_bengio)} & "
                    f"{p_fmt(r.p_nadeau_bengio_holm)} \\\\")
            if not (mod == "tcd" and sel == "lasso"):
                body.append(r"\addlinespace[2pt]")
        if mod == "eeg":
            body.append(r"\midrule")
    hdr = (r"Modality & Selector & Model & $\Delta$ & Cliff's $d$ & "
           r"$p_{\mathrm{W}}$ (Holm) & $p_{\mathrm{NB}}$ & $p_{\mathrm{NB}}$ (Holm)")
    return table(
        r"Label-agnostic reduced set against each supervised selector at matched "
        r"cardinality, Macro-F1, both modalities (the TCD rows extend Table~15 of the "
        r"main text). $\Delta$ is reduced minus selector. $p_{\mathrm{W}}$: Wilcoxon "
        r"signed-rank, uncorrected for fold dependence; $p_{\mathrm{NB}}$: "
        r"Nadeau--Bengio corrected resampled $t$. Holm adjustment is across the ten "
        r"model--modality pairs of each comparison. Inference rests on the corrected "
        r"test; under it no difference is significant.",
        "tab:S4", "lllccccc", hdr, body, size=r"\footnotesize")


def t_chance(vc: pd.DataFrame) -> str:
    body = []
    for mod in ("eeg", "tcd"):
        for i, model in enumerate(MODELS):
            cells = []
            for c in CONDS:
                r = vc[(vc.modality == mod) & (vc.model == model) & (vc.condition == c)
                       & (vc.metric == "macro_f1")].iloc[0]
                star = "$^{*}$" if r.p_holm < 0.05 else ""
                cells.append(f"{f(r.delta, 3, True)}{star}")
            body.append(f"{MOD_LABEL[mod] if i == 0 else ''} & {model} & "
                        + " & ".join(cells) + r" \\")
        if mod == "eeg":
            body.append(r"\midrule")
    hdr = "Modality & Model & " + " & ".join(COND_LABEL[c] for c in CONDS)
    return table(
        r"Difference between fold-wise Macro-F1 and the analytic chance level "
        r"($0.5$ for qEEG, $1/3$ for TCD) under every condition. One-sample Wilcoxon "
        r"test, Holm-adjusted across all 100 tests; $^{*}$ adjusted $p<0.05$. Every TCD "
        r"cell lies far above chance; no qEEG cell lies above it, and the decision tree "
        r"lies significantly below it under every condition.",
        "tab:S5", "llccccc", hdr, body, size=r"\footnotesize")


def t_chains(st: dict) -> str:
    body = []
    for mod in ("eeg", "tcd"):
        for sp in ("native", "core"):
            s = st[mod][sp]
            kept = ", ".join(TEX[n] for n in feature_order(mod) if n in s["kept"])
            dropped = ", ".join(TEX[n] for n in feature_order(mod)
                                if n in s["dropped_corr"]) or "--"
            vif = max(s["final_vif"].values()) if s.get("final_vif") else float("nan")
            body.append(
                rf"{MOD_LABEL[mod] if sp == 'native' else ''} & {sp} & "
                rf"{s['n']}$\rightarrow${s['retained']} & "
                rf"\parbox[t]{{0.33\linewidth}}{{\raggedright {kept}}} & "
                rf"\parbox[t]{{0.27\linewidth}}{{\raggedright {dropped}}} & {vif:.2f} \\")
            body.append(r"\addlinespace[3pt]")
    body = body[:-1]
    hdr = (r"Modality & Set & $n$ & Retained & Removed by $|r|\ge0.6$ screening & "
           r"Max VIF")
    return table(
        r"Reduction chains in detail. Descriptors are listed in taxonomy order. The "
        r"VIF stage ($>10$) removed nothing in any set; the last column is the largest "
        r"VIF among the retained descriptors.",
        "tab:S6", "llcllc", hdr, body, size=r"\footnotesize")


def _block(st, mod):
    fb = st[mod]["family_blocks"]
    return pd.DataFrame(fb["data"], index=fb["index"], columns=fb["columns"])


FAM_ORDER = ["magnitude", "ratio", "asymmetry", "distribution",
             "complexity", "coupling", "morphology"]


def t_blocks(st: dict) -> str:
    parts = []
    for mod in ("eeg", "tcd"):
        b = _block(st, mod)
        fams = [x for x in FAM_ORDER if x in b.index]
        rows = []
        for i, a in enumerate(fams):
            cells = [f"{b.loc[a, c]:.3f}" if j <= i else "" for j, c in enumerate(fams)]
            cells[i] = rf"\textbf{{{b.loc[a, a]:.3f}}}"
            rows.append(f"{a.capitalize()} & " + " & ".join(cells) + r" \\")
        hdr = "& " + " & ".join(x.capitalize() for x in fams)
        parts.append(
            rf"\textit{{{MOD_LABEL[mod]}}}\par\smallskip" "\n"
            rf"\begin{{tabular}}{{l{'c' * len(fams)}}}" "\n" r"\toprule" "\n"
            + hdr + r" \\" "\n" r"\midrule" "\n" + "\n".join(rows) + "\n"
            r"\bottomrule" "\n" r"\end{tabular}")
    return ("\\begin{table}[!htbp]\n\\centering\n\\renewcommand{\\thetable}{S7}\n"
            r"\caption{Mean absolute correlation within (bold diagonal) and between "
            r"feature families, native sets. qEEG populates six families of the taxonomy "
            r"and TCD four; magnitude, ratio and asymmetry are common to both and form "
            r"the matched core. These are the values plotted in Fig.~4 of the main text.}"
            "\n\\label{tab:S7}\n\\footnotesize\n"
            + parts[0] + "\n\n\\bigskip\n" + parts[1] + "\n\\end{table}\n")


def t_partial(ex: dict) -> str:
    pc = ex["partial_correlation"]
    body = []
    for mod in ("eeg", "tcd"):
        p = pc[mod]
        gens = ", ".join(TEX[g] for g in p["generators"])
        body.append(
            rf"{MOD_LABEL[mod]} & \parbox[t]{{0.25\linewidth}}{{\raggedright {gens}}} & "
            rf"{len(p['features'])} & {p['mean_abs_raw']:.3f} & {p['mean_abs_partial']:.3f} & "
            rf"{100 * p['frac_raw_ge_06']:.1f}\% & {100 * p['frac_partial_ge_06']:.1f}\% \\")
    raw = pc["tcd"]["mean_abs_raw"] - pc["eeg"]["mean_abs_raw"]
    par = pc["tcd"]["mean_abs_partial"] - pc["eeg"]["mean_abs_partial"]
    body.append(r"\midrule")
    body.append(rf"Gap (TCD $-$ qEEG) & & & {raw:+.3f} & {par:+.3f} & & \\")
    hdr = (r"Modality & Generating descriptors & $n$ & Raw $|r|$ & Partial $|r|$ & "
           r"Raw $\ge0.6$ & Partial $\ge0.6$")
    return table(
        r"Redundancy among the non-generating descriptors of the native sets before "
        r"and after partialling out the descriptors that generate them by closed-form "
        rf"construction. Partialling removes {100 * (1 - par / raw):.0f}\% of the "
        rf"between-modality gap and leaves {100 * par / raw:.0f}\% standing.",
        "tab:S8", "llccccc", hdr, body, size=r"\footnotesize")


def t_sweep(sw: pd.DataFrame) -> str:
    v = sw[sw.vif_threshold == 10.0]
    cols = [("eeg/native", "qEEG native"), ("tcd/native", "TCD native"),
            ("eeg/core", "qEEG core"), ("tcd/core", "TCD core")]
    body = []
    for ct in sorted(v.corr_threshold.unique()):
        row = v[v.corr_threshold == ct].set_index("set")
        vals = [row.loc[s, "retained_frac"] for s, _ in cols]
        gn, gc = vals[0] - vals[1], vals[2] - vals[3]
        mark = r"$^{\dagger}$" if abs(ct - 0.6) < 1e-9 else ""
        body.append(f"{ct:.2f}{mark} & " + " & ".join(f"{x:.3f}" for x in vals)
                    + f" & {gn:+.3f} & {gc:+.3f} \\\\")
    hdr = ("$|r|$ threshold & " + " & ".join(l for _, l in cols)
           + r" & Gap, native & Gap, core")
    return table(
        r"Fraction of candidate descriptors retained across the correlation-screening "
        r"grid at the VIF threshold of the main analysis ($10$). Gap is qEEG minus TCD; "
        r"it is positive in every row, and in every cell of the full $14\times4$ grid "
        r"(all 224 chains are in Supplementary Data S2). $^{\dagger}$ Threshold used in "
        r"the main analysis.",
        "tab:S9", "lcccccc", hdr, body, size=r"\footnotesize")


def t_selection(stab25: dict, sel50: pd.DataFrame, nog: dict) -> str:
    out = []
    for mod in ("eeg", "tcd"):
        s25 = stab25[mod].set_index("feature").selection_frequency
        g = sel50[sel50.modality == mod]
        body = []
        for name in feature_order(mod):
            v = [g[(g.selector == s) & (g.feature == name)].selection_frequency.iloc[0]
                 for s in ("reduced", "mrmr", "relieff", "lasso")]
            body.append(f"{TEX[name]} & {family(mod, name)} & {s25.get(name, np.nan):.2f} & "
                        + " & ".join(f"{x:.2f}" for x in v) + r" \\")
        body.append(r"\midrule")
        body.append(r"Nogueira index & & -- & " + " & ".join(
            f"{nog[f'{mod}/{s}']['stability']:.3f}"
            for s in ("reduced", "mrmr", "relieff", "lasso")) + r" \\")
        hdr = (r"Descriptor & Family & Reduced (25) & Reduced & mRMR & ReliefF & LASSO")
        out.append(table(
            rf"Per-descriptor selection frequency, {MOD_LABEL[mod]}. The third column is "
            r"the label-agnostic pipeline across the 25 folds of the main evaluation; the "
            r"remaining columns are the four selectors across the 50 subject-grouped "
            r"training partitions of the stability analysis, from which the Nogueira "
            r"index (Table~14 of the main text) is computed.",
            f"tab:S10{'a' if mod == 'eeg' else 'b'}", "llccccc", hdr, body,
            size=r"\footnotesize"))
    return "\n".join(out)


def t_anova(ex: dict) -> str:
    a, ph = ex["anova"], ex["anova_posthoc"]
    body = []
    for mod in ("eeg", "tcd"):
        s, fr = a[mod]["assumptions"], a[mod]["friedman"]
        body.append(
            rf"{MOD_LABEL[mod]} & {s['shapiro_W']:.3f} ({p_fmt(s['shapiro_p'])}) & "
            rf"{s['levene_W']:.2f} ({p_fmt(s['levene_p'])}) & {s['skew']:.2f} & "
            rf"{s['kurtosis']:.2f} & $\chi^2({fr['df']})={fr['chi2']:.2f}$ "
            rf"({p_fmt(fr['p'])}) & {fr['kendall_W']:.3f} \\")
    t1 = table(
        r"Assumption diagnostics for the repeated-measures ANOVA on fold-wise Macro-F1, "
        r"and the rank-based confirmation of the condition effect (Friedman test on "
        r"scores averaged over classifiers, 25 folds $\times$ 5 conditions). Residuals "
        r"depart from normality in both modalities and variances are heterogeneous on "
        r"qEEG, which is why the within-fold permutation test is the one relied on.",
        "tab:S11a", "lcccccc",
        r"Modality & Shapiro--Wilk $W$ ($p$) & Levene $W$ ($p$) & Skew & Kurtosis & "
        r"Friedman ($p$) & Kendall's $W$", body, size=r"\footnotesize")

    body = []
    for mod in ("eeg", "tcd"):
        mm = ph[mod]["marginal_means"]
        fvr = ph[mod]["full_vs_reduced"]
        lo = ph[mod]["leave_one_selector_out"]
        means = " & ".join(f"{mm[c]['mean']:.4f}" for c in CONDS)
        body.append(rf"{MOD_LABEL[mod]} & {means} & {fvr['F']:.2f} ({p_fmt(fvr['p_perm'])}) & "
                    + " & ".join(p_fmt(lo[s]["p_gg"]) for s in ("mrmr", "relieff", "lasso"))
                    + r" \\")
    t2 = table(
        r"Decomposition of the condition effect. Marginal Macro-F1 per condition "
        r"(averaged over classifiers and folds); the condition factor restricted to full "
        r"versus reduced, with its within-fold permutation $p$; and the "
        r"Greenhouse--Geisser $p$ of the omnibus condition effect after dropping one "
        r"supervised selector at a time. The omnibus effect disappears when mRMR is "
        r"dropped on TCD and when LASSO is dropped on qEEG.",
        "tab:S11b", "lcccccccccc",
        r"& \multicolumn{5}{c}{Marginal mean} & Full vs reduced & "
        r"\multicolumn{3}{c}{$p_{\mathrm{GG}}$ without} \\ \cmidrule(lr){2-6}\cmidrule(lr){8-10}"
        "\nModality & " + " & ".join(COND_LABEL[c] for c in CONDS)
        + r" & $F$ ($p_{\mathrm{perm}}$) & mRMR & ReliefF & LASSO",
        body, size=r"\footnotesize")
    return t1 + "\n" + t2


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def fig_equivalence(eq_c: pd.DataFrame, eq_u: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from ncx.report import CATEGORICAL, INK, MUTED, _style
    _style()
    colour = {"eeg": CATEGORICAL[0], "tcd": CATEGORICAL[1]}
    rows = [(m, model) for m in ("eeg", "tcd") for model in MODELS]
    ypos = {r: i + (0.8 if r[0] == "tcd" else 0) for i, r in enumerate(rows)}

    fig, axes = plt.subplots(1, 4, figsize=(17.4 / 2.54, 11.0 / 2.54),
                             sharey=True, constrained_layout=True)
    for ax, metric in zip(axes, METRICS):
        ax.axvspan(-0.05, 0.05, color="#E9E9E6", zorder=0, lw=0)
        ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
        for mod, model in rows:
            y = ypos[(mod, model)]
            u = eq_u[(eq_u.modality == mod) & (eq_u.model == model)
                     & (eq_u.metric == metric)].iloc[0]
            c = eq_c[(eq_c.modality == mod) & (eq_c.model == model)
                     & (eq_c.metric == metric)].iloc[0]
            ax.plot([c.ci90_lo, c.ci90_hi], [y, y], color=colour[mod], lw=2.0,
                    solid_capstyle="round", zorder=2)
            ax.plot([u.ci90_lo, u.ci90_hi], [y - 0.28, y - 0.28], color=MUTED,
                    lw=1.0, solid_capstyle="round", zorder=2)
            ax.plot(c.delta, y, "o", ms=4.2, zorder=3,
                    mfc=colour[mod] if c.equivalent else "white",
                    mec=colour[mod], mew=1.1)
        ax.set_title(METRIC_LABEL[metric], fontsize=7.5, color=INK, pad=4)
        ax.set_xlim(-0.13, 0.13)
        ax.set_xticks([-0.1, 0, 0.1])
        ax.set_xticklabels(["−0.1", "0", "0.1"], fontsize=6)
        ax.tick_params(axis="y", length=0)
        ax.grid(False)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    axes[0].set_yticks([ypos[r] for r in rows])
    axes[0].set_yticklabels([f"{MOD_LABEL[m]} · {mdl}" for m, mdl in rows], fontsize=6.5)
    axes[0].invert_yaxis()
    fig.supxlabel("Full − reduced (positive = loss from pruning)", fontsize=7, color=INK)
    handles = [
        Line2D([0], [0], color=CATEGORICAL[0], lw=2, label="qEEG, corrected 90% CI"),
        Line2D([0], [0], color=CATEGORICAL[1], lw=2, label="TCD, corrected 90% CI"),
        Line2D([0], [0], color=MUTED, lw=1, label="uncorrected 90% CI"),
        Line2D([0], [0], marker="o", ls="", mfc="#555", mec="#555", ms=4.2,
               label="equivalent (corrected, Holm)"),
        Line2D([0], [0], marker="o", ls="", mfc="white", mec="#555", ms=4.2,
               label="not equivalent"),
        plt.Rectangle((0, 0), 1, 1, fc="#E9E9E6", label="±0.05 margin"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, fontsize=6.3,
               frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_S1_equivalence.{ext}", dpi=600)
    plt.close(fig)


def fig_selection(sel50: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ncx.report import INK, SEQUENTIAL, _style
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(17.4 / 2.54, 9.6 / 2.54),
                             constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1]})
    sels = ["reduced", "mrmr", "relieff", "lasso"]
    im = None
    for ax, mod in zip(axes, ("eeg", "tcd")):
        names = feature_order(mod)
        g = sel50[sel50.modality == mod]
        M = np.array([[g[(g.selector == s) & (g.feature == n)].selection_frequency.iloc[0]
                       for s in sels] for n in names])
        im = ax.imshow(M, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(sels)))
        ax.set_xticklabels([COND_LABEL[s] for s in sels], fontsize=6.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([PLAIN[n] for n in names], fontsize=6.3)
        ax.set_title(MOD_LABEL[mod], fontsize=7.5, color=INK, pad=4)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(sels)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(names)), minor=True)
        ax.grid(False, which="major")
        ax.grid(True, which="minor", color="white", lw=1.2)
        ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02)
    cb.set_label("Selection frequency, 50 partitions", fontsize=6.5, color=INK)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_S2_selection_frequency.{ext}", dpi=600)
    plt.close(fig)


# --------------------------------------------------------------------------
# Workbook
# --------------------------------------------------------------------------

def workbook(src: Path, ex: dict, st: dict, out: Path) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    def frame_sheet(wb, name, df, caption):
        ws = wb.create_sheet(name)
        ws["A1"] = caption
        ws["A1"].font = Font(name="Arial", bold=True, size=11)
        ws["A1"].alignment = Alignment(wrap_text=False)
        cols = list(df.columns)
        for j, c in enumerate(cols, 1):
            cell = ws.cell(row=3, column=j, value=str(c))
            cell.font = Font(name="Arial", bold=True, size=10)
            cell.fill = PatternFill("solid", fgColor="E9EEF3")
        for i, row in enumerate(df.itertuples(index=False), 4):
            for j, v in enumerate(row, 1):
                if isinstance(v, (np.floating, float)):
                    v = None if not np.isfinite(v) else float(v)
                elif isinstance(v, (np.integer,)):
                    v = int(v)
                elif isinstance(v, (np.bool_,)):
                    v = bool(v)
                cell = ws.cell(row=i, column=j, value=v)
                cell.font = Font(name="Arial", size=10)
                if isinstance(v, float):
                    cell.number_format = "0.0000"
        for j, c in enumerate(cols, 1):
            width = max([len(str(c))] + [len(f"{x:.4f}") if isinstance(x, float)
                                         else len(str(x)) for x in df.iloc[:200, j - 1]])
            ws.column_dimensions[get_column_letter(j)].width = min(48, max(9, width + 2))
        ws.freeze_panes = "A4"
        return ws

    sheets = []

    def add(name, df, caption, source):
        sheets.append((name, df, caption, source))

    rd = lambda n: pd.read_csv(src / n)  # noqa: E731
    add("Per-fold metrics", rd("per_fold_metrics.csv"),
        "Every metric for every modality x condition x model x repeat x fold",
        "per_fold_metrics.csv")
    add("Summary metrics", rd("summary_metrics.csv"),
        "Mean and SD over the 25 folds, all conditions and metrics", "summary_metrics.csv")
    add("Equivalence corrected", rd("equivalence_tost.csv"),
        "TOST, full vs reduced, Nadeau-Bengio corrected (primary; Table S1)",
        "equivalence_tost.csv")
    add("Equivalence uncorrected", rd("equivalence_tost_uncorrected.csv"),
        "TOST, full vs reduced, without the fold-dependence correction (Table S2)",
        "equivalence_tost_uncorrected.csv")
    add("TOST power corrected", rd("tost_power.csv"),
        "Power at +/-0.05 and minimum margin at 80% power, corrected", "tost_power.csv")
    add("TOST power uncorrected", rd("tost_power_uncorrected.csv"),
        "Power at +/-0.05 and minimum margin at 80% power, uncorrected",
        "tost_power_uncorrected.csv")
    add("Paired comparisons", rd("statistical_comparisons.csv"),
        "Full vs reduced and reduced vs each selector, all metrics (Table S4)",
        "statistical_comparisons.csv")
    add("Chance-level tests", rd("vs_chance.csv"),
        "Each condition and model against analytic chance (Table S5)", "vs_chance.csv")
    add("Threshold sweep", rd("threshold_sweep.csv"),
        "All 224 reduction chains: 14 correlation x 4 VIF thresholds x 4 sets (Table S9)",
        "threshold_sweep.csv")
    s25 = pd.concat([rd("stability_eeg.csv").assign(modality="eeg"),
                     rd("stability_tcd.csv").assign(modality="tcd")])[
        ["modality", "feature", "selection_frequency"]]
    add("Selection freq 25 folds", s25,
        "Selection frequency, label-agnostic pipeline, 25 evaluation folds (Table S10)",
        "stability_eeg.csv, stability_tcd.csv")
    add("Selection freq 50 parts", rd("selector_stability.csv"),
        "Selection frequency, four selectors, 50 partitions (Table S10, Fig. S2)",
        "selector_stability.csv")
    nog = pd.DataFrame([{"modality": k.split("/")[0], "selector": k.split("/")[1], **v}
                        for k, v in ex["selector_stability"].items()])
    add("Stability index", nog, "Nogueira stability index per selector",
        "reviewer_extra.json: selector_stability")
    struct = pd.DataFrame([
        {"modality": m, "set": sp, **{k: v for k, v in st[m][sp].items()
                                      if not isinstance(v, (list, dict))}}
        for m in ("eeg", "tcd") for sp in ("native", "core")])
    add("Structure summary", struct,
        "Correlation, PCA and reduction-chain summary per modality and set",
        "structural.json")
    fb = []
    for m in ("eeg", "tcd"):
        b = _block(st, m)
        for a in b.index:
            for c in b.columns:
                fb.append({"modality": m, "family_a": a, "family_b": c,
                           "mean_abs_r": float(b.loc[a, c])})
    add("Family blocks", pd.DataFrame(fb),
        "Mean |r| within and between families (Table S7)", "structural.json")
    pc = pd.DataFrame([{"modality": m, "generators": ", ".join(v["generators"]),
                        "n_features": len(v["features"]),
                        **{k: v[k] for k in ("mean_abs_raw", "mean_abs_partial",
                                             "frac_raw_ge_06", "frac_partial_ge_06")}}
                       for m, v in ex["partial_correlation"].items()])
    add("Partial correlation", pc, "Redundancy after partialling out generators (Table S8)",
        "reviewer_extra.json: partial_correlation")
    an = []
    for m, v in ex["anova"].items():
        for eff, r in v["rm_anova"].items():
            an.append({"modality": m, "effect": {"A": "condition", "B": "classifier",
                                                  "A:B": "interaction"}[eff],
                       **{k: r[k] for k in ("F", "df", "df_error", "p", "gg_epsilon",
                                            "p_gg", "eta2_partial", "eta2_generalised")},
                       "p_perm": v["permutation"][eff]["p_permutation"]})
    add("ANOVA", pd.DataFrame(an),
        "Two-way repeated-measures ANOVA on fold-wise Macro-F1 (main Table 9)",
        "reviewer_extra.json: anova")
    dec = []
    for m, v in ex["anova_posthoc"].items():
        for c, mm in v["marginal_means"].items():
            dec.append({"modality": m, "quantity": f"marginal mean: {c}",
                        "value": mm["mean"], "sd": mm["std"]})
        for k, x in v["full_vs_reduced"].items():
            dec.append({"modality": m, "quantity": f"full vs reduced: {k}", "value": x})
        for s, x in v["leave_one_selector_out"].items():
            for k, y in x.items():
                dec.append({"modality": m, "quantity": f"without {s}: {k}", "value": y})
    add("ANOVA decomposition", pd.DataFrame(dec),
        "Marginal means, full-vs-reduced contrast, leave-one-selector-out (Table S11)",
        "reviewer_extra.json: anova_posthoc")
    perm = pd.DataFrame([{"modality": m, **v}
                         for m, v in ex["permutation_1000"].items()])
    add("Permutation nulls", perm, "Label-permutation null per feature space, 1000 shuffles",
        "reviewer_extra.json: permutation_1000")
    gap = pd.DataFrame([{"quantity": k, "value": v if not isinstance(v, list) else str(v)}
                        for k, v in ex["redundancy_gap_bootstrap"].items()]
                       + [{"quantity": f"column permutation: {k}", "value": v}
                          for k, v in ex["redundancy_gap_permutation"].items()])
    add("Gap bootstrap", gap, "Subject-level bootstrap and column permutation on the gap",
        "reviewer_extra.json")
    add("Channel pooling", rd("pooling_sensitivity.csv"),
        "Five channel-pooling operators on qEEG (main Table 8)", "pooling_sensitivity.csv")
    add("Within-subject", rd("within_subject.csv"),
        "Within-subject decoding per pooling variant (main Table 8)", "within_subject.csv")
    add("TCD calibration", rd("tcd_calibration.csv"),
        "TCD AUC, Brier score and ECE per model (main Table 12)", "tcd_calibration.csv")
    add("Hyperparameters", rd("hyperparameters.csv"),
        "Search spaces and fixed settings (main Table 3)", "hyperparameters.csv")

    wb = Workbook()
    ws = wb.active
    ws.title = "Read me"
    A = Font(name="Arial", size=10)
    ws["A1"] = "Supplementary Data S2"
    ws["A1"].font = Font(name="Arial", bold=True, size=14)
    ws["A2"] = ("Complete numeric results for: Two origins of feature redundancy in stroke "
                "biosignals: algebraic construction in transcranial Doppler, shared "
                "physiology in quantitative EEG.")
    ws["A2"].font = A
    notes = [
        "All values are aggregate statistics; no record-level or patient-level data are included.",
        "Primary inference uses the Nadeau-Bengio correction for overlapping cross-validation "
        "folds (n_test/n_train = 0.25). Uncorrected equivalence results are given for comparison.",
        "Metric names: macro_f1, balanced_accuracy, sensitivity_macro (identical to balanced "
        "accuracy by construction), specificity_macro, ppv_macro, npv_macro, auc_macro_ovr "
        "(blank where a model gives no calibrated probabilities).",
        "Each sheet names the file in the code repository's reference_results/ folder it is taken from.",
    ]
    for i, n in enumerate(notes, 4):
        ws.cell(row=i, column=1, value=n).font = A
    r0 = 4 + len(notes) + 1
    for j, h in enumerate(["Sheet", "Contents", "Source file"], 1):
        c = ws.cell(row=r0, column=j, value=h)
        c.font = Font(name="Arial", bold=True, size=10)
        c.fill = PatternFill("solid", fgColor="E9EEF3")
    for i, (name, df, cap, srcname) in enumerate(sheets, r0 + 1):
        for j, v in enumerate([name, cap, srcname], 1):
            ws.cell(row=i, column=j, value=v).font = A
        frame_sheet(wb, name, df, cap)

    # live self-checks: formulas, so a reader can confirm the headline counts
    rc = r0 + len(sheets) + 2
    ws.cell(row=rc, column=1, value="Self-checks (live formulas)").font = \
        Font(name="Arial", bold=True, size=10)
    eqc, equ = "'Equivalence corrected'", "'Equivalence uncorrected'"
    col_c = get_column_letter(list(pd.read_csv(src / "equivalence_tost.csv").columns)
                              .index("equivalent") + 1)
    col_u = get_column_letter(list(pd.read_csv(src / "equivalence_tost_uncorrected.csv")
                                   .columns).index("equivalent") + 1)
    mod_c = get_column_letter(list(pd.read_csv(src / "equivalence_tost.csv").columns)
                              .index("modality") + 1)
    checks = [
        ("Equivalent tests, corrected (of 40)", f"=COUNTIF({eqc}!{col_c}4:{col_c}43,TRUE)"),
        ("  of which qEEG", f"=COUNTIFS({eqc}!{mod_c}4:{mod_c}43,\"eeg\",{eqc}!{col_c}4:{col_c}43,TRUE)"),
        ("  of which TCD", f"=COUNTIFS({eqc}!{mod_c}4:{mod_c}43,\"tcd\",{eqc}!{col_c}4:{col_c}43,TRUE)"),
        ("Equivalent tests, uncorrected (of 40)", f"=COUNTIF({equ}!{col_u}4:{col_u}43,TRUE)"),
        ("Reduction chains in the sweep", "=COUNTA('Threshold sweep'!A4:A1000)"),
    ]
    for i, (lab, formula) in enumerate(checks, rc + 1):
        ws.cell(row=i, column=1, value=lab).font = A
        c = ws.cell(row=i, column=2, value=formula)
        c.font = A
        c.number_format = "0"
    ws.column_dimensions["A"].width = 38
    ws.column_dimensions["B"].width = 90
    ws.column_dimensions["C"].width = 36
    wb.save(out)


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, default=ROOT / "reference_results")
    ap.add_argument("--out", type=Path, default=ROOT / "supplement" / "build")
    a = ap.parse_args()
    src, out = a.src, a.out
    (out / "tables").mkdir(parents=True, exist_ok=True)

    ex = json.loads((src / "reviewer_extra.json").read_text())
    st = json.loads((src / "structural.json").read_text())
    eq_c = pd.read_csv(src / "equivalence_tost.csv")
    eq_u = pd.read_csv(src / "equivalence_tost_uncorrected.csv")
    summary = pd.read_csv(src / "summary_metrics.csv")
    comp = pd.read_csv(src / "statistical_comparisons.csv")
    vc = pd.read_csv(src / "vs_chance.csv")
    sw = pd.read_csv(src / "threshold_sweep.csv")
    sel50 = pd.read_csv(src / "selector_stability.csv")
    stab25 = {m: pd.read_csv(src / f"stability_{m}.csv") for m in ("eeg", "tcd")}

    T = out / "tables"
    write(T / "S1.tex", t_equivalence(
        eq_c, "tab:S1",
        r"Equivalence of the full and reduced feature spaces for all 40 model "
        r"$\times$ modality $\times$ metric comparisons, with Nadeau--Bengio corrected "
        r"standard errors (the primary analysis). $\Delta$ is full minus reduced, so "
        r"positive values are a loss from pruning. Margin $\pm0.05$, fixed before "
        r"analysis; $p$ values Holm-adjusted across all 40 tests. Balanced accuracy and "
        r"macro sensitivity are identical by construction, and for the two-class qEEG "
        r"problem macro specificity equals macro sensitivity as well, so the three "
        r"qEEG blocks repeat."))
    write(T / "S2.tex", t_equivalence(
        eq_u, "tab:S2",
        r"As Supplementary Table~S1, but with the standard error computed as if the 25 "
        r"cross-validation folds were independent. These are the results first "
        r"computed; they are reported for comparison only, because they ignore the "
        r"overlap between training sets that the corrected analysis accounts for."))
    write(T / "S3.tex", t_conditions(summary))
    write(T / "S4.tex", t_selectors(comp))
    write(T / "S5.tex", t_chance(vc))
    write(T / "S6.tex", t_chains(st))
    write(T / "S7.tex", t_blocks(st))
    write(T / "S8.tex", t_partial(ex))
    write(T / "S9.tex", t_sweep(sw))
    write(T / "S10.tex", t_selection(stab25, sel50, ex["selector_stability"]))
    write(T / "S11.tex", t_anova(ex))
    print(f"tables -> {T}")

    fig_equivalence(eq_c, eq_u, out)
    fig_selection(sel50, out)
    print(f"figures -> {out}")

    workbook(src, ex, st, out / "Supplementary_Data_S2.xlsx")
    print(f"workbook -> {out / 'Supplementary_Data_S2.xlsx'}")


if __name__ == "__main__":
    main()
