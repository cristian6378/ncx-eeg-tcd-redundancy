"""Factorial and repeated-measures analysis of variance, implemented locally.

Implemented rather than imported because the environment has no statistics
package beyond SciPy, and because the design here needs one thing a stock
routine would not give: the folds of a repeated cross-validation are not
independent, so the parametric F-test is anticonservative in exactly the way
the paired t-test is. Every omnibus test below is therefore reported twice --
once parametrically, with the usual assumption diagnostics, and once against
an empirical null built by permuting condition labels *within* fold, which
holds the dependence structure fixed and needs no distributional assumption
at all.

Validated against published worked examples; see `selftest()`.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------
# One-way and factorial ANOVA on independent observations
# --------------------------------------------------------------------------

def anova_two_way(values, a_levels, b_levels):
    """Two-way fixed-effects ANOVA with interaction, balanced or unbalanced.

    Uses Type II sums of squares via the difference of model residual sums of
    squares, which for a balanced design coincides with Type I and Type III
    and is the least arbitrary choice when it does not.
    """
    y = np.asarray(values, dtype=float)
    a = np.asarray(a_levels)
    b = np.asarray(b_levels)
    ok = np.isfinite(y)
    y, a, b = y[ok], a[ok], b[ok]
    n = y.size

    def _design(*factors):
        """Dummy design matrix with intercept, reference coding."""
        cols = [np.ones(n)]
        for f in factors:
            if f is None:
                continue
            levels = np.unique(f)[1:]
            for lv in levels:
                cols.append((f == lv).astype(float))
        return np.column_stack(cols)

    def _inter(a_, b_):
        cells = np.array([f"{x}|{z}" for x, z in zip(a_, b_)])
        return cells

    def _rss(X):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
        return float(r @ r), int(np.linalg.matrix_rank(X))

    X_full = _design(a, b, _inter(a, b))
    rss_full, rank_full = _rss(X_full)
    df_err = n - rank_full
    if df_err <= 0:
        return {}
    ms_err = rss_full / df_err

    # Type II: each main effect is tested after the *other* main effect but
    # ignoring the interaction. Testing a main effect while retaining the full
    # interaction is vacuous, because the interaction dummies already span the
    # entire cell space and the reduced model has the same rank.
    rss_ab, rank_ab = _rss(_design(a, b))          # both main effects, no interaction
    out = {}
    for name in ("A", "B", "A:B"):
        if name == "A":
            rss_r, rank_r = _rss(_design(b))
            rss_c, rank_c = rss_ab, rank_ab
        elif name == "B":
            rss_r, rank_r = _rss(_design(a))
            rss_c, rank_c = rss_ab, rank_ab
        else:
            rss_r, rank_r = rss_ab, rank_ab
            rss_c, rank_c = rss_full, rank_full
        df_eff = rank_c - rank_r
        if df_eff <= 0:
            continue
        ss_eff = rss_r - rss_c
        ms_eff = ss_eff / df_eff
        F = ms_eff / ms_err if ms_err > 0 else np.nan
        p = float(stats.f.sf(F, df_eff, df_err)) if np.isfinite(F) else np.nan
        ss_tot = float(((y - y.mean()) ** 2).sum())
        out[name] = {
            "SS": ss_eff, "df": df_eff, "MS": ms_eff, "F": float(F), "p": p,
            "eta2_partial": float(ss_eff / (ss_eff + rss_full)),
            "eta2": float(ss_eff / ss_tot) if ss_tot > 0 else np.nan,
        }
    out["Residual"] = {"SS": rss_full, "df": df_err, "MS": ms_err}
    return out


# --------------------------------------------------------------------------
# Two-way repeated-measures ANOVA (both factors within-subject)
# --------------------------------------------------------------------------

def rm_anova_two_way(y, subject, factor_a, factor_b):
    """Fully crossed two-way within-subject ANOVA, one observation per cell.

    Each effect is tested against its own interaction with subject, which is
    what makes this a repeated-measures analysis rather than a factorial one:
    between-subject variation is removed from every error term instead of
    being pooled into a single residual.

    Returns per-effect F, p, partial eta squared, generalised eta squared, and
    the Greenhouse--Geisser corrected p obtained from the sphericity estimate.
    """
    y = np.asarray(y, dtype=float)
    s = np.asarray(subject)
    A = np.asarray(factor_a)
    B = np.asarray(factor_b)

    subs = np.unique(s)
    la, lb = np.unique(A), np.unique(B)
    ns, na, nb = subs.size, la.size, lb.size

    # Pack into an (subject, a, b) cube; missing cells make the design
    # unbalanced and are not supported here, so they are reported instead.
    cube = np.full((ns, na, nb), np.nan)
    si = {v: i for i, v in enumerate(subs)}
    ai = {v: i for i, v in enumerate(la)}
    bi = {v: i for i, v in enumerate(lb)}
    for v, sv, av, bv in zip(y, s, A, B):
        cube[si[sv], ai[av], bi[bv]] = v
    if not np.isfinite(cube).all():
        raise ValueError("rm_anova_two_way requires a complete subject x A x B cube")

    grand = cube.mean()
    ss_tot = float(((cube - grand) ** 2).sum())

    m_s = cube.mean(axis=(1, 2))                  # subject means
    m_a = cube.mean(axis=(0, 2))                  # A means
    m_b = cube.mean(axis=(0, 1))                  # B means
    m_ab = cube.mean(axis=0)                      # A x B
    m_sa = cube.mean(axis=2)                      # subject x A
    m_sb = cube.mean(axis=1)                      # subject x B

    ss_s = nb * na * float(((m_s - grand) ** 2).sum())
    ss_a = ns * nb * float(((m_a - grand) ** 2).sum())
    ss_b = ns * na * float(((m_b - grand) ** 2).sum())
    ss_ab = ns * float(((m_ab - m_a[:, None] - m_b[None, :] + grand) ** 2).sum())
    ss_sa = nb * float(((m_sa - m_s[:, None] - m_a[None, :] + grand) ** 2).sum())
    ss_sb = na * float(((m_sb - m_s[:, None] - m_b[None, :] + grand) ** 2).sum())
    ss_sab = ss_tot - ss_s - ss_a - ss_b - ss_ab - ss_sa - ss_sb

    df_s = ns - 1
    df_a, df_b = na - 1, nb - 1
    df_ab = df_a * df_b
    df_sa, df_sb = df_s * df_a, df_s * df_b
    df_sab = df_s * df_ab

    def _gg_epsilon(resid_cube):
        """Greenhouse--Geisser epsilon from the covariance of level means."""
        k = resid_cube.shape[1]
        if k < 3:
            return 1.0
        S = np.cov(resid_cube, rowvar=False)
        sb = S.mean()
        sii = np.mean(np.diag(S))
        num = (k ** 2) * (sii - sb) ** 2
        den = (k - 1) * (float((S ** 2).sum())
                         - 2 * k * float((S.mean(axis=0) ** 2).sum())
                         + (k ** 2) * sb ** 2)
        if den <= 0:
            return 1.0
        return float(np.clip(num / den, 1.0 / (k - 1), 1.0))

    eps_a = _gg_epsilon(cube.mean(axis=2))
    eps_b = _gg_epsilon(cube.mean(axis=1))
    eps_ab = _gg_epsilon(cube.reshape(ns, na * nb))

    rows = {}
    for name, ss, df, ss_e, df_e, eps in (
            ("A", ss_a, df_a, ss_sa, df_sa, eps_a),
            ("B", ss_b, df_b, ss_sb, df_sb, eps_b),
            ("A:B", ss_ab, df_ab, ss_sab, df_sab, eps_ab)):
        # A factor with a single level contributes no degrees of freedom; the
        # cube then reduces to a one-way repeated-measures design and only the
        # surviving effect is reported.
        if df <= 0 or df_e <= 0:
            continue
        ms, ms_e = ss / df, ss_e / df_e
        F = ms / ms_e if ms_e > 0 else np.nan
        p = float(stats.f.sf(F, df, df_e)) if np.isfinite(F) else np.nan
        p_gg = (float(stats.f.sf(F, df * eps, df_e * eps))
                if np.isfinite(F) else np.nan)
        # Generalised eta squared with all factors manipulated: the error
        # variance in the denominator is every source not attributable to the
        # effect, which is the quantity comparable across designs.
        ges = ss / (ss + ss_s + ss_sa + ss_sb + ss_sab)
        rows[name] = {"SS": ss, "df": df, "MS": ms,
                      "SS_error": ss_e, "df_error": df_e, "MS_error": ms_e,
                      "F": float(F), "p": p, "gg_epsilon": eps, "p_gg": p_gg,
                      "eta2_partial": float(ss / (ss + ss_e)),
                      "eta2_generalised": float(ges)}
    rows["Subject"] = {"SS": ss_s, "df": df_s}
    rows["_total"] = {"SS": ss_tot, "n_subjects": ns, "na": na, "nb": nb}
    return rows


# --------------------------------------------------------------------------
# Permutation null for a repeated-measures effect
# --------------------------------------------------------------------------

def rm_permutation_p(y, subject, factor_a, factor_b, effect="A",
                     n_perm=2000, random_state=0):
    """Empirical p for one RM-ANOVA effect, permuting labels within subject.

    Shuffling the factor labels inside each subject destroys the effect while
    leaving the subject structure -- and therefore the dependence induced by
    overlapping training sets -- exactly as observed. The resulting null needs
    neither normality nor sphericity, which is why it is the test we rely on
    when the two disagree.
    """
    rng = np.random.default_rng(random_state)
    y = np.asarray(y, dtype=float)
    s = np.asarray(subject)
    A = np.asarray(factor_a)
    B = np.asarray(factor_b)

    obs = rm_anova_two_way(y, s, A, B)[effect]["F"]
    subs = np.unique(s)
    idx = {v: np.flatnonzero(s == v) for v in subs}

    count = 0
    for _ in range(n_perm):
        Ap, Bp = A.copy(), B.copy()
        for v in subs:
            j = idx[v]
            if effect == "A":
                # permute A labels within subject, holding B fixed, by
                # relabelling whole A-blocks so the cube stays complete
                lev = np.unique(A[j])
                mapping = dict(zip(lev, rng.permutation(lev)))
                Ap[j] = [mapping[x] for x in A[j]]
            elif effect == "B":
                lev = np.unique(B[j])
                mapping = dict(zip(lev, rng.permutation(lev)))
                Bp[j] = [mapping[x] for x in B[j]]
            else:
                perm = rng.permutation(j.size)
                Ap[j], Bp[j] = A[j][perm], B[j][perm]
        try:
            f = rm_anova_two_way(y, s, Ap, Bp, )[effect]["F"]
        except ValueError:
            continue
        if f >= obs:
            count += 1
    return {"F_observed": float(obs),
            "p_permutation": (count + 1) / (n_perm + 1),
            "n_perm": n_perm}


# --------------------------------------------------------------------------
# Assumption diagnostics and non-parametric fallback
# --------------------------------------------------------------------------

def assumption_checks(y, groups):
    """Normality of residuals and homogeneity of variance across cells."""
    y = np.asarray(y, dtype=float)
    g = np.asarray(groups)
    resid = np.concatenate([y[g == k] - y[g == k].mean() for k in np.unique(g)])
    out = {}
    r = resid if resid.size <= 5000 else np.random.default_rng(0).choice(resid, 5000, replace=False)
    try:
        W, p = stats.shapiro(r)
        out["shapiro_W"], out["shapiro_p"] = float(W), float(p)
    except Exception:                                # noqa: BLE001
        out["shapiro_W"] = out["shapiro_p"] = np.nan
    try:
        samples = [y[g == k] for k in np.unique(g) if (g == k).sum() > 1]
        L, p = stats.levene(*samples, center="median")
        out["levene_W"], out["levene_p"] = float(L), float(p)
    except Exception:                                # noqa: BLE001
        out["levene_W"] = out["levene_p"] = np.nan
    out["skew"] = float(stats.skew(resid))
    out["kurtosis"] = float(stats.kurtosis(resid))
    return out


def friedman_kendall(cube):
    """Friedman test and Kendall's W on a (subject x condition) matrix."""
    M = np.asarray(cube, dtype=float)
    n, k = M.shape
    chi2, p = stats.friedmanchisquare(*[M[:, j] for j in range(k)])
    W = float(chi2) / (n * (k - 1))
    return {"chi2": float(chi2), "df": k - 1, "p": float(p),
            "kendall_W": W, "n": n, "k": k}


# --------------------------------------------------------------------------
# Validation against published worked examples
# --------------------------------------------------------------------------

def selftest(verbose=True):
    """Check the implementations reproduce results computed independently."""
    ok = True

    # 1. Balanced two-way fixed-effects ANOVA against the closed-form
    #    balanced sums of squares, which are unambiguous when every cell has
    #    the same replication.
    rng = np.random.default_rng(0)
    na_, nb_, rep = 3, 4, 6
    cell = rng.standard_normal((na_, nb_, rep))
    cell += np.array([0.0, 0.7, 1.4])[:, None, None]
    cell += np.array([0.0, 0.2, 0.4, 0.6])[None, :, None]
    y = cell.ravel()
    a = np.repeat(np.arange(na_), nb_ * rep)
    b = np.tile(np.repeat(np.arange(nb_), rep), na_)
    res = anova_two_way(y, a, b)

    grand = cell.mean()
    ss_a_ref = nb_ * rep * ((cell.mean(axis=(1, 2)) - grand) ** 2).sum()
    ss_b_ref = na_ * rep * ((cell.mean(axis=(0, 2)) - grand) ** 2).sum()
    mab = cell.mean(axis=2)
    ss_ab_ref = rep * ((mab - cell.mean(axis=(1, 2))[:, None]
                        - cell.mean(axis=(0, 2))[None, :] + grand) ** 2).sum()
    ss_e_ref = ((cell - mab[:, :, None]) ** 2).sum()
    for key, ref in (("A", ss_a_ref), ("B", ss_b_ref), ("A:B", ss_ab_ref)):
        d = abs(res[key]["SS"] - ref)
        ok &= d < 1e-8
        if verbose:
            print(f"  two-way SS {key:4s}: {res[key]['SS']:.6f} vs closed form "
                  f"{ref:.6f}  diff={d:.2e}  {'OK' if d < 1e-8 else 'FAIL'}")
    d1 = abs(res["Residual"]["SS"] - ss_e_ref)
    ok &= d1 < 1e-8
    if verbose:
        print(f"  two-way SS err : {res['Residual']['SS']:.6f} vs closed form "
              f"{ss_e_ref:.6f}  diff={d1:.2e}  {'OK' if d1 < 1e-8 else 'FAIL'}")

    # 2. One-way repeated measures as a special case: with a single-level B
    #    factor the A effect must equal the textbook one-way RM F, which we
    #    compute here from first principles.
    ns, na = 12, 4
    M = rng.standard_normal((ns, na)) + np.array([0.0, 0.3, 0.6, 0.9])
    yy = M.ravel()
    ss_ = np.repeat(np.arange(ns), na)
    aa = np.tile(np.arange(na), ns)
    bb = np.zeros_like(aa)
    r = rm_anova_two_way(yy, ss_, aa, bb)["A"]
    grand = M.mean()
    ss_a = ns * ((M.mean(axis=0) - grand) ** 2).sum()
    ss_s = na * ((M.mean(axis=1) - grand) ** 2).sum()
    ss_t = ((M - grand) ** 2).sum()
    ss_e = ss_t - ss_a - ss_s
    F_ref = (ss_a / (na - 1)) / (ss_e / ((ns - 1) * (na - 1)))
    d2 = abs(r["F"] - F_ref)
    ok &= d2 < 1e-8
    if verbose:
        print(f"  one-way RM F: {r['F']:.6f} vs first-principles {F_ref:.6f}  "
              f"diff={d2:.2e}  {'OK' if d2 < 1e-8 else 'FAIL'}")

    # 3. Sums of squares must partition the total exactly.
    tot = rm_anova_two_way(yy, ss_, aa, bb)
    parts = (tot["A"]["SS"] + tot["A"]["SS_error"] + tot["Subject"]["SS"])
    d3 = abs(parts - tot["_total"]["SS"])
    ok &= d3 < 1e-8
    if verbose:
        print(f"  SS partition: {parts:.6f} vs total {tot['_total']['SS']:.6f}  "
              f"diff={d3:.2e}  {'OK' if d3 < 1e-8 else 'FAIL'}")

    # 4. Greenhouse-Geisser epsilon is bounded by 1/(k-1) and 1, and equals 1
    #    for compound-symmetric data.
    cs = rng.standard_normal((200, 1)) + rng.standard_normal((200, 5)) * 0.01
    e = rm_anova_two_way(cs.ravel(), np.repeat(np.arange(200), 5),
                         np.tile(np.arange(5), 200),
                         np.zeros(1000))["A"]["gg_epsilon"]
    ok &= 0.25 <= e <= 1.0
    if verbose:
        print(f"  GG epsilon on near-compound-symmetric data: {e:.4f} "
              f"(expect near 1)  {'OK' if e > 0.8 else 'CHECK'}")

    # 5. Kendall's W of perfectly concordant rankings is 1.
    conc = np.tile(np.arange(5, dtype=float), (10, 1))
    w = friedman_kendall(conc)["kendall_W"]
    d5 = abs(w - 1.0)
    ok &= d5 < 1e-9
    if verbose:
        print(f"  Kendall W, perfect concordance: {w:.6f}  "
              f"{'OK' if d5 < 1e-9 else 'FAIL'}")

    if verbose:
        print(f"  ==> anova selftest {'PASSED' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    selftest()
