"""Model evaluation with subject-level grouping and per-fold clinical metrics.

Two things here are non-negotiable for a clinical signal-processing paper and
are therefore built into the loop rather than left to a post-hoc script.
First, grouping is by subject, so no trial from a patient appears on both
sides of a split. Second, sensitivity, specificity, PPV and NPV are computed
and stored *per fold*, not reconstructed afterwards from aggregate scores --
predictive values in particular cannot be recovered from macro summaries,
because they depend on the per-class distribution of false positives.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import Config
from .redundancy import RedundancyReducer
from .selectors import make_selector

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:                               # noqa: BLE001
    HAS_LGBM = False


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def clinical_metrics(y_true, y_pred, labels=None) -> dict:
    """Macro-averaged one-vs-rest sensitivity, specificity, PPV and NPV.

    Each class is collapsed to a binary problem against the rest, the four
    operating characteristics are computed per class, and the macro average is
    the unweighted mean across classes. Classes absent from a fold contribute
    NaN and are skipped by the nan-aware mean rather than counted as zero.
    """
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)])) \
        if labels is None else np.asarray(labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()
    sens, spec, ppv, npv = [], [], [], []
    per_class = {}
    for i, lab in enumerate(labels):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        se = tp / (tp + fn) if (tp + fn) else np.nan
        sp = tn / (tn + fp) if (tn + fp) else np.nan
        pv = tp / (tp + fp) if (tp + fp) else np.nan
        nv = tn / (tn + fn) if (tn + fn) else np.nan
        sens.append(se); spec.append(sp); ppv.append(pv); npv.append(nv)
        per_class[str(lab)] = {"sensitivity": se, "specificity": sp,
                               "ppv": pv, "npv": nv, "support": int(tp + fn)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = {"sensitivity_macro": float(np.nanmean(sens)),
               "specificity_macro": float(np.nanmean(spec)),
               "ppv_macro": float(np.nanmean(ppv)),
               "npv_macro": float(np.nanmean(npv))}
    out["per_class"] = per_class
    return out


def _auc(y_true, scores, classes):
    """Macro one-vs-rest AUC from probabilities or decision values.

    Fitting Platt scaling inside every SVM (probability=True) costs an extra
    internal 5-fold refit per candidate configuration and dominates run time
    on this protocol. Decision values rank identically to calibrated
    probabilities, and AUC depends only on the ranking, so the cheaper route
    gives the same number.
    """
    try:
        if scores is None:
            return np.nan
        scores = np.asarray(scores)
        if len(classes) == 2:
            v = scores[:, 1] if scores.ndim == 2 and scores.shape[1] == 2 else scores.ravel()
            return float(roc_auc_score(y_true, v))
        return float(roc_auc_score(y_true, scores, multi_class="ovr",
                                   average="macro", labels=classes))
    except Exception:                           # noqa: BLE001
        return np.nan


def _scores(model, X):
    """Probabilities if cheap, else decision values, else None."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except Exception:                       # noqa: BLE001
            pass
    if hasattr(model, "decision_function"):
        try:
            d = model.decision_function(X)
            return d if np.ndim(d) > 1 else np.asarray(d).reshape(-1, 1)
        except Exception:                       # noqa: BLE001
            pass
    return None


# --------------------------------------------------------------------------
# Model zoo
# --------------------------------------------------------------------------

def build_models(cfg: Config) -> dict:
    rs = cfg.random_state
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=5000, random_state=rs),
            {"clf__C": loguniform(1e-4, 1e2)}),
        "SVM (Linear)": (
            SVC(kernel="linear", random_state=rs),
            {"clf__C": loguniform(1e-4, 1e2)}),
        "SVM (RBF)": (
            SVC(kernel="rbf", random_state=rs),
            {"clf__C": loguniform(1e-2, 1e3),
             "clf__gamma": loguniform(1e-4, 1e0)}),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=rs),
            {"clf__max_depth": randint(2, 12),
             "clf__min_samples_leaf": randint(2, 30),
             "clf__ccp_alpha": uniform(0.0, 0.03)}),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, random_state=rs, n_jobs=1),
            {"clf__max_depth": randint(2, 16),
             "clf__min_samples_leaf": randint(1, 12),
             "clf__max_features": ["sqrt", "log2", None]}),
    }
    if HAS_LGBM:
        models["LightGBM"] = (
            LGBMClassifier(n_estimators=300, random_state=rs, n_jobs=1, verbose=-1),
            {"clf__max_depth": randint(2, 10),
             "clf__num_leaves": randint(8, 64),
             "clf__learning_rate": loguniform(1e-2, 3e-1),
             "clf__reg_alpha": uniform(0.0, 1.0)})
    return models


_MEMORY = None


def _pipeline_memory(cfg):
    """A joblib cache shared by every pipeline in a run.

    The selection step depends only on the training fold, never on the
    classifier's hyperparameters, yet a naive pipeline refits it for every
    candidate configuration -- for the L1 selector that is thousands of
    identical fits. Caching the transformer collapses them to one per fold
    and changes nothing about the result, because every selector here is
    deterministic given its input.
    """
    global _MEMORY
    if _MEMORY is None:
        import tempfile
        from joblib import Memory
        _MEMORY = Memory(location=tempfile.mkdtemp(prefix="ncx_cache_"),
                         verbose=0)
    return _MEMORY


def _make_pipeline(estimator, feature_names, cfg, mode: str, k: int | None):
    steps = [("scale", StandardScaler())]
    if mode == "reduced":
        steps.append(("sel", RedundancyReducer(cfg.corr_threshold,
                                               cfg.vif_threshold,
                                               feature_names)))
    elif mode in {"mrmr", "relieff", "lasso"}:
        steps.append(("sel", make_selector(mode, k or 9, cfg.random_state)))
    steps.append(("clf", estimator))
    if mode in {"reduced", "mrmr", "relieff", "lasso"}:
        return Pipeline(steps, memory=_pipeline_memory(cfg))
    return Pipeline(steps)


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def run_condition(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  feature_names: list[str], cfg: Config, modality: str,
                  mode: str = "full", k: int | None = None,
                  models: dict | None = None) -> pd.DataFrame:
    """Repeated stratified group CV with an inner tuning loop.

    `mode` selects the feature-space condition: 'full' (all candidates),
    'reduced' (the unsupervised two-stage pipeline), or one of the supervised
    baselines. Every mode is fitted strictly within the training partition.
    """
    X = np.nan_to_num(np.asarray(X, dtype=float))
    y = np.asarray(y)
    groups = np.asarray(groups)
    classes = np.unique(y)
    models = models or build_models(cfg)
    rows = []
    n_attempted, first_error = 0, None

    for rep in range(cfg.n_repeats):
        cv = StratifiedGroupKFold(n_splits=cfg.n_splits, shuffle=True,
                                  random_state=cfg.random_state + rep)
        for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
            if len(np.unique(y[tr])) < 2:
                continue
            inner = StratifiedGroupKFold(n_splits=cfg.inner_splits, shuffle=True,
                                         random_state=cfg.random_state + rep)
            for mname, (est, grid) in models.items():
                pipe = _make_pipeline(est, feature_names, cfg, mode, k)
                n_attempted += 1
                try:
                    search = RandomizedSearchCV(
                        pipe, grid, n_iter=cfg.n_search_iter, scoring=cfg.scoring,
                        # A list, not the generator split() returns: the
                        # search may ship `cv` to worker processes, and a
                        # generator cannot be pickled. scikit-learn >= 1.9
                        # does exactly that, and every fit then fails.
                        cv=list(inner.split(X[tr], y[tr], groups[tr])),
                        random_state=cfg.random_state,
                        n_jobs=getattr(cfg, "search_jobs", cfg.n_jobs),
                        refit=True, error_score=np.nan)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        search.fit(X[tr], y[tr])
                    best = search.best_estimator_
                    yp = best.predict(X[te])
                    proba = _scores(best, X[te])
                except Exception as e:          # noqa: BLE001
                    first_error = first_error or e
                    warnings.warn(f"{mname}/{mode} rep{rep} fold{fold}: {e}")
                    continue

                cm = clinical_metrics(y[te], yp, labels=classes)
                sel = best.named_steps.get("sel")
                n_kept = (len(getattr(sel, "support_", [])) if sel is not None
                          else X.shape[1])
                rows.append({
                    "modality": modality, "condition": mode, "model": mname,
                    "repeat": rep, "fold": fold,
                    "macro_f1": float(f1_score(y[te], yp, average="macro",
                                               zero_division=0)),
                    "balanced_accuracy": float(balanced_accuracy_score(y[te], yp)),
                    "auc_macro_ovr": _auc(y[te], proba, classes),
                    "sensitivity_macro": cm["sensitivity_macro"],
                    "specificity_macro": cm["specificity_macro"],
                    "ppv_macro": cm["ppv_macro"],
                    "npv_macro": cm["npv_macro"],
                    "n_features": int(n_kept),
                    "n_test": int(len(te)),
                })
    # An occasional failed fold is tolerated and reported above. A condition
    # in which nothing at all could be fitted is a broken environment or a
    # broken pipeline, and returning an empty table would let it pass silently.
    if n_attempted and not rows:
        raise RuntimeError(
            f"all {n_attempted} fits failed for mode={mode!r}; "
            f"first error: {first_error!r}") from first_error
    return pd.DataFrame(rows)


def selection_stability(X, y, groups, feature_names, cfg, mode="reduced",
                        k: int | None = None) -> pd.DataFrame:
    """How often each feature survives selection across folds.

    Stability is reported because a reduction that keeps a different nine
    features every fold is not a compact representation in any useful sense,
    even if its mean accuracy looks acceptable.
    """
    X = np.nan_to_num(np.asarray(X, dtype=float))
    y, groups = np.asarray(y), np.asarray(groups)
    counts = {n: 0 for n in feature_names}
    n_fits = 0
    for rep in range(cfg.n_repeats):
        cv = StratifiedGroupKFold(n_splits=cfg.n_splits, shuffle=True,
                                  random_state=cfg.random_state + rep)
        for tr, _ in cv.split(X, y, groups):
            Xs = StandardScaler().fit_transform(X[tr])
            sel = (RedundancyReducer(cfg.corr_threshold, cfg.vif_threshold,
                                     feature_names)
                   if mode == "reduced" else make_selector(mode, k or 9,
                                                           cfg.random_state))
            try:
                sel.fit(Xs, y[tr])
            except Exception:                   # noqa: BLE001
                continue
            for i in sel.support_:
                counts[feature_names[i]] += 1
            n_fits += 1
    return (pd.DataFrame({"feature": list(counts),
                          "selection_frequency": [c / max(n_fits, 1)
                                                  for c in counts.values()]})
            .sort_values("selection_frequency", ascending=False)
            .reset_index(drop=True))
