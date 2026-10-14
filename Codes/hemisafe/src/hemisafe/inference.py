"""Permutation inference for map correlations and cross-validated prediction scores."""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut


def perm_pvalue(observed, null, alternative="two-sided"):
    """Monte Carlo p-value with the +1 correction of Phipson & Smyth (2010).

    ``alternative``: "two-sided" (|null| >= |observed|, for statistics centred on 0 under the null such as
    correlations), "greater" or "less"."""
    null = np.asarray(null, float)
    if alternative == "two-sided":
        k = np.sum(np.abs(null) >= np.abs(observed) - 1e-12)
    elif alternative == "greater":
        k = np.sum(null >= observed - 1e-12)
    elif alternative == "less":
        k = np.sum(null <= observed + 1e-12)
    else:
        raise ValueError(alternative)
    return (1.0 + k) / (1.0 + null.size)


def corr_test(x, y, perms, alternative="two-sided"):
    """Correlation between two maps with a permutation/spin null applied to ``y`` (vectorised).

    ``perms``: permutation indices (integer array, n_perm x n) or surrogate maps of ``y`` (float array, n_perm x n,
    for example from ``symmetric_surrogates``). Returns (r, p, null_r)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xs = (x - x.mean()) / x.std()
    P = np.asarray(perms)
    Yp = P.astype(float) if np.issubdtype(P.dtype, np.floating) else y[P]
    Yp = (Yp - Yp.mean(1, keepdims=True)) / Yp.std(1, keepdims=True)
    null = Yp @ xs / len(x)
    r = float(np.mean(xs * (y - y.mean()) / y.std()))
    return r, perm_pvalue(r, null, alternative), null


def nested_cv_predict(estimator, X, y, splits, param_grid=None, inner_groups=None, inner_cv=None):
    """Out-of-fold predictions with optional nested tuning whose inner loop respects the same dependence structure
    as the outer loop. ``inner_groups`` (length n) are passed, restricted to each training set, to the inner splitter
    (default LeaveOneGroupOut): use ``pair_groups(hom)`` for homologue-grouped CV, or the outer fold labels for
    spatial-block CV, so that the tuning never sees a held-out region's homologue or block."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    preds = np.full(len(y), np.nan)
    for tr, te in splits:
        est = clone(estimator)
        if param_grid is not None:
            cv = inner_cv if inner_cv is not None else LeaveOneGroupOut()
            g = None if inner_groups is None else np.asarray(inner_groups)[tr]
            if g is None and isinstance(cv, LeaveOneGroupOut):
                g = np.arange(len(tr))
            est = GridSearchCV(est, param_grid, cv=cv, scoring="neg_mean_squared_error").fit(X[tr], y[tr], groups=g)
            est = est.best_estimator_
        else:
            est.fit(X[tr], y[tr])
        preds[te] = est.predict(X[te])
    return preds


def cv_permutation_test(estimator, X, y, cv, perms, score=r2_score, param_grid=None, inner_groups=None,
                        inner_cv=None):
    """Cross-validated score and its null distribution, re-running the complete CV procedure (including nested
    tuning, see nested_cv_predict) for every null draw.

    ``cv`` must be a partition (each region tested once), e.g. HomologueKFold or BilateralSpatialKFold.
    Returns (observed_score, p_value (greater), null_scores)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    splits = list(cv.split(X, y))
    tested = np.concatenate([te for _, te in splits])
    if len(tested) != len(y) or len(np.unique(tested)) != len(y):
        raise ValueError("cv must partition the regions (each region in exactly one test fold)")

    def run(target):
        return score(target, nested_cv_predict(estimator, X, target, splits, param_grid, inner_groups, inner_cv))

    obs = run(y)
    null = np.array([run(y[p]) for p in np.asarray(perms)])
    return obs, perm_pvalue(obs, null, "greater"), null
