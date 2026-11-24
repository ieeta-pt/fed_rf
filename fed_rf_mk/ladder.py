"""Disclosure ladder for federated Random Forest and XGBoost models.

Notation. K sites, site k holds n_k patients of whom n_k^+ have the outcome,
pi_k = n_k^+ / n_k, pooled prevalence pi = sum n_k^+ / sum n_k,
logit p = log(p / (1 - p)), odds(p) = p / (1 - p), sigma the logistic function.

* Level 0. The source site releases its model, used as it is.
* Level 1. Sites also release n_k and n_k^+, and the source model is shifted
  to the pooled prevalence (``shift_prob``, ``shift_logit``).
* Level 2. The deployment site recalibrates the model on its own labeled
  patients (``fit_recalibration``, ``apply_recalibration``).
* Level 3. Every site releases its model and the K models are combined
  (``allocate_trees``, ``select_trees``, ``rf_federated*``, ``xgb_federated*``).

Only numpy, scipy, scikit-learn and xgboost are used, so this module can be
imported without PySyft. Every function is pure: random choices take a
``numpy.random.Generator``.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.special import expit, logit as _logit

__all__ = [
    "logit", "sigmoid", "odds", "pooled_prevalence", "site_weights",
    "shift_prob", "shift_logit", "clip_prob", "fit_recalibration",
    "apply_recalibration", "allocate_trees", "select_trees", "rf_federated",
    "rf_federated_shift", "rf_federated_global_shift", "xgb_federated",
    "xgb_federated_shift", "xgb_federated_global_shift", "tree_oob_scores",
    "single_record_leaf_fraction", "xgb_single_record_leaf_fraction",
    "share", "citl_intercept", "calibration_slope",
]


# ----------------------------------------------------------------------------
# Basic transforms
# ----------------------------------------------------------------------------

def logit(p):
    return _logit(np.asarray(p, dtype=np.float64))


def sigmoid(z):
    return expit(np.asarray(z, dtype=np.float64))


def odds(p):
    p = np.asarray(p, dtype=np.float64)
    return p / (1.0 - p)


def _check_prevalence(p, name):
    p = np.asarray(p, dtype=np.float64)
    if np.any(p <= 0.0) or np.any(p >= 1.0):
        raise ValueError(f"{name} must lie in (0, 1), got {p}")
    return p


# ----------------------------------------------------------------------------
# Counts and site weights
# ----------------------------------------------------------------------------

def pooled_prevalence(n, n_pos):
    """pi = sum_k n_k^+ / sum_k n_k."""
    n = np.asarray(n, dtype=np.float64)
    n_pos = np.asarray(n_pos, dtype=np.float64)
    if n.shape != n_pos.shape or n.sum() <= 0:
        raise ValueError("n and n_pos must have the same shape and sum(n) > 0")
    return float(n_pos.sum() / n.sum())


def site_weights(n, n_pos, kind):
    """Site weights: ``equal`` 1/K, ``size`` n_k / sum n_j,
    ``positives`` n_k^+ / sum n_j^+."""
    n = np.asarray(n, dtype=np.float64)
    n_pos = np.asarray(n_pos, dtype=np.float64)
    K = n.size
    if kind == "equal":
        return np.full(K, 1.0 / K)
    if kind == "size":
        return n / n.sum()
    if kind == "positives":
        if n_pos.sum() <= 0:
            raise ValueError("positives weights need at least one positive record")
        return n_pos / n_pos.sum()
    raise ValueError(f"unknown weight kind {kind!r}")


# ----------------------------------------------------------------------------
# Level 1, prior shift
# ----------------------------------------------------------------------------

def shift_prob(p, pi, pi_k):
    """Level 1 correction of probabilities.

    p_tilde = c p / ((1 - p) + c p) with c = odds(pi) / odds(pi_k). On (0, 1)
    this equals logit p_tilde = logit p + logit pi - logit pi_k, and it keeps
    0 and 1 unchanged, so no clipping is needed.
    """
    p = np.asarray(p, dtype=np.float64)
    pi = _check_prevalence(pi, "pi")
    pi_k = _check_prevalence(pi_k, "pi_k")
    c = odds(pi) / odds(pi_k)
    return c * p / ((1.0 - p) + c * p)


def shift_logit(z, pi, pi_k):
    """Level 1 correction of log odds: z + logit pi - logit pi_k."""
    pi = _check_prevalence(pi, "pi")
    pi_k = _check_prevalence(pi_k, "pi_k")
    return np.asarray(z, dtype=np.float64) + logit(pi) - logit(pi_k)


def clip_prob(p, T):
    """Clip Random Forest probabilities to [1/(2T), 1 - 1/(2T)], T trees."""
    if T < 1:
        raise ValueError("T must be at least 1")
    eps = 1.0 / (2.0 * T)
    return np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)


# ----------------------------------------------------------------------------
# Level 2, recalibration at the deployment site
# ----------------------------------------------------------------------------

def _unpenalized_logistic(z, y):
    from sklearn.linear_model import LogisticRegression
    z = np.asarray(z, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(y).astype(int).ravel()
    if np.unique(y).size < 2:
        raise ValueError("recalibration needs both classes")
    lr = LogisticRegression(C=1e6, max_iter=10000)
    lr.fit(z, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def fit_recalibration(z_cal, y_cal):
    """(a, b) of a logistic regression of y on z with essentially no penalty."""
    return _unpenalized_logistic(z_cal, y_cal)


def apply_recalibration(z, a, b):
    """sigma(a z + b)."""
    return sigmoid(a * np.asarray(z, dtype=np.float64) + b)


# ----------------------------------------------------------------------------
# Level 3, Random Forest
# ----------------------------------------------------------------------------

def allocate_trees(w, T_k):
    """Largest remainder allocation of trees to sites.

    T = round(sum_k w_k T_k). Site k first gets floor(w_k T_k) trees, capped at
    T_k. The remaining trees go one each to the sites with the largest
    remainders w_k T_k - floor(w_k T_k), ties to the lower site index, never
    above T_k.
    """
    w = np.asarray(w, dtype=np.float64)
    T_k = np.asarray(T_k, dtype=np.int64)
    if w.shape != T_k.shape:
        raise ValueError("w and T_k must have the same shape")
    if np.any(w < 0) or np.any(T_k < 0):
        raise ValueError("weights and tree counts must be non negative")
    exact = w * T_k
    T = int(round(float(exact.sum())))
    T = min(T, int(T_k.sum()))
    base = np.minimum(np.floor(exact + 1e-9).astype(np.int64), T_k)
    remainder = exact - base
    order = np.argsort(-remainder, kind="stable")
    t = base.copy()
    left = T - int(t.sum())
    while left > 0:
        progressed = False
        for k in order:
            if left == 0:
                break
            if t[k] < T_k[k]:
                t[k] += 1
                left -= 1
                progressed = True
        if not progressed:
            break
    return t


def select_trees(T_k, t_k, rng, scores=None):
    """Choose t_k of the T_k trees of one site, without replacement.

    Uniform when ``scores`` is None, otherwise with probability proportional to
    ``scores`` (out of bag accuracies, NaN and negative values set to 0). Falls
    back to uniform when fewer than t_k scores are positive.

    Returns (sorted indices, fallback) where fallback tells whether the
    weighted draw fell back to uniform.
    """
    T_k = int(T_k)
    t_k = int(t_k)
    if not 0 <= t_k <= T_k:
        raise ValueError(f"t_k={t_k} outside [0, {T_k}]")
    if t_k == 0:
        return np.empty(0, dtype=np.int64), False
    fallback = False
    p = None
    if scores is not None:
        s = np.asarray(scores, dtype=np.float64).copy()
        if s.shape != (T_k,):
            raise ValueError("scores must have one value per tree")
        s[~np.isfinite(s)] = 0.0
        s[s < 0] = 0.0
        if np.count_nonzero(s > 0) < t_k:
            fallback = True
        else:
            p = s / s.sum()
    idx = rng.choice(T_k, size=t_k, replace=False, p=p)
    return np.sort(idx.astype(np.int64)), fallback


def _selected_site_means(tree_probs, selections):
    means, counts = [], []
    for probs, sel in zip(tree_probs, selections):
        sel = np.asarray(sel, dtype=np.int64)
        counts.append(sel.size)
        if sel.size:
            means.append(np.asarray(probs)[sel].mean(axis=0, dtype=np.float64))
        else:
            means.append(None)
    return means, np.asarray(counts, dtype=np.int64)


def rf_federated(tree_probs, selections):
    """Mean of P(y=1) over every selected tree of every site.

    ``tree_probs[k]`` has shape (T_k, n), ``selections[k]`` holds tree indices.
    """
    means, counts = _selected_site_means(tree_probs, selections)
    T = counts.sum()
    if T == 0:
        raise ValueError("no tree selected")
    out = np.zeros_like(next(m for m in means if m is not None))
    for m, c in zip(means, counts):
        if c:
            out += (c / T) * m
    return out


def rf_federated_shift(tree_probs, selections, pi, pi_k):
    """Federated forest with the level 1 correction applied per site.

    Each site averages its selected trees, the average is shifted from pi_k to
    pi, and the sites are combined with weights t_k / T. Sites with t_k = 0
    drop out.
    """
    pi_k = np.asarray(pi_k, dtype=np.float64)
    means, counts = _selected_site_means(tree_probs, selections)
    T = counts.sum()
    if T == 0:
        raise ValueError("no tree selected")
    out = None
    for k, (m, c) in enumerate(zip(means, counts)):
        if c:
            term = (c / T) * shift_prob(m, pi, pi_k[k])
            out = term if out is None else out + term
    return out


def rf_federated_global_shift(tree_probs, selections, pi, pi_k):
    """Sensitivity analysis: shift_prob(p_fed, pi, pi_fed) with
    pi_fed = sum_k (t_k / T) pi_k."""
    pi_k = np.asarray(pi_k, dtype=np.float64)
    _, counts = _selected_site_means(tree_probs, selections)
    p_fed = rf_federated(tree_probs, selections)
    pi_fed = float(np.sum(counts / counts.sum() * pi_k))
    return shift_prob(p_fed, pi, pi_fed)


# ----------------------------------------------------------------------------
# Level 3, XGBoost (and any model with log odds outputs)
# ----------------------------------------------------------------------------

def _stack(z):
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    return z


def xgb_federated(z, w):
    """sum_k w_k z_k, with z of shape (K, n)."""
    z = _stack(z)
    w = np.asarray(w, dtype=np.float64)
    return np.tensordot(w, z, axes=(0, 0))


def xgb_federated_shift(z, w, pi, pi_k):
    """sum_k w_k (z_k + logit pi - logit pi_k)."""
    z = _stack(z)
    pi_k = np.asarray(pi_k, dtype=np.float64)
    shifted = np.stack([shift_logit(z[k], pi, pi_k[k]) for k in range(z.shape[0])])
    return xgb_federated(shifted, w)


def xgb_federated_global_shift(z, w, pi, pi_k):
    """sum_k w_k z_k + logit pi - logit(sum_k w_k pi_k)."""
    w = np.asarray(w, dtype=np.float64)
    pi_k = np.asarray(pi_k, dtype=np.float64)
    return shift_logit(xgb_federated(z, w), pi, float(np.sum(w * pi_k)))


# ----------------------------------------------------------------------------
# Out of bag scores and leaf statistics
# ----------------------------------------------------------------------------

def tree_oob_scores(rf, X, y):
    """Accuracy of each tree on the rows outside its bootstrap sample,
    NaN when there is no such row."""
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    n = X.shape[0]
    scores = np.full(len(rf.estimators_), np.nan)
    for t, (tree, samples) in enumerate(zip(rf.estimators_, rf.estimators_samples_)):
        mask = np.ones(n, dtype=bool)
        mask[samples] = False
        if not mask.any():
            continue
        pred = rf.classes_[tree.predict(X[mask]).astype(int)]
        scores[t] = float(np.mean(pred == y[mask]))
    return scores


SMALL_LEAF = 5  # leaves with fewer distinct training rows than this are "small"


def _leaf_counts(leaves, small=SMALL_LEAF):
    """(leaves with one distinct row, leaves with fewer than ``small`` distinct
    rows, number of leaves), for the leaf index of every distinct row."""
    _, counts = np.unique(leaves, return_counts=True)
    return (int(np.count_nonzero(counts == 1)), int(np.count_nonzero(counts < small)),
            int(counts.size))


def _leaf_summary(n_single, n_small, n_leaves):
    nan = float("nan")
    return {"fraction": n_single / n_leaves if n_leaves else nan,
            "fraction_lt5": n_small / n_leaves if n_leaves else nan,
            "n_single": n_single, "n_lt5": n_small, "n_leaves": n_leaves}


def single_record_leaf_fraction(rf, X):
    """Fraction of leaves reached by exactly one distinct training row, and by
    fewer than 5 distinct training rows.

    Each tree is applied to its own bootstrap rows (distinct rows only).
    Returns dict(fraction, fraction_lt5, n_single, n_lt5, n_leaves) summed over trees.
    """
    X = np.asarray(X)
    tot = np.zeros(3, dtype=np.int64)
    for tree, samples in zip(rf.estimators_, rf.estimators_samples_):
        rows = np.unique(samples)
        tot += _leaf_counts(tree.apply(X[rows].astype(np.float32)))
    return _leaf_summary(*(int(v) for v in tot))


def xgb_single_record_leaf_fraction(model, X):
    """Same statistics for XGBoost, with ``pred_leaf=True`` on the training rows.
    Leaves are those reached by at least one training row."""
    import xgboost as xgb
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    leaves = booster.predict(xgb.DMatrix(np.asarray(X)), pred_leaf=True)
    leaves = np.asarray(leaves).reshape(np.asarray(X).shape[0], -1)
    tot = np.zeros(3, dtype=np.int64)
    for j in range(leaves.shape[1]):
        tot += _leaf_counts(leaves[:, j])
    return _leaf_summary(*(int(v) for v in tot))


# ----------------------------------------------------------------------------
# Summaries
# ----------------------------------------------------------------------------

def share(m0, m_level, m3):
    """r = (m_level - m0) / (m3 - m0), the share of the level 3 change that a
    cheaper level recovers."""
    den = m3 - m0
    if den == 0:
        return float("nan")
    return (m_level - m0) / den


def citl_intercept(z, y, lo=-20.0, hi=20.0, tol=1e-10, max_iter=200):
    """The b that solves sum (y - sigma(z + b)) = 0, by bisection on [lo, hi].
    Returns the bound when the root lies outside the interval."""
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    def f(b):
        return float(np.sum(y - expit(z + b)))

    f_lo, f_hi = f(lo), f(hi)
    if f_lo <= 0:
        return lo
    if f_hi >= 0:
        return hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if fm > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def calibration_slope(z, y):
    """Slope of a logistic regression of y on z (essentially no penalty)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a, _ = _unpenalized_logistic(z, y)
    return a
