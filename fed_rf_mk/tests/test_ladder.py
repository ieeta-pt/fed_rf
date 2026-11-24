"""Tests of fed_rf_mk/ladder.py, built on the worked examples of the paper."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

_path = Path(__file__).resolve().parents[1] / "ladder.py"
_spec = importlib.util.spec_from_file_location("ladder", _path)
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)

N = np.array([1000, 600, 400])
NPOS = np.array([100, 180, 220])


def test_prevalences():
    assert np.allclose(NPOS / N, [0.10, 0.30, 0.55])
    assert L.pooled_prevalence(N, NPOS) == pytest.approx(0.25)


def test_logit_offsets():
    pi = L.pooled_prevalence(N, NPOS)
    assert L.logit(pi) - L.logit(0.10) == pytest.approx(1.0986, abs=1e-4)
    assert L.logit(pi) - L.logit(0.55) == pytest.approx(-1.2993, abs=1e-4)
    assert round(float(L.logit(pi) - L.logit(0.10)), 2) == 1.10
    assert round(float(L.logit(pi) - L.logit(0.55)), 2) == -1.30


def test_site_weights():
    assert np.allclose(L.site_weights(N, NPOS, "equal"), [1 / 3] * 3)
    assert np.allclose(L.site_weights(N, NPOS, "size"), [0.5, 0.3, 0.2])
    assert np.allclose(L.site_weights(N, NPOS, "positives"), [0.20, 0.36, 0.44])
    with pytest.raises(ValueError):
        L.site_weights(N, NPOS, "other")


@pytest.mark.parametrize("kind,expected", [
    ("equal", [34, 33, 33]),
    ("size", [50, 30, 20]),
    ("positives", [20, 36, 44]),
])
def test_allocate_trees_examples(kind, expected):
    t = L.allocate_trees(L.site_weights(N, NPOS, kind), [100, 100, 100])
    assert t.tolist() == expected
    assert t.sum() == 100


def test_allocate_trees_cap_and_total():
    rng = np.random.default_rng(0)
    for _ in range(500):
        K = int(rng.integers(1, 11))
        w = rng.dirichlet(np.ones(K))
        T_k = rng.integers(1, 120, size=K)
        t = L.allocate_trees(w, T_k)
        assert np.all(t <= T_k) and np.all(t >= 0)
        assert t.sum() == min(int(round(float(np.sum(w * T_k)))), T_k.sum())
    # a site with few trees is capped and its share goes elsewhere
    t = L.allocate_trees([0.9, 0.1], [10, 100])
    assert t.tolist() == [9, 10]


def test_xgb_example():
    z = np.array([[-1.2], [0.3], [0.0]])
    zf = L.xgb_federated(z, L.site_weights(N, NPOS, "equal"))
    assert zf[0] == pytest.approx(-0.3)
    assert L.sigmoid(zf)[0] == pytest.approx(0.4256, abs=1e-4)
    assert round(float(L.sigmoid(zf)[0]), 2) == 0.43


def test_share_example():
    assert L.share(0.30, 0.22, 0.20) == pytest.approx(0.80)


def test_discussion_example():
    p = np.array([0.02, 0.30])
    assert p.mean() == pytest.approx(0.16)
    pz = L.sigmoid(L.xgb_federated(L.logit(p)[:, None], [0.5, 0.5]))[0]
    assert pz == pytest.approx(0.0855, abs=1e-4)
    assert round(float(pz), 2) == 0.09


def test_shift_prob_matches_logit_form_and_keeps_bounds():
    p = np.linspace(0.001, 0.999, 999)
    for pi, pi_k in [(0.25, 0.10), (0.25, 0.55), (0.05, 0.5)]:
        lhs = L.logit(L.shift_prob(p, pi, pi_k))
        rhs = L.shift_logit(L.logit(p), pi, pi_k)
        assert np.allclose(lhs, rhs)
        q = L.shift_prob(np.array([0.0, 1.0]), pi, pi_k)
        assert q.tolist() == [0.0, 1.0]
        s = L.shift_prob(np.concatenate([[0.0], p, [1.0]]), pi, pi_k)
        assert np.all(np.diff(s) > 0)


def test_shift_and_recalibration_keep_auroc():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    # Random Forest probabilities are multiples of 1/T, so clipping keeps their order
    p = np.round(np.clip(0.3 * y + rng.uniform(0, 0.7, 500), 0, 1) * 100) / 100
    p[:5] = [0.0, 1.0, 0.0, 1.0, 0.5]
    base = roc_auc_score(y, p)
    assert roc_auc_score(y, L.shift_prob(p, 0.25, 0.55)) == pytest.approx(base, abs=1e-12)
    z = L.logit(L.clip_prob(p, 100))
    assert roc_auc_score(y, L.apply_recalibration(z, 0.7, -0.4)) == pytest.approx(base, abs=1e-12)


def test_clip_prob():
    p = L.clip_prob(np.array([0.0, 0.5, 1.0]), 100)
    assert p.tolist() == [0.005, 0.5, 0.995]


def test_fit_recalibration_recovers_parameters():
    rng = np.random.default_rng(2)
    z = rng.normal(size=20000)
    y = (rng.uniform(size=z.size) < L.sigmoid(1.5 * z - 0.7)).astype(int)
    a, b = L.fit_recalibration(z, y)
    assert a == pytest.approx(1.5, abs=0.08)
    assert b == pytest.approx(-0.7, abs=0.08)


@pytest.fixture(scope="module")
def forest_data():
    X, y = make_classification(n_samples=400, n_features=6, weights=[0.8], random_state=3)
    rf = RandomForestClassifier(n_estimators=30, random_state=4, n_jobs=1).fit(X, y)
    probs = np.stack([t.predict_proba(X)[:, 1] for t in rf.estimators_])
    return X, y, rf, probs


def test_rf_federated_full_forest_equals_predict_proba(forest_data):
    X, _, rf, probs = forest_data
    sel = [np.arange(probs.shape[0])]
    assert np.allclose(L.rf_federated([probs], sel), rf.predict_proba(X)[:, 1])


def test_k1_federated_equals_site_model(forest_data):
    X, _, rf, probs = forest_data
    T = probs.shape[0]
    t = L.allocate_trees(L.site_weights([400], [80], "equal"), [T])
    assert t.tolist() == [T]
    sel, _ = L.select_trees(T, t[0], np.random.default_rng(0))
    assert np.allclose(L.rf_federated([probs], [sel]), rf.predict_proba(X)[:, 1])
    pi = 0.2
    assert np.allclose(L.rf_federated_shift([probs], [sel], pi, [pi]),
                       rf.predict_proba(X)[:, 1])
    z = np.array([[-1.0, 0.5, 2.0]])
    assert np.allclose(L.xgb_federated(z, [1.0]), z[0])
    assert np.allclose(L.xgb_federated_shift(z, [1.0], 0.3, [0.3]), z[0])


def test_rf_federated_shift_per_site():
    rng = np.random.default_rng(5)
    probs = [rng.uniform(size=(10, 7)), rng.uniform(size=(10, 7)), rng.uniform(size=(10, 7))]
    sel = [np.arange(4), np.arange(6), np.empty(0, dtype=int)]
    pi, pi_k = 0.25, np.array([0.1, 0.3, 0.55])
    got = L.rf_federated_shift(probs, sel, pi, pi_k)
    want = 0.4 * L.shift_prob(probs[0][:4].mean(0), pi, 0.1) + \
        0.6 * L.shift_prob(probs[1][:6].mean(0), pi, 0.3)
    assert np.allclose(got, want)
    p_fed = L.rf_federated(probs, sel)
    assert np.allclose(p_fed, 0.4 * probs[0][:4].mean(0) + 0.6 * probs[1][:6].mean(0))
    g = L.rf_federated_global_shift(probs, sel, pi, pi_k)
    assert np.allclose(g, L.shift_prob(p_fed, pi, 0.4 * 0.1 + 0.6 * 0.3))


def test_xgb_shift_variants():
    z = np.array([[-1.2, 0.1], [0.3, 0.2], [0.0, -0.5]])
    w = np.array([0.5, 0.3, 0.2])
    pi, pi_k = 0.25, np.array([0.1, 0.3, 0.55])
    want = sum(w[k] * (z[k] + L.logit(pi) - L.logit(pi_k[k])) for k in range(3))
    assert np.allclose(L.xgb_federated_shift(z, w, pi, pi_k), want)
    want_g = (w[:, None] * z).sum(0) + L.logit(pi) - L.logit(np.sum(w * pi_k))
    assert np.allclose(L.xgb_federated_global_shift(z, w, pi, pi_k), want_g)


def test_xgb_stump_margin_is_logit_prevalence():
    from xgboost import XGBClassifier
    rng = np.random.default_rng(6)
    n = 20000
    y = np.zeros(n, dtype=int)
    y[: n // 20] = 1
    rng.shuffle(y)
    X = rng.normal(size=(n, 3))
    m = XGBClassifier(n_estimators=1, max_depth=1, learning_rate=1e-6,
                      tree_method="hist", n_jobs=1, random_state=0).fit(X, y)
    z = m.predict(X, output_margin=True)
    assert float(np.mean(z)) == pytest.approx(float(L.logit(0.05)), abs=1e-4)


def test_select_trees():
    rng_a, rng_b = np.random.default_rng(7), np.random.default_rng(7)
    a, fa = L.select_trees(100, 40, rng_a)
    b, fb = L.select_trees(100, 40, rng_b)
    assert np.array_equal(a, b) and not fa and not fb
    assert len(np.unique(a)) == 40 and a.min() >= 0 and a.max() < 100
    scores = np.linspace(0.5, 1.0, 100)
    c, fc = L.select_trees(100, 40, np.random.default_rng(8), scores)
    d, _ = L.select_trees(100, 40, np.random.default_rng(8), scores)
    assert np.array_equal(c, d) and not fc and len(np.unique(c)) == 40
    # only 3 positive scores for 5 trees -> fallback to uniform
    s = np.full(10, np.nan)
    s[:3] = 0.9
    s[3] = -1.0
    e, fe = L.select_trees(10, 5, np.random.default_rng(9), s)
    u, _ = L.select_trees(10, 5, np.random.default_rng(9))
    assert fe and np.array_equal(e, u)
    # weighted draw never picks a zero score tree
    s = np.zeros(20)
    s[:6] = 1.0
    for seed in range(50):
        f, ff = L.select_trees(20, 6, np.random.default_rng(seed), s)
        assert not ff and set(f.tolist()) == set(range(6))
    z, fz = L.select_trees(10, 0, np.random.default_rng(0))
    assert z.size == 0 and not fz


def test_tree_oob_scores(forest_data):
    X, y, rf, _ = forest_data
    s = L.tree_oob_scores(rf, X, y)
    assert s.shape == (len(rf.estimators_),)
    assert np.all((s >= 0) & (s <= 1))
    t = rf.estimators_[0]
    mask = np.ones(len(y), bool)
    mask[rf.estimators_samples_[0]] = False
    assert s[0] == pytest.approx(np.mean(t.predict(X[mask]) == y[mask]))


def test_single_record_leaf_fraction(forest_data):
    X, _, rf, _ = forest_data
    out = L.single_record_leaf_fraction(rf, X)
    assert out["n_leaves"] == sum(t.get_n_leaves() for t in rf.estimators_)
    assert 0 < out["fraction"] < 1
    # by hand for the first tree
    rows = np.unique(rf.estimators_samples_[0])
    _, c = np.unique(rf.estimators_[0].apply(X[rows].astype(np.float32)), return_counts=True)
    assert c.size == rf.estimators_[0].get_n_leaves()


def test_xgb_single_record_leaf_fraction():
    from xgboost import XGBClassifier
    X, y = make_classification(n_samples=300, n_features=5, random_state=10)
    m = XGBClassifier(n_estimators=5, max_depth=6, n_jobs=1, random_state=0).fit(X, y)
    out = L.xgb_single_record_leaf_fraction(m, X)
    assert out["n_leaves"] > 0 and 0 <= out["fraction"] <= out["fraction_lt5"] <= 1


def test_citl_intercept_and_slope():
    rng = np.random.default_rng(11)
    z = rng.normal(size=5000)
    y = (rng.uniform(size=z.size) < L.sigmoid(z + 0.5)).astype(int)
    b = L.citl_intercept(z, y)
    assert abs(np.sum(y - L.sigmoid(z + b))) < 1e-6
    assert L.calibration_slope(z, y) == pytest.approx(1.0, abs=0.1)
    assert L.citl_intercept(np.zeros(10), np.zeros(10)) == -20.0


def test_leaf_counts_small_leaves():
    leaves = np.array([7, 7, 7, 7, 7, 3, 3, 9, 4, 4, 4, 4])  # sizes 5, 2, 1, 4
    assert L._leaf_counts(leaves) == (1, 3, 4)


def test_leaf_fraction_lt5(forest_data):
    X, y, rf, _ = forest_data
    out = L.single_record_leaf_fraction(rf, X)
    assert out["n_single"] <= out["n_lt5"] <= out["n_leaves"]
    assert out["fraction_lt5"] == pytest.approx(out["n_lt5"] / out["n_leaves"])
    reg = RandomForestClassifier(n_estimators=10, min_samples_leaf=5, random_state=4,
                                 n_jobs=1).fit(X, y)
    r = L.single_record_leaf_fraction(reg, X)
    assert r["n_single"] == 0 and r["n_lt5"] == 0  # min_samples_leaf counts distinct rows
