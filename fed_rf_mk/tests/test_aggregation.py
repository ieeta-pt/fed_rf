import importlib.util
from pathlib import Path
import cloudpickle
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


def load_module(rel_path: str, module_name: str):
    path = Path(rel_path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


agg_mod = load_module("aggregation/models.py", "aggregation_models")
ModelAggregator = agg_mod.ModelAggregator


def test_merge_estimators_rf():
    X, y = make_classification(n_samples=80, n_features=5, random_state=0)
    rf1 = RandomForestClassifier(n_estimators=10, random_state=1).fit(X, y)
    rf2 = RandomForestClassifier(n_estimators=10, random_state=2).fit(X, y)

    ma = ModelAggregator()
    ma.store_model_parameters("s1", {"model": cloudpickle.dumps(rf1)})
    ma.store_model_parameters("s2", {"model": cloudpickle.dumps(rf2)})

    merged = ma.merge_estimators({"s1": 0.6, "s2": 0.4})
    assert merged["seed_model"] is not None
    model = cloudpickle.loads(merged["seed_model"])  # RandomForestClassifier
    # Expected count approximates round(10*0.6) + round(10*0.4)
    expected = round(10 * 0.6) + round(10 * 0.4)
    assert len(model.estimators_) == expected



def _stored(n_sites=3, n_trees=10, oob=False):
    X, y = make_classification(n_samples=300, n_features=5, weights=[0.7], random_state=0)
    ma = ModelAggregator()
    rng = np.random.default_rng(0)
    for k in range(n_sites):
        rows = rng.choice(len(y), size=60 + 40 * k, replace=False)
        rf = RandomForestClassifier(n_estimators=n_trees, random_state=k).fit(X[rows], y[rows])
        mp = {"model": cloudpickle.dumps(rf), "n_train": int(rows.size),
              "n_pos": int(y[rows].sum())}
        if oob:
            mp["oob_scores"] = [float(v) for v in agg_mod.L.tree_oob_scores(rf, X[rows], y[rows])]
        ma.store_model_parameters(f"s{k}", mp)
    return ma


def test_combine_follows_ladder():
    L = agg_mod.L
    ma = _stored()
    order = ["s0", "s1", "s2"]
    n, n_pos = ma.counts(order)
    for kind in ("equal", "size", "positives"):
        out = ma.combine(order, site_weights=kind, seed=[1, 2, 3])
        w = L.site_weights(n, n_pos, kind)
        t = L.allocate_trees(w, np.full(3, 10))
        rng = np.random.default_rng([1, 2, 3])
        sel = [L.select_trees(10, t[k], rng)[0].tolist() for k in range(3)]
        lad = out["ladder"]
        assert lad["t"] == t.tolist() and lad["selections"] == sel
        assert lad["pi"] == L.pooled_prevalence(n, n_pos)
        merged = cloudpickle.loads(out["seed_model"])
        assert len(merged.estimators_) == int(t.sum())
        # same seed, same selection
        assert ma.combine(order, site_weights=kind, seed=[1, 2, 3])["ladder"]["selections"] == sel


def test_combine_oob_needs_scores():
    ma = _stored(oob=True)
    out = ma.combine(["s0", "s1", "s2"], site_weights="size", sampling="oob", seed=5)
    assert out["ladder"]["sampling"] == "oob" and len(out["ladder"]["oob_fallback"]) == 3
    ma = _stored(oob=False)
    try:
        ma.combine(["s0", "s1", "s2"], sampling="oob")
    except ValueError:
        pass
    else:
        raise AssertionError("oob sampling without scores must fail")
