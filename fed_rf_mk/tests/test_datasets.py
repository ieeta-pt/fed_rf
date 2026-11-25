import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def load_module(name):
    path = Path(__file__).resolve().parents[1] / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D = load_module("datasets")
R = load_module("release")


def test_generate_mock_is_synthetic():
    rng = np.random.default_rng(1)
    data = pd.DataFrame({
        "age": rng.integers(18, 90, 300),
        "bmi": rng.normal(27, 4, 300),
        "sex": rng.choice(["F", "M"], 300),
        "smoker": rng.random(300) < 0.3,
        "y": rng.integers(0, 2, 300),
    })
    mock = D.generate_mock(data)
    assert list(mock.columns) == list(data.columns)
    assert list(mock.dtypes) == list(data.dtypes)
    assert len(mock) == D.MOCK_ROWS
    assert mock["age"].between(data["age"].min(), data["age"].max()).all()
    assert mock["bmi"].between(data["bmi"].min(), data["bmi"].max()).all()
    assert set(mock["sex"]) <= set(data["sex"]) and set(mock["y"]) <= {0, 1}
    assert pd.merge(mock, data, how="inner").empty
    assert D.generate_mock(data).equals(mock)  # fixed seed
    assert len(D.generate_mock(data.head(10))) == D.MOCK_ROWS  # rows do not depend on data


def test_release_plan():
    sites = ["a", "b", "c"]
    assert R.site_plan("all", sites) == {s: {"function": "ml_experiment", "release": None}
                                         for s in sites}
    p0 = R.site_plan(0, sites, source_site="b")
    assert list(p0) == ["b"] and R.released_keys(p0["b"]) == {"model_type", "model"}
    p1 = R.site_plan("1", sites, source_site="b")
    assert R.released_keys(p1["b"]) == {"model_type", "model", "n_train", "n_pos"}
    assert p1["a"]["function"] == "site_counts" and R.released_keys(p1["a"]) == {"n_train", "n_pos"}
    p3 = R.site_plan("3", sites, site_weights="equal", tree_sampling="uniform")
    assert all(R.released_keys(e) == {"model_type", "model"} for e in p3.values())
    p3 = R.site_plan("3+2", sites, site_weights="positives", tree_sampling="oob", model_type="xgb")
    assert all(R.released_keys(e) == {"model_type", "model", "n_pos"} for e in p3.values())
    p31 = R.site_plan("3+1", sites, tree_sampling="oob")
    assert all(R.released_keys(e) == {"model_type", "model", "n_train", "n_pos", "oob_scores"}
               for e in p31.values())
    for bad in (dict(level="0"), dict(level="2", source_site="z"), dict(level="4")):
        try:
            R.site_plan(bad.pop("level"), sites, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)


def test_deployment_ladder():
    lad = {"model_type": "rf", "sites": ["a"], "sources": [0], "models": [b"m"], "n": [1],
           "n_pos": [1], "pi": 0.2, "pi_k": [0.2], "weights": [1.0], "t": [10]}
    assert set(R.deployment_ladder("0", lad)) == {"model_type", "sites", "sources", "models"}
    assert set(R.deployment_ladder("1", lad)) == {"model_type", "sites", "sources", "models",
                                                  "pi", "pi_k"}
    assert "n" not in R.deployment_ladder("3+1", lad) and "pi" in R.deployment_ladder("3+1", lad)
    assert R.deployment_ladder("all", lad) == lad
