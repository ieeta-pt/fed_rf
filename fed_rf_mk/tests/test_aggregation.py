import importlib.util
from pathlib import Path
import cloudpickle
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

