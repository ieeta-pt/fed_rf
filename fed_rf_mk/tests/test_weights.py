import importlib.util
from pathlib import Path


def load_module(rel_path: str, module_name: str):
    path = Path(rel_path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


weights_mod = load_module("orchestrator/weights.py", "weights")
WeightManager = weights_mod.WeightManager


def test_normalize_all_none_equal():
    wm = WeightManager()
    wm.set_weight("a", None)
    wm.set_weight("b", None)
    out = wm.normalize_weights(["a", "b"])
    assert abs(out["a"] - 0.5) < 1e-9
    assert abs(out["b"] - 0.5) < 1e-9


def test_normalize_mixed_some_none():
    wm = WeightManager()
    wm.set_weight("a", 0.7)
    wm.set_weight("b", None)
    wm.set_weight("c", None)
    out = wm.normalize_weights(["a", "b", "c"])
    # Remaining = 0.3 split among two -> 0.15 each, then sum to 1.0
    assert abs(out["a"] + out["b"] + out["c"] - 1.0) < 1e-9
    assert abs(out["b"] - out["c"]) < 1e-9


def test_normalize_all_defined_over_one():
    wm = WeightManager()
    wm.set_weight("a", 0.8)
    wm.set_weight("b", 0.6)
    out = wm.normalize_weights(["a", "b"])
    assert abs(out["a"] + out["b"] - 1.0) < 1e-9
    # a should be 0.8/1.4, b 0.6/1.4
    assert abs(out["a"] - (0.8 / 1.4)) < 1e-9
    assert abs(out["b"] - (0.6 / 1.4)) < 1e-9


def test_negative_treated_as_undefined():
    wm = WeightManager()
    wm.set_weight("a", -1.0)
    wm.set_weight("b", None)
    out = wm.normalize_weights(["a", "b"])
    assert abs(out["a"] - 0.5) < 1e-9
    assert abs(out["b"] - 0.5) < 1e-9

