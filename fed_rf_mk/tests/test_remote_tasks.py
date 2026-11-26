import importlib.util
from pathlib import Path
import pandas as pd
from sklearn.datasets import make_classification


def load_module(rel_path: str, module_name: str):
    path = Path(rel_path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


rt = load_module("remote_tasks.py", "remote_tasks")


def make_df(n=100, p=5, seed=0):
    X, y = make_classification(n_samples=n, n_features=p, n_redundant=0, random_state=seed)
    cols = [f"f{i}" for i in range(p)]
    df = pd.DataFrame(X, columns=cols)
    df["y"] = y
    return df


def test_ml_experiment_rf_no_analysis():
    df = make_df(60, 4)
    data_params = {"target": "y", "ignored_columns": ["y"]}
    model_params = {
        "model_type": "rf",
        "n_base_estimators": 5,
        "n_incremental_estimators": 0,
        "train_size": 0.7,
        "test_size": 0.3,
        "allow_analysis": False,
    }
    out = rt.ml_experiment(df, data_params, model_params)
    assert "model" in out and out["model"] is not None
    assert out["analysis_status"] != "enabled"


def test_evaluate_global_model_rf():
    # Train first
    df = make_df(60, 4)
    data_params = {"target": "y", "ignored_columns": ["y"]}
    model_params = {
        "model_type": "rf",
        "n_base_estimators": 5,
        "n_incremental_estimators": 0,
        "train_size": 0.7,
        "test_size": 0.3,
        "allow_analysis": False,
    }
    out = rt.ml_experiment(df, data_params, model_params)
    model_params["model"] = out["model"]

    metrics = rt.evaluate_global_model(df, data_params, model_params)
    assert "accuracy" in metrics or "error" in metrics



def test_ml_experiment_returns_counts_and_oob():
    df = make_df(80, 4)
    data_params = {"target": "y", "ignored_columns": ["y"]}
    model_params = {
        "model_type": "rf",
        "n_base_estimators": 7,
        "n_incremental_estimators": 0,
        "train_size": 1.0,
        "allow_analysis": False,
        "tree_sampling": "oob",
        "random_state": 3,
    }
    out = rt.ml_experiment(df, data_params, model_params)
    assert out["n_train"] == 80 and out["n_pos"] == int(df["y"].sum())
    assert len(out["oob_scores"]) == 7
    model_params["tree_sampling"] = "uniform"
    assert rt.ml_experiment(df, data_params, model_params)["oob_scores"] is None
