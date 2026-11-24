import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import run_task as RT  # noqa: E402

METRICS = RT.METRICS


def _data():
    X, y = make_classification(n_samples=800, n_features=8, n_informative=5,
                               weights=[0.9], flip_y=0.02, random_state=0)
    return X.astype(np.float64), y.astype(int)


@pytest.fixture(scope="module")
def task(tmp_path_factory):
    root = tmp_path_factory.mktemp("task")
    X, y = _data()
    res = RT.run_task("synthetic", 0, 0, K_list=(2, 3), subsample=None, force=False,
                      n_jobs=1, runs=root / "runs", preds=root / "preds",
                      X=X, y=y, nominal={}, write_hyper=False)
    return root, res


def _frames(root):
    d = root / "runs" / "synthetic" / "r0_f0"
    cells = pd.concat([pd.read_parquet(p) for p in sorted(d.glob("*_K?.parquet"))])
    central = pd.read_parquet(d / "central.parquet")
    return cells, central


def test_no_errors(task):
    root, res = task
    assert res["n_error"] == 0, res["status"]
    d = root / "runs" / "synthetic" / "r0_f0"
    for part in config.PARTITIONS:
        for K in (2, 3):
            assert (d / f"{part}_K{K}.parquet").exists()
            assert (d / f"{part}_K{K}_sites.parquet").exists()
            assert (root / "preds" / "synthetic" / "r0_f0" / f"{part}_K{K}.parquet").exists()
    assert (d / "meta.json").exists()


def test_expected_arms(task):
    root, _ = task
    cells, central = _frames(root)
    for part in config.PARTITIONS:
        for K in (2, 3):
            c = cells[(cells.partition == part) & (cells.K == K)]
            for learner in ("rf", "xgb", "lr"):
                for arm in ("src_L0", "src_L1", "src_L2"):
                    for ev in RT.EVALS:
                        sub = c[(c.learner == learner) & (c.arm == arm) & (c["eval"] == ev)]
                        assert set(sub.source) == set(range(K)), (part, K, learner, arm, ev)
            for w in config.WEIGHT_CODE:
                for learner, samplings in (("rf", config.SAMPLING_CODE), ("xgb", ["na"])):
                    for s in samplings:
                        for arm in ("fed_L0", "fed_L1", "fed_L1g", "fed_L2"):
                            sub = c[(c.learner == learner) & (c.arm == arm) & (c.weights == w)
                                    & (c.sampling == s) & (c["eval"] == "test")]
                            assert len(sub) >= 1, (part, K, learner, arm, w, s)
            for learner in ("rf", "xgb"):
                for arm in ("fed_L0", "fed_L1", "fed_L2"):
                    for ev in RT.EVALS:
                        sub = c[(c.learner == learner) & (c.arm == arm) & (c.weights == "equal")
                                & (c.sampling.isin(["uniform", "na"])) & (c["eval"] == ev)]
                        assert len(sub) >= 1
    for learner in ("rf", "xgb", "lr"):
        for arm in ("central_L0", "central_L2"):
            for ev in RT.EVALS:
                assert len(central[(central.learner == learner) & (central.arm == arm)
                                   & (central["eval"] == ev)]) == 1
    # E12 subsets of the calibration part
    e12 = cells[(cells.arm == "src_L2") & (cells["eval"] == "test")]
    assert 25 in set(e12.n_cal)


def test_no_nan_metrics(task):
    root, _ = task
    cells, central = _frames(root)
    for df in (cells, central):
        assert not df[METRICS].isna().any().any()
        l2 = df[df.arm.str.endswith("L2")]
        assert not l2[["rec_a", "rec_b"]].isna().any().any()


def test_auroc_invariance(task):
    root, _ = task
    cells, _ = _frames(root)
    key = ["partition", "K", "learner", "source", "eval"]
    src = cells[cells.arm.str.startswith("src")]
    l0 = src[src.arm == "src_L0"].set_index(key)
    l1 = src[src.arm == "src_L1"].set_index(key)
    assert np.allclose(l0.auroc, l1.loc[l0.index].auroc, atol=1e-12)
    l2 = src[(src.arm == "src_L2") & (src.rec_a > 0)].set_index(key + ["n_cal"])
    ref = l0.loc[l2.index.droplevel("n_cal")].auroc.to_numpy()
    assert len(l2) > 0
    assert np.allclose(l2.auroc.to_numpy(), ref, atol=1e-12)


def test_level1_is_identity_when_prevalence_is_pooled(tmp_path):
    X, y = _data()
    ctx = RT.build_context("synthetic", 0, 0, X=X, y=y, nominal={})
    ctx["out_dir"], ctx["pred_dir"] = tmp_path, None
    sc, sites, _ = RT.run_cell(ctx, "equal", 1)  # one site holds the whole pool
    assert sites.pi_k.iloc[0] == pytest.approx(ctx["pi"])
    df = sc.frame()
    for learner in ("rf", "xgb", "lr"):
        a = df[(df.learner == learner) & (df.arm == "src_L0")].set_index("eval")[METRICS]
        b = df[(df.learner == learner) & (df.arm == "src_L1")].set_index("eval")[METRICS]
        assert np.allclose(a.to_numpy(), b.loc[a.index].to_numpy(), atol=1e-12)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_determinism(tmp_path):
    X, y = _data()
    outs = []
    for i in range(2):
        root = tmp_path / f"run{i}"
        res = RT.run_task("synthetic", 0, 0, K_list=(3,), partitions=("dir0.5",),
                          n_jobs=1, runs=root / "runs", preds=root / "preds",
                          X=X, y=y, nominal={}, write_hyper=False)
        assert res["n_error"] == 0
        d = root / "runs" / "synthetic" / "r0_f0"
        outs.append({p.name: _digest(p) for p in d.glob("*.parquet")}
                    | {"preds_" + p.name: _digest(p)
                       for p in (root / "preds" / "synthetic" / "r0_f0").glob("*.parquet")})
    assert outs[0] == outs[1]


def test_skip_existing(task):
    root, _ = task
    X, y = _data()
    res = RT.run_task("synthetic", 0, 0, K_list=(2, 3), n_jobs=1, runs=root / "runs",
                      preds=root / "preds", X=X, y=y, nominal={}, write_hyper=False)
    assert all(s == "skipped" for s in res["status"]), res["status"]


def test_regularized_main_arms_only(tmp_path):
    X, y = _data()
    res = RT.run_task("synthetic", 0, 0, K_list=(2,), n_jobs=1, runs=tmp_path / "runs",
                      preds=tmp_path / "preds", X=X, y=y, nominal={}, write_hyper=False,
                      hp="regularized")
    assert res["n_error"] == 0
    assert res["meta"]["hyperparameters"]["rf"]["min_samples_leaf"] == 5
    assert res["meta"]["hyperparameters"]["xgb"]["learning_rate"] == 0.1
    assert not (tmp_path / "preds" / "synthetic").exists()
    d = tmp_path / "runs" / "synthetic" / "r0_f0"
    c = pd.read_parquet(d / "equal_K2.parquet")
    central = pd.read_parquet(d / "central.parquet")
    assert set(c["eval"]) == {"test"} and set(central["eval"]) == {"test"}
    assert set(c.weights) == {"na", "equal"} and set(c.sampling) <= {"na", "uniform"}
    assert set(c.arm) == {"src_L0", "src_L1", "src_L2", "fed_L0", "fed_L1", "fed_L1g", "fed_L2"}
    l2 = c[c.arm.str.endswith("L2")]
    assert l2.n_cal.nunique() == 1  # full calibration part only
    assert not c[METRICS].isna().any().any()
    sites = pd.read_parquet(d / "equal_K2_sites.parquet")
    assert np.all(sites.xgb_leaf_frac == 0)  # min_child_weight=1 and hessian <= 0.25
