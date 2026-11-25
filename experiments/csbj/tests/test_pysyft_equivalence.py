"""The PySyft path of fed_rf gives the predictions of the simulation.

Local DataSites (``FLServer`` with ``auto_accept=True``) hold the sites of one
partition of experiments/csbj/run_task.py (same draw, partition and seeds) and
the deployment site. ``FLClient`` trains at the sites, combines the models
with the rule of the ladder and runs every level at the deployment site. For
every level and every source site, the predictions on the test part must equal
those of the simulation within 1e-12 (Random Forest probabilities) and 1e-9
(XGBoost log odds).

Needs PySyft (not part of the E1 environment) and the data of the paper, and is
skipped otherwise. Run it from the repository root, for example

    ../.venv_syft/bin/python -m pytest -q -s experiments/csbj/tests/test_pysyft_equivalence.py

With ``PYSYFT_EQUIV_OUT=<file.json>`` the largest differences are written there.
"""
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sy = pytest.importorskip("syft")

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import config  # noqa: E402
import data as data_mod  # noqa: E402
import run_task as RT  # noqa: E402
from partition import partition, partition_rng, site_counts  # noqa: E402
from fed_rf_mk import ladder as L  # noqa: E402
from fed_rf_mk.client import FLClient  # noqa: E402
from fed_rf_mk.server import FLServer  # noqa: E402

TARGET = "outcome"
import secrets  # noqa: E402

EMAIL = "fedlearning@rf.com"
PASSWORD = secrets.token_urlsafe(16)          # data scientist password, given to spawn_server
ADMIN_PASSWORD = secrets.token_urlsafe(16)    # replaces the PySyft default admin password
TOL = {"rf": 1e-12, "xgb": 1e-9}

CONFIGS = [
    # dataset, learner, partition, K, site weights, tree sampling
    ("aids", "rf", "dir0.5", 3, "size", "oob"),
    ("diabetic", "xgb", "dir0.5", 4, "equal", "uniform"),
]


def _have_data(ds):
    return (config.data_dir() / f"{ds}.csv").exists()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ----------------------------------------------------------------------------
# Simulation side, with the functions of run_task.py in the order of run_cell
# ----------------------------------------------------------------------------

def simulation(ctx, part, K, learner, wname, sname):
    ds, r, f = ctx["ds"], ctx["r"], ctx["f"]
    X, y, tr, de, y_de, pi = ctx["X"], ctx["y"], ctx["tr"], ctx["de"], ctx["y_de"], ctx["pi"]
    dsc = config.DS_CODE[ds]
    pcode = config.PARTITIONS[part][0]
    site = partition(y[tr], K, part, partition_rng(dsc, r, f, K, pcode))
    n_k, pos_k = site_counts(y[tr], site, K)
    pi_k = pos_k / n_k
    X_de = X[de]
    X_de32 = np.ascontiguousarray(X_de, dtype=np.float32)
    cal = ctx["evals"]["test"][0]
    lc = config.LEARNER_CODE[learner]
    seeds = [config.seed_for(dsc, r, f, pcode, K, k, lc) for k in range(K)]
    arms = {}

    def level2(name, z_in):
        a, b = L.fit_recalibration(z_in[cal], y_de[cal])
        z2 = a * z_in + b
        arms[name] = {"p": L.sigmoid(z2), "z": z2, "a": a, "b": b}

    w = L.site_weights(n_k, pos_k, wname)
    if learner == "rf":
        probs, oob = [], []
        for k in range(K):
            rows = tr[site == k]
            pr, ob, _ = RT.fit_rf(X[rows], y[rows], seeds[k], X_de32)
            probs.append(pr)
            oob.append(ob)
        for k in range(K):
            p0 = probs[k].mean(axis=0, dtype=np.float64)
            p1 = L.shift_prob(p0, pi, pi_k[k])
            arms[f"src{k}_L0"] = {"p": p0, "z": RT.rf_logit(p0, 100)}
            arms[f"src{k}_L1"] = {"p": p1, "z": RT.rf_logit(p1, 100)}
            level2(f"src{k}_L2", RT.rf_logit(p0, 100))
        t = L.allocate_trees(w, np.full(K, 100))
        rng = np.random.default_rng([dsc, r, f, pcode, K, config.TREE_SELECT_TAG,
                                     config.WEIGHT_CODE[wname], config.SAMPLING_CODE[sname]])
        sel = [L.select_trees(100, t[k], rng, oob[k] if sname == "oob" else None)[0]
               for k in range(K)]
        T = int(t.sum())
        p0 = L.rf_federated(probs, sel)
        p1 = L.rf_federated_shift(probs, sel, pi, pi_k)
        p1g = L.rf_federated_global_shift(probs, sel, pi, pi_k)
        arms["fed_L0"] = {"p": p0, "z": RT.rf_logit(p0, T)}
        arms["fed_L1"] = {"p": p1, "z": RT.rf_logit(p1, T)}
        arms["fed_L1g"] = {"p": p1g, "z": RT.rf_logit(p1g, T)}
        level2("fed_L2", RT.rf_logit(p0, T))
        extra = {"t": t.tolist(), "selections": [s.tolist() for s in sel]}
    else:
        Z = []
        for k in range(K):
            rows = tr[site == k]
            z, _ = RT.fit_xgb(X[rows], y[rows], seeds[k], X_de)
            Z.append(z)
        Z = np.stack(Z)
        for k in range(K):
            z0 = Z[k]
            z1 = L.shift_logit(z0, pi, pi_k[k])
            arms[f"src{k}_L0"] = {"p": L.sigmoid(z0), "z": z0}
            arms[f"src{k}_L1"] = {"p": L.sigmoid(z1), "z": z1}
            level2(f"src{k}_L2", z0)
        z0 = L.xgb_federated(Z, w)
        z1 = L.xgb_federated_shift(Z, w, pi, pi_k)
        z1g = L.xgb_federated_global_shift(Z, w, pi, pi_k)
        arms["fed_L0"] = {"p": L.sigmoid(z0), "z": z0}
        arms["fed_L1"] = {"p": L.sigmoid(z1), "z": z1}
        arms["fed_L1g"] = {"p": L.sigmoid(z1g), "z": z1g}
        level2("fed_L2", z0)
        extra = {}
    return {"site": site, "n_k": n_k, "pos_k": pos_k, "w": w, "seeds": seeds, "arms": arms,
            **extra}


def check_against_run_cell(ctx, part, K, learner, wname, sname, sim):
    """The simulation above reproduces run_task.run_cell: per row predictions
    of the source arms (and of the default federated arms), and the Brier
    score, AUROC and (a, b) of the federated metric rows of this configuration."""
    sc, _, preds = RT.run_cell(ctx, part, K)
    worst = 0.0
    for k in range(K):
        for lev in ("L0", "L1", "L2"):
            d = np.max(np.abs(preds[f"{learner}_src{k}_{lev}"] - sim["arms"][f"src{k}_{lev}"]["p"]))
            worst = max(worst, float(d))
    if wname == config.DEFAULT_WEIGHTS and sname == config.DEFAULT_SAMPLING:
        for lev in ("L0", "L1", "L2"):
            d = np.max(np.abs(preds[f"{learner}_fed_{lev}"] - sim["arms"][f"fed_{lev}"]["p"]))
            worst = max(worst, float(d))
    m = sc.frame()
    test = ctx["test"]
    y_t = ctx["y_de"][test]
    s_rows = "na" if learner == "xgb" else sname
    for lev in ("L0", "L1", "L1g", "L2"):
        q = m[(m.learner == learner) & (m.arm == f"fed_{lev}") & (m.weights == wname)
              & (m.sampling == s_rows) & (m["eval"] == "test")
              & (m.n_cal == (ctx["cal"].size if lev == "L2" else 0))]
        assert len(q) == 1, (lev, len(q))
        q = q.iloc[0]
        p = sim["arms"][f"fed_{lev}"]["p"][test]
        assert np.mean((p - y_t) ** 2) == pytest.approx(q.brier, abs=1e-15)
        if lev != "L2":
            from sklearn.metrics import roc_auc_score
            assert roc_auc_score(y_t, p) == pytest.approx(q.auroc, abs=1e-15)
        else:
            assert sim["arms"]["fed_L2"]["a"] == q.rec_a and sim["arms"]["fed_L2"]["b"] == q.rec_b
    return worst


# ----------------------------------------------------------------------------
# PySyft side
# ----------------------------------------------------------------------------

def start_servers(specs, auto_accept=True):
    """specs: list of (name, csv path). Returns [(name, port, server)]."""
    out = []
    for name, path in specs:
        port = _free_port()
        srv = FLServer(name, port=port, data_path=str(path), auto_accept=auto_accept,
                       admin_password=ADMIN_PASSWORD, ds_email=EMAIL, ds_password=PASSWORD)
        th = threading.Thread(target=srv.start, daemon=True)
        th.start()
        out.append((name, port, srv))
    t0 = time.time()
    for name, _, srv in out:
        while srv.client is None or srv.thread is None:
            if time.time() - t0 > 300:
                raise TimeoutError(f"DataSite {name} did not start")
            time.sleep(0.5)
    return out


def write_csv(X, y, features, path):
    df = pd.DataFrame(X, columns=features)
    df[TARGET] = y.astype(int)
    df.to_csv(path, index=False)
    back = pd.read_csv(path)  # what the DataSite reads (datasites.create_syft_dataset)
    assert np.array_equal(back[features].to_numpy(dtype=np.float64), X), "CSV round trip"
    assert np.array_equal(back[TARGET].to_numpy(), y.astype(int))


def pysyft_run(tmp, ctx, features, part, K, learner, wname, sname, sim):
    ds, r, f = ctx["ds"], ctx["r"], ctx["f"]
    X, y, tr, de = ctx["X"], ctx["y"], ctx["tr"], ctx["de"]
    dsc = config.DS_CODE[ds]
    pcode = config.PARTITIONS[part][0]
    specs = []
    for k in range(K):
        rows = tr[sim["site"] == k]
        path = tmp / f"site{k}.csv"
        write_csv(X[rows], y[rows], features, path)
        specs.append((f"site{k}", path))
    path = tmp / "deploy.csv"
    write_csv(X[de], y[de], features, path)
    specs.append(("deploy", path))

    servers = start_servers(specs)
    try:
        fl = FLClient()
        for name, port, _ in servers[:-1]:
            fl.add_train_client(name, f"http://localhost:{port}", EMAIL, PASSWORD)
        name, port, _ = servers[-1]
        fl.add_eval_client(name, f"http://localhost:{port}", EMAIL, PASSWORD)
        fl.set_data_params({"target": TARGET, "ignored_columns": [TARGET]})
        params = {
            "model_type": learner,
            "n_base_estimators": 100,
            "n_incremental_estimators": 0,
            "train_size": 1.0,          # every record of a site trains its model
            "test_size": 0.2,
            "allow_analysis": False,
            "site_weights": wname,
            "tree_sampling": sname,
            "seed": [dsc, r, f, pcode, K, config.TREE_SELECT_TAG,
                     config.WEIGHT_CODE[wname], config.SAMPLING_CODE[sname]],
            "site_seeds": {f"site{k}": sim["seeds"][k] for k in range(K)},
            "calibration": {"fraction": config.CAL_FRACTION, "random_state": 1000 * r + f},
            "return_predictions": True,
            "release_level": "all",
        }
        if learner == "xgb":
            params.update(tree_method="hist", n_jobs=1)  # as config.XGB_PARAMS
        fl.set_model_params(params)
        fl.send_request()
        fl.run_model()        # waits for the approval of the training requests
        assert len(fl.model_aggregator.get_model_history()) == K
        lad = fl.model_params["ladder"]
        res = fl.run_evaluate()   # sends the evaluation request, waits for its approval
    finally:
        for _, _, srv in servers:
            srv.shutdown()
    assert isinstance(res, dict) and "error" not in res, res
    return lad, res


@pytest.mark.parametrize("ds,learner,part,K,wname,sname", CONFIGS)
def test_pysyft_equals_simulation(tmp_path, ds, learner, part, K, wname, sname):
    if not _have_data(ds):
        pytest.skip(f"no {ds}.csv in {config.data_dir()}")
    r, f = 0, 0
    d = data_mod.load(ds)
    ctx = RT.build_context(ds, r, f, X=d.X, y=d.y, nominal=d.nominal)
    sim = simulation(ctx, part, K, learner, wname, sname)
    worst_sim = check_against_run_cell(ctx, part, K, learner, wname, sname, sim)
    assert worst_sim == 0.0, worst_sim

    lad, res = pysyft_run(tmp_path, ctx, d.features, part, K, learner, wname, sname, sim)

    # what the sites sent and what the combination computed
    assert lad["n"] == sim["n_k"].tolist() and lad["n_pos"] == sim["pos_k"].tolist()
    assert lad["pi"] == ctx["pi"]
    assert np.array_equal(np.asarray(lad["weights"]), sim["w"])
    if learner == "rf":
        assert lad["t"] == sim["t"] and lad["selections"] == sim["selections"]

    pred = res["predictions"]
    test = ctx["test"]
    assert pred["test_positions"] == test.tolist()
    key = "p" if learner == "rf" else "z"
    diffs = {}
    for name, arm in sim["arms"].items():
        got = np.asarray(pred["arms"][name][key])
        diffs[name] = float(np.max(np.abs(got - arm[key][test])))
    worst = max(diffs.values())
    summary = {
        "dataset": ds, "learner": learner, "partition": part, "K": K, "rep": r, "fold": f,
        "site_weights": wname, "tree_sampling": sname, "compared": key,
        "tolerance": TOL[learner], "n_test": int(test.size), "n_cal": int(ctx["cal"].size),
        "n_k": sim["n_k"].tolist(), "pos_k": sim["pos_k"].tolist(),
        "t": sim.get("t"), "oob_fallback": lad.get("oob_fallback"),
        "max_abs_diff": diffs, "max_abs_diff_all": worst,
        "simulation_vs_run_cell": worst_sim,
        "max_abs_diff_rec_ab": max(
            max(abs(res["ladder"]["arms"][n]["rec_a"] - a["a"]),
                abs(res["ladder"]["arms"][n]["rec_b"] - a["b"]))
            for n, a in sim["arms"].items() if "a" in a),
    }
    print(json.dumps(summary))
    out = os.environ.get("PYSYFT_EQUIV_OUT")
    if out:
        p = Path(out)
        allres = json.loads(p.read_text()) if p.exists() else {}
        allres[f"{ds}_{learner}_{part}_K{K}_{wname}_{sname}"] = summary
        p.write_text(json.dumps(allres, indent=2, sort_keys=True))
    assert worst <= TOL[learner], diffs
