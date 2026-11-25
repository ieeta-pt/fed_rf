"""What leaves each site for every ``release_level``, with manual approval.

Local DataSites (``FLServer`` with the default ``auto_accept=False``) hold a
synthetic dataset split over three training sites and a deployment site. For
every value of ``release_level`` the test approves each request by hand
(``list_pending_requests``, ``request_parameters``, ``approve_request``) and
checks

* which training sites get a request, and that the values bound in each
  request are those the client sent,
* which keys each site returns,
* what the deployment site receives (bound in its request) and which arms it
  computes.

Other tests check the synthetic mock, the accounts, the binding of the
parameter values and the analysis switch. Needs PySyft and is skipped
otherwise. With ``PYSYFT_RELEASE_OUT=<file.json>`` the observed keys are
written there.
"""
import json
import os
import secrets
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sy = pytest.importorskip("syft")

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from fed_rf_mk import datasites  # noqa: E402
from fed_rf_mk.client import FLClient  # noqa: E402
from fed_rf_mk.release import RELEASE_LEVELS  # noqa: E402
from fed_rf_mk.server import FLServer  # noqa: E402

TARGET = "outcome"
EMAIL = "fedlearning@rf.com"
PASSWORD = secrets.token_urlsafe(16)
ADMIN_PASSWORD = secrets.token_urlsafe(16)
K = 3
SITES = [f"site{k}" for k in range(K)]
SOURCE = "site1"

FULL = {"model", "model_type", "n_base_estimators", "n_incremental_estimators", "train_size",
        "sample_size", "test_size", "n_train", "n_pos", "oob_scores", "analysis_status"}
# Random Forest, size weights, out of bag sampling
EXPECTED_SITE = {
    "0": {SOURCE: {"model_type", "model"}},
    "1": {s: ({"model_type", "model", "n_train", "n_pos"} if s == SOURCE else {"n_train", "n_pos"})
          for s in SITES},
    "2": {SOURCE: {"model_type", "model"}},
    "3": {s: {"model_type", "model", "n_train", "oob_scores"} for s in SITES},
    "3+1": {s: {"model_type", "model", "n_train", "n_pos", "oob_scores"} for s in SITES},
    "3+2": {s: {"model_type", "model", "n_train", "oob_scores"} for s in SITES},
    "all": {s: FULL for s in SITES},
}
COMMON = {"model_type", "sites", "sources", "models"}
FED = {"weights", "weights_kind", "sampling", "seed", "T_k", "t", "selections", "oob_fallback"}
EXPECTED_DEPLOY = {
    "0": (COMMON, 1), "1": (COMMON | {"pi", "pi_k"}, 1), "2": (COMMON, 1),
    "3": (COMMON | FED, K), "3+1": (COMMON | FED | {"pi", "pi_k"}, K), "3+2": (COMMON | FED, K),
    "all": (COMMON | FED | {"n", "n_pos", "pi", "pi_k"}, K),
}
SRC_ARMS = {"0": ("L0",), "1": ("L0", "L1"), "2": ("L0", "L2"), "3": ("L0",),
            "3+1": ("L0", "L1"), "3+2": ("L0", "L2"), "all": ("L0", "L1", "L2")}
FED_ARMS = {"0": (), "1": (), "2": (), "3": ("L0",), "3+1": ("L0", "L1", "L1g"),
            "3+2": ("L0", "L2"), "all": ("L0", "L1", "L1g", "L2")}


def expected_arms(level):
    src = [SITES.index(SOURCE)] if level in ("0", "1", "2") else range(K)
    return ({f"src{k}_{a}" for k in src for a in SRC_ARMS[level]}
            | {f"fed_{a}" for a in FED_ARMS[level]})


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start(specs, **kw):
    out = []
    for name, path in specs:
        srv = FLServer(name, port=_free_port(), data_path=str(path), admin_password=ADMIN_PASSWORD,
                       ds_email=EMAIL, ds_password=PASSWORD, **kw)
        threading.Thread(target=srv.start, daemon=True).start()
        out.append(srv)
    t0 = time.time()
    for srv in out:
        while srv.client is None or (srv.auto_accept and srv.thread is None):
            if time.time() - t0 > 300:
                raise TimeoutError(srv.name)
            time.sleep(0.5)
    return out


def url(srv):
    return f"http://localhost:{srv.port}"


@pytest.fixture(scope="module")
def csvs(tmp_path_factory):
    from sklearn.datasets import make_classification
    root = tmp_path_factory.mktemp("release")
    X, y = make_classification(n_samples=800, n_features=6, n_informative=4,
                               weights=[0.75], random_state=3)
    cols = [f"f{i}" for i in range(X.shape[1])]
    rng = np.random.default_rng(0)
    part = rng.integers(0, K + 1, size=y.size)
    paths = {}
    for j, name in enumerate(SITES + ["deploy"]):
        df = pd.DataFrame(X[part == j], columns=cols)
        df[TARGET] = y[part == j]
        paths[name] = root / f"{name}.csv"
        df.to_csv(paths[name], index=False)
    return paths


def approve_all(srv, expected_func=None):
    """Approve every pending request by hand; return the bound values."""
    bound = []
    for idx, func in srv.list_pending_requests():
        if expected_func is not None:
            assert func == expected_func, (srv.name, func)
        bound.append((func, srv.request_parameters(idx)))
        srv.approve_request(idx)
    return bound


@pytest.mark.parametrize("level", RELEASE_LEVELS)
def test_release_level(csvs, level):
    servers = start([(n, csvs[n]) for n in SITES + ["deploy"]])  # auto_accept=False (default)
    by_name = {s.name: s for s in servers}
    try:
        fl = FLClient()
        for n in SITES:
            fl.add_train_client(n, url(by_name[n]), EMAIL, PASSWORD)
        fl.add_eval_client("deploy", url(by_name["deploy"]), EMAIL, PASSWORD)
        fl.set_data_params({"target": TARGET, "ignored_columns": [TARGET]})
        fl.set_model_params({
            "model_type": "rf", "n_base_estimators": 20, "n_incremental_estimators": 0,
            "train_size": 1.0, "allow_analysis": False, "site_weights": "size",
            "tree_sampling": "oob", "seed": 7, "release_level": level, "source_site": SOURCE,
            "approval_timeout": 120,
        })
        fl.send_request()

        # requests: only the sites of the plan, with the values the client sent
        observed = {"site_requests": {}, "site_keys": {}}
        for n in SITES:
            bound = approve_all(by_name[n])
            observed["site_requests"][n] = [f for f, _ in bound]
            if n in EXPECTED_SITE[level]:
                assert len(bound) == 1, (n, bound)
                func, params = bound[0]
                assert func == ("site_counts" if EXPECTED_SITE[level][n] == {"n_train", "n_pos"}
                                else "ml_experiment")
                assert params["modelParams"] == fl.site_params_sent[n]
                assert params["dataParams"] == fl.data_params
            else:
                assert bound == [], (n, bound)
        fl.run_model()

        # keys that left each site
        assert set(fl.site_results) == set(EXPECTED_SITE[level])
        for n, keys in EXPECTED_SITE[level].items():
            assert set(fl.site_results[n]) == keys, (n, set(fl.site_results[n]))
            observed["site_keys"][n] = sorted(fl.site_results[n])

        # what the deployment site receives: only the bound evaluation request
        fl.send_evaluation_request()
        bound = approve_all(by_name["deploy"], "evaluate_global_model")
        assert len(bound) == 1
        mp = bound[0][1]["modelParams"]
        assert mp == fl.eval_params_sent
        lad = mp["ladder"]
        keys, n_models = EXPECTED_DEPLOY[level]
        assert set(lad) == keys, set(lad) ^ keys
        assert len(lad["models"]) == n_models
        if level != "all":
            assert set(mp) == {"model_type", "release_level", "calibration",
                               "return_predictions", "ladder"}
        observed["deploy_model_params"] = sorted(mp)
        observed["deploy_ladder"] = sorted(lad)
        observed["deploy_n_models"] = len(lad["models"])

        res = fl.run_evaluate()
        assert "error" not in res, res
        arms = set(res["ladder"]["arms"])
        assert arms == expected_arms(level), arms ^ expected_arms(level)
        observed["arms"] = sorted(arms)
    finally:
        for s in servers:
            s.shutdown()
    out = os.environ.get("PYSYFT_RELEASE_OUT")
    if out:
        p = Path(out)
        allres = json.loads(p.read_text()) if p.exists() else {}
        allres[level] = observed
        p.write_text(json.dumps(allres, indent=2, sort_keys=True))


def test_mock_accounts_binding_analysis(csvs):
    with pytest.raises(ValueError):
        datasites.spawn_server("x", port=_free_port(), data_path=str(csvs["site0"]),
                               admin_password=None, ds_password="pw")  # no admin password
    with pytest.raises(ValueError):
        datasites.spawn_server("x", port=_free_port(), data_path=str(csvs["site0"]),
                               admin_password="changethis", ds_password="pw")

    datasites.set_analysis_allowed(True)
    on, = start([("on", csvs["site0"])], auto_accept=True, analysis_allowed=True)
    datasites.set_analysis_allowed(False)
    off, = start([("off", csvs["site0"])], auto_accept=True, analysis_allowed=False)
    try:
        real = pd.read_csv(csvs["site0"])
        # synthetic mock seen by the data scientist
        ds = sy.login(email=EMAIL, password=PASSWORD, url=url(on))
        mock = ds.datasets[0].assets[0].mock
        assert list(mock.columns) == list(real.columns)
        assert list(mock.dtypes) == list(real.dtypes)
        assert len(mock) == 100
        assert not mock.equals(real)
        assert pd.merge(mock, real, how="inner").empty  # no real row in the mock
        # accounts: default admin password replaced, guest signup off
        with pytest.raises(Exception):
            sy.login(email=datasites.ADMIN_EMAIL, password="changethis", url=url(on))
        assert on.client.settings.get().signup_enabled is False

        for srv, want in ((on, "enabled"), (off, "disabled_by_silo_policy")):
            fl = FLClient()
            fl.add_train_client(srv.name, url(srv), EMAIL, PASSWORD)
            fl.set_data_params({"target": TARGET, "ignored_columns": [TARGET]})
            fl.set_model_params({
                "model_type": "rf", "n_base_estimators": 5, "n_incremental_estimators": 0,
                "train_size": 0.7, "test_size": 0.3, "allow_analysis": True,
                "analysis": {"enabled": True, "do_shap": False, "do_pfi": True},
                "approval_timeout": 60,
            })
            fl.send_request()
            fl.run_model()
            res = fl.site_results[srv.name]
            # analysis switch of the DataSite reaches the approved code
            assert res["analysis_status"] == want, res["analysis_status"]
            assert ("pfi_data" in res) == (want == "enabled")

        # binding: the approved code runs with the bound values only
        client = sy.login(email=EMAIL, password=PASSWORD, url=url(off))
        asset = client.datasets[0].assets[0]
        n_bound = client.code.ml_experiment(data=asset).get_from(client)["sample_size"]
        n_other = client.code.ml_experiment(
            data=asset, modelParams={"train_size": 0.5, "n_base_estimators": 5,
                                     "n_incremental_estimators": 0, "model_type": "rf"},
        ).get_from(client)["sample_size"]
        assert n_bound == n_other == int(np.floor(0.7 * len(real)))
    finally:
        datasites.set_analysis_allowed(False)
        os.environ.pop(datasites.ENV_ANALYSIS, None)
        on.shutdown()
        off.shutdown()
