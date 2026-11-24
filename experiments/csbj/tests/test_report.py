import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze as A  # noqa: E402
import run_task as RT  # noqa: E402
import tables as T  # noqa: E402


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    root = tmp_path_factory.mktemp("report")
    X, y = make_classification(n_samples=800, n_features=8, n_informative=5,
                               weights=[0.8], flip_y=0.02, random_state=1)
    for f in range(3):
        for hp, runs in (("default", root / "runs"), ("regularized", root / "runs_reg")):
            res = RT.run_task("synthetic", 0, f, K_list=(2, 3), n_jobs=1, runs=runs,
                              preds=root / "preds", X=X, y=y, nominal={}, write_hyper=False,
                              hp=hp)
            assert res["n_error"] == 0
    out = root / "results"
    N = T.build(root / "runs", root / "runs_reg", root / "preds", out, ("synthetic",),
                r_rep=1, r_rep_reg=1, allow_incomplete=True)
    return root, out, N


def test_outputs_exist(report):
    _, out, _ = report
    for stem in ("T1_levels.csv", "T1_brier.tex", "T1_auroc.tex", "T2_shares.csv", "T2_shares.tex",
                 "T3_auroc_gain.csv", "T3_auroc_gain.tex", "perK_differences.csv", "S_E5.csv",
                 "S_E5.tex", "S_L1g.csv", "S_E9_leaves.csv", "S_E10.csv", "S_E11_gain.csv",
                 "S_E12.csv", "S_hp.csv", "S_hp.tex", "S_calibration.csv", "S_rfxgb.csv",
                 "anomalies.csv", "reliability.csv", "sites_r0_f0_K10.csv", "numbers.json",
                 "numbers.md"):
        assert (out / stem).exists(), stem
    nums = json.loads((out / "numbers.json").read_text())
    md = (out / "numbers.md").read_text()
    assert all(f"`{k}`" in md for k in nums)
    assert nums["design.n_draws.synthetic"]["value"] == 3
    assert nums["design.regularized.n_draws.synthetic"]["value"] == 3
    assert not any(k.startswith(("r.", "diff.")) and ".lr." in k for k in nums)  # E13 not reported


def test_t2_matches_direct_computation(report):
    root, out, _ = report
    t2 = pd.read_csv(out / "T2_shares.csv")
    m, _, _ = A.load_runs(root / "runs", ("synthetic",))
    pdw = A.average_over_K(A.levels_per_cell(A.default_view(m), "brier"))
    g = pdw[(pdw.learner == "rf") & (pdw.partition == "dir0.5")]
    r1 = (g.m1 - g.m0).mean() / (g.m31 - g.m0).mean()
    q = t2[(t2.metric == "brier") & (t2.learner == "rf") & (t2.partition == "dir0.5")
           & (t2.level == 1)].iloc[0]
    assert q.r == pytest.approx(r1)
    assert {"r_lo", "r_hi", "reportable"} <= set(t2.columns)
    assert set(t2.learner) == {"rf", "xgb"}


def test_e5_and_l1g(report):
    _, out, _ = report
    e5 = pd.read_csv(out / "S_E5.csv")
    rf = e5[(e5.learner == "rf") & (e5.arm == "fed_L0") & (e5.metric == "auroc")
            & (e5.partition == "dir0.5")]
    assert {(w, s) for w, s in zip(rf.weights, rf.sampling)} == {
        ("equal", "oob"), ("size", "uniform"), ("size", "oob"),
        ("positives", "uniform"), ("positives", "oob")}
    xg = e5[(e5.learner == "xgb") & (e5.arm == "fed_L0") & (e5.metric == "auroc")
            & (e5.partition == "dir0.5")]
    assert set(xg.weights) == {"size", "positives"}
    l1g = pd.read_csv(out / "S_L1g.csv")
    assert set(l1g.metric) == {"brier", "auroc", "mcc_pi"}
    assert set(l1g.learner) == {"rf", "xgb"}
    assert np.isfinite(l1g["mean"]).all()


def test_e12_full_matches_t2(report):
    _, out, _ = report
    e12 = pd.read_csv(out / "S_E12.csv")
    t2 = pd.read_csv(out / "T2_shares.csv")
    a = e12[(e12.n_cal == "full") & (e12.learner == "xgb") & (e12.partition == "equal")].r.iloc[0]
    b = t2[(t2.metric == "brier") & (t2.level == 2) & (t2.learner == "xgb")
           & (t2.partition == "equal")].r.iloc[0]
    assert a == pytest.approx(b)
    assert set(e12.n_cal.astype(str)) >= {"25", "full"}
