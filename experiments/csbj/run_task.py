"""One task of E1 and its variants: one dataset, repetition r and fold f.

    python experiments/csbj/run_task.py --dataset aids --rep 0 --fold 0
    python experiments/csbj/run_task.py --dataset heart --task-id 7      # r = 7 // 5, f = 7 % 5

Smoke runs only: ``--K 2 5 10`` restricts K, ``--subsample 20000`` keeps a
stratified subsample of the dataset. Neither is used in the real runs.

Outputs (RUNS and PREDS from environment variables, see config.py)

    $RUNS/<ds>/r<r>_f<f>/central.parquet           metric rows of the centralized models
    $RUNS/<ds>/r<r>_f<f>/central_sites.parquet     leaf statistics of the centralized models
    $RUNS/<ds>/r<r>_f<f>/<part>_K<K>.parquet       metric rows
    $RUNS/<ds>/r<r>_f<f>/<part>_K<K>_sites.parquet one row per site
    $RUNS/<ds>/r<r>_f<f>/meta.json                 versions, platform, seeds, sizes, time and memory per cell
    $PREDS/<ds>/r0_f0/*.parquet                    per row probabilities, only for r = 0, f = 0
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import config  # noqa: E402
import data as data_mod  # noqa: E402
from partition import partition, partition_rng, site_counts  # noqa: E402
from fed_rf_mk import ladder as L  # noqa: E402

METRIC_COLUMNS = [
    "dataset", "rep", "fold", "partition", "alpha", "K", "learner", "arm",
    "source", "weights", "sampling", "eval", "n_cal",
    "n_eval", "n_eval_pos", "n_k", "pos_k", "pi", "pi_k", "T", "rec_a", "rec_b",
    "auroc", "auprc", "brier", "mcc_pi", "mcc_05", "ppr_pi", "ppr_05",
    "citl_diff", "citl_int", "cal_slope",
]
METRICS = ["auroc", "auprc", "brier", "mcc_pi", "mcc_05", "ppr_pi", "ppr_05",
           "citl_diff", "citl_int", "cal_slope"]
EVALS = ("test", "shift_half", "shift_double")


# ----------------------------------------------------------------------------
# Draws, evaluation sets
# ----------------------------------------------------------------------------

def make_draw(y, r, f):
    from sklearn.model_selection import StratifiedKFold, train_test_split
    tr, de = list(StratifiedKFold(config.N_FOLDS, shuffle=True, random_state=r)
                  .split(np.zeros(len(y)), y))[f]
    cal, test = train_test_split(np.arange(len(de)), train_size=config.CAL_FRACTION,
                                 stratify=y[de], random_state=1000 * r + f)
    return tr, de, np.sort(cal), np.sort(test)


def _shift(pos, y_de, kind, rng):
    """Remove a random half (floor) of the positives (shift_half) or of the
    negatives (shift_double) of the positions ``pos``."""
    cls = 1 if kind == "shift_half" else 0
    members = pos[y_de[pos] == cls]
    drop = rng.choice(members, size=members.size // 2, replace=False)
    return np.setdiff1d(pos, drop)


def make_evals(y_de, cal, test, ds_code, r, f):
    """eval name -> (calibration positions, evaluation positions) inside de."""
    evals = {"test": (cal, test)}
    for kind, code in config.SHIFT_CODE.items():
        rng = np.random.default_rng([ds_code, r, f, config.SHIFT_TAG, code])
        c = _shift(cal, y_de, kind, rng)
        t = _shift(test, y_de, kind, rng)
        evals[kind] = (c, t)
    return evals


def make_ncal_sets(y_de, cal, ds_code, r, f):
    """E12. n_cal -> nested stratified subsets of the calibration part."""
    rng = np.random.default_rng([ds_code, r, f, config.NCAL_TAG])
    pos = cal[y_de[cal] == 1]
    neg = cal[y_de[cal] == 0]
    pos = pos[rng.permutation(pos.size)]
    neg = neg[rng.permutation(neg.size)]
    pi_cal = pos.size / cal.size
    out = {}
    for n_cal in config.N_CAL_LIST:
        if n_cal > cal.size:
            continue
        n_pos = max(2, int(round(n_cal * pi_cal)))
        n_neg = n_cal - n_pos
        if n_pos > pos.size or n_neg > neg.size or n_neg < 2:
            continue
        out[n_cal] = np.sort(np.concatenate([pos[:n_pos], neg[:n_neg]]))
    return out


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def compute_metrics(y, p, z, pi, score=None):
    """Metrics of one arm. ``score`` ranks the records for AUROC and AUPRC
    (default p). Level 2 rows pass sign(a) times the input of the
    recalibration (clipped probability for rf, log odds otherwise), which is
    the ranking of sigma(a z + b) in exact arithmetic. It avoids ties created
    when sigma(a z + b) rounds to 0 or 1, and differences between platforms in
    the last bit of logit."""
    from sklearn.metrics import (average_precision_score, brier_score_loss,
                                 matthews_corrcoef, roc_auc_score)
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    score = p if score is None else np.asarray(score, dtype=np.float64)
    yhat_pi = (p > pi).astype(int)
    yhat_05 = (p > 0.5).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mcc_pi = float(matthews_corrcoef(y, yhat_pi))
        mcc_05 = float(matthews_corrcoef(y, yhat_05))
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "brier": float(brier_score_loss(y, p)),
        "mcc_pi": mcc_pi, "mcc_05": mcc_05,
        "ppr_pi": float(yhat_pi.mean()), "ppr_05": float(yhat_05.mean()),
        "citl_diff": float(p.mean() - y.mean()),
        "citl_int": float(L.citl_intercept(z, y)),
        "cal_slope": float(L.calibration_slope(z, y)),
    }


def rf_logit(p, T):
    return L.logit(L.clip_prob(p, T))


class Scorer:
    """Turns model outputs on the whole deployment site into metric rows."""

    def __init__(self, base, y_de, evals, ncal_sets, pi):
        self.base, self.y_de, self.evals, self.ncal_sets, self.pi = base, y_de, evals, ncal_sets, pi
        self.rows = []
        self.n_nonpos_a = 0

    def _row(self, info, ev, n_cal, eval_pos, p, z, a=np.nan, b=np.nan, rank=None):
        y = self.y_de[eval_pos]
        row = dict(self.base)
        row.update(info)
        row.update(eval=ev, n_cal=int(n_cal), n_eval=int(eval_pos.size),
                   n_eval_pos=int(y.sum()), pi=self.pi, rec_a=a, rec_b=b)
        score = None if rank is None else np.sign(a) * rank[eval_pos]
        row.update(compute_metrics(y, p[eval_pos], z[eval_pos], self.pi, score))
        self.rows.append(row)

    def levels(self, info, levels, z_in, evals, e12=False, prefix="", l2=True, rank_in=None):
        """``levels`` maps arm suffix (L0, L1, L1g) to (p, z) over de.
        Level 2 is fitted on ``z_in``. Returns the level 2 probabilities over
        de fitted on the default calibration part. ``rank_in`` is a score with
        the order of z_in (default z_in), used to rank level 2 rows."""
        p2_default = None
        rank_in = z_in if rank_in is None else rank_in
        for ev in evals:
            cal_pos, test_pos = self.evals[ev]
            for name, (p, z) in levels.items():
                self._row({**info, "arm": f"{prefix}_{name}"}, ev, 0, test_pos, p, z)
            if not l2:
                continue
            a, b = L.fit_recalibration(z_in[cal_pos], self.y_de[cal_pos])
            self.n_nonpos_a += int(a <= 0)
            z2 = a * z_in + b
            p2 = L.sigmoid(z2)
            self._row({**info, "arm": f"{prefix}_L2"}, ev, cal_pos.size, test_pos, p2, z2, a, b,
                      rank=rank_in)
            if ev == "test":
                p2_default = p2
        if e12:
            test_pos = self.evals["test"][1]
            for n_cal, sub in self.ncal_sets.items():
                a, b = L.fit_recalibration(z_in[sub], self.y_de[sub])
                self.n_nonpos_a += int(a <= 0)
                z2 = a * z_in + b
                self._row({**info, "arm": f"{prefix}_L2"}, "test", n_cal, test_pos,
                          L.sigmoid(z2), z2, a, b, rank=rank_in)
        return p2_default

    def frame(self):
        return pd.DataFrame(self.rows, columns=METRIC_COLUMNS)


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------

def make_rf(seed, hp="default"):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(random_state=seed, **config.HP[hp]["rf"])


def make_xgb(seed, hp="default"):
    from xgboost import XGBClassifier
    return XGBClassifier(random_state=seed, **config.HP[hp]["xgb"])


def fit_rf(X, y, seed, X_de32, hp="default", with_oob=True):
    rf = make_rf(seed, hp).fit(X, y)
    assert rf.classes_.tolist() == [0, 1], rf.classes_
    probs = np.stack([t.predict_proba(X_de32, check_input=False)[:, 1]
                      for t in rf.estimators_]).astype(np.float32)
    oob = L.tree_oob_scores(rf, X, y) if with_oob else np.full(len(rf.estimators_), np.nan)
    leaf = L.single_record_leaf_fraction(rf, X)
    del rf
    return probs, oob, leaf


def fit_xgb(X, y, seed, X_de, hp="default"):
    m = make_xgb(seed, hp).fit(X, y)
    assert m.classes_.tolist() == [0, 1], m.classes_
    z = np.asarray(m.predict(X_de, output_margin=True), dtype=np.float64)
    leaf = L.xgb_single_record_leaf_fraction(m, X)
    del m
    return z, leaf


def fit_lr(X, y, seed, X_de, nominal):
    m = data_mod.lr_pipeline(nominal, seed).fit(X, y)
    assert m.classes_.tolist() == [0, 1], m.classes_
    z = np.asarray(m.decision_function(X_de), dtype=np.float64)
    del m
    return z


def hyperparameters(nominal, hp="default"):
    return {
        "hp": hp,
        "rf": make_rf(0, hp).get_params(),
        "xgb": make_xgb(0, hp).get_params(),
        "lr": data_mod.lr_pipeline(nominal, 0).named_steps["lr"].get_params(),
        "seed_note": "random_state is set per model with seed_for(...), see config.py",
    }


# ----------------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------------

def write_parquet(df, path):
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def write_json(obj, path):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default))
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if callable(o) or isinstance(o, type):
        return repr(o)
    return str(o)


def _reset_peak():
    try:
        with open("/proc/self/clear_refs", "w") as fh:
            fh.write("5")
        return True
    except OSError:
        return False


def _peak_mb(reset_ok):
    if reset_ok:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0, "VmHWM of the cell"
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mb = ru / 1024.0 / 1024.0 if sys.platform == "darwin" else ru / 1024.0
    return mb, "ru_maxrss of the worker process (not reset between cells)"


def _warning_summary(caught):
    c = Counter(f"{w.category.__name__}: {str(w.message)[:160]}" for w in caught)
    return dict(c)


# ----------------------------------------------------------------------------
# Central models and cells
# ----------------------------------------------------------------------------

def base_info(ds, r, f, part, K):
    alpha = config.PARTITIONS[part][1] if part in config.PARTITIONS else None
    return {"dataset": ds, "rep": r, "fold": f, "partition": part,
            "alpha": np.nan if alpha is None else alpha, "K": K}


def run_central(ctx):
    ds, r, f = ctx["ds"], ctx["r"], ctx["f"]
    X, y, tr, de = ctx["X"], ctx["y"], ctx["tr"], ctx["de"]
    dsc = config.DS_CODE[ds]
    pi = ctx["pi"]
    Xtr, ytr = X[tr], y[tr]
    X_de = X[de]
    X_de32 = np.ascontiguousarray(X_de, dtype=np.float32)
    seed = {lc: config.seed_for(dsc, r, f, config.CENTRAL_PART_CODE, code)
            for lc, code in config.LEARNER_CODE.items()}
    hp, evals = ctx["hp"], ctx["eval_names"]
    sc = Scorer(base_info(ds, r, f, "central", 0), ctx["y_de"], ctx["evals"], {}, pi)
    info = {"source": -1, "weights": "na", "sampling": "na",
            "n_k": int(tr.size), "pos_k": int(ytr.sum()), "pi_k": pi}
    preds = {}
    probs, _, rf_leaf = fit_rf(Xtr, ytr, seed["rf"], X_de32, hp, with_oob=False)
    p = probs.mean(axis=0, dtype=np.float64)
    sc.levels({**info, "learner": "rf", "T": 100}, {"L0": (p, rf_logit(p, 100))},
              rf_logit(p, 100), evals, prefix="central", rank_in=L.clip_prob(p, 100))
    preds["rf_central_L0"] = p
    del probs
    z, xgb_leaf = fit_xgb(Xtr, ytr, seed["xgb"], X_de, hp)
    sc.levels({**info, "learner": "xgb", "T": 100}, {"L0": (L.sigmoid(z), z)}, z, evals,
              prefix="central")
    preds["xgb_central_L0"] = L.sigmoid(z)
    z = fit_lr(Xtr, ytr, seed["lr"], X_de, ctx["nominal"])
    sc.levels({**info, "learner": "lr", "T": 0}, {"L0": (L.sigmoid(z), z)}, z, evals,
              prefix="central")
    preds["lr_central_L0"] = L.sigmoid(z)
    sites = pd.DataFrame([{
        "dataset": ds, "rep": r, "fold": f, "n": int(tr.size), "n_pos": int(ytr.sum()),
        "rf_leaf_single": rf_leaf["n_single"], "rf_leaf_n": rf_leaf["n_leaves"],
        "rf_leaf_frac": rf_leaf["fraction"], "rf_leaf_lt5": rf_leaf["n_lt5"],
        "rf_leaf_frac_lt5": rf_leaf["fraction_lt5"], "xgb_leaf_single": xgb_leaf["n_single"],
        "xgb_leaf_n": xgb_leaf["n_leaves"], "xgb_leaf_frac": xgb_leaf["fraction"],
        "xgb_leaf_lt5": xgb_leaf["n_lt5"], "xgb_leaf_frac_lt5": xgb_leaf["fraction_lt5"],
    }])
    return sc, sites, preds, seed


def run_cell(ctx, part, K):
    ds, r, f = ctx["ds"], ctx["r"], ctx["f"]
    X, y, tr, de, y_de, pi = ctx["X"], ctx["y"], ctx["tr"], ctx["de"], ctx["y_de"], ctx["pi"]
    dsc = config.DS_CODE[ds]
    pcode = config.PARTITIONS[part][0]
    site = partition(y[tr], K, part, partition_rng(dsc, r, f, K, pcode))
    n_k, pos_k = site_counts(y[tr], site, K)
    pi_k = pos_k / n_k
    X_de = X[de]
    X_de32 = np.ascontiguousarray(X_de, dtype=np.float32)
    hp, evals, main_only = ctx["hp"], ctx["eval_names"], ctx["main_only"]
    e12 = not main_only

    rf_probs, rf_oob, rf_leaf, xgb_z, xgb_leaf, lr_z = [], [], [], [], [], []
    for k in range(K):
        rows = tr[site == k]
        Xk, yk = X[rows], y[rows]
        seeds = {lc: config.seed_for(dsc, r, f, pcode, K, k, code)
                 for lc, code in config.LEARNER_CODE.items()}
        probs, oob, leaf = fit_rf(Xk, yk, seeds["rf"], X_de32, hp, with_oob=not main_only)
        rf_probs.append(probs)
        rf_oob.append(oob)
        rf_leaf.append(leaf)
        z, leaf = fit_xgb(Xk, yk, seeds["xgb"], X_de, hp)
        xgb_z.append(z)
        xgb_leaf.append(leaf)
        lr_z.append(fit_lr(Xk, yk, seeds["lr"], X_de, ctx["nominal"]))
    xgb_z = np.stack(xgb_z)
    lr_z = np.stack(lr_z)

    sc = Scorer(base_info(ds, r, f, part, K), y_de, ctx["evals"], ctx["ncal_sets"], pi)
    preds = {}
    # source arms
    for k in range(K):
        src = {"source": k, "weights": "na", "sampling": "na",
               "n_k": int(n_k[k]), "pos_k": int(pos_k[k]), "pi_k": float(pi_k[k])}
        p0 = rf_probs[k].mean(axis=0, dtype=np.float64)
        p1 = L.shift_prob(p0, pi, pi_k[k])
        lv = {"L0": (p0, rf_logit(p0, 100)), "L1": (p1, rf_logit(p1, 100))}
        p2 = sc.levels({**src, "learner": "rf", "T": 100}, lv, rf_logit(p0, 100), evals,
                       e12=e12, prefix="src", rank_in=L.clip_prob(p0, 100))
        preds.update({f"rf_src{k}_L0": p0, f"rf_src{k}_L1": p1, f"rf_src{k}_L2": p2})
        for lname, Z in (("xgb", xgb_z), ("lr", lr_z)):
            z0 = Z[k]
            z1 = L.shift_logit(z0, pi, pi_k[k])
            lv = {"L0": (L.sigmoid(z0), z0), "L1": (L.sigmoid(z1), z1)}
            p2 = sc.levels({**src, "learner": lname, "T": 100 if lname == "xgb" else 0},
                           lv, z0, evals, e12=e12, prefix="src")
            preds.update({f"{lname}_src{k}_L0": L.sigmoid(z0),
                          f"{lname}_src{k}_L1": L.sigmoid(z1), f"{lname}_src{k}_L2": p2})

    # federated arms
    site_extra = {}
    weight_kinds = [config.DEFAULT_WEIGHTS] if main_only else list(config.WEIGHT_CODE)
    sampling_kinds = [config.DEFAULT_SAMPLING] if main_only else list(config.SAMPLING_CODE)
    for wname in weight_kinds:
        wcode = config.WEIGHT_CODE[wname]
        w = L.site_weights(n_k, pos_k, wname)
        t = L.allocate_trees(w, np.full(K, 100))
        site_extra[f"t_{wname}"] = t
        fed = {"source": -1, "weights": wname, "n_k": int(n_k.sum()), "pos_k": int(pos_k.sum())}
        for sname in sampling_kinds:
            scode = config.SAMPLING_CODE[sname]
            rng = np.random.default_rng([dsc, r, f, pcode, K, config.TREE_SELECT_TAG, wcode, scode])
            sel, fb = [], []
            for k in range(K):
                s, fallback = L.select_trees(100, t[k], rng, rf_oob[k] if sname == "oob" else None)
                sel.append(s)
                fb.append(fallback)
            if sname == "oob":
                site_extra[f"oob_fallback_{wname}"] = np.array(fb)
            T = int(t.sum())
            pi_fed = float(np.sum(t / T * pi_k))
            p0 = L.rf_federated(rf_probs, sel)
            p1 = L.rf_federated_shift(rf_probs, sel, pi, pi_k)
            p1g = L.rf_federated_global_shift(rf_probs, sel, pi, pi_k)
            default = wname == config.DEFAULT_WEIGHTS and sname == config.DEFAULT_SAMPLING
            lv = {"L0": (p0, rf_logit(p0, T)), "L1": (p1, rf_logit(p1, T))}
            info = {**fed, "learner": "rf", "sampling": sname, "T": T, "pi_k": pi_fed}
            if default:
                p2 = sc.levels(info, lv, rf_logit(p0, T), evals, e12=e12, prefix="fed",
                               rank_in=L.clip_prob(p0, T))
                preds.update({"rf_fed_L0": p0, "rf_fed_L1": p1, "rf_fed_L2": p2})
            else:
                sc.levels(info, lv, rf_logit(p0, T), ("test",), prefix="fed",
                          rank_in=L.clip_prob(p0, T))
            sc.levels(info, {"L1g": (p1g, rf_logit(p1g, T))}, None, ("test",),
                      prefix="fed", l2=False)
        z0 = L.xgb_federated(xgb_z, w)
        z1 = L.xgb_federated_shift(xgb_z, w, pi, pi_k)
        z1g = L.xgb_federated_global_shift(xgb_z, w, pi, pi_k)
        info = {**fed, "learner": "xgb", "sampling": "na", "T": 100 * int(np.count_nonzero(w > 0)),
                "pi_k": float(np.sum(w * pi_k))}
        lv = {"L0": (L.sigmoid(z0), z0), "L1": (L.sigmoid(z1), z1)}
        if wname == config.DEFAULT_WEIGHTS:
            p2 = sc.levels(info, lv, z0, evals, e12=e12, prefix="fed")
            preds.update({"xgb_fed_L0": L.sigmoid(z0), "xgb_fed_L1": L.sigmoid(z1), "xgb_fed_L2": p2})
        else:
            sc.levels(info, lv, z0, ("test",), prefix="fed")
        sc.levels(info, {"L1g": (L.sigmoid(z1g), z1g)}, None, ("test",), prefix="fed", l2=False)

    sites = pd.DataFrame({
        "dataset": ds, "rep": r, "fold": f, "partition": part,
        "alpha": base_info(ds, r, f, part, K)["alpha"], "K": K, "site": np.arange(K),
        "n_k": n_k, "pos_k": pos_k, "pi_k": pi_k,
        "rf_leaf_single": [d["n_single"] for d in rf_leaf],
        "rf_leaf_n": [d["n_leaves"] for d in rf_leaf],
        "rf_leaf_frac": [d["fraction"] for d in rf_leaf],
        "rf_leaf_lt5": [d["n_lt5"] for d in rf_leaf],
        "rf_leaf_frac_lt5": [d["fraction_lt5"] for d in rf_leaf],
        "xgb_leaf_single": [d["n_single"] for d in xgb_leaf],
        "xgb_leaf_n": [d["n_leaves"] for d in xgb_leaf],
        "xgb_leaf_frac": [d["fraction"] for d in xgb_leaf],
        "xgb_leaf_lt5": [d["n_lt5"] for d in xgb_leaf],
        "xgb_leaf_frac_lt5": [d["fraction_lt5"] for d in xgb_leaf],
        "rf_oob_mean": [float(np.nanmean(o)) if np.isfinite(o).any() else np.nan for o in rf_oob],
        "rf_oob_nan": [int(np.isnan(o).sum()) for o in rf_oob],
        **site_extra,
    })
    return sc, sites, preds


def _preds_frame(ctx, preds):
    de, cal = ctx["de"], ctx["cal"]
    part = np.zeros(de.size, dtype=np.int8)
    part[cal] = 1
    cols = {"row": de, "y": ctx["y_de"].astype(np.int8), "is_cal": part}
    cols.update({k: np.asarray(v, dtype=np.float32) for k, v in preds.items()})
    return pd.DataFrame(cols)


def cell_paths(out_dir, part, K):
    stem = f"{part}_K{K}"
    return (out_dir / f"{stem}.parquet", out_dir / f"{stem}_sites.parquet",
            out_dir / f"{stem}.error.txt")


def cell_job(ctx, part, K, force):
    """Run one cell in a worker. Returns a status dict, never raises."""
    out_dir = ctx["out_dir"]
    metric_path, sites_path, err_path = cell_paths(out_dir, part, K)
    pred_path = ctx["pred_dir"] / f"{part}_K{K}.parquet" if ctx["pred_dir"] else None
    name = f"{part}_K{K}"
    if not force and metric_path.exists() and sites_path.exists() and \
            (pred_path is None or pred_path.exists()):
        return {"cell": name, "status": "skipped"}
    reset_ok = _reset_peak()
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sc, sites, preds = run_cell(ctx, part, K)
        sites["n_nonpos_a"] = sc.n_nonpos_a
        if pred_path is not None:
            write_parquet(_preds_frame(ctx, preds), pred_path)
        write_parquet(sites, sites_path)
        write_parquet(sc.frame(), metric_path)
        if err_path.exists():
            err_path.unlink()
        peak, how = _peak_mb(reset_ok)
        return {"cell": name, "status": "done", "time_s": time.perf_counter() - t0,
                "peak_mb": peak, "peak_how": how, "rows": len(sc.rows),
                "n_nonpos_a": sc.n_nonpos_a, "warnings": _warning_summary(caught)}
    except Exception:
        err_path.write_text(traceback.format_exc())
        return {"cell": name, "status": "error", "time_s": time.perf_counter() - t0,
                "error": traceback.format_exc().splitlines()[-1]}


# ----------------------------------------------------------------------------
# Task
# ----------------------------------------------------------------------------

def versions():
    import joblib
    import pyarrow
    import scipy
    import sklearn
    import xgboost
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__, "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__, "joblib": joblib.__version__}


def stratified_subsample(y, n, seed=0):
    from sklearn.model_selection import train_test_split
    idx, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=seed)
    return np.sort(idx)


def build_context(ds_name, r, f, subsample=None, X=None, y=None, nominal=None, hp="default"):
    if X is None:
        ds = data_mod.load(ds_name)
        X, y, nominal = ds.X, ds.y, ds.nominal
    if subsample:
        keep = stratified_subsample(y, subsample)
        X, y = X[keep], y[keep]
    dsc = config.DS_CODE[ds_name]
    tr, de, cal, test = make_draw(y, r, f)
    y_de = y[de]
    return {
        "ds": ds_name, "r": r, "f": f, "X": X, "y": y, "nominal": nominal or {},
        "tr": tr, "de": de, "cal": cal, "test": test, "y_de": y_de,
        "pi": float(y[tr].mean()),
        "evals": make_evals(y_de, cal, test, dsc, r, f),
        "ncal_sets": make_ncal_sets(y_de, cal, dsc, r, f),
        "hp": hp, "main_only": hp != "default",
        "eval_names": EVALS if hp == "default" else ("test",),
    }


def run_task(ds_name, r, f, K_list=config.K_LIST, partitions=tuple(config.PARTITIONS),
             subsample=None, force=False, n_jobs=None, runs=None, preds=None,
             X=None, y=None, nominal=None, write_hyper=True, hp="default"):
    """Run one task. With ``hp`` other than "default" only the main arms are
    scored (equal weights, uniform sampling, eval test, full calibration part)
    and no per row predictions are written."""
    if hp not in config.HP:
        raise ValueError(f"unknown hp configuration {hp!r}")
    from joblib import Parallel, delayed
    t_task = time.perf_counter()
    runs = Path(runs) if runs else config.runs_dir(hp)
    preds_root = Path(preds) if preds else config.preds_dir()
    out_dir = runs / ds_name / f"r{r}_f{f}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = None
    if r == 0 and f == 0 and hp == "default":
        pred_dir = preds_root / ds_name / "r0_f0"
        pred_dir.mkdir(parents=True, exist_ok=True)
    ctx = build_context(ds_name, r, f, subsample, X, y, nominal, hp)
    ctx["out_dir"], ctx["pred_dir"] = out_dir, pred_dir
    if n_jobs is None:
        n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.setdefault("cells", {})

    if write_hyper:
        name = "hyperparameters.json" if hp == "default" else f"hyperparameters_{hp}.json"
        hp_path = config.results_dir() / name
        hp_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(hyperparameters(ctx["nominal"], hp), hp_path)

    status = []
    c_path = out_dir / "central.parquet"
    c_sites = out_dir / "central_sites.parquet"
    c_err = out_dir / "central.error.txt"
    if force or not (c_path.exists() and c_sites.exists()):
        reset_ok = _reset_peak()
        t0 = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                sc, sites, cpreds, cseeds = run_central(ctx)
            if pred_dir is not None:
                write_parquet(_preds_frame(ctx, cpreds), pred_dir / "central.parquet")
            write_parquet(sites, c_sites)
            write_parquet(sc.frame(), c_path)
            if c_err.exists():
                c_err.unlink()
            peak, how = _peak_mb(reset_ok)
            meta["cells"]["central"] = {"status": "done", "time_s": time.perf_counter() - t0,
                                        "peak_mb": peak, "peak_how": how,
                                        "warnings": _warning_summary(caught)}
            meta["central_seeds"] = cseeds
        except Exception:
            c_err.write_text(traceback.format_exc())
            meta["cells"]["central"] = {"status": "error",
                                        "error": traceback.format_exc().splitlines()[-1]}
        status.append(meta["cells"]["central"]["status"])
    else:
        status.append("skipped")

    jobs = [(part, K) for part in partitions for K in K_list]
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(cell_job)(ctx, part, K, force) for part, K in jobs)
    for res in results:
        if res["status"] != "skipped":
            meta["cells"][res["cell"]] = res
        status.append(res["status"])

    meta.update({
        "dataset": ds_name, "rep": r, "fold": f, "hp": hp,
        "hyperparameters": {k: dict(v) for k, v in config.HP[hp].items()},
        "main_only": ctx["main_only"], "versions": versions(),
        "platform": {"platform": platform.platform(), "machine": platform.machine(),
                     "processor": platform.processor(), "host": socket.gethostname(),
                     "cpu_count": os.cpu_count(), "n_jobs": n_jobs,
                     "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                     "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID")},
        "sizes": {"n": int(len(ctx["y"])), "tr": int(ctx["tr"].size), "de": int(ctx["de"].size),
                  "cal": int(ctx["cal"].size), "test": int(ctx["test"].size),
                  "tr_pos": int(ctx["y"][ctx["tr"]].sum()),
                  "cal_pos": int(ctx["y_de"][ctx["cal"]].sum()),
                  "test_pos": int(ctx["y_de"][ctx["test"]].sum()),
                  "evals": {k: [int(c.size), int(t.size)] for k, (c, t) in ctx["evals"].items()},
                  "n_cal_subsets": sorted(int(k) for k in ctx["ncal_sets"])},
        "pi": ctx["pi"], "subsample": subsample, "K_list": list(K_list),
        "partitions": list(partitions),
        "seed_rules": {
            "site_model": "seed_for(ds_code, r, f, part_code, K, k, learner_code)",
            "central_model": "seed_for(ds_code, r, f, 99, learner_code)",
            "partition": "default_rng([ds_code, r, f, K, part_code])",
            "tree_selection": "default_rng([ds_code, r, f, part_code, K, 7, w_code, s_code])",
            "shift": "default_rng([ds_code, r, f, 8, shift_code])",
            "n_cal": "default_rng([ds_code, r, f, 9])",
            "draw": "StratifiedKFold(5, shuffle=True, random_state=r), "
                    "train_test_split(random_state=1000 r + f)"},
        "task_time_s_last_call": time.perf_counter() - t_task,
    })
    write_json(meta, meta_path)
    n_err = sum(s == "error" for s in status)
    return {"status": status, "n_error": n_err, "out_dir": str(out_dir), "meta": meta}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=config.DATASETS)
    ap.add_argument("--rep", type=int)
    ap.add_argument("--fold", type=int)
    ap.add_argument("--task-id", type=int, help="r = id // 5, f = id %% 5")
    ap.add_argument("--K", type=int, nargs="+", default=list(config.K_LIST), help="smoke runs only")
    ap.add_argument("--partitions", nargs="+", default=list(config.PARTITIONS))
    ap.add_argument("--subsample", type=int, default=None, help="smoke runs only")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--hp", default="default", choices=list(config.HP),
                    help="regularized: sensitivity run, main arms only, separate run folder")
    a = ap.parse_args(argv)
    if a.task_id is not None:
        r, f = divmod(a.task_id, config.N_FOLDS)
    elif a.rep is not None and a.fold is not None:
        r, f = a.rep, a.fold
    else:
        ap.error("give --task-id or both --rep and --fold")
    if not 0 <= f < config.N_FOLDS:
        ap.error("fold must be in 0..4")
    for K in a.K:
        if K < 2:
            ap.error("K must be at least 2")
    for p in a.partitions:
        if p not in config.PARTITIONS:
            ap.error(f"unknown partition {p}")
    res = run_task(a.dataset, r, f, tuple(a.K), tuple(a.partitions), a.subsample, a.force,
                   a.n_jobs, hp=a.hp)
    cells = res["meta"]["cells"]
    for name, c in sorted(cells.items()):
        extra = f" {c.get('time_s', 0):.1f}s {c.get('peak_mb', 0):.0f}MB" if c.get("status") == "done" else ""
        print(f"{a.dataset} r{r} f{f} {name}: {c.get('status')}{extra} {c.get('error', '')}")
    print(f"errors: {res['n_error']}, output {res['out_dir']}")
    return 1 if res["n_error"] else 0


if __name__ == "__main__":
    sys.exit(main())
