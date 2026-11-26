"""Tables T1 to T3, supplementary tables, numbers.json and numbers.md.

Called by ``analyze.py report``. Every value comes from the run files. The
logistic regression learner (E13) is computed by the runs but not reported.

Outputs in the results folder

    T1_levels.csv / T1_brier.tex / T1_auroc.tex       seven models, mean [95% CI]
    T2_shares.csv / T2_shares.tex                     r_1, r_2 for brier and mcc_pi (mcc_05 in the csv),
                                                      with the denominator mean(m3l - m0) and its interval
    T3_auroc_gain.csv / T3_auroc_gain.tex             AUROC gain of level 3 per K, n/K, skew
    perK_differences.csv                              m - m0 per K with intervals (figures)
    S_E5.csv / S_E5.tex                               weights and sampling against the default
    S_L1g.csv / S_L1g.tex                             fed_L1g against fed_L1
    S_E9_leaves.csv / S_E9_leaves.tex                 leaf fractions
    S_E10.csv / S_E10.tex                             shares under shifted prevalence
    S_E11.csv / S_E11.tex                             largest site as the source
    S_E12.csv / S_E12.tex                             r_2 as a function of n_cal
    S_hp.csv / S_hp.tex                               default against regularized, common draws
    S_calibration.csv                                 calibration slope and intercept per level
    S_L2_nonpositive.csv / .tex                       level 2 fits with a <= 0 and AUROC change
    S_rfxgb.csv                                       rf minus xgb for fed_L1 and fed_L2
    sites_r0_f0_K10.csv                               site prevalences for the figure
    reliability.csv                                   reliability curves, r0 f0 K5 dir0.5
    anomalies.csv
    numbers.json / numbers.md
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import analyze as A
import config

MAIN_LEARNERS = ("rf", "xgb")
PARTS = ("equal", "dir0.5", "dir0.1")
T1_LEVELS = ["m0", "m1", "m2", "m30", "m31", "m32", "mc"]
LEVEL_LABEL = {"m0": "L0", "m1": "L1", "m2": "L2", "m30": "L3", "m31": "L3+L1",
               "m31g": "L3+L1 global", "m32": "L3+L2", "mc": "Central", "mc2": "Central+L2"}
DIFF_LEVELS = ["m1", "m2", "m30", "m31", "m32", "mc"]
EVAL_SHIFTS = ("shift_half", "shift_double")


# ----------------------------------------------------------------------------
# Views and caching
# ----------------------------------------------------------------------------

def ncal_view(df, n):
    """Default configuration, eval test, level 2 rows fitted on n_cal = n
    (n="full" keeps the full calibration part). Central level 2 has the full
    part only and is dropped for subsets."""
    if n == "full":
        return A.default_view(df)
    v = df[(df["eval"] == "test") & df.weights.isin(["na", config.DEFAULT_WEIGHTS])
           & df.sampling.isin(["na", config.DEFAULT_SAMPLING])]
    l2 = v.arm.str.endswith("_L2")
    v = v[(~l2) | (v.n_cal == n)]
    return v[v.arm != "central_L2"]


class Runs:
    """Metric, site and central site frames with cached level tables."""

    def __init__(self, metrics, sites, csites):
        for c in ("dataset", "partition", "learner", "arm", "weights", "sampling", "eval"):
            if c in metrics:
                metrics[c] = metrics[c].astype(str)
        self.m, self.sites, self.csites = metrics, sites, csites
        self._views, self._cells = {}, {}

    def view(self, name="test"):
        if name not in self._views:
            if name == "test":
                self._views[name] = A.default_view(self.m)
            elif name in EVAL_SHIFTS:
                self._views[name] = A.default_view(self.m, name)
            else:  # ("ncal", n)
                self._views[name] = ncal_view(self.m, name[1])
        return self._views[name]

    def cells(self, metric, view="test", source="mean"):
        key = (metric, view, source)
        if key not in self._cells:
            self._cells[key] = A.levels_per_cell(self.view(view), metric, source)
        return self._cells[key]

    def per_draw(self, metric, view="test", source="mean"):
        return A.average_over_K(self.cells(metric, view, source))

    def restrict(self, draws):
        return Runs(*(A._restrict(x, draws) for x in (self.m, self.sites, self.csites)))


def _main(df):
    return df[df.learner.isin(MAIN_LEARNERS)]


def _ci_cols(ci, prefix=""):
    return {f"{prefix}{k}": ci[k] for k in ("mean", "lo", "hi", "nb_lo", "nb_hi", "J")}


# ----------------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------------

def fmt(x, d=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "NA"
    return f"{x:.{d}f}"


def fmt_ci(m, lo, hi, d=3):
    if not np.isfinite(m):
        return "NA"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return fmt(m, d)
    return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def _tex_escape(x):
    return (str(x).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")
            .replace("&", r"\&").replace("#", r"\#"))


SAME = object()


def write_table(df, out, stem, tex_df=SAME, col_format=None):
    """CSV of ``df`` and booktabs LaTeX of ``tex_df`` (``df`` itself by
    default, no LaTeX when None)."""
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{stem}.csv", index=False)
    t = df if tex_df is SAME else tex_df
    if t is None or not len(t):
        return
    cols = list(t.columns)
    fmt_cols = col_format or ("l" * len(cols))
    lines = [r"\begin{tabular}{" + fmt_cols + "}", r"\toprule",
             " & ".join(_tex_escape(c) for c in cols) + r" \\", r"\midrule"]
    prev = None
    for row in t.itertuples(index=False):
        if prev is not None and row[0] != prev:
            lines.append(r"\addlinespace")
        prev = row[0]
        lines.append(" & ".join(_tex_escape(x) for x in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (out / f"{stem}.tex").write_text("\n".join(lines))


def _sort(df, extra=()):
    po = {p: i for i, p in enumerate(PARTS)}
    d = df.copy()
    d["_p"] = d.partition.map(po)
    return d.sort_values(["dataset", "learner", "_p", *extra]).drop(columns="_p")


# ----------------------------------------------------------------------------
# T1, T2, T3 and per K differences
# ----------------------------------------------------------------------------

def table_T1(R):
    rows = []
    for metric in ("brier", "auroc"):
        pdw = _main(R.per_draw(metric))
        for (ds, lr, part), g in pdw.groupby(["dataset", "learner", "partition"]):
            for lv in T1_LEVELS:
                d = A.mean_ci(g[lv] - g["m0"]) if lv != "m0" else dict.fromkeys(
                    ("mean", "lo", "hi", "nb_lo", "nb_hi", "J"), np.nan)
                rows.append(dict(dataset=ds, learner=lr, partition=part, metric=metric,
                                 level=lv, label=LEVEL_LABEL[lv], **_ci_cols(A.mean_ci(g[lv])),
                                 **_ci_cols(d, "diff_")))
    return _sort(pd.DataFrame(rows))


def tex_T1(t1, metric):
    t = t1[t1.metric == metric]
    rows = []
    for (ds, lr, part), g in t.groupby(["dataset", "learner", "partition"], sort=False):
        r = dict(Dataset=ds, Learner=lr, Partition=part)
        for _, q in g.iterrows():
            r[q.label] = fmt_ci(q["mean"], q.lo, q.hi)
        rows.append(r)
    return pd.DataFrame(rows)


def table_T2(R):
    out = [A.shares(R.per_draw(m), m) for m in ("brier", "mcc_pi", "mcc_05")]
    t = _main(pd.concat(out, ignore_index=True))
    return _sort(t, ["metric", "level"])


def tex_T2(t2):
    rows = []
    t = t2[t2.metric.isin(["brier", "mcc_pi"])]
    for (ds, lr, part), g in t.groupby(["dataset", "learner", "partition"], sort=False):
        r = dict(Dataset=ds, Learner=lr, Partition=part)
        for metric, name in (("brier", "Brier"), ("mcc_pi", "MCC")):
            for lvl in (1, 2):
                q = g[(g.metric == metric) & (g.level == lvl)]
                if len(q):
                    q = q.iloc[0]
                    r[f"{name} r{lvl}"] = fmt_ci(q.r, q.r_lo, q.r_hi, 2)
                    r[f"{name} r{lvl} den."] = fmt_ci(q.den, q.den_lo, q.den_hi, 4)
                    r[f"{name} r{lvl} rep."] = "yes" if q.reportable else "no"
        rows.append(r)
    return pd.DataFrame(rows)


def site_geometry(sites):
    """Per cell: n, n/K and skew sum_k (n_k/n) |pi_k - pi| with pi the pooled
    prevalence of the training sites."""
    key = ["dataset", "partition", "K"] + A.DRAW
    s = sites[key + ["n_k", "pos_k", "pi_k"]].copy()
    g = s.groupby(key)
    n = g.n_k.transform("sum")
    pi = g.pos_k.transform("sum") / n
    s["skew"] = s.n_k / n * (s.pi_k - pi).abs()
    cell = s.groupby(key).agg(n=("n_k", "sum"), skew=("skew", "sum")).reset_index()
    cell["n_over_K"] = cell.n / cell.K
    return cell


def table_T3(R):
    c = _main(R.cells("auroc")).copy()
    c["gain"] = c.m30 - c.m0
    geo = site_geometry(R.sites).groupby(["dataset", "partition", "K"])[
        ["n_over_K", "skew"]].mean().reset_index()
    rows = []
    for (ds, lr, part, K), g in c.groupby(["dataset", "learner", "partition", "K"]):
        rows.append(dict(dataset=ds, learner=lr, partition=part, K=int(K),
                         **_ci_cols(A.mean_ci(g.gain))))
    t = pd.DataFrame(rows).merge(geo, on=["dataset", "partition", "K"], how="left")
    return _sort(t, ["K"])


def tex_T3(t3):
    t = t3.assign(gain=[fmt_ci(m, lo, hi) for m, lo, hi in zip(t3["mean"], t3.lo, t3.hi)])
    t = t.assign(nK=t3.n_over_K.round(0).astype("Int64").astype(str), skew=t3["skew"].map(fmt))
    return t[["dataset", "learner", "partition", "K", "gain", "nK", "skew"]].rename(columns={
        "dataset": "Dataset", "learner": "Learner", "partition": "Partition",
        "gain": "AUROC gain L3", "nK": "n/K", "skew": "Skew"})


def perK_differences(R, metrics=("brier", "auroc", "mcc_pi")):
    rows = []
    for metric in metrics:
        c = _main(R.cells(metric))
        for (ds, lr, part, K), g in c.groupby(["dataset", "learner", "partition", "K"]):
            for lv in ["m0"] + DIFF_LEVELS:
                ci = A.mean_ci(g[lv]) if lv == "m0" else A.mean_ci(g[lv] - g["m0"])
                rows.append(dict(dataset=ds, learner=lr, partition=part, K=int(K),
                                 metric=metric, level=lv,
                                 kind="level" if lv == "m0" else "diff", **_ci_cols(ci)))
    return _sort(pd.DataFrame(rows), ["metric", "K", "level"])


# ----------------------------------------------------------------------------
# Supplementary tables
# ----------------------------------------------------------------------------

def table_E5(R, metrics=("brier", "auroc", "mcc_pi")):
    """Every weight and sampling combination against the default (equal,
    uniform), per federated arm, as paired differences per draw after
    averaging over K."""
    m = R.m
    v = m[(m["eval"] == "test") & m.arm.isin(["fed_L0", "fed_L1", "fed_L2"])
          & m.learner.isin(MAIN_LEARNERS)]
    l2 = v.arm == "fed_L2"
    full = v[l2].groupby(["dataset"] + A.DRAW).n_cal.transform("max")
    keep = ~l2
    keep[l2] = v.loc[l2, "n_cal"].to_numpy() == full.to_numpy()
    v = v[keep]
    rows = []
    key = ["dataset", "learner", "partition", "arm", "weights", "sampling"]
    for metric in metrics:
        cell = v.groupby(key + ["K"] + A.DRAW)[metric].mean()
        draw = cell.groupby(key + A.DRAW).mean().reset_index()
        for (ds, lr, part, arm), g in draw.groupby(["dataset", "learner", "partition", "arm"]):
            samp_def = "uniform" if lr == "rf" else "na"
            base = g[(g.weights == config.DEFAULT_WEIGHTS) & (g.sampling == samp_def)]
            base = base.set_index(A.DRAW)[metric]
            for (w, sa), h in g.groupby(["weights", "sampling"]):
                if w == config.DEFAULT_WEIGHTS and sa == samp_def:
                    continue
                d = h.set_index(A.DRAW)[metric] - base
                rows.append(dict(dataset=ds, learner=lr, partition=part, arm=arm, weights=w,
                                 sampling=sa, metric=metric, **_ci_cols(A.mean_ci(d))))
    return _sort(pd.DataFrame(rows), ["arm", "weights", "sampling", "metric"])


def tex_E5(e5):
    t = e5[e5.arm == "fed_L0"]
    rows = []
    for (ds, lr, part, w, sa), g in t.groupby(["dataset", "learner", "partition", "weights",
                                               "sampling"], sort=False):
        r = dict(Dataset=ds, Learner=lr, Partition=part, Weights=w, Sampling=sa)
        for metric, name in (("brier", "Delta Brier"), ("auroc", "Delta AUROC"),
                             ("mcc_pi", "Delta MCC")):
            q = g[g.metric == metric]
            r[name] = fmt_ci(q["mean"].iloc[0], q.lo.iloc[0], q.hi.iloc[0]) if len(q) else "NA"
        rows.append(r)
    return pd.DataFrame(rows)


def table_L1g(R):
    rows = []
    for metric in ("brier", "auroc", "mcc_pi"):
        pdw = _main(R.per_draw(metric))
        for (ds, lr, part), g in pdw.groupby(["dataset", "learner", "partition"]):
            rows.append(dict(dataset=ds, learner=lr, partition=part, metric=metric,
                             fed_L1=g.m31.mean(), fed_L1g=g.m31g.mean(),
                             **_ci_cols(A.mean_ci(g.m31g - g.m31))))
    return _sort(pd.DataFrame(rows), ["metric"])


def _tex_diff_by_metric(t, keys, metrics):
    rows = []
    for k, g in t.groupby(keys, sort=False):
        r = dict(zip([c.capitalize() for c in keys], k))
        for metric, name in metrics:
            q = g[g.metric == metric]
            r[name] = fmt_ci(q["mean"].iloc[0], q.lo.iloc[0], q.hi.iloc[0]) if len(q) else "NA"
        rows.append(r)
    return pd.DataFrame(rows)


def table_E9(R):
    t = A.leaf_summary(R.sites, R.csites)
    pooled = []
    for ds, g in R.sites.groupby("dataset"):
        pooled += A._leaf_rows(g, dataset=ds, partition="all", model="site")
    t = pd.concat([t, pd.DataFrame(pooled)], ignore_index=True)
    return t.sort_values(["dataset", "model", "partition", "learner"])


def table_E10(R):
    out = []
    for ev in EVAL_SHIFTS:
        sh = A.shares(R.per_draw("brier", view=ev), "brier")
        out.append(sh.assign(eval=ev))
    return _sort(_main(pd.concat(out, ignore_index=True)), ["eval", "level"])


def table_E11(R):
    sh = _main(A.shares(R.per_draw("brier", source="largest"), "brier"))
    pdw = _main(R.per_draw("auroc", source="largest"))
    gain = []
    for (ds, lr, part), g in pdw.groupby(["dataset", "learner", "partition"]):
        gain.append(dict(dataset=ds, learner=lr, partition=part,
                         **_ci_cols(A.mean_ci(g.m30 - g.m0), "gain_")))
    return _sort(sh), _sort(pd.DataFrame(gain))


def table_E12(R):
    rows = []
    sizes = R.m[(R.m.arm == "central_L2") & (R.m["eval"] == "test")].groupby(
        "dataset").n_cal.mean()
    for n in list(config.N_CAL_LIST) + ["full"]:
        pdw = _main(R.per_draw("brier", view=("ncal", n)))
        sh = A.shares(pdw, "brier")
        sh = sh[sh.level == 2]
        for _, q in sh.iterrows():
            g = pdw[(pdw.dataset == q.dataset) & (pdw.learner == q.learner)
                    & (pdw.partition == q.partition)]
            rows.append(dict(n_cal=n, n_cal_value=float(sizes.get(q.dataset, np.nan))
                             if n == "full" else float(n),
                             **q.drop(["level"]).to_dict(),
                             **_ci_cols(A.mean_ci(g.m2 - g.m0), "src_diff_"),
                             **_ci_cols(A.mean_ci(g.m32 - g.m0), "fed_diff_")))
    t = pd.DataFrame(rows)
    return _sort(t, ["n_cal_value"]) if len(t) else t


def table_L2_nonpositive(R):
    """Level 2 fits with a <= 0: share per dataset, learner, partition, arm and
    n_cal (E12 subsets and the full calibration part), pooled over K, draws and
    sources, with the mean AUROC change of the level 2 arms (K averaged, paired
    per draw): src_L2 - src_L0, fed_L2 - src_L0 and fed_L2 - fed_L0."""
    m = R.m
    v = m[(m["eval"] == "test") & m.arm.isin(["src_L2", "fed_L2"]) & m.learner.isin(MAIN_LEARNERS)
          & m.weights.isin(["na", config.DEFAULT_WEIGHTS])
          & m.sampling.isin(["na", config.DEFAULT_SAMPLING])].copy()
    full = v.groupby(["dataset"] + A.DRAW).n_cal.transform("max")
    v["n_cal_label"] = np.where(v.n_cal.to_numpy() == full.to_numpy(), "full", v.n_cal.astype(str))
    v["nonpos"] = v.rec_a <= 0
    share = (v.groupby(["dataset", "learner", "partition", "arm", "n_cal_label"])
             .agg(n_fits=("nonpos", "size"), n_nonpos=("nonpos", "sum"), share=("nonpos", "mean"),
                  mean_n_cal=("n_cal", "mean")).reset_index())
    rows = []
    for n in list(config.N_CAL_LIST) + ["full"]:
        pdw = _main(R.per_draw("auroc", view=("ncal", n)))
        for (ds, lr, part), g in pdw.groupby(["dataset", "learner", "partition"]):
            for arm, d in (("src_L2", g.m2 - g.m0), ("fed_L2", g.m32 - g.m0)):
                extra = _ci_cols(A.mean_ci(g.m32 - g.m30), "dauroc_own_L0_") if arm == "fed_L2" else {}
                rows.append(dict(dataset=ds, learner=lr, partition=part, arm=arm,
                                 n_cal_label=str(n), **_ci_cols(A.mean_ci(d), "dauroc_vs_src_L0_"),
                                 **extra))
    t = share.merge(pd.DataFrame(rows), on=["dataset", "learner", "partition", "arm", "n_cal_label"],
                    how="left")
    order = {str(n): i for i, n in enumerate(list(config.N_CAL_LIST) + ["full"])}
    return _sort(t.assign(_n=t.n_cal_label.map(order)), ["arm", "_n"]).drop(columns="_n")


def tex_L2_nonpositive(t):
    rows = []
    for _, q in t.iterrows():
        own = (fmt_ci(q.dauroc_own_L0_mean, q.dauroc_own_L0_lo, q.dauroc_own_L0_hi)
               if q.arm == "fed_L2" else "")
        rows.append({"Dataset": q.dataset, "Learner": q.learner, "Partition": q.partition,
                     "Arm": q.arm, "n_cal": q.n_cal_label, "Share a<=0": fmt(q.share),
                     "Fits": int(q.n_fits),
                     "Delta AUROC vs src_L0": fmt_ci(q.dauroc_vs_src_L0_mean, q.dauroc_vs_src_L0_lo,
                                                     q.dauroc_vs_src_L0_hi),
                     "Delta AUROC vs fed_L0": own})
    return pd.DataFrame(rows)


def table_calibration(R):
    rows = []
    for metric in ("cal_slope", "citl_int", "citl_diff"):
        pdw = _main(R.per_draw(metric))
        for (ds, lr, part), g in pdw.groupby(["dataset", "learner", "partition"]):
            for lv in T1_LEVELS + ["mc2"]:
                rows.append(dict(dataset=ds, learner=lr, partition=part, metric=metric,
                                 level=lv, **_ci_cols(A.mean_ci(g[lv]))))
    return _sort(pd.DataFrame(rows), ["metric", "level"])


def table_rfxgb(R):
    rows = []
    for metric in ("auroc", "brier"):
        pdw = R.per_draw(metric)
        key = ["dataset", "partition"] + A.DRAW
        rf = pdw[pdw.learner == "rf"].set_index(key)
        xg = pdw[pdw.learner == "xgb"].set_index(key)
        for lv in ("m31", "m32"):
            d = (rf[lv] - xg[lv]).dropna().reset_index()
            for (ds, part), g in d.groupby(["dataset", "partition"]):
                rows.append(dict(dataset=ds, partition=part, metric=metric, level=lv,
                                 **_ci_cols(A.mean_ci(g[lv]))))
    t = pd.DataFrame(rows)
    t["learner"] = "rf-xgb"
    return _sort(t, ["metric", "level"])


def table_hp(Rd, Rr):
    """Default against regularized on the common draws."""
    rows = []
    leaves = []
    for hp, R in (("default", Rd), ("regularized", Rr)):
        sh = _main(A.shares(R.per_draw("brier"), "brier"))
        pa = _main(R.per_draw("auroc"))
        pb = _main(R.per_draw("brier"))
        for (ds, lr, part), g in pa.groupby(["dataset", "learner", "partition"]):
            gb = pb[(pb.dataset == ds) & (pb.learner == lr) & (pb.partition == part)]
            r = dict(hp=hp, dataset=ds, learner=lr, partition=part,
                     **_ci_cols(A.mean_ci(g.m30 - g.m0), "gain_"),
                     auroc_m0=g.m0.mean(), auroc_m30=g.m30.mean(), auroc_mc=g.mc.mean(),
                     brier_m0=gb.m0.mean(), brier_m30=gb.m30.mean(), brier_mc=gb.mc.mean())
            for lvl in (1, 2):
                q = sh[(sh.dataset == ds) & (sh.learner == lr) & (sh.partition == part)
                       & (sh.level == lvl)]
                if len(q):
                    q = q.iloc[0]
                    r.update({f"r{lvl}": q.r, f"r{lvl}_lo": q.r_lo, f"r{lvl}_hi": q.r_hi,
                              f"r{lvl}_reportable": q.reportable})
            rows.append(r)
        leaves.append(table_E9(R).assign(hp=hp))
    k = pd.DataFrame(rows)
    leaves = pd.concat(leaves, ignore_index=True)
    trade = A.tradeoff_table(k, leaves[leaves.partition != "all"])
    return _sort(k, ["hp"]), trade, leaves


def tex_hp(trade):
    t = trade.copy()
    for c in t.columns:
        if t[c].dtype.kind == "f":
            t[c] = t[c].map(fmt)
    return t.rename(columns={"dataset": "Dataset", "learner": "Learner",
                             "partition": "Partition", "model": "Model"})


def anomalies(R, runs_root):
    rows = []
    s = R.sites
    for w in config.WEIGHT_CODE:
        c = f"oob_fallback_{w}"
        if c in s:
            for ds, g in s.groupby("dataset"):
                rows.append(dict(item=f"oob_fallback_{w}", dataset=ds,
                                 count=int(g[c].fillna(False).astype(bool).sum()),
                                 out_of=int(len(g)), unit="site models"))
    m = R.m
    l2 = m[m.arm.str.endswith("_L2")]
    for ds, g in l2.groupby("dataset"):
        rows.append(dict(item="level2_a_nonpositive", dataset=ds,
                         count=int((g.rec_a <= 0).sum()), out_of=int(len(g)),
                         unit="level 2 fits (all evals and n_cal)"))
    v = R.view("test")
    l2v = v[v.arm.str.endswith("_L2")]
    for (ds, arm), g in l2v.groupby(["dataset", "arm"]):
        rows.append(dict(item=f"level2_a_nonpositive_{arm}", dataset=ds,
                         count=int((g.rec_a <= 0).sum()), out_of=int(len(g)),
                         unit="level 2 fits, default view"))
    l0 = v[v.arm.isin(["src_L0", "fed_L0", "central_L0"]) & v.learner.isin(MAIN_LEARNERS)]
    for (ds, arm), g in l0.groupby(["dataset", "arm"]):
        one = g.ppr_05.isin([0.0, 1.0])
        rows.append(dict(item=f"single_class_at_0.5_{arm}", dataset=ds, count=int(one.sum()),
                         out_of=int(len(g)), unit="metric rows, default view"))
    errs = A.error_files(runs_root)
    rows.append(dict(item="error_files", dataset="all", count=len(errs), out_of=np.nan,
                     unit="cells"))
    return pd.DataFrame(rows)


def sites_r0_f0(R, K=10):
    s = R.sites
    s = s[(s.rep == 0) & (s.fold == 0) & (s.K == K)]
    return s[["dataset", "partition", "K", "site", "n_k", "pos_k", "pi_k"]].sort_values(
        ["dataset", "partition", "site"])


def reliability(preds_root, datasets, part="dir0.5", K=5, n_bins=10):
    """Reliability curves on the test part of r0 f0 (equal width bins). Source
    models are pooled over sources."""
    rows = []
    for ds in datasets:
        p = Path(preds_root) / ds / "r0_f0" / f"{part}_K{K}.parquet"
        c = Path(preds_root) / ds / "r0_f0" / "central.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        d = d[d.is_cal == 0]
        cen = pd.read_parquet(c) if c.exists() else None
        if cen is not None:
            cen = cen.set_index("row").loc[d.row]
        for lr in MAIN_LEARNERS:
            curves = {}
            for lv in ("L0", "L1", "L2"):
                cols = [x for x in d.columns if x.startswith(f"{lr}_src") and x.endswith(f"_{lv}")]
                curves[f"src_{lv}"] = (np.concatenate([d[x].to_numpy() for x in cols]),
                                       np.tile(d.y.to_numpy(), len(cols)))
                curves[f"fed_{lv}"] = (d[f"{lr}_fed_{lv}"].to_numpy(), d.y.to_numpy())
            if cen is not None:
                curves["central_L0"] = (cen[f"{lr}_central_L0"].to_numpy(), d.y.to_numpy())
            edges = np.linspace(0, 1, n_bins + 1)
            for arm, (pp, yy) in curves.items():
                b = np.clip(np.digitize(pp, edges[1:-1]), 0, n_bins - 1)
                for i in range(n_bins):
                    sel = b == i
                    if sel.sum():
                        rows.append(dict(dataset=ds, learner=lr, arm=arm, bin=i,
                                         p_mean=float(pp[sel].mean()), y_mean=float(yy[sel].mean()),
                                         n=int(sel.sum())))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# numbers.json and numbers.md
# ----------------------------------------------------------------------------

class Numbers:
    def __init__(self):
        self.items = {}

    def add(self, key, value, how, lo=None, hi=None, J=None, reason=None, **extra):
        def clean(x):
            if x is None:
                return None
            if isinstance(x, (np.integer,)):
                return int(x)
            if isinstance(x, (np.floating, float)):
                return float(x) if math.isfinite(float(x)) else None
            if isinstance(x, np.bool_):
                return bool(x)
            return x
        val = clean(value)
        if isinstance(value, (list, tuple)):
            val = [clean(v) for v in value]
        e = {"value": val, "how": how}
        if lo is not None or hi is not None:
            e["lo"], e["hi"] = clean(lo), clean(hi)
        if J is not None:
            e["J"] = clean(J)
        if val is None:
            e["reason"] = reason or "not computable from the run files"
        e.update({k: clean(v) for k, v in extra.items()})
        self.items[key] = e

    def ci(self, key, ci, how, **extra):
        self.add(key, ci["mean"], how, lo=ci["lo"], hi=ci["hi"], J=ci["J"], **extra)

    def range(self, key, keys, how):
        vals = [self.items[k]["value"] for k in keys if k in self.items
                and self.items[k]["value"] is not None]
        if vals:
            self.add(key, [min(vals), max(vals)], how, n=len(vals))
        else:
            self.add(key, None, how, reason="no value available")

    def write(self, out):
        (out / "numbers.json").write_text(json.dumps(self.items, indent=1, sort_keys=False))
        lines = ["# numbers", "",
                 "Every value is computed from the run files by `analyze.py report`.",
                 "Intervals are 95% percentile bootstrap intervals over draws (B = 2000).",
                 "Learners rf and xgb only (E13, logistic regression, is not reported).", "",
                 "| key | value | 95% CI | J | how |", "|---|---|---|---|---|"]
        for k, e in self.items.items():
            v = e["value"]
            if isinstance(v, list):
                vs = "[" + ", ".join(_num(x) for x in v) + "]"
            else:
                vs = _num(v)
            ci = (f"[{_num(e['lo'])}, {_num(e['hi'])}]" if "lo" in e and e["lo"] is not None
                  else "")
            how = e["how"] + (f" Null: {e['reason']}." if v is None else "")
            if "reportable" in e:
                how += f" Reportable: {'yes' if e['reportable'] else 'no'}."
            lines.append(f"| `{k}` | {vs} | {ci} | {e.get('J', '') if e.get('J') is not None else ''} | {how} |")
        (out / "numbers.md").write_text("\n".join(lines) + "\n")


def _num(x):
    if x is None:
        return "null"
    if isinstance(x, bool):
        return str(x).lower()
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return f"{x:.4f}" if abs(x) < 1000 else f"{x:.0f}"
    return str(x)


H_DIFF = "Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight."
H_SHARE = ("r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; "
           "reportable when the interval of mean(m3l - m0) excludes 0.")


def build_numbers(R, t1, t2, t3, perK, e5, l1g, e9, e10, e11, e12, rfxgb, calib, anom, hp, l2neg,
                  meta, hyper):
    N = Numbers()
    # design
    for ds, d in meta["draws"].items():
        N.add(f"design.n_draws.{ds}", d["n_draws"], "Distinct (rep, fold) draws in the main grid.")
        N.add(f"design.n_repetitions.{ds}", d["n_reps"], "Repetitions of 5 fold CV in the main grid.")
    N.add("design.n_folds", config.N_FOLDS, "Folds per repetition.")
    for name, d, how in (("design", meta["draws"], "main grid"),
                         ("design.regularized", meta.get("draws_reg", {}), "regularized grid, common draws")):
        if not d:
            continue
        for k in ("n_draws", "n_reps"):
            vals = {x[k] for x in d.values()}
            key = f"{name}.{'n_draws' if k == 'n_draws' else 'n_repetitions'}"
            if len(vals) == 1:
                N.add(key, vals.pop(), f"Same for every dataset ({how}).")
            else:
                N.add(key, None, f"Per dataset ({how}).", reason="differs between datasets")
    for ds, d in meta.get("draws_reg", {}).items():
        N.add(f"design.regularized.n_draws.{ds}", d["n_draws"],
              "Draws of the regularized grid shared with the main grid (common draws).")
        N.add(f"design.regularized.n_repetitions.{ds}", d["n_reps"],
              "Repetitions in the common draws of the regularized grid.")
    N.add("design.min_per_class", config.MIN_PER_CLASS, "Records of each class given to every site.")
    N.add("design.dirichlet_alpha", sorted(a for _, a in config.PARTITIONS.values() if a),
          "Dirichlet concentrations of the label skewed partitions.")
    N.add("design.K", list(config.K_LIST), "Numbers of sites.")
    for ds, d in meta.get("data", {}).items():
        for k, v in d.items():
            N.add(f"data.{ds}.{k}", v, "From the data loader used by the runs.")
    for lr, keys in (("rf", ("n_estimators",)), ("xgb", ("n_estimators", "max_depth", "learning_rate"))):
        for k in keys:
            v = hyper.get(lr, {}).get(k) if hyper else None
            N.add(f"hyper.{lr}.{k}", v, "get_params() written by the runs (hyperparameters.json).",
                  reason="hyperparameters.json not found")
    N.add("runs.error_files", int(anom[anom.item == "error_files"]["count"].sum()),
          "Error files in the run folders (failed cells).")
    N.add("runs.missing_files", meta.get("missing_files"), "Expected outputs absent.")
    # calibration part
    v = R.view("test")
    cal = v[(v.arm == "central_L2")].groupby("dataset").n_cal.mean()
    for ds, x in cal.items():
        N.add(f"cal.mean_size.{ds}", x, "Mean size of the calibration part (20% of the deployment site).")
    # shares
    for _, q in t2.iterrows():
        key = f"r.{q.metric}.L{q.level}.{q.dataset}.{q.learner}.{q.partition}"
        N.add(key, q.r, H_SHARE, lo=q.r_lo, hi=q.r_hi, J=q.J, reportable=bool(q.reportable),
              r_vs_plain=q.r_vs_plain)
    for metric in ("brier", "mcc_pi"):
        for lvl in (1, 2):
            for part in PARTS:
                ks = [f"r.{metric}.L{lvl}.{ds}.{lr}.{part}" for ds in config.DATASETS
                      for lr in MAIN_LEARNERS]
                N.range(f"r.{metric}.L{lvl}.{part}.range", ks,
                        "Minimum and maximum over the dataset and learner pairs present.")
    # AUROC gain, K averaged, and differences of every level from level 0
    for _, q in t1[t1.level != "m0"].iterrows():
        N.add(f"diff.{q.metric}.{LEVEL_LABEL[q.level]}.{q.dataset}.{q.learner}.{q.partition}",
              q.diff_mean, H_DIFF + " Level minus L0 (mean source model).", lo=q.diff_lo,
              hi=q.diff_hi, J=q.diff_J)
    for _, q in t1.iterrows():
        N.add(f"level.{q.metric}.{LEVEL_LABEL[q.level]}.{q.dataset}.{q.learner}.{q.partition}",
              q["mean"], "Mean over draws, K averaged.", lo=q.lo, hi=q.hi, J=q.J)
    for part in PARTS:
        ks = [f"diff.auroc.L3.{ds}.{lr}.{part}" for ds in config.DATASETS for lr in MAIN_LEARNERS]
        N.range(f"diff.auroc.L3.{part}.range", ks,
                "AUROC gain of fed_L0 over src_L0, minimum and maximum over pairs.")
    # per K
    for _, q in perK[(perK.kind == "diff") & perK.level.isin(["m1", "m30"])
                     & perK.metric.isin(["brier", "auroc"])].iterrows():
        N.add(f"perK.{q.metric}.{LEVEL_LABEL[q.level]}.{q.dataset}.{q.learner}.{q.partition}.K{q.K}",
              q["mean"], "Mean over draws at this K of the level minus L0.", lo=q.lo, hi=q.hi, J=q.J)
    # leaves
    for _, q in e9.iterrows():
        N.add(f"leaf.{q.learner}.{q.model}.{q.partition}.{q.dataset}.single", q.leaf_single,
              "Leaves reached by exactly one distinct training row, pooled over trees, sites, cells and draws.")
        N.add(f"leaf.{q.learner}.{q.model}.{q.partition}.{q.dataset}.lt5", q.leaf_lt5,
              "Leaves reached by fewer than 5 distinct training rows, pooled the same way.")
    # Heart at K = 5 and the Random Forest at 0.5 at K = 10
    if "heart" in set(R.m.dataset):
        ca = R.cells("auroc")
        for lr in MAIN_LEARNERS:
            for part in PARTS:
                g = ca[(ca.dataset == "heart") & (ca.learner == lr) & (ca.partition == part) & (ca.K == 5)]
                N.ci(f"heart.K5.auroc.source.{lr}.{part}", A.mean_ci(g.m0),
                     "AUROC of the mean source model at K = 5.")
                N.ci(f"heart.K5.auroc.central.{lr}.{part}", A.mean_ci(g.mc),
                     "AUROC of the centralized model (draws of K = 5 cells).")
        cp = R.cells("ppr_05")
        cm = R.cells("mcc_pi")
        for part in PARTS:
            sel = lambda c: c[(c.dataset == "heart") & (c.learner == "rf") & (c.partition == part)  # noqa: E731
                              & (c.K == 10)]
            g = sel(cp)
            N.ci(f"heart.rf.K10.ppr05.fed_L0.{part}", A.mean_ci(g.m30),
                 "Share of positive predictions at threshold 0.5 of fed_L0 rf at K = 10.")
            N.ci(f"heart.rf.K10.ppr05.source.{part}", A.mean_ci(g.m0),
                 "Same for the mean source model.")
            g = sel(ca)
            N.ci(f"heart.rf.K10.auroc.fed_L0.{part}", A.mean_ci(g.m30), "AUROC [A] of fed_L0 rf at K = 10.")
            N.ci(f"heart.rf.K10.auroc.source.{part}", A.mean_ci(g.m0),
                 "AUROC [B] of the mean source model rf at K = 10.")
            g = sel(cm)
            N.ci(f"heart.rf.K10.mcc_pi.diff.{part}", A.mean_ci(g.m30 - g.m0),
                 "mcc_pi of fed_L0 rf minus the mean source model at K = 10.")
    else:
        N.add("heart", None, "Heart quantities.", reason="Heart runs not included in this report")
    # calibration in the large of fed_L0 xgb, equal partition
    for _, q in calib[(calib.learner == "xgb") & (calib.partition == "equal") & (calib.level == "m30")
                      & calib.metric.isin(["citl_diff", "citl_int"])].iterrows():
        N.add(f"calib.{q.metric}.fed_L0.xgb.equal.{q.dataset}", q["mean"],
              "Mean over draws, K averaged.", lo=q.lo, hi=q.hi, J=q.J)
    # E5
    for _, q in e5.iterrows():
        N.add(f"e5.{q.metric}.{q.arm}.{q.dataset}.{q.learner}.{q.partition}.{q.weights}_{q.sampling}",
              q["mean"], H_DIFF + " Configuration minus the default (equal weights, uniform sampling).",
              lo=q.lo, hi=q.hi, J=q.J)
    # rf against xgb
    for _, q in rfxgb.iterrows():
        N.add(f"rfxgb.{q.metric}.{LEVEL_LABEL[q.level]}.{q.dataset}.{q.partition}", q["mean"],
              H_DIFF + " rf minus xgb.", lo=q.lo, hi=q.hi, J=q.J)
    # fed_L1g against fed_L1
    for _, q in l1g.iterrows():
        N.add(f"l1g.{q.metric}.{q.dataset}.{q.learner}.{q.partition}", q["mean"],
              H_DIFF + " fed_L1g minus fed_L1.", lo=q.lo, hi=q.hi, J=q.J)
    # E10
    for _, q in e10.iterrows():
        N.add(f"e10.r.brier.L{q.level}.{q['eval']}.{q.dataset}.{q.learner}.{q.partition}", q.r,
              H_SHARE + " Shifted calibration and test parts.", lo=q.r_lo, hi=q.r_hi, J=q.J,
              reportable=bool(q.reportable))
    # E11
    sh11, gain11 = e11
    for _, q in sh11.iterrows():
        N.add(f"e11.r.brier.L{q.level}.{q.dataset}.{q.learner}.{q.partition}", q.r,
              H_SHARE + " Largest site as the source.", lo=q.r_lo, hi=q.r_hi, J=q.J,
              reportable=bool(q.reportable))
    for _, q in gain11.iterrows():
        N.add(f"e11.diff.auroc.L3.{q.dataset}.{q.learner}.{q.partition}", q.gain_mean,
              H_DIFF + " fed_L0 minus the largest source.", lo=q.gain_lo, hi=q.gain_hi, J=q.gain_J)
    # level 2 fits with a <= 0
    for _, q in l2neg.iterrows():
        base = f"l2neg.{q.arm}.{q.dataset}.{q.learner}.{q.partition}.n{q.n_cal_label}"
        N.add(f"{base}.share", q.share, "Share of level 2 fits with a <= 0, pooled over K, draws "
              "and sources (eval test, default configuration).", n_fits=int(q.n_fits))
        N.add(f"{base}.dauroc_vs_src_L0", q.dauroc_vs_src_L0_mean,
              H_DIFF + " Level 2 arm minus src_L0 (mean source model).",
              lo=q.dauroc_vs_src_L0_lo, hi=q.dauroc_vs_src_L0_hi, J=q.dauroc_vs_src_L0_J)
        if q.arm == "fed_L2":
            N.add(f"{base}.dauroc_vs_fed_L0", q.dauroc_own_L0_mean, H_DIFF + " fed_L2 minus fed_L0.",
                  lo=q.dauroc_own_L0_lo, hi=q.dauroc_own_L0_hi, J=q.dauroc_own_L0_J)
    # E12
    for _, q in e12.iterrows():
        N.add(f"e12.r.brier.L2.{q.dataset}.{q.learner}.{q.partition}.n{q.n_cal}", q.r,
              H_SHARE + " src_L2 and fed_L2 fitted on n_cal calibration records.",
              lo=q.r_lo, hi=q.r_hi, J=q.J, reportable=bool(q.reportable))
    # regularized
    k, _, leaves = hp if hp is not None else (None, None, None)
    if k is not None:
        for _, q in k.iterrows():
            base = f"hp.{q.hp}.{q.dataset}.{q.learner}.{q.partition}"
            for lvl in (1, 2):
                if f"r{lvl}" in q and pd.notna(q.get(f"r{lvl}")):
                    N.add(f"{base}.r.brier.L{lvl}", q[f"r{lvl}"], H_SHARE + " Common draws.",
                          lo=q[f"r{lvl}_lo"], hi=q[f"r{lvl}_hi"], reportable=bool(q[f"r{lvl}_reportable"]))
            N.add(f"{base}.diff.auroc.L3", q.gain_mean, H_DIFF + " Common draws.",
                  lo=q.gain_lo, hi=q.gain_hi, J=q.gain_J)
            N.add(f"{base}.auroc.central", q.auroc_mc, "Mean over the common draws.")
            N.add(f"{base}.brier.central", q.brier_mc, "Mean over the common draws.")
        for _, q in leaves.iterrows():
            N.add(f"hp.{q.hp}.leaf.{q.learner}.{q.model}.{q.partition}.{q.dataset}.single",
                  q.leaf_single, "Single row leaf fraction, common draws.")
            N.add(f"hp.{q.hp}.leaf.{q.learner}.{q.model}.{q.partition}.{q.dataset}.lt5",
                  q.leaf_lt5, "Fewer than 5 distinct rows leaf fraction, common draws.")
    # anomalies
    for _, q in anom.iterrows():
        N.add(f"anomaly.{q['item']}.{q.dataset}", int(q["count"]), f"Count of {q.unit}.",
              out_of=None if pd.isna(q.out_of) else int(q.out_of))
    return N


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def check_complete(runs_root, datasets, r_rep):
    """Expected files of every task present and no error file."""
    missing = []
    for ds in datasets:
        for tid in range(5 * r_rep):
            r, f = divmod(tid, 5)
            d = Path(runs_root) / ds / f"r{r}_f{f}"
            names = ["central.parquet", "central_sites.parquet", "meta.json"]
            for part in config.PARTITIONS:
                for K in config.K_LIST:
                    names += [f"{part}_K{K}.parquet", f"{part}_K{K}_sites.parquet"]
            missing += [d / n for n in names if not (d / n).exists()]
    errors = [e for e in A.error_files(runs_root) if e.parts[-3] in datasets]
    return missing, errors


def build(runs, runs_reg, preds, out, datasets, r_rep, r_rep_reg, allow_incomplete=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    missing, errors = check_complete(runs, datasets, r_rep)
    if (missing or errors) and not allow_incomplete:
        raise SystemExit(f"incomplete runs: {len(missing)} missing files, {len(errors)} error files")
    R = Runs(*A.load_runs(runs, datasets))
    meta = {"missing_files": len(missing), "draws": {}, "data": {}}
    for ds, g in R.m.groupby("dataset"):
        meta["draws"][ds] = dict(n_draws=int(g[A.DRAW].drop_duplicates().shape[0]),
                                 n_reps=int(g.rep.nunique()))
    hp = None
    if runs_reg and Path(runs_reg).exists():
        mr, er = check_complete(runs_reg, datasets, r_rep_reg)
        if (mr or er) and not allow_incomplete:
            raise SystemExit(f"incomplete regularized runs: {len(mr)} missing, {len(er)} errors")
        Rr = Runs(*A.load_runs(runs_reg, datasets))
        draws = A.common_draws(R.m, Rr.m)
        meta["draws_reg"] = {ds: dict(n_draws=int(len(g)), n_reps=int(g.rep.nunique()))
                             for ds, g in draws.groupby("dataset")}
        hp = table_hp(R.restrict(draws), Rr.restrict(draws))
    try:
        import data as data_mod
        for ds in datasets:
            d = data_mod.load(ds)
            meta["data"][ds] = dict(n_rows=int(d.X.shape[0]), n_features=int(d.X.shape[1]),
                                    n_positives=int(d.y.sum()))
    except Exception as exc:  # data folder absent: keys become null with the reason
        meta["data_error"] = repr(exc)
    hyper_path = out / "hyperparameters.json"
    hyper = json.loads(hyper_path.read_text()) if hyper_path.exists() else None

    t1 = table_T1(R)
    write_table(t1, out, "T1_levels", tex_df=None)
    for metric in ("brier", "auroc"):
        tex = tex_T1(t1, metric)
        write_table(tex, out, f"T1_{metric}")
    t2 = table_T2(R)
    write_table(t2, out, "T2_shares", tex_df=tex_T2(t2))
    t3 = table_T3(R)
    write_table(t3, out, "T3_auroc_gain", tex_df=tex_T3(t3))
    perK = perK_differences(R)
    perK.to_csv(out / "perK_differences.csv", index=False)
    e5 = table_E5(R)
    write_table(e5, out, "S_E5", tex_df=tex_E5(e5))
    l1g = table_L1g(R)
    write_table(l1g, out, "S_L1g", tex_df=_tex_diff_by_metric(
        l1g, ["dataset", "learner", "partition"],
        (("brier", "Delta Brier"), ("auroc", "Delta AUROC"), ("mcc_pi", "Delta MCC"))))
    e9 = table_E9(R)
    write_table(e9, out, "S_E9_leaves", tex_df=e9.assign(
        leaf_single=e9.leaf_single.map(fmt), leaf_lt5=e9.leaf_lt5.map(fmt)))
    e10 = table_E10(R)
    write_table(e10, out, "S_E10", tex_df=pd.DataFrame([dict(
        Dataset=q.dataset, Learner=q.learner, Partition=q.partition, Eval=q["eval"],
        Level=f"r{q.level}", r=fmt_ci(q.r, q.r_lo, q.r_hi, 2),
        Reportable="yes" if q.reportable else "no") for _, q in e10.iterrows()]))
    e11 = table_E11(R)
    e11[0].to_csv(out / "S_E11_shares.csv", index=False)
    write_table(e11[1], out, "S_E11_gain", tex_df=pd.DataFrame([dict(
        Dataset=q.dataset, Learner=q.learner, Partition=q.partition,
        **{"AUROC gain L3": fmt_ci(q.gain_mean, q.gain_lo, q.gain_hi)}) for _, q in e11[1].iterrows()]))
    e12 = table_E12(R)
    write_table(e12, out, "S_E12", tex_df=pd.DataFrame([dict(
        Dataset=q.dataset, Learner=q.learner, Partition=q.partition, n_cal=q.n_cal,
        r2=fmt_ci(q.r, q.r_lo, q.r_hi, 2), Reportable="yes" if q.reportable else "no")
        for _, q in e12.iterrows()]) if len(e12) else None)
    calib = table_calibration(R)
    calib.to_csv(out / "S_calibration.csv", index=False)
    rfxgb = table_rfxgb(R)
    rfxgb.to_csv(out / "S_rfxgb.csv", index=False)
    if hp is not None:
        write_table(hp[0], out, "S_hp_quantities", tex_df=None)
        write_table(hp[1], out, "S_hp", tex_df=tex_hp(hp[1]))
    sites_r0_f0(R).to_csv(out / "sites_r0_f0_K10.csv", index=False)
    if preds:
        reliability(preds, datasets).to_csv(out / "reliability.csv", index=False)
    anom = anomalies(R, runs)
    anom.to_csv(out / "anomalies.csv", index=False)
    l2neg = table_L2_nonpositive(R)
    write_table(l2neg, out, "S_L2_nonpositive", tex_df=tex_L2_nonpositive(l2neg))
    N = build_numbers(R, t1, t2, t3, perK, e5, l1g, e9, e10, e11, e12, rfxgb, calib, anom, hp, l2neg,
                      meta, hyper)
    N.write(out)
    return N
