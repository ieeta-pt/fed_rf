"""Supplementary tables as booktabs LaTeX, from the CSV files that analyze.py
report already wrote. Nothing is recomputed.

Each table is written to <out>/<name>.tex and holds only the tabular
environment (no table float, no caption), to be typeset with \\small inside a
6.5 in text width. Values are means with their 95% interval as
"0.268 [0.263, 0.273]". A dagger marks a share whose denominator interval
covers zero (not reportable); the caption should say so.

Tables
    si_levels_brier, si_levels_auroc   seven models, all partitions (T1_levels)
    si_shares_brier, si_shares_mcc     r_1 and r_2 with their denominator (T2_shares, mcc_pi)
    si_calibration_citl                calibration in the large, intercept citl_int (S_calibration)
    si_calibration_slope               calibration slope (S_calibration)
    si_l2_nonpositive_<partition>      share of level 2 fits with a <= 0 and the AUROC
                                       change, per n_cal (S_L2_nonpositive)
    si_e5_brier, si_e5_auroc, si_e5_mcc  weights and sampling of the federated model
                                       (Level 3) against the default (S_E5)
    si_l1g                             one shift against one shift per site (S_L1g)
    si_leaves                          leaves reached by one and by fewer than five
                                       records, default and regularized (S_E9_leaves, S_hp)
    si_e10                             shifted prevalence at the deployment site (S_E10)
    si_e11                             largest site as the source (S_E11_shares, S_E11_gain)
    si_e12                             size of the calibration sample (S_E12)
    si_hp                              default and regularized hyperparameters

Usage
    python experiments/csbj/si_tables.py [--results CSBJ/results] [--out CSBJ/results/si_tables]
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd

DATASETS = ("aids", "diabetic", "heart")
LEARNERS = ("rf", "xgb")
PARTS = ("equal", "dir0.5", "dir0.1")
DS = {"aids": "AIDS", "diabetic": "Diabetic", "heart": "Heart"}
LR = {"rf": "Random Forest", "xgb": "XGBoost"}
PART = {"equal": "Equal", "dir0.5": r"Dirichlet $\alpha=0.5$", "dir0.1": r"Dirichlet $\alpha=0.1$"}
PART_FILE = {"equal": "equal", "dir0.5": "dir05", "dir0.1": "dir01"}
LEVELS = ("m0", "m1", "m2", "m30", "m31", "m32", "mc")
LEVEL = {"m0": "Level 0", "m1": "Level 1", "m2": "Level 2", "m30": "Level 3",
         "m31": "Levels 3 and 1", "m32": "Levels 3 and 2", "mc": "Centralized"}
N_CAL = ("25", "50", "100", "200", "full")
DAGGER = r"$^{\dagger}$"


# ----------------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------------

def num(x, d=3):
    """Number with d decimals, a true minus sign and no negative zero."""
    if x is None or not math.isfinite(float(x)):
        return "--"
    s = f"{float(x):.{d}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace("-", "$-$")


def ci(m, lo, hi, d=3):
    if not math.isfinite(float(m)):
        return "--"
    if not (math.isfinite(float(lo)) and math.isfinite(float(hi))):
        return num(m, d)
    return f"{num(m, d)} [{num(lo, d)}, {num(hi, d)}]"


def share(q, d=3, prefix="r"):
    """Share with its interval, with a dagger when it is not reportable."""
    s = ci(q[prefix], q[f"{prefix}_lo"], q[f"{prefix}_hi"], d)
    return s if s == "--" or bool(q["reportable"]) else s + DAGGER


def tex(s):
    return str(s).replace("_", r"\_")


def tabular(spec, header, blocks):
    """booktabs tabular. header is a list of LaTeX header lines (without the
    rules). blocks is a list of (title, rows); a title spans every column and
    rows are lists of cells."""
    ncol = len(spec)
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", *header, r"\midrule"]
    for i, (title, rows) in enumerate(blocks):
        if not rows:
            continue
        if i:
            lines.append(r"\addlinespace")
        if title:
            lines.append(rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\textit{{{title}}}}} \\")
        lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def groups(df):
    """(title, frame) per dataset and learner, in the paper order."""
    for ds in DATASETS:
        for lr in LEARNERS:
            g = df[(df.dataset == ds) & (df.learner == lr)]
            if len(g):
                yield f"{DS[ds]}, {LR[lr]}", g


def one(df, **kw):
    q = df
    for k, v in kw.items():
        q = q[q[k].astype(str) == str(v)]
    return q.iloc[0] if len(q) else None


PART_HEADER = " & ".join(PART[p] for p in PARTS)


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------

def levels(R, metric, d=3, src="T1_levels", level_set=LEVELS):
    t = pd.read_csv(R / f"{src}.csv")
    t = t[t.metric == metric]
    blocks = []
    for title, g in groups(t):
        rows = []
        for lv in level_set:
            cells = []
            for p in PARTS:
                q = one(g, partition=p, level=lv)
                cells.append(ci(q["mean"], q.lo, q.hi, d) if q is not None else "--")
            rows.append([LEVEL[lv]] + cells)
        blocks.append((title, rows))
    return tabular("lccc", [f"Model & {PART_HEADER} \\\\"], blocks)


def shares(R, metric):
    t = pd.read_csv(R / "T2_shares.csv")
    t = t[t.metric == metric]
    blocks = []
    for title, g in groups(t):
        rows = []
        for p in PARTS:
            for lvl in (1, 2):
                q = one(g, partition=p, level=lvl)
                if q is None:
                    continue
                rows.append([PART[p] if lvl == 1 else "", f"Level {lvl}", share(q),
                             ci(q.den, q.den_lo, q.den_hi, 4)])
        blocks.append((title, rows))
    return tabular("llcc", [r"Partition & Level & $r_\ell$ & Denominator \\"], blocks)


def l2_nonpositive(R, part):
    t = pd.read_csv(R / "S_L2_nonpositive.csv")
    t = t[t.partition == part]
    blocks = []
    for title, g in groups(t):
        rows = []
        for n in N_CAL:
            s, f = one(g, arm="src_L2", n_cal_label=n), one(g, arm="fed_L2", n_cal_label=n)
            if s is None and f is None:
                continue
            ref = s if s is not None else f
            label = f"All ({ref.mean_n_cal:,.0f})".replace(",", "{,}") if n == "full" else n
            cells = [label]
            cells += ([num(s.share), ci(s.dauroc_vs_src_L0_mean, s.dauroc_vs_src_L0_lo,
                                        s.dauroc_vs_src_L0_hi)] if s is not None else ["--", "--"])
            cells += ([num(f.share), ci(f.dauroc_own_L0_mean, f.dauroc_own_L0_lo,
                                        f.dauroc_own_L0_hi)] if f is not None else ["--", "--"])
            rows.append(cells)
        blocks.append((title, rows))
    header = [r" & \multicolumn{2}{c}{Level 2} & \multicolumn{2}{c}{Levels 3 and 2} \\",
              r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
              r"$n_{cal}$ & Share $a \le 0$ & $\Delta$AUROC vs Level 0 "
              r"& Share $a \le 0$ & $\Delta$AUROC vs Level 3 \\"]
    return tabular("lcccc", header, blocks)


E5_COMBOS = {"rf": [("equal", "oob"), ("size", "uniform"), ("size", "oob"),
                    ("positives", "uniform"), ("positives", "oob")],
             "xgb": [("size", "na"), ("positives", "na")]}
WEIGHTS = {"equal": "Equal", "size": "Size", "positives": "Positives"}
SAMPLING = {"uniform": "uniform", "oob": "OOB"}


def e5(R, metric, d):
    t = pd.read_csv(R / "S_E5.csv")
    t = t[(t.arm == "fed_L0") & (t.metric == metric)]
    blocks = []
    for title, g in groups(t):
        lr = g.learner.iloc[0]
        rows = []
        for w, sa in E5_COMBOS[lr]:
            cells = []
            for p in PARTS:
                q = one(g, partition=p, weights=w, sampling=sa)
                cells.append(ci(q["mean"], q.lo, q.hi, d) if q is not None else "--")
            name = WEIGHTS[w] + (f", {SAMPLING[sa]}" if sa in SAMPLING else "")
            rows.append([name] + cells)
        blocks.append((title, rows))
    return tabular("lccc", [f"Weights, sampling & {PART_HEADER} \\\\"], blocks)


def l1g(R):
    t = pd.read_csv(R / "S_L1g.csv")
    blocks = []
    for title, g in groups(t):
        rows = []
        for p in PARTS:
            cells = [PART[p]]
            for metric, d in (("brier", 4), ("auroc", 4), ("mcc_pi", 4)):
                q = one(g, partition=p, metric=metric)
                cells.append(ci(q["mean"], q.lo, q.hi, d) if q is not None else "--")
            rows.append(cells)
        blocks.append((title, rows))
    return tabular("lccc", [r"Partition & $\Delta$Brier & $\Delta$AUROC & $\Delta$MCC \\"], blocks)


def leaves(R):
    e9 = pd.read_csv(R / "S_E9_leaves.csv")
    hp = pd.read_csv(R / "S_hp.csv")
    blocks = []
    for title, g in groups(e9):
        ds, lr = g.dataset.iloc[0], g.learner.iloc[0]
        h = hp[(hp.dataset == ds) & (hp.learner == lr)]
        rows = []
        for p in PARTS + ("central",):
            if p == "central":
                q, r = one(g, model="central"), one(h, model="central")
                name = "Centralized"
            else:
                q, r = one(g, model="site", partition=p), one(h, model="source L0", partition=p)
                name = f"Sites, {PART[p]}"
            rows.append([name,
                         num(q.leaf_single) if q is not None else "--",
                         num(r.leaf_single_reg) if r is not None else "--",
                         num(q.leaf_lt5) if q is not None else "--",
                         num(r.leaf_lt5_reg) if r is not None else "--"])
        blocks.append((title, rows))
    header = [r" & \multicolumn{2}{c}{One record} & \multicolumn{2}{c}{Fewer than 5 records} \\",
              r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
              r"Forests & Default & Regularized & Default & Regularized \\"]
    return tabular("lcccc", header, blocks)


def e10(R):
    t = pd.read_csv(R / "S_E10.csv")
    t = t[t.metric == "brier"]
    blocks = []
    for title, g in groups(t):
        rows = []
        for p in PARTS:
            for lvl in (1, 2):
                cells = [PART[p] if lvl == 1 else "", f"Level {lvl}"]
                for ev in ("shift_half", "shift_double"):
                    q = one(g, partition=p, level=lvl, eval=ev)
                    cells.append(share(q) if q is not None else "--")
                rows.append(cells)
        blocks.append((title, rows))
    return tabular("llcc", [r"Partition & Level & Odds halved & Odds doubled \\"], blocks)


def e11(R):
    sh = pd.read_csv(R / "S_E11_shares.csv")
    sh = sh[sh.metric == "brier"]
    gain = pd.read_csv(R / "S_E11_gain.csv")
    blocks = []
    for title, g in groups(sh):
        ds, lr = g.dataset.iloc[0], g.learner.iloc[0]
        rows = []
        for p in PARTS:
            cells = [PART[p]]
            for lvl in (1, 2):
                q = one(g, partition=p, level=lvl)
                cells.append(share(q) if q is not None else "--")
            q = one(gain, dataset=ds, learner=lr, partition=p)
            cells.append(ci(q.gain_mean, q.gain_lo, q.gain_hi) if q is not None else "--")
            rows.append(cells)
        blocks.append((title, rows))
    return tabular("lccc", [r"Partition & $r_1$, Brier & $r_2$, Brier & AUROC gain of Level 3 \\"],
                   blocks)


def e12(R):
    t = pd.read_csv(R / "S_E12.csv")
    t = t[t.metric == "brier"]
    blocks = []
    for title, g in groups(t):
        rows = []
        for n in N_CAL:
            h = g[g.n_cal.astype(str) == n]
            if not h.r.notna().any():  # n_cal larger than the calibration part
                continue
            label = (f"All ({h.n_cal_value.iloc[0]:,.0f})".replace(",", "{,}")
                     if n == "full" else n)
            cells = [label]
            for p in PARTS:
                q = one(h, partition=p)
                cells.append(share(q) if q is not None else "--")
            rows.append(cells)
        blocks.append((title, rows))
    return tabular("lccc", [f"$n_{{cal}}$ & {PART_HEADER} \\\\"], blocks)


def hp(R):
    d = json.loads((R / "hyperparameters.json").read_text())
    r = json.loads((R / "hyperparameters_regularized.json").read_text())

    def show(k, v):
        if k == "random_state":
            return "per model"
        if v is None:
            return "None"
        if isinstance(v, float) and math.isnan(v):
            return "NaN"
        return tex(v)

    blocks = []
    for lr in LEARNERS:
        rows = []
        for k in sorted(set(d[lr]) | set(r[lr])):
            a, b = d[lr].get(k), r[lr].get(k)
            if lr == "xgb" and a is None and b is None:  # None leaves the XGBoost default
                continue
            rows.append([rf"\texttt{{{tex(k)}}}", show(k, a), show(k, b)])
        blocks.append((LR[lr], rows))
    return tabular("lll", [r"Parameter & Default & Regularized \\"], blocks)


def build(R, out):
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "si_levels_brier": levels(R, "brier"),
        "si_levels_auroc": levels(R, "auroc"),
        "si_shares_brier": shares(R, "brier"),
        "si_shares_mcc": shares(R, "mcc_pi"),
        "si_calibration_citl": levels(R, "citl_int", src="S_calibration"),
        "si_calibration_slope": levels(R, "cal_slope", src="S_calibration"),
        **{f"si_l2_nonpositive_{PART_FILE[p]}": l2_nonpositive(R, p) for p in PARTS},
        "si_e5_brier": e5(R, "brier", 4),
        "si_e5_auroc": e5(R, "auroc", 3),
        "si_e5_mcc": e5(R, "mcc_pi", 3),
        "si_l1g": l1g(R),
        "si_leaves": leaves(R),
        "si_e10": e10(R),
        "si_e11": e11(R),
        "si_e12": e12(R),
        "si_hp": hp(R),
    }
    for name, body in tables.items():
        (out / f"{name}.tex").write_text(body)
    return list(tables)


def main(argv=None):
    root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=root / "results")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    out = a.out or a.results / "si_tables"
    names = build(a.results, out)
    print(f"{len(names)} tables written to {out}")


if __name__ == "__main__":
    main()
