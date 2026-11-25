"""Figures from the CSV files written by ``analyze.py report``.

    python experiments/csbj/figures.py [--results DIR]

Writes PDF and PNG (300 dpi, final printed size) into <results>/figures.

    F1_<learner>_<partition>   improvement over level 0 per K (dir0.5 main, others supplementary)
    F2_shares                  r_1 and r_2 for Brier, reportable values only
    F3_gain_vs_site_size       AUROC gain of level 3 against n/K
    S_prevalence               pi_k of every site, r0 f0, K = 10
    S_E5                       weights and sampling against the default, fed_L0, dir0.5
    S_E10                      r_1 and r_2 for Brier under shifted prevalence, dir0.5
    S_E12                      r_2 for Brier against n_cal, dir0.5
    S_calibration              calibration slope and intercept per level, dir0.5
    S_reliability              reliability curves, r0 f0, K = 5, dir0.5, bins with >= 10 rows
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

TWO_COL, ONE_COL = 7.2, 3.5
GREEN = "#009E73"
STYLE = {  # level -> (color, linestyle, label)
    "m0": ("#999999", "-", "Level 0"),
    "m1": ("#0072B2", "-", "Level 1"),
    "m2": ("#E69F00", "-", "Level 2"),
    "m30": (GREEN, "-", "Level 3"),
    "m31": (GREEN, "--", "Level 3 with level 1"),
    "m32": (GREEN, ":", "Level 3 with level 2"),
    "mc": ("#000000", "-", "Centralized"),
}
ARM_LEVEL = {"src_L0": "m0", "src_L1": "m1", "src_L2": "m2", "fed_L0": "m30",
             "fed_L1": "m31", "fed_L2": "m32", "central_L0": "mc"}
PART_STYLE = {"equal": ("#56B4E9", "o", "Equal"), "dir0.5": ("#D55E00", "s", "Dirichlet 0.5"),
              "dir0.1": ("#CC79A7", "^", "Dirichlet 0.1")}
DS_LABEL = {"aids": "AIDS", "diabetic": "Diabetic", "heart": "Heart", "synthetic": "Synthetic"}
DS_MARKER = {"aids": "o", "diabetic": "s", "heart": "^", "synthetic": "D"}
LR_LABEL = {"rf": "Random Forest", "xgb": "XGBoost"}
LR_SHORT = {"rf": "RF", "xgb": "XGB"}
MIN_BIN = 10  # reliability bins with fewer test rows are not drawn
PARTS = ("equal", "dir0.5", "dir0.1")


def setup():
    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "lines.linewidth": 1,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    })


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)


def letter(ax, i):
    ax.text(-0.02, 1.06, "abcdefghijklmnop"[i], transform=ax.transAxes, fontweight="bold",
            ha="right", va="bottom")


def plain_log_x(ax, decades=None):
    """Log x axis with plain numbers.

    Labels sit at 1, 2 and 5 times powers of 10 when the data span at most two
    decades, and at powers of 10 only when they span more, so labels never overlap.
    """
    ax.set_xscale("log")
    if decades is None:
        lo, hi = ax.dataLim.intervalx
        decades = np.log10(hi / lo) if lo > 0 and hi > lo else 0
    subs = (1.0, 2.0, 5.0) if decades <= 2 else (1.0,)
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=subs))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _datasets(df):
    return [d for d in ("aids", "diabetic", "heart", "synthetic") if d in set(df.dataset)]


# ----------------------------------------------------------------------------
# Main figures
# ----------------------------------------------------------------------------

def fig_F1(res, out, learner, partition):
    p = pd.read_csv(res / "perK_differences.csv")
    p = p[(p.learner == learner) & (p.partition == partition) & (p.kind == "diff")]
    dss = _datasets(p)
    if not dss:
        return
    fig, axes = plt.subplots(2, len(dss), figsize=(TWO_COL, 3.4), squeeze=False)
    for j, ds in enumerate(dss):
        for i, (metric, sign, ylab) in enumerate((("brier", -1, "Decrease in Brier score\nfrom level 0"),
                                                   ("auroc", 1, "AUROC gain\nover level 0"))):
            ax = axes[i, j]
            ax.axhline(0, color=STYLE["m0"][0], lw=1)
            g = p[(p.dataset == ds) & (p.metric == metric)]
            for lv in ("m1", "m2", "m30", "m31", "m32", "mc"):
                h = g[g.level == lv].sort_values("K")
                if not len(h):
                    continue
                c, ls, _ = STYLE[lv]
                y, lo, hi = sign * h["mean"], sign * h.lo, sign * h.hi
                ax.plot(h.K, y, color=c, ls=ls)
                ax.fill_between(h.K, np.minimum(lo, hi), np.maximum(lo, hi), color=c, alpha=0.12, lw=0)
            ax.set_xticks(list(config.K_LIST))
            if i == 1:
                ax.set_xlabel("Number of sites K")
            if j == 0:
                ax.set_ylabel(ylab)
            letter(ax, i * len(dss) + j)
            if i == 0:
                ax.text(0.5, 1.02, DS_LABEL[ds], transform=ax.transAxes, ha="center", va="bottom")
    handles = [Line2D([], [], color=STYLE[lv][0], ls=STYLE[lv][1], label=STYLE[lv][2])
               for lv in ("m1", "m2", "m30", "m31", "m32", "mc")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1), h_pad=1.2, w_pad=1.0)
    save(fig, out, f"F1_{learner}_{partition}")


def _share_panel(ax, t, rows, level, groups, group_style, offsets):
    for gi, gname in enumerate(groups):
        color, marker, _ = group_style[gname]
        for ri, (ds, lr) in enumerate(rows):
            q = t[(t.dataset == ds) & (t.learner == lr) & (t.level == level) & (t["_g"] == gname)]
            if not len(q) or not bool(q.reportable.iloc[0]):
                continue
            q = q.iloc[0]
            y = ri + offsets[gi]
            ax.errorbar(q.r, y, xerr=[[q.r - q.r_lo], [q.r_hi - q.r]], fmt=marker, color=color,
                        ms=3, elinewidth=0.8, capsize=0)
    ax.axvline(0, color="#999999", lw=0.6)
    ax.axvline(1, color="#999999", lw=0.6, ls=":")
    ax.set_yticks(range(len(rows)))
    ax.set_ylim(len(rows) - 0.5, -0.5)


def _rows(t):
    return [(ds, lr) for ds in _datasets(t) for lr in ("rf", "xgb")
            if len(t[(t.dataset == ds) & (t.learner == lr)])]


def fig_F2(res, out):
    t = pd.read_csv(res / "T2_shares.csv")
    t = t[t.metric == "brier"].assign(_g=lambda d: d.partition)
    rows = _rows(t)
    fig, axes = plt.subplots(1, 2, figsize=(ONE_COL, 0.35 * len(rows) + 1.0), sharey=True)
    for k, (ax, lvl) in enumerate(zip(axes, (1, 2))):
        _share_panel(ax, t, rows, lvl, PARTS, PART_STYLE, (-0.2, 0.0, 0.2))
        ax.set_xlabel(f"$r_{lvl}$, Brier score")
        letter(ax, k)
    axes[0].set_yticklabels([f"{DS_LABEL[d]} {LR_SHORT[lr]}" for d, lr in rows])
    handles = [Line2D([], [], color=c, marker=m, ls="", ms=3, label=l) for c, m, l in PART_STYLE.values()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.8)
    save(fig, out, "F2_shares")


def fig_F3(res, out):
    t = pd.read_csv(res / "T3_auroc_gain.csv")
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 2.6), sharey=True)
    for k, (ax, lr) in enumerate(zip(axes, ("rf", "xgb"))):
        g = t[t.learner == lr]
        for ds in _datasets(g):
            for part in PARTS:
                h = g[(g.dataset == ds) & (g.partition == part)]
                if not len(h):
                    continue
                c = PART_STYLE[part][0]
                ax.errorbar(h.n_over_K, h["mean"], yerr=[h["mean"] - h.lo, h.hi - h["mean"]],
                            fmt=DS_MARKER[ds], color=c, ms=3, elinewidth=0.6, capsize=0,
                            mfc="none" if part == "equal" else c)
        plain_log_x(ax)
        ax.axhline(0, color="#999999", lw=0.6)
        ax.set_xlabel("Mean site size n/K (records, log scale)")
        if k == 0:
            ax.set_ylabel("AUROC gain of level 3 over level 0")
        ax.text(0.5, 1.02, LR_LABEL[lr], transform=ax.transAxes, ha="center", va="bottom")
        letter(ax, k)
    dsh = [Line2D([], [], color="#555555", marker=DS_MARKER[d], ls="", ms=3, label=DS_LABEL[d])
           for d in _datasets(t)]
    ph = [Line2D([], [], color=c, marker="o", ls="", ms=3, label=l, mfc="none" if p == "equal" else c)
          for p, (c, _, l) in PART_STYLE.items()]
    fig.legend(handles=dsh + ph, loc="lower center", ncol=len(dsh) + len(ph), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.0)
    save(fig, out, "F3_gain_vs_site_size")


# ----------------------------------------------------------------------------
# Supplementary figures
# ----------------------------------------------------------------------------

def fig_prevalence(res, out):
    s = pd.read_csv(res / "sites_r0_f0_K10.csv")
    dss = _datasets(s)
    if not dss:
        return
    fig, axes = plt.subplots(len(dss), 3, figsize=(TWO_COL, 1.5 * len(dss) + 0.4), squeeze=False)
    for i, ds in enumerate(dss):
        for j, part in enumerate(PARTS):
            ax = axes[i, j]
            g = s[(s.dataset == ds) & (s.partition == part)].sort_values("site")
            pi = g.pos_k.sum() / g.n_k.sum()
            ax.bar(g.site + 1, g.pi_k, color=PART_STYLE[part][0], width=0.7)
            ax.axhline(pi, color="#000000", lw=0.8, ls="--")
            ax.set_ylim(0, 1)
            ax.set_xticks(range(1, 11))
            if i == len(dss) - 1:
                ax.set_xlabel("Site")
            if j == 0:
                ax.set_ylabel(f"{DS_LABEL[ds]}\nprevalence $\\pi_k$")
            if i == 0:
                ax.text(0.5, 1.04, PART_STYLE[part][2], transform=ax.transAxes, ha="center", va="bottom")
            letter(ax, i * 3 + j)
    fig.tight_layout(h_pad=0.8)
    save(fig, out, "S_prevalence")


def fig_E5(res, out, partition="dir0.5"):
    e = pd.read_csv(res / "S_E5.csv")
    e = e[(e.arm == "fed_L0") & (e.partition == partition)]
    keys = e[["dataset", "learner", "weights", "sampling"]].drop_duplicates()
    keys = keys.assign(_d=keys.dataset.map({d: i for i, d in enumerate(_datasets(e))}))
    keys = keys.sort_values(["_d", "learner", "weights", "sampling"]).reset_index(drop=True)
    if not len(keys):
        return
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COL, 0.17 * len(keys) + 0.9), sharey=True)
    for k, (ax, (metric, lab)) in enumerate(zip(axes, (("brier", "Difference in Brier score"),
                                                       ("auroc", "Difference in AUROC"),
                                                       ("mcc_pi", "Difference in MCC")))):
        for ri, q in keys.iterrows():
            h = e[(e.dataset == q.dataset) & (e.learner == q.learner) & (e.weights == q.weights)
                  & (e.sampling == q.sampling) & (e.metric == metric)]
            if len(h):
                h = h.iloc[0]
                ax.errorbar(h["mean"], ri, xerr=[[h["mean"] - h.lo], [h.hi - h["mean"]]], fmt="o",
                            color=GREEN, ms=2.5, elinewidth=0.8, capsize=0)
        ax.axvline(0, color="#999999", lw=0.6)
        ax.set_xlabel(lab)
        letter(ax, k)
    axes[0].set_yticks(range(len(keys)))
    axes[0].set_yticklabels([f"{DS_LABEL[q.dataset]} {LR_SHORT[q.learner]}, {q.weights}"
                             + ("" if q.sampling == "na" else f", {q.sampling}")
                             for _, q in keys.iterrows()])
    axes[0].set_ylim(len(keys) - 0.5, -0.5)
    fig.tight_layout(w_pad=0.8)
    save(fig, out, "S_E5")


def fig_E10(res, out, partition="dir0.5"):
    t = pd.read_csv(res / "S_E10.csv")
    t = t[t.partition == partition].assign(_g=lambda d: d["eval"])
    rows = _rows(t)
    if not rows:
        return
    st = {"shift_half": ("#000000", "v", "Prevalence odds halved"),
          "shift_double": ("#000000", "^", "Prevalence odds doubled")}
    fig, axes = plt.subplots(1, 2, figsize=(ONE_COL, 0.35 * len(rows) + 1.0), sharey=True)
    for k, (ax, lvl) in enumerate(zip(axes, (1, 2))):
        _share_panel(ax, t, rows, lvl, list(st), st, (-0.12, 0.12))
        ax.set_xlabel(f"$r_{lvl}$, Brier score")
        letter(ax, k)
    axes[0].set_yticklabels([f"{DS_LABEL[d]} {LR_SHORT[lr]}" for d, lr in rows])
    handles = [Line2D([], [], color=c, marker=m, ls="", ms=3, label=l) for c, m, l in st.values()]
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=0.8)
    save(fig, out, "S_E10")


def fig_E12(res, out, partition="dir0.5"):
    t = pd.read_csv(res / "S_E12.csv")
    t = t[(t.partition == partition) & t.reportable.astype(bool)]
    dss = _datasets(t)
    if not dss:
        return
    fig, axes = plt.subplots(1, len(dss), figsize=(TWO_COL, 2.2), squeeze=False)
    lr_style = {"rf": ("#E69F00", "o"), "xgb": ("#E69F00", "s")}
    for k, ds in enumerate(dss):
        ax = axes[0, k]
        for lr, (c, m) in lr_style.items():
            h = t[(t.dataset == ds) & (t.learner == lr)].sort_values("n_cal_value")
            if not len(h):
                continue
            ax.errorbar(h.n_cal_value * (0.96 if lr == "rf" else 1.04), h.r,
                        yerr=[h.r - h.r_lo, h.r_hi - h.r], fmt=m + ("-" if lr == "rf" else "--"),
                        color=c, ms=3, elinewidth=0.6, capsize=0,
                        mfc=c if lr == "rf" else "white", label=LR_LABEL[lr])
        plain_log_x(ax)
        ax.axhline(1, color="#999999", lw=0.6, ls=":")
        ax.axhline(0, color="#999999", lw=0.6)
        ax.set_xlabel("Labeled records for level 2, $n_{cal}$")
        if k == 0:
            ax.set_ylabel("$r_2$, Brier score")
        ax.text(0.5, 1.02, DS_LABEL[ds], transform=ax.transAxes, ha="center", va="bottom")
        letter(ax, k)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=1.0)
    save(fig, out, "S_E12")


def fig_calibration(res, out, partition="dir0.5"):
    t = pd.read_csv(res / "S_calibration.csv")
    t = t[(t.partition == partition) & t.level.isin(list(STYLE))]
    rows = _rows(t)
    if not rows:
        return
    levels = ["m0", "m1", "m2", "m30", "m31", "m32", "mc"]
    off = np.linspace(-0.3, 0.3, len(levels))
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 0.4 * len(rows) + 1.0), sharey=True)
    for k, (ax, (metric, lab, ref)) in enumerate(zip(axes, (("cal_slope", "Calibration slope", 1),
                                                            ("citl_int", "Calibration in the large (intercept)", 0)))):
        for ri, (ds, lr) in enumerate(rows):
            for li, lv in enumerate(levels):
                q = t[(t.dataset == ds) & (t.learner == lr) & (t.metric == metric) & (t.level == lv)]
                if not len(q):
                    continue
                q = q.iloc[0]
                c, ls, _ = STYLE[lv]
                mfc = "white" if ls in ("--", ":") else c
                mk = {"-": "o", "--": "s", ":": "D"}[ls]
                ax.errorbar(q["mean"], ri + off[li], xerr=[[q["mean"] - q.lo], [q.hi - q["mean"]]],
                            fmt=mk, color=c, mfc=mfc, ms=2.5, elinewidth=0.6, capsize=0)
        ax.axvline(ref, color="#999999", lw=0.6)
        ax.set_xlabel(lab)
        letter(ax, k)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([f"{DS_LABEL[d]} {LR_SHORT[lr]}" for d, lr in rows])
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    handles = []
    for lv in levels:
        c, ls, lab = STYLE[lv]
        handles.append(Line2D([], [], color=c, marker={"-": "o", "--": "s", ":": "D"}[ls], ls="",
                              mfc="white" if ls in ("--", ":") else c, ms=3, label=lab))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=1.0)
    save(fig, out, "S_calibration")


def fig_reliability(res, out):
    p = res / "reliability.csv"
    if not p.exists() or p.stat().st_size < 10:
        return
    t = pd.read_csv(p)
    t = t[t.n >= MIN_BIN]
    dss = _datasets(t)
    if not dss:
        return
    fig, axes = plt.subplots(2, len(dss), figsize=(TWO_COL, 4.4), squeeze=False)
    for j, ds in enumerate(dss):
        for i, lr in enumerate(("rf", "xgb")):
            ax = axes[i, j]
            ax.plot([0, 1], [0, 1], color="#999999", lw=0.6)
            for arm, lv in ARM_LEVEL.items():
                h = t[(t.dataset == ds) & (t.learner == lr) & (t.arm == arm)].sort_values("bin")
                if len(h):
                    c, ls, _ = STYLE[lv]
                    ax.plot(h.p_mean, h.y_mean, color=c, ls=ls, marker="o", ms=1.5)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            if i == 1:
                ax.set_xlabel("Mean predicted probability")
            if j == 0:
                ax.set_ylabel(f"{LR_LABEL[lr]}\nobserved frequency")
            if i == 0:
                ax.text(0.5, 1.03, DS_LABEL[ds], transform=ax.transAxes, ha="center", va="bottom")
            letter(ax, i * len(dss) + j)
    handles = [Line2D([], [], color=STYLE[lv][0], ls=STYLE[lv][1], label=STYLE[lv][2])
               for lv in ("m0", "m1", "m2", "m30", "m31", "m32", "mc")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, out, "S_reliability")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(config.results_dir()))
    a = ap.parse_args(argv)
    res = Path(a.results)
    out = res / "figures"
    setup()
    for lr in ("rf", "xgb"):
        for part in PARTS:
            fig_F1(res, out, lr, part)
    fig_F2(res, out)
    fig_F3(res, out)
    fig_prevalence(res, out)
    fig_E5(res, out)
    fig_E10(res, out)
    fig_E12(res, out)
    fig_calibration(res, out)
    fig_reliability(res, out)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
