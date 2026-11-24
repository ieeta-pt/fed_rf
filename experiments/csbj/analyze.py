"""Analysis of the run files.

    python experiments/csbj/analyze.py summary [--runs DIR] [--runs-reg DIR]

Levels per dataset, learner, partition, K and draw (default configuration:
equal weights, uniform sampling, eval test, full calibration part)

    m0, m1, m2      means over sources of src_L0, src_L1, src_L2
    m30, m31, m32   fed_L0, fed_L1, fed_L2      (m31g: fed_L1g, sensitivity)
    mc, mc2         central_L0, central_L2

Values are averaged over K with equal weight per K, which gives one value per
draw. Shares of the level 3 change recovered by a cheaper level

    r_l        = mean(m_l - m0) / mean(m3l - m0)      (main definition, l = 1, 2)
    r_vs_plain = mean(m_l - m0) / mean(m30 - m0)      (secondary, uncorrected federated model)

with percentile bootstrap intervals over draws that recompute the ratio in
each resample. r_l is reportable when the interval of mean(m3l - m0) excludes 0.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402

B_BOOT = 2000
BOOT_SEED = 20251209
LEVELS = ["m0", "m1", "m2", "m30", "m31", "m31g", "m32", "mc", "mc2"]
ARM_OF = {"m0": "src_L0", "m1": "src_L1", "m2": "src_L2", "m30": "fed_L0",
          "m31": "fed_L1", "m31g": "fed_L1g", "m32": "fed_L2",
          "mc": "central_L0", "mc2": "central_L2"}
DRAW = ["rep", "fold"]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_runs(runs_root, datasets=config.DATASETS):
    """Concatenate every metric file (cells and central) and site file."""
    runs_root = Path(runs_root)
    metrics, sites, central_sites = [], [], []
    for ds in datasets:
        for d in sorted((runs_root / ds).glob("r*_f*")):
            for p in sorted(d.glob("*.parquet")):
                if p.name == "central_sites.parquet":
                    central_sites.append(pd.read_parquet(p))
                elif p.name.endswith("_sites.parquet"):
                    sites.append(pd.read_parquet(p))
                else:
                    metrics.append(pd.read_parquet(p))
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()  # noqa: E731
    return cat(metrics), cat(sites), cat(central_sites)


def error_files(runs_root):
    return sorted(Path(runs_root).glob("*/r*_f*/*.error.txt"))


# ----------------------------------------------------------------------------
# Default view and levels per draw
# ----------------------------------------------------------------------------

def default_view(df, ev="test"):
    """Default configuration rows, full calibration part for level 2."""
    v = df[(df["eval"] == ev) & df.weights.isin(["na", config.DEFAULT_WEIGHTS])
           & df.sampling.isin(["na", config.DEFAULT_SAMPLING])].copy()
    l2 = v.arm.str.endswith("_L2")
    full = v[l2].groupby(["dataset"] + DRAW).n_cal.transform("max")
    keep = ~l2
    keep[l2] = v.loc[l2, "n_cal"].to_numpy() == full.to_numpy()
    return v[keep]


def levels_per_cell(v, metric, source="mean"):
    """One row per dataset, learner, partition, K and draw with m0 .. mc2.

    source="mean" averages the source arms over sources, source="largest"
    keeps the source with the largest n_k (ties to the lower index, E11)."""
    cells = v[v.partition != "central"]
    central = v[v.partition == "central"]
    key = ["dataset", "learner", "partition", "K"] + DRAW
    src = cells[cells.arm.str.startswith("src_")]
    if source == "largest":
        best = (src[src.arm == "src_L0"].sort_values(key + ["n_k", "source"],
                                                     ascending=[True] * len(key) + [False, True])
                .groupby(key).head(1)[key + ["source"]])
        src = src.merge(best, on=key + ["source"])
    out = src.pivot_table(index=key, columns="arm", values=metric, aggfunc="mean")
    fed = cells[cells.arm.str.startswith("fed_")].pivot_table(
        index=key, columns="arm", values=metric, aggfunc="mean")
    out = out.join(fed, how="left")
    cen = central.pivot_table(index=["dataset", "learner"] + DRAW, columns="arm",
                              values=metric, aggfunc="mean")
    out = out.reset_index().merge(cen.reset_index(), on=["dataset", "learner"] + DRAW, how="left")
    for lv, arm in ARM_OF.items():
        out[lv] = out[arm] if arm in out else np.nan
    return out[key + LEVELS]


def average_over_K(cells):
    """Equal weight per K, one row per dataset, learner, partition and draw."""
    return (cells.groupby(["dataset", "learner", "partition"] + DRAW)[LEVELS]
            .mean().reset_index())


# ----------------------------------------------------------------------------
# Intervals
# ----------------------------------------------------------------------------

def _boot_idx(J, B=B_BOOT, seed=BOOT_SEED):
    return np.random.default_rng(seed).integers(0, J, size=(B, J))


def mean_ci(x, B=B_BOOT, seed=BOOT_SEED):
    """Mean, 95% percentile bootstrap interval over draws, and the corrected
    resampled t interval (variance factor 1/J + 1/4)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    J = x.size
    if J == 0:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, nb_lo=np.nan, nb_hi=np.nan, J=0)
    m = float(x.mean())
    if J == 1:
        return dict(mean=m, lo=np.nan, hi=np.nan, nb_lo=np.nan, nb_hi=np.nan, J=1)
    bm = x[_boot_idx(J, B, seed)].mean(axis=1)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    se = np.sqrt((1.0 / J + 0.25) * x.var(ddof=1))
    t = stats.t.ppf(0.975, J - 1)
    return dict(mean=m, lo=float(lo), hi=float(hi), nb_lo=m - t * se, nb_hi=m + t * se, J=J)


def ratio_ci(num, den, B=B_BOOT, seed=BOOT_SEED):
    """mean(num) / mean(den) with a percentile bootstrap interval that
    recomputes the ratio in each resample of draws."""
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    ok = np.isfinite(num) & np.isfinite(den)
    num, den = num[ok], den[ok]
    J = num.size
    if J == 0 or den.mean() == 0:
        return dict(r=np.nan, lo=np.nan, hi=np.nan)
    r = float(num.mean() / den.mean())
    if J == 1:
        return dict(r=r, lo=np.nan, hi=np.nan)
    idx = _boot_idx(J, B, seed)
    with np.errstate(divide="ignore", invalid="ignore"):
        br = num[idx].mean(axis=1) / den[idx].mean(axis=1)
    br = br[np.isfinite(br)]
    lo, hi = np.percentile(br, [2.5, 97.5])
    return dict(r=r, lo=float(lo), hi=float(hi))


def _excludes_zero(ci):
    return bool(np.isfinite(ci["lo"]) and (ci["lo"] > 0 or ci["hi"] < 0))


def shares(per_draw, metric_name):
    """r_1 and r_2 (with m3l) and r_vs_plain (with m30) per dataset, learner and partition."""
    rows = []
    for (ds, lr, part), g in per_draw.groupby(["dataset", "learner", "partition"]):
        if g["m30"].isna().all():
            continue
        for l in (1, 2):
            num = g[f"m{l}"] - g["m0"]
            den = g[f"m3{l}"] - g["m0"]
            den_plain = g["m30"] - g["m0"]
            r = ratio_ci(num, den)
            rp = ratio_ci(num, den_plain)
            ci_den = mean_ci(den)
            ci_plain = mean_ci(den_plain)
            rows.append(dict(
                dataset=ds, learner=lr, partition=part, metric=metric_name, level=l,
                r=r["r"], r_lo=r["lo"], r_hi=r["hi"],
                reportable=_excludes_zero(ci_den),
                r_vs_plain=rp["r"], r_vs_plain_lo=rp["lo"], r_vs_plain_hi=rp["hi"],
                reportable_vs_plain=_excludes_zero(ci_plain),
                num=float(num.mean()), den=float(den.mean()), den_plain=float(den_plain.mean()),
                J=int(num.notna().sum())))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Comparison of two hyperparameter configurations
# ----------------------------------------------------------------------------

def leaf_summary(sites, central_sites):
    rows = []
    if len(sites):
        for (ds, part), g in sites.groupby(["dataset", "partition"]):
            rows.append(dict(dataset=ds, partition=part, model="site",
                             rf=g.rf_leaf_single.sum() / g.rf_leaf_n.sum(),
                             xgb=g.xgb_leaf_single.sum() / g.xgb_leaf_n.sum()))
    if len(central_sites):
        for ds, g in central_sites.groupby("dataset"):
            rows.append(dict(dataset=ds, partition="central", model="central",
                             rf=g.rf_leaf_single.sum() / g.rf_leaf_n.sum(),
                             xgb=g.xgb_leaf_single.sum() / g.xgb_leaf_n.sum()))
    return pd.DataFrame(rows)


def key_quantities(metrics, partitions=None):
    """Brier shares, AUROC gain of level 3 and centralized AUROC and Brier,
    per dataset, learner and partition, averaged over K and draws."""
    v = default_view(metrics)
    if partitions:
        v = v[v.partition.isin(list(partitions) + ["central"])]
    out = []
    for metric in ("brier", "auroc"):
        pd_ = average_over_K(levels_per_cell(v, metric))
        g = pd_.groupby(["dataset", "learner", "partition"])
        s = g[LEVELS].mean()
        s.columns = [f"{metric}_{c}" for c in s.columns]
        out.append(s)
    res = pd.concat(out, axis=1).reset_index()
    res["auroc_gain_L3"] = res["auroc_m30"] - res["auroc_m0"]
    res["auroc_central_minus_L3"] = res["auroc_mc"] - res["auroc_m30"]
    sh = shares(average_over_K(levels_per_cell(v, "brier")), "brier")
    if len(sh):
        wide = sh.pivot_table(index=["dataset", "learner", "partition"], columns="level",
                              values=["r", "r_vs_plain"])
        wide.columns = [f"brier_{a}{b}" for a, b in wide.columns]
        res = res.merge(wide.reset_index(), on=["dataset", "learner", "partition"], how="left")
    return res


def compare_hp(runs_default, runs_reg, partitions=None):
    """Side by side key quantities and leaf fractions of the two configurations."""
    md, sd, cd = load_runs(runs_default)
    mr, sr, cr = load_runs(runs_reg)
    kd = key_quantities(md, partitions).assign(hp="default")
    kr = key_quantities(mr, partitions).assign(hp="regularized")
    ld = leaf_summary(sd, cd).assign(hp="default")
    lr = leaf_summary(sr, cr).assign(hp="regularized")
    return pd.concat([kd, kr], ignore_index=True), pd.concat([ld, lr], ignore_index=True)


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------

SUMMARY_COLS = ["m0", "m1", "m2", "m30", "m31", "m31g", "m32", "mc"]


def print_summary(runs, partitions=("dir0.5",)):
    m, sites, csites = load_runs(runs)
    v = default_view(m)
    pd.set_option("display.width", 250)
    for metric in ("brier", "auroc"):
        pdraw = average_over_K(levels_per_cell(v, metric))
        tab = pdraw[pdraw.partition.isin(partitions)].groupby(
            ["dataset", "learner", "partition"])[SUMMARY_COLS].mean()
        print(f"\n{metric}, mean over K and draws")
        print(tab.round(4).to_string())
    sh = shares(average_over_K(levels_per_cell(v, "brier")), "brier")
    sh = sh[sh.partition.isin(partitions)]
    print("\nBrier shares (r with m3l, r_vs_plain with m30)")
    print(sh[["dataset", "learner", "partition", "level", "r", "r_vs_plain", "num", "den",
              "den_plain", "J"]].round(3).to_string(index=False))
    print("\nleaf fractions")
    print(leaf_summary(sites, csites).round(3).to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["summary", "compare-hp"])
    ap.add_argument("--runs", default=str(config.runs_dir()))
    ap.add_argument("--runs-reg", default=str(config.runs_dir("regularized")))
    ap.add_argument("--partitions", nargs="+", default=["dir0.5"])
    a = ap.parse_args(argv)
    if a.command == "summary":
        print_summary(a.runs, tuple(a.partitions))
    else:
        k, leaves = compare_hp(a.runs, a.runs_reg, a.partitions)
        cols = ["dataset", "learner", "partition", "hp", "brier_r1", "brier_r2",
                "brier_r_vs_plain1", "brier_r_vs_plain2", "auroc_gain_L3", "auroc_mc",
                "auroc_m30", "auroc_m0", "brier_mc", "brier_m30"]
        pd.set_option("display.width", 250)
        print(k[[c for c in cols if c in k]].sort_values(["dataset", "learner", "partition", "hp"])
              .round(3).to_string(index=False))
        print(leaves.sort_values(["dataset", "partition", "hp"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
