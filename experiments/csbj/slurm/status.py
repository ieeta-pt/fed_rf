"""Expected, present and failed outputs of the E1 runs.

    python experiments/csbj/slurm/status.py [--r-rep 4] [--hp default|regularized] [--list]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def expected_files(task_dir):
    names = ["central.parquet", "central_sites.parquet", "meta.json"]
    for part in config.PARTITIONS:
        for K in config.K_LIST:
            names += [f"{part}_K{K}.parquet", f"{part}_K{K}_sites.parquet"]
    return [task_dir / n for n in names]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--r-rep", type=int, default=config.R_REP)
    ap.add_argument("--hp", default="default", choices=list(config.HP))
    ap.add_argument("--list", action="store_true", help="list the missing files")
    a = ap.parse_args(argv)
    runs = config.runs_dir(a.hp)
    print(f"runs: {runs}")
    total_missing = total_err = 0
    for ds in config.DATASETS:
        done = partial = 0
        missing, errors = [], []
        for tid in range(5 * a.r_rep):
            r, f = divmod(tid, 5)
            d = runs / ds / f"r{r}_f{f}"
            exp = expected_files(d)
            miss = [p for p in exp if not p.exists()]
            errs = sorted(d.glob("*.error.txt")) if d.exists() else []
            errors += errs
            if not miss and not errs:
                done += 1
            elif len(miss) < len(exp):
                partial += 1
            missing += miss
        n = 5 * a.r_rep
        print(f"{ds:9s} tasks complete {done}/{n}, partial {partial}, "
              f"missing files {len(missing)}, error files {len(errors)}")
        for e in errors:
            print(f"  ERROR {e}")
        if a.list:
            for m in missing:
                print(f"  missing {m}")
        total_missing += len(missing)
        total_err += len(errors)
    return 1 if total_err else 0


if __name__ == "__main__":
    sys.exit(main())
