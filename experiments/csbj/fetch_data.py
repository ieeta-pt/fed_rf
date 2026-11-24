"""Download AIDS (UCI 890) and Diabetic Retinopathy Debrecen (UCI 329) and
check the Heart file that is downloaded by hand from Kaggle.

Usage: python experiments/csbj/fetch_data.py
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

DIABETIC_ZIP = ("https://archive.ics.uci.edu/static/public/329/"
                "diabetic+retinopathy+debrecen.zip")
DIABETIC_COLUMNS = (["quality", "pre_screening"]
                    + [f"ma{i}" for i in range(1, 7)]
                    + [f"exudate{i}" for i in range(1, 9)]
                    + ["macula_opticdisc_distance", "opticdisc_diameter",
                       "am_fm_classification"])


def check(name, got, expected):
    if got != expected:
        raise SystemExit(f"{name}: expected {expected}, found {got}. Stopping.")
    print(f"  {name}: {got} (ok)")


def fetch_aids(out):
    from ucimlrepo import fetch_ucirepo
    d = fetch_ucirepo(id=890)
    X = d.data.features.copy()
    y = d.data.targets["cid"].astype(int)
    print("AIDS")
    check("rows", len(X), 2139)
    check("features", X.shape[1], 23)
    check("positives", int(y.sum()), 521)
    if "pidnum" in X.columns:
        raise SystemExit("AIDS: patient ID among the features. Stopping.")
    df = X.assign(cid=y.values)
    df.to_csv(out / "aids.csv", index=False)


def parse_arff(text):
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip().lower() == "@data")
    rows = [ln.split(",") for ln in lines[start + 1:] if ln.strip() and not ln.startswith("%")]
    return pd.DataFrame(rows).astype(float)


def fetch_diabetic(out):
    with urllib.request.urlopen(DIABETIC_ZIP, timeout=60) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    name = next(n for n in z.namelist() if n.endswith(".arff"))
    raw = parse_arff(z.read(name).decode("utf-8"))
    print("Diabetic")
    check("columns", raw.shape[1], 20)
    raw.columns = DIABETIC_COLUMNS + ["Class"]
    raw["Class"] = raw["Class"].astype(int)
    check("rows", len(raw), 1151)
    check("features", raw.shape[1] - 1, 19)
    check("positives", int(raw["Class"].sum()), 611)
    if (raw["exudate3"] == raw["exudate4"]).all():
        raise SystemExit("Diabetic: exudate4 equals exudate3. Stopping.")
    n_dup = int(raw.duplicated(keep="first").sum())
    n_dup_x = int(raw[DIABETIC_COLUMNS].duplicated(keep="first").sum())
    (out / "diabetic_duplicates.txt").write_text(
        f"exact duplicate rows (features and Class), counted beyond the first copy: {n_dup}\n"
        f"duplicate feature vectors, ignoring Class: {n_dup_x}\n"
        "All rows are kept.\n")
    print(f"  exact duplicate rows: {n_dup} (kept), duplicate feature vectors: {n_dup_x}")
    raw.to_csv(out / "diabetic.csv", index=False)


def check_heart(out):
    path = out / "heart_2022_no_nans.csv"
    print("Heart")
    if not path.exists():
        raise SystemExit(f"{path} is missing. Download it from Kaggle (see INSTRUCOES_E1.md, section 10).")
    df = pd.read_csv(path)
    check("rows", len(df), 246022)
    check("columns", df.shape[1], 40)
    pos = int((df[config.HEART_TARGET] == "Yes").sum())
    print(f"  positives: {pos} ({pos / len(df):.4f})")


def main():
    out = config.data_dir()
    out.mkdir(parents=True, exist_ok=True)
    fetch_aids(out)
    fetch_diabetic(out)
    check_heart(out)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
