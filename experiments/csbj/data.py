"""Load the three datasets as float64 feature arrays and 0/1 labels.

    python experiments/csbj/data.py      prints sizes, writes the Heart cache and encoding map
"""
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

AIDS_DROP = ("cid", "time", "cd420", "cd820", "offtrt")


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    features: list
    # column index -> number of categories, for nominal columns (Heart only)
    nominal: dict = field(default_factory=dict)
    coltypes: dict = field(default_factory=dict)


def load_aids():
    df = pd.read_csv(config.data_dir() / "aids.csv")
    y = df["cid"].to_numpy().astype(int)
    X = df.drop(columns=list(AIDS_DROP))
    assert X.shape[1] == 19, X.shape
    return Dataset("aids", X.to_numpy(dtype=np.float64), y, list(X.columns))


def load_diabetic():
    df = pd.read_csv(config.data_dir() / "diabetic.csv")
    y = df["Class"].to_numpy().astype(int)
    X = df.drop(columns=["Class"])
    assert X.shape[1] == 19, X.shape
    return Dataset("diabetic", X.to_numpy(dtype=np.float64), y, list(X.columns))


def heart_maps(df, verbose=True):
    """Build the value -> code map of every non numeric Heart column from the
    distinct values found in the data. Stops on any value no map covers."""
    maps, kinds = {}, {}
    text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    for c in text_cols:
        values = sorted(df[c].unique().tolist())
        if verbose:
            print(f"  {c}: {values}")
        if c in config.HEART_ORDERED:
            order = config.HEART_ORDERED[c]
            m, kind = {v: i for i, v in enumerate(order)}, "ordered"
        elif c in config.HEART_NOMINAL:
            m, kind = {v: i for i, v in enumerate(values)}, "nominal"
        elif c in config.HEART_BINARY_EXTRA:
            m, kind = dict(config.HEART_BINARY_EXTRA[c]), "binary"
        elif set(values) <= {"No", "Yes"}:
            m, kind = {"No": 0, "Yes": 1}, "binary"
        else:
            raise SystemExit(f"Heart column {c!r} has no encoding rule. Stopping.")
        missing = [v for v in values if v not in m]
        if missing:
            raise SystemExit(f"Heart column {c!r}: values {missing} not covered. Stopping.")
        maps[c], kinds[c] = m, kind
    return maps, kinds


def write_heart_encoding(maps, kinds):
    out = config.results_dir()
    out.mkdir(parents=True, exist_ok=True)
    rows = [(c, kinds[c], v, code) for c, m in maps.items()
            for v, code in sorted(m.items(), key=lambda kv: kv[1])]
    path = out / "heart_encoding.csv"
    tmp = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows, columns=["column", "kind", "value", "code"]).to_csv(tmp, index=False)
    os.replace(tmp, path)


def encode_heart(verbose=True):
    df = pd.read_csv(config.data_dir() / "heart_2022_no_nans.csv")
    if verbose:
        print("Heart, distinct values of non numeric columns")
    maps, kinds = heart_maps(df, verbose)
    enc = pd.DataFrame(index=df.index)
    for c in df.columns:
        enc[c] = df[c].map(maps[c]) if c in maps else df[c]
        if enc[c].isna().any():
            raise SystemExit(f"Heart column {c!r} has missing values after encoding. Stopping.")
    y = enc.pop(config.HEART_TARGET).to_numpy().astype(int)
    X = enc.to_numpy(dtype=np.float64)
    features = list(enc.columns)
    assert X.shape[1] == 39, X.shape
    coltypes = {c: kinds.get(c, "numeric") for c in features}
    nominal = {features.index(c): len(maps[c]) for c in config.HEART_NOMINAL}
    write_heart_encoding(maps, kinds)
    path = config.data_dir() / "heart_encoded.npz"
    tmp = path.with_name("heart_encoded.tmp.npz")
    np.savez_compressed(tmp, X=X, y=y, features=np.array(features),
                        coltypes=np.array([coltypes[c] for c in features]),
                        nominal_idx=np.array(list(nominal.keys())),
                        nominal_n=np.array(list(nominal.values())))
    os.replace(tmp, path)
    return Dataset("heart", X, y, features, nominal, coltypes)


def load_heart(verbose=False):
    path = config.data_dir() / "heart_encoded.npz"
    if not path.exists():
        return encode_heart(verbose=True)
    z = np.load(path, allow_pickle=False)
    features = z["features"].tolist()
    coltypes = dict(zip(features, z["coltypes"].tolist()))
    nominal = dict(zip(z["nominal_idx"].tolist(), z["nominal_n"].tolist()))
    return Dataset("heart", z["X"], z["y"].astype(int), features, nominal, coltypes)


def load(name):
    if name == "aids":
        return load_aids()
    if name == "diabetic":
        return load_diabetic()
    if name == "heart":
        return load_heart()
    raise ValueError(name)


def lr_pipeline(nominal, seed):
    """Logistic regression for E13: one hot encoding of the nominal Heart
    columns, then StandardScaler, then LogisticRegression(max_iter=5000).
    ``nominal`` maps column index to number of categories (empty for AIDS and
    Diabetic). Fitted on the training rows of one site only."""
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    steps = []
    if nominal:
        idx = sorted(nominal)
        cats = [np.arange(nominal[i], dtype=np.float64) for i in idx]
        steps.append(("onehot", ColumnTransformer(
            [("nominal", OneHotEncoder(categories=cats, handle_unknown="ignore",
                                       sparse_output=False), idx)],
            remainder="passthrough")))
    steps.append(("scale", StandardScaler()))
    steps.append(("lr", LogisticRegression(random_state=seed, **config.LR_PARAMS)))
    return Pipeline(steps)


if __name__ == "__main__":
    for name in config.DATASETS:
        if name == "heart":
            ds = encode_heart(verbose=True)
        else:
            ds = load(name)
        print(f"{name}: X {ds.X.shape}, positives {int(ds.y.sum())} ({ds.y.mean():.4f})")
        if ds.nominal:
            print("  nominal columns:", {ds.features[i]: n for i, n in ds.nominal.items()})
