import numpy as np
import pandas as pd

MOCK_ROWS = 100
MOCK_SEED = 0


def generate_mock(data: pd.DataFrame, n_rows: int = MOCK_ROWS, seed: int = MOCK_SEED) -> pd.DataFrame:
    """Synthetic mock with the columns and dtypes of ``data``.

    Values are drawn independently for each column with a fixed seed: numeric
    columns uniformly between their minimum and maximum (integers inclusive),
    other columns (including booleans) uniformly over their observed values.
    The number of rows is fixed (``n_rows``) and does not depend on ``data``.
    No row of ``data`` is copied.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for col in data.columns:
        s = data[col]
        dtype = s.dtype
        if pd.api.types.is_bool_dtype(dtype) or not pd.api.types.is_numeric_dtype(dtype):
            values = pd.unique(s.dropna())
            if len(values) == 0:
                out[col] = pd.Series([None] * n_rows, dtype=dtype)
            else:
                out[col] = pd.Series(values[rng.integers(0, len(values), size=n_rows)]).astype(dtype)
            continue
        valid = s.dropna()
        if valid.empty:
            out[col] = pd.Series(np.full(n_rows, np.nan)).astype(dtype)
            continue
        lo, hi = valid.min(), valid.max()
        if pd.api.types.is_integer_dtype(dtype):
            vals = rng.integers(int(lo), int(hi), size=n_rows, endpoint=True)
        else:
            vals = rng.uniform(float(lo), float(hi), size=n_rows)
        out[col] = pd.Series(vals).astype(dtype)
    return pd.DataFrame(out, columns=data.columns)
