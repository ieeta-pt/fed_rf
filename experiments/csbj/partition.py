"""Split the training pool into K sites.

``partition(y, K, kind, rng)`` returns, for each record of the pool, the index
of its site (0 .. K-1).

* ``equal``. For each class, permute its records and deal them round robin.
* ``dir<alpha>``. For each class separately, permute its records, give
  ``MIN_PER_CLASS`` of them to every site, draw q ~ Dirichlet(alpha 1_K) and
  counts ~ Multinomial(remaining, q), and hand the remaining records out in
  consecutive blocks. Negative and positive fractions are drawn independently.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def partition_rng(ds_code, r, f, K, part_code):
    return np.random.default_rng([ds_code, r, f, K, part_code])


def partition(y, K, kind, rng, m=config.MIN_PER_CLASS):
    y = np.asarray(y).astype(int)
    if K < 1:
        raise ValueError("K must be at least 1")
    if kind not in config.PARTITIONS:
        raise ValueError(f"unknown partition {kind!r}")
    alpha = config.PARTITIONS[kind][1]
    site = np.full(y.size, -1, dtype=np.int64)
    for c in (0, 1):
        idx = np.flatnonzero(y == c)
        idx = idx[rng.permutation(idx.size)]
        if alpha is None:
            site[idx] = np.arange(idx.size) % K
            continue
        if idx.size < K * m:
            raise ValueError(f"class {c} has {idx.size} records, fewer than K*m = {K * m}")
        site[idx[: K * m]] = np.repeat(np.arange(K), m)
        rest = idx[K * m:]
        q = rng.dirichlet(np.full(K, alpha))
        counts = rng.multinomial(rest.size, q)
        site[rest] = np.repeat(np.arange(K), counts)
    assert np.all(site >= 0)
    return site


def site_counts(y, site, K):
    y = np.asarray(y).astype(int)
    n = np.bincount(site, minlength=K)
    n_pos = np.bincount(site, weights=y, minlength=K).astype(np.int64)
    return n, n_pos
