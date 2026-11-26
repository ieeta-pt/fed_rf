import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from partition import partition, partition_rng, site_counts  # noqa: E402


def _labels(n=900, prev=0.2, seed=0):
    y = np.zeros(n, dtype=int)
    y[: int(n * prev)] = 1
    np.random.default_rng(seed).shuffle(y)
    return y


@pytest.mark.parametrize("kind", list(config.PARTITIONS))
@pytest.mark.parametrize("K", list(config.K_LIST))
def test_partition_properties(kind, K):
    y = _labels()
    code = config.PARTITIONS[kind][0]
    site = partition(y, K, kind, partition_rng(1, 0, 0, K, code))
    assert site.shape == y.shape
    assert set(np.unique(site)) == set(range(K))  # every record in exactly one site
    n, n_pos = site_counts(y, site, K)
    assert n.sum() == y.size and n_pos.sum() == y.sum()
    assert np.all(n_pos >= config.MIN_PER_CLASS)
    assert np.all(n - n_pos >= config.MIN_PER_CLASS)
    if kind == "equal":
        assert n_pos.max() - n_pos.min() <= 1
        assert (n - n_pos).max() - (n - n_pos).min() <= 1
    again = partition(y, K, kind, partition_rng(1, 0, 0, K, code))
    assert np.array_equal(site, again)


def test_partition_differs_across_seeds():
    y = _labels()
    a = partition(y, 5, "dir0.5", partition_rng(1, 0, 0, 5, 5))
    b = partition(y, 5, "dir0.5", partition_rng(1, 1, 0, 5, 5))
    assert not np.array_equal(a, b)


def test_dirichlet_too_few_records():
    y = _labels(n=100, prev=0.2)  # 20 positives < 10 * 5
    with pytest.raises(ValueError):
        partition(y, 10, "dir0.1", partition_rng(1, 0, 0, 10, 1))


def test_dirichlet_is_skewed():
    y = _labels(n=5000, prev=0.2)
    site = partition(y, 10, "dir0.1", partition_rng(1, 0, 0, 10, 1))
    n, n_pos = site_counts(y, site, 10)
    pi_k = n_pos / n
    assert pi_k.max() - pi_k.min() > 0.2


@pytest.mark.skipif(not (config.data_dir() / "heart_2022_no_nans.csv").exists(),
                    reason="Heart data not present")
def test_heart_features_without_hadangina():
    import data
    ds = data.load_heart()
    assert ds.X.shape[1] == 38 == config.HEART_N_FEATURES
    assert "HadAngina" not in ds.features and config.HEART_TARGET not in ds.features
