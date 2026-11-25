"""Model aggregation with the rule of the disclosure ladder (``fed_rf_mk/ladder.py``).

Random Forest. Site k gets t_k trees from ``ladder.allocate_trees`` (largest
remainder), chosen with ``ladder.select_trees`` from one seeded numpy
generator, uniformly or in proportion to the out of bag score of each tree.

XGBoost. The site models are kept as ensemble members, and the deployment site
averages their log odds with the site weights (``ladder.xgb_federated``).

Site weights are ``equal``, ``size`` or ``positives`` (``ladder.site_weights``)
from the counts n_k and n_k^+ that each site returns, or ``manual`` (weights
given to the client).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union
import copy
import importlib.util
import logging
import pickle
from pathlib import Path

import cloudpickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


def _load_ladder():
    try:
        from .. import ladder  # package import
        return ladder
    except ImportError:
        pass
    try:
        from fed_rf_mk import ladder  # type: ignore
        return ladder
    except ImportError:
        pass
    path = Path(__file__).resolve().parents[1] / "ladder.py"  # file loaded on its own
    spec = importlib.util.spec_from_file_location("ladder", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


L = _load_ladder()

WEIGHT_KINDS = ("equal", "size", "positives", "manual")
SAMPLING_KINDS = ("uniform", "oob")


class ModelAggregator:
    """Handles model aggregation and ensemble creation."""

    def __init__(self) -> None:
        self.model_parameters_history: Dict[str, Dict] = {}
        self.xgb_ensemble_members: List[Tuple[bytes, float]] = []  # (model_bytes, weight)

    def store_model_parameters(self, silo_name: str, model_params: Dict) -> None:
        self.model_parameters_history[silo_name] = copy.deepcopy(model_params)

    def get_model_history(self) -> Dict[str, Dict]:
        return self.model_parameters_history

    # ------------------------------------------------------------------
    # Counts and weights
    # ------------------------------------------------------------------

    def counts(self, site_order: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """(n_k, n_k^+) of the training part of each site, in ``site_order``."""
        n, n_pos = [], []
        for name in site_order:
            mp = self.model_parameters_history[name]
            if mp.get("n_train") is None or mp.get("n_pos") is None:
                raise ValueError(f"site {name!r} did not return n_train and n_pos")
            n.append(int(mp["n_train"]))
            n_pos.append(int(mp["n_pos"]))
        return np.asarray(n, dtype=np.int64), np.asarray(n_pos, dtype=np.int64)

    def site_weights(self, site_order: Sequence[str], kind: str = "equal",
                     manual: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Weights of the ladder (``equal``, ``size``, ``positives``) or the
        normalized ``manual`` weights."""
        if kind not in WEIGHT_KINDS:
            raise ValueError(f"site_weights must be one of {WEIGHT_KINDS}, got {kind!r}")
        if kind == "manual":
            if manual is None:
                raise ValueError("manual site weights need the weights")
            return {s: float(manual.get(s, 0.0)) for s in site_order}
        n, n_pos = self.counts(site_order)
        w = L.site_weights(n, n_pos, kind)
        return {s: float(v) for s, v in zip(site_order, w)}

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def merge_estimators(
        self,
        weights: Dict[str, float],
        site_order: Optional[Sequence[str]] = None,
        sampling: str = "uniform",
        seed: Union[int, Sequence[int], None] = 42,
    ) -> Dict[str, object]:
        """Combine the site models with the given site weights.

        ``site_order`` fixes the order in which sites draw trees from the one
        generator ``numpy.random.default_rng(seed)`` (default: the order in
        which the models were stored). Returns the merged Random Forest
        (``seed_model``), the XGBoost members (``ensemble_members``) and
        ``ladder``, what the deployment site needs for every level.
        """
        if sampling not in SAMPLING_KINDS:
            raise ValueError(f"tree_sampling must be one of {SAMPLING_KINDS}, got {sampling!r}")
        order = [s for s in (site_order or list(self.model_parameters_history))
                 if self.model_parameters_history.get(s, {}).get("model")]
        if not order:
            return {"seed_model": None, "ensemble_members": None, "ladder": None}

        models = {s: pickle.loads(self.model_parameters_history[s]["model"]) for s in order}
        kinds = {("rf" if isinstance(m, RandomForestClassifier)
                  else "xgb" if type(m).__name__ == "XGBClassifier" else "other")
                 for m in models.values()}
        if kinds - {"rf", "xgb"} or len(kinds) != 1:
            raise ValueError(f"all sites must return the same model type, got {kinds}")
        w = np.asarray([float(weights.get(s, 0.0)) for s in order])
        if np.any(w < 0) or w.sum() <= 0:
            raise ValueError(f"site weights must be non negative with a positive sum, got {w}")
        w = w / w.sum() if not np.isclose(w.sum(), 1.0, rtol=0, atol=1e-12) else w

        n_train = [self.model_parameters_history[s].get("n_train") for s in order]
        n_pos = [self.model_parameters_history[s].get("n_pos") for s in order]
        have_counts = all(v is not None for v in n_train + n_pos)
        ladder = {
            "sites": list(order),
            "models": [self.model_parameters_history[s]["model"] for s in order],
            "weights": [float(v) for v in w],
            "n": [int(v) for v in n_train] if have_counts else None,
            "n_pos": [int(v) for v in n_pos] if have_counts else None,
            "pi": L.pooled_prevalence(n_train, n_pos) if have_counts else None,
        }

        if kinds == {"xgb"}:
            members = [(self.model_parameters_history[s]["model"], float(v))
                       for s, v in zip(order, w)]
            best = int(np.argmax(w))  # seed for a next round, ties to the first site
            ladder.update(model_type="xgb")
            return {"seed_model": members[best][0], "ensemble_members": members,
                    "ladder": ladder}

        # Random Forest: largest remainder allocation, seeded selection
        T_k = np.asarray([len(models[s].estimators_) for s in order], dtype=np.int64)
        t = L.allocate_trees(w, T_k)
        rng = np.random.default_rng(seed)
        selections, fallbacks = [], []
        for s, Tk, tk in zip(order, T_k, t):
            scores = None
            if sampling == "oob":
                scores = self.model_parameters_history[s].get("oob_scores")
                if scores is None:
                    raise ValueError(f"site {s!r} did not return out of bag scores")
            sel, fb = L.select_trees(int(Tk), int(tk), rng, scores)
            selections.append(sel)
            fallbacks.append(bool(fb))
            if fb:
                logger.warning("Site %s: fewer than %d positive out of bag scores, "
                               "uniform selection used", s, tk)
        merged = copy.deepcopy(models[order[0]])
        merged.estimators_ = [models[s].estimators_[i] for s, sel in zip(order, selections)
                              for i in sel]
        merged.n_estimators = len(merged.estimators_)
        ladder.update(model_type="rf", sampling=sampling,
                      seed=list(seed) if isinstance(seed, (list, tuple)) else seed,
                      T_k=[int(v) for v in T_k], t=[int(v) for v in t],
                      selections=[[int(i) for i in sel] for sel in selections],
                      oob_fallback=fallbacks)
        return {"seed_model": cloudpickle.dumps(merged), "ensemble_members": None,
                "ladder": ladder}

    def combine(
        self,
        site_order: Sequence[str],
        site_weights: str = "equal",
        sampling: str = "uniform",
        seed: Union[int, Sequence[int], None] = 42,
        manual_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, object]:
        """Site weights of the ladder, then ``merge_estimators``."""
        weights = self.site_weights(site_order, site_weights, manual_weights)
        out = self.merge_estimators(weights, site_order, sampling, seed)
        if out.get("ladder") is not None:
            out["ladder"]["weights_kind"] = site_weights
        out["weights"] = weights
        return out
