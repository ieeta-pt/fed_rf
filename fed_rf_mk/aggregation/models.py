from __future__ import annotations

from typing import Dict, List, Tuple, Union
import copy
import logging
import pickle
import random

import cloudpickle
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


class ModelAggregator:
    """Handles model aggregation and ensemble creation."""

    def __init__(self) -> None:
        self.model_parameters_history: Dict[str, Dict] = {}
        self.xgb_ensemble_members: List[Tuple[bytes, float]] = []  # (model_bytes, weight)

    def store_model_parameters(self, silo_name: str, model_params: Dict) -> None:
        self.model_parameters_history[silo_name] = copy.deepcopy(model_params)

    def merge_estimators(self, weights: Dict[str, float]) -> Dict[str, Union[bytes, List[Tuple[bytes, float]], None]]:
        all_estimators = []
        merged_rf = None
        xgb_members: List[Tuple[bytes, float]] = []
        seed_xgb = None
        best_w = -1.0

        for silo_name, model_params in self.model_parameters_history.items():
            model_bytes = model_params.get("model")
            if not model_bytes:
                continue
            clf = pickle.loads(model_bytes)
            w = float(weights.get(silo_name, 0.0))

            if isinstance(clf, RandomForestClassifier):
                # RF path: sample estimators by weight
                n_to_take = int(round(clf.n_estimators * w))
                n_to_take = max(0, min(n_to_take, len(clf.estimators_)))
                if n_to_take > 0:
                    all_estimators.extend(random.sample(clf.estimators_, n_to_take))
                if merged_rf is None:
                    merged_rf = clf
            elif type(clf).__name__ == "XGBClassifier":
                # XGB path: keep members for evaluation; pick best-weight seed for next round
                xgb_members.append((model_bytes, w))
                if w > best_w:
                    seed_xgb = model_bytes
                    best_w = w
            else:
                logger.warning("Unknown model type for silo '%s'. Skipping.", silo_name)

        # Prefer XGB branch when present
        if xgb_members:
            return {"seed_model": seed_xgb, "ensemble_members": xgb_members}

        # Otherwise RF: build merged RF and serialize
        if merged_rf is not None:
            merged_rf.estimators_ = all_estimators
            return {"seed_model": cloudpickle.dumps(merged_rf), "ensemble_members": None}

        # Nothing to merge
        return {"seed_model": None, "ensemble_members": None}

    def get_model_history(self) -> Dict[str, Dict]:
        return self.model_parameters_history

