from __future__ import annotations

from typing import Dict, Optional, Tuple
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SHAPHandler:
    """Handles SHAP value processing and aggregation."""

    def __init__(self) -> None:
        self.silo_shap_values: Dict[str, Dict] = {}
        self.averaged_shap_values: Optional[Dict] = None
        self.arrays_fixed: bool = False

    def store_silo_shap_values(self, silo_name: str, shap_data: Dict) -> None:
        self.silo_shap_values[silo_name] = shap_data

    def average_shap_values(self, weights: Dict[str, float]) -> Optional[Dict]:
        if not self.silo_shap_values:
            return None

        # Get all feature names from first silo
        first_silo = next(iter(self.silo_shap_values.values()))
        all_features = set(first_silo["feature_names"])

        # Verify all silos have the same features
        for silo_name, shap_data in self.silo_shap_values.items():
            silo_features = set(shap_data["feature_names"])
            if silo_features != all_features:
                logger.warning("Feature mismatch in silo %s", silo_name)
                all_features = all_features.intersection(silo_features)

        # Calculate weighted average for each feature's mean absolute SHAP value
        averaged_shap = {}
        for feature in all_features:
            weighted_sum = 0.0
            total_weight = 0.0
            for silo_name, shap_data in self.silo_shap_values.items():
                if feature in shap_data["mean_abs_shap"] and silo_name in weights:
                    weight = float(weights[silo_name])
                    weighted_sum += shap_data["mean_abs_shap"][feature] * weight
                    total_weight += weight
            if total_weight > 0:
                averaged_shap[feature] = weighted_sum / total_weight

        self.averaged_shap_values = {
            "feature_names": list(all_features),
            "mean_abs_shap": averaged_shap,
        }
        return self.averaged_shap_values

    def fix_arrays(self) -> None:
        # Fix individual silo data
        for silo_name, shap_data in self.silo_shap_values.items():
            fixed_mean_abs_shap = {}
            for feature, value in shap_data["mean_abs_shap"].items():
                if isinstance(value, np.ndarray):
                    fixed_value = float(value[0])
                else:
                    fixed_value = float(value)
                fixed_mean_abs_shap[feature] = fixed_value
            self.silo_shap_values[silo_name]["mean_abs_shap"] = fixed_mean_abs_shap

        # Fix averaged data
        if self.averaged_shap_values and "mean_abs_shap" in self.averaged_shap_values:
            fixed_avg_shap = {}
            for feature, value in self.averaged_shap_values["mean_abs_shap"].items():
                if isinstance(value, np.ndarray):
                    fixed_value = float(value[0])
                else:
                    fixed_value = float(value)
                fixed_avg_shap[feature] = fixed_value
            self.averaged_shap_values["mean_abs_shap"] = fixed_avg_shap

        self.arrays_fixed = True

    def get_silo_values(self, silo_name: str) -> Optional[Tuple[Dict, pd.DataFrame]]:
        if not self.arrays_fixed:
            raise RuntimeError("SHAP arrays not generated. Please run fix_arrays() first.")

        if silo_name not in self.silo_shap_values:
            logger.error(
                "Silo '%s' not found. Available silos: %s",
                silo_name,
                list(self.silo_shap_values.keys()),
            )
            return None

        silo_data = self.silo_shap_values[silo_name].copy()
        df = pd.DataFrame(
            list(silo_data["mean_abs_shap"].items()),
            columns=["Feature", "Mean_Abs_SHAP"],
        ).sort_values("Mean_Abs_SHAP", ascending=False)
        return silo_data, df

    def get_feature_importance_df(self) -> Optional[pd.DataFrame]:
        if not self.arrays_fixed:
            raise RuntimeError("SHAP arrays not generated. Please run fix_arrays() first.")
        if not self.averaged_shap_values:
            return None
        return pd.DataFrame(
            list(self.averaged_shap_values["mean_abs_shap"].items()),
            columns=["Feature", "Mean_Abs_SHAP"],
        ).sort_values("Mean_Abs_SHAP", ascending=False)

