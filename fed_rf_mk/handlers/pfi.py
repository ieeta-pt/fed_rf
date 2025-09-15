from __future__ import annotations

from typing import Dict, Optional, Tuple
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PFIHandler:
    """Handles Permutation Feature Importance processing and aggregation."""

    def __init__(self) -> None:
        self.silo_pfi_values: Dict[str, Dict] = {}
        self.averaged_pfi_values: Optional[Dict] = None
        self.arrays_fixed: bool = False

    def store_silo_pfi_values(self, silo_name: str, pfi_data: Dict) -> None:
        self.silo_pfi_values[silo_name] = pfi_data

    def average_pfi_values(self, weights: Dict[str, float]) -> Optional[Dict]:
        if not self.silo_pfi_values:
            return None

        # Get all feature names from first silo
        first_silo = next(iter(self.silo_pfi_values.values()))
        all_features = set(first_silo["feature_names"])

        # Verify all silos have the same features
        for silo_name, pfi_data in self.silo_pfi_values.items():
            silo_features = set(pfi_data["feature_names"])
            if silo_features != all_features:
                logger.warning("Feature mismatch in silo %s", silo_name)
                all_features = all_features.intersection(silo_features)

        # Calculate weighted average for each feature's PFI mean and std
        averaged_pfi_mean = {}
        averaged_pfi_std = {}

        for feature in all_features:
            weighted_sum_mean = 0.0
            weighted_sum_std = 0.0
            total_weight = 0.0

            for silo_name, pfi_data in self.silo_pfi_values.items():
                if feature in pfi_data["pfi_mean"] and silo_name in weights:
                    weight = float(weights[silo_name])
                    weighted_sum_mean += pfi_data["pfi_mean"][feature] * weight
                    weighted_sum_std += pfi_data["pfi_std"][feature] * weight
                    total_weight += weight

            if total_weight > 0:
                averaged_pfi_mean[feature] = weighted_sum_mean / total_weight
                averaged_pfi_std[feature] = weighted_sum_std / total_weight

        self.averaged_pfi_values = {
            "feature_names": list(all_features),
            "pfi_mean": averaged_pfi_mean,
            "pfi_std": averaged_pfi_std,
        }

        return self.averaged_pfi_values

    def fix_arrays(self) -> None:
        # Fix individual silo data
        for silo_name, pfi_data in self.silo_pfi_values.items():
            fixed_pfi_mean = {}
            fixed_pfi_std = {}

            for feature, value in pfi_data["pfi_mean"].items():
                if isinstance(value, np.ndarray):
                    fixed_value = float(value[0])
                else:
                    fixed_value = float(value)
                fixed_pfi_mean[feature] = fixed_value

            for feature, value in pfi_data["pfi_std"].items():
                if isinstance(value, np.ndarray):
                    fixed_value = float(value[0])
                else:
                    fixed_value = float(value)
                fixed_pfi_std[feature] = fixed_value

            self.silo_pfi_values[silo_name]["pfi_mean"] = fixed_pfi_mean
            self.silo_pfi_values[silo_name]["pfi_std"] = fixed_pfi_std

        # Fix averaged data
        if self.averaged_pfi_values:
            if "pfi_mean" in self.averaged_pfi_values:
                fixed_avg_pfi_mean = {}
                fixed_avg_pfi_std = {}

                for feature, value in self.averaged_pfi_values["pfi_mean"].items():
                    if isinstance(value, np.ndarray):
                        fixed_value = float(value[0])
                    else:
                        fixed_value = float(value)
                    fixed_avg_pfi_mean[feature] = fixed_value

                for feature, value in self.averaged_pfi_values["pfi_std"].items():
                    if isinstance(value, np.ndarray):
                        fixed_value = float(value[0])
                    else:
                        fixed_value = float(value)
                    fixed_avg_pfi_std[feature] = fixed_value

                self.averaged_pfi_values["pfi_mean"] = fixed_avg_pfi_mean
                self.averaged_pfi_values["pfi_std"] = fixed_avg_pfi_std

        self.arrays_fixed = True

    def get_silo_values(self, silo_name: str) -> Optional[Tuple[Dict, pd.DataFrame]]:
        if not self.arrays_fixed:
            raise RuntimeError("PFI arrays not generated. Please run fix_arrays() first.")
        if silo_name not in self.silo_pfi_values:
            logger.error(
                "Silo '%s' not found. Available silos: %s",
                silo_name,
                list(self.silo_pfi_values.keys()),
            )
            return None

        silo_data = self.silo_pfi_values[silo_name].copy()

        data = []
        for feature in silo_data["feature_names"]:
            data.append(
                {
                    "Feature": feature,
                    "PFI_Mean": silo_data["pfi_mean"][feature],
                    "PFI_Std": silo_data["pfi_std"][feature],
                }
            )

        df = pd.DataFrame(data).sort_values("PFI_Mean", ascending=False)
        return silo_data, df

    def get_feature_importance_df(self) -> Optional[pd.DataFrame]:
        if not self.arrays_fixed:
            raise RuntimeError("PFI arrays not generated. Please run fix_arrays() first.")
        if not self.averaged_pfi_values:
            return None

        data = []
        for feature in self.averaged_pfi_values["feature_names"]:
            data.append(
                {
                    "Feature": feature,
                    "PFI_Mean": self.averaged_pfi_values["pfi_mean"][feature],
                    "PFI_Std": self.averaged_pfi_values["pfi_std"][feature],
                }
            )

        return pd.DataFrame(data).sort_values("PFI_Mean", ascending=False)

