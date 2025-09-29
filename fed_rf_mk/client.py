from __future__ import annotations

import logging
import concurrent.futures
from typing import List, TypedDict, Any, Dict, Tuple, Optional

import pandas as pd
import syft as sy
from syft.service.policy.policy import MixedInputPolicy

try:
    # Package import
    from .orchestrator.clients import DataSiteManager
    from .orchestrator.weights import WeightManager
    from .handlers.shap import SHAPHandler
    from .handlers.pfi import PFIHandler
    from .aggregation.models import ModelAggregator
except ImportError:  # pragma: no cover - fallback for script-style usage
    # Script-style import (when running from inside package directory)
    from fed_rf_mk.orchestrator.clients import DataSiteManager  # type: ignore
    from fed_rf_mk.orchestrator.weights import WeightManager  # type: ignore
    from fed_rf_mk.handlers.shap import SHAPHandler  # type: ignore
    from fed_rf_mk.handlers.pfi import PFIHandler  # type: ignore
    from fed_rf_mk.aggregation.models import ModelAggregator  # type: ignore

logger = logging.getLogger(__name__)


class DataParamsDict(TypedDict):
    target: str
    ignored_columns: List[Any]


class ModelParamsDict(TypedDict):
    model: bytes
    n_base_estimators: int
    n_incremental_estimators: int
    train_size: float
    test_size: float
    sample_size: int
    model_type: str
    ensemble_members: Optional[List[Tuple[bytes, float]]]


class FLClient:
    """
    Refactored Federated Learning Client with separated concerns.

    Coordinates between specialized handlers and managers instead of
    managing everything directly.
    """

    def __init__(self) -> None:
        # Specialized handlers
        self.datasite_manager = DataSiteManager()
        self.weight_manager = WeightManager()
        self.shap_handler = SHAPHandler()
        self.pfi_handler = PFIHandler()
        self.model_aggregator = ModelAggregator()

        # Configuration
        self.data_params: dict = {}
        self.model_params: dict = {}

    def add_train_client(
        self, name: str, url: str, email: str, password: str, weight: Optional[float] = None
    ) -> None:
        """Add a training client with optional weight."""
        success = self.datasite_manager.add_train_client(name, url, email, password)
        if success:
            self.weight_manager.set_weight(name, weight)

    def add_eval_client(self, name: str, url: str, email: str, password: str) -> None:
        """Add an evaluation client."""
        self.datasite_manager.add_eval_client(name, url, email, password)

    def check_status(self) -> None:
        """Check status of all connected silos."""
        self.datasite_manager.check_status()

    def set_data_params(self, data_params: dict) -> str:
        """Set data parameters."""
        self.data_params = data_params
        return f"Data parameters set: {data_params}"

    def set_model_params(self, model_params: dict) -> str:
        """Set model parameters."""
        self.model_params = model_params
        return f"Model parameters set: {model_params}"

    def get_data_params(self) -> dict:
        """Get data parameters."""
        return self.data_params

    def get_model_params(self) -> dict:
        """Get model parameters."""
        return self.model_params

    def send_request(self) -> None:
        """Send requests to all connected sites."""
        if not self.datasite_manager.get_train_datasites():
            logger.warning("No clients connected. Please add clients first.")
            return

        if not self.data_params or not self.model_params:
            logger.warning("DataParams and ModelParams must be set before sending the request.")
            return

        self.datasite_manager.send_training_requests(self.data_params, self.model_params)
        self.datasite_manager.send_evaluation_requests(self.data_params, self.model_params)

    def check_status_last_code_requests(self) -> None:
        """Check status of last code requests."""
        self.datasite_manager.check_status_last_code_requests()

    def start_analysis(self) -> None:
        """Prepare feature importance analysis results."""
        logger.info("Generating analysis arrays...")

        self.shap_handler.fix_arrays()
        self.pfi_handler.fix_arrays()

        logger.info("SHAP and PFI arrays generated.")

    # SHAP-related methods (delegated to SHAPHandler)
    def get_averaged_shap_values(self) -> Optional[Dict]:
        """Get averaged SHAP values from all silos."""
        return self.shap_handler.averaged_shap_values

    def get_shap_feature_importance_df(self) -> Optional[pd.DataFrame]:
        """Get SHAP feature importance as DataFrame."""
        return self.shap_handler.get_feature_importance_df()

    def get_silo_shap_values(self, silo_name: str) -> Optional[Tuple[Dict, pd.DataFrame]]:
        """Get SHAP values for a specific silo."""
        return self.shap_handler.get_silo_values(silo_name)

    # PFI-related methods (delegated to PFIHandler)
    def get_averaged_pfi_values(self) -> Optional[Dict]:
        """Get averaged PFI values from all silos."""
        return self.pfi_handler.averaged_pfi_values

    def get_pfi_feature_importance_df(self) -> Optional[pd.DataFrame]:
        """Get PFI feature importance as DataFrame."""
        return self.pfi_handler.get_feature_importance_df()

    def get_silo_pfi_values(self, silo_name: str) -> Optional[Tuple[Dict, pd.DataFrame]]:
        """Get PFI values for a specific silo."""
        return self.pfi_handler.get_silo_values(silo_name)

    def run_model(self) -> None:
        """Execute the federated learning training process."""
        datasites = self.datasite_manager.get_train_datasites()

        # First epoch - parallel execution
        logger.info("Launching first-epoch training on all clients in parallel…")
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(datasites)) as executor:
            futures: dict[concurrent.futures.Future, str] = {}
            for name, datasite in datasites.items():
                data_asset = datasite.datasets[0].assets[0]
                futures[executor.submit(
                    lambda ds, da, dp: ds.code.ml_experiment(
                        data=da, dataParams=dp, modelParams={**self.model_params, "model": None}
                    ).get_from(ds),
                    datasite, data_asset, self.data_params,
                )] = name

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    mp = future.result()
                    self.model_aggregator.store_model_parameters(name, mp)

                    # Store SHAP values
                    if "shap_data" in mp:
                        self.shap_handler.store_silo_shap_values(name, mp["shap_data"])
                        logger.info("%s completed with SHAP analysis", name)

                    # Store PFI values
                    if "pfi_data" in mp:
                        self.pfi_handler.store_silo_pfi_values(name, mp["pfi_data"])
                        logger.info("%s completed with PFI analysis", name)

                except Exception as e:
                    logger.warning("%s failed: %s", name, e)

        # Normalize weights and average feature importance values
        successful_silos = list(self.model_aggregator.get_model_history().keys())
        normalized_weights = self.weight_manager.normalize_weights(successful_silos)

        # Average feature importance values
        self.shap_handler.average_shap_values(normalized_weights)
        self.pfi_handler.average_pfi_values(normalized_weights)

        # Merge models
        merged_model = self.model_aggregator.merge_estimators(normalized_weights)
        self.model_params["model"] = merged_model["seed_model"]
        self.model_params["ensemble_members"] = merged_model.get("ensemble_members")

        logger.info("SHAP values averaged across silos")
        logger.info("PFI values averaged across silos")
        logger.info("Models merged successfully")

    def run_evaluate(self) -> Optional[Dict]:
        """Run evaluation on evaluation sites."""
        eval_datasites = self.datasite_manager.get_eval_datasites()
        logger.info("Number of evaluation sites: %s", len(eval_datasites))

        for name, datasite in eval_datasites.items():
            data_asset = datasite.datasets[0].assets[0]
            logger.info("Evaluating model at %s", name)

            try:
                # Preferred path: use registered remote function on datasite
                model = datasite.code.evaluate_global_model(
                    data=data_asset, dataParams=self.data_params, modelParams=self.model_params
                ).get_from(datasite)
                return model
            except Exception as e:
                # Fallback: construct and invoke an ad-hoc remote function with policy
                logger.info(
                    "Falling back to ad-hoc evaluate_global_model syft function: %s", e
                )
                try:
                    try:
                        from .remote_tasks import evaluate_global_model as eval_fn
                    except Exception:  # script-style fallback
                        from remote_tasks import evaluate_global_model as eval_fn  # type: ignore

                    eval_remote = sy.syft_function(
                        input_policy=MixedInputPolicy(
                            client=datasite, data=data_asset, dataParams=dict, modelParams=dict
                        )
                    )(eval_fn)
                    model = eval_remote(
                        data=data_asset, dataParams=self.data_params, modelParams=self.model_params
                    ).get_from(datasite)
                    return model
                except Exception as e2:
                    logger.error("Evaluation failed on %s: %s", name, e2)
                    return {"error": str(e2)}

        return None

    # Legacy method
    def start_shap_analysis(self) -> None:
        """Legacy method - use start_analysis() instead."""
        logger.warning("start_shap_analysis() is deprecated. Use start_analysis() instead.")
        self.start_analysis()
