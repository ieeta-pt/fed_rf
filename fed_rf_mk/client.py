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


# Options of the disclosure ladder in model_params, with the defaults of the paper.
LADDER_DEFAULTS = {
    "site_weights": "equal",      # equal, size, positives (ladder.site_weights) or manual
    "tree_sampling": "uniform",   # uniform or oob (out of bag score of each tree)
    "seed": 42,                   # int or list of ints, numpy generator of the tree selection
    "site_seeds": None,           # optional {site name: random_state of its local model}
    "calibration": {"n_cal": None, "fraction": 0.2, "random_state": 42},  # level 2
    "return_predictions": False,  # per record outputs of the test part of the deployment site
}


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
        """Set model parameters. Ladder options missing from ``model_params``
        take the defaults of ``LADDER_DEFAULTS``."""
        params = {k: (dict(v) if isinstance(v, dict) else v) for k, v in LADDER_DEFAULTS.items()}
        cal = {**params["calibration"], **(model_params.get("calibration") or {})}
        params.update(model_params)
        params["calibration"] = cal
        self.model_params = params
        return f"Model parameters set: {params}"

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
            site_seeds = self.model_params.get("site_seeds") or {}
            for name, datasite in datasites.items():
                data_asset = datasite.datasets[0].assets[0]
                mp = {**self.model_params, "model": None}
                if name in site_seeds:
                    mp["random_state"] = int(site_seeds[name])
                futures[executor.submit(
                    lambda ds, da, dp, mp: ds.code.ml_experiment(
                        data=da, dataParams=dp, modelParams=mp
                    ).get_from(ds),
                    datasite, data_asset, self.data_params, mp,
                )] = name

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    mp = future.result()
                    if not isinstance(mp, dict) or "error" in mp or not mp.get("model"):
                        raise RuntimeError(mp.get("error") if isinstance(mp, dict) else mp)
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

        # Sites in the order they were added (not the order in which they
        # finished), so that the seeded tree selection is reproducible.
        history = self.model_aggregator.get_model_history()
        successful_silos = [name for name in datasites if name in history]
        manual = None
        if self.model_params.get("site_weights", "equal") == "manual":
            manual = self.weight_manager.normalize_weights(successful_silos)

        # Merge models with the rule of the ladder (site weights, largest
        # remainder allocation, seeded tree selection)
        merged_model = self.model_aggregator.combine(
            successful_silos,
            site_weights=self.model_params.get("site_weights", "equal"),
            sampling=self.model_params.get("tree_sampling", "uniform"),
            seed=self.model_params.get("seed", 42),
            manual_weights=manual,
        )
        weights = merged_model["weights"]

        # Average feature importance values with the same site weights
        self.shap_handler.average_shap_values(weights)
        self.pfi_handler.average_pfi_values(weights)

        self.model_params["model"] = merged_model["seed_model"]
        self.model_params["ensemble_members"] = merged_model.get("ensemble_members")
        self.model_params["ladder"] = merged_model.get("ladder")
        lad = merged_model.get("ladder") or {}
        logger.info("Site weights (%s): %s", self.model_params.get("site_weights"), weights)
        logger.info("Pooled prevalence: %s", lad.get("pi"))
        if lad.get("t") is not None:
            logger.info("Trees per site: %s", dict(zip(lad["sites"], lad["t"])))

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


def hello_world() -> str:
    return "Hello! This is fed_rf_mk client module. Installed correctly.\nVersion 1.0.0.\nMade by Alexandre Cotorobai."