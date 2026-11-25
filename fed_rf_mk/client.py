from __future__ import annotations

import copy
import logging
import concurrent.futures
from typing import List, TypedDict, Any, Dict, Tuple, Optional

import pandas as pd
import numpy as np

try:
    # Package import
    from .orchestrator.clients import DataSiteManager
    from .orchestrator.weights import WeightManager
    from .handlers.shap import SHAPHandler
    from .handlers.pfi import PFIHandler
    from .aggregation.models import ModelAggregator, L as _ladder
    from . import release as R
    from .remote_tasks import ml_experiment, site_counts
except ImportError:  # pragma: no cover - fallback for script-style usage
    # Script-style import (when running from inside package directory)
    from fed_rf_mk.orchestrator.clients import DataSiteManager  # type: ignore
    from fed_rf_mk.orchestrator.weights import WeightManager  # type: ignore
    from fed_rf_mk.handlers.shap import SHAPHandler  # type: ignore
    from fed_rf_mk.handlers.pfi import PFIHandler  # type: ignore
    from fed_rf_mk.aggregation.models import ModelAggregator, L as _ladder  # type: ignore
    from fed_rf_mk import release as R  # type: ignore
    from fed_rf_mk.remote_tasks import ml_experiment, site_counts  # type: ignore

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
    "release_level": "all",       # 0, 1, 2, 3, 3+1, 3+2 or all (fed_rf_mk/release.py)
    "source_site": None,          # name of the source site, for release levels 0, 1 and 2
    "approval_timeout": 300,      # seconds to wait for the data owner to approve a request
}

_SITE_FUNCTIONS = {"ml_experiment": ml_experiment, "site_counts": site_counts}
# Keys of model_params that stay at the coordinator
_COORDINATOR_ONLY = ("ladder", "ensemble_members", "site_seeds", "approval_timeout")


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

        # Release plan and what was sent and received (kept for inspection)
        self._plan: Optional[Dict[str, dict]] = None
        self.site_params_sent: Dict[str, dict] = {}
        self.site_results: Dict[str, dict] = {}
        self.eval_params_sent: Optional[dict] = None

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

    def _release_level(self) -> str:
        return R.normalize_level(self.model_params.get("release_level", "all"))

    def _site_params(self, name: str, entry: dict) -> dict:
        mp = {k: v for k, v in self.model_params.items() if k not in _COORDINATOR_ONLY}
        mp["model"] = None
        seeds = self.model_params.get("site_seeds") or {}
        if name in seeds:
            mp["random_state"] = int(seeds[name])
        if entry["release"] is not None:
            mp["release"] = dict(entry["release"])
        return mp

    def send_request(self) -> None:
        """Send the training requests of the release plan. The values of
        dataParams and modelParams are bound in each request, so the data
        owner approves them with the code. The evaluation request carries the
        released models, so it is sent after training
        (``send_evaluation_request``, or by ``run_evaluate``)."""
        if not self.datasite_manager.get_train_datasites():
            logger.warning("No clients connected. Please add clients first.")
            return

        if not self.data_params or not self.model_params:
            logger.warning("DataParams and ModelParams must be set before sending the request.")
            return

        mp = self.model_params
        self._plan = R.site_plan(
            self._release_level(), list(self.datasite_manager.get_train_datasites()),
            source_site=mp.get("source_site"), site_weights=mp.get("site_weights", "equal"),
            tree_sampling=mp.get("tree_sampling", "uniform"),
            model_type=mp.get("model_type", "rf"),
        )
        self.site_params_sent = {n: self._site_params(n, e) for n, e in self._plan.items()}
        self.eval_params_sent = None
        self.datasite_manager.send_training_requests(
            self.data_params,
            {n: (_SITE_FUNCTIONS[e["function"]], self.site_params_sent[n])
             for n, e in self._plan.items()},
        )
        skipped = [n for n in self.datasite_manager.get_train_datasites() if n not in self._plan]
        if skipped:
            logger.info("Release level %s: no request to %s", self._release_level(), skipped)

    def send_evaluation_request(self) -> None:
        """Send the evaluation request, with only what the release level lets
        the deployment site receive."""
        lad = self.model_params.get("ladder")
        if lad is None:
            raise RuntimeError("run_model must run before the evaluation request is sent")
        level = self._release_level()
        if level == "all":
            params = {k: v for k, v in self.model_params.items()
                      if k not in ("site_seeds", "approval_timeout")}
        else:
            params = {
                "model_type": self.model_params.get("model_type", "rf"),
                "release_level": level,
                "calibration": self.model_params.get("calibration"),
                "return_predictions": self.model_params.get("return_predictions", False),
                "ladder": R.deployment_ladder(level, lad),
            }
        self.eval_params_sent = params
        self._eval_for = id(lad)
        self.datasite_manager.send_evaluation_requests(self.data_params, params)

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

    def _call_approved(self, name: str, client: Any, func_name: str) -> dict:
        timeout = float(self.model_params.get("approval_timeout", 300))
        if not self.datasite_manager.wait_for_approval(client, func_name, timeout):
            raise TimeoutError(f"the {func_name} request at {name} is not approved "
                               f"after {timeout:.0f} s")
        asset = client.datasets[0].assets[0]
        # dataParams and modelParams are bound in the approved request
        return getattr(client.code, func_name)(data=asset).get_from(client)

    def run_model(self) -> None:
        """Execute the federated learning training process."""
        if self._plan is None:
            raise RuntimeError("send_request must run before run_model")
        datasites = self.datasite_manager.get_train_datasites()
        plan = self._plan
        level = self._release_level()

        # First epoch - parallel execution
        logger.info("Launching first-epoch training on %d clients in parallel…", len(plan))
        self.site_results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(plan))) as executor:
            futures = {executor.submit(self._call_approved, name, datasites[name],
                                       entry["function"]): name
                       for name, entry in plan.items()}

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    mp = future.result()
                    if not isinstance(mp, dict) or "error" in mp:
                        raise RuntimeError(mp.get("error") if isinstance(mp, dict) else mp)
                    self.site_results[name] = copy.deepcopy(mp)
                    if plan[name]["function"] != "ml_experiment":
                        continue
                    if not mp.get("model"):
                        raise RuntimeError("no model returned")
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

        order = [name for name in datasites if name in plan]
        if level in R.SOURCE_LEVELS:
            # every training site, so that the source keeps its index
            self._source_ladder(list(datasites), level)
            return

        # Sites in the order they were added (not the order in which they
        # finished), so that the seeded tree selection is reproducible.
        history = self.model_aggregator.get_model_history()
        successful_silos = [name for name in order if name in history]
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

        logger.info("Models merged successfully")

    def _source_ladder(self, order: List[str], level: str) -> None:
        """Levels 0, 1 and 2: the model of the source site, and for level 1
        the pooled prevalence from the counts of every site."""
        src = self.model_params.get("source_site")
        res = self.site_results.get(src)
        if res is None or not res.get("model"):
            raise RuntimeError(f"the source site {src!r} did not return its model")
        lad = {"model_type": res.get("model_type", self.model_params.get("model_type")),
               "sites": [src], "sources": [order.index(src)], "models": [res["model"]]}
        if level == "1":
            missing = [s for s in order if s not in self.site_results]
            if missing:
                raise RuntimeError(f"level 1 needs the counts of every site, missing {missing}")
            n = [int(self.site_results[s]["n_train"]) for s in order]
            n_pos = [int(self.site_results[s]["n_pos"]) for s in order]
            j = order.index(src)
            lad.update(n=n, n_pos=n_pos, pi=_ladder.pooled_prevalence(n, n_pos),
                       pi_k=[float(np.int64(n_pos[j]) / np.int64(n[j]))])
        self.model_params["model"] = res["model"]
        self.model_params["ensemble_members"] = None
        self.model_params["ladder"] = lad
        logger.info("Release level %s: model of %s%s", level, src,
                    ", pooled prevalence %s" % lad.get("pi") if level == "1" else "")

    def run_evaluate(self) -> Optional[Dict]:
        """Run evaluation on evaluation sites. Sends the evaluation request
        first if it was not sent for the current models, then waits for its
        approval (``approval_timeout``)."""
        eval_datasites = self.datasite_manager.get_eval_datasites()
        logger.info("Number of evaluation sites: %s", len(eval_datasites))
        lad = self.model_params.get("ladder")
        if self.eval_params_sent is None or getattr(self, "_eval_for", None) != id(lad):
            self.send_evaluation_request()

        for name, datasite in eval_datasites.items():
            logger.info("Evaluating model at %s", name)
            try:
                return self._call_approved(name, datasite, "evaluate_global_model")
            except Exception as e:
                logger.error("Evaluation failed on %s: %s", name, e)
                return {"error": str(e)}

        return None

    # Legacy method
    def start_shap_analysis(self) -> None:
        """Legacy method - use start_analysis() instead."""
        logger.warning("start_shap_analysis() is deprecated. Use start_analysis() instead.")
        self.start_analysis()


def hello_world() -> str:
    return "Hello! This is fed_rf_mk client module. Installed correctly.\nVersion 1.1.0.\nMade by Alexandre Cotorobai."