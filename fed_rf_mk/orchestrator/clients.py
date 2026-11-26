from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple
import logging
import time

import syft as sy
from syft.service.policy.policy import MixedInputPolicy

try:
    from ..utils import check_status_last_code_requests as _check_status_last_code_requests
    from ..remote_tasks import ml_experiment, evaluate_global_model, site_counts
except ImportError:  # pragma: no cover - fallback when imported as top-level module
    from fed_rf_mk.utils import check_status_last_code_requests as _check_status_last_code_requests  # type: ignore
    from fed_rf_mk.remote_tasks import ml_experiment, evaluate_global_model, site_counts  # type: ignore

logger = logging.getLogger(__name__)


class DataSiteManager:
    """Manages connections to data sites and handles communication."""

    def __init__(self) -> None:
        self.train_datasites: Dict[str, Any] = {}
        self.eval_datasites: Dict[str, Any] = {}

    def add_train_client(self, name: str, url: str, email: str, password: str) -> bool:
        try:
            client = sy.login(email=email, password=password, url=url)
            self.train_datasites[name] = client
            logger.info("Connected to training datasite '%s' at %s", name, url)
            return True
        except Exception as e:
            logger.error("Failed to connect to training datasite '%s' at %s: %s", name, url, e)
            return False

    def add_eval_client(self, name: str, url: str, email: str, password: str) -> bool:
        try:
            client = sy.login(email=email, password=password, url=url)
            self.eval_datasites[name] = client
            logger.info("Connected to evaluation datasite '%s' at %s", name, url)
            return True
        except Exception as e:
            logger.error("Failed to connect to evaluation datasite '%s' at %s: %s", name, url, e)
            return False

    def check_status(self) -> None:
        for name, client in self.train_datasites.items():
            try:
                datasets = client.datasets
                logger.info("%s: Connected (%d datasets available)", name, len(datasets))
            except Exception as e:
                logger.error("%s: Connection failed (%s)", name, e)

    def get_train_datasites(self) -> Dict[str, Any]:
        return self.train_datasites

    def get_eval_datasites(self) -> Dict[str, Any]:
        return self.eval_datasites

    @staticmethod
    def _send_code_request(client: Any, fn: Callable, data_params: dict, model_params: dict,
                           project_name: str, description: str) -> None:
        """One code request whose ``dataParams`` and ``modelParams`` values are
        bound in the request (``sy.Constant``). The data owner approves the
        code with these values, and the approved code runs only with them."""
        data_asset = client.datasets[0].assets[0]
        syft_fn = sy.syft_function(
            input_policy=MixedInputPolicy(
                client=client,
                data=data_asset,
                dataParams=sy.Constant(val=data_params),
                modelParams=sy.Constant(val=model_params),
            )
        )(fn)
        project = sy.Project(name=project_name, description=description, members=[client])
        project.create_code_request(syft_fn, client)
        _ = project.send()

    def send_training_requests(self, data_params: dict,
                               plan: Dict[str, Tuple[Callable, dict]]) -> None:
        """``plan`` maps a training site to (function, modelParams). Sites not
        in ``plan`` get no request."""
        if not self.train_datasites:
            logger.warning("No training clients connected. Please add clients first.")
            return

        for site, (fn, params) in plan.items():
            client = self.train_datasites[site]
            self._send_code_request(client, fn, data_params, params,
                                    "ML Experiment for FL", "Test project to run a ML experiment")
            logger.info("Training code request (%s) sent to '%s'", fn.__name__, site)

    def send_evaluation_requests(self, data_params: dict, model_params: dict) -> None:
        for site in self.eval_datasites:
            client = self.eval_datasites[site]
            self._send_code_request(client, evaluate_global_model, data_params, model_params,
                                    "ML Evaluation for FL", "Test project to evaluate a ML model")
            logger.info("Evaluation code request sent to '%s'", site)

    @staticmethod
    def wait_for_approval(client: Any, func_name: str, timeout: float = 0.0) -> bool:
        """True once the latest request for ``func_name`` at this site is
        approved. Waits up to ``timeout`` seconds. Raises if it was rejected."""
        t0 = time.time()
        while True:
            reqs = [r for r in client.requests
                    if getattr(r.code, "service_func_name", None) == func_name]
            if reqs:
                status = reqs[-1].status.value
                if status == 2:  # APPROVED
                    return True
                if status == 1:  # REJECTED
                    raise PermissionError(f"the request for {func_name} was rejected")
            if time.time() - t0 >= timeout:
                return False
            time.sleep(1.0)

    def check_status_last_code_requests(self) -> None:
        _check_status_last_code_requests(self.train_datasites)
        _check_status_last_code_requests(self.eval_datasites)
