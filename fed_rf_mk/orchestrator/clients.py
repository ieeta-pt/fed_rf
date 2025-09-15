from __future__ import annotations

from typing import Any, Dict
import logging

import syft as sy
from syft.service.policy.policy import MixedInputPolicy

try:
    from ..utils import check_status_last_code_requests as _check_status_last_code_requests
    from ..remote_tasks import ml_experiment, evaluate_global_model
except ImportError:  # pragma: no cover - fallback when imported as top-level module
    from utils import check_status_last_code_requests as _check_status_last_code_requests  # type: ignore
    from remote_tasks import ml_experiment, evaluate_global_model  # type: ignore

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

    def send_training_requests(self, data_params: dict, model_params: dict) -> None:
        if not self.train_datasites:
            logger.warning("No training clients connected. Please add clients first.")
            return

        for site in self.train_datasites:
            data_asset = self.train_datasites[site].datasets[0].assets[0]
            client = self.train_datasites[site]
            syft_fl_experiment = sy.syft_function(
                input_policy=MixedInputPolicy(
                    client=client, data=data_asset, dataParams=dict, modelParams=dict
                )
            )(ml_experiment)
            ml_training_project = sy.Project(
                name="ML Experiment for FL",
                description="Test project to run a ML experiment",
                members=[client],
            )
            ml_training_project.create_code_request(syft_fl_experiment, client)
            _ = ml_training_project.send()
            logger.info("Training code request sent to '%s'", site)

    def send_evaluation_requests(self, data_params: dict, model_params: dict) -> None:
        for site in self.eval_datasites:
            data_asset = self.eval_datasites[site].datasets[0].assets[0]
            client = self.eval_datasites[site]
            syft_eval_experiment = sy.syft_function(
                input_policy=MixedInputPolicy(
                    client=client, data=data_asset, dataParams=dict, modelParams=dict
                )
            )(evaluate_global_model)
            ml_eval_project = sy.Project(
                name="ML Evaluation for FL",
                description="Test project to evaluate a ML model",
                members=[client],
            )
            ml_eval_project.create_code_request(syft_eval_experiment, client)
            _ = ml_eval_project.send()
            logger.info("Evaluation code request sent to '%s'", site)

    def check_status_last_code_requests(self) -> None:
        _check_status_last_code_requests(self.train_datasites)
        _check_status_last_code_requests(self.eval_datasites)
