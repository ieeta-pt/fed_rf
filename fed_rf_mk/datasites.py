#!/usr/bin/env python
# coding: utf-8

import logging
import syft as sy
from syft.service.user.user import UserCreate, ServiceRole

from datasets import generate_mock

from threading import current_thread
from time import sleep
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)


def create_syft_dataset(name: str, data_path: str, mock_path: str) -> Optional[sy.Dataset]:
    """Creates a new syft.Dataset for the selected datasite/dataset.
    None is returned is the matching dataset cannot be found/load from disk.
    """
    if data_path is None:
        logger.error("No data_path provided for dataset '%s'", name)
        return None

    try:
        data = pd.read_csv(data_path)
    except Exception as e:
        logger.error("Failed to read CSV at '%s' for dataset '%s': %s", data_path, name, e)
        return None

    # Generate or load mock data
    try:
        if mock_path is not None:
            mock = pd.read_csv(mock_path)
        else:
            mock = generate_mock(data)
    except Exception as e:
        logger.warning(
            "Mock generation/load failed for '%s' (data=%s, mock=%s): %s. Falling back to data.head(100).",
            name,
            data_path,
            mock_path,
            e,
        )
        mock = data.head(100)

    dataset = sy.Dataset(
        name=name,
        summary=(sumry := f"Dataset from {name}"),
        description=f"""
Detailed Description of the dataset from {name} goes here.
""",
    )  # type: ignore
    dataset.add_asset(
        sy.Asset(
            name="Asset",
            data=data,
            mock=mock,
        )
    )
    return dataset


def _get_welcome_message(name: str, full_name: str) -> str:
    return f"""

## Welcome to the {name} Datasite

**Institute**: {full_name}

**Deployment Type**: Local
"""


def spawn_server(name: str, port: int = 8080, data_path: str = None, mock_path: str = None):
    """Utility function to launch a new instance of a PySyft Datasite"""

    data_site = sy.orchestra.launch(
        name=name,
        port=port,
        reset=True,
        n_consumers=3,
        create_producer=True,
    )
    client = data_site.login(email="info@openmined.org", password="changethis")

    # Customise Settings
    client.settings.allow_guest_signup(True)
    client.settings.welcome_customize(
        markdown=_get_welcome_message(name=name, full_name=name)
    )
    client.users.create(
        email="fedlearning@rf.com",
        password="****",
        password_verify="****",
        name="Researcher Name",
        institution="Institution",
        website="https://institution.com",
        role=ServiceRole.DATA_SCIENTIST,
    )

    user = client.users[-1]
    # user.allow_mock_execution(True)

    ds = create_syft_dataset(name=name, data_path=data_path, mock_path=mock_path)
    if not ds is None:
        client.upload_dataset(ds)

    logger.info("Datasite %s is up and running: %s:%s", name, data_site.url, data_site.port)
    return data_site, client

# --- Server-side analysis policy (module-scoped) ---
ANALYSIS_ALLOWED: bool = False

def set_analysis_allowed(enabled: bool) -> None:
    global ANALYSIS_ALLOWED
    ANALYSIS_ALLOWED = bool(enabled)

def is_analysis_allowed() -> bool:
    return ANALYSIS_ALLOWED


def check_and_approve_incoming_requests(client):
    """This utility function will set the server in busy-waiting
    to constantly check and auto-approve any incoming code requests.

    Note: This function is only intended for the tutorial as demonstration
    of the PoC example.
    For further information about please check out the official for the
    Requests API: https://docs.openmined.org/en/latest/components/requests-api.html
    """
    while not current_thread().stopped():  # type: ignore
        requests = client.requests
        for r in filter(lambda r: r.status.value != 2, requests):  # 2 == APPROVED
            # Server is authoritative for analysis permission. We approve requests
            # regardless; the remote function will consult is_analysis_allowed().
            # You can add request inspection here if you want to reject analysis
            # attempts outright when ANALYSIS_ALLOWED is False.
            r.approve(approve_nested=True)
            logger.info("Approved incoming request (nested approval)")
        sleep(1)
