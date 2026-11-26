#!/usr/bin/env python
# coding: utf-8

import logging
import os
import syft as sy
from syft.service.user.user import UserCreate, ServiceRole

from fed_rf_mk.datasets import MOCK_SEED, generate_mock

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

    # Mock data: a file given by the data owner, or a synthetic mock. The real
    # data is never uploaded as the mock.
    if mock_path is not None:
        try:
            mock = pd.read_csv(mock_path)
        except Exception as e:
            logger.error("Failed to read mock CSV at '%s' for dataset '%s': %s", mock_path, name, e)
            return None
        logger.info("Dataset '%s': mock read from %s", name, mock_path)
    else:
        mock = generate_mock(data)
        logger.info(
            "Dataset '%s': synthetic mock used (%d rows, columns drawn independently, seed %d)",
            name, len(mock), MOCK_SEED,
        )

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


ADMIN_EMAIL = "info@openmined.org"  # PySyft default admin account
_ADMIN_DEFAULT_PASSWORD = "changethis"  # PySyft default, replaced at start
ENV_ADMIN_PASSWORD = "FEDRF_ADMIN_PASSWORD"
ENV_DS_EMAIL = "FEDRF_DS_EMAIL"
ENV_DS_PASSWORD = "FEDRF_DS_PASSWORD"
ENV_GUEST_SIGNUP = "FEDRF_ALLOW_GUEST_SIGNUP"
ENV_ANALYSIS = "FEDRF_ANALYSIS_ALLOWED"
DEFAULT_DS_EMAIL = "fedlearning@rf.com"


def _setting(value, env, required=True, what=""):
    if value is None:
        value = os.environ.get(env)
    if required and not value:
        raise ValueError(f"{what} must be given as an argument or in the environment variable {env}")
    return value


def spawn_server(
    name: str,
    port: int = 8080,
    data_path: str = None,
    mock_path: str = None,
    admin_password: str = None,
    ds_email: str = None,
    ds_password: str = None,
    allow_guest_signup: bool = None,
):
    """Utility function to launch a new instance of a PySyft Datasite.

    The data owner gives the new admin password and the password of the data
    scientist account, as arguments or in the environment variables
    FEDRF_ADMIN_PASSWORD and FEDRF_DS_PASSWORD (FEDRF_DS_EMAIL for the email,
    default fedlearning@rf.com). Guest signup stays disabled unless
    ``allow_guest_signup`` (or FEDRF_ALLOW_GUEST_SIGNUP=1) asks for it.
    """
    admin_password = _setting(admin_password, ENV_ADMIN_PASSWORD, what="admin password")
    ds_password = _setting(ds_password, ENV_DS_PASSWORD, what="data scientist password")
    ds_email = _setting(ds_email, ENV_DS_EMAIL, required=False) or DEFAULT_DS_EMAIL
    if allow_guest_signup is None:
        allow_guest_signup = os.environ.get(ENV_GUEST_SIGNUP, "0").lower() in ("1", "true", "yes")
    if admin_password == _ADMIN_DEFAULT_PASSWORD:
        raise ValueError("the admin password must differ from the PySyft default")

    data_site = sy.orchestra.launch(
        name=name,
        port=port,
        reset=True,
        n_consumers=3,
        create_producer=True,
    )
    client = data_site.login(email=ADMIN_EMAIL, password=_ADMIN_DEFAULT_PASSWORD)
    client.account.set_password(admin_password, confirm=False)
    client = data_site.login(email=ADMIN_EMAIL, password=admin_password)

    # Customise Settings
    client.settings.allow_guest_signup(enable=bool(allow_guest_signup))
    client.settings.welcome_customize(
        markdown=_get_welcome_message(name=name, full_name=name)
    )
    client.users.create(
        email=ds_email,
        password=ds_password,
        password_verify=ds_password,
        name="Researcher Name",
        institution="Institution",
        website="https://institution.com",
        role=ServiceRole.DATA_SCIENTIST,
    )

    ds = create_syft_dataset(name=name, data_path=data_path, mock_path=mock_path)
    if not ds is None:
        client.upload_dataset(ds)

    logger.info(
        "Datasite %s is up and running: %s:%s (guest signup %s)",
        name, data_site.url, data_site.port, "enabled" if allow_guest_signup else "disabled",
    )
    return data_site, client

# --- Server-side analysis policy (module-scoped) ---
ANALYSIS_ALLOWED: bool = False

def set_analysis_allowed(enabled: bool) -> None:
    """Set the policy in this process and in its environment, so that a
    DataSite launched afterwards (a child process) inherits it. A DataSite
    that is already running keeps the value it started with."""
    global ANALYSIS_ALLOWED
    ANALYSIS_ALLOWED = bool(enabled)
    os.environ[ENV_ANALYSIS] = "1" if enabled else "0"

def is_analysis_allowed() -> bool:
    env = os.environ.get(ENV_ANALYSIS)
    if env is not None:
        return env == "1"
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
