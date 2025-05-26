#!/usr/bin/env python
# coding: utf-8

import syft as sy
from syft.service.user.user import UserCreate, ServiceRole
from threading import current_thread
from time import sleep
from typing import Optional
import pandas as pd


from fed_rf_mk.datasets import generate_mock


def create_syft_dataset(name: str, data_path: str, mock_path: str) -> Optional[sy.Dataset]:
    """Creates a new syft.Dataset for the selected datasite/dataset.
    None is returned is the matching dataset cannot be found/load from disk.
    """
    if data_path is None:
        return None
    
    data = pd.read_csv(data_path)

    if data is None:
        return None
    
    if mock_path is not None:
        mock = pd.read_csv(mock_path)
    else:
        mock = generate_mock(data)

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

    print(f"Datasite {name} is up and running: {data_site.url}:{data_site.port}")
    return data_site, client


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
            r.approve(approve_nested=True)
            # print("New Request approved in ")
        sleep(1)
