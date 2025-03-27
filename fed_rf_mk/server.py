#!/usr/bin/env python
# coding: utf-8

from threading import Thread, Event
from time import sleep
from fed_rf_mk.datasites import spawn_server, check_and_approve_incoming_requests


class DataSiteThread(Thread):
    """
    Thread class with a stop() method.
    The thread itself has to check regularly for the stopped() condition.
    """

    def __init__(self, *args, **kwargs):
        super(DataSiteThread, self).__init__(*args, **kwargs)
        self._stop_event = Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


def launch_datasite(name: str, port: int, data_path: str, mock_path: str):
    """
    Launches a single datasite with given parameters.

    Args:
        name (str): The name of the datasite.
        port (int): The port on which the datasite runs.
        data_path (str): Path to the dataset.
        mock_path (str): Path to mock dataset.
    """
    print(f"Starting DataSite {name} on port {port} with data at {data_path} and mock at {mock_path}")

    data_site, client = spawn_server(name=name, port=port, data_path=data_path, mock_path=mock_path)

    # Start background thread to approve incoming requests
    client_thread = DataSiteThread(target=check_and_approve_incoming_requests, args=(client,), daemon=True)
    client_thread.start()

    try:
        while True:
            sleep(2)
    except KeyboardInterrupt:
        print(f"Shutting down {name}...")
        data_site.land()
        client_thread.stop()