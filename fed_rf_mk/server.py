"""Local datasite server utilities."""

import logging
from threading import Thread, Event
from time import sleep
from datasites import (
    spawn_server,
    check_and_approve_incoming_requests,
    set_analysis_allowed,
    is_analysis_allowed,
)

logger = logging.getLogger(__name__)


class DataSiteThread(Thread):
    def __init__(self, *args, **kwargs):
        super(DataSiteThread, self).__init__(*args, **kwargs)
        self._stop_event = Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class FLServer:
    def __init__(self, name: str, port: int, data_path: str, mock_path: str = None, auto_accept: bool = True, analysis_allowed: bool | None = None):
        self.name = name
        self.port = port
        self.data_path = data_path
        self.mock_path = mock_path if mock_path else data_path
        self.auto_accept = auto_accept
        # Optional default analysis policy to set at startup. If None, leaves current policy.
        self._analysis_allowed_default = analysis_allowed
        self.thread = None
        self.data_site = None
        self.client = None

    def start(self):
        logger.info(
            "Starting DataSite %s on port %s with data at %s and mock at %s",
            self.name,
            self.port,
            self.data_path,
            self.mock_path,
        )

        # If a default policy was provided, set/override the module policy before launching
        if self._analysis_allowed_default is not None:
            set_analysis_allowed(self._analysis_allowed_default)

        self.data_site, self.client = spawn_server(
            name=self.name,
            port=self.port,
            data_path=self.data_path,
            mock_path=self.mock_path
        )

        # Announce local analysis policy to make behavior explicit
        logger.info("Analysis policy: %s", "ENABLED" if is_analysis_allowed() else "DISABLED")

        if self.auto_accept:
            self.thread = DataSiteThread(
                target=check_and_approve_incoming_requests, args=(self.client,), daemon=True
            )
            self.thread.start()
        else:
            logger.info("Server running in manual mode. Use `.list_pending_requests()` to view requests.")

        try:
            while True:
                sleep(2)
        except KeyboardInterrupt:
            logger.info("Shutting down %s...", self.name)
            self.shutdown()

    def list_pending_requests(self):
        if self.client is None:
            logger.error("Client not initialized.")
            return

        logger.info("Pending requests:")
        for idx, code in enumerate(self.client.code):
            if not code.status.approved:
                logger.info("[%s] Status: %s", idx, code.status)

    def approve_request(self, request_index: int):
        """
        Approve a single incoming request by index, using the same
        nested-approval flag as the auto‑approve loop.
        """
        try:
            req = self.client.requests[request_index]
            # mirrors the tutorial helper’s r.approve(approve_nested=True)
            req.approve(approve_nested=True)
            logger.info("Approved request at index %s.", request_index)
        except IndexError:
            logger.error("No request at index %s.", request_index)
        except Exception as e:
            logger.error("Error approving request: %s", e)

    def inspect_request(self, request_index: int):
        """
        Return the code object attached to the incoming request at `request_index`.
        """
        if self.client is None:
            logger.error("Client not initialized.")
            return None

        try:
            req = self.client.requests[request_index]
        except IndexError:
            logger.error("No request at index %s.", request_index)
            return None

        return req.code
    
    def shutdown(self):
        if self.data_site:
            self.data_site.land()
        if self.thread:
            self.thread.stop()

    # --- Convenience controls for analysis policy ---
    def set_analysis_policy(self, enabled: bool) -> None:
        """Enable/disable analysis at runtime for this server process.

        Takes effect for subsequent code executions that consult datasites.is_analysis_allowed().
        """
        set_analysis_allowed(enabled)
        logger.info("Analysis policy set to: %s", "ENABLED" if enabled else "DISABLED")

    def is_analysis_allowed(self) -> bool:
        """Return current effective policy from the datasites module."""
        return is_analysis_allowed()
