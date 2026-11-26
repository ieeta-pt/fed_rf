"""Local datasite server utilities."""

import logging
from threading import Thread, Event
from time import sleep
from fed_rf_mk.datasites import (
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
    """One DataSite. Code requests are approved by hand unless ``auto_accept``
    is True (``list_pending_requests``, ``inspect_request``,
    ``request_parameters``, ``approve_request``).

    ``mock_path`` is a mock file of the data owner. Without it a synthetic mock
    is built (``datasets.generate_mock``); the real data is never the mock.
    ``admin_password``, ``ds_email``, ``ds_password`` and ``allow_guest_signup``
    go to ``datasites.spawn_server`` (or come from its environment variables).
    """

    def __init__(
        self,
        name: str,
        port: int,
        data_path: str,
        mock_path: str = None,
        auto_accept: bool = False,
        analysis_allowed: bool | None = None,
        admin_password: str | None = None,
        ds_email: str | None = None,
        ds_password: str | None = None,
        allow_guest_signup: bool | None = None,
    ):
        self.name = name
        self.port = port
        self.data_path = data_path
        self.mock_path = mock_path
        self.auto_accept = auto_accept
        # Optional default analysis policy to set at startup. If None, leaves current policy.
        self._analysis_allowed_default = analysis_allowed
        self._accounts = dict(admin_password=admin_password, ds_email=ds_email,
                              ds_password=ds_password, allow_guest_signup=allow_guest_signup)
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
            mock_path=self.mock_path,
            **self._accounts,
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
        """Indices (in ``client.requests``) of the requests not yet approved
        or rejected, with the name of the function each one would run."""
        if self.client is None:
            logger.error("Client not initialized.")
            return []
        pending = []
        for idx, req in enumerate(self.client.requests):
            if req.status.value == 0:  # 0 == PENDING
                func = getattr(req.code, "service_func_name", "?")
                logger.info("[%s] %s, status %s", idx, func, req.status)
                pending.append((idx, func))
        return pending

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

    def request_parameters(self, request_index: int) -> dict:
        """Values bound in the request at ``request_index`` (``dataParams`` and
        ``modelParams``). The approved code runs only with these values."""
        code = self.inspect_request(request_index)
        if code is None:
            return {}
        out = {}
        for rules in (code.input_policy_init_kwargs or {}).values():
            for kw, rule in rules.items():
                if hasattr(rule, "val") and kw != "data":
                    out[kw] = rule.val
        return out

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
