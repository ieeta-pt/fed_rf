# server.py

from threading import Thread, Event
from time import sleep


from fed_rf_mk.datasites import spawn_server, check_and_approve_incoming_requests


class DataSiteThread(Thread):
    def __init__(self, *args, **kwargs):
        super(DataSiteThread, self).__init__(*args, **kwargs)
        self._stop_event = Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class FLServer:
    def __init__(self, name: str, port: int, data_path: str, mock_path: str = None, auto_accept: bool = True):
        self.name = name
        self.port = port
        self.data_path = data_path
        self.mock_path = mock_path if mock_path else data_path
        self.auto_accept = auto_accept
        self.thread = None
        self.data_site = None
        self.client = None

    def start(self):
        print(f"Starting DataSite {self.name} on port {self.port} with data at {self.data_path} and mock at {self.mock_path}")
        self.data_site, self.client = spawn_server(
            name=self.name,
            port=self.port,
            data_path=self.data_path,
            mock_path=self.mock_path
        )

        if self.auto_accept:
            self.thread = DataSiteThread(target=check_and_approve_incoming_requests, args=(self.client,), daemon=True)
            self.thread.start()
        else:
            print("Server running in manual mode. Use `.list_pending_requests()` to view requests.")

        try:
            while True:
                sleep(2)
        except KeyboardInterrupt:
            print(f"Shutting down {self.name}...")
            self.shutdown()

    def list_pending_requests(self):
        if self.client is None:
            print("Client not initialized.")
            return

        print("Pending requests:")
        for idx, code in enumerate(self.client.code):
            if not code.status.approved:
                print(f"[{idx}] Status: {code.status}")

    def approve_request(self, request_index: int):
        """
        Approve a single incoming request by index, using the same
        nested-approval flag as the auto‑approve loop.
        """
        try:
            req = self.client.requests[request_index]
            # mirrors the tutorial helper’s r.approve(approve_nested=True)
            req.approve(approve_nested=True)
            print(f"✅ Approved request at index {request_index}.")
        except IndexError:
            print(f"❌ No request at index {request_index}.")
        except Exception as e:
            print(f"❌ Error approving request: {e}")

    def inspect_request(self, request_index: int):
        """
        Return the code object attached to the incoming request at `request_index`.
        """
        if self.client is None:
            print("Client not initialized.")
            return None

        try:
            req = self.client.requests[request_index]
        except IndexError:
            print(f"❌ No request at index {request_index}.")
            return None

        return req.code
    
    def shutdown(self):
        if self.data_site:
            self.data_site.land()
        if self.thread:
            self.thread.stop()
