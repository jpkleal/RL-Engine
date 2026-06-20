"""
A minimal stand-in for the real test runner, used to develop/test
TestAbstraction without needing the actual runner up. It:

  1. Accepts a connection.
  2. Reads the start_test JSON.
  3. Immediately replies with an ack carrying a timeout value.
  4. Waits `result_delay` seconds (simulating the test actually running),
     then sends back a result message ({"failed": ..., "result_file": ...}).

Not part of the production package -- just a dev/test helper.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Optional


class MockTestRunner:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        ack_timeout_value: int = 30,
        result_delay: float = 0.2,
        failed: bool = False,
        result_file: str = "results/mock_run",
        send_ack: bool = True,
        send_result: bool = True,
        omit_timeout_in_ack: bool = False,
    ):
        self.host = host
        self.ack_timeout_value = ack_timeout_value
        self.result_delay = result_delay
        self.failed = failed
        self.result_file = result_file
        self.send_ack = send_ack
        self.send_result = send_result
        self.omit_timeout_in_ack = omit_timeout_in_ack

        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((host, port))
        self._server.listen(1)
        self.port = self._server.getsockname()[1]

        self.last_request: Optional[dict] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._serve_once, daemon=True)
        self._thread.start()

    def _serve_once(self) -> None:
        conn, _ = self._server.accept()
        with conn:
            buf = bytearray()
            while b"\n" not in buf:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                buf.extend(chunk)
            self.last_request = json.loads(buf.split(b"\n", 1)[0].decode("utf-8"))
            print(f"[mock_runner] received start_test: {self.last_request}")

            if self.send_ack:
                ack = {} if self.omit_timeout_in_ack else {"timeout": self.ack_timeout_value}
                conn.sendall((json.dumps(ack) + "\n").encode("utf-8"))

            if not self.send_result:
                # Simulate a runner that accepts the test but never responds.
                # Hold the connection open well past any reasonable client
                # timeout rather than closing it (a different failure mode).
                time.sleep(max(self.result_delay, self.ack_timeout_value + 5))
                return

            time.sleep(self.result_delay)
            result = {"failed": self.failed, "result_file": self.result_file}
            conn.sendall((json.dumps(result) + "\n").encode("utf-8"))

    def stop(self) -> None:
        self._server.close()