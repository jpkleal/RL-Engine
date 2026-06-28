from __future__ import annotations

import json
import logging
import socket
import time
from typing import Any, Dict, Optional

from .commons import TestResult, TestCase, TestAbstractionError, InputModule

logger = logging.getLogger("rl_engine.test_abstraction")


class TestAbstraction:
    """
    TCP client for a single test-runner test case.
    """

    def __init__(
        self,
        host: str,
        port: int,
        connect_timeout: float = 5.0,
        ack_timeout: float = 5.0,
        default_result_timeout: float = 600.0,
        max_result_timeout: Optional[float] = 3600.0,
        max_batch_size: Optional[int] = None,
        encoding: str = "utf-8",
    ):
        """
        `default_result_timeout` is only used as a fallback if the
        runner's ack is somehow missing the "timeout" field (it's
        required by spec, but real-world runners do ship bugs).

        `max_result_timeout` is a safety cap: if the runner asks us to
        wait longer than this, we clip it and log a warning, so a buggy
        or malicious ack can't hang a training loop indefinitely. Set to
        None to disable the cap.

        `max_batch_size` enforces the spec's "needs a limit" note on
        batch_size, checked locally before ever contacting the runner.
        Set to None to skip the check (e.g. if the real limit isn't
        known yet).
        """
        self.host = host
        self.port = port
        self.connect_timeout = connect_timeout
        self.ack_timeout = ack_timeout
        self.default_result_timeout = default_result_timeout
        self.max_result_timeout = max_result_timeout
        self.max_batch_size = max_batch_size
        self.encoding = encoding

    def start_test(
        self,
        test_case: "TestCase | str",
        batch_size: int,
        input_module: "InputModule | str",
        module_config: Optional[Dict[str, Any]] = None,
        verbose_out: bool = True,
    ) -> TestResult:
        """
        Starts a test case on the test runner and blocks until either the
        final result arrives, the runner rejects the test, or a timeout
        happens. Never raises for protocol/network/validation problems --
        those are reported back as a failed TestResult so callers (RL
        training loops) don't need a try/except around every call.

        `module_config` is nested under "{input_module.lower()}_config"
        (e.g. "neonfc_config") and omitted from the payload entirely if
        None, matching the spec's "optional[json]".
        """
        if self.max_batch_size is not None and batch_size > self.max_batch_size:
            return TestResult(
                success=False,
                error=f"batch_size {batch_size} exceeds max_batch_size {self.max_batch_size}",
            )

        test_case_value = test_case.value if isinstance(test_case, TestCase) else test_case
        module_value = (
            input_module.value if isinstance(input_module, InputModule) else input_module
        )

        payload: Dict[str, Any] = {
            "teste_case": test_case_value,
            "batch_size": batch_size,
            "input_module": module_value,
            "verbose_out": verbose_out,
        }
        if module_config is not None:
            payload[f"{module_value.lower()}_config"] = module_config

        sock: Optional[socket.socket] = None
        try:
            sock = socket.create_connection(
                (self.host, self.port), timeout=self.connect_timeout
            )
            logger.info("Connected to test runner at %s:%s", self.host, self.port)

            self._send_json(sock, payload)
            logger.debug("Sent start_test payload: %s", payload)

            result_timeout = self._await_ack(sock)
            logger.debug(
                "Received ack from runner, waiting up to %.1fs for result", result_timeout
            )

            return self._await_result(sock, result_timeout)

        except socket.timeout as e:
            logger.error("Timed out talking to test runner: %s", e)
            return TestResult(success=False, error=f"timeout: {e}")
        except (ConnectionError, OSError) as e:
            logger.error("Connection error talking to test runner: %s", e)
            return TestResult(success=False, error=f"connection_error: {e}")
        except TestAbstractionError as e:
            logger.error("Protocol error: %s", e)
            return TestResult(success=False, error=str(e))
        finally:
            if sock is not None:
                sock.close()

    def _send_json(self, sock: socket.socket, obj: Dict[str, Any]) -> None:
        data = (json.dumps(obj) + "\n").encode(self.encoding)
        sock.sendall(data)

    def _recv_line(self, sock: socket.socket, timeout: float) -> str:
        """Reads bytes until a newline, with `timeout` covering the whole wait."""
        deadline = time.monotonic() + timeout
        buf = bytearray()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise socket.timeout("timed out waiting for newline-terminated message")
            sock.settimeout(remaining)
            chunk = sock.recv(4096)
            if not chunk:
                raise TestAbstractionError(
                    "connection closed by test runner before message completed"
                )
            buf.extend(chunk)
            if b"\n" in chunk:
                break
        line, _, _ = buf.partition(b"\n")
        return line.decode(self.encoding)

    def _await_ack(self, sock: socket.socket) -> float:
        """Reads the ack and returns the result timeout (seconds) it specifies."""
        line = self._recv_line(sock, self.ack_timeout)
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            raise TestAbstractionError(f"malformed ack from runner: {e}") from e

        if not isinstance(msg, dict) or "timeout" not in msg:
            logger.warning(
                "ack missing required 'timeout' field (%s); falling back to default %.1fs",
                msg,
                self.default_result_timeout,
            )
            return self.default_result_timeout

        timeout = msg["timeout"]
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise TestAbstractionError(f"ack 'timeout' field is invalid: {msg}")

        timeout = float(timeout)
        if self.max_result_timeout is not None and timeout > self.max_result_timeout:
            logger.warning(
                "runner requested timeout %.1fs exceeds max_result_timeout %.1fs; clipping",
                timeout,
                self.max_result_timeout,
            )
            timeout = self.max_result_timeout
        return timeout

    def _await_result(self, sock: socket.socket, timeout: float) -> TestResult:
        line = self._recv_line(sock, timeout)
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            return TestResult(success=False, error=f"malformed result message: {e}")

        if not isinstance(msg, dict) or "failed" not in msg:
            return TestResult(
                success=False, error=f"result message missing required 'failed' field: {msg}"
            )

        failed = bool(msg["failed"])
        result_file = msg.get("result_file")

        return TestResult(
            success=not failed,
            result_file=result_file,
            error=None if not failed else "test runner reported failure (failed=true)",
        )
