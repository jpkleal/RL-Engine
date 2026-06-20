from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TestCase(str, Enum):
    """Known test cases. Extend as the runner adds more."""

    SHOOTOUT = "SHOOTOUT"


class InputModule(str, Enum):
    NEONFC = "NeonFC"
    AUTOREF = "AutoRef"


class TestAbstractionError(Exception):
    """Protocol-level failure: bad handshake, malformed message, etc."""


@dataclass
class TestResult:
    success: bool
    result_file: Optional[str] = None
    error: Optional[str] = None
