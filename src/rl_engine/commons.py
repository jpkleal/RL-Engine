from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any


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


def build_model_update_event(model_paths: Dict[str, str]) -> Dict[str, Any]:
    """
    Builds the `event_data` payload for a NEONFC "ModelUpdate" event
    (see test_abstraction.py) from a {model_id: checkpoint_path}
    mapping, e.g.:

        build_model_update_event({
            "striker": "models/current/striker.pt",
            "goalkeeper": "models/current/goalkeeper.pt",
        })
        # -> {"models": [
        #       {"id": "striker", "file_path": "models/current/striker.pt"},
        #       {"id": "goalkeeper", "file_path": "models/current/goalkeeper.pt"},
        #    ]}
    """
    return {"models": [{"id": mid, "file_path": path} for mid, path in model_paths.items()]}
