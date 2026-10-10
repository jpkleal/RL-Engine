from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any


class TestCase(str, Enum):
    """Known test cases. Extend as the runner adds more."""
    SHOOTOUT = "SHOOTOUT"
    TEST = "TEST"


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
    # A batch runs one file per episode; result_file stays as the
    # first (or None) so single-file callers keep working unchanged.
    result_files: List[str] = field(default_factory=list)


def build_model_update_event(
    model_paths: Dict[str, str],
    stochastic: Optional[bool] = None,
    hold_frames: Optional[int] = None,
) -> Dict[str, Any]:
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
