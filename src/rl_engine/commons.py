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

    `stochastic` and `hold_frames` are only emitted when given, so the
    default payload stays byte-identical to what an older playing service
    understands. (It isn't the other way round: an older service raises
    TypeError on the unknown keys, so only send them to one that has the
    matching support.)
    """
    extras: Dict[str, Any] = {}
    if stochastic is not None:
        extras["stochastic"] = stochastic
    if hold_frames is not None:
        extras["hold_frames"] = hold_frames

    return {
        "models": [
            {"id": mid, "file_path": path, **extras} for mid, path in model_paths.items()
        ]
    }
