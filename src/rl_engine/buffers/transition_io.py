"""
Shared parsing/file-cursor logic for the out.jsonl transition format,
used by both ReplayMemory and RolloutBuffer so the two buffer types
(off-policy / on-policy) don't duplicate this.

FILE FORMAT, one JSON object per line:
    {"s": [...], "s'": [...], "a": ..., "r": ...}

Optionally tagged with a role/agent id (see ROLE_ALIASES below) -- this
isn't in the original spec, but is needed to let a single shared/
centralized algorithm instance pull separate per-robot batches out of
what may be one combined transitions file. If the real out.jsonl is
written per-role instead (one file per robot), just don't tag lines and
don't pass a role_filter -- everything still works as a single stream.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import torch

logger = logging.getLogger("rl_engine.transition_io")

PathLike = Union[str, Path]

NEXT_STATE_ALIASES = ("s'", "s_prime", "next_state", "s2")
DONE_ALIASES = ("done", "terminal", "is_terminal")
ROLE_ALIASES = ("role", "agent_id", "id")


def read_new_complete_lines(path: PathLike, offset: int) -> Tuple[List[bytes], int]:
    """
    Reads bytes from `offset` to EOF, keeps only up to the last complete
    "\\n" (a trailing partial line, still being written, is left for next
    time), and returns (non-empty raw lines, new offset to resume from).
    """
    with open(path, "rb") as f:
        f.seek(offset)
        chunk = f.read()

    if not chunk:
        return [], offset

    last_newline = chunk.rfind(b"\n")
    if last_newline == -1:
        return [], offset

    complete = chunk[: last_newline + 1]
    lines = [line for line in complete.split(b"\n") if line.strip()]
    return lines, offset + len(complete)


def extract_role(obj: dict) -> Optional[str]:
    """Cheap pre-check: get a transition's role/agent tag, if any, without
    doing full shape-validated parsing. Lets callers skip lines for other
    roles before paying the cost of (and logging spurious warnings from)
    parse_transition's shape validation against the wrong role's shapes."""
    role = next((obj[k] for k in ROLE_ALIASES if k in obj), None)
    return str(role) if role is not None else None


def parse_transition(
    obj: dict,
    state_shape: Sequence[int],
    action_shape: Sequence[int],
    state_dtype: torch.dtype,
    action_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, bool, Optional[str]]:
    """Returns (s, a, r, s_next, done, role). Raises ValueError on any
    missing field or shape mismatch -- callers should catch and skip."""
    if "s" not in obj or "a" not in obj or "r" not in obj:
        raise ValueError(f"transition missing required field(s): {obj}")

    next_state_raw = next((obj[k] for k in NEXT_STATE_ALIASES if k in obj), None)
    if next_state_raw is None:
        raise ValueError(
            f"transition missing next-state field (tried {NEXT_STATE_ALIASES}): {obj}"
        )

    done_raw = next((obj[k] for k in DONE_ALIASES if k in obj), False)
    role = next((obj[k] for k in ROLE_ALIASES if k in obj), None)

    s = torch.tensor(obj["s"], dtype=state_dtype)
    s_next = torch.tensor(next_state_raw, dtype=state_dtype)
    a = torch.tensor(obj["a"], dtype=action_dtype)
    r = float(obj["r"])
    done = bool(done_raw)

    state_shape, action_shape = tuple(state_shape), tuple(action_shape)
    if tuple(s.shape) != state_shape:
        raise ValueError(f"state shape {tuple(s.shape)} != expected {state_shape}")
    if tuple(s_next.shape) != state_shape:
        raise ValueError(f"next-state shape {tuple(s_next.shape)} != expected {state_shape}")
    if tuple(a.shape) != action_shape:
        raise ValueError(f"action shape {tuple(a.shape)} != expected {action_shape}")

    return s, a, r, s_next, done, (str(role) if role is not None else None)
