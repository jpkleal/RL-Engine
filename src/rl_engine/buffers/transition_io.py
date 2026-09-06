"""
Shared parsing/file-cursor logic for the out.jsonl transition format,
used by both ReplayMemory and RolloutBuffer so the two buffer types
(off-policy / on-policy) don't duplicate this.

FILE FORMAT, one JSON object per line:
    {"next_state": [...], "cur_state": [...], "actions": ..., "rewards": ..., "done": true|false}

Optionally tagged with a role/agent id (see ROLE_ALIASES below) -- this
isn't in the original spec, but is needed to let a single shared/
centralized algorithm instance pull separate per-robot batches out of
what may be one combined transitions file. If the real out.jsonl is
written per-role instead (one file per robot), just don't tag lines and
don't pass a role_filter -- everything still works as a single stream.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import torch

logger = logging.getLogger("rl_engine.transition_io")

PathLike = Union[str, Path]

NEXT_STATE = "next_state"
CURRENT_STATE = "cur_state"
REWARD = "rewards"
ACTION = "actions"
DONE = "done"


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


def parse_transition(
        obj: dict,
        role_id: str,
        state_shape: Sequence[int],
        action_shape: Sequence[int],
        state_dtype: torch.dtype,
        action_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, bool]:
    """
    Parse one timestep JSON object into a single-role transition.
    Returns (s, a, r, s_next, done).

    `role_id` selects which agent's action to extract from the actions
    dict. The reward is signed per REWARD_SIGN so each role's buffer
    holds its correct reward without the algorithm needing to negate.

    Raises ValueError on missing fields or shape mismatches.
    """

    s = torch.tensor(_get_field(obj, CURRENT_STATE), dtype=state_dtype)
    s_next = torch.tensor(_get_field(obj, NEXT_STATE), dtype=state_dtype)
    a = torch.tensor(_get_field(_get_field(obj, ACTION), role_id), dtype=action_dtype)
    r = float(_get_field(obj, REWARD))
    done = bool(_get_field(obj, DONE))

    state_shape, action_shape = tuple(state_shape), tuple(action_shape)
    if tuple(s.shape) != state_shape:
        raise ValueError(f"state shape {tuple(s.shape)} != expected {state_shape}")
    if tuple(s_next.shape) != state_shape:
        raise ValueError(f"next-state shape {tuple(s_next.shape)} != expected {state_shape}")
    if tuple(a.shape) != action_shape:
        raise ValueError(f"action shape {tuple(a.shape)} != expected {action_shape}")

    return s, a, r, s_next, done


def _get_field(obj, field_name):
    if field_name not in obj:
        raise ValueError(f"transition missing required '{field_name}' field: {obj}")
    return obj[field_name]
