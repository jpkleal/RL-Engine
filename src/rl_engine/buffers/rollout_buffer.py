"""
RolloutBuffer
=============

The on-policy counterpart to ReplayMemory. Where ReplayMemory is a
fixed-capacity circular buffer sampled randomly (right for off-policy
algorithms like SAC), RolloutBuffer just grows as transitions are
ingested and is meant to be fully consumed and cleared after each
update -- the usual on-policy pattern (PPO and friends): collect a
batch of fresh experience under the current policy, train on exactly
that batch (often for several epochs), then discard it, since stale
on-policy data is invalid for the next update.

Shares the same out.jsonl parsing/file-cursor logic as ReplayMemory
(see transition_io.py), including the same optional role_filter for
pulling one role's data out of a combined multi-role file.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch

from .transition_io import extract_role, parse_transition, read_new_complete_lines

logger = logging.getLogger("rl_engine.rollout_buffer")

PathLike = Union[str, Path]


class RolloutBuffer:
    def __init__(
        self,
        state_shape: Sequence[int],
        action_shape: Sequence[int] = (),
        state_dtype: torch.dtype = torch.float32,
        action_dtype: torch.dtype = torch.float32,
        reward_dtype: torch.dtype = torch.float32,
    ):
        self.state_shape = tuple(state_shape)
        self.action_shape = tuple(action_shape)
        self.state_dtype = state_dtype
        self.action_dtype = action_dtype
        self.reward_dtype = reward_dtype

        self._states: List[torch.Tensor] = []
        self._actions: List[torch.Tensor] = []
        self._rewards: List[float] = []
        self._next_states: List[torch.Tensor] = []
        self._dones: List[bool] = []

        # Cursors persist across clear() -- clearing the *data* doesn't
        # mean we should re-read transitions tagged for other roles or
        # already-consumed lines from the same file.
        self._file_cursors: Dict[str, int] = {}

        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._rewards)

    def __repr__(self) -> str:
        return f"RolloutBuffer(size={len(self)}, state_shape={self.state_shape}, action_shape={self.action_shape})"

    # ------------------------------------------------------------------ #
    # Ingesting new transitions from out.jsonl
    # ------------------------------------------------------------------ #

    def ingest_new_transitions(self, path: PathLike, role_filter: Optional[str] = None) -> int:
        """Same semantics as ReplayMemory.ingest_new_transitions: reads
        only newly-appended complete lines, optionally filtered by role,
        and appends them. Returns the number ingested."""
        path_str = str(Path(path).resolve())
        with self._lock:
            offset = self._file_cursors.get(path_str, 0)

        lines, new_offset = read_new_complete_lines(path, offset)
        if not lines:
            return 0

        added = 0
        skipped = 0
        for raw_line in lines:
            try:
                obj = json.loads(raw_line)
            except Exception as e:  # noqa: BLE001
                skipped += 1
                logger.warning("Skipping malformed transition in %s: %s", path, e)
                continue

            if role_filter is not None:
                role = extract_role(obj)
                if role is not None and role != role_filter:
                    continue

            try:
                s, a, r, s_next, done, _role = parse_transition(
                    obj, self.state_shape, self.action_shape, self.state_dtype, self.action_dtype
                )
            except Exception as e:  # noqa: BLE001
                skipped += 1
                logger.warning("Skipping malformed transition in %s: %s", path, e)
                continue

            with self._lock:
                self._states.append(s)
                self._actions.append(a)
                self._rewards.append(r)
                self._next_states.append(s_next)
                self._dones.append(done)
            added += 1

        with self._lock:
            self._file_cursors[path_str] = new_offset

        if skipped:
            logger.warning("Ingested %d transitions from %s (%d skipped)", added, path, skipped)
        else:
            logger.debug("Ingested %d transitions from %s", added, path)
        return added

    # ------------------------------------------------------------------ #
    # Consumption
    # ------------------------------------------------------------------ #

    def get_all(self, device: Optional[Union[str, torch.device]] = None) -> Dict[str, torch.Tensor]:
        """Returns every transition currently held, in ingestion order
        (on-policy algorithms generally want this order preserved for
        e.g. GAE computation, unlike ReplayMemory's random sampling)."""
        if len(self) == 0:
            raise ValueError("cannot get_all from an empty rollout buffer")
        with self._lock:
            states = torch.stack(self._states)
            actions = torch.stack(self._actions)
            rewards = torch.tensor(self._rewards, dtype=self.reward_dtype)
            next_states = torch.stack(self._next_states)
            dones = torch.tensor(self._dones, dtype=torch.bool)

        if device is not None:
            states, actions, rewards, next_states, dones = (
                t.to(device) for t in (states, actions, rewards, next_states, dones)
            )
        return {"s": states, "a": actions, "r": rewards, "s_next": next_states, "done": dones}

    def clear(self) -> None:
        """Discards all currently-held transitions (call after the
        algorithm has consumed get_all() for its update). File read
        cursors are preserved -- this only clears the in-memory data,
        not the bookkeeping of what's already been read from disk."""
        with self._lock:
            self._states.clear()
            self._actions.clear()
            self._rewards.clear()
            self._next_states.clear()
            self._dones.clear()
