"""
ReplayMemory
============

PyTorch-native circular replay buffer for off-policy RL training, plus
two pieces of file I/O specific to this project:

  * save() / load() / load_or_create() -- persist/restore the *entire*
    buffer as a torch checkpoint. Meant to be called on RL-Engine
    startup so training can resume without losing accumulated
    experience.
  * ingest_new_transitions(path) -- incrementally read newly appended
    lines from a transitions file (the "out.jsonl" produced by a test
    run -- presumably the `result_file` a TestAbstraction.start_test()
    call returns) and push them into the buffer. Designed to be called
    repeatedly as the file grows; it remembers how far it has already
    read (per file path) so each call only consumes new lines.

See transition_io.py for the file format and parsing details shared
with RolloutBuffer (the on-policy counterpart used by e.g. PPO-style
algorithms).

ASSUMPTIONS (please confirm against the real producer of out.jsonl):
  * No "done"/terminal flag is in the spec'd format. Every ingested
    transition defaults to done=False unless a "done" or "terminal" key
    is present.
  * Malformed lines (bad JSON, wrong shape, missing field) are logged
    and skipped rather than aborting ingestion of the rest of the file.
  * `out.jsonl` may still be being appended to while we read it. Only
    bytes up to the last complete "\n" are consumed.
  * `role_filter` (optional) lets one combined file serve multiple
    roles/agents if lines are tagged with a "role"/"agent_id" key --
    see transition_io.ROLE_ALIASES. Untagged lines are always included
    regardless of filter, so this is fully backward compatible with a
    single-role file that has no role tagging at all.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import torch

from .transition_io import extract_role, parse_transition, read_new_complete_lines

logger = logging.getLogger("rl_engine.replay_memory")

PathLike = Union[str, Path]


class ReplayMemory:
    def __init__(
        self,
        capacity: int,
        state_shape: Sequence[int],
        action_shape: Sequence[int] = (),
        state_dtype: torch.dtype = torch.float32,
        action_dtype: torch.dtype = torch.float32,
        reward_dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ):
        self.capacity = capacity
        self.state_shape = tuple(state_shape)
        self.action_shape = tuple(action_shape)
        self.state_dtype = state_dtype
        self.action_dtype = action_dtype
        self.reward_dtype = reward_dtype
        self.device = torch.device(device)

        # Storage always lives on CPU regardless of `device` -- buffers
        # are often far bigger than comfortably fits on a GPU. Samples
        # are moved to `device` only when drawn via sample().
        self._states = torch.zeros((capacity, *self.state_shape), dtype=state_dtype)
        self._next_states = torch.zeros((capacity, *self.state_shape), dtype=state_dtype)
        self._actions = torch.zeros((capacity, *self.action_shape), dtype=action_dtype)
        self._rewards = torch.zeros((capacity,), dtype=reward_dtype)
        self._dones = torch.zeros((capacity,), dtype=torch.bool)

        self._position = 0  # next index to write to
        self._size = 0  # number of valid entries (<= capacity)

        # path -> byte offset already consumed.
        self._file_cursors: Dict[str, int] = {}

        self._lock = threading.Lock()

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return (
            f"ReplayMemory(size={self._size}/{self.capacity}, "
            f"state_shape={self.state_shape}, action_shape={self.action_shape})"
        )

    # ------------------------------------------------------------------ #
    # Insertion
    # ------------------------------------------------------------------ #

    def push(self, state, action, reward: float, next_state, done: bool = False) -> None:
        """Add a single transition."""
        self._push_batch(
            torch.as_tensor(state, dtype=self.state_dtype).unsqueeze(0),
            torch.as_tensor(action, dtype=self.action_dtype).unsqueeze(0),
            torch.as_tensor([reward], dtype=self.reward_dtype),
            torch.as_tensor(next_state, dtype=self.state_dtype).unsqueeze(0),
            torch.as_tensor([done], dtype=torch.bool),
        )

    def _push_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        n = states.shape[0]
        if n == 0:
            return
        if n > self.capacity:
            states, actions, rewards, next_states, dones = (
                t[-self.capacity :] for t in (states, actions, rewards, next_states, dones)
            )
            n = self.capacity

        with self._lock:
            idx = (self._position + torch.arange(n)) % self.capacity
            self._states[idx] = states
            self._actions[idx] = actions
            self._rewards[idx] = rewards
            self._next_states[idx] = next_states
            self._dones[idx] = dones

            self._position = (self._position + n) % self.capacity
            self._size = min(self._size + n, self.capacity)

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #

    def sample(
        self, batch_size: int, device: Optional[Union[str, torch.device]] = None
    ) -> Dict[str, torch.Tensor]:
        """Uniform random sample with replacement, the common DRL default."""
        if self._size == 0:
            raise ValueError("cannot sample from an empty replay memory")
        target_device = torch.device(device) if device is not None else self.device

        with self._lock:
            indices = torch.randint(0, self._size, (batch_size,))
            return {
                "s": self._states[indices].to(target_device),
                "a": self._actions[indices].to(target_device),
                "r": self._rewards[indices].to(target_device),
                "s_next": self._next_states[indices].to(target_device),
                "done": self._dones[indices].to(target_device),
            }

    # ------------------------------------------------------------------ #
    # Whole-buffer persistence (startup load/save)
    # ------------------------------------------------------------------ #

    def to_state_dict(self) -> dict:
        """In-memory snapshot for embedding inside an algorithm checkpoint.
        Use save(path) for standalone buffer files."""
        with self._lock:
            return {
                "states": self._states.clone(),
                "next_states": self._next_states.clone(),
                "actions": self._actions.clone(),
                "rewards": self._rewards.clone(),
                "dones": self._dones.clone(),
                "position": self._position,
                "size": self._size,
                "file_cursors": dict(self._file_cursors),
            }

    def load_state_dict(self, state: dict) -> None:
        """Restores from a snapshot produced by to_state_dict(), in place."""
        with self._lock:
            self._states = state["states"]
            self._next_states = state["next_states"]
            self._actions = state["actions"]
            self._rewards = state["rewards"]
            self._dones = state["dones"]
            self._position = state["position"]
            self._size = state["size"]
            self._file_cursors = dict(state.get("file_cursors", {}))

    def save(self, path: PathLike) -> None:
        """Persists the entire buffer (contents + ingestion cursors) to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            checkpoint = {
                "capacity": self.capacity,
                "state_shape": self.state_shape,
                "action_shape": self.action_shape,
                "state_dtype": self.state_dtype,
                "action_dtype": self.action_dtype,
                "reward_dtype": self.reward_dtype,
                "states": self._states,
                "next_states": self._next_states,
                "actions": self._actions,
                "rewards": self._rewards,
                "dones": self._dones,
                "position": self._position,
                "size": self._size,
                "file_cursors": dict(self._file_cursors),
            }
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, tmp_path)
        tmp_path.replace(path)
        logger.info("Saved replay memory (%d/%d entries) to %s", self._size, self.capacity, path)

    @classmethod
    def load(cls, path: PathLike, device: Union[str, torch.device] = "cpu") -> "ReplayMemory":
        """Restores a buffer previously written by save()."""
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
        rm = cls(
            capacity=checkpoint["capacity"],
            state_shape=checkpoint["state_shape"],
            action_shape=checkpoint["action_shape"],
            state_dtype=checkpoint["state_dtype"],
            action_dtype=checkpoint["action_dtype"],
            reward_dtype=checkpoint["reward_dtype"],
            device=device,
        )
        rm._states = checkpoint["states"]
        rm._next_states = checkpoint["next_states"]
        rm._actions = checkpoint["actions"]
        rm._rewards = checkpoint["rewards"]
        rm._dones = checkpoint["dones"]
        rm._position = checkpoint["position"]
        rm._size = checkpoint["size"]
        rm._file_cursors = checkpoint.get("file_cursors", {})
        logger.info("Loaded replay memory (%d/%d entries) from %s", rm._size, rm.capacity, path)
        return rm

    @classmethod
    def load_or_create(
        cls,
        path: PathLike,
        capacity: int,
        state_shape: Sequence[int],
        action_shape: Sequence[int] = (),
        **kwargs,
    ) -> "ReplayMemory":
        """
        Typical RL-Engine startup call: resume from a checkpoint if one
        exists at `path`, otherwise start with a fresh, empty buffer.
        """
        if Path(path).exists():
            return cls.load(path, device=kwargs.get("device", "cpu"))
        logger.info("No replay memory checkpoint at %s; starting empty", path)
        return cls(capacity=capacity, state_shape=state_shape, action_shape=action_shape, **kwargs)

    # ------------------------------------------------------------------ #
    # Ingesting new transitions from out.jsonl
    # ------------------------------------------------------------------ #

    def ingest_new_transitions(self, path: PathLike, role_filter: Optional[str] = None) -> int:
        """
        Reads any lines appended to `path` since the last call against
        this same path (or the whole file, the first time), parses each
        as a transition, and pushes them into the buffer.

        `role_filter`: if given, lines tagged with a different role/agent
        id are skipped (untagged lines are always included). Lets one
        combined transitions file serve multiple roles -- see
        transition_io.ROLE_ALIASES.

        Returns the number of transitions actually pushed (after any
        role filtering).
        """
        path_str = str(Path(path).resolve())
        with self._lock:
            offset = self._file_cursors.get(path_str, 0)

        lines, new_offset = read_new_complete_lines(path, offset)
        if not lines:
            return 0

        states, actions, rewards, next_states, dones = [], [], [], [], []
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
                    continue  # belongs to a different role -- not an error, just not ours

            try:
                s, a, r, s_next, done, _role = parse_transition(
                    obj, self.state_shape, self.action_shape, self.state_dtype, self.action_dtype
                )
            except Exception as e:  # noqa: BLE001 -- one bad line shouldn't kill ingestion
                skipped += 1
                logger.warning("Skipping malformed transition in %s: %s", path, e)
                continue

            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(s_next)
            dones.append(done)

        if states:
            self._push_batch(
                torch.stack(states),
                torch.stack(actions),
                torch.tensor(rewards, dtype=self.reward_dtype),
                torch.stack(next_states),
                torch.tensor(dones, dtype=torch.bool),
            )

        with self._lock:
            self._file_cursors[path_str] = new_offset

        if skipped:
            logger.warning("Ingested %d transitions from %s (%d skipped)", len(states), path, skipped)
        else:
            logger.debug("Ingested %d transitions from %s", len(states), path)
        return len(states)
