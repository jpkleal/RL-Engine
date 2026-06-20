from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import torch

logger = logging.getLogger("rl_engine.replay_memory")

PathLike = Union[str, Path]

# Primary keys are "s", "a", "r" per spec. The next-state field is given
# as "s'" -- valid JSON, just unusual -- with a couple of fallback names
# in case the real producer uses something else.
_NEXT_STATE_ALIASES = ("s'", "s_prime", "next_state", "s2")
_DONE_ALIASES = ("done", "terminal", "is_terminal")


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
        """
        `action_shape=()` means a scalar (discrete) action -- pass an int
        per transition. For continuous/vector actions pass e.g. (3,) and
        an int dtype isn't appropriate, so default action_dtype is float;
        override to torch.int64 for discrete action ids if you want them
        stored as integers rather than floats.
        """
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
        return f"ReplayMemory(size={self._size}/{self.capacity}, state_shape={self.state_shape}, action_shape={self.action_shape})"

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
            # Only the most recent `capacity` entries in this batch can
            # possibly survive; drop the older ones up front.
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
        # Write to a temp file and rename, so a crash mid-save can't leave
        # a truncated/corrupt checkpoint behind.
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

    def ingest_new_transitions(self, path: PathLike) -> int:
        """
        Reads any lines appended to `path` since the last call against
        this same path (or the whole file, the first time), parses each
        as a transition, and pushes them into the buffer.

        Only consumes bytes up to the last complete newline -- a line
        still being written by the producer is left for next call -- so
        this is safe to call while the file is actively growing.

        Returns the number of transitions successfully ingested.
        """
        path_str = str(Path(path).resolve())
        with self._lock:
            offset = self._file_cursors.get(path_str, 0)

        with open(path, "rb") as f:
            f.seek(offset)
            chunk = f.read()

        if not chunk:
            return 0

        last_newline = chunk.rfind(b"\n")
        if last_newline == -1:
            return 0  # no complete line since last call yet

        complete = chunk[: last_newline + 1]
        new_offset = offset + len(complete)

        states, actions, rewards, next_states, dones = [], [], [], [], []
        skipped = 0
        for raw_line in complete.split(b"\n"):
            if not raw_line.strip():
                continue
            try:
                obj = json.loads(raw_line)
                s, a, r, s_next, done = self._parse_transition(obj)
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

    def _parse_transition(self, obj: dict):
        if "s" not in obj or "a" not in obj or "r" not in obj:
            raise ValueError(f"transition missing required field(s): {obj}")

        next_state_raw = None
        for key in _NEXT_STATE_ALIASES:
            if key in obj:
                next_state_raw = obj[key]
                break
        if next_state_raw is None:
            raise ValueError(
                f"transition missing next-state field (tried {_NEXT_STATE_ALIASES}): {obj}"
            )

        done_raw = False
        for key in _DONE_ALIASES:
            if key in obj:
                done_raw = obj[key]
                break

        s = torch.tensor(obj["s"], dtype=self.state_dtype)
        s_next = torch.tensor(next_state_raw, dtype=self.state_dtype)
        a = torch.tensor(obj["a"], dtype=self.action_dtype)
        r = float(obj["r"])
        done = bool(done_raw)

        if tuple(s.shape) != self.state_shape:
            raise ValueError(f"state shape {tuple(s.shape)} != expected {self.state_shape}")
        if tuple(s_next.shape) != self.state_shape:
            raise ValueError(f"next-state shape {tuple(s_next.shape)} != expected {self.state_shape}")
        if tuple(a.shape) != self.action_shape:
            raise ValueError(f"action shape {tuple(a.shape)} != expected {self.action_shape}")

        return s, a, r, s_next, done