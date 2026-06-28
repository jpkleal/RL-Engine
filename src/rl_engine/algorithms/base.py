"""
base.py
=======

Shared utilities for the algorithms package.

  * Algorithm  -- Protocol defining the contract both SharedCriticSAC
                  and SharedCriticPPO satisfy. Checked at construction
                  time via isinstance() in Trainer (runtime_checkable).
                  No inheritance required -- structural conformance only.
  * PathLike, Metrics  -- type aliases shared across algorithm files
  * ingest_to_buffers  -- the one piece of logic genuinely identical
                          across both implementations
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Protocol, Union, runtime_checkable

import torch

from .buffer import Buffer
from ..model import TrainableModel

PathLike = Union[str, Path]
Metrics = Dict[str, float]


@runtime_checkable
class Algorithm(Protocol):
    """Structural contract Trainer expects. Any class with this shape
    qualifies -- no inheritance from this Protocol required."""

    @property
    def deployable_models(self) -> Dict[str, TrainableModel]: ...

    def ingest(self, path: PathLike) -> Dict[str, int]: ...

    def ready(self) -> bool: ...

    def update(self) -> Metrics: ...

    def act(self, role_id: str, state: torch.Tensor) -> torch.Tensor: ...

    def save_checkpoint(self, path: PathLike) -> None: ...

    def load_checkpoint(
        self, path: PathLike, device: Union[str, torch.device] = "cpu"
    ) -> None: ...

    def save_for_playing(self, dir_path: PathLike) -> Dict[str, str]: ...


def ingest_to_buffers(
    buffers: Dict[str, Buffer],
    path: PathLike,
) -> Dict[str, int]:
    """Fan out ingest calls to per-role buffers from a shared transitions
    file, each buffer filtering for its own role-tagged lines."""
    return {
        role_id: buf.ingest_new_transitions(path, role_filter=role_id)
        for role_id, buf in buffers.items()
    }