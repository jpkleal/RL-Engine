from __future__ import annotations

from pathlib import Path
from typing import Dict, Protocol, Union, runtime_checkable
import torch

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
