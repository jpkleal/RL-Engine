from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from src.rl_engine.model.playing_model import Model

logger = logging.getLogger("rl_engine.model")

PathLike = Union[str, Path]


class TrainableModel(Model):
    def __init__(self, layer_sizes: list[int], model_id: str) -> None:
        super().__init__(layer_sizes, model_id)
        # The base class doesn't keep this around, but we need it both
        # to write playing-compatible checkpoints and to reconstruct
        # architecture-identical clones/targets.
        self.layer_sizes: list[int] = list(layer_sizes)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Gradient-enabled forward pass. `inference()` (inherited from
        the playing service's Model) wraps this in torch.no_grad(),
        which is correct for serving but wrong for computing a loss."""
        return self.network(state.float())

    # ------------------------------------------------------------------ #
    # Target-network style utilities (generic across off-policy algos)
    # ------------------------------------------------------------------ #

    def clone(self, model_id: Optional[str] = None) -> "TrainableModel":
        """Architecture-identical copy with its own weights (a deep
        copy, not a reference) -- the usual starting point for a target
        network."""
        twin = TrainableModel(self.layer_sizes, model_id=model_id or self.id)
        twin.load_state_dict(self.state_dict())
        device = next(self.parameters()).device
        twin.to(device)
        return twin

    def hard_update(self, source: "TrainableModel") -> None:
        """Copies `source`'s weights into self exactly (e.g. periodic
        target-network sync in DQN)."""
        self.load_state_dict(source.state_dict())

    def soft_update(self, source: "TrainableModel", tau: float) -> None:
        """Polyak averaging: self = tau * source + (1 - tau) * self
        (e.g. DDPG/TD3/SAC-style target network updates)."""
        if not 0.0 < tau <= 1.0:
            raise ValueError(f"tau must be in (0, 1], got {tau}")
        with torch.no_grad():
            for target_param, source_param in zip(self.parameters(), source.parameters()):
                target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)

    # ------------------------------------------------------------------ #
    # Playing-service-compatible checkpointing
    # ------------------------------------------------------------------ #

    def save(self, path: PathLike) -> None:
        """
        Writes exactly the format playing_model.Model.load()/.update()
        expect. This is the method to call when handing a network off
        to the playing service.
        """
        self._atomic_save(path, {"layer_sizes": self.layer_sizes, "state_dict": self.state_dict()})

    # ------------------------------------------------------------------ #
    # Training-resume checkpointing (superset of the above)
    # ------------------------------------------------------------------ #

    def save_training_checkpoint(
        self,
        path: PathLike,
        optimizer: Optional[torch.optim.Optimizer] = None,
        step: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Superset of save(): everything the playing service needs, plus
        whatever's needed to resume training itself.
        """
        checkpoint: Dict[str, Any] = {
            "layer_sizes": self.layer_sizes,
            "state_dict": self.state_dict(),
            "model_id": self.id,
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if step is not None:
            checkpoint["step"] = step
        if extra:
            checkpoint["extra"] = extra
        self._atomic_save(path, checkpoint)

    @staticmethod
    def _atomic_save(path: PathLike, payload: Dict[str, Any]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
        logger.info("Saved model checkpoint to %s", path)
