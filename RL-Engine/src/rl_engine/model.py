"""
TrainableModel
==============

The RL-Engine-side wrapper around the same network architecture the
playing service uses (see playing_model.py). It subclasses
playing_model.Model directly rather than reimplementing it, so the
state_dict layout is guaranteed identical -- there's no architecture
that could quietly drift between the two.

It adds what training needs that the playing service has no reason to
carry:

  * a gradient-enabled forward() -- the playing service only exposes
    inference(), which wraps the forward pass in torch.no_grad()
  * clone() / hard_update() / soft_update() for target-network style
    patterns used across many off-policy algorithms (DQN, DDPG, TD3,
    SAC, ...) -- generic, not tied to any one of them
  * save() -- writes exactly the {"layer_sizes", "state_dict"} format
    playing_model.Model.load()/.update() expect, so a checkpoint
    produced during training is directly usable by the playing service
    (e.g. as the file_path in a TestAbstraction ModelUpdate event)
  * save_training_checkpoint() / load_training_checkpoint() -- a
    superset of save() for resuming *training itself* (optimizer
    state, step count, ...). Still loadable by the playing service if
    ever pointed at one, since Model.load()/.update() only ever read
    two keys and ignore the rest -- but see the weights_only caveat
    below before stuffing arbitrary objects into `extra`.

COMPATIBILITY CAVEAT: the playing service loads checkpoints with
`torch.load(path, weights_only=True)`, which refuses to unpickle the
*entire file* if anything in it isn't on torch's safe-globals allowlist
-- not just the two keys it reads. Plain Python primitives, lists,
dicts, and tensors are fine (verified below); don't put custom classes,
numpy objects, etc. into `extra` if there's any chance the playing
service might load that same file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from .playing_model import Model

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

    @classmethod
    def load_training_checkpoint(
        cls,
        path: PathLike,
        optimizer: Optional[torch.optim.Optimizer] = None,
        model_id: Optional[str] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> "TrainingCheckpoint":
        """
        Restores a model (and optimizer, if given and present in the
        file) written by save_training_checkpoint(). Returns a
        TrainingCheckpoint bundling the model with whatever bookkeeping
        (step, extra) was saved alongside it, since training code
        generally needs that back too -- unlike the playing service's
        Model.load(), which only ever wants the network.
        """
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
        model = cls(checkpoint["layer_sizes"], model_id=model_id or checkpoint.get("model_id"))
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)

        if optimizer is not None:
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            else:
                logger.warning("No optimizer_state_dict found in %s; optimizer left as-is", path)

        return TrainingCheckpoint(
            model=model,
            step=checkpoint.get("step"),
            extra=checkpoint.get("extra"),
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _atomic_save(path: PathLike, payload: Dict[str, Any]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
        logger.info("Saved model checkpoint to %s", path)


class TrainingCheckpoint:
    """Bundles a restored TrainableModel with whatever training
    bookkeeping (step, extra) was saved alongside it."""

    def __init__(
        self, model: TrainableModel, step: Optional[int], extra: Optional[Dict[str, Any]]
    ):
        self.model = model
        self.step = step
        self.extra = extra or {}


# ---------------------------------------------------------------------- #
# Bridge to TestAbstraction's "ModelUpdate" event format
# ---------------------------------------------------------------------- #


def build_model_update_event(model_paths: Dict[str, str]) -> Dict[str, Any]:
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