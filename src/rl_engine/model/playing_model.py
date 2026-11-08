"""
This is a  copy of the network class defined in the
playing service. RL-Engine doesn't import the playing service's code
directly (separate processes/repos, no shared package today), so this
copy IS the contract: whatever checkpoint format this class's
`load()`/`update()` can read is what RL-Engine has to write.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

# Must match the live NeonFC model.py exactly -- both copies need to
# agree on input scale or a checkpoint trained here means something
# different once deployed.
X_MAX, Y_MAX = 9.0, 6.0
V_MAX = 4.0
VTHETA_MAX = 10.0


class Model(nn.Module):
    def __init__(self, layer_sizes: list[int], model_id: str) -> None:
        super().__init__()
        assert len(layer_sizes) >= 2, "Need at least an input and output layer."

        self.id: str = model_id
        self.register_buffer("state_scale", self._default_state_scale(layer_sizes[0]))

        layers = []
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layers.append(nn.Linear(in_size, out_size))
            is_last_layer = i == len(layer_sizes) - 2
            if not is_last_layer:
                layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

    @staticmethod
    def _default_state_scale(state_dim: int) -> torch.Tensor:
        per_robot = torch.tensor([X_MAX, Y_MAX, math.pi, V_MAX, V_MAX, VTHETA_MAX])
        ball = torch.tensor([X_MAX, Y_MAX, V_MAX, V_MAX])
        n_robots = (state_dim - ball.numel()) // per_robot.numel()
        scale = torch.cat([per_robot] * n_robots + [ball])
        assert scale.numel() == state_dim, (
            f"state_dim {state_dim} doesn't match the expected robot/ball "
            f"layout ({n_robots} robots -> {scale.numel()} features)"
        )
        return scale

    def _normalize(self, state: torch.Tensor) -> torch.Tensor:
        return state.float() / self.state_scale

    def inference(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.tanh(self.network(self._normalize(state)))

    def update(self, path: str) -> None:
        checkpoint = torch.load(path, weights_only=True)
        self.load_state_dict(checkpoint["state_dict"])

    @classmethod
    def load(cls, path: str, model_id: str | None = None) -> "Model":
        checkpoint = torch.load(path, weights_only=True)
        model = cls(checkpoint["layer_sizes"], model_id=model_id)
        model.load_state_dict(checkpoint["state_dict"])
        return model