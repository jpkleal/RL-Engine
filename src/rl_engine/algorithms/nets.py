"""
nets.py
=======

Small shared building blocks used by the algorithm implementations.
Not meant to be used outside of the algorithms/ package.
"""

from __future__ import annotations

from typing import Tuple, Type

import torch
import torch.nn as nn


def mlp(
    sizes: Tuple[int, ...],
    activation: Type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    """Builds a fully-connected network with `activation` between every
    pair of layers and no activation after the final layer."""
    layers = []
    for i, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(in_size, out_size))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """Q(s, a) -> scalar. Used by SAC's twin critics."""

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...] = (256, 256)):
        super().__init__()
        self.net = mlp((state_dim + action_dim, *hidden_sizes, 1))

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1))


class ValueNetwork(nn.Module):
    """V(s) -> scalar. Used by PPO's critic."""

    def __init__(self, state_dim: int, hidden_sizes: Tuple[int, ...] = (256, 256)):
        super().__init__()
        self.net = mlp((state_dim, *hidden_sizes, 1))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)