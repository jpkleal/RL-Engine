"""
config.py
=========

Typed configuration for the entire RL-Engine. The TOML file structure
is unchanged -- only the Python side moves from raw dicts to dataclasses.

Usage:
    cfg = Config.from_toml("config.toml")
    cfg.sac.actor_lr       # float, IDE-complete, typo-safe
    cfg.training.role_ids  # Tuple[str, str]

__post_init__ on each class coerces TOML lists to tuples where needed,
so Config.from_toml() can do SACConfig(**raw["sac"]) without extra
conversion logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Union


# ────────────────────────────────────────────────────────────────────────
# Section configs
# ────────────────────────────────────────────────────────────────────────

@dataclass
class RunnerConfig:
    host: str = "127.0.0.1"
    port: int = 9999
    connect_timeout: float = 5.0
    ack_timeout: float = 5.0


@dataclass
class TrainingConfig:
    algorithm: str = "sac"
    role_ids: Tuple[str, str] = ("striker", "goalkeeper")
    state_dim: int = 16
    action_dim: int = 4
    rollout_batch_size: int = 32
    save_every_n_epochs: int = 10
    test_case: str = "SHOOTOUT"
    input_module: str = "NeonFC"

    def __post_init__(self):
        self.role_ids = tuple(self.role_ids)


@dataclass
class SACConfig:
    hidden_sizes: Tuple[int, ...] = (256, 256)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    initial_alpha: float = 0.2
    auto_alpha: bool = True
    replay_capacity: int = 200_000
    min_buffer_size: int = 1_000
    batch_size: int = 256

    def __post_init__(self):
        self.hidden_sizes = tuple(self.hidden_sizes)


@dataclass
class PPOConfig:
    hidden_sizes: Tuple[int, ...] = (256, 256)
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    epochs: int = 10
    minibatch_size: int = 64
    entropy_coef: float = 0.01
    value_coef: float = 0.5

    def __post_init__(self):
        self.hidden_sizes = tuple(self.hidden_sizes)


@dataclass
class PathsConfig:
    save_dir: str = "runs/training"
    log_dir: str = "runs/tensorboard"


# ────────────────────────────────────────────────────────────────────────
# Top-level config
# ────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @classmethod
    def from_toml(cls, path: Union[str, Path]) -> "Config":
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[no-redef]

        with open(path, "rb") as f:
            raw = tomllib.load(f)

        return cls(
            runner=RunnerConfig(**raw.get("runner", {})),
            training=TrainingConfig(**raw.get("training", {})),
            sac=SACConfig(**raw.get("sac", {})),
            ppo=PPOConfig(**raw.get("ppo", {})),
            paths=PathsConfig(**raw.get("paths", {})),
        )