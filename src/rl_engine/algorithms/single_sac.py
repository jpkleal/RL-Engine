"""
SAC
===

Standard Soft Actor-Critic for a single agent (one robot).

Network layout:
    actor                 state -> action mean (deployed to playing service)
    log_std               learnable, state-independent action std (training only)
    q1, q2                (state, action) -> Q value
    q1_target, q2_target  slow-moving copies for stable bootstrap targets

Conforms structurally to the `Algorithm` protocol. The actor is exposed under
`role_id` so the Trainer / playing service can treat it like any other role.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.config import SACConfig
from ..model import TrainableModel
from ..buffers.replay_memory import ReplayMemory
from .base import Metrics, PathLike
from .nets import QNetwork, ingest_to_buffers


class SAC:

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        role_id: str = "robot",
        cfg: Optional[SACConfig] = None,
        device: str = "cpu",
        reward_key: Optional[str] = None,
        checkpoint_buffer: bool = True,
    ):
        cfg = cfg if cfg is not None else SACConfig()
        self.device = torch.device(device)
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.auto_alpha = cfg.auto_alpha
        self._batch_size = cfg.batch_size
        self._min_buffer_size = cfg.min_buffer_size
        self._role_id = role_id
        self._target_entropy = float(-action_dim)
        self._checkpoint_buffer = checkpoint_buffer
        hidden_sizes = cfg.hidden_sizes

        # ── Actor ───────────────────────────────────────────────────────
        self._actor = TrainableModel(
            [state_dim, *hidden_sizes, action_dim], model_id=role_id
        ).to(self.device)
        self._log_std = nn.Parameter(torch.zeros(action_dim, device=self.device))

        # ── Twin critics + targets ──────────────────────────────────────
        self._q1 = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self._q2 = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self._q1_target = copy.deepcopy(self._q1)
        self._q2_target = copy.deepcopy(self._q2)
        for p in [*self._q1_target.parameters(), *self._q2_target.parameters()]:
            p.requires_grad_(False)

        # ── Optimizers ──────────────────────────────────────────────────
        self._critic_optimizer = torch.optim.Adam(
            list(self._q1.parameters()) + list(self._q2.parameters()),
            lr=cfg.critic_lr,
        )
        self._actor_optimizer = torch.optim.Adam(
            list(self._actor.parameters()) + [self._log_std], lr=cfg.actor_lr
        )

        # ── Entropy temperature ─────────────────────────────────────────
        self._log_alpha = nn.Parameter(
            torch.tensor(cfg.initial_alpha, device=self.device).log()
        )
        if self.auto_alpha:
            self._alpha_optimizer = torch.optim.Adam(
                [self._log_alpha], lr=cfg.alpha_lr
            )

        # ── Replay buffer (keyed by role so ingest_to_buffers is reused) ─
        self._buffers: Dict[str, ReplayMemory] = {
            role_id: ReplayMemory(
                capacity=cfg.replay_capacity,
                state_shape=(state_dim,),
                action_shape=(action_dim,),
                device=device,
            )
        }
        self._reward_keys = {role_id: reward_key or role_id}
        self._reward_signs = {role_id: 1.0}

        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.auto_alpha = cfg.auto_alpha
        self._critic_lr = cfg.critic_lr  # new
        self._actor_lr = cfg.actor_lr  # new
        self._alpha_lr = cfg.alpha_lr  # new

    # ------------------------------------------------------------------ #
    # Algorithm Protocol
    # ------------------------------------------------------------------ #

    @property
    def _buffer(self) -> ReplayMemory:
        return self._buffers[self._role_id]

    @property
    def deployable_models(self) -> Dict[str, TrainableModel]:
        return {self._role_id: self._actor}

    def ingest(self, path: PathLike) -> Dict[str, int]:
        return ingest_to_buffers(
            self._buffers, path, self._reward_keys, self._reward_signs
        )

    def ready(self) -> bool:
        return len(self._buffer) >= self._min_buffer_size

    def act(self, role_id: str, state: torch.Tensor) -> torch.Tensor:
        if role_id != self._role_id:
            raise KeyError(f"Unknown role '{role_id}', expected '{self._role_id}'")
        return self._actor.inference(state)

    def save_for_playing(self, dir_path: PathLike) -> Dict[str, str]:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        p = dir_path / f"{self._role_id}.pt"
        self._actor.save(p, extra={"log_std": self._log_std.detach().cpu()})
        return {self._role_id: str(p)}

    def save_checkpoint(self, path: PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.state_dict(), tmp)
        tmp.replace(path)

    def load_checkpoint(
        self, path: PathLike, device: Union[str, torch.device] = "cpu"
    ) -> None:
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        self.load_state_dict(state, device=device)

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #

    def update(self) -> Metrics:
        alpha = self._log_alpha.exp()

        batch = self._buffer.sample(self._batch_size, device=self.device)
        s = batch["s"].float()
        a = batch["a"].float()
        r = batch["r"].float().reshape(-1, 1)
        s_next = batch["s_next"].float()
        done = batch["done"].float().reshape(-1, 1)

        # ── Critic ──────────────────────────────────────────────────────
        with torch.no_grad():
            a_next, log_pi_next = self._sample(s_next)
            q_next = (
                torch.min(
                    self._q1_target(s_next, a_next),
                    self._q2_target(s_next, a_next),
                )
                - alpha.detach() * log_pi_next
            )
            target = r + self.gamma * (1.0 - done) * q_next

        q1_pred = self._q1(s, a)
        q2_pred = self._q2(s, a)
        critic_loss = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)
        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_optimizer.step()

        # ── Actor (critic frozen so no grads accumulate in it) ──────────
        self._set_critic_grad(False)
        pi, log_pi = self._sample(s)
        q_pi = torch.min(self._q1(s, pi), self._q2(s, pi))
        actor_loss = (alpha.detach() * log_pi - q_pi).mean()
        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        self._actor_optimizer.step()
        self._set_critic_grad(True)

        # ── Alpha ───────────────────────────────────────────────────────
        alpha_loss = torch.zeros((), device=self.device)
        if self.auto_alpha:
            alpha_loss = -(
                self._log_alpha * (log_pi.detach() + self._target_entropy)
            ).mean()
            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()

        # ── Soft target update ──────────────────────────────────────────
        self._soft_update(self._q1_target, self._q1)
        self._soft_update(self._q2_target, self._q2)

        return {
            "critic_loss": critic_loss.item() / 2,
            "actor_loss": actor_loss.item(),
            "alpha": alpha.item(),
            "alpha_loss": alpha_loss.item(),
            "entropy": -log_pi.mean().item(),
            "q_mean": q1_pred.mean().item(),
        }

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict:
        state = {
            "actor": self._actor.state_dict(),
            "log_std": self._log_std.detach().cpu(),
            "q1": self._q1.state_dict(),
            "q2": self._q2.state_dict(),
            "q1_target": self._q1_target.state_dict(),
            "q2_target": self._q2_target.state_dict(),
            "critic_optimizer": self._critic_optimizer.state_dict(),
            "actor_optimizer": self._actor_optimizer.state_dict(),
            "log_alpha": self._log_alpha.detach().cpu(),
            "auto_alpha": self.auto_alpha,
        }
        if self.auto_alpha:
            state["alpha_optimizer"] = self._alpha_optimizer.state_dict()
        if self._checkpoint_buffer:
            state["buffer"] = self._buffer.to_state_dict()
        return state

    def load_state_dict(
            self, state: dict, device: Union[str, torch.device, None] = None
    ) -> None:
        device = torch.device(device) if device is not None else self.device

        self._actor.load_state_dict(state["actor"])
        self._actor.to(device)
        self._log_std.data = state["log_std"].to(device)

        self._q1.load_state_dict(state["q1"])
        self._q2.load_state_dict(state["q2"])
        self._q1_target.load_state_dict(state["q1_target"])
        self._q2_target.load_state_dict(state["q2_target"])
        for net in (self._q1, self._q2, self._q1_target, self._q2_target):
            net.to(device)

        self._log_alpha.data = state["log_alpha"].to(device)

        # Move parameters first, then restore optimizer state (Adam moves its
        # state tensors to the parameters' device on load).
        self._critic_optimizer.load_state_dict(state["critic_optimizer"])
        self._actor_optimizer.load_state_dict(state["actor_optimizer"])
        if self.auto_alpha and "alpha_optimizer" in state:
            self._alpha_optimizer.load_state_dict(state["alpha_optimizer"])

        # load_state_dict() restores each param group's lr from the
        # checkpoint, silently overriding whatever the current config says.
        # Re-apply the current cfg's lr after restoring momentum/Adam state,
        # so a config change actually takes effect on a resumed run.
        for g in self._critic_optimizer.param_groups:
            g["lr"] = self._critic_lr
        for g in self._actor_optimizer.param_groups:
            g["lr"] = self._actor_lr
        if self.auto_alpha:
            for g in self._alpha_optimizer.param_groups:
                g["lr"] = self._alpha_lr

        if "buffer" in state:
            self._buffer.load_state_dict(state["buffer"])

        self.device = device

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _sample(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reparameterised tanh-squashed sample and its log-prob."""
        mean = self._actor(states)
        log_std = self._log_std.clamp(-20, 2)
        std = log_std.exp().expand_as(mean)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        log_pi = (
            normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        ).sum(-1, keepdim=True)
        return y_t, log_pi

    def _set_critic_grad(self, flag: bool) -> None:
        for p in [*self._q1.parameters(), *self._q2.parameters()]:
            p.requires_grad_(flag)

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1 - self.tau).add_(sp.data, alpha=self.tau)