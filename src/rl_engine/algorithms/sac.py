"""
SharedCriticSAC
===============

SAC for two zero-sum adversarial agents sharing a single Q-network pair.

Network layout:
    actor_<role>              one per role  --  state -> action mean (deployed to playing service)
    log_std_<role>            one per role  --  learnable action std (training only)
    q1, q2                    shared        --  (state, action) -> Q value
    q1_target, q2_target      shared        --  slow-moving copies for stable targets

Zero-sum convention:
    The critic is trained on the protagonist's (first role's) reward
    convention throughout. The antagonist's rewards are negated when
    computing critic targets since r_antagonmist = -r_protagonist.
    The antagonist's actor loss is also negated -- it minimises Q
    rather than maximises it.

Each role has its own alpha (entropy temperature) since their
exploration needs may differ even in a symmetric game.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import SACConfig
from ..model import TrainableModel
from src.rl_engine.buffers.replay_memory import ReplayMemory
from .base import Metrics, PathLike, ingest_to_buffers
from .nets import QNetwork


class SharedCriticSAC:

    def __init__(
        self,
        role_ids: Tuple[str, str],
        state_dim: int,
        action_dim: int,
        cfg: SACConfig = SACConfig(),
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.auto_alpha = cfg.auto_alpha
        self._batch_size = cfg.batch_size
        self._min_buffer_size = cfg.min_buffer_size
        self._role_ids = role_ids
        self._protagonist = role_ids[0]
        self._antagonist = role_ids[1]
        self._target_entropy = float(-action_dim)
        hidden_sizes = cfg.hidden_sizes

        # ── Actors (one per role, deployed to playing service) ──────────
        self._actors: Dict[str, TrainableModel] = {
            role_id: TrainableModel(
                [state_dim, *hidden_sizes, action_dim], model_id=role_id
            ).to(self.device)
            for role_id in role_ids
        }
        self._log_stds: Dict[str, nn.Parameter] = {
            role_id: nn.Parameter(torch.zeros(action_dim, device=self.device))
            for role_id in role_ids
        }

        # ── Shared twin critics ──────────────────────────────────────────
        self._q1 = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self._q2 = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self._q1_target = copy.deepcopy(self._q1)
        self._q2_target = copy.deepcopy(self._q2)
        for p in [*self._q1_target.parameters(), *self._q2_target.parameters()]:
            p.requires_grad_(False)

        # ── Optimizers ───────────────────────────────────────────────────
        self._critic_optimizer = torch.optim.Adam(
            list(self._q1.parameters()) + list(self._q2.parameters()), lr=cfg.critic_lr
        )
        self._actor_optimizers: Dict[str, torch.optim.Adam] = {
            role_id: torch.optim.Adam(
                list(self._actors[role_id].parameters()) + [self._log_stds[role_id]],
                lr=cfg.actor_lr,
            )
            for role_id in role_ids
        }

        # ── Per-role entropy temperature ─────────────────────────────────
        self._log_alphas: Dict[str, nn.Parameter] = {
            role_id: nn.Parameter(
                torch.tensor(cfg.initial_alpha, device=self.device).log()
            )
            for role_id in role_ids
        }
        if self.auto_alpha:
            self._alpha_optimizers: Dict[str, torch.optim.Adam] = {
                role_id: torch.optim.Adam([self._log_alphas[role_id]], lr=cfg.alpha_lr)
                for role_id in role_ids
            }

        # ── Per-role replay buffers ───────────────────────────────────────
        self._buffers: Dict[str, ReplayMemory] = {
            role_id: ReplayMemory(
                capacity=cfg.replay_capacity,
                state_shape=(state_dim,),
                action_shape=(action_dim,),
                device=device,
            )
            for role_id in role_ids
        }

    # ------------------------------------------------------------------ #
    # Algorithm Protocol
    # ------------------------------------------------------------------ #

    @property
    def deployable_models(self) -> Dict[str, TrainableModel]:
        return dict(self._actors)

    def ingest(self, path: PathLike) -> Dict[str, int]:
        return ingest_to_buffers(self._buffers, path)

    def ready(self) -> bool:
        return all(len(buf) >= self._min_buffer_size for buf in self._buffers.values())

    def act(self, role_id: str, state: torch.Tensor) -> torch.Tensor:
        return self._actors[role_id].inference(state)

    def save_for_playing(self, dir_path: PathLike) -> Dict[str, str]:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        paths = {}
        for role_id, actor in self._actors.items():
            p = dir_path / f"{role_id}.pt"
            actor.save(p)
            paths[role_id] = str(p)
        return paths

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
        pro, ant = self._protagonist, self._antagonist
        alphas = {r: self._log_alphas[r].exp() for r in self._role_ids}

        batch_pro = self._buffers[pro].sample(self._batch_size, device=self.device)
        batch_ant = self._buffers[ant].sample(self._batch_size, device=self.device)

        s_pro      = batch_pro["s"].float()
        a_pro      = batch_pro["a"].float()
        r_pro      = batch_pro["r"].float().unsqueeze(-1)
        s_next_pro = batch_pro["s_next"].float()
        done_pro   = batch_pro["done"].float().unsqueeze(-1)

        s_ant      = batch_ant["s"].float()
        a_ant      = batch_ant["a"].float()
        r_ant      = batch_ant["r"].float().unsqueeze(-1)
        s_next_ant = batch_ant["s_next"].float()
        done_ant   = batch_ant["done"].float().unsqueeze(-1)

        # ── Critic ────────────────────────────────────────────────────
        with torch.no_grad():
            next_a_pro, log_pi_next_pro = self._sample(s_next_pro, pro)
            q1_t_pro = self._q1_target(s_next_pro, next_a_pro)
            q2_t_pro = self._q2_target(s_next_pro, next_a_pro)
            q_next_pro = torch.min(q1_t_pro, q2_t_pro) - alphas[pro] * log_pi_next_pro
            target_pro = r_pro + self.gamma * (1 - done_pro) * q_next_pro

            next_a_ant, log_pi_next_ant = self._sample(s_next_ant, ant)
            q1_t_ant = self._q1_target(s_next_ant, next_a_ant)
            q2_t_ant = self._q2_target(s_next_ant, next_a_ant)
            q_next_ant = torch.min(q1_t_ant, q2_t_ant) - alphas[ant] * log_pi_next_ant
            # Negate: critic always in protagonist convention
            target_ant = -r_ant + self.gamma * (1 - done_ant) * q_next_ant

        critic_loss = (
            F.mse_loss(self._q1(s_pro, a_pro), target_pro)
            + F.mse_loss(self._q2(s_pro, a_pro), target_pro)
            + F.mse_loss(self._q1(s_ant, a_ant), target_ant)
            + F.mse_loss(self._q2(s_ant, a_ant), target_ant)
        )
        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_optimizer.step()

        # ── Actors (freeze critic during actor updates) ────────────────
        for p in [*self._q1.parameters(), *self._q2.parameters()]:
            p.requires_grad_(False)

        # Protagonist: maximise Q
        pi_pro, log_pi_pro = self._sample(s_pro, pro)
        q_pi_pro = torch.min(self._q1(s_pro, pi_pro), self._q2(s_pro, pi_pro))
        actor_loss_pro = (alphas[pro].detach() * log_pi_pro - q_pi_pro).mean()
        self._actor_optimizers[pro].zero_grad()
        actor_loss_pro.backward()
        self._actor_optimizers[pro].step()

        # Antagonist: minimise Q (+ q_pi instead of - q_pi)
        pi_ant, log_pi_ant = self._sample(s_ant, ant)
        q_pi_ant = torch.min(self._q1(s_ant, pi_ant), self._q2(s_ant, pi_ant))
        actor_loss_ant = (alphas[ant].detach() * log_pi_ant + q_pi_ant).mean()
        self._actor_optimizers[ant].zero_grad()
        actor_loss_ant.backward()
        self._actor_optimizers[ant].step()

        for p in [*self._q1.parameters(), *self._q2.parameters()]:
            p.requires_grad_(True)

        # ── Alpha ──────────────────────────────────────────────────────
        alpha_loss_pro = alpha_loss_ant = torch.tensor(0.0)
        if self.auto_alpha:
            alpha_loss_pro = -(
                self._log_alphas[pro] * (log_pi_pro + self._target_entropy).detach()
            ).mean()
            self._alpha_optimizers[pro].zero_grad()
            alpha_loss_pro.backward()
            self._alpha_optimizers[pro].step()

            alpha_loss_ant = -(
                self._log_alphas[ant] * (log_pi_ant + self._target_entropy).detach()
            ).mean()
            self._alpha_optimizers[ant].zero_grad()
            alpha_loss_ant.backward()
            self._alpha_optimizers[ant].step()

        # ── Soft target update ─────────────────────────────────────────
        self._soft_update(self._q1_target, self._q1)
        self._soft_update(self._q2_target, self._q2)

        return {
            "critic_loss":        critic_loss.item() / 4,
            f"{pro}/actor_loss":  actor_loss_pro.item(),
            f"{ant}/actor_loss":  actor_loss_ant.item(),
            f"{pro}/alpha":       alphas[pro].item(),
            f"{ant}/alpha":       alphas[ant].item(),
            f"{pro}/alpha_loss":  alpha_loss_pro.item() if self.auto_alpha else 0.0,
            f"{ant}/alpha_loss":  alpha_loss_ant.item() if self.auto_alpha else 0.0,
        }

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict:
        state = {
            "actors":           {r: a.state_dict() for r, a in self._actors.items()},
            "log_stds":         {r: p.detach().cpu() for r, p in self._log_stds.items()},
            "q1":               self._q1.state_dict(),
            "q2":               self._q2.state_dict(),
            "q1_target":        self._q1_target.state_dict(),
            "q2_target":        self._q2_target.state_dict(),
            "critic_optimizer": self._critic_optimizer.state_dict(),
            "actor_optimizers": {r: o.state_dict() for r, o in self._actor_optimizers.items()},
            "log_alphas":       {r: p.detach().cpu() for r, p in self._log_alphas.items()},
            "auto_alpha":       self.auto_alpha,
            "buffers":          {r: b.to_state_dict() for r, b in self._buffers.items()},
        }
        if self.auto_alpha:
            state["alpha_optimizers"] = {
                r: o.state_dict() for r, o in self._alpha_optimizers.items()
            }
        return state

    def load_state_dict(
        self, state: dict, device: Union[str, torch.device, None] = None
    ) -> None:
        device = torch.device(device) if device is not None else self.device

        for r, s in state["actors"].items():
            if r in self._actors:
                self._actors[r].load_state_dict(s)
                self._actors[r].to(device)

        for r, log_std in state["log_stds"].items():
            if r in self._log_stds:
                self._log_stds[r].data = log_std.to(device)

        self._q1.load_state_dict(state["q1"])
        self._q2.load_state_dict(state["q2"])
        self._q1_target.load_state_dict(state["q1_target"])
        self._q2_target.load_state_dict(state["q2_target"])
        for net in (self._q1, self._q2, self._q1_target, self._q2_target):
            net.to(device)

        self._critic_optimizer.load_state_dict(state["critic_optimizer"])
        for r, s in state["actor_optimizers"].items():
            if r in self._actor_optimizers:
                self._actor_optimizers[r].load_state_dict(s)

        for r, la in state["log_alphas"].items():
            if r in self._log_alphas:
                self._log_alphas[r].data = la.to(device)

        if self.auto_alpha and "alpha_optimizers" in state:
            for r, s in state["alpha_optimizers"].items():
                if r in self._alpha_optimizers:
                    self._alpha_optimizers[r].load_state_dict(s)

        for r, buf_state in state["buffers"].items():
            if r in self._buffers:
                self._buffers[r].load_state_dict(buf_state)

        self.device = device

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _sample(
        self, states: torch.Tensor, role_id: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reparameterised sample + log prob for role_id's actor."""
        mean = self._actors[role_id](states)
        log_std = self._log_stds[role_id].clamp(-20, 2)
        std = log_std.exp().expand_as(mean)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        log_pi = (
            normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        ).sum(-1, keepdim=True)
        return y_t, log_pi

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1 - self.tau).add_(sp.data, alpha=self.tau)