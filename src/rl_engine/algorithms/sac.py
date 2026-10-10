"""
SharedCriticSAC
===============

SAC for two roles, each with its own actor and its own twin critics.

Network layout:
    actor_<role>                        one per role  --  state -> action mean (deployed to playing service)
    log_std_<role>                      one per role  --  learnable action std (training only)
    q1_<role>, q2_<role>                one pair per role --  (state, action) -> Q value
    q1_target_<role>, q2_target_<role>  one pair per role --  slow-moving copies for stable targets

Why per-role critics:
    The two roles don't necessarily share a reward function. In the TEST
    scenario each robot gets its own shaped reward, and with
    `zero_sum = true` the second role's rewards are already negated at
    ingestion. Either way each role's buffer holds that role's own
    (s, a, r), so one critic pair fitted to both roles' samples is
    fitting two different functions at once -- and with no role input
    it cannot tell them apart.

Zero-sum convention:
    Every actor maximises its own critic. There is no protagonist /
    antagonist asymmetry here: adversarial behaviour comes from the
    reward *signs* applied at ingestion (see `zero_sum` in the config),
    not from flipping the sign of an actor loss.

Each role has its own alpha (entropy temperature) since their
exploration needs may differ even in a symmetric game.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.config import SACConfig
from ..model import TrainableModel
from ..buffers.replay_memory import ReplayMemory
from .base import Metrics, PathLike
from .nets import QNetwork, ingest_to_buffers

logger = logging.getLogger("rl_engine.algorithms.sac")


class SharedCriticSAC:

    def __init__(
        self,
        role_ids: Tuple[str, str],
        state_dim: int,
        action_dim: int,
        cfg: SACConfig = SACConfig(),
        device: str = "cpu",
        reward_keys=None,
        zero_sum=False
    ):
        self.device = torch.device(device)
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.auto_alpha = cfg.auto_alpha
        self._batch_size = cfg.batch_size
        self._min_buffer_size = cfg.min_buffer_size
        self._updates_per_transition = cfg.updates_per_transition
        self._max_updates_per_call = cfg.max_updates_per_call
        self._pending = 0
        self._role_ids = role_ids
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

        # ── Per-role twin critics ────────────────────────────────────────
        self._q1: Dict[str, QNetwork] = {
            role_id: QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
            for role_id in role_ids
        }
        self._q2: Dict[str, QNetwork] = {
            role_id: QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
            for role_id in role_ids
        }
        self._q1_target: Dict[str, QNetwork] = {}
        self._q2_target: Dict[str, QNetwork] = {}
        for role_id in role_ids:
            q1_target = copy.deepcopy(self._q1[role_id])
            q2_target = copy.deepcopy(self._q2[role_id])
            for p in [*q1_target.parameters(), *q2_target.parameters()]:
                p.requires_grad_(False)
            self._q1_target[role_id] = q1_target
            self._q2_target[role_id] = q2_target

        # ── Optimizers ───────────────────────────────────────────────────
        self._critic_optimizers: Dict[str, torch.optim.Adam] = {
            role_id: torch.optim.Adam(
                list(self._q1[role_id].parameters()) + list(self._q2[role_id].parameters()),
                lr=cfg.critic_lr,
            )
            for role_id in role_ids
        }
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

        self._reward_keys = {r: (reward_keys or {}).get(r, r) for r in role_ids}
        self._reward_signs = {
            role_ids[0]: 1.0,
            role_ids[1]: -1.0 if zero_sum else 1.0,
        }

    # ------------------------------------------------------------------ #
    # Algorithm Protocol
    # ------------------------------------------------------------------ #

    @property
    def deployable_models(self) -> Dict[str, TrainableModel]:
        return dict(self._actors)

    def ingest(self, path: PathLike) -> Dict[str, int]:
        stats = ingest_to_buffers(self._buffers, path, self._reward_keys, self._reward_signs)
        # Both roles see the same frames, so the max is the batch size
        # rather than the sum.
        self._pending += max((st.n for st in stats.values()), default=0)
        return stats

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
        """Runs as many gradient steps as the newly ingested data justifies.

        Returns metrics averaged over those steps, plus the step count
        itself so the ratio is visible in TensorBoard.
        """
        k = min(
            self._max_updates_per_call,
            max(1, round(self._pending * self._updates_per_transition)),
        )
        self._pending = 0

        totals: Dict[str, float] = {}
        for _ in range(k):
            for tag, value in self._update_once().items():
                totals[tag] = totals.get(tag, 0.0) + value

        metrics = {tag: value / k for tag, value in totals.items()}
        metrics["sac/gradient_steps"] = float(k)
        return metrics

    def _update_once(self) -> Metrics:
        alphas = {r: self._log_alphas[r].exp() for r in self._role_ids}
        metrics: Dict[str, float] = {}
        critic_losses = []

        for role_id in self._role_ids:
            batch = self._buffers[role_id].sample(self._batch_size, device=self.device)
            s = batch["s"].float()
            a = batch["a"].float()
            r = batch["r"].float().unsqueeze(-1)
            s_next = batch["s_next"].float()
            done = batch["done"].float().unsqueeze(-1)
            alpha = alphas[role_id]

            # ── Critic ────────────────────────────────────────────────
            with torch.no_grad():
                next_a, log_pi_next = self._sample(s_next, role_id)
                q_next = (
                    torch.min(
                        self._q1_target[role_id](s_next, next_a),
                        self._q2_target[role_id](s_next, next_a),
                    )
                    - alpha * log_pi_next
                )
                target = r + self.gamma * (1 - done) * q_next

            q1, q2 = self._q1[role_id], self._q2[role_id]
            critic_loss = F.mse_loss(q1(s, a), target) + F.mse_loss(q2(s, a), target)
            critic_optimizer = self._critic_optimizers[role_id]
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            critic_losses.append(critic_loss.item())
            metrics[f"{role_id}/critic_loss"] = critic_loss.item()

            # ── Actor (freeze this role's critics during its update) ───
            for p in [*q1.parameters(), *q2.parameters()]:
                p.requires_grad_(False)

            pi, log_pi = self._sample(s, role_id)
            q_pi = torch.min(q1(s, pi), q2(s, pi))
            actor_loss = (alpha.detach() * log_pi - q_pi).mean()
            actor_optimizer = self._actor_optimizers[role_id]
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            for p in [*q1.parameters(), *q2.parameters()]:
                p.requires_grad_(True)

            metrics[f"{role_id}/actor_loss"] = actor_loss.item()
            metrics[f"{role_id}/alpha"] = alpha.item()

            # ── Alpha ──────────────────────────────────────────────────
            if self.auto_alpha:
                alpha_loss = -(
                    self._log_alphas[role_id] * (log_pi + self._target_entropy).detach()
                ).mean()
                self._alpha_optimizers[role_id].zero_grad()
                alpha_loss.backward()
                self._alpha_optimizers[role_id].step()
                metrics[f"{role_id}/alpha_loss"] = alpha_loss.item()
            else:
                metrics[f"{role_id}/alpha_loss"] = 0.0

            # ── Soft target update ─────────────────────────────────────
            self._soft_update(self._q1_target[role_id], self._q1[role_id])
            self._soft_update(self._q2_target[role_id], self._q2[role_id])

        # Reported as a single headline number (the mean over roles).
        metrics["critic_loss"] = sum(critic_losses) / len(critic_losses)
        return metrics

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict:
        state = {
            "format_version":    2,
            "actors":           {r: a.state_dict() for r, a in self._actors.items()},
            "log_stds":         {r: p.detach().cpu() for r, p in self._log_stds.items()},
            "q1":               {r: q.state_dict() for r, q in self._q1.items()},
            "q2":               {r: q.state_dict() for r, q in self._q2.items()},
            "q1_target":        {r: q.state_dict() for r, q in self._q1_target.items()},
            "q2_target":        {r: q.state_dict() for r, q in self._q2_target.items()},
            "critic_optimizers": {r: o.state_dict() for r, o in self._critic_optimizers.items()},
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

        if state.get("format_version") != 2:
            # Pre-per-role-critics checkpoints carry a single shared Q
            # pair, which has no per-role counterpart. Everything else
            # transfers; the critics start fresh and relearn.
            logger.warning("legacy shared-critic checkpoint: critics re-initialised")
        else:
            for r in self._role_ids:
                self._q1[r].load_state_dict(state["q1"][r])
                self._q2[r].load_state_dict(state["q2"][r])
                self._q1_target[r].load_state_dict(state["q1_target"][r])
                self._q2_target[r].load_state_dict(state["q2_target"][r])
            for r in self._role_ids:
                for net in (
                    self._q1[r], self._q2[r],
                    self._q1_target[r], self._q2_target[r],
                ):
                    net.to(device)
            for r, o in state["critic_optimizers"].items():
                if r in self._critic_optimizers:
                    self._critic_optimizers[r].load_state_dict(o)

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