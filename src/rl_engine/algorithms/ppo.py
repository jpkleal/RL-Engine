"""
SharedCriticPPO
===============

MAPPO for two zero-sum adversarial agents sharing a single value network.

Network layout:
    actor_<role>   one per role  --  state -> action mean (deployed to playing service)
    log_std_<role> one per role  --  learnable action std (training only)
    value          shared        --  state -> V(s), trained on protagonist convention

Zero-sum convention:
    The value network is trained on the protagonist's (first role's)
    return throughout. The antagonist's returns are negated when computing
    GAE and value targets -- since r_antagonist = -r_protagonist, the
    shared critic learns V from one consistent perspective.
    The antagonist's policy gradient is also negated (it minimises the
    protagonist's value rather than maximising it).

Each role has its own rollout buffer. Both are cleared after every update
since on-policy data is stale once the policy moves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import PPOConfig
from ..model import TrainableModel
from ..rollout_buffer import RolloutBuffer
from .base import Metrics, PathLike, ingest_to_buffers
from .nets import ValueNetwork


class SharedCriticPPO:

    def __init__(
        self,
        role_ids: Tuple[str, str],
        state_dim: int,
        action_dim: int,
        cfg: PPOConfig = PPOConfig(),
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = cfg.gamma
        self.gae_lambda = cfg.gae_lambda
        self.clip_eps = cfg.clip_eps
        self.epochs = cfg.epochs
        self.minibatch_size = cfg.minibatch_size
        self.entropy_coef = cfg.entropy_coef
        self.value_coef = cfg.value_coef
        self._role_ids = role_ids
        self._protagonist = role_ids[0]
        self._antagonist = role_ids[1]
        hidden_sizes = cfg.hidden_sizes

        # ── Actors (one per role) ────────────────────────────────────────
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

        # ── Shared value network ─────────────────────────────────────────
        self._value = ValueNetwork(state_dim, hidden_sizes).to(self.device)

        # ── One optimizer for everything ─────────────────────────────────
        # All parameters updated jointly: value loss and both actor losses
        # are summed into a single backward pass.
        self._optimizer = torch.optim.Adam(
            list(self._value.parameters())
            + [p for role in role_ids
               for p in list(self._actors[role].parameters()) + [self._log_stds[role]]],
            lr=cfg.lr,
        )

        # ── Per-role rollout buffers ──────────────────────────────────────
        self._buffers: Dict[str, RolloutBuffer] = {
            role_id: RolloutBuffer(
                state_shape=(state_dim,),
                action_shape=(action_dim,),
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
        return all(len(buf) > 0 for buf in self._buffers.values())

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

        batch_pro = self._buffers[pro].get_all(device=self.device)
        batch_ant = self._buffers[ant].get_all(device=self.device)

        # Compute GAE advantages and value targets for both roles.
        # Protagonist uses rewards as-is; antagonist rewards are negated
        # so the shared value network stays in protagonist convention.
        adv_pro, ret_pro = self._gae(batch_pro, sign=1.0)
        adv_ant, ret_ant = self._gae(batch_ant, sign=-1.0)

        # Compute old log-probs (needed for the clipped surrogate ratio).
        # Done before any weight updates so these are truly "old policy".
        with torch.no_grad():
            old_log_pi_pro = self._log_prob(
                batch_pro["s"].float(), batch_pro["a"].float(), pro
            )
            old_log_pi_ant = self._log_prob(
                batch_ant["s"].float(), batch_ant["a"].float(), ant
            )

        # Concatenate both roles' data for the value update so the shared
        # critic sees the full game experience each epoch.
        s_all  = torch.cat([batch_pro["s"].float(), batch_ant["s"].float()])
        ret_all = torch.cat([ret_pro, ret_ant])

        policy_losses_pro, policy_losses_ant, value_losses, entropies = [], [], [], []

        n_pro = batch_pro["s"].shape[0]
        n_ant = batch_ant["s"].shape[0]
        n_all = n_pro + n_ant

        for _ in range(self.epochs):

            # ── Value update (shared critic, both roles) ─────────────
            idx_all = torch.randperm(n_all)
            for start in range(0, n_all, self.minibatch_size):
                mb = idx_all[start : start + self.minibatch_size]
                v_pred = self._value(s_all[mb]).squeeze(-1)
                v_loss = F.mse_loss(v_pred, ret_all[mb])
                value_losses.append(v_loss.item())

                self._optimizer.zero_grad()
                (self.value_coef * v_loss).backward()
                self._optimizer.step()

            # ── Actor update (protagonist) ───────────────────────────
            idx_pro = torch.randperm(n_pro)
            for start in range(0, n_pro, self.minibatch_size):
                mb = idx_pro[start : start + self.minibatch_size]
                s_mb  = batch_pro["s"].float()[mb]
                a_mb  = batch_pro["a"].float()[mb]
                adv_mb = adv_pro[mb]

                log_pi, entropy = self._log_prob_and_entropy(s_mb, a_mb, pro)
                ratio = (log_pi - old_log_pi_pro[mb]).exp()
                clipped = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps)
                # Protagonist maximises advantage
                policy_loss = -torch.min(ratio * adv_mb, clipped * adv_mb).mean()
                loss = policy_loss - self.entropy_coef * entropy.mean()

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                policy_losses_pro.append(policy_loss.item())
                entropies.append(entropy.mean().item())

            # ── Actor update (antagonist) ────────────────────────────
            idx_ant = torch.randperm(n_ant)
            for start in range(0, n_ant, self.minibatch_size):
                mb = idx_ant[start : start + self.minibatch_size]
                s_mb  = batch_ant["s"].float()[mb]
                a_mb  = batch_ant["a"].float()[mb]
                adv_mb = adv_ant[mb]

                log_pi, entropy = self._log_prob_and_entropy(s_mb, a_mb, ant)
                ratio = (log_pi - old_log_pi_ant[mb]).exp()
                clipped = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps)
                # Antagonist maximises its own advantage (already negated in GAE)
                policy_loss = -torch.min(ratio * adv_mb, clipped * adv_mb).mean()
                loss = policy_loss - self.entropy_coef * entropy.mean()

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                policy_losses_ant.append(policy_loss.item())

        # Clear buffers -- on-policy data is stale after the update.
        for buf in self._buffers.values():
            buf.clear()

        return {
            f"{pro}/policy_loss": sum(policy_losses_pro) / len(policy_losses_pro),
            f"{ant}/policy_loss": sum(policy_losses_ant) / len(policy_losses_ant),
            "value_loss":         sum(value_losses) / len(value_losses),
            "entropy":            sum(entropies) / len(entropies),
        }

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict:
        # Buffer data excluded (cleared after every update, stale on-policy
        # data is invalid to resume from). File cursors ARE saved so a
        # resumed process doesn't re-ingest already-seen lines.
        return {
            "actors":       {r: a.state_dict() for r, a in self._actors.items()},
            "log_stds":     {r: p.detach().cpu() for r, p in self._log_stds.items()},
            "value":        self._value.state_dict(),
            "optimizer":    self._optimizer.state_dict(),
            "file_cursors": {
                r: dict(buf._file_cursors) for r, buf in self._buffers.items()
            },
        }

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

        self._value.load_state_dict(state["value"])
        self._value.to(device)

        self._optimizer.load_state_dict(state["optimizer"])

        for r, cursors in state.get("file_cursors", {}).items():
            if r in self._buffers:
                self._buffers[r]._file_cursors = dict(cursors)

        self.device = device

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _gae(
        self, batch: Dict[str, torch.Tensor], sign: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generalised Advantage Estimation.

        `sign` is +1 for the protagonist (rewards used as-is) and -1 for
        the antagonist (rewards negated to protagonist convention before
        computing value targets with the shared critic).
        """
        s    = batch["s"].float()
        r    = batch["r"].float() * sign
        s_next = batch["s_next"].float()
        done = batch["done"].float()
        n = s.shape[0]

        with torch.no_grad():
            v      = self._value(s).squeeze(-1)
            v_next = self._value(s_next).squeeze(-1) * (1 - done)

        advantages = torch.zeros(n, device=self.device)
        gae = 0.0
        for t in reversed(range(n)):
            delta = r[t] + self.gamma * v_next[t] - v[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + v
        return advantages, returns

    def _log_prob(
        self, states: torch.Tensor, actions: torch.Tensor, role_id: str
    ) -> torch.Tensor:
        mean = self._actors[role_id](states)
        log_std = self._log_stds[role_id].clamp(-20, 2)
        std = log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(actions).sum(-1)

    def _log_prob_and_entropy(
        self, states: torch.Tensor, actions: torch.Tensor, role_id: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self._actors[role_id](states)
        log_std = self._log_stds[role_id].clamp(-20, 2)
        std = log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(actions).sum(-1), dist.entropy().sum(-1)