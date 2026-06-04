#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sac.py

Soft Actor-Critic (SAC) training loop.

This is a production-grade, self-contained implementation that:
- Fixes missing imports + syntax errors in your uploaded sac.py
- Works with the LONG multi-index environment that emits **2D observations**
  (assets × features). The networks flatten internally (still MLP).
- Uses ReplayBuffer that supports tuple observation shapes.

Expected environment API: gym-like
    obs = env.reset()
    obs2, rew, done, info = env.step(action)

Notes for your setup:
- Your environment scales actions with a running_scaler inside get_state.
  SAC does not need to touch that; it just outputs actions in env bounds.
- If your env is initialized with use_2d=True, obs is (n_assets, n_features).
"""

from __future__ import annotations

import json
import logging
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.optim import Adam

from ReplayBuffers import ReplayBuffer, DEVICE
from sac_core import MLPActorCritic


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------


class SimpleLogger:
    """Lightweight logger compatible with your previous logger.store usage."""

    def __init__(self):
        self._store: Dict[str, list] = {}

    def store(self, **kwargs):
        for k, v in kwargs.items():
            self._store.setdefault(k, []).append(v)

    def get_stats(self) -> Dict[str, float]:
        out = {}
        for k, vals in self._store.items():
            try:
                out[k] = float(np.mean(vals))
            except Exception:
                pass
        return out


def set_seed(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# SAC
# -----------------------------------------------------------------------------


class SAC(object):
    def __init__(
        self,
        state_dim: Optional[int],
        action_dim: Optional[int],
        env: Union[Callable[[], Any], Any],
        test_env: Optional[Union[Callable[[], Any], Any]] = None,
        hidden1_dim: int = 256,
        hidden2_dim: int = 256,
        lr: float = 3e-4,
        alpha: float = 0.2,
        gamma: float = 0.99,
        tau: float = 0.005,
        replay_buffer_size: int = int(1e6),
        batch_size: int = 256,
        auto_alpha: bool = False,
        alpha_lr: float = 3e-4,
        actor_critic=MLPActorCritic,
        ac_kwargs: dict = None,
        seed: int = 42,
        steps_per_epoch: int = 4000,
        epochs: int = 100,
        polyak: float = 0.995,
        start_steps: int = 10000,
        update_after: int = 1000,
        update_every: int = 50,
        num_test_episodes: int = 10,
        max_ep_len: int = 1000,
        save_freq: int = 10,
        reward_scale: float = 1.0,
        logger_kwargs: dict = None,
        **unused,
    ):
        self.logger = SimpleLogger()
        self.python_logger = logging.getLogger("SAC")

        ac_kwargs = ac_kwargs or {}
        logger_kwargs = logger_kwargs or {}

        set_seed(seed)

        # env can be instance or fn
        self.env = env() if callable(env) else env
        self.test_env = (test_env() if callable(test_env) else test_env) if test_env is not None else None

        self.obs_shape = tuple(int(x) for x in self.env.observation_space.shape)
        self.act_dim = int(self.env.action_space.shape[0])

        self.gamma = float(gamma)
        self.polyak = float(polyak)
        self.tau = float(tau)
        self.reward_scale = float(reward_scale)

        self.steps_per_epoch = int(steps_per_epoch)
        self.epochs = int(epochs)
        self.start_steps = int(start_steps)
        self.update_after = int(update_after)
        self.update_every = int(update_every)
        self.num_test_episodes = int(num_test_episodes)
        self.max_ep_len = int(max_ep_len)
        self.save_freq = int(save_freq)

        # Actor-Critic
        hidden_sizes = (int(hidden1_dim), int(hidden2_dim))
        self.ac = actor_critic(self.env.observation_space, self.env.action_space, hidden_sizes=hidden_sizes, **ac_kwargs)
        self.ac_targ = deepcopy(self.ac)

        # Freeze target parameters
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # Replay buffer (tuple obs shape supported)
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_shape, act_dim=self.act_dim, size=int(replay_buffer_size))
        self.batch_size = int(batch_size)

        # Optimizers
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=float(lr))
        self.q_params = list(self.ac.q1.parameters()) + list(self.ac.q2.parameters())
        self.q_optimizer = Adam(self.q_params, lr=float(lr))

        # Entropy temperature
        self.auto_alpha = bool(auto_alpha)
        self.target_entropy = -float(self.act_dim)
        if self.auto_alpha:
            self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, device=DEVICE, requires_grad=True)
            self.alpha_optimizer = Adam([self.log_alpha], lr=float(alpha_lr))
        else:
            self.alpha = float(alpha)

    @property
    def alpha_value(self) -> torch.Tensor:
        if self.auto_alpha:
            return torch.exp(self.log_alpha)
        return torch.tensor(self.alpha, dtype=torch.float32, device=DEVICE)

    def get_action(self, o: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self.ac.act(o, deterministic=deterministic)

    def compute_loss_q(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        o, a, r, o2, d = data["obs"], data["act"], data["rew"], data["obs2"], data["done"]
        r = r * self.reward_scale

        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        with torch.no_grad():
            a2, logp_a2 = self.ac.pi(o2)
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)

            backup = r + self.gamma * (1.0 - d) * (q_pi_targ - self.alpha_value * logp_a2)

        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        info = {
            "Q1": float(q1.mean().detach().cpu().item()),
            "Q2": float(q2.mean().detach().cpu().item()),
            "LossQ1": float(loss_q1.detach().cpu().item()),
            "LossQ2": float(loss_q2.detach().cpu().item()),
        }
        return loss_q, info

    def compute_loss_pi(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        o = data["obs"]
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        loss_pi = (self.alpha_value * logp_pi - q_pi).mean()

        info = {
            "LogPi": float(logp_pi.mean().detach().cpu().item()),
            "QPi": float(q_pi.mean().detach().cpu().item()),
            "LossPi": float(loss_pi.detach().cpu().item()),
        }
        return loss_pi, logp_pi.detach(), info

    def update(self, data: Dict[str, torch.Tensor]) -> None:
        # Q update
        self.q_optimizer.zero_grad(set_to_none=True)
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(self.q_params, max_norm=10.0)
        self.q_optimizer.step()

        self.logger.store(LossQ=float(loss_q.detach().cpu().item()), **q_info)

        # Freeze Q for policy update
        for p in self.q_params:
            p.requires_grad = False

        # Policy update
        self.pi_optimizer.zero_grad(set_to_none=True)
        loss_pi, logpi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(self.ac.pi.parameters(), max_norm=10.0)
        self.pi_optimizer.step()

        # Temperature update
        if self.auto_alpha:
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss = -(self.alpha_value * (logpi + self.target_entropy)).mean()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.logger.store(Alpha=float(self.alpha_value.detach().cpu().item()), AlphaLoss=float(alpha_loss.detach().cpu().item()))

        self.logger.store(**pi_info)

        # Unfreeze Q
        for p in self.q_params:
            p.requires_grad = True

        # Polyak averaging for target networks
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    @torch.no_grad()
    def test_agent(self) -> Dict[str, float]:
        if self.test_env is None:
            return {}
        env = self.test_env
        ep_returns = []
        for _ in range(self.num_test_episodes):
            o = env.reset()
            d = False
            ep_ret = 0.0
            ep_len = 0
            while not d and ep_len < self.max_ep_len:
                a = self.get_action(o, deterministic=True)
                o, r, d, _ = env.step(a)
                ep_ret += float(r)
                ep_len += 1
            ep_returns.append(ep_ret)
        return {
            "TestEpRet": float(np.mean(ep_returns)) if len(ep_returns) else 0.0,
            "TestEpRetStd": float(np.std(ep_returns)) if len(ep_returns) else 0.0,
        }

    def run(self, plot_freq: int = 0):
        """Train SAC. Returns simple history arrays for convenience."""

        total_steps = self.steps_per_epoch * self.epochs
        o = self.env.reset()

        ep_ret = 0.0
        ep_len = 0

        train_ep_returns = []
        train_ep_lengths = []

        test_ep_returns = []
        test_ep_lengths = []

        for t in range(total_steps):
            # Action selection
            if t < self.start_steps:
                a = self.env.action_space.sample()
            else:
                a = self.get_action(o, deterministic=False)

            o2, r, d, info = self.env.step(a)
            ep_ret += float(r)
            ep_len += 1

            # Ignore time-limit termination when computing bootstraps
            done_for_buffer = bool(d) if ep_len < self.max_ep_len else False

            self.replay_buffer.store(o, a, r, o2, done_for_buffer)
            o = o2

            # End of trajectory
            if d or (ep_len >= self.max_ep_len):
                train_ep_returns.append(ep_ret)
                train_ep_lengths.append(ep_len)
                o = self.env.reset()
                ep_ret, ep_len = 0.0, 0

            # Update
            if t >= self.update_after and (t % self.update_every == 0):
                for _ in range(self.update_every):
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    self.update(batch)

            # End of epoch
            if (t + 1) % self.steps_per_epoch == 0:
                test_stats = self.test_agent()
                if test_stats:
                    test_ep_returns.append(test_stats.get("TestEpRet", 0.0))
                    test_ep_lengths.append(self.max_ep_len)

        return train_ep_returns, train_ep_lengths, test_ep_returns, test_ep_lengths
