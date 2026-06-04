#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sac_core.py

Core neural-network modules for Soft Actor-Critic.

This is a drop-in replacement for your SAC stack and is compatible with
**2D observations** coming from the LONG (timestamp, tic) environment:

    obs.shape == (n_assets, n_features)

The policy / Q networks accept (batch, n_assets, n_features) and flatten
internally to (batch, n_assets*n_features) for MLP consumption.

It also supports classic flat observations.

Key fixes vs. your uploaded file:
- Works even if optional exploration modules are missing.
- Correct tanh-squash log-prob computation (and includes affine action scaling).
- Removes references to undefined globals (device, init_weights_*).
- Computes obs_dim as prod(observation_space.shape) (instead of shape[0]).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------

def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_default_device()


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

LOG_STD_MAX = 2.0
LOG_STD_MIN = -5.0


def count_vars(module: nn.Module) -> int:
    return int(sum(int(np.prod(p.shape)) for p in module.parameters()))


def _prod_shape(shape: Sequence[int]) -> int:
    out = 1
    for s in shape:
        out *= int(s)
    return int(out)


def _flatten_obs(obs: torch.Tensor) -> torch.Tensor:
    """Flatten observations to (batch, dim).

    Accepts:
      - (batch, dim)
      - (batch, A, F)
    """
    if obs.ndim <= 2:
        return obs
    return obs.reshape(obs.shape[0], -1)


# -----------------------------------------------------------------------------
# Weight init (safe defaults)
# -----------------------------------------------------------------------------

def init_weights_relu(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def init_weights_elu(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def init_weights_tanh(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("tanh"))
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# -----------------------------------------------------------------------------
# MLP builders
# -----------------------------------------------------------------------------

def mlp(
    sizes: Sequence[int],
    activation: type[nn.Module] = nn.ReLU,
    output_activation: type[nn.Module] = nn.Identity,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    for j in range(len(sizes) - 1):
        in_dim, out_dim = int(sizes[j]), int(sizes[j + 1])
        layers.append(nn.Linear(in_dim, out_dim))
        act = activation if j < len(sizes) - 2 else output_activation
        layers.append(act())
    return nn.Sequential(*layers)


# -----------------------------------------------------------------------------
# Optional exploration modules referenced in your original file
# -----------------------------------------------------------------------------

try:  # pragma: no cover
    from modules_models_exploration import (  # type: ignore
        NoisyLazyLinear,
        NoisyLinear,
        LazygSDEModule,
        reset_noise,
        gSDEModule,
    )

    _EXPLORATION_OK = True
except Exception:  # pragma: no cover
    NoisyLazyLinear = None  # type: ignore
    NoisyLinear = None  # type: ignore
    LazygSDEModule = None  # type: ignore
    reset_noise = None  # type: ignore
    gSDEModule = None  # type: ignore
    _EXPLORATION_OK = False


# -----------------------------------------------------------------------------
# Actor
# -----------------------------------------------------------------------------

class SquashedGaussianMLPActor(nn.Module):
    """Gaussian policy with tanh squashing and affine action rescaling."""

    def __init__(
        self,
        obs_dim_flat: int,
        action_space,
        hidden_sizes: Sequence[int],
        activation: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.obs_dim_flat = int(obs_dim_flat)
        self.act_dim = int(action_space.shape[0])
        self.activation = activation

        hs = [int(x) for x in hidden_sizes]
        # Keep the spirit of your deeper net by doubling the stack.
        if len(hs) > 0:
            hs = hs + hs

        # backbone
        if len(hs) == 0:
            self.net = nn.Identity()
            last_dim = self.obs_dim_flat
        else:
            self.net = mlp([self.obs_dim_flat] + hs, activation=activation, output_activation=activation)
            last_dim = hs[-1]

        self.mu_layer = nn.Linear(last_dim, self.act_dim)
        self.log_std_layer = nn.Linear(last_dim, self.act_dim)

        # Action rescaling buffers (affine transform to env bounds)
        high = np.asarray(action_space.high, dtype=np.float32)
        low = np.asarray(action_space.low, dtype=np.float32)
        scale = (high - low) / 2.0
        bias = (high + low) / 2.0
        self.register_buffer("action_scale", torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(bias, dtype=torch.float32))

        # init
        self.net.apply(init_weights_relu if activation is nn.ReLU else init_weights_elu)
        self.mu_layer.apply(init_weights_tanh)
        self.log_std_layer.apply(init_weights_tanh)

    def _std_from_raw(self, raw: torch.Tensor) -> torch.Tensor:
        """raw -> std, with log_std constrained to [LOG_STD_MIN, LOG_STD_MAX]."""
        log_std = torch.tanh(raw)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0)
        return torch.exp(log_std)

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        with_logprob: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # obs: (batch, dim) or (batch, A, F)
        obs = _flatten_obs(obs)

        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        std = self._std_from_raw(self.log_std_layer(net_out))
        dist = Normal(mu, std)

        if deterministic:
            pre_tanh = mu
        else:
            pre_tanh = dist.rsample()

        tanh_action = torch.tanh(pre_tanh)
        action = tanh_action * self.action_scale + self.action_bias

        logp: Optional[torch.Tensor]
        if with_logprob:
            # Gaussian log-prob
            logp = dist.log_prob(pre_tanh).sum(dim=-1)

            # Tanh correction (stable equivalent)
            correction = (2.0 * (np.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))).sum(dim=-1)
            logp = logp - correction

            # Affine scaling correction (constant shift)
            scale = torch.clamp(self.action_scale, min=1e-12)
            logp = logp - torch.log(scale).sum()
        else:
            logp = None

        return action, logp


# -----------------------------------------------------------------------------
# Critic
# -----------------------------------------------------------------------------

class MLPQFunction(nn.Module):
    def __init__(
        self,
        obs_dim_flat: int,
        act_dim: int,
        hidden_sizes: Sequence[int],
        activation: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.obs_dim_flat = int(obs_dim_flat)
        self.act_dim = int(act_dim)

        hs = [int(x) for x in hidden_sizes]
        if len(hs) > 0:
            hs = hs + hs

        self.q = mlp([self.obs_dim_flat + self.act_dim] + hs + [1], activation=activation, output_activation=nn.Identity)
        self.q.apply(init_weights_relu if activation is nn.ReLU else init_weights_elu)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        obs = _flatten_obs(obs)
        x = torch.cat([obs, act], dim=-1)
        q = self.q(x)
        return torch.squeeze(q, -1)


# -----------------------------------------------------------------------------
# Actor-Critic
# -----------------------------------------------------------------------------

class MLPActorCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()

        obs_shape = tuple(int(x) for x in observation_space.shape)
        obs_dim_flat = _prod_shape(obs_shape)
        act_dim = int(action_space.shape[0])

        self.obs_shape = obs_shape
        self.obs_dim_flat = obs_dim_flat
        self.act_dim = act_dim

        self.pi = SquashedGaussianMLPActor(
            obs_dim_flat=obs_dim_flat,
            action_space=action_space,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )
        self.q1 = MLPQFunction(obs_dim_flat, act_dim, hidden_sizes, activation=activation)
        self.q2 = MLPQFunction(obs_dim_flat, act_dim, hidden_sizes, activation=activation)

        self.to(DEVICE)

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False) -> np.ndarray:
        """obs can be numpy or torch. Supports 2D obs (A,F) by adding batch dim."""
        if torch.is_tensor(obs):
            obs_t = obs.to(device=DEVICE, dtype=torch.float32)
        else:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)

        if tuple(obs_t.shape) == self.obs_shape:
            obs_t = obs_t.unsqueeze(0)

        a, _ = self.pi(obs_t, deterministic=deterministic, with_logprob=False)
        a = a.squeeze(0)
        return a.cpu().numpy()
