#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ppo_core_.py

Clean, production-grade PPO actor/critic networks.

Drop-in goals for your LONG env:
- Env can emit observations as **2D matrices**: (n_assets, n_features) per timestep
  (or batched: (batch, n_assets, n_features)).
- PPO training can also feed batched tensors from the replay/trajectory buffer.
- We therefore do **shape-safe** parsing:
    * Flat obs: (dim,), (batch, dim)
    * Matrix obs: (A, F), (batch, A, F)
    * Flattened matrix obs: (A*F,), (batch, A*F)

This revision adds a Set-Transformer-style Actor-Critic for matrix observations:
- Per-asset encoder + cross-asset self-attention
- Actor outputs per-asset actions (dim = n_assets)
- Critic pools across assets to produce V(s)

It auto-selects the SetTransformer backend when:
- observation_space.shape is 2D (A, F), AND
- action_space is Box with act_dim == A

Otherwise it falls back to the original MLP actor/critic (with correct flattening).

Compatible with:
- Box continuous actions (your env uses Box)
- MPI training (no MPI code here, but works with your ppo_mpi helpers)

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


# -----------------------------------------------------------------------------
# Minimal Gym spaces fallback
# -----------------------------------------------------------------------------
try:  # gymnasium
    from gymnasium.spaces import Box, Discrete
except Exception:  # gym
    try:
        from gym.spaces import Box, Discrete
    except Exception:
        # fallback minimal types (duck-typed)
        class Box:  # type: ignore
            def __init__(self, low, high, shape, dtype=np.float32):
                self.low = np.full(shape, low, dtype=dtype)
                self.high = np.full(shape, high, dtype=dtype)
                self.shape = tuple(shape)
                self.dtype = dtype

        class Discrete:  # type: ignore
            def __init__(self, n: int):
                self.n = int(n)


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
EPS = 1e-6


def combined_shape(length: int, shape: Union[int, Tuple[int, ...]] = ()) -> Tuple[int, ...]:
    if isinstance(shape, int):
        return (length, shape)
    return (length, *shape)


def count_vars(module: nn.Module) -> int:
    return int(sum(np.prod(p.shape) for p in module.parameters()))


def _prod_shape(shape: Sequence[int]) -> int:
    out = 1
    for s in shape:
        out *= int(s)
    return int(out)


def _to_tensor(x, *, device: torch.device = DEVICE, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, dtype=dtype, device=device)


def _ensure_flat_batch(obs_t: torch.Tensor, obs_dim_flat: int) -> torch.Tensor:
    """
    Return obs as (B, obs_dim_flat), accepting:
      - (obs_dim_flat,)
      - (B, obs_dim_flat)
      - (A, F) or (B, A, F) which will be flattened
    """
    if obs_t.ndim == 1:
        if obs_t.numel() != obs_dim_flat:
            raise ValueError(f"Flat obs expected size {obs_dim_flat}, got {obs_t.numel()}.")
        return obs_t.view(1, obs_dim_flat)

    if obs_t.ndim == 2:
        # Could be (B, obs_dim_flat) OR a single (A, F) matrix.
        # If second dim matches obs_dim_flat, treat as (B, obs_dim_flat).
        if obs_t.shape[1] == obs_dim_flat:
            return obs_t
        # Otherwise treat as a single matrix obs and flatten.
        if obs_t.numel() != obs_dim_flat:
            raise ValueError(f"Obs total size {obs_t.numel()} != expected flat dim {obs_dim_flat}.")
        return obs_t.reshape(1, obs_dim_flat)

    # ndim >= 3: assume (B, ...) -> flatten last dims
    B = obs_t.shape[0]
    if obs_t.numel() != B * obs_dim_flat:
        raise ValueError(
            f"Obs total size {obs_t.numel()} != B*obs_dim_flat {B*obs_dim_flat} "
            f"(B={B}, obs_dim_flat={obs_dim_flat})."
        )
    return obs_t.reshape(B, obs_dim_flat)


def _ensure_matrix_batch(obs_t: torch.Tensor, n_assets: int, n_features: int) -> torch.Tensor:
    """
    Return obs as (B, A, F), accepting:
      - (A, F)
      - (B, A, F)
      - flattened: (A*F,) or (B, A*F)
    """
    A, F = int(n_assets), int(n_features)
    AF = A * F

    if obs_t.ndim == 2:
        # Either (A,F) or (B, AF) or (B, A, F) not possible at ndim=2
        if obs_t.shape == (A, F):
            return obs_t.unsqueeze(0)
        if obs_t.shape[1] == AF:
            # (B, AF) -> (B, A, F)
            return obs_t.reshape(obs_t.shape[0], A, F)
        # maybe (batch, A) for some reason -> not supported
        raise ValueError(f"Expected (A,F)=({A},{F}) or (B, A*F)=(*,{AF}), got {tuple(obs_t.shape)}.")

    if obs_t.ndim == 3:
        if obs_t.shape[1:] != (A, F):
            raise ValueError(f"Expected (B,A,F)=(*,{A},{F}), got {tuple(obs_t.shape)}.")
        return obs_t

    if obs_t.ndim == 1:
        if obs_t.numel() != AF:
            raise ValueError(f"Flattened matrix obs expected size {AF}, got {obs_t.numel()}.")
        return obs_t.view(1, A, F)

    # ndim >=4: flatten all but batch then reshape
    B = obs_t.shape[0]
    if obs_t.numel() != B * AF:
        raise ValueError(f"Obs total size {obs_t.numel()} != B*AF {B*AF}.")
    return obs_t.reshape(B, A, F)


# -----------------------------------------------------------------------------
# Squashed Gaussian distribution
# -----------------------------------------------------------------------------
def atanh(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, -1.0 + EPS, 1.0 - EPS)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


@dataclass
class SquashedGaussian:
    """A tanh-squashed Normal with optional affine scaling to action bounds."""
    mean: torch.Tensor
    std: torch.Tensor
    action_scale: torch.Tensor
    action_bias: torch.Tensor

    def __post_init__(self):
        self.normal = Normal(self.mean, self.std)

    @property
    def mode(self) -> torch.Tensor:
        y = torch.tanh(self.mean)
        return y * self.action_scale + self.action_bias

    def sample(self) -> torch.Tensor:
        # IMPORTANT: no sample_shape passed here. Returns same shape as mean/std.
        z = self.normal.sample()
        y = torch.tanh(z)
        return y * self.action_scale + self.action_bias

    def rsample(self) -> torch.Tensor:
        z = self.normal.rsample()
        y = torch.tanh(z)
        return y * self.action_scale + self.action_bias

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        # action: env-scale
        y = (action - self.action_bias) / (self.action_scale + EPS)
        y = torch.clamp(y, -1.0 + EPS, 1.0 - EPS)
        z = atanh(y)
        logp = self.normal.log_prob(z)
        log_det = torch.log(self.action_scale + EPS) + torch.log(1.0 - y * y + EPS)
        logp = logp - log_det
        return logp.sum(axis=-1)

    def entropy(self) -> torch.Tensor:
        # Exact entropy under tanh is not closed form. Use base normal entropy proxy.
        return self.normal.entropy().sum(axis=-1)


# -----------------------------------------------------------------------------
# Network builders / init
# -----------------------------------------------------------------------------
def mlp(
    sizes: Sequence[int],
    activation: type[nn.Module],
    output_activation: type[nn.Module] = nn.Identity,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    for j in range(len(sizes) - 1):
        out_dim = int(sizes[j + 1])
        layers.append(nn.LazyLinear(out_dim, bias=True))
        act = activation if j < len(sizes) - 2 else output_activation
        layers.append(act())
    return nn.Sequential(*layers)


def orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# -----------------------------------------------------------------------------
# Actor / Critic interfaces
# -----------------------------------------------------------------------------
class Actor(nn.Module):
    def _distribution(self, obs: Union[np.ndarray, torch.Tensor], deterministic: bool = False):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        act: Optional[torch.Tensor] = None,
        determinstic: bool = False,
        debug: bool = False,
    ):
        pi = self._distribution(obs, deterministic=determinstic)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
            if debug:
                print(f"||| logp_a.shape: {tuple(logp_a.shape)} ||| act.shape: {tuple(act.shape)}")
        return pi, logp_a


# -----------------------------------------------------------------------------
# Categorical Actor
# -----------------------------------------------------------------------------
class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim_flat: int, act_dim: int, hidden_sizes: Sequence[int], activation: type[nn.Module]):
        super().__init__()
        self.obs_dim_flat = int(obs_dim_flat)
        self.logits_net = mlp([self.obs_dim_flat] + list(hidden_sizes) + [act_dim], activation)
        self.logits_net.apply(lambda m: orthogonal_init(m, gain=math.sqrt(2)))

    def _distribution(self, obs, deterministic: bool = False):
        obs_t = _to_tensor(obs)
        x = _ensure_flat_batch(obs_t, self.obs_dim_flat)
        logits = self.logits_net(x)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi: Categorical, act: torch.Tensor) -> torch.Tensor:
        return pi.log_prob(act)


# -----------------------------------------------------------------------------
# Squashed Gaussian Actor (MLP over flattened obs)
# -----------------------------------------------------------------------------
class MLPSquashedGaussianActor(Actor):
    def __init__(
        self,
        obs_dim_flat: int,
        act_dim: int,
        action_space: Box,
        hidden_sizes: Sequence[int],
        activation: type[nn.Module],
        log_std_init: float = -0.5,
    ):
        super().__init__()
        self.obs_dim_flat = int(obs_dim_flat)
        self.act_dim = int(act_dim)

        # backbone
        self.net = mlp([self.obs_dim_flat] + list(hidden_sizes), activation, output_activation=activation)
        last_dim = int(hidden_sizes[-1]) if len(hidden_sizes) else self.obs_dim_flat

        self.mu_layer = nn.Linear(last_dim, self.act_dim)
        self.log_std = nn.Parameter(torch.ones(self.act_dim, dtype=torch.float32) * float(log_std_init))

        high = np.asarray(action_space.high, dtype=np.float32)
        low = np.asarray(action_space.low, dtype=np.float32)
        scale = (high - low) / 2.0
        bias = (high + low) / 2.0
        self.register_buffer("action_scale", torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(bias, dtype=torch.float32))

        self.to(DEVICE)

    def _distribution(self, obs, deterministic: bool = False) -> SquashedGaussian:
        obs_t = _to_tensor(obs)
        x = _ensure_flat_batch(obs_t, self.obs_dim_flat)   # (B, dim)
        h = self.net(x)
        mu = self.mu_layer(h)
        std = torch.exp(torch.clamp(self.log_std, -20.0, 2.0)).expand_as(mu)
        return SquashedGaussian(mu, std, self.action_scale, self.action_bias)

    def _log_prob_from_distribution(self, pi: SquashedGaussian, act: torch.Tensor) -> torch.Tensor:
        return pi.log_prob(act)


# -----------------------------------------------------------------------------
# MLP Critic (flattened obs)
# -----------------------------------------------------------------------------
class MLPCritic(nn.Module):
    def __init__(self, obs_dim_flat: int, hidden_sizes: Sequence[int], activation: type[nn.Module]):
        super().__init__()
        self.obs_dim_flat = int(obs_dim_flat)
        self.v_net = mlp([self.obs_dim_flat] + list(hidden_sizes) + [1], activation)
        self.to(DEVICE)

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        obs_t = _to_tensor(obs)
        x = _ensure_flat_batch(obs_t, self.obs_dim_flat)
        v = self.v_net(x)
        return torch.squeeze(v, -1)


# -----------------------------------------------------------------------------
# Set-Transformer blocks (cross-asset attention)
# -----------------------------------------------------------------------------
class _AssetEncoder(nn.Module):
    """Per-asset MLP encoder: (F) -> (D) applied over last dim."""
    def __init__(self, n_features: int, d_model: int, hidden: int = 256, activation: type[nn.Module] = nn.Tanh, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            activation(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            activation(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, A, F) -> apply over last dim
        return self.net(x)


class _CrossAssetBlock(nn.Module):
    """Self-attention across assets. Input/Output: (B, A, D)."""
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # key_padding_mask: (B, A) True means "ignore"
        h, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.ln1(x + self.drop(h))
        x = self.ln2(x + self.drop(self.ff(x)))
        return x


class SetTransformerSquashedGaussianActor(Actor):
    """
    Actor for matrix obs (A,F):
      - Encode each asset -> (B,A,D)
      - Cross-asset self-attention -> (B,A,D)
      - Per-asset head -> mu (B,A) -> SquashedGaussian with act_dim=A
    """
    def __init__(
        self,
        n_assets: int,
        n_features: int,
        action_space: Box,
        d_model: int = 128,
        n_blocks: int = 2,
        n_heads: int = 4,
        dropout: float = 0.0,
        use_asset_id_embedding: bool = True,
        log_std_init: float = -0.5,
    ):
        super().__init__()
        self.n_assets = int(n_assets)
        self.n_features = int(n_features)
        self.act_dim = int(n_assets)

        self.asset_encoder = _AssetEncoder(
            n_features=self.n_features, d_model=d_model, hidden=256, activation=nn.ReLU, # nn.Tanh,
            dropout=dropout
        )
        self.asset_id = nn.Embedding(self.n_assets, d_model) if use_asset_id_embedding else None
        self.blocks = nn.ModuleList(
            [
                _CrossAssetBlock(d_model=d_model, n_heads=n_heads, dropout=dropout) for _ in range(n_blocks)
            ]
        )

        self.mu_head = nn.Linear(d_model, 1)
        self.log_std = nn.Parameter(torch.as_tensor(torch.ones(self.act_dim, dtype=torch.float32) * float(log_std_init))) # Original.
        # self.log_std = nn.Linear(d_model, 1)

        high = np.asarray(action_space.high, dtype=np.float32)
        low = np.asarray(action_space.low, dtype=np.float32)
        scale = (high - low) / 2.0
        bias = (high + low) / 2.0
        self.register_buffer("action_scale", torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(bias, dtype=torch.float32))

        self.to(DEVICE)

    def _encode(self, obs_t: torch.Tensor) -> torch.Tensor:
        x = _ensure_matrix_batch(obs_t, self.n_assets, self.n_features)  # (B,A,F)
        h = self.asset_encoder(x)  # (B,A,D)
        if self.asset_id is not None:
            B = h.shape[0]
            ids = torch.arange(self.n_assets, device=h.device).unsqueeze(0).expand(B, -1)
            h = h + self.asset_id(ids)
        for blk in self.blocks:
            h = blk(h)
        return h  # (B,A,D)

    def _distribution(self, obs, deterministic: bool = False) -> SquashedGaussian:
        obs_t = _to_tensor(obs)
        h = self._encode(obs_t)  # (B,A,D)
        mu = self.mu_head(h).squeeze(-1)  # (B,A)
        # std = torch.exp(torch.clamp(self.log_std, -5.0, 2.0)).view(1, -1).expand_as(mu) # Original.
        std = self.constrain_logsigma().view(1, -1).expand_as(mu)
        # log_sigma = self.log_std(h).squeeze(-1)
        # std = self.constrain_logsigma(log_sigma=log_sigma)
        return SquashedGaussian(mu, std, self.action_scale, self.action_bias) # Original.
        # return torch.distributions.Normal(loc=mu, scale=std)

    def _log_prob_from_distribution(self, pi: SquashedGaussian, act: torch.Tensor) -> torch.Tensor:
        return pi.log_prob(act) # Original.
        # return pi.log_prob(act).sum(-1)

    def constrain_logsigma(self): # , log_sigma):
        # logvar = self.log_std.to(device, non_blocking=True)
        # self.log_std_bounds = [-20, 2]
        # constrain log_std inside [log_std_min, log_std_max]
        log_sigma = torch.tanh(self.log_std) # log_sigma)
        # log_std_min, log_std_max = self.log_std_bounds
        log_std_min = -5.
        log_std_max = 2.
        # print(f'log_std_min: {log_std_min} ||| log_std_max: {log_std_max} ||| std_min: {np.exp(log_std_min)} ||| std_max: {np.exp(log_std_max)}')
        log_std = log_std_min + 0.5 * (
            log_std_max - log_std_min
            ) * (
                log_sigma + 1
                # log_sigma.pow(2)
                )

        # sigma = log_sigma.exp()
        # return sigma
        return log_std.exp()



class SetTransformerCritic(nn.Module):
    """Critic for matrix obs (A,F): pooled asset embeddings -> scalar value."""
    def __init__(
        self,
        n_assets: int,
        n_features: int,
        d_model: int = 128,
        n_blocks: int = 2,
        n_heads: int = 4,
        dropout: float = 0.0,
        use_asset_id_embedding: bool = True,
    ):
        super().__init__()
        self.n_assets = int(n_assets)
        self.n_features = int(n_features)

        self.asset_encoder = _AssetEncoder(
            n_features=self.n_features, d_model=d_model, hidden=256, activation=nn.ReLU, # nn.Tanh,
            dropout=dropout
        )
        self.asset_id = nn.Embedding(self.n_assets, d_model) if use_asset_id_embedding else None
        self.blocks = nn.ModuleList(
            [
                _CrossAssetBlock(d_model=d_model, n_heads=n_heads, dropout=dropout) for _ in range(n_blocks)
            ]
        )

        self.v_head = nn.Sequential(
            nn.Linear(d_model, 256),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.to(DEVICE)

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        obs_t = _to_tensor(obs)
        x = _ensure_matrix_batch(obs_t, self.n_assets, self.n_features)  # (B,A,F)
        h = self.asset_encoder(x)  # (B,A,D)

        if self.asset_id is not None:
            B = h.shape[0]
            ids = torch.arange(self.n_assets, device=h.device).unsqueeze(0).expand(B, -1)
            h = h + self.asset_id(ids)

        for blk in self.blocks:
            h = blk(h)

        pooled = h.mean(dim=1)  # (B,D)
        v = self.v_head(pooled)  # (B,1)
        return torch.squeeze(v, -1)


# -----------------------------------------------------------------------------
# Actor-Critic wrapper (drop-in name: MLPActorCritic)
# -----------------------------------------------------------------------------
class MLPActorCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: type[nn.Module] = nn.ReLU, # nn.Tanh,
        # SetTransformer knobs (safe defaults)
        use_set_transformer: Optional[bool] = None,
        st_d_model: int = 128,
        st_n_blocks: int = 2,
        st_n_heads: int = 4,
        st_dropout: float = 0.0,
        st_use_asset_id_embedding: bool = True,
        **kwargs,
    ):
        super().__init__()

        obs_shape = tuple(getattr(observation_space, "shape", ()))
        if len(obs_shape) == 0:
            raise ValueError("observation_space.shape is required")

        # Determine action type
        is_box = hasattr(action_space, "shape") and hasattr(action_space, "low") and hasattr(action_space, "high")
        is_discrete = hasattr(action_space, "n") and not (hasattr(action_space, "low") and hasattr(action_space, "high"))

        if is_box:
            act_dim = int(action_space.shape[0])
            # Decide backend
            if use_set_transformer is None:
                use_set_transformer = (len(obs_shape) == 2 and int(obs_shape[0]) == act_dim)

            if use_set_transformer:
                if len(obs_shape) != 2:
                    raise ValueError(
                        f"SetTransformer requires 2D obs shape (A,F). Got obs_shape={obs_shape}."
                    )
                n_assets, n_features = int(obs_shape[0]), int(obs_shape[1])
                if act_dim != n_assets:
                    raise ValueError(
                        f"SetTransformer expects act_dim == n_assets. Got act_dim={act_dim}, n_assets={n_assets}."
                    )

                self.pi: Actor = SetTransformerSquashedGaussianActor(
                    n_assets=n_assets,
                    n_features=n_features,
                    action_space=action_space,
                    d_model=st_d_model,
                    n_blocks=st_n_blocks,
                    n_heads=st_n_heads,
                    dropout=st_dropout,
                    use_asset_id_embedding=st_use_asset_id_embedding,
                )
                self.v = SetTransformerCritic(
                    n_assets=n_assets,
                    n_features=n_features,
                    d_model=st_d_model,
                    n_blocks=int(st_n_blocks - 1),
                    n_heads=st_n_heads,
                    dropout=st_dropout,
                    use_asset_id_embedding=st_use_asset_id_embedding,
                )
                self._obs_dim_flat = n_assets * n_features
                self._matrix_obs = True
            else:
                obs_dim_flat = _prod_shape(obs_shape)
                self.pi = MLPSquashedGaussianActor(
                    obs_dim_flat=obs_dim_flat,
                    act_dim=act_dim,
                    action_space=action_space,
                    hidden_sizes=tuple(hidden_sizes),
                    activation=activation,
                )
                self.v = MLPCritic(obs_dim_flat, hidden_sizes, activation)
                self._obs_dim_flat = obs_dim_flat
                self._matrix_obs = False

        elif is_discrete:
            obs_dim_flat = _prod_shape(obs_shape)
            act_dim = int(action_space.n)
            self.pi = MLPCategoricalActor(obs_dim_flat, act_dim, hidden_sizes, activation)
            self.v = MLPCritic(obs_dim_flat, hidden_sizes, activation)
            self._obs_dim_flat = obs_dim_flat
            self._matrix_obs = False
        else:
            raise TypeError("Unsupported action_space type")

        self.to(DEVICE)
        self._dummy_forward(obs_shape)
        self._init_weights()

    @torch.no_grad()
    def _dummy_forward(self, obs_shape: Tuple[int, ...]) -> None:
        # Materialize LazyLinear weights (and generally validate shapes).
        if len(obs_shape) == 1:
            dummy = torch.randn(4, int(obs_shape[0]), device=DEVICE)
        elif len(obs_shape) == 2:
            dummy = torch.randn(4, int(obs_shape[0]), int(obs_shape[1]), device=DEVICE)
        else:
            # Flatten higher dims for safety
            dummy = torch.randn(4, _prod_shape(obs_shape), device=DEVICE)

        _ = self.pi._distribution(dummy)
        _ = self.v(dummy)

    def _init_weights(self) -> None:
        # Conservative orthogonal init for Linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                orthogonal_init(m, gain=math.sqrt(2))

        # Policy output head: small init
        if hasattr(self.pi, "mu_layer") and isinstance(self.pi.mu_layer, nn.Linear):
            orthogonal_init(self.pi.mu_layer, gain=0.01)
        if hasattr(self.pi, "mu_head") and isinstance(self.pi.mu_head, nn.Linear):
            orthogonal_init(self.pi.mu_head, gain=0.01)

        # Critic last layer: keep gain=1.0
        # Try to locate final Linear in MLP critic or SetTransformer critic head
        if hasattr(self.v, "v_net") and isinstance(self.v.v_net, nn.Sequential):
            # last Linear in v_net
            for mod in reversed(list(self.v.v_net.modules())):
                if isinstance(mod, nn.Linear):
                    orthogonal_init(mod, gain=1.0)
                    break
        if hasattr(self.v, "v_head") and isinstance(self.v.v_head, nn.Sequential):
            for mod in reversed(list(self.v.v_head.modules())):
                if isinstance(mod, nn.Linear):
                    orthogonal_init(mod, gain=1.0)
                    break

    @torch.no_grad()
    def step(self, obs: Union[np.ndarray, torch.Tensor], determinstic: bool = False):
        """
        Returns SpinningUp-style outputs for a *single* env step:
          a: (act_dim,)  numpy
          v: scalar numpy
          logp: scalar numpy
        Also supports batched obs (then returns batched arrays).
        """
        pi = self.pi._distribution(obs, deterministic=determinstic)

        a = pi.mode if determinstic else pi.sample()   # shape matches pi.mean
        logp_a = self.pi._log_prob_from_distribution(pi, a)
        v = self.v(obs)

        # If single obs -> squeeze batch dim
        if isinstance(a, torch.Tensor) and a.ndim == 2 and a.shape[0] == 1:
            a = a[0]
        if isinstance(v, torch.Tensor) and v.ndim == 1 and v.shape[0] == 1:
            v = v[0]
        if isinstance(logp_a, torch.Tensor) and logp_a.ndim == 1 and logp_a.shape[0] == 1:
            logp_a = logp_a[0]

        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    @torch.no_grad()
    def act(self, obs: Union[np.ndarray, torch.Tensor], determinstic: bool = False):
        return self.step(obs, determinstic=determinstic)[0]
