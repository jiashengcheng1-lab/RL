import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# -----------------------------
# Utilities
# -----------------------------
def _init_layer(layer: nn.Linear, gain: float = 1.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0.0)
    return layer


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Tuple[int, ...],
        out_dim: int,
        activation: nn.Module = nn.Tanh,
        layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        dims = (in_dim,) + hidden_dims + (out_dim,)
        layers = []
        for i in range(len(dims) - 1):
            lin = _init_layer(nn.Linear(dims[i], dims[i + 1]), gain=nn.init.calculate_gain("tanh"))
            layers.append(lin)
            if i < len(dims) - 2:
                if layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(activation())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAssetAttentionBlock(nn.Module):
    """
    Self-attention across assets.
    Input:  x: (B, A, D)
    Output: x: (B, A, D)
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            _init_layer(nn.Linear(d_model, 4 * d_model), gain=nn.init.calculate_gain("relu")),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            _init_layer(nn.Linear(4 * d_model, d_model), gain=1.0),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # key_padding_mask: (B, A) True means "ignore"
        h, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.ln1(x + self.drop(h))
        x = self.ln2(x + self.drop(self.ff(x)))
        return x


# -----------------------------
# Squashed Gaussian distribution helpers
# -----------------------------
LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0

def atanh(x: torch.Tensor) -> torch.Tensor:
    # numerically stable inverse tanh
    eps = 1e-6
    x = torch.clamp(x, -1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def squash_action_and_logp(a_pre: torch.Tensor, logp_pre: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply tanh squashing and correct log-prob (change-of-variables).
    a = tanh(a_pre)
    logp = logp_pre - sum(log(1 - tanh(a_pre)^2))
    """
    a = torch.tanh(a_pre)
    # correction term
    correction = torch.log(torch.clamp(1.0 - a * a, min=1e-6))
    logp = logp_pre - correction.sum(dim=-1)
    return a, logp


# -----------------------------
# Set/Asset Transformer Actor-Critic
# -----------------------------
class SetTransformerSquashedGaussianActorCritic(nn.Module):
    """
    Obs per step is a matrix: (A, F) or batched (B, A, F)

    Policy: per-asset mu/log_std -> diagonal Gaussian -> tanh squash -> action in [-1,1]^A
    Critic: pooled asset embedding -> scalar value
    """
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
        self.d_model = int(d_model)

        # Per-asset feature encoder: (F) -> (D)
        self.asset_encoder = MLP(
            in_dim=n_features,
            hidden_dims=(256, 256),
            out_dim=d_model,
            activation=nn.Tanh,
            layer_norm=True,
            dropout=dropout,
        )

        self.asset_id = nn.Embedding(n_assets, d_model) if use_asset_id_embedding else None

        self.blocks = nn.ModuleList(
            [CrossAssetAttentionBlock(d_model=d_model, n_heads=n_heads, dropout=dropout) for _ in range(n_blocks)]
        )

        # Actor heads produce per-asset params from per-asset embeddings
        self.mu_head = _init_layer(nn.Linear(d_model, 1), gain=0.01)
        self.log_std_head = _init_layer(nn.Linear(d_model, 1), gain=0.01)

        # Critic: pool across assets then value head
        self.v_head = nn.Sequential(
            _init_layer(nn.Linear(d_model, 256), gain=nn.init.calculate_gain("tanh")),
            nn.Tanh(),
            _init_layer(nn.Linear(256, 1), gain=1.0),
        )

    def _ensure_batched(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)  # (1, A, F)
        if obs.dim() != 3:
            raise ValueError(f"Expected obs dim 2 or 3, got {obs.shape}")
        return obs

    def encode(self, obs: torch.Tensor, asset_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        obs: (B, A, F)
        asset_mask: (B, A) where True means "asset is present", False means missing.
        MultiheadAttention uses key_padding_mask where True means "ignore".
        """
        obs = self._ensure_batched(obs)
        B, A, Fdim = obs.shape
        if A != self.n_assets or Fdim != self.n_features:
            raise ValueError(f"Obs shape mismatch. Expected (B,{self.n_assets},{self.n_features}), got {obs.shape}")

        # Encode each asset independently
        x = self.asset_encoder(obs)  # (B, A, D)

        if self.asset_id is not None:
            ids = torch.arange(self.n_assets, device=obs.device).unsqueeze(0).expand(B, -1)  # (B, A)
            x = x + self.asset_id(ids)

        key_padding_mask = None
        if asset_mask is not None:
            if asset_mask.shape != (B, A):
                raise ValueError(f"asset_mask must be (B,A) = {(B,A)}, got {asset_mask.shape}")
            key_padding_mask = ~asset_mask.bool()  # True means ignore

        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        return x  # (B, A, D)

    def pi(self, obs: torch.Tensor, asset_mask: Optional[torch.Tensor] = None) -> Tuple[Normal, torch.Tensor, torch.Tensor]:
        """
        Returns:
          dist: Normal with batch shape (B, A)
          mu:   (B, A)
          log_std: (B, A)
        """
        x = self.encode(obs, asset_mask=asset_mask)  # (B, A, D)
        mu = self.mu_head(x).squeeze(-1)            # (B, A)
        log_std = self.log_std_head(x).squeeze(-1)  # (B, A)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        return dist, mu, log_std

    def v(self, obs: torch.Tensor, asset_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.encode(obs, asset_mask=asset_mask)  # (B, A, D)

        # Masked mean pooling across assets if mask provided
        if asset_mask is None:
            pooled = x.mean(dim=1)  # (B, D)
        else:
            m = asset_mask.float()  # (B, A)
            denom = torch.clamp(m.sum(dim=1, keepdim=True), min=1.0)
            pooled = (x * m.unsqueeze(-1)).sum(dim=1) / denom  # (B, D)

        return self.v_head(pooled).squeeze(-1)  # (B,)

    def step(self, obs: torch.Tensor, asset_mask: Optional[torch.Tensor] = None, deterministic: bool = False):
        """
        For PPO-style interaction:
          returns action in [-1,1]^A (shape (A,)), value scalar, and logp scalar
        """
        obs_b = self._ensure_batched(obs)

        dist, mu, _ = self.pi(obs_b, asset_mask=asset_mask)
        if deterministic:
            a_pre = mu
        else:
            # ✅ IMPORTANT: do NOT pass sample_shape=(n_assets,) here
            a_pre = dist.rsample()

        logp_pre = dist.log_prob(a_pre).sum(dim=-1)  # (B,)
        a, logp = squash_action_and_logp(a_pre, logp_pre)  # (B,A), (B,)

        val = self.v(obs_b, asset_mask=asset_mask)  # (B,)

        # return single-env shapes
        return a[0], val[0], logp[0]
