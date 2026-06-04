#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ppo_buffer.py

Trajectory buffer for PPO with:
- tuple observation shapes (supports 2D obs: assets × features)
- MPI-compatible advantage normalization

Drop-in replacement for your current ppo_buffer.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from ppo_mpi import mpi_statistics_scalar


def combined_shape(length: int, shape: Union[int, Tuple[int, ...]] = ()) -> Tuple[int, ...]:
    if isinstance(shape, int):
        return (length, shape)
    return (length, *shape)


def discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    """Compute discounted cumulative sums of vectors."""
    x = np.asarray(x, dtype=np.float64)
    y = np.zeros_like(x, dtype=np.float64)
    running = 0.0
    for i in reversed(range(len(x))):
        running = x[i] + discount * running
        y[i] = running
    return y


@dataclass
class PPOBuffer:
    obs_dim: Union[int, Tuple[int, ...]]
    act_dim: int
    size: int
    gamma: float = 0.99
    lam: float = 0.97
    is_hyper_tune: bool = False
    device: Optional[torch.device] = None

    def __post_init__(self):
        self.max_size = int(self.size)
        self.ptr = 0
        self.path_start_idx = 0

        self.obs_buf = np.zeros(combined_shape(self.max_size, self.obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(self.max_size, self.act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(self.max_size, dtype=np.float32)
        self.rew_buf = np.zeros(self.max_size, dtype=np.float32)
        self.ret_buf = np.zeros(self.max_size, dtype=np.float32)
        self.val_buf = np.zeros(self.max_size, dtype=np.float32)
        self.logp_buf = np.zeros(self.max_size, dtype=np.float32)

        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = np.asarray(obs, dtype=np.float32)
        self.act_buf[self.ptr] = np.asarray(act, dtype=np.float32)
        self.rew_buf[self.ptr] = float(rew)
        self.val_buf[self.ptr] = float(val)
        self.logp_buf[self.ptr] = float(logp) if np.isscalar(logp) else float(np.asarray(logp).reshape(-1)[0])
        self.ptr += 1

    def finish_path(self, last_val: float = 0.0):
        """Call this at the end of a trajectory, or when an epoch ends."""
        path_slice = slice(self.path_start_idx, self.ptr)

        rews = np.append(self.rew_buf[path_slice], float(last_val))
        vals = np.append(self.val_buf[path_slice], float(last_val))

        # GAE-Lambda advantage
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam).astype(np.float32)

        # rewards-to-go
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1].astype(np.float32)

        self.path_start_idx = self.ptr

    def get(self) -> Dict[str, torch.Tensor]:
        assert self.ptr == self.max_size, f"Buffer not full: {self.ptr} / {self.max_size}"

        self.ptr = 0
        self.path_start_idx = 0

        # normalize advantages
        adv = self.adv_buf
        if not self.is_hyper_tune:
            adv_mean, adv_std = mpi_statistics_scalar(adv)
        else:
            adv_mean, adv_std = float(np.mean(adv)), float(np.std(adv) + 1e-8)
        self.adv_buf = (adv - adv_mean) / (adv_std + 1e-8)

        data = dict(
            obs=torch.as_tensor(self.obs_buf, dtype=torch.float32, device=self.device),
            act=torch.as_tensor(self.act_buf, dtype=torch.float32, device=self.device),
            ret=torch.as_tensor(self.ret_buf, dtype=torch.float32, device=self.device),
            adv=torch.as_tensor(self.adv_buf, dtype=torch.float32, device=self.device),
            logp=torch.as_tensor(self.logp_buf, dtype=torch.float32, device=self.device),
            val=torch.as_tensor(self.val_buf, dtype=torch.float32, device=self.device),
        )
        return data

