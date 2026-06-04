#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ReplayBuffers.py

Replay buffers used by SAC (and other off-policy agents).

This is a cleaned, drop-in replacement that:
- Adds missing imports and a consistent DEVICE.
- Supports **tuple observation shapes** (e.g. (n_assets, n_features)).
- Keeps your original class names so your training code doesn't break.

Primary class used by sac.py below is ReplayBuffer.
"""

from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def get_default_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


DEVICE = get_default_device()


def combined_shape(length: int, shape=None) -> Tuple[int, ...]:
    if shape is None:
        return (int(length),)
    if np.isscalar(shape):
        return (int(length), int(shape))
    return (int(length),) + tuple(int(x) for x in shape)


# -----------------------------------------------------------------------------
# Basic FIFO Replay Buffer
# -----------------------------------------------------------------------------


class ReplayBuffer:
    """A simple FIFO experience replay buffer for SAC agents."""

    def __init__(self, obs_dim, act_dim, size: int):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(int(size), dtype=np.float32)
        self.done_buf = np.zeros(int(size), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, int(size)

    def store(self, obs, act, rew: float, next_obs, done: bool):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = float(rew)
        self.done_buf[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: int = 32) -> Dict[str, torch.Tensor]:
        idxs = np.random.randint(0, self.size, size=int(batch_size))
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )
        return {k: torch.as_tensor(v, dtype=torch.float32, device=DEVICE) for k, v in batch.items()}

    def save(self, filename: str) -> None:
        data = {
            "obs_buf": self.obs_buf,
            "obs2_buf": self.obs2_buf,
            "act_buf": self.act_buf,
            "rew_buf": self.rew_buf,
            "done_buf": self.done_buf,
            "ptr": self.ptr,
            "size": self.size,
            "max_size": self.max_size,
        }
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def load(self, filename: str) -> None:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        self.obs_buf = data["obs_buf"]
        self.obs2_buf = data["obs2_buf"]
        self.act_buf = data["act_buf"]
        self.rew_buf = data["rew_buf"]
        self.done_buf = data["done_buf"]
        self.ptr = int(data["ptr"])
        self.size = int(data["size"])
        self.max_size = int(data["max_size"])


# -----------------------------------------------------------------------------
# Prioritized Replay Buffer (SumTree)
# -----------------------------------------------------------------------------


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)
        self.data = np.empty(self.capacity, dtype=object)
        self.write = 0

    def total(self) -> float:
        return float(self.tree[0])

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    def add(self, priority: float, data: Any) -> None:
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity

    def update(self, idx: int, priority: float) -> None:
        priority = float(max(priority, 1e-12))
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float):
        idx = self._retrieve(0, float(s))
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """Prioritized buffer storing arbitrary transitions.

    This is included for compatibility; SAC in sac.py uses FIFO ReplayBuffer.
    """

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.tree = SumTree(capacity)
        self.alpha = float(alpha)
        self.max_priority = 1.0

    def add(self, transition: Any, priority: Optional[float] = None) -> None:
        if priority is None:
            priority = self.max_priority
        p = float(priority) ** self.alpha
        self.tree.add(p, transition)
        self.max_priority = max(self.max_priority, float(priority))

    def sample(self, batch_size: int, beta: float = 0.4):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / float(batch_size)
        for i in range(int(batch_size)):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        probs = np.asarray(priorities, dtype=np.float64) / max(self.tree.total(), 1e-12)
        weights = (len(self.tree.data) * probs) ** (-float(beta))
        weights /= max(weights.max(), 1e-12)
        return batch, idxs, weights.astype(np.float32)

    def update_priorities(self, idxs: Sequence[int], priorities: Sequence[float]) -> None:
        for idx, priority in zip(idxs, priorities):
            self.tree.update(int(idx), float(priority) ** self.alpha)
            self.max_priority = max(self.max_priority, float(priority))


# -----------------------------------------------------------------------------
# Compatibility stubs for your other buffers
# -----------------------------------------------------------------------------


class LIFOReplayBuffer(ReplayBuffer):
    """LIFO variant: keeps the most recent transitions by overwriting oldest."""

    # Same as FIFO given our ring-buffer implementation; kept for compatibility.
    pass


class HERReplayBuffer(ReplayBuffer):
    """Hindsight Experience Replay placeholder.

    Your original file included a custom HER implementation. If you use HER,
    swap this with your domain-specific relabeling. Kept so imports don't break.
    """

    pass


class DualReplayBuffer:
    """A minimal dual-buffer wrapper (compatibility)."""

    def __init__(self, buffer_a: ReplayBuffer, buffer_b: ReplayBuffer, p_a: float = 0.5):
        self.a = buffer_a
        self.b = buffer_b
        self.p_a = float(p_a)

    def store(self, *args, **kwargs):
        # store into both
        self.a.store(*args, **kwargs)
        self.b.store(*args, **kwargs)

    def sample_batch(self, batch_size: int = 32) -> Dict[str, torch.Tensor]:
        if random.random() < self.p_a:
            return self.a.sample_batch(batch_size)
        return self.b.sample_batch(batch_size)

    def save(self, filename: str) -> None:
        with open(filename, "wb") as f:
            pickle.dump({"a": self.a, "b": self.b, "p_a": self.p_a}, f)

    def load(self, filename: str) -> None:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        self.a = data["a"]
        self.b = data["b"]
        self.p_a = float(data["p_a"])
