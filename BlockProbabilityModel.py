#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BlockProbabilityModel.py

Tracks per-asset probability that an attempted trade will be blocked.

- Uses a Beta-Bernoulli posterior with prior Beta(alpha0, beta0).
- update_with_info_gain(...) returns an information-gain proxy (KL), with a safe fallback
  when SciPy is unavailable.

Backward-compatible alias:
- BlockProbabilityTracker -> BlockProbabilityModel
"""
from __future__ import annotations

from typing import Optional

import math
import numpy as np


# ---------------------------------------------------------------------
# Optional SciPy acceleration
# ---------------------------------------------------------------------
try:  # pragma: no cover
    from scipy.special import betaln as _betaln  # type: ignore
    from scipy.special import digamma as _digamma  # type: ignore
    _SCIPY_OK = True
except Exception:  # pragma: no cover
    _SCIPY_OK = False

    def _betaln(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # betaln(a,b) = lgamma(a)+lgamma(b)-lgamma(a+b)
        lg = np.vectorize(math.lgamma)
        return lg(a) + lg(b) - lg(a + b)

    def _digamma_scalar(x: float) -> float:
        # Digamma approximation (sufficient for simple IG proxy)
        # Use recurrence to move x to > 6 then asymptotic expansion.
        if x <= 0.0:
            return float("nan")
        result = 0.0
        while x < 6.0:
            result -= 1.0 / x
            x += 1.0
        inv = 1.0 / x
        inv2 = inv * inv
        # asymptotic series
        result += math.log(x) - 0.5 * inv - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))
        return result

    def _digamma(x: np.ndarray) -> np.ndarray:
        v = np.vectorize(_digamma_scalar)
        return v(x)


class BlockProbabilityModel:
    def __init__(self, n_assets: int, alpha0: float = 1.0, beta0: float = 1.0):
        self.n_assets = int(n_assets)
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        self.blocked_counts = np.zeros(self.n_assets, dtype=np.int64)
        self.allowed_counts = np.zeros(self.n_assets, dtype=np.int64)

    def reset(self) -> None:
        self.blocked_counts[:] = 0
        self.allowed_counts[:] = 0

    def _posterior_params(self) -> tuple[np.ndarray, np.ndarray]:
        a = self.alpha0 + self.blocked_counts.astype(np.float64)
        b = self.beta0 + self.allowed_counts.astype(np.float64)
        return a, b

    @staticmethod
    def beta_kl(a_post: np.ndarray, b_post: np.ndarray, a_prior: np.ndarray, b_prior: np.ndarray) -> np.ndarray:
        """
        KL( Beta(a_post,b_post) || Beta(a_prior,b_prior) ).

        If SciPy isn't present, this still works using math.lgamma + digamma approximation.
        """
        a_post = np.asarray(a_post, dtype=np.float64)
        b_post = np.asarray(b_post, dtype=np.float64)
        a_prior = np.asarray(a_prior, dtype=np.float64)
        b_prior = np.asarray(b_prior, dtype=np.float64)

        term1 = _betaln(a_prior, b_prior) - _betaln(a_post, b_post)
        psi_sum = _digamma(a_post + b_post)
        term2 = (a_post - a_prior) * (_digamma(a_post) - psi_sum)
        term3 = (b_post - b_prior) * (_digamma(b_post) - psi_sum)
        kl = term1 + term2 + term3
        kl = np.where(np.isfinite(kl), kl, 0.0)
        return np.maximum(kl, 0.0)

    def update_with_info_gain(self, asset_idx: int, blocked: bool) -> float:
        i = int(asset_idx)
        a_old = self.alpha0 + float(self.blocked_counts[i])
        b_old = self.beta0 + float(self.allowed_counts[i])

        if blocked:
            self.blocked_counts[i] += 1
        else:
            self.allowed_counts[i] += 1

        a_new = self.alpha0 + float(self.blocked_counts[i])
        b_new = self.beta0 + float(self.allowed_counts[i])

        ig = self.beta_kl(
            np.array([a_new], dtype=np.float64),
            np.array([b_new], dtype=np.float64),
            np.array([a_old], dtype=np.float64),
            np.array([b_old], dtype=np.float64),
        )[0]
        return float(ig)

    def posterior_mean(self) -> np.ndarray:
        a, b = self._posterior_params()
        return a / (a + b)

    def posterior_var(self) -> np.ndarray:
        a, b = self._posterior_params()
        denom = (a + b) ** 2 * (a + b + 1.0)
        return (a * b) / np.where(denom <= 0.0, np.nan, denom)

    def sample_posterior(self, num_samples: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(random_state)
        a, b = self._posterior_params()
        return rng.beta(a[None, :], b[None, :], size=(int(num_samples), self.n_assets))


BlockProbabilityTracker = BlockProbabilityModel
