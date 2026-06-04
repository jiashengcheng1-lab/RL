#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RewardShaper.py

Lightweight, dependency-minimal reward shaping utility.

This module intentionally stays simple:
- Keeps a rolling window of portfolio returns.
- Exposes common risk-adjusted metrics (Sharpe/Sortino/CVaR/MaxDD).
- Can combine them into a single scalar reward.

Your environment can call:
    shaper.add_return(ret)
    shaped = shaper.reward(...)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np


@dataclass
class RewardWeights:
    sharpe: float = 1.0
    sortino: float = 0.5
    max_drawdown: float = 1.0
    cvar: float = 0.5


class RewardShaper:
    def __init__(self, window: int = 20, trading_days: int = 252):
        self.window = int(window)
        self.trading_days = int(trading_days)
        self.returns: Deque[float] = deque(maxlen=self.window)

    def reset(self) -> None:
        self.returns.clear()

    def add_return(self, ret: float) -> None:
        if ret is None or not np.isfinite(ret):
            ret = 0.0
        self.returns.append(float(ret))

    def _as_array(self) -> np.ndarray:
        if len(self.returns) == 0:
            return np.zeros((0,), dtype=np.float64)
        return np.asarray(self.returns, dtype=np.float64)

    def sharpe(self) -> float:
        rets = self._as_array()
        if rets.size < 3:
            return 0.0
        mu = float(np.mean(rets)) * self.trading_days
        sd = float(np.std(rets, ddof=1)) * np.sqrt(self.trading_days)
        if sd <= 1e-12:
            return 0.0
        return float(mu / sd)

    def sortino(self) -> float:
        rets = self._as_array()
        if rets.size < 3:
            return 0.0
        mu = float(np.mean(rets)) * self.trading_days
        downside = rets[rets < 0.0]
        if downside.size < 2:
            return 0.0
        dd = float(np.std(downside, ddof=1)) * np.sqrt(self.trading_days)
        if dd <= 1e-12:
            return 0.0
        return float(mu / dd)

    def cvar(self, alpha: float = 0.05) -> float:
        rets = self._as_array()
        if rets.size < 3:
            return 0.0
        a = float(alpha)
        a = min(max(a, 1e-4), 0.5)
        q = float(np.quantile(rets, a))
        tail = rets[rets <= q]
        if tail.size == 0:
            return 0.0
        return float(np.mean(tail))

    def max_drawdown(self) -> float:
        rets = self._as_array()
        if rets.size < 3:
            return 0.0
        nav = np.cumprod(1.0 + rets)
        peak = np.maximum.accumulate(nav)
        dd = nav / np.where(peak <= 0.0, np.nan, peak) - 1.0
        out = float(np.nanmin(dd))
        if not np.isfinite(out):
            return 0.0
        return out  # negative or 0

    def reward(
        self,
        w_sharpe: float = 1.0,
        w_sortino: float = 0.5,
        w_drawdown: float = 1.0,
        w_cvar: float = 0.5,
        *,
        cvar_alpha: float = 0.05,
    ) -> float:
        s = self.sharpe()
        so = self.sortino()
        mdd = self.max_drawdown()  # negative
        cv = self.cvar(alpha=cvar_alpha)  # could be negative
        # drawdown is penalty -> subtract |mdd|
        return float(w_sharpe * s + w_sortino * so - w_drawdown * abs(mdd) + w_cvar * cv)
