#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PortfolioRiskFeatureEngine.py

Rolling portfolio risk/alpha features, designed for RL state + reward.

- record(portfolio_value, market_return, risk_free_rate, timestamp)
- get_latest_features()

This file replaces the broken syntax in the uploaded version while keeping
the same class name + method signatures.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional

import numpy as np


def _safe_std(x: np.ndarray, ddof: int = 1, eps: float = 1e-12) -> float:
    if x.size < (ddof + 1):
        return 0.0
    s = float(np.std(x, ddof=ddof))
    return s if np.isfinite(s) and s > eps else 0.0


def _max_drawdown_from_nav(nav: np.ndarray) -> float:
    nav = np.asarray(nav, dtype=np.float64)
    if nav.size < 2:
        return 0.0
    peak = np.maximum.accumulate(nav)
    dd = nav / np.where(peak <= 0.0, np.nan, peak) - 1.0
    out = float(np.nanmin(dd))
    return out if np.isfinite(out) else 0.0


class PortfolioRiskFeatureEngine:
    def __init__(
        self,
        window: int = 30,
        maxlen: int = 10000,
        risk_free_rate: float = 0.000118,
        compute_market_beta: bool = True,
        prefix: str = "pf",
        annualization: int = 365,
    ):
        self.window = int(window)
        self.maxlen = int(maxlen)
        self.prefix = str(prefix)
        self.compute_market_beta = bool(compute_market_beta)
        self.annualization = int(annualization)

        self.static_risk_free = float(risk_free_rate)

        self.portfolio_values: Deque[float] = deque(maxlen=self.maxlen)
        self.market_returns: Deque[float] = deque(maxlen=self.maxlen)
        self.risk_free_rates: Deque[float] = deque(maxlen=self.maxlen)
        self.timestamps: Deque[Any] = deque(maxlen=self.maxlen)

        # filled on record()
        self.latest_features: Dict[str, float] = {}

        self.col_names = [
            f"{self.prefix}_alpha_monthly",
            f"{self.prefix}_vol_monthly",
            f"{self.prefix}_sharpe",
            f"{self.prefix}_sortino",
            f"{self.prefix}_max_drawdown",
            f"{self.prefix}_m_squared_ratio",
            f"{self.prefix}_beta",
            f"{self.prefix}_beta_adj_sharpe",
            f"{self.prefix}_cagr",
            f"{self.prefix}_cvar",
        ]
        self.reset()

    def reset(self) -> None:
        self.portfolio_values.clear()
        self.market_returns.clear()
        self.risk_free_rates.clear()
        self.timestamps.clear()
        self.latest_features = {c: 0.0 for c in self.col_names}

    def _returns_from_values(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.size < 2:
            return np.zeros((0,), dtype=np.float64)
        r = values[1:] / np.where(values[:-1] <= 0.0, np.nan, values[:-1]) - 1.0
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        return r

    def record(
        self,
        portfolio_value: float,
        market_return: float = 0.0,
        risk_free_rate: Optional[float] = None,
        timestamp: Optional[Any] = None,
    ) -> Dict[str, float]:
        pv = float(portfolio_value)
        mr = float(market_return) if np.isfinite(market_return) else 0.0
        rf = float(self.static_risk_free if risk_free_rate is None else risk_free_rate)
        rf = rf if np.isfinite(rf) else 0.0

        self.portfolio_values.append(pv)
        self.market_returns.append(mr)
        self.risk_free_rates.append(rf)
        self.timestamps.append(timestamp)

        values = np.asarray(self.portfolio_values, dtype=np.float64)
        rets = self._returns_from_values(values)
        w = min(int(self.window), int(rets.size))
        if w < 2:
            return self.latest_features

        r = rets[-w:]
        rf_per_step = float(np.mean(np.asarray(self.risk_free_rates, dtype=np.float64)[-w:]))
        x = r - rf_per_step

        mu = float(np.mean(x))
        sd = _safe_std(x, ddof=1)
        sharpe = (mu / sd) * np.sqrt(self.annualization) if sd > 0.0 else 0.0

        downside = x[x < 0.0]
        dd = _safe_std(downside, ddof=1)
        sortino = (mu / dd) * np.sqrt(self.annualization) if dd > 0.0 else 0.0

        # monthly-ish vol/alpha using 30 steps as "month" in daily data
        month = min(30, w)
        r_m = r[-month:]
        vol_m = float(np.std(r_m, ddof=1)) * np.sqrt(30.0) if month >= 2 else 0.0
        alpha_m = float(np.mean(r_m)) * 30.0

        nav = np.cumprod(1.0 + r)
        mdd = _max_drawdown_from_nav(nav)

        # CVaR
        q = float(np.quantile(r, 0.05))
        tail = r[r <= q]
        cvar = float(np.mean(tail)) if tail.size > 0 else 0.0

        # M^2 ratio proxy: (Sharpe_pf - Sharpe_mkt) * sigma_mkt + rf
        # For robustness, use available market returns in window.
        mkt = np.asarray(self.market_returns, dtype=np.float64)[-w:]
        mkt = np.nan_to_num(mkt, nan=0.0, posinf=0.0, neginf=0.0)
        mkt_x = mkt - rf_per_step
        mu_m = float(np.mean(mkt_x))
        sd_m = _safe_std(mkt_x, ddof=1)
        sharpe_m = (mu_m / sd_m) * np.sqrt(self.annualization) if sd_m > 0.0 else 0.0
        sigma_m = float(np.std(mkt, ddof=1)) if mkt.size >= 2 else 0.0
        m2 = (sharpe - sharpe_m) * sigma_m + rf_per_step

        beta = 0.0
        if self.compute_market_beta and mkt.size >= 3:
            cov = float(np.cov(r, mkt, ddof=1)[0, 1])
            var = float(np.var(mkt, ddof=1))
            beta = cov / var if var > 1e-12 else 0.0
        beta_adj_sharpe = sharpe / (1.0 + abs(beta)) if np.isfinite(beta) else sharpe

        # CAGR proxy (robust)
        # NOTE: (end/start)**(1/years) is undefined for negative ratios and will emit
        #       RuntimeWarning: invalid value encountered in scalar power.
        # We only compute CAGR when the ratio is positive and finite.
        years = (values.size - 1) / float(self.annualization)
        cagr = 0.0
        if years > 0:
            start = float(values[0])
            end = float(values[-1])
            ratio = (end / start) if (np.isfinite(start) and start > 0.0) else np.nan
            if np.isfinite(ratio) and ratio > 0.0:
                # Use log/exp form for numerical stability.
                with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
                    cagr = float(np.exp(np.log(ratio) / years) - 1.0)
            else:
                # If equity curve crosses <= 0 (or end <= 0), CAGR is not defined.
                # Returning 0.0 keeps downstream feature dimensions stable.
                cagr = 0.0

        self.latest_features = {
            f"{self.prefix}_alpha_monthly": float(alpha_m),
            f"{self.prefix}_vol_monthly": float(vol_m),
            f"{self.prefix}_sharpe": float(sharpe),
            f"{self.prefix}_sortino": float(sortino),
            f"{self.prefix}_max_drawdown": float(mdd),
            f"{self.prefix}_m_squared_ratio": float(m2),
            f"{self.prefix}_beta": float(beta),
            f"{self.prefix}_beta_adj_sharpe": float(beta_adj_sharpe),
            f"{self.prefix}_cagr": float(cagr),
            f"{self.prefix}_cvar": float(cvar),
        }
        return self.latest_features

    def get_latest_features(self) -> Dict[str, float]:
        return dict(self.latest_features)
