#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CryptoTradingEnv.py (LONG / MultiIndex drop-in)

This file is a production-grade, drop-in replacement that:
- Accepts LONG panel data (timestamp, tic) as the main dataset.
- Preserves the high-level step flow you rely on:
    update previous vars -> process actions -> record -> get_state -> step += 1
- Returns 2D observations (assets × features) when use_2d=True (recommended),
  while supporting a flat 1D fallback when use_2d=False.

Expected LONG data format:
    columns: ['timestamp','tic', ...features...]
    or MultiIndex with levels (timestamp, tic).

Minimum columns for trading:
    'Close' (used for execution + valuation)

Recommended columns present in your uploaded dataset.csv:
    Close, High, Low, Open, Volume, vix, turbulence

You can pass pre-scaled data from env_starter.neutralization() to avoid look-ahead.

Key env outputs:
- self.historical_trades (WIDE, index=timestamp)  -> used for running_scaler fitting, same spirit as your old setup
- self.historical_trades_long (LONG, MultiIndex=(timestamp,tic)) -> exact per-asset per-time audit ledger

Action modes:
- execute_trade_tanh: actions in [-1,1] per asset -> delta weight signal
- execute_trade_action_norm: actions in [0,1] per asset -> normalized weights (no cash)
- execute_trade_target_weights: actions in [0,1] incl cash -> normalized target weights

Important:
- To use 2D obs, set use_2d=True.
- PPO/SAC MLP can still ingest 2D by flattening inside the policy network.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Gym compatibility (gym / gymnasium optional)
# ------------------------------------------------------------------
try:
    import gym  # type: ignore
    _GYM_AVAILABLE = True
    EnvBase = gym.Env
    BoxSpace = gym.spaces.Box
except Exception:  # pragma: no cover
    _GYM_AVAILABLE = False

    class EnvBase:  # minimal stand-in
        metadata = {}

        def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
            return None

    class BoxSpace:
        def __init__(self, low, high, shape, dtype=np.float32):
            self.low = low
            self.high = high
            self.shape = tuple(shape)
            self.dtype = dtype

        def sample(self):
            low = np.asarray(self.low, dtype=np.float64)
            high = np.asarray(self.high, dtype=np.float64)
            if low.size == 1 and high.size == 1:
                return np.random.uniform(float(low), float(high), size=self.shape).astype(self.dtype)
            # broadcast
            return np.random.uniform(low, high, size=self.shape).astype(self.dtype)

from sklearn.preprocessing import MinMaxScaler

from RewardShaper import RewardShaper
from PortfolioRiskFeatureEngine import PortfolioRiskFeatureEngine
from LivePnLCalculator import LivePnLCalculator
from BlockProbabilityModel import BlockProbabilityTracker


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    t = float(temperature)
    if not np.isfinite(t) or t <= 0.0:
        t = 1.0
    z = x / t
    z = z - np.max(z)
    e = np.exp(z)
    s = np.sum(e)
    if s <= 0.0 or not np.isfinite(s):
        return np.ones_like(x, dtype=np.float64) / float(x.size)
    return e / s


def _safe_log_return(v_prev: float, v_next: float, eps: float = 1e-12) -> float:
    a = max(float(v_prev), eps)
    b = max(float(v_next), eps)
    return float(np.log(b / a))


def _ensure_long_multiindex(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    tic_col: str = "tic",
) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.index, pd.MultiIndex) and out.index.nlevels >= 2:
        # attempt to reorder to (timestamp, tic)
        names = list(out.index.names)
        if timestamp_col in names and tic_col in names:
            if names[:2] != [timestamp_col, tic_col]:
                out = out.reorder_levels([timestamp_col, tic_col] + [n for n in names if n not in (timestamp_col, tic_col)])
        else:
            # fallback: reset and set
            out = out.reset_index()
    if not (isinstance(out.index, pd.MultiIndex) and out.index.nlevels >= 2 and out.index.names[:2] == [timestamp_col, tic_col]):
        if timestamp_col not in out.columns or tic_col not in out.columns:
            raise ValueError("LONG df must have columns ['timestamp','tic'] or MultiIndex with names (timestamp,tic).")
        out[timestamp_col] = pd.to_datetime(out[timestamp_col])
        out = out.set_index([timestamp_col, tic_col], drop=True)
    out = out.sort_index(level=[0, 1])
    return out


@dataclass
class TradeResult:
    executed: bool
    blocked: bool
    reason: str
    tx_fee_paid: float
    notional: float
    qty_delta: float


class CryptoTradingEnv(EnvBase):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        data=None,
        # Backward-compatible alias used by some runners
        data_long=None,
        selected_assets=[
            "BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD", "XRP-USD",
            "SOL-USD", "TRX-USD", "STETH-USD", "DOGE-USD", "ADA-USD",
            "BCH-USD", "XMR-USD", "LEO-USD", "LINK-USD", "XLM-USD",
            "ZEC-USD", "DAI-USD", "CRO-USD", "DOT-USD", "UNI7083-USD",
            "XAUT-USD", "PAXG-USD", "OKB-USD", "AAVE-USD", "NEAR-USD",
            "MSTR", "V", "MA", "PYPL",
            # # "SQ",
            "CME", "ICE", "STT", "GS",
            "MS", "JPM", "EQIX", "DLR", "MARA", "RIOT", "WULF", "HUT",
            "BITF", "NVDA", "AMD", "INTC",
        ],
        initial_balance=1000,
        transaction_fee=0.001,
        start_step=3,
        scaler=MinMaxScaler(),
        running_scaler=MinMaxScaler(),
        train=True,
        price_data=None,
        execution_price_col: str = "Close_raw",
        seq_len=30,
        window_size=30,
        train_ratio=0.7,
        buffer_size=1000,
        env_name="CryptoTradingEnv",
        use_2d=False,
        if_discrete=False,
        use_seq_obs=False,
        use_action_norm=False,
        initialized=False,
        is_td3_softmax=False,
        use_sequential_actions=False,
        is_ppo=True,
        fixed_fee=1.0,
        min_trade_value=10.0,
        pct_risk_free_rate=0.000118,
        risk_free_rate=0.000114715,
        ema_short_period: int = 12,
        ema_long_period: int = 26,
        rsi_window: int = 14,
        entropy_bins: int = 10,
        var_ratio_lag: int = 5,
        horizon: int = 0, # Optional[int] = None,
        reward_weights=dict(
            pf_sharpe=1.0,
            log_return=1.0,
            vol_monthly=0.05,
            pf_max_drawdown=0.05,
            alpha_monthly=1.0,
            m_squared_ratio=0.05,
            beta_adj_sharpe=0.5,
            pf_cvar=0.001,
            pf_sortino=0.1,
            pf_beta=0.1,
            blocked_actions_w=0.001,
            tx_fee_w=0.002,
            prospect_theory_loss_aversion_alpha=2.0,
            prospect_theory_loss_aversion_w=0.5,
        ),
            random_start: Optional[bool] = None,
        seed: Optional[int] = None,
        obs_clip: float = 10.0,
        pf_clip: float = 5.0,
        reward_clip: Optional[float] = 10.0,
        normalize_positions: bool = True,
        normalize_pnl: bool = True,
        normalize_tx_fee: bool = True,
        normalize_blocked: bool = True,
    ):
        self.env_name = str(env_name)
        self.train = bool(train)
        self.use_2d = bool(use_2d)
        self.use_seq_obs = bool(use_seq_obs)
        self.use_action_norm = bool(use_action_norm)
        self.is_td3_softmax = bool(is_td3_softmax)

        self.initial_balance = float(initial_balance)
        self.balance = float(initial_balance)
        self.transaction_fee = float(transaction_fee)
        self.fixed_fee = float(fixed_fee)
        self.min_trade_value = float(min_trade_value)

        self.seq_len = int(seq_len)
        self.window_size = int(window_size)
        self.start_step = int(start_step)

        self.reward_weights = dict(reward_weights)
        self.pct_risk_free_rate = float(pct_risk_free_rate)
        self.risk_free_rate = float(risk_free_rate)

        self.scaler = scaler
        self.running_scaler = running_scaler

        # ---- Sampling & stabilization ----
        self._rng = np.random.default_rng(seed)
        self.random_start = bool(self.train) if random_start is None else bool(random_start)
        self.obs_clip = float(obs_clip)
        self.pf_clip = float(pf_clip)
        self.reward_clip = None if reward_clip is None else float(reward_clip)
        self.normalize_positions = bool(normalize_positions)
        self.normalize_pnl = bool(normalize_pnl)
        self.normalize_tx_fee = bool(normalize_tx_fee)
        self.normalize_blocked = bool(normalize_blocked)

        # ---- Load/normalize data (LONG) ----
        if data is None and data_long is not None:
            data = data_long

        if data is None:
            raise ValueError("CryptoTradingEnv requires LONG 'data' DataFrame (timestamp,tic).")
        if isinstance(data, str):
            # allow passing a csv/parquet path
            if data.lower().endswith(".csv"):
                data = pd.read_csv(data)
            else:
                data = pd.read_parquet(data)

        self.data_long = _ensure_long_multiindex(data, timestamp_col="timestamp", tic_col="tic")

        # ------------------------------------------------------------------
        # Execution prices (NO LEVERAGE / NO SHORTING safety)
        # ------------------------------------------------------------------
        # DO NOT use standardized prices for execution/valuation.
        # If Close is standardized it can be negative and PV will go negative
        # even with non-negative holdings. We therefore support a separate
        # execution price column (default: Close_raw) or a separate price_data
        # input. This keeps trading/valuation in real price space while
        # allowing observation features to be scaled.
        self.execution_price_col = str(execution_price_col)

        self.price_long = None
        if price_data is not None:
            if isinstance(price_data, str):
                if price_data.lower().endswith(".csv"):
                    price_df = pd.read_csv(price_data)
                else:
                    price_df = pd.read_parquet(price_data)
            else:
                price_df = price_data.copy()
            self.price_long = _ensure_long_multiindex(price_df, timestamp_col="timestamp", tic_col="tic")
        else:
            self.price_long = self.data_long

        # Choose a valid execution price column
        if self.execution_price_col not in self.price_long.columns:
            if "Close" in self.price_long.columns:
                self.execution_price_col = "Close"
            else:
                raise ValueError("Price data must contain 'Close' or the specified execution_price_col.")

        # Filter assets to those available
        avail_assets = sorted(self.data_long.index.get_level_values(1).unique().tolist())
        self.selected_assets = [a for a in list(selected_assets) if a in avail_assets]
        if len(self.selected_assets) == 0:
            # fallback: use all assets in data
            self.selected_assets = avail_assets

        self.n_assets = int(len(self.selected_assets))

        # feature columns (market)
        # 'Close' (scaled) can remain a feature, but execution uses execution_price_col.
        self.price_col = "Close"  # feature name only
        if self.price_col not in self.data_long.columns:
            # Allow feature Close missing if user provides other features, but execution must exist.
            self.price_col = self.execution_price_col

        # The env uses all numeric columns as market features by default,
        # but you can override by scaling/preprocessing upstream.
        # Exclude raw execution columns from observation features by default
        raw_cols = [c for c in self.data_long.columns if str(c).lower().endswith("_raw")]
        self.market_feature_cols = [c for c in self.data_long.columns if c not in raw_cols]
        # ensure deterministic order and numeric-ish
        self.market_feature_cols = list(dict.fromkeys(self.market_feature_cols))

        # Ensure timestamp axis
        self.timestamps = self.data_long.index.get_level_values(0).unique().sort_values()
        self.n_steps_total = int(len(self.timestamps))
        if self.n_steps_total <= (self.start_step + 2):
            raise ValueError("Not enough timestamps for the requested start_step.")

        # ---- Precompute feature tensor for fast get_state() ----
        # market tensor shape: (T, N, F_market)
        # price tensor shape:  (T, N)
        self._build_market_tensor()

        # ---- Portfolio state ----
        self.portfolio_qty = np.zeros(self.n_assets, dtype=np.float64)
        self.portfolio_cash = float(self.initial_balance)
        self.total_transaction_fee = 0.0

        # action tracking
        self.actions: List[List[float]] = []
        self.raw_actions = None
        self.previous_actions = None
        self.last_actions_raw = np.zeros(self.n_assets, dtype=np.float64)
        self.last_actions_scaled = np.zeros(self.n_assets, dtype=np.float64)

        # Performance history
        self.portfolio_values: List[float] = []
        self.returns: List[float] = []
        self.cumulative_returns_history: List[float] = []

        # trackers / helpers
        self.reward_shaper = RewardShaper(window=self.window_size, trading_days=365)
        self.risk_engine = PortfolioRiskFeatureEngine(
            window=self.window_size, risk_free_rate=self.pct_risk_free_rate,
            annualization=365, prefix="pf"
        )
        self.live_pnl = LivePnLCalculator(assets=self.selected_assets)
        self.block_tracker = BlockProbabilityTracker(self.n_assets, alpha0=1.0, beta0=1.0)

        # Historical records
        self.historical_trades = pd.DataFrame()
        self.historical_trades_long = pd.DataFrame()

        # episode bounds / horizon (index-based, NOT length-based)
        self.end_step = self.n_steps_total - 1
        if horizon !=0.: # is not None:
            self.max_ep_len = int(
                max(1, min(int(horizon), self.end_step - self.start_step))
            )
        else:
            self.max_ep_len = int(
                max(1, self.end_step - self.start_step)
            )
        self.current_step = self.start_step
        self.episode_step = 0

        # ---- Spaces ----
        # Actions:
        if not self.use_action_norm and not self.is_td3_softmax:
            self.action_space = BoxSpace(
                low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
            )
        elif self.is_td3_softmax:
            # includes cash
            self.action_space = BoxSpace(
                low=0.0, high=1.0, shape=(self.n_assets + 1,), dtype=np.float32
            )
        else:
            self.action_space = BoxSpace(
                low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
            )

        # Observation: 2D assets × features if use_2d else flat
        # We'll infer the feature_dim after reset() once state is built.
        obs = self.reset(debug=False)
        if self.use_2d:
            self.observation_space = BoxSpace(
                low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
            )
        else:
            self.observation_space = BoxSpace(
                low=-np.inf, high=np.inf, shape=(obs.size,), dtype=np.float32
            )

        logging.info(
            "CryptoTradingEnv initialized: n_assets=%s, timestamps=%s", self.n_assets, self.n_steps_total
        )

    # ------------------------------------------------------------------
    # Public scaler hooks (used by env_starter.prefill...)
    # ------------------------------------------------------------------
    def set_scaler(self, scaler) -> None:
        self.scaler = scaler

    def get_scaler(self):
        return self.scaler

    def set_running_scaler(self, running_scaler) -> None:
        self.running_scaler = running_scaler

    def get_running_scaler(self):
        return self.running_scaler

    # ------------------------------------------------------------------
    # Data tensor build
    # ------------------------------------------------------------------
    def _build_market_tensor(self) -> None:
        """Build (market_tensor, price_tensor) aligned on (timestamps × selected_assets)."""
        F = len(self.market_feature_cols)
        T = len(self.timestamps)
        N = self.n_assets

        idx = pd.MultiIndex.from_product([self.timestamps, self.selected_assets], names=["timestamp", "tic"])

        # ----------------------------
        # Market features (can be scaled)
        # ----------------------------
        df_feat = self.data_long.reindex(idx)
        df_feat = df_feat.groupby(level='tic').ffill() # .groupby(level='tic').bfill()
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan).fillna(value=0.0)
        X = df_feat[self.market_feature_cols].to_numpy(dtype=np.float32)
        market_tensor = X.reshape(T, N, F)

        # ----------------------------
        # Execution prices (MUST be raw/non-negative)
        # ----------------------------
        df_px = self.price_long.reindex(idx)
        df_px = df_px.groupby(level=1).ffill().groupby(level=1).bfill()
        px = df_px[self.execution_price_col].to_numpy(dtype=np.float64)
        px = np.nan_to_num(px, nan=0.0, posinf=0.0, neginf=0.0)
        price_tensor = px.reshape(T, N)

        # If execution prices are standardized/negative, fail fast.
        # (Allow zeros for missing data, but not negative.)
        if np.nanmin(price_tensor) < -1e-12:
            raise ValueError(
                f"Execution prices contain negative values (min={float(np.nanmin(price_tensor))}). "
                "Do not standardize prices for execution/valuation. "
                "Preserve raw prices in a Close_raw column (recommended) and set execution_price_col='Close_raw'."
            )

        self._market_tensor = market_tensor
        self._price_tensor = price_tensor
        self._market_index = idx
        self._T = T
        self._F_market = F

    def _prices_at(self, t_idx: int) -> np.ndarray:
        """Execution/valuation prices for time index t_idx."""
        px = self._price_tensor[t_idx, :].astype(np.float64)
        return np.nan_to_num(px, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    # Reset / State
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, debug: bool = False):
        if _GYM_AVAILABLE:
            try:
                super().reset(seed=seed)
            except Exception:
                pass

        # (Optional) reseed the env RNG
        if seed is not None:
            try:
                self._rng = np.random.default_rng(int(seed))
            except Exception:
                pass

        # episode counter (used for horizon-based termination)
        self.episode_step = 0

        # Randomize the start for TRAINING to avoid repeatedly overfitting the same prefix.
        # For test/validation, keep deterministic start.
        if self.random_start and bool(getattr(self, 'train', False)):
            # We use (self._T - 2) as the latest step that still allows a forward price.
            max_start = int((self._T - 2) - (self.max_ep_len - 1))
            if max_start > int(self.start_step):
                self.current_step = int(self._rng.integers(int(self.start_step), max_start + 1))
            else:
                self.current_step = int(self.start_step)
        else:
            self.current_step = int(self.start_step)

        self.portfolio_cash = float(self.initial_balance)
        self.balance = float(self.initial_balance)
        self.portfolio_qty[:] = 0.0
        self.total_transaction_fee = 0.0

        self.actions = []
        self.raw_actions = None
        self.previous_actions = None
        self.last_actions_raw[:] = 0.0
        self.last_actions_scaled[:] = 0.0

        self.portfolio_values = []
        self.returns = []
        self.cumulative_returns_history = []

        self.reward_shaper.reset()
        self.risk_engine.reset()
        self.live_pnl.reset()
        self.block_tracker.reset()

        self._init_historical_ledgers()

        # initial portfolio value at current timestamp using current prices
        pv0 = self._portfolio_value_at(self.current_step)
        self.portfolio_values.append(float(pv0))

        obs = self.get_state(debug=debug)
        return obs

    def _init_historical_ledgers(self) -> None:
        # wide ledger indexed by timestamp (your running_scaler expects a row)
        # We'll create as we go to avoid huge empty frame.
        self.historical_trades = pd.DataFrame()
        # long ledger for exact auditing
        self.historical_trades_long = pd.DataFrame()

        # pre-store feature name ordering for deterministic columns later
        self._asset_fields = [
            "action_raw",
            "action_effective",
            "target_weight",
            "weight",
            "qty",
            "qty_delta",
            "price",
            "notional",
            "tx_fee_paid",
            "blocked",
            "blocked_prob",
            "fifo_realized",
            "fifo_unrealized",
            "lifo_realized",
            "lifo_unrealized",
            "fifo_total",
            "lifo_total",
        ]
        self._global_fields = [
            "portfolio_value",
            "cash",
            "return",
            # "log_return",
            # "reward",
            "total_tx_fee",
        ]
        # plus pf features from risk_engine (dynamic)
        self._pf_fields = list(self.risk_engine.get_latest_features().keys())

    def _portfolio_value_at(self, t_idx: int) -> float:
        px = self._prices_at(t_idx)
        return float(self.portfolio_cash + np.sum(self.portfolio_qty * px))

    def _weights_at(self, t_idx: int) -> np.ndarray:
        pv = self._portfolio_value_at(t_idx)
        if pv <= 1e-12:
            return np.zeros((self.n_assets,), dtype=np.float64)
        px = self._prices_at(t_idx)
        vals = self.portfolio_qty * px
        return vals / pv

    def get_state(
        self,
        window_size: int = 30, # 7,
        debug: bool = False,
        *,
        apply_running_scaler: bool = True,
    ):
        """
        Build the per-asset observation matrix at the current step.

        Returns:
            - if self.use_2d=True: np.ndarray shape (n_assets, feature_dim)
            - else: flat vector shape (n_assets * feature_dim,)

        Scaling:
            If apply_running_scaler=True and self.running_scaler is set + fitted, we scale the FULL state
            matrix (market + portfolio + pnl + pf + action history). This is the single intended place
            to normalize observations when neutralization() returns UNSCALED frames.
        """
        t = int(self.current_step)

        # Market features for obs (can be raw; execution prices are handled separately)
        market = self._market_tensor[t, :, :].astype(np.float32)  # (N, Fm)

        # Portfolio features (raw magnitudes; will be normalized by running_scaler if enabled)
        px = self._prices_at(t)
        pv = self._portfolio_value_at(t)
        w = self._weights_at(t)

        qty_units = self.portfolio_qty.astype(np.float32)
        if self.normalize_positions:
            # Stable across price regimes: log(1+qty) and value as portfolio-weight.
            qty = np.log1p(np.maximum(qty_units, 0.0)).astype(np.float32)
            val = ((qty_units * px) / (pv if pv > 1e-12 else 1.0)).astype(np.float32)
        else:
            qty = qty_units
            val = (qty_units * px).astype(np.float32)
        cash_w = (self.portfolio_cash / pv) if pv > 1e-12 else 0.0

        # Block probabilities
        blocked_prob = self.block_tracker.posterior_mean().astype(np.float32)

        # PnL features (EWMA smoothed)
        pnl_rows = []
        denom_pnl = (pv if (self.normalize_pnl and pv > 1e-12) else 1.0)
        for i, a in enumerate(self.selected_assets):
            m = self.live_pnl.get_pnl(a, current_price=float(px[i]))
            row = [
                float(m.get("fifo_realized", 0.0)),
                float(m.get("fifo_unrealized", 0.0)),
                float(m.get("lifo_realized", 0.0)),
                float(m.get("lifo_unrealized", 0.0)),
                float(m.get("fifo_total", 0.0)),
                float(m.get("lifo_total", 0.0)),
            ]
            if denom_pnl != 1.0:
                row = [x / denom_pnl for x in row]
            pnl_rows.append(row)
        pnl = np.asarray(pnl_rows, dtype=np.float32)

        # Risk features (global, repeated)
        pf = self.risk_engine.get_latest_features()
        pf_vec = np.asarray([float(pf.get(k, 0.0)) for k in self._pf_fields], dtype=np.float32)
        # Clip/tanh-scale global risk features to avoid rare huge spikes dominating the networks.
        if self.pf_clip is not None and self.pf_clip > 0:
            c = float(self.pf_clip)
            pf_vec = (c * np.tanh(pf_vec / c)).astype(np.float32)
        pf_rep = np.repeat(pf_vec[None, :], self.n_assets, axis=0)

        # Last action history (we keep raw values as features; scaling is done on the full state below)
        a_raw = self.last_actions_raw.astype(np.float32)
        # a_scaled = a_raw  # kept for backward compatibility in feature names
        a_scaled = _softmax(a_raw) # Edited
        self.last_actions_scaled = a_scaled.astype(np.float64)

        # Assemble per-asset state matrix:
        # market + [qty, val, weight, cash_w, a_raw, a_scaled, blocked_prob] + pnl + pf
        per_asset = np.column_stack([
            market, # Original.
            qty.reshape(-1, 1),
            val.reshape(-1, 1),
            w.astype(np.float32).reshape(-1, 1),
            np.full((self.n_assets, 1), float(cash_w), dtype=np.float32),
            a_raw.reshape(-1, 1),
            a_scaled.reshape(-1, 1),
            blocked_prob.reshape(-1, 1),
            pnl,
            pf_rep,
        ]).astype(np.float32)

        self.state_feature_names = (
            [f"mkt__{c}" for c in self.market_feature_cols]
            + ["pos__qty", "pos__value", "pos__weight", "pos__cash_weight"]
            + ["act__raw", "act__scaled", "blk__prob"]
            + [
                "pnl__fifo_realized", "pnl__fifo_unrealized",
                "pnl__lifo_realized", "pnl__lifo_unrealized",
                "pnl__fifo_total", "pnl__lifo_total",
            ]
            + [f"pf__{k}" for k in self._pf_fields]
        )

        # Apply running_scaler to the FULL state matrix (row-wise over assets)
        if apply_running_scaler and self.running_scaler is not None and hasattr(self.running_scaler, "transform"):
            try:
                Xs = self.running_scaler.transform(per_asset.astype(np.float64, copy=False)) # Original.
                per_asset_scaled = Xs.astype(np.float32, copy=False)                         # Original.
                # print(f'Xs.shape: {Xs.shape} | market.shape: {market.shape}')
                # market = self.scaler.transfrom(market.astype(np.float64, copy=False))
                # per_asset_scaled = np.column_stack([
                #     market,
                #     Xs,
                # ]).astype(np.float32)
                # print(f'per_asset_scaled.shape: {per_asset_scaled.shape}')
            except Exception as e:
                print(f'exception: {e}')
                # self.running_scaler.fit(per_asset.astype(np.float64, copy=False))
                # Xs = self.running_scaler.transform(per_asset.astype(np.float64, copy=False))
                # per_asset_scaled = Xs.astype(np.float32, copy=False)
                per_asset_scaled = per_asset                                                   # Original.
        else:
            per_asset_scaled = per_asset
        # Final safety clip to bound outliers (helps critic stability).
        if self.obs_clip is not None and self.obs_clip > 0:
            c = float(self.obs_clip)
            # per_asset_scaled = np.clip(per_asset_scaled, -c, c).astype(np.float32, copy=False) # Original.
            # per_asset_scaled = np.clip(per_asset_scaled, 0, 1.0).astype(np.float32, copy=False)

        if self.use_2d:
            return per_asset_scaled
        return per_asset_scaled.reshape(-1)

    # ------------------------------------------------------------------
    # Step / Trading
    # ------------------------------------------------------------------
    def step(self, actions, debug: bool = False, is_env_action_sample: bool = False):
        # --- Update Previous Values ---
        self.previous_actions = deepcopy(self.actions[-1]) if len(self.actions) > 0 else None
        prev_value = float(self._portfolio_value_at(self.current_step - 1)) if self.current_step > 0 else float(self._portfolio_value_at(self.current_step))
        self.previous_portfolio_value = prev_value
        self.prev_balance = float(self.portfolio_cash)
        self.prev_total_transaction_fee = float(self.total_transaction_fee)

        # --- Sanitize actions ---
        a = np.asarray(actions, dtype=np.float64).reshape(-1)
        if self.is_td3_softmax:
            if a.size != (self.n_assets + 1):
                raise ValueError(f"is_td3_softmax=True expects action dim {self.n_assets+1}, got {a.size}.")
        else:
            if a.size != self.n_assets:
                raise ValueError(f"Expected action dim {self.n_assets}, got {a.size}.")

        self.raw_actions = a.copy()
        # keep last actions (pre normalization)
        if self.is_td3_softmax:
            self.last_actions_raw = a[:-1].copy()
        else:
            self.last_actions_raw = a.copy()

        # --- Determine timestamp triplet ---
        t = int(self.current_step)
        ts = self.timestamps[t]
        next_t = t + 1
        done = False
        # --- Determine timestamp triplet --- # Original.
        '''if next_t >= self._T:
            # done = True # Original.
            done = (
                next_t >= self._T or
                self.portfolio_cash < 0. or self.portfolio_value < 0 or
                any(
                    # self.portfolio[asset] < 0. for asset in self.selected_assets
                    self.portfolio_qty < 0.
                )
            )
            # still record and return last obs
            obs = self.get_state(debug=debug)
            return obs, 0.0, True, {"reason": "eod"}'''

        # --- Determine timestamp triplet --- # Edited
        portfolio_value = self.portfolio_values[-1]
        if self.portfolio_cash < 0. or portfolio_value < 0. or any(self.portfolio_qty < 0.):
            done = True
            # still record and return last obs
            obs = self.get_state(debug=debug)
            return obs, 0.0, True, {"reason": "actions_oob"}

        # --- Execute trade at current timestamp (t) close ---
        px_t = self._prices_at(t)
        trade_blocked = np.zeros(self.n_assets, dtype=np.int8)
        tx_fee_paid_vec = np.zeros(self.n_assets, dtype=np.float64)
        qty_delta_vec = np.zeros(self.n_assets, dtype=np.float64)
        notional_vec = np.zeros(self.n_assets, dtype=np.float64)
        action_effective = np.zeros(self.n_assets, dtype=np.float64)
        target_weights = np.zeros(self.n_assets, dtype=np.float64)

        if self.is_td3_softmax:
            tr = self.execute_trade_target_weights(a, px_t)
            trade_blocked, tx_fee_paid_vec, qty_delta_vec, notional_vec, action_effective, target_weights = tr
        elif self.use_action_norm:
            tr = self.execute_trade_action_norm(a, px_t, is_env_action_sample=is_env_action_sample)
            trade_blocked, tx_fee_paid_vec, qty_delta_vec, notional_vec, action_effective, target_weights = tr
        else:
            tr = self.execute_trade_tanh(a, px_t)
            trade_blocked, tx_fee_paid_vec, qty_delta_vec, notional_vec, action_effective, target_weights = tr

        # update last action effective
        self.actions.append(action_effective.tolist())

        # --- Portfolio value at next timestamp (t+1) after price move ---
        pv_next = float(self._portfolio_value_at(next_t))
        log_ret = _safe_log_return(prev_value, pv_next)
        ret = float(np.expm1(log_ret))

        self.returns.append(ret)
        self.reward_shaper.add_return(ret)

        # market return proxy: equal-weight average of asset returns from t-1 to t (or t to t+1)
        px_next = self._prices_at(next_t)
        mkt_arr = px_next / np.where(px_t <= 0.0, np.nan, px_t) - 1.0
        if np.all(np.isnan(mkt_arr)):
            mkt_ret = 0.0
        else:
            mkt_ret = float(np.nanmean(mkt_arr))
            if not np.isfinite(mkt_ret):
                mkt_ret = 0.0

        pf_feats = self.risk_engine.record(portfolio_value=pv_next, market_return=mkt_ret, risk_free_rate=self.pct_risk_free_rate, timestamp=ts)

        reward = self._compute_reward(
            log_ret=log_ret,
            ret=ret,
            pf_feats=pf_feats,
            blocked_vec=trade_blocked,
            tx_fee_vec=tx_fee_paid_vec,
            pv_prev=float(prev_value),
        )

        self.portfolio_values.append(pv_next)
        self.total_transaction_fee += float(np.sum(tx_fee_paid_vec))

        # --- Record historical data at timestamp ts for auditing ---
        self.record_historical_data(
            timestamp=ts,
            prices=px_t,
            actions_raw=(a[:-1] if self.is_td3_softmax else a),
            actions_effective=action_effective,
            target_weights=target_weights,
            qty_delta=qty_delta_vec,
            notional=notional_vec,
            tx_fee_paid=tx_fee_paid_vec,
            blocked=trade_blocked,
            reward=reward,
            ret=ret,
            log_ret=log_ret,
            pv=float(self._portfolio_value_at(t)),  # value right after execution at t
            pv_next=pv_next,
        )

        # --- Determine timestamp triplet --- # Edited
        if next_t >= self._T:
            # done = True # Original.
            portfolio_value = self.portfolio_values[-1]
            done = (
                next_t >= self._T or
                self.portfolio_cash < 0. or portfolio_value < 0 or
                # is_done or
                # self.is_month_end() or
                # self.is_week_end() or
                any(
                    # self.portfolio[asset] < 0. for asset in self.selected_assets
                    self.portfolio_qty < 0.
                )
            )
            # still record and return last obs
            obs = self.get_state(debug=debug)
            return obs, reward, True, {"reason": "eod"}

        # obs = self.get_state(debug=debug)     # Edited

        # --- Advance step ---
        self.current_step += 1
        self.episode_step = int(getattr(self, 'episode_step', 0) + 1)
        if (self.current_step >= (self._T - 2)) or (self.episode_step >= (self.max_ep_len - 1)):
            # done = True # Original.
            portfolio_value = self.portfolio_values[-1]
            done = (
                (self.current_step >= (self._T - 2)) or
                (self.episode_step >= (self.max_ep_len - 1)) or
                self.portfolio_cash < 0. or portfolio_value < 0 or
                any(
                    self.portfolio_qty < 0.
                )
            )


        obs = self.get_state(debug=debug) # Original.
        # print(f'obs.shape: {obs.shape} | max(obs): {np.max(obs)} | min(obs): | {np.min(obs)}')
        info = {
            "timestamp": ts,
            "portfolio_value_next": pv_next,
            "return": ret,
            "log_return": log_ret,
            "blocked_actions": int(np.sum(trade_blocked)),
            "tx_fee_paid": float(np.sum(tx_fee_paid_vec)),
        }
        # >>> ADD THIS <<<
        if bool(done) and self.current_step >= (self._T - 2):
            info["truncated"] = True
            info["reason"] = "end_of_data"
        return obs, float(reward), bool(done), info

    # ------------------------------------------------------------------
    # Trading engines
    # ------------------------------------------------------------------
    def execute_trade_tanh(self, actions: np.ndarray, prices: np.ndarray):
        """
        Actions in [-1,1] per asset.
        Interpret as delta-weight signal around current weights.

        Overtrading guard:
        - cap per-step weight change (max_weight_change)
        - enforce min_trade_value
        """
        a = np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0)
        px = np.asarray(prices, dtype=np.float64)
        pv = self._portfolio_value_at(self.current_step)
        if pv <= 1e-12:
            pv = float(self.portfolio_cash)

        cur_w = self._weights_at(self.current_step)
        # max_weight_change = 0.10  # 10% per step # Original.
        delta = a # * max_weight_change            # Original.
        tgt_w = cur_w + delta                    # Original.
        tgt_w = np.clip(tgt_w, 0.0, 1.0)
        s = float(np.sum(tgt_w))
        if s <= 1e-12:
            tgt_w = np.ones_like(tgt_w) / float(self.n_assets)
        else:
            tgt_w = tgt_w / s

        # tgt_w = _softmax(a)
        return self._rebalance_to_target_weights(tgt_w, px, actions_effective=tgt_w)

    def execute_trade_action_norm(self, actions: np.ndarray, prices: np.ndarray, is_env_action_sample: bool = False):
        """
        Actions in [0,1] per asset. The env normalizes to weights (no cash asset).
        """
        a = np.asarray(actions, dtype=np.float64).reshape(-1)
        a = np.clip(a, 0.0, 1.0)

        # robust normalization
        if is_env_action_sample:
            w = _softmax(a)
        else:
            s = float(np.sum(a))
            w = a / s if s > 1e-12 else (np.ones_like(a) / float(a.size))
        w = np.clip(w, 0.0, 1.0)
        w = w / float(np.sum(w)) if float(np.sum(w)) > 1e-12 else (np.ones_like(w) / float(w.size))

        return self._rebalance_to_target_weights(w, np.asarray(prices, dtype=np.float64), actions_effective=w)

    def execute_trade_target_weights(self, actions: np.ndarray, prices: np.ndarray):
        """
        Actions in [0,1] of length n_assets+1, including cash as last entry.
        """
        a = np.asarray(actions, dtype=np.float64).reshape(-1)
        a = np.clip(a, 0.0, 1.0)
        if a.size != (self.n_assets + 1):
            raise ValueError("execute_trade_target_weights expects (n_assets+1) actions incl cash.")

        s = float(np.sum(a))
        w_all = a / s if s > 1e-12 else (np.ones_like(a) / float(a.size))
        w_assets = w_all[:-1]
        w_cash = float(w_all[-1])

        # Normalize assets to (1 - cash_weight)
        sA = float(np.sum(w_assets))
        if sA <= 1e-12:
            w_assets = np.ones_like(w_assets) * (1.0 - w_cash) / float(self.n_assets)
        else:
            w_assets = w_assets / sA * (1.0 - w_cash)

        return self._rebalance_to_target_weights(w_assets, np.asarray(prices, dtype=np.float64), actions_effective=w_assets)

    def _rebalance_to_target_weights(self, target_weights: np.ndarray, prices: np.ndarray, actions_effective: np.ndarray):
        """
        Rebalance with sells first then buys, enforcing:
        - min_trade_value
        - transaction_fee + fixed_fee
        """
        tgt_w = np.asarray(target_weights, dtype=np.float64)
        px = np.asarray(prices, dtype=np.float64)

        pv = self._portfolio_value_at(self.current_step)
        if pv <= 1e-12:
            pv = float(self.portfolio_cash)

        # current values
        cur_vals = self.portfolio_qty * px
        cur_w = cur_vals / pv if pv > 1e-12 else np.zeros_like(cur_vals)

        tgt_vals = tgt_w * pv
        delta_vals = tgt_vals - cur_vals  # positive buy, negative sell

        blocked = np.zeros(self.n_assets, dtype=np.int8)
        tx_fee_paid = np.zeros(self.n_assets, dtype=np.float64)
        qty_delta = np.zeros(self.n_assets, dtype=np.float64)
        notional = np.zeros(self.n_assets, dtype=np.float64)

        # sell first
        sell_idx = np.where(delta_vals < 0.0)[0]
        for i in sell_idx:
            if px[i] <= 0.0:
                blocked[i] = 1
                self.block_tracker.update_with_info_gain(i, blocked=True)
                continue
            sell_val = -float(delta_vals[i])
            if sell_val < self.min_trade_value:
                blocked[i] = 1
                self.block_tracker.update_with_info_gain(i, blocked=True)
                continue
            sell_qty = sell_val / float(px[i])
            sell_qty = min(sell_qty, float(self.portfolio_qty[i]))  # cannot sell more than holdings
            if sell_qty <= 1e-18:
                blocked[i] = 1
                self.block_tracker.update_with_info_gain(i, blocked=True)
                continue

            fee = sell_val * self.transaction_fee + self.fixed_fee
            proceeds = sell_val - fee
            if proceeds < 0.0:
                blocked[i] = 1
                self.block_tracker.update_with_info_gain(i, blocked=True)
                continue

            self.portfolio_qty[i] -= sell_qty
            self.portfolio_cash += proceeds
            self.balance = self.portfolio_cash

            tx_fee_paid[i] = fee
            qty_delta[i] = -sell_qty
            notional[i] = sell_val
            blocked[i] = 0
            self.block_tracker.update_with_info_gain(i, blocked=False)

            # pnl tracker
            self.live_pnl.process_trade(self.selected_assets[i], volume=sell_qty, price=float(px[i]), trade_type="sell")

        # buy second
        buy_idx = np.where(delta_vals > 0.0)[0]
        for i in buy_idx:
            if px[i] <= 0.0:
                blocked[i] = 1
                self.block_tracker.update_with_info_gain(i, blocked=True)
                continue
            buy_val = float(delta_vals[i])
            if buy_val < self.min_trade_value:
                blocked[i] = 1
                self.block_tracker.update_with_info_gain(i, blocked=True)
                continue

            fee = buy_val * self.transaction_fee + self.fixed_fee
            cost = buy_val + fee

            if cost > self.portfolio_cash:
                # partial fill
                cost = float(self.portfolio_cash)
                # solve for buy_val such that buy_val + buy_val*fee_rate + fixed_fee = cash
                # approximate by removing fixed fee first
                available = max(0.0, cost - self.fixed_fee)
                buy_val = available / (1.0 + self.transaction_fee) if (1.0 + self.transaction_fee) > 0 else 0.0
                fee = buy_val * self.transaction_fee + self.fixed_fee
                cost = buy_val + fee
                if buy_val < self.min_trade_value or buy_val <= 1e-12:
                    blocked[i] = 1
                    self.block_tracker.update_with_info_gain(i, blocked=True)
                    continue

            buy_qty = buy_val / float(px[i])
            if buy_qty <= 1e-18:
                blocked[i] = 1
                self.block_tracker.update_with_info_gain(i, blocked=True)
                continue

            self.portfolio_qty[i] += buy_qty
            self.portfolio_cash -= cost
            self.balance = self.portfolio_cash

            tx_fee_paid[i] = fee
            qty_delta[i] = buy_qty
            notional[i] = buy_val
            blocked[i] = 0
            self.block_tracker.update_with_info_gain(i, blocked=False)

            self.live_pnl.process_trade(self.selected_assets[i], volume=buy_qty, price=float(px[i]), trade_type="buy")

        # ---------------------------------------------------------------
        # Hard safety clamps: NO leverage / NO shorting
        # ---------------------------------------------------------------
        # Numerical drift can make cash slightly negative; clamp to 0.
        if not np.isfinite(self.portfolio_cash) or self.portfolio_cash < 0.0:
            self.portfolio_cash = max(0.0, float(np.nan_to_num(self.portfolio_cash, nan=0.0, posinf=0.0, neginf=0.0)))
        # Holdings must never be negative.
        self.portfolio_qty = np.maximum(self.portfolio_qty, 0.0)
        self.balance = float(self.portfolio_cash)

        return blocked, tx_fee_paid, qty_delta, notional, np.asarray(actions_effective, dtype=np.float64), tgt_w

    # ------------------------------------------------------------------
    # Recording / Reward
    # ------------------------------------------------------------------
    def record_historical_data(
        self,
        *,
        timestamp,
        prices: np.ndarray,
        actions_raw: np.ndarray,
        actions_effective: np.ndarray,
        target_weights: np.ndarray,
        qty_delta: np.ndarray,
        notional: np.ndarray,
        tx_fee_paid: np.ndarray,
        blocked: np.ndarray,
        reward: float,
        ret: float,
        log_ret: float,
        pv: float,
        pv_next: float,
    ) -> None:
        ts = pd.to_datetime(timestamp)
        px = np.asarray(prices, dtype=np.float64)
        w = self._weights_at(self.current_step)

        blocked_prob = self.block_tracker.posterior_mean()

        # global row
        pf_feats = self.risk_engine.get_latest_features()
        global_row = {
            "portfolio_value": float(pv),
            "cash": float(self.portfolio_cash),
            "return": float(ret),
            # "log_return": float(log_ret),
            # "reward": float(reward),
            "total_tx_fee": float(self.total_transaction_fee),
        }
        global_row.update({k: float(pf_feats.get(k, 0.0)) for k in self._pf_fields})

        # WIDE
        wide = dict(global_row)
        for i, a in enumerate(self.selected_assets):
            prefix = f"{a}__"
            pnl = self.live_pnl.get_pnl(a, current_price=float(px[i]))
            fields = {
                "action_raw": float(actions_raw[i]) if actions_raw.size == self.n_assets else float(actions_raw[i]) if i < actions_raw.size else 0.0,
                "action_effective": float(actions_effective[i]),
                "target_weight": float(target_weights[i]),
                "weight": float(w[i]),
                "qty": float(self.portfolio_qty[i]),
                "qty_delta": float(qty_delta[i]),
                "price": float(px[i] / pv),
                "notional": float(notional[i]),
                "tx_fee_paid": float(tx_fee_paid[i] / pv),
                "blocked": int(blocked[i]),
                "blocked_prob": float(blocked_prob[i]),
                "fifo_realized": float(pnl.get("fifo_realized", 0.0) / pv),
                "fifo_unrealized": float(pnl.get("fifo_unrealized", 0.0) / pv),
                "lifo_realized": float(pnl.get("lifo_realized", 0.0) / pv),
                "lifo_unrealized": float(pnl.get("lifo_unrealized", 0.0) / pv),
                "fifo_total": float(pnl.get("fifo_total", 0.0) / pv),
                "lifo_total": float(pnl.get("lifo_total", 0.0) / pv),
            }
            for k, v in fields.items():
                wide[prefix + k] = v

        self.historical_trades.loc[ts, list(wide.keys())] = list(wide.values())

        # LONG
        long_rows = []
        for i, a in enumerate(self.selected_assets):
            pnl = self.live_pnl.get_pnl(a, current_price=float(px[i]))
            long_rows.append({
                "timestamp": ts,
                "tic": a,
                "action_raw": float(actions_raw[i]) if actions_raw.size >= self.n_assets else 0.0,
                "action_effective": float(actions_effective[i]),
                "target_weight": float(target_weights[i]),
                "weight": float(w[i]),
                "qty": float(self.portfolio_qty[i]),
                "qty_delta": float(qty_delta[i]),
                "price": float(px[i] / pv),
                "notional": float(notional[i]),
                "tx_fee_paid": float(tx_fee_paid[i] / pv),
                "blocked": int(blocked[i]),
                "blocked_prob": float(blocked_prob[i]),
                "fifo_realized": float(pnl.get("fifo_realized", 0.0) / pv),
                "fifo_unrealized": float(pnl.get("fifo_unrealized", 0.0) / pv),
                "lifo_realized": float(pnl.get("lifo_realized", 0.0) / pv),
                "lifo_unrealized": float(pnl.get("lifo_unrealized", 0.0) / pv),
                "fifo_total": float(pnl.get("fifo_total", 0.0) / pv),
                "lifo_total": float(pnl.get("lifo_total", 0.0) / pv),
                "portfolio_value": float(pv / self.initial_balance),
                "cash": float(self.portfolio_cash / pv),
                "return": float(ret),
                # "log_return": float(log_ret),
                # "reward": float(reward),
                "total_tx_fee": float(self.total_transaction_fee / pv),
                **{k: float(pf_feats.get(k, 0.0)) for k in self._pf_fields},
            })
        df_long = pd.DataFrame(long_rows).set_index(["timestamp", "tic"]).sort_index()
        if self.historical_trades_long.empty:
            self.historical_trades_long = df_long
        else:
            self.historical_trades_long = pd.concat([self.historical_trades_long, df_long], axis=0)

    def _compute_reward(
        self,
        *,
        log_ret: float,
        ret: float,
        pf_feats: Dict[str, float],
        blocked_vec: np.ndarray,
        tx_fee_vec: np.ndarray,
        pv_prev: float,
    ) -> float:
        w = self.reward_weights

        # core return term
        reward = float(w.get("log_return", 1.0)) * float(log_ret)
        # Robustify: bound extremely large PF metrics so they don't dominate learning.
        pf_c = float(self.pf_clip) if (self.pf_clip is not None and self.pf_clip > 0) else 0.0
        def _pf(x: float) -> float:
            x = float(x)
            if not np.isfinite(x):
                return 0.0
            if pf_c > 0.0:
                return float(pf_c * np.tanh(x / pf_c))
            return x

        # risk terms # Original.
        '''reward += float(w.get("pf_sharpe", 0.0)) * _pf(pf_feats.get("pf_sharpe", 0.0))
        reward += float(w.get("pf_sortino", 0.0)) * _pf(pf_feats.get("pf_sortino", 0.0))
        reward += float(w.get("alpha_monthly", 0.0)) * _pf(pf_feats.get("pf_alpha_monthly", 0.0))
        reward += float(w.get("vol_monthly", 0.0)) * (-abs(_pf(pf_feats.get("pf_vol_monthly", 0.0))))
        reward += float(w.get("pf_max_drawdown", 0.0)) * (-abs(_pf(pf_feats.get("pf_max_drawdown", 0.0))))
        reward += float(w.get("m_squared_ratio", 0.0)) * _pf(pf_feats.get("pf_m_squared_ratio", 0.0))
        reward += float(w.get("pf_beta", 0.0)) * (-abs(_pf(pf_feats.get("pf_beta", 0.0))))
        reward += float(w.get("beta_adj_sharpe", 0.0)) * _pf(pf_feats.get("pf_beta_adj_sharpe", 0.0))
        reward += float(w.get("pf_cvar", 0.0)) * _pf(pf_feats.get("pf_cvar", 0.0))'''

        # risk terms
        '''reward += float(w.get("pf_sharpe", 0.0)) * pf_feats.get("pf_sharpe", 0.0)
        reward += float(w.get("pf_sortino", 0.0)) * pf_feats.get("pf_sortino", 0.0)
        reward += float(w.get("alpha_monthly", 0.0)) * pf_feats.get("pf_alpha_monthly", 0.0)'''
        reward += float(w.get("vol_monthly", 0.0)) * (-abs(pf_feats.get("pf_vol_monthly", 0.0)))
        '''reward += float(w.get("pf_max_drawdown", 0.0)) * (-abs(pf_feats.get("pf_max_drawdown", 0.0)))
        reward += float(w.get("m_squared_ratio", 0.0)) * pf_feats.get("pf_m_squared_ratio", 0.0)
        reward += float(w.get("pf_beta", 0.0)) * (-abs(pf_feats.get("pf_beta", 0.0)))
        reward += float(w.get("beta_adj_sharpe", 0.0)) * pf_feats.get("pf_beta_adj_sharpe", 0.0)
        reward += float(w.get("pf_cvar", 0.0)) * pf_feats.get("pf_cvar", 0.0)'''

        # penalties
        blocked_cnt = float(np.sum(blocked_vec))
        if self.normalize_blocked:
            blocked_cnt = blocked_cnt / max(1.0, float(self.n_assets))
        reward -= float(w.get("blocked_actions_w", 0.0)) * blocked_cnt

        tx_fee = float(np.sum(tx_fee_vec))
        if self.normalize_tx_fee:
            tx_fee = tx_fee / max(1e-12, float(pv_prev))
        reward -= float(w.get("tx_fee_w", 0.0)) * tx_fee

        '''# prospect theory loss aversion
        alpha = float(w.get("prospect_theory_loss_aversion_alpha", 2.0))
        lam = float(w.get("prospect_theory_loss_aversion_w", 0.5))
        if ret < 0.0:
            reward -= lam * (abs(ret) ** alpha)'''

        if not np.isfinite(reward):
            print(f'reward is infinite: {reward}')
            reward = 0.0

        # (Optional) reward clip to keep targets in a learnable scale
        # if self.reward_clip is not None and self.reward_clip > 0:
        #     rc = float(self.reward_clip)
        #     reward = float(np.clip(reward, -rc, rc))

        return float(reward)

    # ------------------------------------------------------------------
    # Rendering (optional)
    # ------------------------------------------------------------------
    def render(self, mode="human"):
        t = int(self.current_step)
        ts = self.timestamps[t]
        pv = self._portfolio_value_at(t)
        print(f"[{self.env_name}] t={t} ts={ts} PV={pv:.2f} Cash={self.portfolio_cash:.2f}")

