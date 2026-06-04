#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
env_starter.py (LONG / MultiIndex drop-in)

See docstring in the previous version for details.
"""
from __future__ import annotations

import logging
import os
import pickle
from copy import deepcopy
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


# -------------------------------------------------------------------
# Pickle helpers
# -------------------------------------------------------------------
def save_pickle(obj, filename: str) -> None:
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filename: str):
    with open(filename, "rb") as f:
        return pickle.load(f)


def is_valid_pickle(filename: str) -> bool:
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        return False
    try:
        with open(filename, "rb") as f:
            pickle.load(f)
        return True
    except Exception:
        return False


# -------------------------------------------------------------------
# LONG data utilities
# -------------------------------------------------------------------
def _ensure_long_multiindex(df: pd.DataFrame, ts_col: str = "timestamp", tic_col: str = "tic") -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.index, pd.MultiIndex) and out.index.nlevels >= 2:
        names = list(out.index.names)
        if ts_col in names and tic_col in names:
            if names[:2] != [ts_col, tic_col]:
                out = out.reorder_levels([ts_col, tic_col] + [n for n in names if n not in (ts_col, tic_col)])
        else:
            out = out.reset_index()
    if not (isinstance(out.index, pd.MultiIndex) and out.index.nlevels >= 2 and out.index.names[:2] == [ts_col, tic_col]):
        if ts_col not in out.columns or tic_col not in out.columns:
            raise ValueError("Data must be LONG with ['timestamp','tic'] columns or MultiIndex(timestamp,tic).")
        out[ts_col] = pd.to_datetime(out[ts_col])
        out = out.set_index([ts_col, tic_col], drop=True)
    out = out.sort_index(level=[0, 1])
    return out


def _pivot_close(df_long: pd.DataFrame, close_col: str = "Close") -> pd.DataFrame:
    out = df_long.reset_index().pivot(index="timestamp", columns="tic", values=close_col)
    return out.sort_index()


def neutralization_(
    data: Union[pd.DataFrame, str, None],
    start_step_train: int,
    train_ratio: float = 0.7,
):
    """
    Drop-in neutralization that outputs LONG scaled frames (MultiIndex timestamp,tic).

    Returns (same count/order as your current wide code):
        train_data_scaled,
        validation_data_scaled,
        train_price_data,
        validation_price_data,
        train_scaler,
        test_scaler
    """
    if data is None:
        default_path = "/mnt/data/dataset.csv"
        if os.path.exists(default_path):
            data = default_path
        else:
            raise ValueError("neutralization: data is None and default /mnt/data/dataset.csv not found.")

    if isinstance(data, str):
        if data.lower().endswith(".csv"):
            df = pd.read_csv(data)
        else:
            df = pd.read_parquet(data)
    else:
        df = data.copy()

    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    df_long = _ensure_long_multiindex(df, ts_col="timestamp", tic_col="tic")
    # ---------------------------------------------------------------
    # IMPORTANT (no leverage/shorting safety):
    # Never standardize the execution price used for valuation.
    # If prices are standardized they can go negative, making PV negative
    # even with non-negative holdings (appears like "leverage").
    #
    # Preserve a raw execution price column:
    #   Close_raw = original Close
    # and allow Close (and other features) to be scaled for observations.
    # ---------------------------------------------------------------
    if "Close" not in df_long.columns:
        raise ValueError("neutralization: LONG data must contain a 'Close' column.")
    if "Close_raw" not in df_long.columns:
        df_long["Close_raw"] = df_long["Close"].astype("float64")

    # df_long = df_long.groupby(level=1).ffill().groupby(level=1).bfill() # Original.
    df_long = df_long.groupby(level='tic').ffill()
    df_long = df_long.replace([np.inf, -np.inf], np.nan).fillna(value=0.0)

    timestamps = df_long.index.get_level_values(0).unique().sort_values()
    nT = len(timestamps)
    if nT < 20:
        raise ValueError("Dataset has too few timestamps to split.")

    start_step_train = int(max(0, start_step_train))
    if start_step_train >= nT - 5:
        raise ValueError("start_step_train too large for dataset length.")

    usable = timestamps[start_step_train:]
    nU = len(usable)

    train_end = int(max(2, np.floor(train_ratio * nU)))
    val_end = int(max(train_end + 2, min(nU, train_end + max(2, int(0.15 * nU)))))

    train_ts = usable[:train_end]
    val_ts = usable[train_end:val_end]
    test_ts = usable[val_end:]

    train_df = df_long.loc[(train_ts, slice(None)), :]
    val_df = df_long.loc[(val_ts, slice(None)), :]
    test_df = df_long.loc[(test_ts, slice(None)), :]


    # ---------------------------------------------------------------
    # Ensure SPLITS share a consistent asset universe (fixed n_assets).
    #
    # Why this is needed:
    # - If a ticker is present in train but absent in validation/test,
    #   env instances can end up with different n_assets, causing policy
    #   shape mismatches during evaluation (e.g. obs (25,F) vs expected (26,F)).
    #
    # What we do:
    # - Use TRAIN tickers as the canonical universe (keeps training action
    #   dimension unchanged), and reindex each split onto the full
    #   (timestamp × ticker) panel. Missing rows are safely filled.
    # ---------------------------------------------------------------
    asset_universe = train_df.index.get_level_values(1).unique().tolist()
    asset_universe = sorted(asset_universe)

    def _complete_panel(df_part: pd.DataFrame, ts_index, assets) -> pd.DataFrame:
        ts_index = pd.to_datetime(pd.Index(ts_index))
        idx_full = pd.MultiIndex.from_product([ts_index, assets], names=["timestamp", "tic"])
        out = df_part.reindex(idx_full)
        # Forward/back fill within each asset where possible; then zero-fill.
        out = out.groupby(level=1).ffill().groupby(level=1).bfill()
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    train_df = _complete_panel(train_df, train_ts, asset_universe)
    val_df = _complete_panel(val_df, val_ts, asset_universe)
    test_df = _complete_panel(test_df, test_ts, asset_universe)

    # Scale all feature columns EXCEPT raw execution price columns
    raw_cols = [c for c in train_df.columns if c.lower().endswith("_raw")]
    feature_cols = [c for c in train_df.columns if c not in raw_cols]
    X_train = train_df[feature_cols].to_numpy(dtype=np.float64)

    scaler = MinMaxScaler() # StandardScaler()
    scaler.fit(X_train)

    def transform(df_part: pd.DataFrame) -> pd.DataFrame:
        X = df_part[feature_cols].to_numpy(dtype=np.float64)
        Xs = scaler.transform(X)
        out = df_part.copy()
        out[feature_cols] = Xs.astype(np.float32)
        # keep *_raw columns unscaled
        for c in raw_cols:
            out[c] = out[c].astype(np.float32)
        return out

    train_scaled = transform(train_df)
    val_scaled = transform(val_df)
    _ = transform(test_df)  # computed but not returned to keep drop-in return order/count

    # Return raw prices for downstream plots/diagnostics
    train_price = _pivot_close(train_df, close_col="Close_raw")
    val_price = _pivot_close(val_df, close_col="Close_raw")

    test_scaler = deepcopy(scaler)
    # return train_scaled, val_scaled, train_price, val_price, scaler, test_scaler
    return train_df, val_df, train_price, val_price, scaler, test_scaler # Return UNSCALED frames


def neutralization(
    data: Union[pd.DataFrame, str, None],
    start_step_train: int,
    train_ratio: float = 0.7,
):
    """
    Drop-in neutralization that outputs LONG *UNSCALED* frames (MultiIndex timestamp,tic).

    Returns (same count/order as your current wide code):
        train_data_unscaled,
        validation_data_unscaled,
        train_price_data,
        validation_price_data,
        train_scaler,
        test_scaler
    """
    if data is None:
        default_path = "/mnt/data/dataset.csv"
        if os.path.exists(default_path):
            data = default_path
        else:
            raise ValueError("neutralization: data is None and default /mnt/data/dataset.csv not found.")

    if isinstance(data, str):
        if data.lower().endswith(".csv"):
            df = pd.read_csv(data)
        else:
            df = pd.read_parquet(data)
    else:
        df = data.copy()

    # Drop "Unnamed: 0" style columns
    drop_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    # Ensure LONG MultiIndex (timestamp, tic)
    df_long = _ensure_long_multiindex(df, ts_col="timestamp", tic_col="tic")

    # Preserve raw execution price for valuation (never scale this column)
    if "Close" not in df_long.columns:
        raise ValueError("neutralization: LONG data must contain a 'Close' column.")
    if "Close_raw" not in df_long.columns:
        df_long["Close_raw"] = df_long["Close"].astype("float64")

    # Fill missing within each asset (level=1 is 'tic' in your code)
    # df_long = df_long.groupby(level=1).ffill().groupby(level=1).bfill() # Original.
    df_long = df_long.groupby(level='tic').ffill()

    # Clean inf/nan
    df_long = df_long.replace([np.inf, -np.inf], np.nan)
    # avoid FutureWarning downcasting surprises
    df_long = df_long.infer_objects(copy=False).fillna(value=0.0)

    timestamps = df_long.index.get_level_values(0).unique().sort_values()
    nT = len(timestamps)
    if nT < 20:
        raise ValueError("Dataset has too few timestamps to split.")

    start_step_train = int(max(0, start_step_train))
    if start_step_train >= nT - 5:
        raise ValueError("start_step_train too large for dataset length.")

    usable = timestamps[start_step_train:]
    nU = len(usable)

    train_end = int(max(2, np.floor(train_ratio * nU)))
    val_end = int(max(train_end + 2, min(nU, train_end + max(2, int(0.15 * nU)))))

    train_ts = usable[:train_end]
    val_ts   = usable[train_end:val_end]
    test_ts  = usable[val_end:]

    train_df = df_long.loc[(train_ts, slice(None)), :]
    val_df   = df_long.loc[(val_ts, slice(None)), :]
    test_df  = df_long.loc[(test_ts, slice(None)), :]

    # ------------------------------------------------------------------
    # Enforce consistent asset universe across splits: use TRAIN tickers
    # ------------------------------------------------------------------
    asset_universe = sorted(train_df.index.get_level_values(1).unique().tolist())

    def _complete_panel(df_part: pd.DataFrame, ts_index, assets) -> pd.DataFrame:
        ts_index = pd.to_datetime(pd.Index(ts_index))
        idx_full = pd.MultiIndex.from_product([ts_index, assets], names=["timestamp", "tic"])
        out = df_part.reindex(idx_full)
        out = out.groupby(level=1).ffill().groupby(level=1).bfill()
        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.infer_objects(copy=False).fillna(0.0)
        return out

    train_df = _complete_panel(train_df, train_ts, asset_universe)
    val_df   = _complete_panel(val_df, val_ts, asset_universe)
    test_df  = _complete_panel(test_df, test_ts, asset_universe)  # kept for scaler-fit sanity

    # ------------------------------------------------------------------
    # Fit scalers (OPTIONAL future use) but DO NOT transform outputs.
    # ------------------------------------------------------------------
    raw_cols = [c for c in train_df.columns if str(c).lower().endswith("_raw")]
    feature_cols = [c for c in train_df.columns if c not in raw_cols]

    # Fit on TRAIN only (standard ML hygiene)
    X_train = train_df[feature_cols].to_numpy(dtype=np.float64, copy=False)
    train_scaler = MinMaxScaler() # StandardScaler()
    train_scaler.fit(X_train)

    # In your original code you deepcopy into test_scaler; keep that for drop-in compatibility
    test_scaler = deepcopy(train_scaler)

    # Price pivots for plotting/diagnostics (raw prices)
    train_price = _pivot_close(train_df, close_col="Close_raw")
    val_price   = _pivot_close(val_df, close_col="Close_raw")

    # Return UNSCALED frames
    return train_df, val_df, train_price, val_price, train_scaler, test_scaler


# -------------------------------------------------------------------
# Prefill replay buffer + scalers (2D-safe + OBS-based running_scaler)
# -------------------------------------------------------------------
def _obs_dim_from_space(space) -> int:
    shp = getattr(space, "shape", None)
    if shp is None:
        return 0
    dim = 1
    for s in shp:
        dim *= int(s)
    return int(dim)


def prefill_replay_buffer_and_scalers(
    # agent,
    env, test_env, replay_buffer, replay_buffer_path,
    scaler_path, running_scaler_path, replay_buffer_size, device,
    max_ep_len, logger=None, is_ppo=False,
):
    """
    Prefill with random actions to:
      1) optionally populate an off-policy replay buffer, AND
      2) fit env.running_scaler **on observation/state features** (NOT actions).

    The running scaler is used inside CryptoTradingEnv.get_state() to scale the
    per-asset state matrix (n_assets × feature_dim) row-wise.

    Notes
    -----
    - We fit on per-asset rows (feature_dim columns). This keeps the scaler
      independent of n_assets, and works for both 2D and flattened 1D obs.
    - We attach metadata to the scaler object so that stale action-based scalers
      from older runs are automatically rejected and rebuilt.
    """
    if env is None:
        env = test_env

    obs_dim = _obs_dim_from_space(env.observation_space)
    act_dim = int(env.action_space.shape[0])

    scaler_path = f"obs_dim_{obs_dim}_act_dim_{act_dim}_" + scaler_path
    running_scaler_path = f"obs_dim_{obs_dim}_act_dim_{act_dim}_" + running_scaler_path
    replay_buffer_path = f"obs_dim_{obs_dim}_act_dim_{act_dim}_" + replay_buffer_path

    scaler_exists = is_valid_pickle(scaler_path)
    running_scaler_exists = is_valid_pickle(running_scaler_path)

    # -------------------------------
    # helpers
    # -------------------------------
    def _get_raw_obs(default_obs=None):
        # Prefer env.get_state(apply_running_scaler=False) when available
        try:
            return env.get_state(apply_running_scaler=False)
        except Exception:
            return default_obs

    def _obs_to_rows(obs, n_assets: int):
        arr = np.asarray(obs)
        if arr.ndim == 3:
            # e.g., (L, N, F) -> rows (L*N, F)
            return arr.reshape(-1, arr.shape[-1]).astype(np.float64, copy=False)
        if arr.ndim == 2:
            return arr.astype(np.float64, copy=False)
        if arr.ndim == 1:
            if n_assets and (arr.size % int(n_assets) == 0):
                return arr.reshape(int(n_assets), -1).astype(np.float64, copy=False)
            return arr.reshape(1, -1).astype(np.float64, copy=False)
        return arr.reshape(1, -1).astype(np.float64, copy=False)

    def _is_obs_scaler(s):
        return (
            s is not None
            and getattr(s, "_fit_purpose", None) == "obs_per_asset_v1"
            and isinstance(getattr(s, "_feature_dim", None), (int, np.integer))
        )

    # infer expected feature_dim from one reset() observation
    try:
        o0 = env.reset()
    except Exception:
        o0 = None
    n_assets = int(getattr(env, "n_assets", 0) or 0)
    rows0 = _obs_to_rows(o0 if o0 is not None else np.zeros((1, 1), dtype=np.float64), n_assets)
    expected_feature_dim = int(rows0.shape[1])

    # -------------------------------
    # load scalers if valid
    # -------------------------------
    if scaler_exists and running_scaler_exists:
        try:
            scaler = load_pickle(scaler_path)
            running_scaler = load_pickle(running_scaler_path)

            # reject stale action-based scalers or dim-mismatched scalers
            if not _is_obs_scaler(running_scaler) or int(getattr(running_scaler, "_feature_dim", -1)) != expected_feature_dim:
                raise ValueError("Stale or incompatible running scaler on disk; rebuilding.")

            env.set_running_scaler(running_scaler)
            if test_env is not None:
                test_env.set_running_scaler(running_scaler)
            if logger:
                logger.info("Scalers loaded from disk (obs-based running scaler).")
            return (running_scaler, scaler) if is_ppo else (running_scaler, scaler, replay_buffer)
        except Exception as e:
            if logger:
                logger.info("Rebuilding scalers (reason: %s)", e)

    # -------------------------------
    # build / fit running obs scaler
    # -------------------------------
    # choose a sensible prefill length
    total_steps = int(max(1, int(max_ep_len) * 30))
    # if caller provided a large replay buffer size, cap prefill to it
    try:
        if replay_buffer_size is not None and int(replay_buffer_size) >= 1000:
            total_steps = int(min(total_steps, int(replay_buffer_size)))
    except Exception:
        pass

    # total_steps = np.clip(total_steps, 50000, 50000)
    pbar = tqdm(total=total_steps, desc="Prefill (fit obs scaler)") if tqdm is not None else None

    running_scaler = env.get_running_scaler()
    if running_scaler is None:
        running_scaler = MinMaxScaler() # StandardScaler()

    # Disable state scaling during prefill so collected observations (and any stored buffer)
    # remain consistently *raw*. We'll install the fitted scaler at the end.
    try:
        env.set_running_scaler(None)
        if test_env is not None:
            test_env.set_running_scaler(None)
    except Exception:
        pass

    # If available, do streaming updates (preferred).
    use_partial = hasattr(running_scaler, "partial_fit")

    # Reset (again) to start stepping
    o = env.reset()
    ep_len = 0
    steps = 0

    # warm-start with initial obs
    try:
        raw_o = _get_raw_obs(o)
        rows = _obs_to_rows(raw_o, n_assets)
        if use_partial:
            running_scaler.partial_fit(rows)
        else:
            obs_rows = [rows]
    except Exception:
        obs_rows = [] if not use_partial else None

    while steps < total_steps:
        a = env.action_space.sample()

        o2, r, d, info = env.step(a, is_env_action_sample=True)
        ep_len += 1
        steps += 1

        # optional replay buffer fill (off-policy agents)
        if replay_buffer is not None:
            try:
                replay_buffer.store(o, a, r, o2, d)
            except Exception:
                pass

        # accumulate obs rows for scaler fitting
        try:
            raw_o2 = _get_raw_obs(o2)
            rows = _obs_to_rows(raw_o2, n_assets)
            if use_partial:
                running_scaler.partial_fit(rows)
            else:
                obs_rows.append(rows)
        except Exception:
            pass

        o = o2
        if pbar is not None:
            pbar.update(1)

        if d or ep_len >= max_ep_len - 1:
            o = env.reset()
            ep_len = 0

    if pbar is not None:
        pbar.close()

    if not use_partial:
        if len(obs_rows) == 0:
            X = np.zeros((1, expected_feature_dim), dtype=np.float64)
        else:
            X = np.vstack(obs_rows).astype(np.float64, copy=False)
        running_scaler.fit(X)

    # attach metadata so we can validate on reload
    try:
        running_scaler._fit_purpose = "obs_per_asset_v1"
        running_scaler._feature_dim = int(expected_feature_dim)
    except Exception:
        pass

    env.set_running_scaler(running_scaler)
    if test_env is not None:
        test_env.set_running_scaler(running_scaler)

    scaler = env.get_scaler() if hasattr(env, "get_scaler") else None

    # Save
    try:
        save_pickle(scaler, scaler_path)
        save_pickle(running_scaler, running_scaler_path)
    except Exception as e:
        if logger:
            logger.warning("Failed saving scalers: %s", e)

    try:
        if replay_buffer is not None and hasattr(replay_buffer, "save"):
            replay_buffer.save(replay_buffer_path)
        else:
            save_pickle(replay_buffer, replay_buffer_path)
    except Exception:
        pass

    return (running_scaler, scaler) if is_ppo else (running_scaler, scaler, replay_buffer)

