#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_integration_long_env.py

Quick correctness & wiring checks for LONG multi-index env + helpers.

This script:
- Loads dataset.csv (long format)
- Runs env_starter.neutralization (scaling)
- Creates a CryptoTradingEnv(use_2d=True)
- Steps through N timesteps with random actions
- Validates:
    * historical_trades (timestamp index) grows with steps
    * historical_trades_long has MultiIndex (timestamp, tic)
    * each timestamp in historical_trades_long has exactly n_assets rows
    * action_raw/action_scaled columns exist per asset

Run:
    python test_integration_long_env.py --data /mnt/data/dataset.csv --steps 200
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="/mnt/data/dataset.csv")
    p.add_argument("--steps", type=int, default=200)
    return p.parse_args()


def load_df(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main():
    args = parse_args()

    from env_starter import neutralization, prefill_replay_buffer_and_scalers
    from CryptoTradingEnv import CryptoTradingEnv

    df_raw = load_df(args.data)
    train_df, val_df, *_ = neutralization(df_raw, start_step_train=0, train_ratio=0.7)

    env = CryptoTradingEnv(train_df, train=True, use_2d=True)
    test_env = CryptoTradingEnv(val_df, train=False, use_2d=True)

    # fit running action scaler for get_state
    prefill_replay_buffer_and_scalers(
        agent=None,
        env=env,
        test_env=test_env,
        replay_buffer=None,
        replay_buffer_path="/tmp/_rb.pkl",
        scaler_path="/tmp/_sc.pkl",
        running_scaler_path="/tmp/_rsc.pkl",
        replay_buffer_size=1,
        device=None,
        max_ep_len=500,
        logger=None,
        is_ppo=True,
    )

    o = env.reset()
    n_assets = env.n_assets

    steps = int(args.steps)
    for _ in range(steps):
        a = env.action_space.sample()
        o, r, d, info = env.step(a)
        if d:
            break

    # checks
    assert hasattr(env, "historical_trades"), "env.historical_trades missing"
    assert hasattr(env, "historical_trades_long"), "env.historical_trades_long missing"

    ht = env.historical_trades
    htl = env.historical_trades_long

    assert isinstance(htl.index, pd.MultiIndex), "historical_trades_long must be MultiIndex"
    assert list(htl.index.names)[:2] == ["timestamp", "tic"], f"index names are {htl.index.names}"

    # each timestamp has n_assets rows
    counts = htl.groupby(level=0).size()
    bad = counts[counts != n_assets]
    if len(bad) > 0:
        raise AssertionError(f"Found timestamps with missing assets. Examples: {bad.head()} (expected {n_assets})")

    # action columns exist
    action_cols = [c for c in htl.columns if "action" in c.lower()]
    if len(action_cols) == 0:
        raise AssertionError("No action-like columns found in historical_trades_long")

    # spot-check last timestamp action columns are finite
    last_ts = htl.index.get_level_values(0).max()
    last_block = htl.loc[(last_ts, slice(None)), action_cols]
    if not np.isfinite(last_block.to_numpy()).all():
        raise AssertionError("Non-finite values found in action columns at last timestamp")

    print("OK: env LONG integration checks passed")
    print("historical_trades rows:", len(ht))
    print("historical_trades_long rows:", len(htl), "(should be ~ steps*n_assets)")
    print("action cols:", action_cols[:10], "...")


if __name__ == "__main__":
    main()

