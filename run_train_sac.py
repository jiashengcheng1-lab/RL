#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_train_sac.py

End-to-end SAC training runner for your LONG multi-index env.

- Loads local LONG dataframe
- Runs env_starter.neutralization
- Prefills replay buffer + running action scaler
- Trains SAC
- Optional plotext plots via sac_train_test_plot.py

Example:
    python run_train_sac.py --data /mnt/data/dataset.csv --epochs 5 --steps-per-epoch 2000
"""

from __future__ import annotations

import argparse
import os

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="/mnt/data/dataset.csv")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--steps-per-epoch", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-2d", action="store_true")
    p.add_argument("--max-ep-len", type=int, default=2000)
    p.add_argument("--results-dir", type=str, default="./results/sac_run")
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def load_df(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    from env_starter import neutralization, prefill_replay_buffer_and_scalers
    from CryptoTradingEnv import CryptoTradingEnv
    from ReplayBuffers import ReplayBuffer
    from sac import SAC

    from sac_train_test_plot import train_sac

    df_raw = load_df(args.data)

    train_df, val_df, *_ = neutralization(df_raw, start_step_train=0, train_ratio=0.7)

    env = CryptoTradingEnv(
        data_long=train_df,
        train=True,
        use_2d=bool(args.use_2d)
    )
    test_env = CryptoTradingEnv(
        data_long=val_df,
        train=False,
        use_2d=bool(args.use_2d)
    )

    replay_buffer = ReplayBuffer(
        obs_dim=env.observation_space.shape,
        act_dim=int(env.action_space.shape[0]),
        size=int(1e6),
    )

    sac_agent = SAC(
        state_dim=env.observation_space.shape,
        action_dim=int(env.action_space.shape[0]),
        env=env,
        test_env=test_env,
        replay_buffer=replay_buffer,
        seed=args.seed,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
    )

    prefill_replay_buffer_and_scalers(
        env=env,
        test_env=test_env,
        replay_buffer=replay_buffer,
        replay_buffer_path=os.path.join(args.results_dir, "sac_replay_buffer.pkl"),
        scaler_path=os.path.join(args.results_dir, "sac_obs_scaler.pkl"),
        running_scaler_path=os.path.join(args.results_dir, "sac_running_action_scaler.pkl"),
        replay_buffer_size=int(1e6),
        device=None,
        logger=None,
        max_ep_len=args.max_ep_len,
        is_ppo=False,
    )

    if args.plot:
        train_sac(sac_agent, plot_freq=1)
    else:
        sac_agent.run(plot_freq=0)


if __name__ == "__main__":
    main()

