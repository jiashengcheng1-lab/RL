#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_train_ppo.py

End-to-end PPO training runner for your LONG multi-index env.

- Loads local LONG dataframe (timestamp,tic) from CSV/parquet
- Runs env_starter.neutralization (train/val/test split + scaling)
- Prefills the env running action scaler (to scale actions inside get_state)
- Trains PPO with MPI (ppo_mpi.mpi_fork)
- Uses plotext terminal plots (ppo_train_test_plot.py)

Example:
    python run_train_ppo.py --data /mnt/data/dataset.csv --procs 4 --train-epochs 10 --test-episodes 2

Notes:
- MPI requires `mpirun`/OpenMPI installed.
- If --procs 1, it runs in a single process.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

from ppo_mpi import mpi_fork, proc_id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="/mnt/data/dataset.csv", help="Path to LONG dataframe CSV/parquet")
    p.add_argument("--procs", type=int, default=1, help="Number of MPI processes")
    p.add_argument("--train-epochs", type=int, default=10, help="Number of training epochs")
    p.add_argument("--steps-per-epoch", type=int, default=2048, help="Environment steps per epoch (global)")
    p.add_argument("--test-episodes", type=int, default=2, help="Test episodes per epoch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-2d", action="store_true", help="Use 2D obs (assets×features). Recommended.")
    p.add_argument("--max-ep-len", type=int, default=2000)
    p.add_argument("--horizon", type=int, default=512)
    p.add_argument("--use-action-norm", action="store_true", help="Env will softmax-normalize actions internally")
    p.add_argument("--is-td3-softmax", action="store_true", help="Env expects cash as extra asset (n_assets+1) weights")
    p.add_argument("--results-dir", type=str, default="./results/ppo_run")
    return p.parse_args()


def load_df(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main():
    args = parse_args()

    # MPI fork must happen before heavy imports/allocations
    if args.procs and args.procs > 1:
        mpi_fork(args.procs)

    # Now in worker process context
    os.makedirs(args.results_dir, exist_ok=True)

    # local imports (after mpi_fork)
    from env_starter import neutralization, prefill_replay_buffer_and_scalers
    from CryptoTradingEnv import CryptoTradingEnv

    from ppo import PPOAgent
    from ppo_train_test_plot import train_ppo

    df_raw = load_df(args.data)

    # Split + scale (no look-ahead: scaler fit on train then applied to validation)
    # neutralization returns 6 items; we only need the scaled LONG frames.
    # train_df, val_df, *_ = neutralization(df_raw, start_step_train=0, train_ratio=0.7) # Original.
    train_df, val_df, train_price, val_price, scaler, test_scaler, test_df = neutralization(df_raw, start_step_train=0, train_ratio=0.7)
    print(
        f'train_df: {train_df} | val_df: {val_df}'
    )

    # envs
    env = CryptoTradingEnv(
        data_long=train_df,
        train=True,
        scaler=scaler,
        use_2d=bool(args.use_2d),
        use_action_norm=bool(args.use_action_norm),
        is_td3_softmax=bool(args.is_td3_softmax),
        horizon=args.horizon,
    )
    test_env = CryptoTradingEnv(
        data_long=val_df,
        train=False,
        scaler=test_scaler,
        use_2d=bool(args.use_2d),
        use_action_norm=bool(args.use_action_norm),
        is_td3_softmax=bool(args.is_td3_softmax),
        horizon=args.horizon,
    )

    print(f'env.max_ep_len: {env.max_ep_len}')
    # Build PPO agent
    # ppo_agent = PPOAgent(
    #     env=env,
    #     test_env=test_env,
    #     seed=args.seed,
    #     steps_per_epoch=args.steps_per_epoch,
    #     epochs=args.train_epochs,
    #     max_ep_len=int(env.max_ep_len), #  * proc_id()),
    #     logger_kwargs=dict(output_dir=args.results_dir, exp_name="ppo_long_env"),
    #     save_freq=50,
    #     is_hyper_tune=False,
    # )
    # print(f'ppo_agent.ac: {ppo_agent.ac}')

    # Prefill env running action scaler (used inside env.get_state to scale raw actions)
    running_scaler, scaler = prefill_replay_buffer_and_scalers(
        # agent=ppo_agent,
        env=env,
        test_env=test_env,
        replay_buffer=None,
        replay_buffer_path=os.path.join(args.results_dir, "ppo_prefill_buffer.pkl"),
        scaler_path=os.path.join(args.results_dir, "ppo_obs_scaler.pkl"),
        running_scaler_path=os.path.join(args.results_dir, "ppo_running_action_scaler.pkl"),
        replay_buffer_size=1,
        device=None,
        max_ep_len=int(env.max_ep_len), #  * proc_id()),
        logger=None,
        is_ppo=True,
    )

    test_env.running_scaler = running_scaler
    env.running_scaler = running_scaler

    # Re-Build PPO agent
    ppo_agent = PPOAgent(
        env=env,
        test_env=test_env,
        seed=args.seed,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.train_epochs,
        max_ep_len=int(env.max_ep_len), #  * proc_id()),
        logger_kwargs=dict(output_dir=args.results_dir, exp_name="ppo_long_env"),
        save_freq=50,
        is_hyper_tune=False,
    )
    print(f'ppo_agent.ac: {ppo_agent.ac}')

    if proc_id() == 0:
        print(f"[PPO] Starting training. data={args.data} use_2d={args.use_2d} procs={args.procs}")

    train_ppo(
        ppo_agent=ppo_agent,
        num_episodes_train=args.train_epochs,
        num_episodes_test=args.test_episodes,
        is_hyper_tune=False,
        plot_freq=5,
    )


if __name__ == "__main__":
    main()

