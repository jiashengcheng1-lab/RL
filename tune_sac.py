#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tune_sac.py

Lightweight tuning scaffold for SAC.

This runs sequentially and saves a CSV with trial metrics.
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Dict, List

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="/mnt/data/dataset.csv")
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--steps-per-epoch", type=int, default=2000)
    p.add_argument("--results-dir", type=str, default="./results/sac_tune")
    p.add_argument("--use-2d", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_df(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def sample_config(rng: random.Random) -> Dict:
    return {
        "gamma": rng.choice([0.95, 0.99]),
        "polyak": rng.choice([0.995, 0.99]),
        "lr": 10 ** rng.uniform(-4.5, -3.0),
        "alpha": rng.choice([0.1, 0.2, 0.05]),
        "batch_size": rng.choice([128, 256]),
        "start_steps": rng.choice([1000, 5000]),
        "update_after": rng.choice([1000, 2000]),
        "update_every": rng.choice([50, 100]),
        "hidden_sizes": rng.choice([(256, 256), (512, 256)]),
    }


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    from env_starter import neutralization, prefill_replay_buffer_and_scalers
    from CryptoTradingEnv import CryptoTradingEnv
    from ReplayBuffers import ReplayBuffer
    from sac import SAC

    df_raw = load_df(args.data)
    train_df, val_df, *_ = neutralization(df_raw, start_step_train=0, train_ratio=0.7)

    rng = random.Random(args.seed)
    results: List[Dict] = []

    for trial in range(int(args.trials)):
        cfg = sample_config(rng)

        env = CryptoTradingEnv(train_df, train=True, use_2d=bool(args.use_2d))
        test_env = CryptoTradingEnv(val_df, train=False, use_2d=bool(args.use_2d))

        rb = ReplayBuffer(
            obs_dim=env.observation_space.shape,
            act_dim=int(env.action_space.shape[0]),
            size=int(2e5)
        )

        agent = SAC(
            state_dim=env.observation_space.shape,
            action_dim=int(env.action_space.shape[0]),
            env=env,
            test_env=test_env,
            replay_buffer=rb,
            seed=args.seed + trial,
            steps_per_epoch=args.steps_per_epoch,
            epochs=args.epochs,
            **cfg,
        )

        prefill_replay_buffer_and_scalers(
            agent=agent,
            env=env,
            test_env=test_env,
            replay_buffer=rb,
            replay_buffer_path=os.path.join(args.results_dir, f"trial_{trial}", "replay.pkl"),
            scaler_path=os.path.join(args.results_dir, f"trial_{trial}", "obs_scaler.pkl"),
            running_scaler_path=os.path.join(args.results_dir, f"trial_{trial}", "running_action_scaler.pkl"),
            replay_buffer_size=int(2e5),
            device=None,
            logger=None,
            is_ppo=False,
        )

        train_ep_returns, train_ep_lengths, test_ep_returns, test_ep_lengths = agent.run(plot_freq=0)
        score = float(sum(test_ep_returns[-min(len(test_ep_returns), 3):]) / max(min(len(test_ep_returns), 3), 1)) if test_ep_returns else 0.0

        row = {"trial": trial, "score": score, "last_test_return": (test_ep_returns[-1] if test_ep_returns else 0.0), **cfg}
        results.append(row)
        pd.DataFrame(results).to_csv(os.path.join(args.results_dir, "sac_tune_results.csv"), index=False)

    print("Saved:", os.path.join(args.results_dir, "sac_tune_results.csv"))


if __name__ == "__main__":
    main()

