#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tune_ppo.py

Lightweight hyper-parameter tuning scaffold for PPO.

- Runs sequentially (no MPI fork)
- Evaluates a small random/grid search
- Writes a CSV summary to results_dir

You can extend the search space as you like.
"""

from __future__ import annotations

import argparse
import itertools
import os
import random
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="/mnt/data/dataset.csv")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--epochs", type=int, default=5, help="Train epochs per trial")
    p.add_argument("--test-episodes", type=int, default=2)
    p.add_argument("--results-dir", type=str, default="./results/ppo_tune")
    p.add_argument("--use-2d", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_df(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def sample_config(rng: random.Random) -> Dict:
    return {
        "pi_lr": 10 ** rng.uniform(-4.5, -3.0),
        "vf_lr": 10 ** rng.uniform(-4.0, -3.0),
        "clip_ratio": rng.choice([0.1, 0.2, 0.3]),
        "train_pi_iters": rng.choice([40, 80]),
        "train_v_iters": rng.choice([40, 80]),
        "ent_coef": rng.choice([0.0, 0.001, 0.005]),
        "target_kl": rng.choice([0.01, 0.02, None]),
        "hidden_sizes": rng.choice([(256, 256), (512, 256)]),
    }


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    from env_starter import neutralization, prefill_replay_buffer_and_scalers
    from CryptoTradingEnv import CryptoTradingEnv
    from ppo import PPOAgent
    from ppo_train_test_plot import train_ppo

    df_raw = load_df(args.data)
    train_df, val_df, *_ = neutralization(df_raw, start_step_train=0, train_ratio=0.7)

    rng = random.Random(args.seed)
    results: List[Dict] = []

    for trial in range(int(args.trials)):
        cfg = sample_config(rng)

        env = CryptoTradingEnv(data_long=train_df, train=True, use_2d=bool(args.use_2d))
        test_env = CryptoTradingEnv(data_long=val_df, train=False, use_2d=bool(args.use_2d))

        agent = PPOAgent(
            env=env,
            test_env=test_env,
            seed=args.seed + trial,
            steps_per_epoch=2048,
            epochs=args.epochs,
            logger_kwargs=dict(output_dir=os.path.join(args.results_dir, f"trial_{trial}"), exp_name="ppo_tune"),
            save_freq=999999,
            is_hyper_tune=True,
            **cfg,
        )

        prefill_replay_buffer_and_scalers(
            env=env,
            test_env=test_env,
            replay_buffer=None,
            replay_buffer_path=os.path.join(args.results_dir, f"trial_{trial}", "prefill.pkl"),
            scaler_path=os.path.join(args.results_dir, f"trial_{trial}", "obs_scaler.pkl"),
            running_scaler_path=os.path.join(args.results_dir, f"trial_{trial}", "running_action_scaler.pkl"),
            replay_buffer_size=1,
            device=None,
            max_ep_len=2000,
            logger=None,
            is_ppo=True,
        )

        avg_test_reward, max_test_reward, avg_test_len, max_test_len, *_ = train_ppo(
            ppo_agent=agent,
            num_episodes_train=args.epochs,
            num_episodes_test=args.test_episodes,
            is_hyper_tune=True,
            plot_freq=999999,
        )

        row = {
            "trial": trial,
            "avg_test_reward": avg_test_reward,
            "max_test_reward": max_test_reward,
            "avg_test_len": avg_test_len,
            "max_test_len": max_test_len,
            **cfg,
        }
        results.append(row)

        pd.DataFrame(results).to_csv(os.path.join(args.results_dir, "ppo_tune_results.csv"), index=False)

    print("Saved:", os.path.join(args.results_dir, "ppo_tune_results.csv"))


if __name__ == "__main__":
    main()

