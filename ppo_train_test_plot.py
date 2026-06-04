#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ppo_train_test_plot.py

Terminal plotting + training harness for PPO.

This keeps your original plotting behavior (plotext) and training loop style,
while fixing:
- missing imports
- 2D observation support (env obs shape = (assets, features))
- device handling (delegated to actor/critic)
- MPI compatibility (still uses ppo_mpi.num_procs)

Primary entrypoints:
- train_ppo(ppo_agent, ...)
- test(ppo_agent, ...)

You can call these from `run_train_ppo.py`.
"""

from __future__ import annotations

import os
import time
from copy import deepcopy

import numpy as np
import torch

import plotext as plt
from tqdm import tqdm

from ppo_mpi import num_procs, proc_id


# -----------------------------------------------------------------------------
# Plotting utilities
# -----------------------------------------------------------------------------

def _safe_clear():
    try:
        plt.clf()
    except Exception:
        try:
            plt.clear_terminal()
        except Exception:
            pass


def plot_results(episode_rewards, episode_lengths, test_episode_rewards, test_episode_lengths):
    """Plot key training metrics using plotext in the terminal."""
    if not episode_rewards or not episode_lengths or not test_episode_rewards or not test_episode_lengths:
        return

    _safe_clear()
    plt.plot(episode_rewards, label="Training Rewards")
    plt.title("Training Episode Rewards over Time")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.show()

    _safe_clear()
    plt.plot(test_episode_rewards, label="Testing Rewards")
    plt.title("Testing Episode Rewards over Time")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.show()


def plot_training_results(results, metric="Portfolio Value"):
    """Plot a single list of scalars in terminal using plotext."""
    if results is None or len(results) == 0:
        return
    _safe_clear()
    plt.plot(results, label=str(metric))
    plt.title(str(metric))
    plt.xlabel("Episodes")
    plt.ylabel(str(metric))
    plt.show()



# -----------------------------------------------------------------------------
# Sharpe helpers (robust)
# -----------------------------------------------------------------------------

def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _compute_episode_sharpe_from_env(env) -> float:
    """Annualized Sharpe computed from env.returns (robust, no NaNs).

    Why this exists:
    - Your env ledger stores rolling portfolio metrics under keys like 'pf_sharpe'
      (from PortfolioRiskFeatureEngine), NOT 'annual_sharpe'.
    - The previous training loop attempted to read 'annual_sharpe', which does not
      exist, causing EpSharpe/TestEpSharpe to stay empty and therefore log as NaN.
    """
    # Prefer realized per-step returns from the env
    try:
        rets = np.asarray(getattr(env, "returns", []), dtype=np.float64).reshape(-1)
        rets = rets[np.isfinite(rets)]
    except Exception:
        rets = np.asarray([], dtype=np.float64)

    # Risk-free rate: use the same per-step rate the env passes to the risk engine
    rf = _safe_float(getattr(env, "pct_risk_free_rate", getattr(env, "risk_free_rate", 0.0)), 0.0)

    # Annualization factor (defaults to 365 to match PortfolioRiskFeatureEngine)
    ann = 365.0
    try:
        ann = float(getattr(getattr(env, "risk_engine", None), "annualization", 365.0))
        if not np.isfinite(ann) or ann <= 0:
            ann = 365.0
    except Exception:
        ann = 365.0

    if rets.size < 2:
        # Fall back to the risk engine rolling sharpe, if available
        try:
            feats = getattr(env, "risk_engine").get_latest_features()
            return _safe_float(feats.get("pf_sharpe", 0.0), 0.0)
        except Exception:
            return 0.0

    excess = rets - rf
    mu = float(np.mean(excess))
    sd = float(np.std(excess, ddof=1))
    if (not np.isfinite(mu)) or (not np.isfinite(sd)) or sd < 1e-12:
        return 0.0
    return float((mu / sd) * np.sqrt(ann))




# -----------------------------------------------------------------------------
# PPO train / test
# -----------------------------------------------------------------------------

def train_ppo(
    ppo_agent, num_episodes_train: int, num_episodes_test: int, is_hyper_tune: bool = False, plot_freq: int = 25
):
    """Train PPO using the agent + env already constructed."""

    ppo_agent.ac.train()
    ppo_agent.ac.pi.train()
    ppo_agent.ac.v.train()

    model_dir = "./results/training_model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "ppo_actor_critic.pth")

    # steps/epochs
    steps_per_epoch = int(2048 * num_procs())
    epochs = int(num_episodes_train)

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = ppo_agent.env.reset(), 0.0, 0

    total_steps = int(ppo_agent.local_steps_per_epoch * epochs)

    train_episode_rewards: list[float] = []
    train_episode_lengths: list[int] = []
    test_episode_rewards: list[float] = []
    test_episode_lengths: list[int] = []

    train_episode_portfolio_values: list[float] = []
    test_episode_portfolio_values: list[float] = []

    train_sharpe_ratios: list[float] = []
    test_sharpe_ratios: list[float] = []

    progress_bar = tqdm(total=total_steps, desc="PPO Training Progress")

    max_ep_len = int(ppo_agent.max_ep_len)

    for epoch in range(epochs):
        for t in range(ppo_agent.local_steps_per_epoch):
            a, v, logp = ppo_agent.ac.step(o, determinstic=False)
            next_o, r, d, info = ppo_agent.env.step(a)

            ep_ret += float(r)
            ep_len += 1

            # save and log
            ppo_agent.remember(
                state=o,
                action=a,
                reward=float(r),
                new_state=next_o,
                done=bool(d),
                value=v,
                log_probs=logp,
            )
            ppo_agent.logger.store(VVals=float(np.asarray(v).reshape(-1)[0]))

            o = next_o

            # treat "ran out of data" as truncation/time-limit (bootstrap!)
            truncated = bool(
                (info or {}).get("truncated", False)
                or (info or {}).get("TimeLimit.truncated", False)
                or (info or {}).get("reason", "") in ("end_of_data", "eod", "eod_window")
            )

            timeout = (ep_len == max_ep_len) or truncated
            terminal = bool(d) or timeout
            # timeout = ep_len == max_ep_len
            # terminal = bool(d) or timeout
            epoch_ended = t == ppo_agent.local_steps_per_epoch - 1

            # -----------------------------------------------------------------
            # "Real episodes" behavior:
            # - We always cut the *trajectory buffer* at epoch end (so PPO can
            #   compute advantages/returns for the batch), with bootstrapping.
            # - We only reset the *environment* when an episode truly ends
            #   (done or timeout). This allows episodes to span epochs.
            # -----------------------------------------------------------------
            if terminal or epoch_ended:
                # bootstrap value if needed (timeout or epoch cut)
                if timeout or epoch_ended:
                    _, v_boot, _ = ppo_agent.ac.step(o, determinstic=False)
                    v_boot = float(np.asarray(v_boot).reshape(-1)[0])
                else:
                    v_boot = 0.0

                ppo_agent.replay_buffer.finish_path(v_boot)

                if terminal:
                    ppo_agent.logger.store(EpRet=ep_ret, EpLen=ep_len)
                    train_episode_rewards.append(float(ep_ret))
                    train_episode_lengths.append(int(ep_len))

                    # portfolio stats (best effort)
                    try:
                        train_episode_portfolio_values.append(float(np.mean(ppo_agent.env.portfolio_values)))
                    except Exception:
                        pass
                    try:
                        sharpe = float(np.mean(ppo_agent.env.historical_trades.iloc[ppo_agent.env.start_step + 1 :]["annual_sharpe"]))
                        ppo_agent.logger.store(EpSharpe=sharpe)
                        train_sharpe_ratios.append(sharpe)
                    except Exception:
                        pass

                    sharpe = _compute_episode_sharpe_from_env(ppo_agent.env)
                    ppo_agent.logger.store(EpSharpe=sharpe)
                    train_sharpe_ratios.append(sharpe)

                    # Reset env + episode counters ONLY when terminal
                    o, ep_ret, ep_len = ppo_agent.env.reset(), 0.0, 0

                # export train ledger at the end
                if epoch == epochs - 1:
                    try:
                        train_csv = deepcopy(ppo_agent.env.historical_trades)
                        if "timestamp" not in train_csv.index.names:
                            train_csv.index.name = "timestamp"
                        train_csv.reset_index(inplace=True)
                        train_csv.to_csv("ppo_cleaned_train.csv", index=False)
                    except Exception:
                        pass

            # progress
            progress_bar.set_postfix(
                {
                    "Epoch": epoch,
                    "r": float(r),
                    "EpRet": float(ep_ret),
                    "AvgTrainEpRet": float(np.mean(train_episode_rewards)) if train_episode_rewards else 0.0,
                    "AvgTestEpRet": float(np.mean(test_episode_rewards)) if test_episode_rewards else 0.0,
                    "t": t,
                    "ep_len": ep_len,
                }
            )
            progress_bar.update(1)

        # Update policy/value
        ppo_agent.update(step=epoch)

        # Save model periodically (rank 0 only)
        if (epoch % ppo_agent.save_freq == 0 or epoch == epochs - 1) and (epoch != 0) and (proc_id() == 0):
            if not is_hyper_tune:
                try:
                    if os.path.exists(model_path):
                        torch.save(ppo_agent.ac.state_dict(), model_path)
                    else:
                        torch.save(ppo_agent.ac.state_dict(), model_path)
                except Exception:
                    pass

        # Evaluate
        avg_test_reward, max_test_reward, avg_test_lengths, max_test_lengths, te_rewards, te_lens, te_pv, te_sharpes, test_csv = test(agent=ppo_agent, num_episodes=num_episodes_test)
        test_episode_rewards.extend(te_rewards)
        test_episode_lengths.extend(te_lens)
        test_episode_portfolio_values.extend(te_pv)
        test_sharpe_ratios.extend(te_sharpes)

        if (epoch % plot_freq == 0) and (epoch != 0):
            plot_training_results(results=train_episode_portfolio_values, metric="Training Portfolio Value")
            plot_results(train_episode_rewards, train_episode_lengths, test_episode_rewards, test_episode_lengths)
            plot_training_results(results=test_episode_portfolio_values, metric="Testing Portfolio Value")

        # epoch_fallback_metrics:
        # If no *terminal* episode finished during this epoch (common in finance / long horizons),
        # EpochLogger would otherwise print NaN for EpRet/EpLen/EpSharpe.
        # We log the in-progress episode snapshot so dashboards stay informative.
        if len(ppo_agent.logger.epoch_dict.get("EpRet", [])) == 0:
            ppo_agent.logger.store(
                EpRet=float(ep_ret),
                EpLen=int(ep_len),
                EpSharpe=_compute_episode_sharpe_from_env(ppo_agent.env),
            )


        # Log epoch summary
        ppo_agent.logger.log_tabular("Epoch", epoch)
        ppo_agent.logger.log_tabular("EpRet", with_min_and_max=True)
        ppo_agent.logger.log_tabular("EpLen", average_only=True)
        ppo_agent.logger.log_tabular("TestEpRet", with_min_and_max=True)
        ppo_agent.logger.log_tabular("TestEpLen", average_only=True)
        ppo_agent.logger.log_tabular("EpSharpe", average_only=True)
        ppo_agent.logger.log_tabular("TestEpSharpe", average_only=True)
        ppo_agent.logger.log_tabular("VVals", with_min_and_max=True)
        ppo_agent.logger.log_tabular("TotalEnvInteracts", (epoch + 1) * steps_per_epoch)
        ppo_agent.logger.log_tabular("LossPi", average_only=True)
        ppo_agent.logger.log_tabular("PolicyLoss", average_only=True)
        ppo_agent.logger.log_tabular("ValueLoss", average_only=True)
        ppo_agent.logger.log_tabular("DeltaLossV", average_only=True)
        ppo_agent.logger.log_tabular("Entropy", average_only=True)
        ppo_agent.logger.log_tabular("KL", average_only=True)
        ppo_agent.logger.log_tabular("ClipFrac", average_only=True)
        ppo_agent.logger.log_tabular("StopIter", average_only=True)
        ppo_agent.logger.log_tabular("ExplainedVariance", average_only=True)
        ppo_agent.logger.log_tabular("PolicyLR", average_only=True)
        ppo_agent.logger.log_tabular("Time", time.time() - start_time)
        ppo_agent.logger.dump_tabular()

    progress_bar.close()

    try:
        test_csv.to_csv("ppo_test_historical_trades.csv")
    except Exception:
        pass

    return (
        float(np.mean(test_episode_rewards)) if test_episode_rewards else 0.0,
        float(np.max(test_episode_rewards)) if test_episode_rewards else 0.0,
        float(np.mean(test_episode_lengths)) if test_episode_lengths else 0.0,
        float(np.max(test_episode_lengths)) if test_episode_lengths else 0.0,
        test_episode_rewards,
        test_episode_lengths,
        test_episode_portfolio_values,
        [],
        test_csv,
    )


@torch.no_grad()
def test(agent, num_episodes: int):
    """Test the agent in the test environment using deterministic actions."""

    agent.ac.eval()
    agent.ac.v.eval()
    agent.ac.pi.eval()

    test_episode_rewards: list[float] = []
    test_episode_lengths: list[int] = []
    test_episode_portfolio_values: list[float] = []
    test_sharpe_ratios: list[float] = []

    for _ in range(int(num_episodes)):
        o, d, ep_ret, ep_len = agent.test_env.reset(), False, 0.0, 0

        while not (d or (agent.test_env.current_step >= agent.test_env.end_step - 1)):
            a = agent.ac.act(o, determinstic=True)
            o, r, d, _ = agent.test_env.step(a)
            ep_ret += float(r)
            ep_len += 1

        agent.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
        test_episode_rewards.append(float(ep_ret))
        test_episode_lengths.append(int(ep_len))
        sharpe = _compute_episode_sharpe_from_env(agent.test_env)
        agent.logger.store(TestEpSharpe=sharpe)
        test_sharpe_ratios.append(sharpe)

        try:
            sharpe = float(np.mean(agent.test_env.historical_trades.iloc[agent.test_env.start_step + 1 :]["annual_sharpe"]))
            agent.logger.store(TestEpSharpe=sharpe)
            test_sharpe_ratios.append(sharpe)
        except Exception:
            test_sharpe_ratios.append(0.0)

        try:
            test_episode_portfolio_values.append(float(np.mean(agent.test_env.portfolio_values)))
        except Exception:
            pass

    avg_test_reward = float(np.mean(test_episode_rewards)) if test_episode_rewards else 0.0
    max_test_reward = float(np.max(test_episode_rewards)) if test_episode_rewards else 0.0
    avg_test_lengths = float(np.mean(test_episode_lengths)) if test_episode_lengths else 0.0
    max_test_lengths = float(np.max(test_episode_lengths)) if test_episode_lengths else 0.0

    return (
        avg_test_reward,
        max_test_reward,
        avg_test_lengths,
        max_test_lengths,
        test_episode_rewards,
        test_episode_lengths,
        test_episode_portfolio_values,
        test_sharpe_ratios,
        agent.test_env.historical_trades,
    )

