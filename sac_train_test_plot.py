#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sac_train_test_plot.py

Terminal plotting harness for SAC (plotext), matching the behavior you built for PPO.

Usage:
- Construct SAC agent (from sac.py)
- Call train_sac(agent, plot_freq=...) to train and plot.

This does not modify SAC internals; it just wraps `SAC.run()`.
"""

from __future__ import annotations

import plotext as plt


def _safe_clear():
    try:
        plt.clf()
    except Exception:
        try:
            plt.clear_terminal()
        except Exception:
            pass


def plot_results(train_rewards, train_lengths, test_rewards, test_lengths):
    if not train_rewards or not test_rewards:
        return

    _safe_clear()
    plt.plot(train_rewards, label="Training EpRet")
    plt.title("SAC Training Episode Returns")
    plt.xlabel("Episodes")
    plt.ylabel("Return")
    plt.show()

    _safe_clear()
    plt.plot(test_rewards, label="Testing EpRet")
    plt.title("SAC Testing Episode Returns")
    plt.xlabel("Epoch")
    plt.ylabel("Return")
    plt.show()


def train_sac(sac_agent, plot_freq: int = 1):
    """Train SAC and periodically plot."""

    train_ep_returns, train_ep_lengths, test_ep_returns, test_ep_lengths = sac_agent.run(plot_freq=0)

    # If user wants plots, do one combined plot at end (and optionally intermediate via your run script)
    if plot_freq is not None and int(plot_freq) > 0:
        plot_results(train_ep_returns, train_ep_lengths, test_ep_returns, test_ep_lengths)

    return train_ep_returns, train_ep_lengths, test_ep_returns, test_ep_lengths

