#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ppo.py

PPO agent wrapper (MPI-capable) compatible with your LONG multi-index env.

This is a *drop-in* replacement for your current PPOAgent file:
- Fixes missing imports
- Supports 2D observations (assets × features) from the env
  by flattening inside `ppo_core_.MLPActorCritic`.
- Keeps the public API used by `ppo_train_test_plot.py`:
    - ppo_agent.ac.step / act
    - ppo_agent.remember
    - ppo_agent.update
    - ppo_agent.replay_buffer
    - ppo_agent.logger

Action bounds:
- Uses a tanh-squashed Gaussian policy (log-prob corrected) so actions stay in
  env bounds (commonly [-1, 1]).

"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ExponentialLR, LambdaLR, ReduceLROnPlateau, CosineAnnealingLR

from logger import EpochLogger
from ppo_buffer import PPOBuffer
from ppo_core_ import MLPActorCritic
from ppo_mpi import (
    mpi_avg,
    mpi_avg_grads,
    num_procs,
    proc_id,
    setup_pytorch_for_mpi,
    sync_params,
)
from base_agent import Agent, GradCheckConfig

# from TransformerSquashedGaussianActorCritic import SetTransformerSquashedGaussianActorCritic


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """1 - Var[y-ypred] / Var[y]."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    var_y = np.var(y_true)
    if var_y < 1e-12:
        return 0.0
    return float(1.0 - np.var(y_true - y_pred) / (var_y + 1e-12))




class PPOAgent(Agent):
    def __init__(
        self,
        env,
        test_env=None,
        *,
        seed: int = 0,
        steps_per_epoch: int = 2048,
        epochs: int = 100,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        pi_lr: float = 3e-4,
        vf_lr: float = 1e-3,
        train_pi_iters: int = 80,
        train_v_iters: int = 80,
        lam: float = 0.97,
        max_ep_len: int = 2000,
        target_kl: Optional[float] = 0.01,
        hidden_sizes: Sequence[int] = (256, 256),
        logger_kwargs: Optional[Dict[str, Any]] = None,
        save_freq: int = 10,
        ent_coef: float = 0.0,
        clip_range_vf: Optional[float] = None,
        is_hyper_tune: bool = False,
        use_smoothed_kl: bool = False,
        kl_smoothing_coef: float = 0.2,
        shared_minibatch_shuffle: bool = True,
        shared_shuffle_seed: int = 12345,
        minibatch_size: int = 256,

        scheduler_type='cosine',
        exp_decay=0.99,      # γ = decay rate per update
        total_updates=100000,
        early_stop_backoff_patience=2,
        early_stop_backoff_factor=0.8,
        min_lr=1e-6,
        lr_recovery_patience=5,
        lr_recovery_factor=1.1,
    ):
        """Create PPO agent.

        Parameters are intentionally aligned with your prior file, but only the
        essentials are used.
        """

        # --- MPI setup ---
        setup_pytorch_for_mpi()

        self.env = env
        self.test_env = test_env
        self.is_hyper_tune = bool(is_hyper_tune)

        # seed per-rank
        seed = int(seed) + 10000 * proc_id()
        set_seed(seed)

        # logger
        logger_kwargs = logger_kwargs or {}
        self.logger = EpochLogger(**logger_kwargs)

        # dims
        obs_shape = tuple(getattr(env.observation_space, "shape", ()))
        if len(obs_shape) == 0:
            raise ValueError("env.observation_space.shape is required")
        act_dim = int(env.action_space.shape[0])

        # actor-critic
        self.ac = MLPActorCritic(
            observation_space=env.observation_space,
            action_space=env.action_space,
            hidden_sizes=hidden_sizes,
            activation=nn.Tanh,
            # SetTransformer knobs (safe defaults)
            use_set_transformer=None,
            st_d_model=128,
            st_n_blocks=2,
            st_n_heads=4,
            st_dropout=0.0,
            st_use_asset_id_embedding=True,
        ).to(DEVICE)
        sync_params(self.ac)  # sync across ranks

        # buffer stores obs in original env shape (2D-safe)
        self.replay_buffer = PPOBuffer(
            obs_dim=obs_shape,
            act_dim=act_dim,
            size=int(steps_per_epoch // max(num_procs(), 1)), # Original.
            # size=int(steps_per_epoch),
            gamma=gamma,
            lam=lam,
            is_hyper_tune=self.is_hyper_tune,
            device=DEVICE,
        )

        # optimizers
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=float(pi_lr))
        self.v_optimizer = Adam(self.ac.v.parameters(), lr=float(vf_lr))


        self.early_stop_backoff_patience = early_stop_backoff_patience
        self.early_stop_backoff_factor = early_stop_backoff_factor
        self.min_lr = min_lr
        self.lr_recovery_patience = lr_recovery_patience
        self.lr_recovery_factor = lr_recovery_factor
        self.clip_range_vf = clip_range_vf


        ################################

        # Adaptive LR state
        self.original_pi_lr = pi_lr
        self.early_stop_counter = 0
        self.no_early_stop_counter = 0

        # step scheduler: drop LR by half at ~66k & ~133k updates
        step_kwargs = dict(
            step_size=total_updates // 3,  # ~66666
            gamma=0.5
        )

        # cosine scheduler: smooth decay to 1% of initial_lr over full run
        cosine_kwargs = dict(
            T_max=total_updates,
            # eta_min=vf_lr * 0.01
            eta_min=vf_lr * 0.001
        )

        # Choose scheduler
        if scheduler_type == "linear":
            # Linear decay: lr = initial_lr * (1 - step/total_updates)
            lr_lambda = lambda step: max(0.0, 1.0 - step/total_updates)
            self.scheduler = LambdaLR(
                self.v_optimizer, lr_lambda=lr_lambda, # verbose=True
            )

        elif scheduler_type == "step":
            scheduler_kwargs = step_kwargs
            # Drops LR by gamma every step_size updates
            step_size = scheduler_kwargs.get("step_size", total_updates // 3)
            gamma     = scheduler_kwargs.get("gamma", 0.5)
            self.scheduler = StepLR(
                self.v_optimizer, step_size=step_size, gamma=gamma, verbose=True
            )

        elif scheduler_type == "cosine":
            scheduler_kwargs = cosine_kwargs
            # Cosine annealing from lr to eta_min
            T_max  = scheduler_kwargs.get("T_max", total_updates)
            eta_min = scheduler_kwargs.get("eta_min", 0)
            self.scheduler = CosineAnnealingLR(
                self.v_optimizer, T_max=T_max, eta_min=eta_min, verbose=True
            )

            self.pi_scheduler = CosineAnnealingLR(
                self.pi_optimizer, T_max=T_max, eta_min=float(pi_lr * 0.001), verbose=True
            )

        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")

        # base_agent config
        self.gradcheck_cfg = GradCheckConfig

        # training hyperparams
        self.steps_per_epoch = int(steps_per_epoch)
        self.local_steps_per_epoch = int(self.steps_per_epoch // max(num_procs(), 1)) # Original.
        # self.local_steps_per_epoch = int(self.steps_per_epoch) # * max(num_procs(), 1))
        self.epochs = int(epochs)
        self.gamma = float(gamma)
        self.clip_ratio = float(clip_ratio)
        self.train_pi_iters = int(train_pi_iters)
        self.train_v_iters = int(train_v_iters)
        self.max_ep_len = int(max_ep_len)
        self.target_kl = None if target_kl is None else float(target_kl)
        self.ent_coef = float(ent_coef)
        self.clip_range_vf = clip_range_vf if clip_range_vf is None else float(clip_range_vf)

        # KL smoothing / shuffle options (kept to match your previous behavior)
        self.use_smoothed_kl = bool(use_smoothed_kl)
        self.kl_smoothing_coef = float(kl_smoothing_coef)
        self.shared_minibatch_shuffle = bool(shared_minibatch_shuffle)
        self.shared_shuffle_seed = int(shared_shuffle_seed)
        self.minibatch_size = int(minibatch_size)
        # self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)

        self.save_freq = int(save_freq)
        self.logger.setup_pytorch_saver(self.ac)

    # ---------------------------------------------------------------------
    # Losses
    # ---------------------------------------------------------------------

    def compute_loss_pi(self, data: Dict[str, torch.Tensor], mb_idx=None, debug: bool = False):
        obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]
        if mb_idx is not None:
            obs = obs[mb_idx]
            act = act[mb_idx]
            adv = adv[mb_idx]
            logp_old = logp_old[mb_idx]

        pi_dist, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)

        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # entropy bonus (pi_dist.entropy() exists for both categorical and our squashed gaussian proxy)
        ent = pi_dist.entropy().mean()
        loss = loss_pi - self.ent_coef * ent

        with torch.no_grad():
            approx_kl = (logp_old - logp).mean().item()
            clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
            clipfrac = clipped.float().mean().item()

        pi_info = dict(kl=float(approx_kl), ent=float(ent.item()), cf=float(clipfrac))

        if debug:
            logging.info(f"PPO loss_pi={loss_pi.item():.6f} kl={approx_kl:.6f} ent={ent.item():.6f} cf={clipfrac:.4f}")

        return loss, pi_info

    def compute_loss_v(self, data: Dict[str, torch.Tensor], mb_idx=None, debug: bool = False):
        obs, ret = data["obs"], data["ret"]
        old_v = data.get("val", None)

        if mb_idx is not None:
            obs = obs[mb_idx]
            ret = ret[mb_idx]
            if old_v is not None:
                old_v = old_v[mb_idx]

        v = self.ac.v(obs)

        if self.clip_range_vf is not None and old_v is not None:
            v_clipped = old_v + torch.clamp(v - old_v, -self.clip_range_vf, self.clip_range_vf)
            loss_v = F.mse_loss(v_clipped, ret)
        else:
            loss_v = F.mse_loss(v, ret)

        if debug:
            logging.info(f"Value loss={loss_v.item():.6f}")
        return loss_v

    # ---------------------------------------------------------------------
    # API used by ppo_train_test_plot
    # ---------------------------------------------------------------------

    def remember(self, state, action, reward, new_state, done, value, log_probs, debug: bool = False):
        # value/log_probs can come as numpy arrays
        v = float(np.asarray(value).reshape(-1)[0])
        lp = float(np.asarray(log_probs).reshape(-1)[0])
        self.replay_buffer.store(obs=state, act=action, rew=reward, val=v, logp=lp)

    def get_epoch_permutation(self, batch_size: int, epoch: int) -> np.ndarray:
        if self.shared_minibatch_shuffle:
            seed = (self.shared_shuffle_seed + epoch) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            return rng.permutation(batch_size)
        return np.random.permutation(batch_size)

    def update(self, step: int = 0, debug: bool = False):
        # only update when buffer full
        if self.replay_buffer.ptr < self.replay_buffer.max_size:
            return

        data = self.replay_buffer.get()

        # baseline losses
        pi_loss_old, pi_info_old = self.compute_loss_pi(data)
        v_loss_old = self.compute_loss_v(data).item()

        # policy update
        total_size = int(data["obs"].shape[0])
        mb_size = min(self.minibatch_size, total_size)

        policy_kl_vals = []
        policy_ent_vals = []
        policy_clipfracs = []
        policy_losses = []
        early_stop = 0
        early_stop_flag_total = False

        for epoch in range(self.train_pi_iters):
            idxs = self.get_epoch_permutation(total_size, epoch)
            kl_list = []
            kl_ema = None
            for start in range(0, total_size, mb_size):
                mb_idx = idxs[start : start + mb_size]

                self.pi_optimizer.zero_grad(set_to_none=True)
                loss_pi, pi_info = self.compute_loss_pi(data, mb_idx=mb_idx, debug=debug)

                # MPI average KL for early stopping
                kl = mpi_avg(pi_info["kl"]) if not self.is_hyper_tune else float(pi_info["kl"])

                if self.use_smoothed_kl:
                    kl_ema = kl if kl_ema is None else (self.kl_smoothing_coef * kl + (1 - self.kl_smoothing_coef) * kl_ema)
                    # metric_kl = float(kl_ema) # Original.
                    metric_kl = kl_ema
                else:
                    # metric_kl = float(kl)     # Original.
                    kl_list.append(kl)
                    metric_kl = np.mean(kl_list)

                policy_kl_vals.append(float(kl))
                policy_ent_vals.append(float(pi_info["ent"]))
                policy_clipfracs.append(float(pi_info["cf"]))
                policy_losses.append(float(loss_pi.item()))

                if self.target_kl is not None and metric_kl > 1.5 * self.target_kl:
                    early_stop = epoch + 1
                    self.logger.log(f"Early stopping policy at epoch {epoch}, minibatch starting {start} due to KL metric {metric_kl:.6f}")
                    early_stop_this_epoch = True
                    early_stop_flag_total = True
                    break

                loss_pi.backward()
                if not self.is_hyper_tune:
                    mpi_avg_grads(self.ac.pi)
                # if self.max_grad_norm is not None:                                              # Original.
                #     torch.nn.utils.clip_grad_norm_(self.ac.pi.parameters(), self.max_grad_norm) # Original.
                # self.pi_optimizer.step()                                                        # Original.
                self.check_gradients(self.ac.pi)
                self.pi_optimizer.step()

            if early_stop:
                break


        # Adaptive learning rate based on early stops
        backoff_applied = False
        recovery_applied = False
        if early_stop_flag_total:
            self.early_stop_counter += 1
            self.no_early_stop_counter = 0
        else:
            self.no_early_stop_counter += 1
            self.early_stop_counter = 0

        if self.early_stop_counter >= self.early_stop_backoff_patience:
            for pg in self.pi_optimizer.param_groups:
                new_lr = max(pg['lr'] * self.early_stop_backoff_factor, self.min_lr)
                if new_lr < pg['lr']:
                    pg['lr'] = new_lr 
                    backoff_applied = True
            self.early_stop_counter = 0

        if self.no_early_stop_counter >= self.lr_recovery_patience:
            for pg in self.pi_optimizer.param_groups:
                restored = min(pg['lr'] * self.lr_recovery_factor, self.original_pi_lr)
                if restored > pg['lr']:
                    pg['lr'] = restored 
                    recovery_applied = True
            self.no_early_stop_counter = 0




        # value update
        value_losses = []
        for _ in range(self.train_v_iters):
            # Original:
            # self.v_optimizer.zero_grad(set_to_none=True)
            # loss_v = self.compute_loss_v(data)
            # loss_v.backward()
            # if not self.is_hyper_tune:
            #     mpi_avg_grads(self.ac.v)
            # if self.max_grad_norm is not None:
            #     torch.nn.utils.clip_grad_norm_(self.ac.v.parameters(), self.max_grad_norm)
            # self.v_optimizer.step()
            # value_losses.append(float(loss_v.item()))
            # End of Original.

            idxs = self.get_epoch_permutation(total_size, epoch)
            for start in range(0, total_size, mb_size):
                mb_idx = idxs[start:start+mb_size]
                self.v_optimizer.zero_grad(set_to_none=True)
                loss_v = self.compute_loss_v(data, mb_idx=mb_idx, debug=debug)
                loss_v.backward()
                if not self.is_hyper_tune:
                    mpi_avg_grads(self.ac.v)
                self.check_gradients(self.ac.v)
                self.v_optimizer.step()
                value_losses.append(loss_v.item())



        # diagnostics
        with torch.no_grad():
            v_pred = self.ac.v(data["obs"]).cpu().numpy()
            ev = explained_variance(v_pred, data["ret"].cpu().numpy())

        # log
        self.logger.store(
            LossPi=float(pi_loss_old.item()),
            KL=float(np.mean(policy_kl_vals)) if policy_kl_vals else float(pi_info_old.get("kl", 0.0)),
            Entropy=float(np.mean(policy_ent_vals)) if policy_ent_vals else float(pi_info_old.get("ent", 0.0)),
            ClipFrac=float(np.mean(policy_clipfracs)) if policy_clipfracs else float(pi_info_old.get("cf", 0.0)),
            PolicyLoss=float(np.mean(policy_losses)) if policy_losses else float(pi_loss_old.item()),
            ValueLoss=float(np.mean(value_losses)) if value_losses else float(v_loss_old),
            DeltaLossV=float((np.mean(value_losses) - v_loss_old) if value_losses else 0.0),
            StopIter=int(early_stop),
            ExplainedVariance=float(ev),
            PolicyLR=float(self.pi_optimizer.param_groups[0]["lr"]),
        )




