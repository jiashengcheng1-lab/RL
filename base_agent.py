from __future__ import annotations

import abc
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class GradCheckConfig:
    # Only log extremes; keep it lightweight
    min_norm: float = 1e-5
    max_norm: float = 50.0
    skip_bias: bool = True
    # Optional: skip LayerNorm/BatchNorm params from checks
    skip_norm_layers: bool = True


class Agent(abc.ABC):
    """
    Base class for RL agents.

    Provides:
      - unified seeding
      - device management
      - gradient diagnostics
      - gradient clipping helper
      - optimizer LR helpers

    Keeps algorithm-specific logic out (buffers/losses/env loops).
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        device: Optional[Union[str, torch.device]] = None,
        logger: Optional[Any] = None,
        gradcheck_cfg: Optional[GradCheckConfig] = None,
        max_grad_norm: Optional[float] = None,
    ) -> None:
        self.seed = int(seed)
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.logger = logger  # can be your EpochLogger or None
        self.gradcheck_cfg = gradcheck_cfg or GradCheckConfig()
        self.max_grad_norm = max_grad_norm

        self._set_global_seeds(self.seed)

    # -----------------------
    # Seeding (global + env)
    # -----------------------
    @staticmethod
    def _set_global_seeds(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Determinism tradeoff: enable only if you *need* strict reproducibility.
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    @staticmethod
    def seed_env(env: Any, seed: int) -> None:
        """
        Best-effort seeding for Gym/Gymnasium envs.
        """
        if env is None:
            return
        try:
            env.reset(seed=seed)
        except Exception:
            pass
        try:
            env.action_space.seed(seed)
        except Exception:
            pass
        try:
            env.observation_space.seed(seed)
        except Exception:
            pass

    # -----------------------
    # Gradient utilities
    # -----------------------
    def check_gradients(self, model: nn.Module) -> None:
        """
        Logs layers with extreme gradient norms or missing gradients.
        Does not modify gradients.
        """
        cfg = self.gradcheck_cfg

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if cfg.skip_bias and name.endswith("bias"):
                continue

            if cfg.skip_norm_layers:
                # common patterns; cheap heuristic
                if "layernorm" in name.lower() or "batchnorm" in name.lower() or ".ln" in name.lower():
                    continue

            if param.grad is None:
                logging.info(f"Layer {name} | No Gradient")
                continue

            # Use float() to avoid tensor->python overhead issues
            gn = float(param.grad.detach().norm().item())
            if gn > cfg.max_norm or gn < cfg.min_norm:
                logging.info(f"Layer {name} | Gradient Norm: {gn:.6g}")

    def clip_gradients(self, model: nn.Module, max_norm: Optional[float] = None) -> float:
        """
        Clips gradients in-place and returns total norm (pre-clip).
        """
        clip_value = float(max_norm if max_norm is not None else (self.max_grad_norm or 0.0))
        if clip_value <= 0:
            return 0.0
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        return float(total_norm.item()) if hasattr(total_norm, "item") else float(total_norm)

    # -----------------------
    # Optimizer/LR utilities
    # -----------------------
    @staticmethod
    def get_lr(optimizer: torch.optim.Optimizer) -> float:
        return float(optimizer.param_groups[0]["lr"])

    @staticmethod
    def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
        lr = float(lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

    @staticmethod
    def scale_lr(optimizer: torch.optim.Optimizer, factor: float, *, min_lr: float = 0.0, max_lr: float = float("inf")) -> float:
        """
        Multiply LR by factor for all param groups; clamps to [min_lr, max_lr].
        Returns new LR for group 0.
        """
        factor = float(factor)
        for pg in optimizer.param_groups:
            new_lr = float(pg["lr"]) * factor
            new_lr = max(min_lr, min(new_lr, max_lr))
            pg["lr"] = new_lr
        return float(optimizer.param_groups[0]["lr"])

    # -----------------------
    # Lifecycle (optional)
    # -----------------------
    # @abc.abstractmethod
    # def train(self) -> None:
    #     
    #     Main training loop driver (epoch loop). You can decide if you want this abstract.
    #     
    #     raise NotImplementedError

    @abc.abstractmethod
    def update(self, *args, **kwargs) -> Dict[str, Any]:
        """
        One update step for the agent (policy/value updates).
        """
        raise NotImplementedError
