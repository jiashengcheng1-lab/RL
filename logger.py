#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""logger.py

A compact, robust Logger + EpochLogger in the style of OpenAI SpinningUp.

Why replace:
- Your uploaded logger file had imports and definitions out of order, which
  caused runtime NameError and made it fragile.

What this provides (and what PPO/SAC here rely on):
- Logger.log(msg)
- EpochLogger.store(**kwargs)
- EpochLogger.log_tabular(key, ...)
- EpochLogger.dump_tabular()
- EpochLogger.setup_pytorch_saver(model)

This is intentionally dependency-light (no TensorBoard).
"""

from __future__ import annotations

import atexit
import json
import os
import os.path as osp
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch


def colorize(string: str, color: str, bold: bool = False, highlight: bool = False) -> str:
    """Colorize text for terminal printing."""
    color2num = dict(
        gray=30,
        red=31,
        green=32,
        yellow=33,
        blue=34,
        magenta=35,
        cyan=36,
        white=37,
        crimson=38,
    )
    attr = []
    num = color2num.get(color, 37)
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")
    return f"\x1b[{';'.join(attr)}m{string}\x1b[0m"


def _now_str() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


class Logger:
    def __init__(
        self,
        output_dir: Optional[str] = None,
        exp_name: Optional[str] = None,
        output_fname: str = "progress.txt",
    ):
        exp_name = exp_name or "experiment"
        if output_dir is None:
            output_dir = osp.join("./results", f"{exp_name}_{_now_str()}")
        self.output_dir = output_dir
        self.output_fname = output_fname
        os.makedirs(self.output_dir, exist_ok=True)

        self._output_file = open(osp.join(self.output_dir, self.output_fname), "w")
        atexit.register(self._output_file.close)

        self.log(f"Logging data to {self._output_file.name}", color="green")

        self._first_row = True
        self._log_headers = []
        self._log_current_row: Dict[str, Any] = {}

        self._pytorch_saver = None

    def log(self, msg: str, color: str = "green"):
        print(colorize(msg, color, bold=True))

    def save_config(self, config: Dict[str, Any], filename: str = "config.json"):
        path = osp.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump(config, f, indent=2, default=str)

    def setup_pytorch_saver(self, model: torch.nn.Module):
        self._pytorch_saver = model

    def save_state(self, state: Dict[str, Any], itr: Optional[int] = None):
        """Save model + ancillary training state."""
        if self._pytorch_saver is None:
            return
        fname = "pyt_save"
        if itr is not None:
            fname += f"_{itr}"
        path = osp.join(self.output_dir, fname + ".pt")
        payload = {"model": self._pytorch_saver.state_dict(), "state": state}
        torch.save(payload, path)


class EpochLogger(Logger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_dict = defaultdict(list)

    def store(self, **kwargs):
        for k, v in kwargs.items():
            self.epoch_dict[k].append(v)

    def log_tabular(
        self,
        key: str,
        val: Optional[float] = None,
        with_min_and_max: bool = False,
        average_only: bool = False,
    ):
        """Log a value or stats of stored values."""
        if val is not None:
            v = float(val)
        else:
            vals = self.epoch_dict.get(key, [])
            if len(vals) == 0:
                v = float("nan")
            else:
                arr = np.asarray(vals, dtype=np.float64)
                v = float(np.mean(arr))
                if with_min_and_max:
                    self._log_current_row[key + "Min"] = float(np.min(arr))
                    self._log_current_row[key + "Max"] = float(np.max(arr))
                if not average_only:
                    self._log_current_row[key + "Std"] = float(np.std(arr))
        self._log_current_row[key] = v

    def dump_tabular(self):
        keys = list(self._log_current_row.keys())
        if self._first_row:
            self._log_headers = keys
            self._output_file.write("\t".join(keys) + "\n")
            self._first_row = False

        vals = [self._log_current_row.get(k, "") for k in self._log_headers]
        valstrs = []
        for v in vals:
            if isinstance(v, float):
                valstrs.append(f"{v:8.3g}")
            else:
                valstrs.append(str(v))

        # pretty print
        max_key_len = max(len(k) for k in self._log_headers) if self._log_headers else 0
        print("-" * (max_key_len + 35))
        for k in self._log_headers:
            v = self._log_current_row.get(k, "")
            if isinstance(v, float):
                v = f"{v:8.3g}"
            print(f"{k:<{max_key_len}} : {v}")
        print("-" * (max_key_len + 35))

        self._output_file.write("\t".join(map(str, vals)) + "\n")
        self._output_file.flush()

        self._log_current_row.clear()
        self.epoch_dict.clear()

