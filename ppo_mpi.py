#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ppo_mpi.py

MPI helpers (SpinningUp-style) with graceful fallback when mpi4py is not installed.

Why:
- Your PPO training code expects MPI utilities for multi-process speedups.
- Many local setups (including some notebooks/containers) may not have mpi4py.

Behavior:
- If mpi4py is available: true MPI support.
- If not: stubs behave like single-process training.

Important:
- `mpi_fork(n)` still requires a working MPI runtime (e.g., OpenMPI) and mpi4py.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
import torch


# -----------------------------------------------------------------------------
# MPI detection / fallback
# -----------------------------------------------------------------------------

try:
    from mpi4py import MPI  # type: ignore

    _MPI_AVAILABLE = True
except Exception:  # pragma: no cover
    MPI = None  # type: ignore
    _MPI_AVAILABLE = False



# Set environment variables for running as root
os.environ['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
os.environ['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'

def mpi_fork(n: int, bind_to_core: bool = False):
    """Re-launch the current script with `mpirun -np n`.

    If mpi4py is missing, this raises a clear error.
    """
    n = int(n)
    if n <= 1:
        return
    if not _MPI_AVAILABLE:
        raise RuntimeError("mpi4py is not installed. Install mpi4py and an MPI runtime to use mpi_fork.")

    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        # env["IN_MPI"] = "1"
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        args = ["mpirun", "-np", str(n)]
        if bind_to_core:
            args += ["-bind-to", "core"]
        args += [sys.executable] + sys.argv
        subprocess.check_call(args, env=env)
        sys.exit()


def proc_id() -> int:
    if not _MPI_AVAILABLE:
        return 0
    return int(MPI.COMM_WORLD.Get_rank())


def num_procs() -> int:
    if not _MPI_AVAILABLE:
        return 1
    return int(MPI.COMM_WORLD.Get_size())


def mpi_avg(x: Any) -> float:
    x = float(x)
    if not _MPI_AVAILABLE:
        return x
    buf = np.array(x, dtype=np.float64)
    MPI.COMM_WORLD.Allreduce(buf, buf, op=MPI.SUM)
    return float(buf / num_procs())


def mpi_sum(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if not _MPI_AVAILABLE:
        return x
    out = np.zeros_like(x)
    MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
    return out


def mpi_statistics_scalar(x, with_min_and_max: bool = False) -> Tuple[float, float, float | None, float | None]:
    """Compute mean/std (and optionally min/max) across MPI processes."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        mean, std = 0.0, 0.0
        return (mean, std, None, None) if with_min_and_max else (mean, std)

    if not _MPI_AVAILABLE:
        mean = float(np.mean(x))
        std = float(np.std(x) + 1e-8)
        if with_min_and_max:
            return mean, std, float(np.min(x)), float(np.max(x))
        return mean, std

    sum_x = mpi_sum(np.sum(x))
    sum_x2 = mpi_sum(np.sum(np.square(x)))
    count = mpi_sum(np.array(x.size, dtype=np.float64))

    mean = float(sum_x / count)
    var = float(sum_x2 / count - mean**2)
    std = float(np.sqrt(max(var, 0.0)) + 1e-8)

    if with_min_and_max:
        global_min = float(MPI.COMM_WORLD.allreduce(np.min(x), op=MPI.MIN))
        global_max = float(MPI.COMM_WORLD.allreduce(np.max(x), op=MPI.MAX))
        return mean, std, global_min, global_max

    return mean, std


def setup_pytorch_for_mpi():
    """Limit torch threads per process to reduce contention."""
    if torch.get_num_threads() == 1:
        return
    fair = max(int(torch.get_num_threads() / max(num_procs(), 1)), 1)
    torch.set_num_threads(fair)


def sync_params(module: torch.nn.Module):
    """Sync parameters across MPI ranks."""
    if not _MPI_AVAILABLE:
        return
    comm = MPI.COMM_WORLD
    for p in module.parameters():
        p_data = p.data.cpu().numpy()
        comm.Bcast(p_data, root=0)
        p.data.copy_(torch.as_tensor(p_data, device=p.data.device))


def mpi_avg_grads(module: torch.nn.Module):
    """All-reduce gradients across ranks and average."""
    if not _MPI_AVAILABLE:
        return
    comm = MPI.COMM_WORLD
    world = num_procs()
    for p in module.parameters():
        if p.grad is None:
            continue
        grad = p.grad.data.cpu().numpy()
        buf = np.zeros_like(grad)
        comm.Allreduce(grad, buf, op=MPI.SUM)
        p.grad.data.copy_(torch.as_tensor(buf / world, device=p.grad.device))

