#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LivePnLCalculator.py

FIFO/LIFO realized + unrealized PnL tracker per asset.

Fix included (critical):
- For sells, consume the *requested* volume, not the last buy volume.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np


class LivePnLCalculator:
    def __init__(self, assets: List[str], smoothing_span: int = 20):
        self.assets = list(assets)
        self.smoothing_span = int(smoothing_span)

        self.fifo_positions: Dict[str, Deque[Dict[str, float]]] = {a: deque() for a in self.assets}
        self.lifo_positions: Dict[str, List[Dict[str, float]]] = {a: [] for a in self.assets}

        self.realized_pnl_fifo: Dict[str, float] = {a: 0.0 for a in self.assets}
        self.realized_pnl_lifo: Dict[str, float] = {a: 0.0 for a in self.assets}
        self.last_price: Dict[str, float] = {a: 0.0 for a in self.assets}

        self._locks: Dict[str, threading.Lock] = {a: threading.Lock() for a in self.assets}

        self.alpha = 2.0 / (self.smoothing_span + 1.0) if self.smoothing_span > 0 else 1.0
        self.metrics = [
            "fifo_realized", "fifo_unrealized",
            "lifo_realized", "lifo_unrealized",
            "fifo_total", "lifo_total",
        ]
        self.ewm_state: Dict[str, Dict[str, float]] = {
            a: {m: 0.0 for m in self.metrics} for a in self.assets
        }

    def reset(self) -> None:
        for a in self.assets:
            with self._locks[a]:
                self.fifo_positions[a].clear()
                self.lifo_positions[a].clear()
                self.realized_pnl_fifo[a] = 0.0
                self.realized_pnl_lifo[a] = 0.0
                self.last_price[a] = 0.0
                for m in self.metrics:
                    self.ewm_state[a][m] = 0.0

    def _ewm_update(self, asset: str, metric: str, value: float) -> float:
        prev = float(self.ewm_state[asset][metric])
        v = float(value)
        out = self.alpha * v + (1.0 - self.alpha) * prev
        self.ewm_state[asset][metric] = out
        return out

    def process_trade(self, asset: str, volume: float, price: float, trade_type: str) -> None:
        a = str(asset)
        vol = float(volume)
        px = float(price)
        tt = str(trade_type).lower()
        if not np.isfinite(px) or px <= 0.0:
            return
        if not np.isfinite(vol) or abs(vol) <= 1e-18:
            self.last_price[a] = px
            return

        with self._locks[a]:
            self.last_price[a] = px

            if tt == "buy":
                lot = {"volume": vol, "price": px}
                self.fifo_positions[a].append(lot.copy())
                self.lifo_positions[a].append(lot.copy())
            elif tt == "sell":
                sell_vol = abs(vol)

                # FIFO consume
                remaining = sell_vol
                while remaining > 1e-18 and len(self.fifo_positions[a]) > 0:
                    lot = self.fifo_positions[a][0]
                    take = min(remaining, lot["volume"])
                    pnl = (px - lot["price"]) * take
                    self.realized_pnl_fifo[a] += pnl
                    lot["volume"] -= take
                    remaining -= take
                    if lot["volume"] <= 1e-18:
                        self.fifo_positions[a].popleft()

                # LIFO consume
                remaining = sell_vol
                while remaining > 1e-18 and len(self.lifo_positions[a]) > 0:
                    lot = self.lifo_positions[a][-1]
                    take = min(remaining, lot["volume"])
                    pnl = (px - lot["price"]) * take
                    self.realized_pnl_lifo[a] += pnl
                    lot["volume"] -= take
                    remaining -= take
                    if lot["volume"] <= 1e-18:
                        self.lifo_positions[a].pop()
            else:
                # unknown trade type -> ignore
                return

    def get_pnl(self, asset: str, current_price: float) -> Dict[str, float]:
        a = str(asset)
        px = float(current_price)
        if not np.isfinite(px) or px <= 0.0:
            px = self.last_price.get(a, 0.0)

        with self._locks[a]:
            # unrealized = sum((px - lot_price)*lot_volume)
            fifo_unreal = sum((px - lot["price"]) * lot["volume"] for lot in self.fifo_positions[a])
            lifo_unreal = sum((px - lot["price"]) * lot["volume"] for lot in self.lifo_positions[a])

            fifo_real = float(self.realized_pnl_fifo[a])
            lifo_real = float(self.realized_pnl_lifo[a])

            fifo_total = fifo_real + fifo_unreal
            lifo_total = lifo_real + lifo_unreal

            # EWMA smoothing
            out = {
                "fifo_realized": self._ewm_update(a, "fifo_realized", fifo_real),
                "fifo_unrealized": self._ewm_update(a, "fifo_unrealized", fifo_unreal),
                "lifo_realized": self._ewm_update(a, "lifo_realized", lifo_real),
                "lifo_unrealized": self._ewm_update(a, "lifo_unrealized", lifo_unreal),
                "fifo_total": self._ewm_update(a, "fifo_total", fifo_total),
                "lifo_total": self._ewm_update(a, "lifo_total", lifo_total),
                # raw (non-smoothed) for debugging
                "fifo_realized_raw": fifo_real,
                "fifo_unrealized_raw": float(fifo_unreal),
                "lifo_realized_raw": lifo_real,
                "lifo_unrealized_raw": float(lifo_unreal),
            }
            return out
