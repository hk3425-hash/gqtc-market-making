"""
algorithms/signals/obi_signal.py — Order Book Imbalance signals signal.

Scans a configurable depth band on each side of the order book, computes
the raw or ratio-normalised imbalance, then standardises it over a rolling
window to produce a z-score signals adjustment in USD.

    signals(t) = c1 * (I(t) - mean(I)) / std(I)

where I(t) = bid_qty - ask_qty  (raw mode, default)
          or (bid_qty - ask_qty) / (bid_qty + ask_qty)  (ratio mode).
"""

from __future__ import annotations

import math
from collections import deque

from .base_signal import AlphaSignal


class OBISignal(AlphaSignal):
    """
    Order Book Imbalance (OBI) signals signal.

    Parameters
    ----------
    c1 : float
        Sensitivity — USD shift per standard deviation of imbalance.
    looking_depth : float
        Fraction of mid price to scan on each side (e.g. 0.01 = 1 %).
    window : int
        Rolling window length in strategy steps.
    roi_lb : float
        Lower bound of the region of interest in USD (same as AssetConfig.roi_lb).
    roi_ub : float
        Upper bound of the region of interest in USD (same as AssetConfig.roi_ub).
    ratio_mode : bool
        If True, normalise imbalance by total depth; default is raw difference.
    """

    def __init__(
            self,
            c1: float = 0.5,
            looking_depth: float = 0.01,
            window: int = 600,
            roi_lb: float = 0.0,
            roi_ub: float = 1200.0,
            ratio_mode: bool = False,
    ) -> None:
        self.c1 = float(c1)
        self.looking_depth = float(looking_depth)
        self.window = int(window)
        self.roi_lb = float(roi_lb)
        self.roi_ub = float(roi_ub)
        self.ratio_mode = bool(ratio_mode)
        self._buf: deque = deque(maxlen=self.window)

    def reset(self) -> None:
        self._buf.clear()

    def compute(self, depth, mid: float, position: float) -> float:
        tick = depth.tick_size
        roi_lb_tick = int(round(self.roi_lb / tick))
        roi_ub_tick = int(round(self.roi_ub / tick))

        # --- Scan ask side ---
        sum_ask = 0.0
        f_ask = max(depth.best_ask_tick, roi_lb_tick)
        u_ask = min(int(math.floor(mid * (1.0 + self.looking_depth) / tick)), roi_ub_tick)
        for pt in range(f_ask, u_ask):
            sum_ask += depth.ask_depth[pt - roi_lb_tick]

        # --- Scan bid side ---
        sum_bid = 0.0
        f_bid = min(depth.best_bid_tick, roi_ub_tick)
        u_bid = max(int(math.ceil(mid * (1.0 - self.looking_depth) / tick)), roi_lb_tick)
        for pt in range(f_bid, u_bid, -1):
            sum_bid += depth.bid_depth[pt - roi_lb_tick]

        # --- Imbalance ---
        if self.ratio_mode:
            total = sum_bid + sum_ask
            imb = (sum_bid - sum_ask) / total if total > 0.0 else 0.0
        else:
            imb = sum_bid - sum_ask

        self._buf.append(imb)

        if len(self._buf) < 5:
            return 0.0

        vals = list(self._buf)
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var) if var > 1e-12 else 1.0
        return self.c1 * (imb - mean) / std

    @property
    def params_dict(self) -> dict:
        return {
            "signal": "OBISignal",
            "c1": self.c1,
            "looking_depth": self.looking_depth,
            "window": self.window,
            "roi_lb": self.roi_lb,
            "roi_ub": self.roi_ub,
            "ratio_mode": self.ratio_mode,
        }
