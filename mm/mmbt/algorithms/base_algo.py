"""
algorithms/base_algo.py — Base Strategy class

Every strategy is a Python class. No @njit required.

The run loop is implemented once here in BaseStrategy.run().
Subclasses only implement _step(hbt, asset_no) — the per-step logic.

Usage
-----
    class MyStrategy(BaseStrategy):
        def __init__(self, my_param, **kwargs):
            super().__init__(**kwargs)
            self.my_param = my_param

        def _step(self, hbt, asset_no):
            depth = hbt.depth(asset_no)
            mid   = (depth.best_bid + depth.best_ask) / 2.0
            # place orders...

    strat  = MyStrategy(my_param=1.0, interval=100_000_000)
    runner = BacktestRunner(assets=[...], strategy=strat, book_size=2000.0)
    result = runner.run()
"""

from __future__ import annotations
import math
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np


class BaseStrategy(ABC):
    """
    Pure-Python base for all mmbt algorithms.

    Parameters
    ----------
    interval  : nanoseconds between updates (e.g. 100_000_000 = 100ms)

    Note: ``n_assets`` and ``days`` are no longer constructor parameters.
    They are injected automatically by BacktestRunner via _configure()
    once the asset list and date range are known.
    """

    def __init__(self, interval: int) -> None:
        self.interval = int(interval)

        # Defaults — overwritten by BacktestRunner._configure() before run()
        self.n_assets = 1
        self.days = 2.0
        self._max_steps = math.ceil(2.0 * 86_400 * 1_000_000_000 / self.interval) + 500
        self._reset_recording()

        # Progress bar — injected by BacktestRunner when show_progress=True
        self._progress_bar = None

    def _configure(self, n_assets: int, days: float) -> None:
        """
        Called by BacktestRunner before run() to inject asset count and
        backtest duration derived from the actual AssetConfig list.
        Do not call this manually.
        """
        self.n_assets = int(n_assets)
        self.days = float(days)
        self._max_steps = math.ceil(days * 86_400 * 1_000_000_000 / self.interval) + 500
        self._reset_recording()

    # ── Per-asset recording ────────────────────────────────────────────

    def _reset_recording(self) -> None:
        """Reset all recording state. Called before each run."""
        self._rec: List[dict] = [
            {"ts": [], "mid": [], "bid": [], "ask": [], "pos": []}
            for _ in range(self.n_assets)
        ]
        self._orders: List[dict] = [
            {
                "sub_ts": [], "sub_side": [], "sub_price": [], "sub_qty": [], "sub_id": [],
                "can_ts": [], "can_id": [],
            }
            for _ in range(self.n_assets)
        ]

    def _record(self, ts: int, mid: float, bid: float, ask: float, pos: float,
                asset_no: int = 0) -> None:
        r = self._rec[asset_no]
        r["ts"].append(ts);
        r["mid"].append(mid)
        r["bid"].append(bid);
        r["ask"].append(ask);
        r["pos"].append(pos)

    def _log_submit(self, ts: int, side: str, price: float, qty: float, oid: int,
                    asset_no: int = 0) -> None:
        o = self._orders[asset_no]
        o["sub_ts"].append(ts);
        o["sub_side"].append(side)
        o["sub_price"].append(price);
        o["sub_qty"].append(qty);
        o["sub_id"].append(oid)

    def _log_cancel(self, ts: int, oid: int, asset_no: int = 0) -> None:
        o = self._orders[asset_no]
        o["can_ts"].append(ts);
        o["can_id"].append(oid)

    # ── Public data accessors ──────────────────────────────────────────

    def mm_data(self) -> Optional[dict]:
        return self._build_mm_data(0)

    def mm_data_all(self) -> List[Optional[dict]]:
        return [self._build_mm_data(i) for i in range(self.n_assets)]

    def order_history(self) -> Optional[dict]:
        return self._build_order_history(0)

    def order_history_all(self) -> List[Optional[dict]]:
        return [self._build_order_history(i) for i in range(self.n_assets)]

    def _build_mm_data(self, asset_no: int) -> Optional[dict]:
        r = self._rec[asset_no]
        n = len(r["ts"])
        if n == 0:
            return None
        return {
            "timestamps": np.array(r["ts"], dtype=np.int64),
            "mid_prices": np.array(r["mid"], dtype=np.float64),
            "bid_quotes": np.array(r["bid"], dtype=np.float64),
            "ask_quotes": np.array(r["ask"], dtype=np.float64),
            "positions": np.array(r["pos"], dtype=np.float64),
        }

    def _build_order_history(self, asset_no: int) -> Optional[dict]:
        o = self._orders[asset_no]
        if not o["sub_ts"] and not o["can_ts"]:
            return None
        return {
            "submissions": {
                "timestamp_ns": o["sub_ts"],
                "side": o["sub_side"],
                "price": o["sub_price"],
                "qty": o["sub_qty"],
                "order_id": o["sub_id"],
            },
            "cancellations": {
                "timestamp_ns": o["can_ts"],
                "order_id": o["can_id"],
            },
        }

    # ── Main loop ──────────────────────────────────────────────────────

    def run(self, hbt, stat) -> bool:
        """Main loop — do not override. Implement _step() instead."""
        self._reset_recording()
        self._on_start(hbt)
        pb = getattr(self, "_progress_bar", None)
        step = 0

        while hbt.elapse(self.interval) == 0:
            for asset_no in range(self.n_assets):
                hbt.clear_inactive_orders(asset_no)
                self._step(hbt, asset_no)
            stat.record(hbt)
            step += 1
            if pb is not None:
                pb.update(step)

        return True

    # ── Hooks ──────────────────────────────────────────────────────────

    def _on_start(self, hbt) -> None:
        """Called once before the main loop starts. Override for warmup logic."""

    @abstractmethod
    def _step(self, hbt, asset_no: int) -> None:
        """Per-step strategy logic for one asset."""

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _snap_bid(price: float, tick: float) -> float:
        return math.floor(price / tick) * tick

    @staticmethod
    def _snap_ask(price: float, tick: float) -> float:
        return math.ceil(price / tick) * tick

    @staticmethod
    def _order_qty(order_qty_usd: float, mid: float, lot_size: float) -> float:
        qty = round((order_qty_usd / mid) / lot_size) * lot_size
        return max(qty, lot_size)

    @property
    def params_dict(self) -> dict:
        return {"strategy": type(self).__name__, "interval_ns": self.interval}

    # ── Backwards-compat shim ──────────────────────────────────────────
    # Warn loudly if old code passes n_assets/days to the constructor.
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
