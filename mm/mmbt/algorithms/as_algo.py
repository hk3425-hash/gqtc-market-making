"""
as_algo.py — Avellaneda-Stoikov Market Maker

Pure Python, no numba. Pluggable signals signals.

Model
-----
  reservation_price = mid − q·γ·σ²·τ  + Σ signal.compute(...)
  optimal_spread    = γ·σ²·τ + (2/γ)·ln(1 + γ/k)
  bid = reservation − spread/2  (clamped to ≤ best_bid)
  ask = reservation + spread/2  (clamped to ≥ best_ask)

Sigma is estimated online from a rolling window of mid-price levels.
k can be fixed or estimated from market message rate (k_auto).

Usage
-----
    from mmbt.algorithms import AvellanedaStoikovStrategy
    from mmbt.algorithms.signals import OBISignal

    strat = AvellanedaStoikovStrategy(
        gamma=0.01, k=20.0, horizon=300.0,
        order_qty_usd=100.0, max_position_usd=1000.0,
        signals=[OBISignal(c1=0.5, roi_ub=1200.0)],   # optional
        k_auto=True,                                    # optional
    )
"""

from __future__ import annotations
import math
from collections import deque
from typing import List, Optional
from hftbacktest import BUY, SELL, GTX, LIMIT

try:
    from .base_algo import BaseStrategy
    from .signals import AlphaSignal
except ImportError:
    from mmbt_clean.strategies.base import BaseStrategy
    from mmbt_clean.strategies.alpha import AlphaSignal


class AvellanedaStoikovStrategy(BaseStrategy):
    """
    Avellaneda-Stoikov optimal market maker.

    Parameters
    ----------
    gamma             : risk-aversion γ (typical range 0.001–0.1)
    k                 : order-arrival intensity κ (initial value if k_auto=True)
    horizon           : session horizon in seconds (spread widens as τ → 0)
    order_qty_usd     : order size in USD
    max_position_usd  : max absolute inventory in USD (symmetric)
    sigma_window      : rolling window size for σ estimation
    min_half_spread   : minimum half-spread floor in USD
    interval          : nanoseconds between updates
    k_auto            : if True, estimate k from message rate (EMA)
    k_min             : floor on k when k_auto=True
    k_window          : steps per k-estimation window
    signals           : list of AlphaSignal objects (default: none → signals=0)
    """

    def __init__(
            self,
            gamma: float,
            k: float,
            horizon: float,
            order_qty_usd: float,
            max_position_usd: float,
            sigma_window: int = 200,
            min_half_spread: float = 0.005,
            interval: int = 100_000_000,
            k_auto: bool = False,
            k_min: float = 5.0,
            k_window: int = 100,
            signals: Optional[List[AlphaSignal]] = None,
    ) -> None:
        super().__init__(interval=interval)
        self.gamma = float(gamma)
        self.k_init = float(k)
        self.horizon = float(horizon)
        self.order_qty_usd = float(order_qty_usd)
        self.max_position_usd = float(max_position_usd)
        self.sigma_window = int(sigma_window)
        self.min_half_spread = float(min_half_spread)
        self.k_auto = bool(k_auto)
        self.k_min = float(k_min)
        self.k_window = int(k_window)
        self.signals = signals or []

        # Mutable state (reset on each run)
        self._sigma_buf: deque = deque(maxlen=sigma_window)
        self._k: float = float(k)
        self._k_steps: int = 0
        self._elapsed_s: float = 0.0
        self.interval_s: float = self.interval / 1e9

    def _on_start(self, hbt) -> None:
        """Reset all dynamic state before each run."""
        self._sigma_buf.clear()
        self._k = self.k_init
        self._k_steps = 0
        self._elapsed_s = 0.0
        for sig in self.signals:
            sig.reset()

    def _estimate_sigma(self, mid: float) -> float:
        """Update rolling sigma from mid-price levels, return current estimate."""
        self._sigma_buf.append(mid)
        n = len(self._sigma_buf)
        if n < 5:
            return 0.001
        prices = list(self._sigma_buf)
        mean = sum(prices) / n
        var = sum((p - mean) ** 2 for p in prices) / n
        # Convert variance of levels to per-second volatility
        raw = math.sqrt(var / self.interval_s) if var > 0 else 0.001
        cap = mid * 0.001  # cap at 0.1% of mid
        return max(min(raw, cap), 0.001)

    def _update_k_auto(self) -> None:
        """Count message steps; estimate k every k_window steps."""
        self._k_steps += 1
        if self._k_steps >= self.k_window:
            # Rate = steps per second — a proxy for market activity
            rate = self.k_window / (self.k_window * self.interval_s)
            self._k = max(self.k_min, 0.3 * rate + 0.7 * self._k)
            self._k_steps = 0

    def _compute_quotes(
            self,
            mid: float,
            position: float,
            sigma: float,
            alpha: float,
            best_bid: float,
            best_ask: float,
            tick: float,
    ) -> tuple[float, float]:
        """Return (bid_price, ask_price) snapped to tick grid."""
        self._elapsed_s += self.interval_s
        tau = max(self.horizon - (self._elapsed_s % self.horizon), 1.0)

        # Reservation price
        res = mid - position * self.gamma * sigma * sigma * tau + alpha

        # Optimal spread
        g, k = self.gamma, self._k
        if g > 0 and k > 0:
            spread = g * sigma * sigma * tau + (2.0 / g) * math.log(1 + g / k)
        else:
            spread = 2.0 / max(k, 0.01)

        half = max(spread / 2.0, self.min_half_spread)
        half = min(half, mid * 0.02)

        bid = self._snap_bid(min(res - half, best_bid), tick)
        ask = self._snap_ask(max(res + half, best_ask), tick)

        if bid <= 0:
            bid = tick
        if ask <= bid:
            ask = bid + tick

        return bid, ask

    def _step(self, hbt, asset_no: int) -> None:
        depth = hbt.depth(asset_no)
        position = hbt.position(asset_no)
        orders = hbt.orders(asset_no)
        tick = depth.tick_size
        lot = depth.lot_size
        best_bid = depth.best_bid
        best_ask = depth.best_ask
        mid = (best_bid + best_ask) / 2.0
        pos_val = position * mid

        if self.k_auto:
            self._update_k_auto()

        sigma = self._estimate_sigma(mid)
        alpha = sum(s.compute(depth, mid, position) for s in self.signals)

        bid_price, ask_price = self._compute_quotes(
            mid, position, sigma, alpha, best_bid, best_ask, tick
        )

        order_qty = self._order_qty(self.order_qty_usd, mid, lot)

        # Single bid / single ask (A-S is a single-quote strategy)
        bid_id = int(round(bid_price / tick))
        ask_id = int(round(ask_price / tick)) + 1_000_000_000

        target_bids = {bid_id: bid_price} if pos_val < self.max_position_usd and math.isfinite(bid_price) else {}
        target_asks = {ask_id: ask_price} if pos_val > -self.max_position_usd and math.isfinite(ask_price) else {}

        # Cancel stale
        ov = orders.values()
        while ov.has_next():
            o = ov.get()
            if o.cancellable:
                if (o.side == BUY and o.order_id not in target_bids) or \
                        (o.side == SELL and o.order_id not in target_asks):
                    self._log_cancel(hbt.current_timestamp, o.order_id)
                    hbt.cancel(asset_no, o.order_id, False)

        for oid, price in target_bids.items():
            if oid not in orders:
                self._log_submit(hbt.current_timestamp, "BUY", price, order_qty, oid)
                hbt.submit_buy_order(asset_no, oid, price, order_qty, GTX, LIMIT, False)

        for oid, price in target_asks.items():
            if oid not in orders:
                self._log_submit(hbt.current_timestamp, "SELL", price, order_qty, oid)
                hbt.submit_sell_order(asset_no, oid, price, order_qty, GTX, LIMIT, False)

        self._record(hbt.current_timestamp, mid, bid_price, ask_price, position)

    @property
    def params_dict(self) -> dict:
        d = dict(
            strategy="AvellanedaStoikovStrategy",
            gamma=self.gamma, k=self.k_init, horizon_s=self.horizon,
            order_qty_usd=self.order_qty_usd, max_position_usd=self.max_position_usd,
            sigma_window=self.sigma_window, min_half_spread_usd=self.min_half_spread,
            interval_ns=self.interval, k_auto=self.k_auto, k_min=self.k_min,
        )
        if self.signals:
            d["signals"] = [s.params_dict for s in self.signals]
        return d
