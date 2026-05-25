"""
OBI-MM — Order Book Imbalance Market Maker

Pure Python, no numba.
"""

from __future__ import annotations
import math
from typing import List, Optional
from hftbacktest import BUY, SELL, GTX, LIMIT

try:
    from .base_algo import BaseStrategy
    from .signals import AlphaSignal, OBISignal
except ImportError:
    from mmbt_clean.strategies.base import BaseStrategy
    from mmbt_clean.strategies.alpha import AlphaSignal, OBISignal


class OBIStrategy(BaseStrategy):
    """
    Order Book Imbalance Market Maker.

    Quotes a symmetric grid around a reservation price derived from:
        res_price = mid + signals(OBI) − skew × position

    Parameters
    ----------
    half_spread       : USD half-spread from reservation price to first quote
    skew              : inventory skew in USD per unit position
    c1                : OBI signal sensitivity (convenience shorthand —
                        creates an OBISignal internally; ignored if signals= is set)
    looking_depth     : depth scan fraction (e.g. 0.01 = 1%)
    window            : OBI rolling window steps
    interval          : nanoseconds between updates
    order_qty_usd     : order size per level in USD
    max_position_usd  : max absolute inventory in USD
    grid_num          : grid levels per side
    grid_interval_usd : USD spacing between grid levels
    roi_lb / roi_ub   : region-of-interest bounds
    signals           : optional list of AlphaSignal objects
                        (overrides the built-in OBI shorthand)
    """

    def __init__(
            self,
            half_spread: float,
            skew: float,
            interval: int,
            order_qty_usd: float,
            max_position_usd: float,
            grid_num: int = 4,
            grid_interval_usd: float = 2.0,
            roi_lb: float = 0.0,
            roi_ub: float = 1200.0,
            # OBI shorthand (ignored if signals= is set)
            c1: float = 0.5,
            looking_depth: float = 0.01,
            window: int = 600,
            # Override signals completely
            signals: Optional[List[AlphaSignal]] = None,
    ) -> None:
        super().__init__(interval=interval)
        self.half_spread = float(half_spread)
        self.skew = float(skew)
        self.order_qty_usd = float(order_qty_usd)
        self.max_position_usd = float(max_position_usd)
        self.grid_num = int(grid_num)
        self.grid_interval_usd = float(grid_interval_usd)
        self.roi_lb = float(roi_lb)
        self.roi_ub = float(roi_ub)

        if signals is not None:
            self.signals = signals
        else:
            self.signals = [OBISignal(c1=c1, looking_depth=looking_depth,
                                      window=window, roi_lb=roi_lb, roi_ub=roi_ub)]

    def _on_start(self, hbt) -> None:
        for sig in self.signals:
            sig.reset()

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

        # Alpha
        alpha = sum(s.compute(depth, mid, position) for s in self.signals)

        # Reservation price
        order_qty = self._order_qty(self.order_qty_usd, mid, lot)
        norm_pos = position / order_qty if order_qty > 0 else 0.0
        res_price = mid + alpha - self.skew * norm_pos

        bid_price = self._snap_bid(min(round(res_price - self.half_spread), best_bid), tick)
        ask_price = self._snap_ask(max(round(res_price + self.half_spread), best_ask), tick)

        # Build target grids
        target_bids: dict[int, float] = {}
        if pos_val < self.max_position_usd and math.isfinite(bid_price):
            p = bid_price
            for _ in range(self.grid_num):
                oid = int(round(p / tick))
                target_bids[oid] = p
                p -= self.grid_interval_usd

        target_asks: dict[int, float] = {}
        if pos_val > -self.max_position_usd and math.isfinite(ask_price):
            p = ask_price
            for _ in range(self.grid_num):
                oid = int(round(p / tick)) + 1_000_000_000
                target_asks[oid] = p
                p += self.grid_interval_usd

        # Cancel stale orders
        ov = orders.values()
        while ov.has_next():
            o = ov.get()
            if o.cancellable:
                if (o.side == BUY and o.order_id not in target_bids) or \
                        (o.side == SELL and o.order_id not in target_asks):
                    self._log_cancel(hbt.current_timestamp, o.order_id)
                    hbt.cancel(asset_no, o.order_id, False)

        # Submit new orders
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
            strategy="OBIStrategy",
            half_spread=self.half_spread, skew=self.skew,
            order_qty_usd=self.order_qty_usd, max_position_usd=self.max_position_usd,
            grid_num=self.grid_num, grid_interval_usd=self.grid_interval_usd,
            roi_lb=self.roi_lb, roi_ub=self.roi_ub, interval_ns=self.interval,
        )
        if self.signals:
            d["signals"] = [s.params_dict for s in self.signals]
        return d
