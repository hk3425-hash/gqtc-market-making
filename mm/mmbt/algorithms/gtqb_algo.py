"""
gtqb_algo.py — Grid Trading + Queue-Based Market Maker (multi-asset, pure Python)
"""

from __future__ import annotations
import math
from typing import List, Union
import numpy as np
from hftbacktest import BUY, SELL, GTX, LIMIT

try:
    from .base_algo import BaseStrategy
except ImportError:
    from mmbt_clean.strategies.base import BaseStrategy


class GTQBStrategy(BaseStrategy):
    """
    Grid Trading + Queue-Based Market Maker for multiple assets.

    QB component : quotes at BBO, steps back 1 tick when queue is thin
    GT component : quotes around a reservation price with fixed spread

    All per-asset array params can be a single value (applied to all assets)
    or a list of values (one per asset).

    Parameters
    ----------
    interval              : ns between updates
    order_qty_usd         : order size per level in USD
    max_position_usd_qb   : QB max abs position per asset
    qty_threshold         : thin-queue detection threshold per asset
    grid_num_qb           : QB grid levels per side per asset
    max_position_usd_gt   : GT max abs position per asset
    skew_gt               : GT inventory skew (USD per unit) per asset
    grid_num_gt           : GT grid levels per side per asset
    grid_interval_usd_gt  : GT grid spacing in USD per asset
    half_spread_usd_gt    : GT half-spread in USD per asset
    """

    def __init__(
            self,
            interval: int,
            order_qty_usd: float,
            max_position_usd_qb: Union[float, List[float]],
            qty_threshold: Union[float, List[float]],
            grid_num_qb: Union[int, List[int]],
            max_position_usd_gt: Union[float, List[float]],
            skew_gt: Union[float, List[float]],
            grid_num_gt: Union[int, List[int]],
            grid_interval_usd_gt: Union[float, List[float]],
            half_spread_usd_gt: Union[float, List[float]],
    ) -> None:
        super().__init__(interval=interval)
        self.order_qty_usd = float(order_qty_usd)

        # Store raw (scalar or list) values.
        # _configure() expands them to per-asset lists once n_assets is known.
        self._raw_max_position_usd_qb = max_position_usd_qb
        self._raw_qty_threshold = qty_threshold
        self._raw_grid_num_qb = grid_num_qb
        self._raw_max_position_usd_gt = max_position_usd_gt
        self._raw_skew_gt = skew_gt
        self._raw_grid_num_gt = grid_num_gt
        self._raw_grid_interval_usd_gt = grid_interval_usd_gt
        self._raw_half_spread_usd_gt = half_spread_usd_gt

        # Placeholders so params_dict works before the runner is attached
        self.max_position_usd_qb = max_position_usd_qb
        self.qty_threshold = qty_threshold
        self.grid_num_qb = grid_num_qb
        self.max_position_usd_gt = max_position_usd_gt
        self.skew_gt = skew_gt
        self.grid_num_gt = grid_num_gt
        self.grid_interval_usd_gt = grid_interval_usd_gt
        self.half_spread_usd_gt = half_spread_usd_gt

    def _configure(self, n_assets: int, days: float) -> None:
        """Expand scalar/list params to per-asset lists, then delegate to base."""

        def _arr(v):
            return [v] * n_assets if not isinstance(v, (list, tuple)) else list(v)

        self.max_position_usd_qb = _arr(self._raw_max_position_usd_qb)
        self.qty_threshold = _arr(self._raw_qty_threshold)
        self.grid_num_qb = _arr(self._raw_grid_num_qb)
        self.max_position_usd_gt = _arr(self._raw_max_position_usd_gt)
        self.skew_gt = _arr(self._raw_skew_gt)
        self.grid_num_gt = _arr(self._raw_grid_num_gt)
        self.grid_interval_usd_gt = _arr(self._raw_grid_interval_usd_gt)
        self.half_spread_usd_gt = _arr(self._raw_half_spread_usd_gt)

        super()._configure(n_assets=n_assets, days=days)

    QB_OFFSET = 0
    GT_OFFSET = 1_000_000_000

    def _step(self, hbt, asset_no: int) -> None:
        a = asset_no
        depth = hbt.depth(a)
        position = hbt.position(a)
        orders = hbt.orders(a)
        tick = depth.tick_size
        lot = depth.lot_size
        best_bid = depth.best_bid
        best_ask = depth.best_ask
        mid = (best_bid + best_ask) / 2.0
        pos_val = position * mid

        # Guard: skip step if book is not yet populated
        if (not math.isfinite(best_bid) or not math.isfinite(best_ask)
                or best_bid <= 0.0 or best_ask <= 0.0
                or depth.best_bid_tick < 0 or depth.best_ask_tick < 0):
            return

        order_qty = self._order_qty(self.order_qty_usd, mid, lot)

        # ── QB quotes ────────────────────────────────────────────────────
        bid_qty = depth.bid_depth[depth.best_bid_tick]
        ask_qty = depth.ask_depth[depth.best_ask_tick]
        skew_qb = position / order_qty if order_qty > 0 else 0.0

        bid_qb = self._snap_bid(
            best_bid - tick if bid_qty < self.qty_threshold[a] and skew_qb > 0 else best_bid, tick)
        ask_qb = self._snap_ask(
            best_ask + tick if ask_qty < self.qty_threshold[a] and skew_qb < 0 else best_ask, tick)

        target_bids: dict[int, float] = {}
        target_asks: dict[int, float] = {}

        if pos_val < self.max_position_usd_qb[a] and math.isfinite(bid_qb):
            rem = self.max_position_usd_qb[a] - pos_val
            lvl = min(self.grid_num_qb[a], int(math.floor((rem / mid) / order_qty)))
            p = bid_qb
            for _ in range(lvl):
                target_bids[self.QB_OFFSET + int(round(p / tick))] = p
                p -= tick

        if pos_val > -self.max_position_usd_qb[a] and math.isfinite(ask_qb):
            rem = self.max_position_usd_qb[a] + pos_val
            lvl = min(self.grid_num_qb[a], int(math.floor((rem / mid) / order_qty)))
            p = ask_qb
            for _ in range(lvl):
                target_asks[self.QB_OFFSET + int(round(p / tick))] = p
                p += tick

        # ── GT quotes ────────────────────────────────────────────────────
        res = mid - self.skew_gt[a] * position
        bid_gt = self._snap_bid(min(res - self.half_spread_usd_gt[a], best_bid), tick)
        ask_gt = self._snap_ask(max(res + self.half_spread_usd_gt[a], best_ask), tick)

        if pos_val < self.max_position_usd_gt[a] and math.isfinite(bid_gt):
            rem = self.max_position_usd_gt[a] - pos_val
            lvl = min(self.grid_num_gt[a], int(math.floor(rem / self.order_qty_usd)))
            p = bid_gt
            for _ in range(lvl):
                target_bids[self.GT_OFFSET + int(round(p / tick))] = p
                p -= self.grid_interval_usd_gt[a]

        if pos_val > -self.max_position_usd_gt[a] and math.isfinite(ask_gt):
            rem = self.max_position_usd_gt[a] + pos_val
            lvl = min(self.grid_num_gt[a], int(math.floor(rem / self.order_qty_usd)))
            p = ask_gt
            for _ in range(lvl):
                target_asks[self.GT_OFFSET + int(round(p / tick))] = p
                p += self.grid_interval_usd_gt[a]

        # ── Sync orders ───────────────────────────────────────────────────
        ov = orders.values()
        while ov.has_next():
            o = ov.get()
            if o.cancellable:
                if (o.side == BUY and o.order_id not in target_bids) or \
                        (o.side == SELL and o.order_id not in target_asks):
                    self._log_cancel(hbt.current_timestamp, o.order_id, asset_no=a)
                    hbt.cancel(a, o.order_id, False)

        for oid, price in target_bids.items():
            if oid not in orders:
                self._log_submit(hbt.current_timestamp, "BUY", price, order_qty, oid, asset_no=a)
                hbt.submit_buy_order(a, oid, price, order_qty, GTX, LIMIT, False)

        for oid, price in target_asks.items():
            if oid not in orders:
                self._log_submit(hbt.current_timestamp, "SELL", price, order_qty, oid, asset_no=a)
                hbt.submit_sell_order(a, oid, price, order_qty, GTX, LIMIT, False)

        # Record every asset
        self._record(hbt.current_timestamp, mid, bid_qb, ask_qb, position, asset_no=a)

    @property
    def params_dict(self) -> dict:
        return dict(
            strategy="GTQBStrategy", interval_ns=self.interval,
            order_qty_usd=self.order_qty_usd,
            max_position_usd_qb=self.max_position_usd_qb,
            qty_threshold=self.qty_threshold, grid_num_qb=self.grid_num_qb,
            max_position_usd_gt=self.max_position_usd_gt,
            skew_gt=self.skew_gt, grid_num_gt=self.grid_num_gt,
            grid_interval_usd_gt=self.grid_interval_usd_gt,
            half_spread_usd_gt=self.half_spread_usd_gt,
        )
