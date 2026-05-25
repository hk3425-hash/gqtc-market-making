"""
quant_bot.py  —  Multi-coin market making bot

Supports three trading modes via TradingMode Enum:
  QUEUE_BASED   — Quote at best bid/ask; step back 1 tick when queue is thin
  GRID_TRADING  — Avellaneda-Stoikov reservation price with configurable spread/grid
  BOTH          — Run QB + GT simultaneously with independent capital caps

Architecture
────────────
  TradingMode          — Python Enum mirroring the C++ enum
  BotConfig            — All parameters for one coin, one mode
  StrategyMeta         — Metaclass that builds a registry mapping each TradingMode
                         to the static methods that implement that mode's:
                           • brain_factory  — build the C++ OrderBookTracker
                           • config_summary — human-readable parameter dump
  QuantBot             — One bot instance per coin, inherits StrategyMeta dispatch
  MultiCoinRunner      — Spins up N QuantBot instances concurrently
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime   import datetime
from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN
from enum       import Enum
from typing     import Dict, List, Optional
import argparse
import asyncio
import math
import os
import signal as os_signal
import sys
import time

from qb.event_publisher import EventPublisher
from qb.exchanges        import GeminiExchange, MockExchange
from qb                  import tracker


# ═══════════════════════════════════════════════════════════════════════════
# TRADING MODE ENUM
# ═══════════════════════════════════════════════════════════════════════════

class TradingMode(Enum):
    """Mirrors the C++ TradingMode enum.  Values must stay in sync."""
    QUEUE_BASED  = 0   # QB only
    GRID_TRADING = 1   # GT only
    BOTH         = 2   # QB + GT with independent capital caps


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BotConfig:
    """All parameters for a single coin + mode combination.

    Parameters used only by one mode are documented with their mode tag.
    In BOTH mode all parameters are used.
    """
    symbol:  str   = "USDCUSD"
    mode:    TradingMode = TradingMode.BOTH

    # ── Exchange tick / lot sizes (overridden by fetch_symbol_details) ──────
    tick_size: float = 0.000001
    lot_size:  float = 0.1

    # ── Shared ───────────────────────────────────────────────────────────────
    order_qty_usd:   float = 1.0   # USD notional per order level
    fee_rate:        float = 0.0000
    status_interval: float = 5.0

    # ── QB parameters ─────────────────────────────────────────────────────
    max_position_usd_qb: float = 10.0  # USD cap for QB inventory
    qty_threshold_qb:    float = 10.0    # BBO queue depth → step-back threshold
    grid_num_qb:         int   = 1       # max levels per side (QB)

    # ── GT parameters ─────────────────────────────────────────────────────
    max_position_usd_gt:   float = 10.0   # USD cap for GT inventory
    skew_gt:               float = 0.0      # inventory skew multiplier
    half_spread_usd_gt:    float = 0.001    # USD half-spread from reservation
    grid_interval_usd_gt:  float = 0.0001   # USD between GT grid levels
    grid_num_gt:           int   = 10       # max levels per side (GT)

    # ── Risk / ops ──────────────────────────────────────────────────────────
    max_concurrent_orders: int   = 8
    zmq_port:              int   = 5557
    jsonl_path:            str   = field(
        default_factory=lambda: f"logs/events_{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"
    )


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY META — registry of mode-specific static methods
# ═══════════════════════════════════════════════════════════════════════════

class StrategyMeta(type):
    """Metaclass that builds a registry of static-method handlers keyed by
    TradingMode.  Each entry maps a TradingMode to a dict of callables:

        _STRATEGY_REGISTRY[TradingMode.XXX] = {
            "brain_factory":  fn(logger, cfg) -> tracker.OrderBookTracker,
            "config_summary": fn(cfg)         -> str,
        }

    QuantBot calls these through ``self._dispatch(key, *args)`` so adding a
    new mode only requires adding one entry here — the bot logic is untouched.
    """

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)

        cls._STRATEGY_REGISTRY: Dict[TradingMode, Dict[str, callable]] = {

            # ── QUEUE_BASED ──────────────────────────────────────────────
            TradingMode.QUEUE_BASED: {
                "brain_factory": staticmethod(
                    lambda logger, cfg: tracker.OrderBookTracker(
                        logger,
                        tracker.TradingMode.QUEUE_BASED,
                        cfg.tick_size,
                        cfg.lot_size,
                        cfg.order_qty_usd,
                        # QB
                        cfg.max_position_usd_qb,
                        cfg.qty_threshold_qb,
                        cfg.grid_num_qb,
                        # GT (ignored in QB-only mode but required by C++ ctor)
                        0.0, 0.0, 0.0, 0.0, 0,
                    )
                ),
                "config_summary": staticmethod(
                    lambda cfg: (
                        f"Mode:QB | Symbol:{cfg.symbol} | "
                        f"QtyUSD:{cfg.order_qty_usd:.2f} | "
                        f"MaxPosQB:${cfg.max_position_usd_qb:.2f} | "
                        f"QtyThresh:{cfg.qty_threshold_qb} | "
                        f"GridNumQB:{cfg.grid_num_qb}"
                    )
                ),
            },

            # ── GRID_TRADING ─────────────────────────────────────────────
            TradingMode.GRID_TRADING: {
                "brain_factory": staticmethod(
                    lambda logger, cfg: tracker.OrderBookTracker(
                        logger,
                        tracker.TradingMode.GRID_TRADING,
                        cfg.tick_size,
                        cfg.lot_size,
                        cfg.order_qty_usd,
                        # QB (ignored in GT-only mode)
                        0.0, 0.0, 0,
                        # GT
                        cfg.skew_gt,
                        cfg.half_spread_usd_gt,
                        cfg.grid_interval_usd_gt,
                        cfg.max_position_usd_gt,
                        cfg.grid_num_gt,
                    )
                ),
                "config_summary": staticmethod(
                    lambda cfg: (
                        f"Mode:GT | Symbol:{cfg.symbol} | "
                        f"QtyUSD:{cfg.order_qty_usd:.2f} | "
                        f"MaxPosGT:${cfg.max_position_usd_gt:.2f} | "
                        f"HalfSpread:{cfg.half_spread_usd_gt} | "
                        f"GridInterval:{cfg.grid_interval_usd_gt} | "
                        f"GridNumGT:{cfg.grid_num_gt} | "
                        f"Skew:{cfg.skew_gt}"
                    )
                ),
            },

            # ── BOTH ─────────────────────────────────────────────────────
            TradingMode.BOTH: {
                "brain_factory": staticmethod(
                    lambda logger, cfg: tracker.OrderBookTracker(
                        logger,
                        tracker.TradingMode.BOTH,
                        cfg.tick_size,
                        cfg.lot_size,
                        cfg.order_qty_usd,
                        # QB
                        cfg.max_position_usd_qb,
                        cfg.qty_threshold_qb,
                        cfg.grid_num_qb,
                        # GT
                        cfg.skew_gt,
                        cfg.half_spread_usd_gt,
                        cfg.grid_interval_usd_gt,
                        cfg.max_position_usd_gt,
                        cfg.grid_num_gt,
                    )
                ),
                "config_summary": staticmethod(
                    lambda cfg: (
                        f"Mode:BOTH | Symbol:{cfg.symbol} | "
                        f"QtyUSD:{cfg.order_qty_usd:.2f} | "
                        f"QB→MaxPos:${cfg.max_position_usd_qb:.2f} "
                        f"Thresh:{cfg.qty_threshold_qb} Levels:{cfg.grid_num_qb} | "
                        f"GT→MaxPos:${cfg.max_position_usd_gt:.2f} "
                        f"HSpread:{cfg.half_spread_usd_gt} "
                        f"Interval:{cfg.grid_interval_usd_gt} "
                        f"Levels:{cfg.grid_num_gt} Skew:{cfg.skew_gt}"
                    )
                ),
            },
        }

        return cls


# ═══════════════════════════════════════════════════════════════════════════
# ACTIVE ORDER TRACKING
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ActiveOrder:
    id:         str
    price:      float
    size:       float
    side:       str
    timestamp:  float
    grid_level: int = 0
    algo_tag:   int = tracker.ALGO_GT   # ALGO_QB or ALGO_GT


# ═══════════════════════════════════════════════════════════════════════════
# QUANT BOT — single coin
# ═══════════════════════════════════════════════════════════════════════════

class QuantBot(metaclass=StrategyMeta):
    """Market-making bot for one coin.

    Mode-specific behaviour (C++ brain construction, config logging) is
    dispatched through the StrategyMeta registry rather than if/elif chains.
    """

    def __init__(self, exchange, logger, config: BotConfig):
        self.ex      = exchange
        self.logger  = logger
        self.cfg     = config
        self.running = True

        # Brain and stats are built/rebuilt after fetch_symbol_details()
        self.brain: Optional[tracker.OrderBookTracker] = None
        self.stats = tracker.TradeTracker(logger, config.fee_rate)
        self.pub   = EventPublisher(config.zmq_port, config.jsonl_path, logger)

        self.book = tracker.OrderBook()

        # Active orders — keyed by order ID, split by side
        self.active_bids: Dict[str, ActiveOrder] = {}
        self.active_asks: Dict[str, ActiveOrder] = {}

        # Pending prices (in-flight REST calls) — also carry algo_tag
        self.pending_bids: Dict[float, int] = {}   # price → algo_tag
        self.pending_asks: Dict[float, int] = {}

        self._place_backoff_until = 0.0
        self.last_signal = None
        self.bg_tasks: List[asyncio.Task] = []
        self._tick_n   = 0
        self._tick_gen = 0
        self._exec_sem = asyncio.Semaphore(config.max_concurrent_orders)
        self._pending  = asyncio.Queue()

    # ── Internal dispatch ────────────────────────────────────────────────
    def _dispatch(self, key: str, *args, **kwargs):
        """Call the static method registered for this bot's TradingMode."""
        fn = self._STRATEGY_REGISTRY[self.cfg.mode][key]
        return fn(*args, **kwargs)

    def _rebuild_brain(self):
        self.brain = self._dispatch("brain_factory", self.logger, self.cfg)

    # ── Lifecycle ────────────────────────────────────────────────────────
    async def run(self):
        await self.ex.connect()

        # Fetch exchange tick/lot sizes and adjust config
        details = await self.ex.fetch_symbol_details(self.cfg.symbol)
        if details:
            qi = details["quote_increment"]
            self.cfg.tick_size = qi
            self.cfg.lot_size = 0.1
            self.cfg.min_order_size = details.get("min_order_size", 0.0)
            # Snap GT spread/interval to valid ticks
            self.cfg.half_spread_usd_gt   = max(qi, round(self.cfg.half_spread_usd_gt   / qi) * qi)
            self.cfg.grid_interval_usd_gt = max(qi, round(self.cfg.grid_interval_usd_gt / qi) * qi)
            self.logger.info(
                f"Config adjusted | tick={self.cfg.tick_size} | lot={self.cfg.lot_size} | "
                f"half_spread={self.cfg.half_spread_usd_gt} | grid_interval={self.cfg.grid_interval_usd_gt}"
            )
        else:
            self.logger.warn("Could not fetch symbol details — using config defaults")

        self._rebuild_brain()

        summary = self._dispatch("config_summary", self.cfg)
        self.logger.info(f"Started | {summary}")
        self.pub.system(summary)

        try:
            async with asyncio.TaskGroup() as tg:
                self.bg_tasks = [
                    tg.create_task(self._listen_market()),
                    tg.create_task(self._listen_orders()),
                    tg.create_task(self._monitor_status()),
                    tg.create_task(self._action_executor()),
                ]
        except* asyncio.CancelledError:
            pass
        except* Exception as eg:
            err_lines = [f"Task: {e}" for e in eg.exceptions]
            self.logger.error("\n".join(err_lines))  # ONE boundary crossing
        finally:
            self.logger.warn("Shutdown — cancelling all orders...")
            self.pub.system("Shutdown — cancelling all orders", "warn")
            await self._cancel_all()
            await self.ex.close()
            self.pub.system("Shutdown complete")
            self.pub.close()
            self.logger.warn("Shutdown complete.")

    async def shutdown(self):
        if not self.running:
            return
        self.running = False
        self.logger.warn("Shutdown signal...")
        self.pub.system("Shutdown signal", "warn")
        for t in self.bg_tasks:
            t.cancel()

    # ── Market data ──────────────────────────────────────────────────────
    async def _listen_market(self):
        async for data in self.ex.market_stream(self.cfg.symbol):
            if not self.running:
                break
            mt = data.get("type")
            if mt == "l2_updates":
                changes = [(s, float(p), float(q)) for s, p, q in data.get("changes", [])]
                self.book.update(changes)
                await self._process_tick()
            elif mt == "trade":
                evts = data.get("events", [])
                if not evts and "side" in data:
                    evts = [data]
                for t in evts:
                    a = t.get("amount") or t.get("quantity")
                    s = t.get("side")
                    if s and a:
                        self.brain.on_public_trade(s, float(a))

    # ── Core tick ────────────────────────────────────────────────────────
    async def _process_tick(self):
        sb, sa = self.book.sorted_bids(), self.book.sorted_asks()
        if not sb or not sa:
            return
        self._tick_n   += 1
        self._tick_gen += 1
        inv = self.stats.inventory()

        # Build price + tag lists for active + pending orders
        active_bid_px   = [o.price    for o in self.active_bids.values()]
        active_bid_tags = [o.algo_tag for o in self.active_bids.values()]
        active_ask_px   = [o.price    for o in self.active_asks.values()]
        active_ask_tags = [o.algo_tag for o in self.active_asks.values()]

        # Include pending (in-flight) prices so C++ doesn't re-emit PLACE actions
        for px, tag in self.pending_bids.items():
            active_bid_px.append(px);   active_bid_tags.append(tag)
        for px, tag in self.pending_asks.items():
            active_ask_px.append(px);   active_ask_tags.append(tag)

        sig, actions = self.brain.on_tick(
            sb, sa, inv,
            active_bid_px,
            active_ask_px,
            active_bid_tags,
            active_ask_tags,
        )
        self.last_signal = sig

        if self._tick_n % 5 == 0:
            st = self.stats.get_stats(sig.mid_price).to_dict()
            self.pub.tick(sig.to_dict(), st, len(self.active_bids), len(self.active_asks), len(actions))

        for a in actions:
            await self._pending.put((self._tick_gen, a))

    # ── Action executor ──────────────────────────────────────────────────
    async def _action_executor(self):
        while self.running:
            try:
                gen, action = await asyncio.wait_for(self._pending.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            # Drain whatever else is immediately available in the queue
            items = [(gen, action)]
            while not self._pending.empty():
                try:
                    items.append(self._pending.get_nowait())
                except asyncio.QueueEmpty:
                    break

            stale_logs = []
            for g, act in items:
                if g < self._tick_gen:
                    stale_logs.append(
                        f"SKIP stale gen={g} cur={self._tick_gen} | "
                        f"act={act.action} price={act.price:.6f} reason={act.reason}"
                    )
                else:
                    asyncio.create_task(self._exec(act))

            # Make ONE boundary crossing for all skipped actions
            if stale_logs:
                self.logger.info("\n".join(stale_logs))

    async def _exec(self, action):
        async with self._exec_sem:
            a = action.action
            if   a == tracker.ACTION_PLACE_BID:
                await self._place("buy",  action.price, action.size, action.grid_level, action.algo_tag)
            elif a == tracker.ACTION_PLACE_ASK:
                await self._place("sell", action.price, action.size, action.grid_level, action.algo_tag)
            elif a == tracker.ACTION_CANCEL_BID:
                await self._cancel_side("buy",  action.price, action.algo_tag)
            elif a == tracker.ACTION_CANCEL_ASK:
                await self._cancel_side("sell", action.price, action.algo_tag)
            elif a == tracker.ACTION_CANCEL_ALL:
                await self._cancel_all()

    # ── Order placement ──────────────────────────────────────────────────
    async def _place(self, side: str, price: float, size: float,
                     grid_level: int, algo_tag: int, options=None):
        if time.time() < getattr(self, "_place_backoff_until", 0.0):
            return

        if price <= 0 or size <= 0:
            self.logger.info(f"SKIP {side.upper()} | Invalid target price or size ({price} @ {size})")
            return

        mid = self.last_signal.mid_price if self.last_signal else price

        if side == "buy":
            exposure = self.stats.inventory() * mid
            cap = (self.cfg.max_position_usd_qb if algo_tag == tracker.ALGO_QB
                   else self.cfg.max_position_usd_gt)
            if exposure + size * mid > cap:
                self.logger.info(
                    f"SKIP BUY @ {price:.6f} | pos limit "
                    f"({exposure:.2f} + {size*mid:.2f} > {cap:.2f}) "
                    f"[{'QB' if algo_tag == tracker.ALGO_QB else 'GT'}]"
                )
                return

        if side == "sell":
            inv = self.stats.inventory()
            if inv < size:
                self.logger.info(
                    f"SKIP SELL @ {price:.6f} | insufficient inventory "
                    f"({inv:.4f} < {size:.4f})"
                )
                return

        lot_dec  = Decimal(str(self.cfg.lot_size))
        tick_dec = Decimal(str(self.cfg.tick_size))

        # Snap size down to nearest lot_size and format
        size_dec = (Decimal(str(size)) / lot_dec).quantize(Decimal('1'), rounding=ROUND_DOWN) * lot_dec
        size_str = f"{size_dec.normalize():f}"

        # Snap price to nearest tick_size and format
        price_dec = (Decimal(str(price)) / tick_dec).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * tick_dec
        price_str = f"{price_dec.normalize():f}"

        # Guard against zero-size or zero-price orders after snapping
        if float(size_str) <= 0 or float(price_str) <= 0:
             self.logger.warn(f"SKIP {side.upper()} | size or price snapped to 0. {size_str} @ {price_str}")
             return

        pending = self.pending_bids if side == "buy" else self.pending_asks
        pending[price] = algo_tag

        oid = None
        try:
            # Pass the strings (price_str, size_str) instead of floats
            oid = await self.ex.place_order(self.cfg.symbol, side, price_str, size_str, options)
        except Exception as e:
            self.logger.error(f"Order placement threw exception: {e}")
        finally:
            # --- FIX: Rate limit backoff loop ---
            if not oid:
                # Keep it in pending for 2 seconds to prevent the C++ tracker from immediately firing it again
                asyncio.create_task(self._delayed_pop(pending, price, 2.0))
            else:
                pending.pop(price, None)

        if oid:
            c_side = tracker.Side.BUY if side == "buy" else tracker.Side.SELL
            o = ActiveOrder(oid, price, size, side, time.time(), grid_level, algo_tag)
            (self.active_bids if side == "buy" else self.active_asks)[oid] = o
            self.stats.on_order_submitted(c_side, size)
            self.pub.order("PLACED", oid, side, price, size,
                           grid_level=grid_level,
                           options=options or ["maker-or-cancel"],
                           algo="QB" if algo_tag == tracker.ALGO_QB else "GT")

            # Mock exchange: simulate immediate fill
            if isinstance(self.ex, MockExchange):
                r = self.stats.on_fill(c_side, price, size)
                self.pub.order("FILLED", oid, side, price, size)
                self.pub.fill(side, price, size, oid, r.to_dict())
                (self.active_bids if side == "buy" else self.active_asks).pop(oid, None)
                self.stats.on_order_cancelled(c_side)
        else:
            self.pub.order("REJECTED", "", side, price, size)

    async def _delayed_pop(self, pending_dict: dict, price: float, delay: float):
        """Removes an item from pending_dict after a delay to throttle rejected orders."""
        await asyncio.sleep(delay)
        pending_dict.pop(price, None)

    # ── Order cancellation ───────────────────────────────────────────────
    async def _cancel_side(self, side: str, price: float, algo_tag: int):
        c_side = tracker.Side.BUY if side == "buy" else tracker.Side.SELL
        orders = self.active_bids if side == "buy" else self.active_asks
        tol    = self.cfg.tick_size * 0.5
        to_cancel = [
            oid for oid, o in orders.items()
            if abs(o.price - price) < tol and o.algo_tag == algo_tag
        ]
        for oid in to_cancel:
            if await self.ex.cancel_order(oid):
                o = orders.pop(oid, None)
                self.stats.on_order_cancelled(c_side)
                if o:
                    self.pub.order("CANCELLED", oid, side, o.price, o.size,
                                   algo="QB" if algo_tag == tracker.ALGO_QB else "GT")

    async def _cancel_all(self):
        all_o = list(self.active_bids.values()) + list(self.active_asks.values())
        if all_o:
            res = await asyncio.gather(
                *[self.ex.cancel_order(o.id) for o in all_o],
                return_exceptions=True,
            )
            for o, r in zip(all_o, res):
                if r is True:
                    self.pub.order("CANCELLED", o.id, o.side, o.price, o.size, reason="cancel_all")
        self.active_bids.clear()
        self.active_asks.clear()
        self.pending_bids.clear()
        self.pending_asks.clear()
        self.logger.warn(f"Cancelled {len(all_o)} orders.")

    # ── Order events ─────────────────────────────────────────────────────
    async def _listen_orders(self):
        async for msg in self.ex.order_stream():
            if not self.running:
                break
            ms = [msg] if isinstance(msg, dict) else (msg if isinstance(msg, list) else [])
            for m in ms:
                await self._handle_evt(m)

    async def _handle_evt(self, msg: dict):
        if msg.get("symbol") and msg.get("symbol").upper() != self.cfg.symbol.upper():
            return
        et   = msg.get("type")
        oid  = str(msg.get("order_id", ""))
        side = msg.get("side", "unknown")
        price = float(msg.get("price") or 0.0)

        if et == "fill":
            fill_data  = msg.get("fill", {})
            size       = float(fill_data.get("amount", 0) if fill_data else 0)
            fill_price = float(fill_data.get("price")  or price or 0.0)
            if size <= 0:
                size = float(msg.get("fill_amount") or msg.get("amount", 0))
            if fill_price <= 0:
                fill_price = price

            if size > 0 and fill_price > 0:
                c_side = tracker.Side.BUY if side == "buy" else tracker.Side.SELL
                r = self.stats.on_fill(c_side, fill_price, size)
                self.logger.decision(
                    f"FILL {side.upper()} {size:.8f} @ {fill_price:.6f} "
                    f"PnL:{r.trade_pnl:.8f} Inv:{r.inventory_after:.8f} Eq:{r.equity_after:.8f}"
                )
                self.pub.order("FILLED", oid, side, fill_price, size)
                self.pub.fill(side, fill_price, size, oid, r.to_dict())

                remaining = float(msg.get("remaining_amount", -1))
                if remaining == 0:
                    (self.active_bids if side == "buy" else self.active_asks).pop(oid, None)
                    self.logger.info(f"ORDER FULLY FILLED | {side.upper()} [{oid}] — removed")
            else:
                self.logger.warn(
                    f"FILL PARSE FAIL | {side.upper()} [{oid}] size={size} price={fill_price}"
                )

        elif et in ("accepted", "booked"):
            size = float(msg.get("original_amount") or msg.get("amount", 0))
            self.logger.info(f"ORDER {et.upper()} | {side.upper()} {size:.2f} @ {price:.6f} [{oid}]")
            self.pub.order(et.upper(), oid, side, price, size)

        elif et in ("cancelled", "closed"):
            size = float(msg.get("original_amount") or msg.get("amount", 0))
            self.logger.info(f"ORDER {et.upper()} | {side.upper()} {size:.2f} @ {price:.6f} [{oid}]")
            self.pub.order(et.upper(), oid, side, price, size)
            (self.active_bids if side == "buy" else self.active_asks).pop(oid, None)

        elif et == "rejected":
            reason = msg.get("reason", "Unknown")
            self.logger.error(f"ORDER REJECTED | [{oid}] Reason: {reason}")
            self.pub.order("REJECTED", oid, side, price, 0, reason=reason)

    # ── Status reporting ─────────────────────────────────────────────────
    async def _monitor_status(self):
        while self.running:
            await asyncio.sleep(self.cfg.status_interval)
            mid = self.last_signal.mid_price if self.last_signal else 0
            st  = self.stats.get_stats(mid) if mid > 0 else None

            if st:
                self.pub.status(st.to_dict(), self.last_signal,
                                self.active_bids, self.active_asks)

            # 1. Use a list to accumulate all log lines
            log_lines = []

            # 2. Build the main status line
            status_line = f"STATUS [{self.cfg.symbol}] | Bids:{len(self.active_bids)} Asks:{len(self.active_asks)}"
            if st:
                status_line += (
                    f" | Eq:${st.equity:.8f} | Inv:{st.inventory:.8f} | "
                    f"AvgEntry:{st.avg_entry:.6f} | RPnL:{st.realized_pnl:.8f} | "
                    f"UPnL:{st.unrealized_pnl:.8f} | TotPnL:{st.total_pnl:.8f} | "
                    f"DD:${st.max_drawdown:.8f} | Fees:${st.total_fees:.8f} | "
                    f"Vol:${st.volume:.8f} | WR:{st.win_rate:.1f}% | "
                    f"Fills:{int(st.fills)}({int(st.buy_fills)}B/{int(st.sell_fills)}S) | "
                    f"PF:{st.profit_factor:.3f} | Exp:{st.expectancy:.8f}"
                )
            if self.last_signal:
                status_line += f" | Mid:{self.last_signal.mid_price:.6f} | Spr:{self.last_signal.spread:.8f}"

            log_lines.append(status_line)

            # 3. Process orders (only sort if we actually have them)
            if self.active_bids or self.active_asks:
                log_lines.append(f"--- ORDERS [{self.cfg.symbol}] ---")

                # OPTIMIZATION: Use reverse=True instead of reversed(sorted(...))
                if self.active_asks:
                    sorted_asks = sorted(self.active_asks.values(), key=lambda x: x.price, reverse=True)
                    for o in sorted_asks:
                        algo = 'QB' if o.algo_tag == tracker.ALGO_QB else 'GT'
                        log_lines.append(f"ASK {algo} | {o.price:.6f} | {o.size:.1f} | {o.id}")
                    log_lines.append("---")

                if self.active_bids:
                    # Negating the price gives descending order
                    sorted_bids = sorted(self.active_bids.values(), key=lambda x: -x.price)
                    for o in sorted_bids:
                        algo = 'QB' if o.algo_tag == tracker.ALGO_QB else 'GT'
                        log_lines.append(f"BID {algo} | {o.price:.6f} | {o.size:.1f} | {o.id}")
                    log_lines.append("---")

            # 4. Make ONE single boundary-crossing call to C++
            self.logger.info("\n".join(log_lines))

# ═══════════════════════════════════════════════════════════════════════════
# MULTI-COIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════

class MultiCoinRunner:
    """Runs N QuantBot instances concurrently (one per coin).

    Each bot has its own exchange connection, logger, and ZMQ port so they
    are fully independent at the I/O layer.  They do share the same process
    and the same C++ AsyncLogger infrastructure.

    Notebook mapping:
        coins[0]  →  USDCUSD   (index 0 in malogtqb_mm)
        coins[1]  →  USDTUSD   (index 1 in malogtqb_mm)
    """

    def __init__(self, configs: List[BotConfig], exchange_factory, logger_factory):
        """
        Args:
            configs:           One BotConfig per coin.
            exchange_factory:  Callable(logger, config) → exchange instance.
            logger_factory:    Callable(tag: str)       → tracker.AsyncLogger.
        """
        self.configs          = configs
        self.exchange_factory = exchange_factory
        self.logger_factory   = logger_factory
        self.bots: List[QuantBot] = []

    async def run(self):
        # ✅ ADDED the signal handler here, controlling the runner's shutdown
        loop = asyncio.get_running_loop()
        for sig in (os_signal.SIGINT, os_signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
            except NotImplementedError:
                pass

        for i, cfg in enumerate(self.configs):
            log = self.logger_factory(f"qb_{cfg.symbol.lower()}_{cfg.mode.name.lower()}")
            ex  = self.exchange_factory(log, cfg)
            self.bots.append(QuantBot(ex, log, cfg))

        try:
            async with asyncio.TaskGroup() as tg:
                for bot in self.bots:
                    tg.create_task(bot.run())
        except* asyncio.CancelledError:
            pass
        except* Exception as eg:
            for e in eg.exceptions:
                print(f"[MultiCoinRunner] Task error: {e}", file=sys.stderr)

    async def shutdown(self):
        for bot in self.bots:
            await bot.shutdown()


# ═══════════════════════════════════════════════════════════════════════════
# CLI HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _load(fn: str) -> Optional[str]:
    try:
        with open(os.path.expanduser(fn)) as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def resolve_keys(args) -> tuple[Optional[str], Optional[str]]:
    k = args.key or (
        _load("~/.gemini/gak.txt") if args.exchange_mode == "live"
        else _load("~/.gemini/gak_sb.txt"))
    s = args.secret or (
        _load("~/.gemini/gas.txt") if args.exchange_mode == "live"
        else _load("~/.gemini/gas_sb.txt"))
    return k, s


def build_configs(args) -> List[BotConfig]:
    """Build one BotConfig per symbol from CLI args, using notebook defaults."""
    mode = TradingMode[args.trading_mode.upper()]

    # Notebook defaults per coin (index 0 = USDCUSD, index 1 = USDTUSD)
    notebook_defaults = {
        "USDCUSD": dict(
            max_position_usd_qb  = 10.0,
            qty_threshold_qb     = 10.0,
            grid_num_qb          = 1,
            max_position_usd_gt  = 10.0,
            skew_gt              = 0.0,
            grid_num_gt          = 10,
            grid_interval_usd_gt = 1e-4,
            half_spread_usd_gt   = 0.001,
        ),
        "USDTUSD": dict(
            max_position_usd_qb  = 10.0,
            qty_threshold_qb     = 10.0,
            grid_num_qb          = 1,
            max_position_usd_gt  = 10.0,
            skew_gt              = 0.0,
            grid_num_gt          = 10,
            grid_interval_usd_gt = 1e-4,
            half_spread_usd_gt   = 0.0001,
        ),
    }

    symbols = args.symbols if args.symbols else ["USDCUSD", "USDTUSD"]
    configs = []
    for i, sym in enumerate(symbols):
        defaults = notebook_defaults.get(sym.upper(), notebook_defaults["USDCUSD"])
        cfg = BotConfig(
            symbol   = sym.upper(),
            mode     = mode,
            zmq_port = (args.zmq_port or 5557) + i,
            **defaults,
        )
        # CLI overrides (apply to all coins — fine for symmetric algorithms)
        if args.max_pos_qb  is not None: cfg.max_position_usd_qb  = args.max_pos_qb
        if args.max_pos_gt  is not None: cfg.max_position_usd_gt  = args.max_pos_gt
        if args.grid_num_qb is not None: cfg.grid_num_qb          = args.grid_num_qb
        if args.grid_num_gt is not None: cfg.grid_num_gt          = args.grid_num_gt
        if args.qty_thresh  is not None: cfg.qty_threshold_qb     = args.qty_thresh
        if args.skew        is not None: cfg.skew_gt              = args.skew
        if args.half_spread is not None: cfg.half_spread_usd_gt   = args.half_spread
        if args.grid_int    is not None: cfg.grid_interval_usd_gt = args.grid_int
        configs.append(cfg)
    return configs


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="QB Multi-Coin MM Bot")

    # Exchange mode
    ap.add_argument("exchange_mode", choices=["mock", "paper", "live"],
                    help="mock=simulated, paper=sandbox, live=real money")

    # Trading mode
    ap.add_argument("--trading-mode",
                    choices=["queue_based", "grid_trading", "both"],
                    default="queue_based",
                    help="Strategy mode (default: both)")

    # Coins
    ap.add_argument("--symbols", nargs="+", default=None,
                    metavar="SYM",
                    help="Symbols to trade (default: USDCUSD USDTUSD)")

    # Auth
    ap.add_argument("--key");    ap.add_argument("--secret")

    # Shared
    ap.add_argument("--qty-usd",   type=float, default=None, help="USD per order level")
    ap.add_argument("--fee-rate",  type=float, default=None, help="Taker fee rate (default 0.0001)")
    ap.add_argument("--zmq-port",  type=int,   default=None, help="Base ZMQ port (coin N gets port+N)")

    # QB
    ap.add_argument("--max-pos-qb",  type=float, default=None, help="QB max position USD")
    ap.add_argument("--grid-num-qb", type=int,   default=None, help="QB grid levels per side")
    ap.add_argument("--qty-thresh",  type=float, default=None, help="QB queue depth step-back threshold")

    # GT
    ap.add_argument("--max-pos-gt",  type=float, default=None, help="GT max position USD")
    ap.add_argument("--grid-num-gt", type=int,   default=None, help="GT grid levels per side")
    ap.add_argument("--skew",        type=float, default=None, help="GT inventory skew multiplier")
    ap.add_argument("--half-spread", type=float, default=None, help="GT half-spread USD")
    ap.add_argument("--grid-int",    type=float, default=None, help="GT grid interval USD")

    args = ap.parse_args()

    configs = build_configs(args)

    def logger_factory(tag: str) -> tracker.AsyncLogger:
        return tracker.AsyncLogger(tag)

    if args.exchange_mode == "mock":
        def exchange_factory(log, cfg):
            return MockExchange(log)
    else:
        k, s = resolve_keys(args)
        if not k or not s:
            sys.exit("No API keys found.")
        def exchange_factory(log, cfg):
            return GeminiExchange(k, s, log, sandbox=(args.exchange_mode == "paper"))

    root_log = tracker.AsyncLogger("qb_live_confirm")
    root_log.info(f"\n{args=}\n\n{configs=}\n")

    if args.exchange_mode == "live":
        root_log.warn("*** REAL MONEY — ALL COINS ***")
        if input("Type 'I understand': ").strip() != "I understand":
            sys.exit(0)

    runner = MultiCoinRunner(configs, exchange_factory, logger_factory)

    try:
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        print("Interrupted.")


if __name__ == "__main__":
    main()
