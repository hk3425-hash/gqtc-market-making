"""
logger.py — Post-run structured event log (per-asset).

All timestamps are backtest replay time (nanosecond epoch), not wall-clock.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .runner import BacktestResult

_LOG_FMT = "%(asctime)s  %(levelname)-8s  %(message)s"
_TIME_FMT = "%H:%M:%S.%f"


def _ns_to_str(ts_ns: int) -> str:
    dt = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d}"


def _make_logger(path: Path, console: bool) -> logging.Logger:
    name = f"mmbt.run.{path.stem}.{id(path)}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()

    fmt = logging.Formatter(_LOG_FMT, datefmt=_TIME_FMT)

    fh = logging.FileHandler(path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger


def _order_history_to_dfs(oh: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    subs_data = oh.get("submissions", {})
    cans_data = oh.get("cancellations", {})

    if subs_data and subs_data.get("timestamp_ns"):
        subs_df = pd.DataFrame(subs_data)
    else:
        subs_df = pd.DataFrame(
            columns=["timestamp_ns", "side", "price", "qty", "order_id"]
        )

    if cans_data and cans_data.get("timestamp_ns"):
        cans_df = pd.DataFrame(cans_data)
    else:
        cans_df = pd.DataFrame(columns=["timestamp_ns", "order_id"])

    return subs_df, cans_df


def generate_run_log(
        result: "BacktestResult",
        output_path: Path | str,
        asset_no: int = 0,
        console: bool = False,
) -> Path:
    """
    Write a chronological event log for a single asset.

    Parameters
    ----------
    result      : BacktestResult from BacktestRunner.run()
    output_path : destination path for the .log file
    asset_no    : which asset to log (default 0)
    console     : if True, also stream INFO-level events to stdout
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log = _make_logger(output_path, console)
    rec = result.records[asset_no]
    symbol = result.assets[asset_no].symbol.upper()
    start_d = result.assets[asset_no].start_date
    end_d = result.assets[asset_no].end_date

    log.info("=" * 70)
    log.info("  BACKTEST LOG  --  %s  %s to %s", symbol, start_d, end_d)
    log.info("=" * 70)
    log.info(
        "  Book size: $%s   Runtime: %.1fs   "
        "All timestamps are BACKTEST REPLAY TIME (UTC)",
        f"{result.book_size:,.0f}", result.elapsed_s,
    )
    log.info("=" * 70)

    has_oh = (
            bool(result.order_history_list)
            and asset_no < len(result.order_history_list)
            and bool(result.order_history_list[asset_no])
    )

    subs_df = pd.DataFrame()
    cans_df = pd.DataFrame()
    fills_df = pd.DataFrame()

    if has_oh:
        oh = result.order_history_list[asset_no]
        subs_df, cans_df = _order_history_to_dfs(oh)
        fills_df = result.fills_df(asset_no=asset_no)

    mm_data = (
        result.mm_data_list[asset_no]
        if result.mm_data_list and asset_no < len(result.mm_data_list)
        else None
    )

    if mm_data is not None:
        ts_arr = mm_data["timestamps"]

        def _lu(df: pd.DataFrame, col: str = "timestamp_ns") -> dict:
            lu: dict = {}
            if df.empty:
                return lu
            for row in df.itertuples(index=False):
                t = int(getattr(row, col))
                lu.setdefault(t, []).append(row)
            return lu

        sub_lu = _lu(subs_df)
        can_lu = _lu(cans_df)
        fill_lu = _lu(fills_df)

        cancel_prices: dict[int, float] = {}
        if not subs_df.empty and "order_id" in subs_df.columns:
            cancel_prices = dict(zip(subs_df["order_id"], subs_df["price"]))

        for ts in ts_arr:
            ts_i = int(ts)

            for row in sub_lu.get(ts_i, []):
                log.info(
                    "[%s]  SUBMIT  %-4s  id=%-12s  price=%10.4f  qty=%.4f",
                    _ns_to_str(ts_i), row.side, row.order_id, row.price, row.qty,
                )

            for row in can_lu.get(ts_i, []):
                sp = cancel_prices.get(row.order_id, float("nan"))
                log.info(
                    "[%s]  CANCEL  id=%-12s  submit_price=%10.4f",
                    _ns_to_str(ts_i), row.order_id, sp,
                )

            for row in fill_lu.get(ts_i, []):
                log.info(
                    "[%s]  FILL    %-4s  qty=%-10.4f  net_cf=%+10.4f  fee=%+8.4f",
                    _ns_to_str(ts_i), row.side, row.qty,
                    row.net_cash_flow, row.fee_at_fill,
                )

    # --- Session summary ---
    log.info("")
    log.info("=" * 70)
    log.info("  SESSION SUMMARY  —  %s", symbol)
    log.info("=" * 70)

    try:
        t_start = int(rec["timestamp"][0])
        t_end = int(rec["timestamp"][-1])
        log.info("  Backtest start  : %s", _ns_to_str(t_start))
        log.info("  Backtest end    : %s", _ns_to_str(t_end))
    except (IndexError, KeyError):
        pass

    if not fills_df.empty:
        n_buys = (fills_df["side"] == "BUY").sum()
        n_sells = (fills_df["side"] == "SELL").sum()
        log.info("  Fills           : %d  (%d buys / %d sells)",
                 len(fills_df), n_buys, n_sells)
        log.info("  Total fees      : %+.4f USD", fills_df["fee_at_fill"].sum())

    if not subs_df.empty:
        log.info("  Orders submitted: %d", len(subs_df))
    if not cans_df.empty:
        log.info("  Orders cancelled: %d", len(cans_df))

    try:
        final_bal = float(rec["balance"][-1])
        final_fee = float(rec["fee"][-1])
        final_pos = float(rec["position"][-1])
        eqwf = (float(rec["equity_wo_fee"][-1])
                if "equity_wo_fee" in rec.dtype.names
                else final_bal + final_pos * float(rec["price"][-1]))
        log.info("  Final position  : %+.4f", final_pos)
        log.info("  Gross PnL       : %+.4f USD  (equity_wo_fee)", eqwf)
        log.info("  Cumulative fees : %+.4f USD", final_fee)
        log.info("  Net PnL         : %+.4f USD", eqwf - final_fee)
    except (IndexError, ValueError, KeyError):
        pass

    from .metrics import extract_summary
    try:
        so = result.stats_list[asset_no]
        asset_metrics = extract_summary(so)
    except (IndexError, AttributeError):
        asset_metrics = result.metrics

    if asset_metrics:
        log.info("")
        log.info("  METRICS  (fractions of book_size unless noted)")
        log.info("  " + "-" * 50)
        for k, v in asset_metrics.items():
            if isinstance(v, float):
                log.info("  %-30s %.6f", k, v)
            else:
                log.info("  %-30s %s", k, v)

    log.info("=" * 70)

    for h in log.handlers:
        h.flush()
        h.close()

    return output_path
