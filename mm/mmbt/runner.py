"""
runner.py — BacktestRunner and BacktestResult
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hftbacktest import ROIVectorMarketDepthBacktest, Recorder
from hftbacktest.stats import LinearAssetRecord

from .asset import AssetConfig
from .charts import (
    build_single_asset_dashboard, build_multi_asset_dashboard,
    plot_pnl_single, plot_pnl_combined,
    plot_inventory_single, plot_quotes_single,
    _extract_pnl_series, _extract_pnl_from_rec,
    _time_mask, _zoom_mm, _C, _show,
)
from .metrics import DEFAULT_METRICS, MetricsRegistry, extract_summary
from .progress import BacktestProgressBar

log = logging.getLogger(__name__)


# ── Date / interval helpers ──────────────────────────────────────────────────

def _estimate_days(assets: List[AssetConfig]) -> float:
    """Estimate total backtest span in days from start/end dates."""
    if not assets:
        return 2.0
    try:
        from .data.utils import _parse_date_hour
        start_dt, start_h = _parse_date_hour(assets[0].start_date)
        end_dt, end_h = _parse_date_hour(assets[0].end_date)
        start_h = start_h if start_h is not None else 0
        end_h = end_h if end_h is not None else 23
        total_hours = (end_dt - start_dt).days * 24 + (end_h - start_h) + 1
        return max(total_hours / 24.0, 0.1)
    except Exception:
        return 2.0


def _estimate_recorder_capacity(days: float, interval_ns: int) -> int:
    """Estimate recorder capacity from days and strategy interval."""
    steps_per_day = 86_400 * 1_000_000_000 / max(interval_ns, 1)
    return math.ceil(steps_per_day * days) + 10_000


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """All outputs of a completed backtest run."""

    stats: Any
    stats_list: List[Any]
    records: List[np.ndarray]
    metrics: dict
    assets: List[AssetConfig]
    book_size: float
    elapsed_s: float
    mm_data_list: List[Optional[dict]] = field(default_factory=list)
    order_history_list: List[Optional[dict]] = field(default_factory=list)

    @property
    def mm_data(self) -> Optional[dict]:
        return self.mm_data_list[0] if self.mm_data_list else None

    def save(self, name=None, base_dir="results_backtest",
             dpi=150, verbose=False, params=None) -> Path:
        from .saver import save_result
        return save_result(self, name=name, base_dir=base_dir,
                           dpi=dpi, verbose=verbose, params=params)

    def time_range_ns(self, start: str, end: str) -> tuple:
        """Convert 'HH:MM' or 'HH:MM:SS' to (start_ns, end_ns)."""
        date_str = self.assets[0].start_date.split("_")[0]
        base_date = datetime.strptime(date_str, "%Y%m%d").date()

        def _p(t: str) -> int:
            parts = t.split(":")
            h, m = int(parts[0]), int(parts[1])
            s = int(parts[2]) if len(parts) > 2 else 0
            dt = datetime(base_date.year, base_date.month, base_date.day,
                          h, m, s, tzinfo=timezone.utc)
            return int(dt.timestamp() * 1e9)

        return _p(start), _p(end)

    # ── charts ─────────────────────────────────────────────────────────

    def plot(self, asset_no=0, downsample=4, figsize=(20, 18),
             show=True, time_range=None):
        if len(self.assets) == 1:
            fig = build_single_asset_dashboard(
                stats_obj=self.stats, rec=self.records[0],
                mm_data=self.mm_data_list[0] if self.mm_data_list else None,
                symbol=self.assets[0].symbol,
                start_date=self.assets[0].start_date,
                end_date=self.assets[0].end_date,
                downsample=downsample, figsize=figsize, time_range=time_range,
            )
        else:
            fig = build_multi_asset_dashboard(
                stats_list=self.stats_list, records=self.records,
                mm_data_list=self.mm_data_list,
                symbols=[a.symbol for a in self.assets],
                start_date=self.assets[0].start_date,
                end_date=self.assets[0].end_date,
                downsample=downsample, figsize=figsize, time_range=time_range,
            )
        return _show(fig, show)

    def plot_asset(self, asset_no=0, downsample=4, figsize=(20, 18),
                   show=True, time_range=None):
        mm = self.mm_data_list[asset_no] if asset_no < len(self.mm_data_list) else None
        so = self.stats_list[asset_no] if asset_no < len(self.stats_list) else self.stats
        fig = build_single_asset_dashboard(
            stats_obj=so, rec=self.records[asset_no], mm_data=mm,
            symbol=self.assets[asset_no].symbol,
            start_date=self.assets[asset_no].start_date,
            end_date=self.assets[asset_no].end_date,
            downsample=downsample, figsize=figsize, time_range=time_range,
        )
        return _show(fig, show)

    def plot_pnl(self, asset_no=0, show=True, time_range=None, **kw):
        so = self.stats_list[asset_no] if asset_no < len(self.stats_list) else self.stats
        ts, net, gross, df = _extract_pnl_series(so)
        if ts is None:
            ts, net, gross, df = _extract_pnl_from_rec(self.records[asset_no])
        mask = _time_mask(ts, time_range)
        fig, ax = plt.subplots(figsize=kw.get("figsize", (16, 6)))
        fig.patch.set_facecolor(_C["bg"])
        plot_pnl_single(ax, ts[mask], net[mask], gross[mask], df[mask])
        plt.tight_layout()
        return _show(fig, show)

    def plot_inventory(self, asset_no=0, show=True, time_range=None, **kw):
        rec = self.records[asset_no]
        mask = _time_mask(rec["timestamp"].astype(np.int64), time_range)
        fig, ax = plt.subplots(figsize=kw.get("figsize", (16, 5)))
        fig.patch.set_facecolor(_C["bg"])
        plot_inventory_single(ax, rec["timestamp"][mask].astype(np.int64),
                              rec["position"][mask], label=self.assets[asset_no].symbol)
        plt.tight_layout()
        return _show(fig, show)

    def plot_quotes(self, asset_no=0, downsample=4, show=True, time_range=None, **kw):
        mm = self.mm_data_list[asset_no] if asset_no < len(self.mm_data_list) else None
        if not mm:
            log.warning("No mm_data for asset %d — quotes plot skipped.", asset_no)
            return None
        mm_z = _zoom_mm(mm, time_range)
        if mm_z is None:
            return None
        fig, ax = plt.subplots(figsize=kw.get("figsize", (16, 6)))
        fig.patch.set_facecolor(_C["bg"])
        plot_quotes_single(ax, mm_z["timestamps"], mm_z["mid_prices"],
                           mm_z["bid_quotes"], mm_z["ask_quotes"],
                           mm_z["positions"], label=self.assets[asset_no].symbol,
                           downsample=downsample)
        plt.tight_layout()
        return _show(fig, show)

    def plot_combined_pnl(self, show=True, **kw):
        fig, ax = plt.subplots(figsize=kw.get("figsize", (16, 6)))
        fig.patch.set_facecolor(_C["bg"])
        plot_pnl_combined(ax, self.stats_list, self.records, [a.symbol for a in self.assets])
        plt.tight_layout()
        return _show(fig, show)

    def fills_df(self, asset_no: int = 0) -> pd.DataFrame:
        """Reconstruct fills from recorder position changes."""
        rec = self.records[asset_no]
        ts = rec["timestamp"].astype(np.int64)
        pos = rec["position"].astype(np.float64)
        bal = rec["balance"].astype(np.float64)
        fee = rec["fee"].astype(np.float64)
        n_tr = rec["num_trades"].astype(np.int64)

        d_pos = np.diff(pos, prepend=pos[0])
        d_trades = np.diff(n_tr, prepend=n_tr[0])
        d_bal = np.diff(bal, prepend=bal[0])
        d_fee = np.diff(fee, prepend=fee[0])
        mask = d_trades > 0

        if not mask.any():
            return pd.DataFrame(columns=["datetime", "side", "qty", "net_cash_flow", "fee"])

        idx = np.where(mask)[0]
        dts = [datetime.fromtimestamp(t / 1e9, tz=timezone.utc) for t in ts[idx]]
        qty = np.abs(d_pos[idx])
        net_cf = d_bal[idx]
        fee_cf = d_fee[idx]
        sides = ["BUY" if d > 0 else "SELL" for d in d_pos[idx]]
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fill_price = np.where(qty > 0, np.abs(net_cf) / qty, np.nan)
        return pd.DataFrame({
            "datetime": dts,
            "timestamp_ns": ts[idx],
            "side": sides,
            "qty": qty,
            "fill_price": fill_price,
            "net_cash_flow": net_cf,
            "fee_at_fill": fee_cf,
        })


# ── Runner ────────────────────────────────────────────────────────────────────

class BacktestRunner:
    """
    Run a backtest.

        runner = BacktestRunner(
            assets   = [AssetConfig(...)],
            strategy = AvellanedaStoikovStrategy(...),
            book_size= 2000.0,
        )
        result = runner.run()

    Parameters
    ----------
    assets            : list of AssetConfig
    strategy          : any BaseStrategy subclass instance
    book_size         : capital for metric normalisation
    metrics           : tuple of metric classes (default: DEFAULT_METRICS)
    extra_metrics     : extra metric classes
    recorder_capacity : hftbacktest Recorder pre-allocation
                        (auto-estimated from dates + interval if None)
    show_progress     : show a live progress bar during the run (default True)
    """

    def __init__(
            self,
            assets: List[AssetConfig],
            strategy,
            book_size: float = 10_000.0,
            metrics: tuple = DEFAULT_METRICS,
            extra_metrics: tuple = (),
            recorder_capacity: Optional[int] = None,
            show_progress: bool = True,
    ) -> None:
        if not assets:
            raise ValueError("At least one AssetConfig is required.")
        self.assets = assets
        self.strategy = strategy
        self.book_size = book_size
        self.registry = MetricsRegistry(base=metrics, extra=extra_metrics)
        self.show_progress = show_progress

        # Auto-estimate recorder capacity if not provided
        days = _estimate_days(assets)
        interval_ns = getattr(strategy, "interval", 100_000_000)
        self.recorder_capacity = recorder_capacity or _estimate_recorder_capacity(days, interval_ns)

        # Inject n_assets and days into strategy (replaces old manual params)
        strategy._configure(n_assets=len(assets), days=days)

    def run(self) -> BacktestResult:
        symbols = ", ".join(a.symbol.upper() for a in self.assets)
        start_date = self.assets[0].start_date
        end_date = self.assets[0].end_date
        strategy = type(self.strategy).__name__

        days = _estimate_days(self.assets)
        interval_ns = getattr(self.strategy, "interval", 100_000_000)
        total_steps = _estimate_recorder_capacity(days, interval_ns)

        log.info(
            "Backtest starting  |  strategy=%s  assets=%s  period=%s to %s  "
            "book_size=%.0f  estimated_steps=%d",
            strategy, symbols, start_date, end_date,
            self.book_size, total_steps,
        )

        t0 = time.perf_counter()
        hbt_assets = [cfg.build() for cfg in self.assets]
        n_assets = len(hbt_assets)
        hbt = ROIVectorMarketDepthBacktest(hbt_assets)
        recorder = Recorder(n_assets, self.recorder_capacity)

        if self.show_progress:
            bar = BacktestProgressBar(
                total_steps=total_steps,
                desc=f"{strategy} | {symbols}",
            )
            self.strategy._progress_bar = bar
            bar.start()

        self.strategy.run(hbt, recorder.recorder)

        if self.show_progress:
            bar.finish()
            if hasattr(self.strategy, "_progress_bar"):
                del self.strategy._progress_bar

        hbt.close()

        elapsed = time.perf_counter() - t0
        records = [recorder.get(i) for i in range(n_assets)]
        combined = _combine(records)
        stats = LinearAssetRecord(combined).stats(
            metrics=self.registry.metrics, book_size=self.book_size)
        metrics_dict = extract_summary(stats)
        stats_list = [
            LinearAssetRecord(rec).stats(
                metrics=self.registry.metrics, book_size=self.book_size)
            for rec in records
        ]

        mm_data_per_asset = self.strategy.mm_data_all()
        oh_data_per_asset = self.strategy.order_history_all()

        result = BacktestResult(
            stats=stats,
            stats_list=stats_list,
            records=records,
            metrics=metrics_dict,
            assets=self.assets,
            book_size=self.book_size,
            elapsed_s=elapsed,
            mm_data_list=mm_data_per_asset,
            order_history_list=oh_data_per_asset,
        )

        _log_run_summary(result, elapsed, strategy, symbols, start_date, end_date)
        return result


def _log_run_summary(result: BacktestResult, elapsed: float,
                     strategy: str, symbols: str,
                     start_date: str, end_date: str) -> None:
    sep = "=" * 60
    log.info(sep)
    log.info("Backtest complete  |  strategy=%s  assets=%s  runtime=%.2fs",
             strategy, symbols, elapsed)
    log.info("Period: %s to %s  |  book_size=%.0f",
             start_date, end_date, result.book_size)
    if result.metrics:
        log.info("Metrics:")
        for k, v in result.metrics.items():
            log.info("  %-30s %.6f", k, v)
    log.info(sep)


def _combine(records):
    if len(records) == 1:
        return records[0]
    c = records[0].copy()
    for r in records[1:]:
        for f in ("position", "balance", "fee", "num_trades", "trading_volume", "trading_value"):
            if f in c.dtype.names:
                c[f] += r[f]
    return c
