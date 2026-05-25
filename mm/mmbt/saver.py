"""
saver.py — Persist all backtest artefacts to disk.

Directory layout
----------------
results_backtest/
  {run_name}_{YYYYMMDD_HHMMSS}/
      ├── dashboard.png          ← full multi-asset overview
      ├── pnl_combined.png       ← (multi-asset only)
      ├── reports/
      │     summary.txt
      │     params.json          ← if params= supplied
      │     metrics.json
      │     metrics.csv
      └── asset_{i}_{sym}/
            ├── data/
            │     orders_submitted.csv
            │     orders_cancelled.csv
            │     trades.csv
            ├── logs/
            │     run.log        ← per-asset event log
            ├── plots/
            │     dashboard.png  ← single-asset overview
            │     pnl.png
            │     inventory.png
            │     quotes.png
            └── reports/
                  summary.txt
                  metrics.json
                  metrics.csv
"""

from __future__ import annotations

import csv
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

if TYPE_CHECKING:
    from .runner import BacktestResult

logger = logging.getLogger(__name__)

_METRIC_UNITS: dict[str, str] = {
    "SR": "annualised Sharpe ratio",
    "Sortino": "annualised Sortino ratio",
    "Return": "total return (fraction of book_size)",
    "MaxDrawdown": "maximum drawdown (fraction of book_size)",
    "DailyNumberOfTrades": "average trades per calendar day",
    "DailyTurnover": "average daily turnover (fraction of book_size)",
    "ReturnOverMDD": "return / max drawdown",
    "ReturnOverTrade": "return per trade (fraction of book_size)",
    "MaxPositionValue": "maximum absolute position value (USD)",
    "DailyTradingVolume": "average daily trading volume (base currency)",
    "DailyTradingValue": "average daily trading value (USD)",
}


def save_result(
        result: "BacktestResult",
        name: Optional[str] = None,
        base_dir: str | Path = "results_backtest",
        dpi: int = 150,
        verbose: bool = False,
        params: Optional[dict] = None,
) -> Path:
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = name or f"{result.assets[0].symbol}_{result.assets[0].start_date}"
    run_dir = Path(base_dir) / f"{run_name}_{ts_str}"
    n_assets = len(result.assets)

    # ── Global directories ────────────────────────────────────────────────
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Global reports ────────────────────────────────────────────────────
    (reports_dir / "summary.txt").write_text(_build_summary(result), encoding="utf-8")
    logger.info("Summary written: reports/summary.txt")

    if params is not None:
        _json_dump(_serialise_params(params), reports_dir / "params.json")
        logger.info("Params written: reports/params.json")

    _json_dump(result.metrics, reports_dir / "metrics.json")
    _metrics_csv(result.metrics, reports_dir / "metrics.csv")
    logger.info("Metrics written: reports/metrics.json / metrics.csv")

    # ── Global dashboard ──────────────────────────────────────────────────
    _save_fig(result, "plot", plots_dir / "dashboard.png", dpi)
    logger.info("Global dashboard written: plots/dashboard.png")

    if n_assets > 1:
        _save_fig(result, "plot_combined_pnl", plots_dir / "pnl_combined.png", dpi)
        logger.info("Combined PnL written: plots/pnl_combined.png")

    # ── Per-asset artefacts ───────────────────────────────────────────────
    for i in range(n_assets):
        sym = result.assets[i].symbol
        asset_dir = run_dir / f"asset_{i}_{sym}"

        data_dir_a = asset_dir / "data"
        logs_dir_a = asset_dir / "logs"
        plots_dir_a = asset_dir / "plots"
        reports_dir_a = asset_dir / "reports"
        for d in (data_dir_a, logs_dir_a, plots_dir_a, reports_dir_a):
            d.mkdir(parents=True, exist_ok=True)

        # ── Per-asset data CSVs ───────────────────────────────────────────
        oh = result.order_history_list[i] if i < len(result.order_history_list) else None
        if oh:
            from .logger import _order_history_to_dfs
            subs_df, cans_df = _order_history_to_dfs(oh)
            fills_df = result.fills_df(asset_no=i)
            subs_df.to_csv(data_dir_a / "orders_submitted.csv", index=False)
            cans_df.to_csv(data_dir_a / "orders_cancelled.csv", index=False)
            fills_df.to_csv(data_dir_a / "trades.csv", index=False)
            logger.info(
                "Data CSVs  asset %d (%s): %d submitted, %d cancelled, %d fills",
                i, sym, len(subs_df), len(cans_df), len(fills_df),
            )

        # ── Per-asset event log ───────────────────────────────────────────
        from .logger import generate_run_log
        generate_run_log(result, logs_dir_a / "run.log", asset_no=i, console=verbose)
        logger.info("Event log  asset %d (%s): logs/run.log", i, sym)

        # ── Per-asset reports ─────────────────────────────────────────────
        asset_metrics = _per_asset_metrics(result, i)
        (reports_dir_a / "summary.txt").write_text(
            _build_summary(result, asset_no=i), encoding="utf-8"
        )
        _json_dump(asset_metrics, reports_dir_a / "metrics.json")
        _metrics_csv(asset_metrics, reports_dir_a / "metrics.csv")
        logger.info("Reports  asset %d (%s): reports/", i, sym)

        # ── Per-asset plots ───────────────────────────────────────────────
        _save_fig(result, "plot_asset", plots_dir_a / "dashboard.png", dpi, asset_no=i)
        _save_fig(result, "plot_pnl", plots_dir_a / "pnl.png", dpi, asset_no=i)
        _save_fig(result, "plot_inventory", plots_dir_a / "inventory.png", dpi, asset_no=i)
        _save_fig(result, "plot_quotes", plots_dir_a / "quotes.png", dpi, asset_no=i)
        logger.info("Plots  asset %d (%s): plots/", i, sym)

    logger.info("Backtest saved in: %s", run_dir)
    return run_dir


# ── helpers ──────────────────────────────────────────────────────────────────

def _per_asset_metrics(result: "BacktestResult", asset_no: int) -> dict:
    """Extract metrics for a single asset from stats_list."""
    from .metrics import extract_summary
    try:
        so = result.stats_list[asset_no]
        m = extract_summary(so)
        if m:
            return m
    except (IndexError, AttributeError):
        pass
    return result.metrics


def _build_summary(result: "BacktestResult", asset_no: int | None = None) -> str:
    if asset_no is None:
        symbols = ", ".join(a.symbol.upper() for a in result.assets)
        metrics = result.metrics
        heading = "BACKTEST SUMMARY"
    else:
        symbols = result.assets[asset_no].symbol.upper()
        metrics = _per_asset_metrics(result, asset_no)
        heading = f"ASSET SUMMARY  —  {symbols}"

    lines = [
        "=" * 70,
        heading,
        "=" * 70,
        f"Symbol    : {symbols}",
        f"Period    : {result.assets[0].start_date} to {result.assets[0].end_date}",
        f"Book size : ${result.book_size:,.0f}",
        f"Runtime   : {result.elapsed_s:.1f}s",
        "",
        "METRICS",
        f"{'Metric':<30}  {'Value':>14}  Unit",
        "-" * 70,
    ]
    for k, v in metrics.items():
        val_str = f"{v:>14.6f}" if isinstance(v, float) else f"{v:>14}"
        unit_str = _METRIC_UNITS.get(k, "")
        lines.append(f"  {k:<28}  {val_str}  {unit_str}")
    lines += [
        "",
        "PNL ACCOUNTING",
        "-" * 70,
        "  equity_wo_fee  Gross PnL = balance + position x price  (fees NOT deducted)",
        "  fee            Cumulative fees paid (always positive, growing)",
        "  Net PnL        equity_wo_fee - fee",
        "  Return         Net PnL / book_size (fraction)",
        "=" * 70,
    ]
    return "\n".join(lines)


def _serialise_params(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        if isinstance(v, list):
            out[k] = [_serialise_params(x) if isinstance(x, dict) else
                      (x.tolist() if hasattr(x, "tolist") else x) for x in v]
        elif hasattr(v, "tolist"):
            out[k] = v.tolist()
        elif isinstance(v, (int, float, str, bool, type(None))):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _json_dump(d: dict, path: Path) -> None:
    safe = {}
    for k, v in d.items():
        try:
            json.dumps(v)
            safe[k] = v
        except (TypeError, ValueError):
            safe[k] = str(v)
    path.write_text(json.dumps(safe, indent=2), encoding="utf-8")


def _metrics_csv(d: dict, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "unit"])
        for k, v in d.items():
            w.writerow([k, v, _METRIC_UNITS.get(k, "")])


def _save_fig(
        result: "BacktestResult",
        method: str,
        path: Path,
        dpi: int,
        asset_no: int = 0,
) -> None:
    try:
        kwargs: dict = {}
        if method in ("plot_pnl", "plot_inventory", "plot_quotes", "plot_asset"):
            kwargs["asset_no"] = asset_no
        fig = getattr(result, method)(show=False, **kwargs)
        if fig is not None:
            fig.savefig(path, dpi=dpi, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)
    except Exception as exc:
        logger.warning("Plot '%s' (asset %d) failed: %s", method, asset_no, exc)
