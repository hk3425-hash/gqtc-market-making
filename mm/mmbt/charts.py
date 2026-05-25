"""
Charts module — market-making backtest visualisations.

PnL source of truth
-------------------
hftbacktest computes `equity_wo_fee` in stats.entire.
As the name implies, this is the equity WITHOUT fees deducted:

    equity_wo_fee = Gross PnL (before fees)
    fee           = cumulative fee cost paid
    Net PnL       = equity_wo_fee - fee      <- THE correct PnL curve

We plot:
  · Net PnL        : True bottom-line equity after fees
  · Gross PnL      : equity_wo_fee (useful to see the fee drag distance)
  · Incremental fee: delta fee per period (small per-step cost, stays readable)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone
from typing import Optional, List

_C = dict(
    bg="#0e1117",
    surface="#1a1d27",
    border="#2c2f3f",
    text="#e2e8f0",
    muted="#64748b",
    green="#22c55e",
    red="#ef4444",
    blue="#3b82f6",
    purple="#a855f7",
    orange="#f97316",
    yellow="#eab308",
    teal="#14b8a6",
    pink="#ec4899",
)
_ASSET_COLORS = [_C["blue"], _C["orange"], _C["teal"], _C["pink"], _C["yellow"]]


def _apply_dark(ax, title: str = "") -> None:
    ax.set_facecolor(_C["surface"])
    ax.tick_params(colors=_C["muted"], labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(_C["border"])
    ax.xaxis.label.set_color(_C["muted"])
    ax.yaxis.label.set_color(_C["muted"])
    if title:
        ax.set_title(title, color=_C["text"], fontsize=11, fontweight="bold", pad=10)
    ax.grid(True, color=_C["border"], linewidth=0.5, alpha=0.7)


def _ns_to_mpl(ts_ns: np.ndarray) -> list:
    return mdates.date2num(
        [datetime.fromtimestamp(t / 1e9, tz=timezone.utc) for t in ts_ns]
    )


def _fmt_xaxis(ax) -> None:
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    for lbl in ax.get_xticklabels():
        lbl.set_color(_C["muted"])


def _show(fig: plt.Figure, show: bool) -> Optional[plt.Figure]:
    """Show figure safely across interactive and non-interactive backends."""
    if show:
        try:
            import matplotlib
            backend = matplotlib.get_backend().lower()
            if "inline" in backend or "nbagg" in backend or "widget" in backend:
                plt.show()
            else:
                try:
                    from IPython.display import display
                    display(fig)
                except ImportError:
                    pass
        except Exception:
            pass
        plt.close(fig)
        return None
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PnL extraction — always from stats.entire (Polars or Pandas)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_pnl_series(stats_obj, asset_no: int = 0):
    """
    Extract (timestamps_ns, net_pnl, gross_pnl, delta_fee) from stats.entire.
    """
    try:
        entire = stats_obj.entire
        try:
            ts_ns = entire["timestamp"].to_numpy(zero_copy_only=False).astype(np.int64)
            eqwf = entire["equity_wo_fee"].to_numpy(zero_copy_only=False).astype(np.float64)
            fee = entire["fee"].to_numpy(zero_copy_only=False).astype(np.float64)
        except AttributeError:
            ts_ns = entire["timestamp"].values.astype(np.int64)
            eqwf = entire["equity_wo_fee"].values.astype(np.float64)
            fee = entire["fee"].values.astype(np.float64)

        net_pnl = eqwf - fee
        delta_fee = np.diff(fee, prepend=fee[0])
        return ts_ns, net_pnl, eqwf, delta_fee

    except Exception:
        return None, None, None, None


def _extract_pnl_from_rec(rec: np.ndarray):
    """Fallback: extract PnL directly from recorder structured array."""
    ts_ns = rec["timestamp"].astype(np.int64)
    if "equity_wo_fee" in rec.dtype.names:
        eqwf = rec["equity_wo_fee"].astype(np.float64)
    else:
        eqwf = (rec["balance"].astype(np.float64)
                + rec["position"].astype(np.float64) * rec["price"].astype(np.float64))

    fee = rec["fee"].astype(np.float64)
    net_pnl = eqwf - fee
    delta_fee = np.diff(fee, prepend=fee[0])
    return ts_ns, net_pnl, eqwf, delta_fee


# ─────────────────────────────────────────────────────────────────────────────
# Individual plot functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_pnl_single(ax, ts_ns, net_pnl, gross_pnl, delta_fee, label=""):
    """Plot Net PnL (after fees), Gross PnL, and cumulative incremental fee drag."""
    _apply_dark(ax, f"PnL{'  —  ' + label if label else ''}")
    dt = _ns_to_mpl(ts_ns)

    ax.plot(dt, gross_pnl, color=_C["muted"], lw=1.2, linestyle="--",
            label="Gross PnL (equity_wo_fee)", zorder=2)
    ax.plot(dt, net_pnl, color=_C["green"], lw=1.8,
            label="Net PnL (after fees)", zorder=3)

    cum_fee = np.cumsum(delta_fee)
    if np.nanmax(np.abs(cum_fee)) < 10 * max(np.nanmax(np.abs(net_pnl)), 1e-9):
        ax.plot(dt, -cum_fee, color=_C["red"], lw=1.0,
                linestyle=":", alpha=0.8, label="Cumulative fees paid", zorder=2)
        ax.fill_between(dt, net_pnl, gross_pnl,
                        alpha=0.08, color=_C["red"], zorder=1)

    ax.axhline(0, color=_C["border"], lw=0.8, zorder=1)
    ax.set_ylabel("PnL (USD)")
    ax.legend(fontsize=8, facecolor=_C["surface"], labelcolor=_C["text"], framealpha=0.9)
    _fmt_xaxis(ax)


def plot_pnl_combined(ax, stats_list, records, symbols):
    """Overlay all assets' net equity plus a total line."""
    _apply_dark(ax, "Net PnL — All Assets")

    combined_ts = None
    combined_net = None

    for i, (stats_obj, rec, sym) in enumerate(zip(stats_list, records, symbols)):
        ts_ns, net_pnl, _, _ = _extract_pnl_series(stats_obj)
        if ts_ns is None:
            ts_ns, net_pnl, _, _ = _extract_pnl_from_rec(rec)

        color = _ASSET_COLORS[i % len(_ASSET_COLORS)]
        dt = _ns_to_mpl(ts_ns)
        ax.plot(dt, net_pnl, color=color, lw=1.2, alpha=0.85, label=sym.upper(), zorder=3)

        if combined_ts is None:
            combined_ts = ts_ns
            combined_net = net_pnl.copy()
        else:
            combined_net = combined_net + np.interp(
                combined_ts.astype(float), ts_ns.astype(float), net_pnl
            )

    if combined_ts is not None and len(records) > 1:
        ax.plot(_ns_to_mpl(combined_ts), combined_net, color=_C["text"],
                lw=2.0, linestyle="--", label="Total Net PnL", zorder=4)

    ax.axhline(0, color=_C["border"], lw=0.8)
    ax.set_ylabel("PnL (USD)")
    ax.legend(fontsize=8, facecolor=_C["surface"], labelcolor=_C["text"], framealpha=0.9)
    _fmt_xaxis(ax)


def plot_inventory_single(ax, rec_ts, positions, label="", color=None):
    """Plot inventory (position) over time."""
    _apply_dark(ax, f"Inventory{'  —  ' + label if label else ''}")
    c = color or _C["blue"]
    dt = _ns_to_mpl(rec_ts)
    ax.plot(dt, positions, color=c, lw=1.2, alpha=0.9, label=label or "Position", zorder=3)
    ax.fill_between(dt, positions, 0, alpha=0.12, color=c, zorder=2)
    ax.axhline(0, color=_C["border"], lw=0.8, zorder=1)
    ax.set_ylabel("Position (base)")
    if label:
        ax.legend(fontsize=8, facecolor=_C["surface"], labelcolor=_C["text"], framealpha=0.9)
    _fmt_xaxis(ax)


def plot_quotes_single(ax, ts, mid, bid, ask, positions, label="", downsample=4):
    """Plot mid price, bid/ask quotes, and fill markers."""
    _apply_dark(ax, f"Quotes{'  —  ' + label if label else ''}")
    dt = np.array(_ns_to_mpl(ts))
    mid = np.asarray(mid, dtype=float)
    bid = np.asarray(bid, dtype=float)
    ask = np.asarray(ask, dtype=float)

    ax.plot(dt, mid, color=_C["muted"], lw=0.8, alpha=0.6, zorder=2, label="Mid")

    bid_change = np.concatenate([[True], np.diff(np.nan_to_num(bid)) != 0])
    ask_change = np.concatenate([[True], np.diff(np.nan_to_num(ask)) != 0])

    bid_mask = bid_change & np.isfinite(bid)
    ask_mask = ask_change & np.isfinite(ask)

    if downsample > 1:
        idx = np.arange(len(dt))
        bid_mask = bid_mask & ((idx % downsample == 0) | bid_change)
        ask_mask = ask_mask & ((idx % downsample == 0) | ask_change)

    if bid_mask.any():
        ax.scatter(dt[bid_mask], bid[bid_mask],
                   color=_C["green"], s=8, alpha=0.3, zorder=3,
                   marker="o", label="Bid quote")
    if ask_mask.any():
        ax.scatter(dt[ask_mask], ask[ask_mask],
                   color=_C["red"], s=8, alpha=0.3, zorder=3,
                   marker="o", label="Ask quote")

    if len(positions) > 1:
        delta = np.diff(positions, prepend=positions[0])
        buy_mask = delta > 0
        sell_mask = delta < 0
        if buy_mask.any():
            ax.scatter(dt[buy_mask], mid[buy_mask],
                       color=_C["green"], s=100, zorder=5, marker="x",
                       linewidths=2.2, label="Buy fill")
        if sell_mask.any():
            ax.scatter(dt[sell_mask], mid[sell_mask],
                       color=_C["red"], s=100, zorder=5, marker="x",
                       linewidths=2.2, label="Sell fill")

    ax.set_ylabel("Price (USD)")
    ax.legend(fontsize=8, facecolor=_C["surface"], labelcolor=_C["text"],
              framealpha=0.9, markerscale=1.6)
    _fmt_xaxis(ax)


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard builders — no metrics panel
# ─────────────────────────────────────────────────────────────────────────────

def build_single_asset_dashboard(stats_obj=None, rec=None, mm_data=None,
                                 symbol="", start_date="", end_date="",
                                 downsample=4, figsize=(18, 14),
                                 time_range=None):
    ts_ns, net_pnl, gross_pnl, dfee = _extract_pnl_series(stats_obj)
    if ts_ns is None:
        ts_ns, net_pnl, gross_pnl, dfee = _extract_pnl_from_rec(rec)

    pmask = _time_mask(ts_ns, time_range)
    rmask = _time_mask(rec["timestamp"].astype(np.int64), time_range)
    rec_z = rec[rmask]
    mm_z = _zoom_mm(mm_data, time_range)
    zoom_s = _zoom_label(time_range)

    fig, axes = plt.subplots(
        3, 1, figsize=figsize, facecolor=_C["bg"], sharex=False,
        gridspec_kw=dict(hspace=0.42, left=0.07, right=0.97, top=0.93, bottom=0.05),
    )
    fig.patch.set_facecolor(_C["bg"])
    fig.suptitle(
        f"Market Making Backtest  ·  {symbol.upper()}  ·  {start_date} to {end_date}{zoom_s}",
        color=_C["text"], fontsize=13, fontweight="bold", y=0.98,
    )

    plot_pnl_single(axes[0], ts_ns[pmask], net_pnl[pmask], gross_pnl[pmask], dfee[pmask])
    plot_inventory_single(axes[1], rec_z["timestamp"].astype(np.int64), rec_z["position"])

    if mm_z is not None:
        plot_quotes_single(axes[2], mm_z["timestamps"], mm_z["mid_prices"],
                           mm_z["bid_quotes"], mm_z["ask_quotes"],
                           mm_z["positions"], downsample=downsample)
    else:
        _apply_dark(axes[2], "Quotes")
        axes[2].text(0.5, 0.5, "No mm_data available for this asset.",
                     transform=axes[2].transAxes, ha="center", va="center",
                     color=_C["muted"], fontsize=10)

    return fig


def build_multi_asset_dashboard(stats_list, records, mm_data_list, symbols,
                                start_date, end_date,
                                downsample=4, figsize=None, time_range=None):
    n = len(records)
    zoom_s = _zoom_label(time_range)
    if figsize is None:
        figsize = (20, 5 + 5 * n)

    # Row 0: combined PnL spanning full width
    # Rows 1..n: PnL | Inventory | Quotes  (3 cols per asset)
    n_rows = 1 + n
    n_cols = 3

    fig = plt.figure(figsize=figsize, facecolor=_C["bg"])
    gs = fig.add_gridspec(
        n_rows, n_cols,
        hspace=0.45, wspace=0.28,
        left=0.06, right=0.97, top=0.93, bottom=0.04,
    )
    fig.suptitle(
        f"Backtest  ·  {' + '.join(s.upper() for s in symbols)}  ·  {start_date} to {end_date}{zoom_s}",
        color=_C["text"], fontsize=13, fontweight="bold", y=0.98,
    )

    # Row 0: combined PnL
    ax_comb = fig.add_subplot(gs[0, :])
    plot_pnl_combined(ax_comb, stats_list, records, symbols)

    # Rows 1..n: per-asset PnL + inventory + quotes
    for i, (stats_obj, rec, mm, sym) in enumerate(zip(stats_list, records, mm_data_list, symbols)):
        color = _ASSET_COLORS[i % len(_ASSET_COLORS)]
        row = i + 1

        # PnL
        ts_ns, net_pnl, gross_pnl, dfee = _extract_pnl_series(stats_obj)
        if ts_ns is None:
            ts_ns, net_pnl, gross_pnl, dfee = _extract_pnl_from_rec(rec)
        pmask = _time_mask(ts_ns, time_range)
        ax_pnl = fig.add_subplot(gs[row, 0])
        plot_pnl_single(ax_pnl, ts_ns[pmask], net_pnl[pmask],
                        gross_pnl[pmask], dfee[pmask], label=sym.upper())

        # Inventory
        rmask = _time_mask(rec["timestamp"].astype(np.int64), time_range)
        rec_z = rec[rmask]
        ax_inv = fig.add_subplot(gs[row, 1])
        plot_inventory_single(ax_inv, rec_z["timestamp"].astype(np.int64),
                              rec_z["position"], label=sym.upper(), color=color)

        # Quotes
        mm_z = _zoom_mm(mm, time_range)
        ax_q = fig.add_subplot(gs[row, 2])
        if mm_z is not None:
            plot_quotes_single(ax_q, mm_z["timestamps"], mm_z["mid_prices"],
                               mm_z["bid_quotes"], mm_z["ask_quotes"],
                               mm_z["positions"], label=sym.upper(), downsample=downsample)
        else:
            _apply_dark(ax_q, f"Quotes — {sym.upper()}")
            ax_q.text(0.5, 0.5, "No mm_data", transform=ax_q.transAxes,
                      ha="center", va="center", color=_C["muted"], fontsize=9)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _time_mask(ts_ns: np.ndarray, tr) -> np.ndarray:
    if tr is None:
        return np.ones(len(ts_ns), dtype=bool)
    return (ts_ns >= tr[0]) & (ts_ns <= tr[1])


def _zoom_mm(mm, tr) -> Optional[dict]:
    if mm is None:
        return None
    if tr is None:
        return mm
    mask = _time_mask(mm["timestamps"], tr)
    return {k: v[mask] for k, v in mm.items()} if mask.any() else None


def _zoom_label(tr) -> str:
    if tr is None:
        return ""
    t0 = datetime.fromtimestamp(tr[0] / 1e9, tz=timezone.utc)
    t1 = datetime.fromtimestamp(tr[1] / 1e9, tz=timezone.utc)
    return f"  ·  {t0.strftime('%H:%M')} to {t1.strftime('%H:%M')} UTC"
