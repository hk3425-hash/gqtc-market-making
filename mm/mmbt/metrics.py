"""
Metrics utilities.
"""

from __future__ import annotations
from typing import Any

from hftbacktest.stats.metrics import (
    SR, Sortino, Ret, MaxDrawdown, DailyTradingValue,
    ReturnOverMDD, ReturnOverTrade, MaxPositionValue,
    DailyNumberOfTrades, DailyTradingVolume,
)

DEFAULT_METRICS: tuple = (
    SR, Sortino, Ret, MaxDrawdown, DailyNumberOfTrades,
    DailyTradingValue, ReturnOverMDD, ReturnOverTrade,
    MaxPositionValue, DailyTradingVolume,
)

_KNOWN_ATTRS = [
    "sr", "sharpe", "sharpe_ratio", "sortino", "sortino_ratio",
    "ret", "return_", "total_return", "max_drawdown", "mdd",
    "daily_number_of_trades", "num_trades", "daily_trading_value",
    "trading_value", "return_over_mdd", "romdd", "return_over_trade",
    "rot", "max_position_value", "daily_trading_volume", "trading_volume",
]


class MetricsRegistry:
    """Deduplicating, ordered collection of metric classes."""

    def __init__(self, base: tuple = DEFAULT_METRICS, extra: tuple = ()) -> None:
        seen, merged = set(), []
        for m in (*base, *extra):
            if m not in seen:
                seen.add(m);
                merged.append(m)
        self._metrics: tuple = tuple(merged)

    @property
    def metrics(self) -> tuple:
        return self._metrics

    def add(self, *metric_classes) -> "MetricsRegistry":
        return MetricsRegistry(base=self._metrics, extra=metric_classes)

    def remove(self, *metric_classes) -> "MetricsRegistry":
        return MetricsRegistry(base=tuple(m for m in self._metrics if m not in set(metric_classes)))

    def __repr__(self) -> str:
        return f"MetricsRegistry([{', '.join(getattr(m, '__name__', str(m)) for m in self._metrics)}])"


def extract_summary(stats_obj) -> dict[str, Any]:
    """
    Pull computed metric values out of an hftbacktest Stats object.

    hftbacktest's Stats object has:
        .splits  — list of per-period dicts: [{'SR': ..., 'Return': ..., ...}]
        .entire  — Polars DataFrame (raw timeseries, NOT the metric values)
        .kwargs  — {'book_size': ...}

    We extract from splits[0] and keep only numeric scalar values.
    """
    import datetime

    def _is_scalar_numeric(v) -> bool:
        """True for plain numbers and numpy scalars, False for DataFrames/dicts/dates."""
        if isinstance(v, (int, float)):
            return True
        # numpy scalar
        try:
            import numpy as np
            if isinstance(v, np.generic) and v.ndim == 0:
                return True
        except ImportError:
            pass
        return False

    def _extract_from_dict(d: dict) -> dict:
        """Keep only scalar numeric entries, skip start/end/datetime keys."""
        out = {}
        for k, v in d.items():
            k_str = str(k)
            if k_str in ("start", "end"):
                continue
            if isinstance(v, datetime.datetime):
                continue
            if _is_scalar_numeric(v):
                out[k_str] = float(v)
        return out

    # ── 1. hftbacktest primary layout: .splits list of period dicts ──────
    try:
        splits = stats_obj.splits
        if splits and isinstance(splits, (list, tuple)) and len(splits) > 0:
            first = splits[0]
            if isinstance(first, dict):
                result = _extract_from_dict(first)
                if result:
                    return result
    except (AttributeError, TypeError, IndexError):
        pass

    # ── 2. .metrics iterable (older hftbacktest versions) ────────────────
    try:
        result: dict[str, Any] = {}
        for item in stats_obj.metrics:
            if hasattr(item, "name") and hasattr(item, "value"):
                if _is_scalar_numeric(item.value):
                    result[str(item.name)] = float(item.value)
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                if _is_scalar_numeric(item[1]):
                    result[str(item[0])] = float(item[1])
        if result:
            return result
    except (AttributeError, TypeError):
        pass

    # ── 3. dataclass fields ───────────────────────────────────────────────
    try:
        result = {}
        for f in stats_obj.__dataclass_fields__:
            v = getattr(stats_obj, f)
            if _is_scalar_numeric(v):
                result[f] = float(v)
        if result:
            return result
    except AttributeError:
        pass

    # ── 4. probe known attribute names ────────────────────────────────────
    result = {}
    for attr in _KNOWN_ATTRS:
        try:
            v = getattr(stats_obj, attr)
            if _is_scalar_numeric(v):
                result[attr] = float(v)
        except AttributeError:
            pass
    if result:
        return result

    return {}
