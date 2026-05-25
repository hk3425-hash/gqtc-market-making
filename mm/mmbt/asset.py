"""
Asset configuration for the backtest runner.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from hftbacktest import BacktestAsset

from .data.utils import get_hourly_files, get_snapshot


@dataclass
class AssetConfig:
    """
    Full configuration for a single asset in the backtest.

    tick_size, lot_size, and roi_ub are optional: if not provided they are
    loaded automatically from {data_dir}/{symbol}/info.json (written by the
    data collector).

    Args:
        symbol:         Trading pair identifier (e.g. "xrpusd").
        start_date:     Backtest start date, YYYYMMDD or YYYYMMDD_HH.
        end_date:       Backtest end date (inclusive), YYYYMMDD or YYYYMMDD_HH.
        tick_size:      Minimum price increment. Auto-loaded from info.json if None.
        lot_size:       Minimum order quantity increment. Auto-loaded if None.
        maker_fee:      Maker fee rate (e.g. 0.0004 = 4 bps).
        taker_fee:      Taker fee rate (e.g. 0.0008 = 8 bps).
        latency_ns:     Round-trip order latency in nanoseconds (default 100ms).
        roi_lb:         Region-of-interest lower price bound (default 0.0).
        roi_ub:         Region-of-interest upper price bound. Auto-loaded if None.
        last_trades_capacity: Ring-buffer size for last-trades data (default 1000).
        data_dir:       Root data directory (default Path("data")).
    """

    symbol: str
    start_date: str
    end_date: str

    tick_size: Optional[float] = None
    lot_size: Optional[float] = None

    maker_fee: float = 0.0004
    taker_fee: float = 0.0008
    latency_ns: int = 100_000_000

    roi_lb: float = 0.0
    roi_ub: Optional[float] = None
    last_trades_capacity: int = 1_000

    data_dir: Path = field(default_factory=lambda: Path("data"))

    # ── resolved at build time ───────────────────────────────────────────
    _data_files: list = field(default_factory=list, init=False, repr=False)
    _snapshot: Optional[str] = field(default=None, init=False, repr=False)

    def _load_info(self) -> dict:
        """Load info.json for this symbol, return empty dict if missing."""
        info_path = self.data_dir / self.symbol / "info.json"
        if not info_path.exists():
            return {}
        try:
            return json.loads(info_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def build(self) -> BacktestAsset:
        """Resolve data files/snapshot and return a configured BacktestAsset."""
        # Fill missing instrument params from info.json
        info = self._load_info()

        tick_size = self.tick_size or info.get("tick_size")
        lot_size  = self.lot_size  or info.get("lot_size")
        roi_ub    = self.roi_ub    or info.get("roi_ub") or 1_200.0

        if tick_size is None:
            raise ValueError(
                f"tick_size for {self.symbol!r} is not set and not found in info.json. "
                "Run the data collector first, or pass tick_size explicitly."
            )
        if lot_size is None:
            raise ValueError(
                f"lot_size for {self.symbol!r} is not set and not found in info.json. "
                "Run the data collector first, or pass lot_size explicitly."
            )

        self._data_files = get_hourly_files(
            self.symbol, self.start_date, self.end_date, self.data_dir
        )
        self._snapshot = get_snapshot(self.symbol, self.start_date, self.data_dir)

        if not self._data_files:
            raise FileNotFoundError(
                f"No data files found for {self.symbol} "
                f"between {self.start_date} and {self.end_date} in {self.data_dir}"
            )

        asset = (
            BacktestAsset()
            .data(self._data_files)
            .linear_asset(1.0)
            .constant_order_latency(self.latency_ns, self.latency_ns)
            .power_prob_queue_model(2)
            .trading_value_fee_model(self.maker_fee, self.taker_fee)
            .tick_size(tick_size)
            .lot_size(lot_size)
            .roi_lb(self.roi_lb)
            .roi_ub(roi_ub)
            .last_trades_capacity(self.last_trades_capacity)
        )

        if self._snapshot:
            asset = asset.initial_snapshot(self._snapshot)

        return asset