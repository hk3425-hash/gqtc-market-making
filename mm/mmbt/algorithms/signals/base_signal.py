"""
algorithms/signals/base_signal.py — AlphaSignal abstract base class.

An signals signal adjusts the reservation/fair price by returning a USD offset
at each strategy step:
  - positive  → bullish (shift quotes up, buy more aggressively)
  - negative  → bearish (shift quotes down, sell more aggressively)
  - 0         → no directional view (default)

Usage
-----
    class MySignal(AlphaSignal):
        def compute(self, depth, mid: float, position: float) -> float:
            return 0.0  # your logic here

        @property
        def params_dict(self) -> dict:
            return {"signal": "MySignal"}
"""

from __future__ import annotations
from abc import ABC, abstractmethod


class AlphaSignal(ABC):
    """Abstract base class for all signals signals."""

    @abstractmethod
    def compute(self, depth, mid: float, position: float) -> float:
        """
        Compute the signals price adjustment in USD.

        Parameters
        ----------
        depth    : hftbacktest depth object (bid_depth, ask_depth, tick_size, …)
        mid      : current mid price in USD
        position : current inventory in base currency

        Returns
        -------
        float : adjustment in USD (positive = bullish, negative = bearish)
        """

    @property
    def params_dict(self) -> dict:
        """Return serialisable signal parameters. Override in subclasses."""
        return {"signal": type(self).__name__}

    def reset(self) -> None:
        """Reset internal state. Called automatically before each backtest run."""
