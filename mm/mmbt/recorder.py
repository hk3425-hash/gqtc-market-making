"""
MMRecorder — lightweight, numba-compatible data recorder.

Usage inside an @njit strategy function
-----------------------------------------
The runner pre-allocates six numpy arrays and a step-counter and passes them
to your strategy.  Call ``mm_record(...)`` at the *end* of each main loop
iteration (once per interval, after you have computed bid/ask prices for
asset 0 — or whichever asset you want to visualise).

Example (add to your @njit function signature):

    mm_ts, mm_mid, mm_bid, mm_ask, mm_pos, mm_step

Then, once per outer loop iteration (after the last asset loop):

    mmbt.recorder.record_step(
        mm_ts, mm_mid, mm_bid, mm_ask, mm_pos, mm_step,
        hbt.current_timestamp,
        mid_price_asset0,
        bid_quote_asset0,
        ask_quote_asset0,
        position_asset0,
    )

Or, writing the arrays directly (if you prefer explicit numba):

    i = mm_step[0]
    mm_ts[i]     = hbt.current_timestamp
    mm_mid[i]    = mid_price
    mm_bid[i]    = bid_price   # most-aggressive bid quote
    mm_ask[i]    = ask_price   # most-aggressive ask quote
    mm_pos[i]    = position
    mm_step[0]  += 1
"""

from __future__ import annotations

import numpy as np
from numba import njit


# ── numba helper (called from inside @njit strategy) ────────────────────────

@njit
def record_step(
        mm_ts: np.ndarray,  # int64[max_steps]
        mm_mid: np.ndarray,  # float64[max_steps]
        mm_bid: np.ndarray,  # float64[max_steps]
        mm_ask: np.ndarray,  # float64[max_steps]
        mm_pos: np.ndarray,  # float64[max_steps]
        mm_step: np.ndarray,  # int64[1]  — mutable step counter
        timestamp: np.int64,
        mid_price: np.float64,
        bid_price: np.float64,
        ask_price: np.float64,
        position: np.float64,
) -> None:
    """Write one row into the pre-allocated recording arrays."""
    i = mm_step[0]
    if i < mm_ts.shape[0]:
        mm_ts[i] = timestamp
        mm_mid[i] = mid_price
        mm_bid[i] = bid_price
        mm_ask[i] = ask_price
        mm_pos[i] = position
        mm_step[0] = i + 1


# ── Python-side container ────────────────────────────────────────────────────

class MMRecorder:
    """
    Allocates the per-step recording arrays and exposes them for
    injection into the @njit strategy and post-run data retrieval.

    Parameters
    ----------
    max_steps : int
        Upper bound on the number of strategy iterations.
        A safe value: ``(end_ts - start_ts) // interval + 1``
        or simply ``days * 86_400 * 1_000_000_000 // interval + 1``.
    """

    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps

        # Pre-allocated arrays written to by @njit strategy
        self.ts = np.zeros(max_steps, dtype=np.int64)
        self.mid = np.zeros(max_steps, dtype=np.float64)
        self.bid = np.full(max_steps, np.nan, dtype=np.float64)
        self.ask = np.full(max_steps, np.nan, dtype=np.float64)
        self.pos = np.zeros(max_steps, dtype=np.float64)
        self.step = np.zeros(1, dtype=np.int64)  # [0] = current idx

    # ── convenience: return tuple for direct **kwargs injection ──────────

    @property
    def arrays(self) -> dict:
        """
        Return a dict ready for ``**``-unpacking into the strategy call.

        Expected parameter names in your @njit function:
            mm_ts, mm_mid, mm_bid, mm_ask, mm_pos, mm_step
        """
        return dict(
            mm_ts=self.ts,
            mm_mid=self.mid,
            mm_bid=self.bid,
            mm_ask=self.ask,
            mm_pos=self.pos,
            mm_step=self.step,
        )

    # ── post-run data access ─────────────────────────────────────────────

    def get(self) -> dict[str, np.ndarray]:
        """Return a dict of trimmed arrays (recorded steps only)."""
        n = int(self.step[0])
        return {
            "timestamps": self.ts[:n],
            "mid_prices": self.mid[:n],
            "bid_quotes": self.bid[:n],
            "ask_quotes": self.ask[:n],
            "positions": self.pos[:n],
        }

    def __len__(self) -> int:
        return int(self.step[0])
