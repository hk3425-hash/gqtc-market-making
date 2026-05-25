"""
OrderLogger — tracks order submissions, cancellations, and fills from @njit.

PnL / cash flow notes
---------------------
In hftbacktest the ``balance`` recorder field tracks *realized cash flow*:
  - BUY  fill:  balance -= fill_price * fill_qty  (cash leaves)
  - SELL fill:  balance += fill_price * fill_qty  (cash arrives)
  Fees are already deducted inside ``balance`` AND tracked separately in ``fee``.

So over the recorder's sampling interval (e.g. 100ms) the balance change at
a fill event is approximately ±order_qty_usd — that is correct and expected.
It is NOT "realized PnL" (which only crystallizes on a round-trip); it is the
gross cash-flow impact of that fill.  We label it accordingly.

True mark-to-market PnL at time t:
    pnl_net(t)   = balance(t) + position(t) * mid(t)
    pnl_gross(t) = balance(t) + fee(t) + position(t) * mid(t)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from numba import njit


# ── numba helpers (called from inside @njit strategy) ────────────────────────

@njit
def log_submit(
        ol_sub_ts: np.ndarray,  # int64[cap]
        ol_sub_side: np.ndarray,  # int8[cap]    1=BUY  -1=SELL
        ol_sub_price: np.ndarray,  # float64[cap]
        ol_sub_qty: np.ndarray,  # float64[cap]
        ol_sub_id: np.ndarray,  # uint64[cap]
        ol_sub_n: np.ndarray,  # int64[1]
        timestamp: np.int64,
        side: np.int8,
        price: np.float64,
        qty: np.float64,
        order_id,  # uint64
) -> None:
    i = ol_sub_n[0]
    if i < ol_sub_ts.shape[0]:
        ol_sub_ts[i] = timestamp
        ol_sub_side[i] = side
        ol_sub_price[i] = price
        ol_sub_qty[i] = qty
        ol_sub_id[i] = order_id
        ol_sub_n[0] = i + 1


@njit
def log_cancel(
        ol_can_ts: np.ndarray,  # int64[cap]
        ol_can_id: np.ndarray,  # uint64[cap]
        ol_can_n: np.ndarray,  # int64[1]
        timestamp: np.int64,
        order_id,  # uint64
) -> None:
    i = ol_can_n[0]
    if i < ol_can_ts.shape[0]:
        ol_can_ts[i] = timestamp
        ol_can_id[i] = order_id
        ol_can_n[0] = i + 1


# ── Python-side container ─────────────────────────────────────────────────────

class OrderLogger:
    """
    Pre-allocates order-event arrays and reconstructs DataFrames after the run.

    Parameters
    ----------
    max_orders : int
        Capacity for submitted / cancelled events.
        A safe estimate: ``max_steps * grid_levels * 2 * 2``
    """

    def __init__(self, max_orders: int) -> None:
        self.max_orders = max_orders

        self._sub_ts = np.zeros(max_orders, dtype=np.int64)
        self._sub_side = np.zeros(max_orders, dtype=np.int8)
        self._sub_price = np.zeros(max_orders, dtype=np.float64)
        self._sub_qty = np.zeros(max_orders, dtype=np.float64)
        self._sub_id = np.zeros(max_orders, dtype=np.uint64)
        self._sub_n = np.zeros(1, dtype=np.int64)

        self._can_ts = np.zeros(max_orders, dtype=np.int64)
        self._can_id = np.zeros(max_orders, dtype=np.uint64)
        self._can_n = np.zeros(1, dtype=np.int64)

    @property
    def arrays(self) -> dict:
        """``**``-unpack into your @njit strategy call."""
        return dict(
            ol_sub_ts=self._sub_ts,
            ol_sub_side=self._sub_side,
            ol_sub_price=self._sub_price,
            ol_sub_qty=self._sub_qty,
            ol_sub_id=self._sub_id,
            ol_sub_n=self._sub_n,
            ol_can_ts=self._can_ts,
            ol_can_id=self._can_id,
            ol_can_n=self._can_n,
        )

    # ── DataFrames ────────────────────────────────────────────────────────

    def submissions_df(self) -> pd.DataFrame:
        """All submitted orders."""
        n = int(self._sub_n[0])
        if n == 0:
            return pd.DataFrame(columns=["datetime", "timestamp_ns",
                                         "side", "price", "qty", "order_id"])
        ts = self._sub_ts[:n]
        return pd.DataFrame({
            "datetime": [datetime.fromtimestamp(t / 1e9, tz=timezone.utc) for t in ts],
            "timestamp_ns": ts,
            "side": ["BUY" if s == 1 else "SELL" for s in self._sub_side[:n]],
            "price": self._sub_price[:n],
            "qty": self._sub_qty[:n],
            "order_id": self._sub_id[:n],
        })

    def cancellations_df(self) -> pd.DataFrame:
        """All cancellations, enriched with submission price/side."""
        n = int(self._can_n[0])
        if n == 0:
            return pd.DataFrame(columns=["datetime", "timestamp_ns",
                                         "order_id", "submit_price", "side"])
        can_ts = self._can_ts[:n]
        can_id = self._can_id[:n]

        # Build lookup from submission data
        sub_n = int(self._sub_n[0])
        lookup = {
            oid: (price, side)
            for oid, price, side in zip(
                self._sub_id[:sub_n],
                self._sub_price[:sub_n],
                self._sub_side[:sub_n],
            )
        }
        prices = [lookup.get(oid, (float("nan"), 0))[0] for oid in can_id]
        sides = ["BUY" if lookup.get(oid, (0, 0))[1] == 1 else "SELL" for oid in can_id]

        return pd.DataFrame({
            "datetime": [datetime.fromtimestamp(t / 1e9, tz=timezone.utc) for t in can_ts],
            "timestamp_ns": can_ts,
            "order_id": can_id,
            "submit_price": prices,
            "side": sides,
        })

    def fills_df(
            self,
            recorder_data: np.ndarray,
            mm_data: "dict | None" = None,
    ) -> pd.DataFrame:
        """
        Infer fill events from the recorder's position and num_trades fields.

        Columns
        -------
        datetime, timestamp_ns
            When the fill was recorded.
        side
            "BUY" (position increased) or "SELL" (position decreased).
        qty
            Filled quantity in base currency.
        fill_price
            Estimated fill price (from MMRecorder bid/ask quotes if available,
            else from balance-change ÷ qty).
        gross_cash_flow
            Cash-flow impact BEFORE fees: roughly ±fill_price × qty.
            Positive = cash in (SELL fill), Negative = cash out (BUY fill).
        net_cash_flow
            Cash-flow impact AFTER fees (as recorded by hftbacktest balance field).
            This is NOT realized PnL — round-trip PnL = sell net_cf + buy net_cf.
        fee_at_fill
            Fee charged for this fill (net_cash_flow - gross_cash_flow, typically
            negative — i.e. a cost).

        Note on ±99 values
        ------------------
        If order_qty_usd ≈ $100 you will see gross_cash_flow ≈ ±100 per fill.
        That is correct — it IS the cash flow for that trade.  Realized PnL only
        crystallises after a round-trip buy + sell of the same inventory.
        """
        ts_rec = recorder_data["timestamp"].astype(np.int64)
        pos = recorder_data["position"].astype(np.float64)
        bal = recorder_data["balance"].astype(np.float64)
        fee_cum = recorder_data["fee"].astype(np.float64)
        n_trades = recorder_data["num_trades"].astype(np.int64)

        delta_pos = np.diff(pos, prepend=pos[0])
        delta_trades = np.diff(n_trades, prepend=n_trades[0])
        delta_bal = np.diff(bal, prepend=bal[0])
        delta_fee = np.diff(fee_cum, prepend=fee_cum[0])
        fill_mask = delta_trades > 0

        if not fill_mask.any():
            return pd.DataFrame(columns=[
                "datetime", "timestamp_ns", "side", "qty",
                "fill_price", "gross_cash_flow", "net_cash_flow", "fee_at_fill",
            ])

        idx = np.where(fill_mask)[0]
        ts_fills = ts_rec[idx]
        dts = [datetime.fromtimestamp(t / 1e9, tz=timezone.utc) for t in ts_fills]
        qty = np.abs(delta_pos[idx])
        sides = ["BUY" if d > 0 else "SELL" for d in delta_pos[idx]]

        # ── net cash flow from hftbacktest balance ────────────────────────
        net_cf = delta_bal[idx]  # balance change at fill (fees already out)
        fee_cf = delta_fee[idx]  # fees at this fill (negative = cost)
        # gross cash flow = net + fees (add back cost to get pre-fee figure)
        gross_cf = net_cf - fee_cf  # fee_cf is negative, so subtracting makes gross larger

        # ── fill price estimation ─────────────────────────────────────────
        if mm_data is not None:
            mm_ts_f = mm_data["timestamps"].astype(np.float64)
            rec_ts_f = ts_fills.astype(np.float64)
            bid_q = mm_data.get("bid_quotes")
            ask_q = mm_data.get("ask_quotes")

            prices = []
            for t, side, g_cf, q in zip(rec_ts_f, sides, gross_cf, qty):
                idx_mm = int(np.searchsorted(mm_ts_f, t))
                idx_mm = min(idx_mm, len(mm_ts_f) - 1)
                if side == "BUY" and bid_q is not None and np.isfinite(bid_q[idx_mm]):
                    prices.append(float(bid_q[idx_mm]))
                elif side == "SELL" and ask_q is not None and np.isfinite(ask_q[idx_mm]):
                    prices.append(float(ask_q[idx_mm]))
                elif q > 0:
                    # fallback: derive from gross cash flow
                    prices.append(abs(float(g_cf)) / float(q))
                else:
                    prices.append(float("nan"))
        else:
            # fallback: |gross_cash_flow| / qty
            with np.errstate(divide="ignore", invalid="ignore"):
                prices = np.where(qty > 0, np.abs(gross_cf) / qty, float("nan")).tolist()

        return pd.DataFrame({
            "datetime": dts,
            "timestamp_ns": ts_fills,
            "side": sides,
            "qty": qty,
            "fill_price": prices,
            "gross_cash_flow": gross_cf,  # pre-fee cash movement (~±order_qty_usd)
            "net_cash_flow": net_cf,  # post-fee cash movement
            "fee_at_fill": fee_cf,  # negative = cost paid
        })
