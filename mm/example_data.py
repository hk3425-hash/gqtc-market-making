"""
example_data.py
===============
Standalone data-collection script.  Run from the `mm/` directory:

    python example_data.py

Configuration
-------------
Edit the SYMBOLS list below.  Exchange is auto-detected from the symbol name:

  Gemini  → symbols ending in  usd   (btcusd, ethusd, solusd, xrpusd, bnbusd, usdtusd)
  Binance → symbols ending in  usdt  (btcusdt, ethusdt, solusdt, …)

Mixed lists are supported: one connector per exchange runs concurrently and
both write into the same data/ directory.

Stop
----
Type  stop  (or quit / q / exit)  in the PyCharm Run console and press Enter.
All open hourly NPZ buffers are flushed cleanly before exit.

Output layout
-------------
mm/data/
  {symbol}/
    npz/        {symbol}_{YYYYMMDD}_{HH}.npz     ← hourly hftbacktest files
    snapshots/  {symbol}_{YYYYMMDD}_{HHMMSS}_book.npz
    info.json   tick_size, lot_size, roi_ub, …
"""
import asyncio
import sys
from pathlib import Path

# ── Make sure mmbt is importable from this script's location ─────────────────
_HERE = Path(__file__).resolve().parent          # mm/
sys.path.insert(0, str(_HERE))

from mmbt.data.collect import run_auto           # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
#  ★  EDIT THIS LIST  ★
# ══════════════════════════════════════════════════════════════════════════════
SYMBOLS = [
    # ── Gemini (competition assets) ───────────────────────────────────────────
    # "usdtusd",   # USDT/USD  — stable-coin pair, very tight spreads
    # "btcusd",    # BTC/USD
    # "xrpusd",    # XRP/USD
    # "solusd",    # SOL/USD
    "bnbusd",    # BNB/USD

    # ── Binance futures (uncomment to add) ────────────────────────────────────
    # "btcusdt",
    # "ethusdt",
    # "solusdt",
]

# Root data directory (relative to this file → mm/data/)
DATA_DIR = str(_HERE / "data")

# Fetch REST book snapshot before the WS stream starts (recommended: True)
SNAPSHOT_ON_START = True
# ══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    asyncio.run(
        run_auto(
            symbols=SYMBOLS,
            data_dir=DATA_DIR,
            snapshot_on_start=SNAPSHOT_ON_START,
        )
    )