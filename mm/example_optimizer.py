"""
example_optimize.py — Full worked example of BayesianOptimizer.

Shows how to:
  1. Configure AssetConfig (tick/lot sizes, fees, date range)
  2. Define fixed vs variable hyperparameters
  3. Run the optimiser with a live progress bar
  4. Inspect results — top-N table, convergence plot, full DataFrame
  5. Re-run the best config and save all artefacts

Run with:
    python example_optimize.py
"""

from mmbt import (
    AssetConfig, BacktestRunner, BayesianOptimizer,
    AvellanedaStoikovStrategy, OBISignal,
)

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Asset configuration
#     tick_size / lot_size depend on the exchange and instrument.
#     No need to specify 'days' — BacktestRunner derives it from the dates.
# ─────────────────────────────────────────────────────────────────────────────

ASSET = AssetConfig(
    symbol     = "solusd",
    start_date = "20260308_13",       # YYYYMMDD  (full day: 00h–23h)
    end_date   = "20260308_14",       # inclusive
    tick_size  = 0.01,
    lot_size   = 0.01,
    maker_fee  = 0.0002,
    taker_fee  = 0.0004,
)

BOOK_SIZE = 2_000.0   # USD  — used only for metric normalisation

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Hyperparameter search space
#
#     fixed    — passed to the strategy constructor unchanged every trial
#     variable — tuned by the optimiser: {name: (low, high, "uniform"|"log-uniform")}
#
#     Note: 'days' and 'interval' are in fixed; the runner will auto-inject
#     the actual date span into every strategy instance anyway.
# ─────────────────────────────────────────────────────────────────────────────

FIXED = dict(
    order_qty_usd    = 100.0,
    max_position_usd = 1_000.0,
    interval         = 100_000_000,   # 100 ms in nanoseconds
)

VARIABLE = {
    # Avellaneda-Stoikov core parameters
    "gamma":           (0.0001, 10,   "log-uniform"),
    "k":               (20.0,   300.0, "log-uniform"),
    "horizon":         (30.0,  900.0, "uniform"),
    # Spread floor
    "min_half_spread": (0.001, 0.1,  "log-uniform"),
}

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Objective function
#
#     Returns a scalar to MAXIMISE.  Use any combination of BacktestResult
#     fields.  Here we penalise negative Sharpe by returning -inf.
# ─────────────────────────────────────────────────────────────────────────────

def objective(result) -> float:
    ret  = result.metrics.get("Return", 0.0)
    vol = result.metrics.get("DailyTradingVolume", 0.0)

    # Maximise SR/MDD ratio — rewards consistency and low drawdown
    return vol * (2 + ret) * (2 + ret)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Run the optimiser
# ─────────────────────────────────────────────────────────────────────────────

opt = BayesianOptimizer(
    strategy_class = AvellanedaStoikovStrategy,
    assets         = [ASSET],
    book_size      = BOOK_SIZE,
    fixed          = FIXED,
    variable       = VARIABLE,
    objective      = objective,
    n_calls        = 20,    # total evaluations
    n_initial      = 8,    # random seed points before GP takes over
    show_progress  = True,  # live in-place progress bar  ← NEW
)

result = opt.run()
result.save(name="as_sol_optim")