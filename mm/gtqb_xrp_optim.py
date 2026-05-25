"""
gtqb_xrp_sol_optim.py — GTQB optimisation across 2 assets (XRP + SOL).

Demonstrates all three variable-param formats accepted by BayesianOptimizer:

  1. Scalar range      — one GP dimension, same optimised value for both assets
       qty_threshold = (1.0, 100.0, "log-uniform")

  2. Fixed list        — passed to the strategy unchanged every trial
       max_position_usd_gt = [0.0, 0.0]

  3. Per-asset range   — one independent GP dimension per asset,
     reassembled into [v0, v1] before the strategy is instantiated
       max_position_usd_qb = [(200.0, 3000.0), (500.0, 8000.0)]

How per-asset ranges work internally
--------------------------------------
  max_position_usd_qb = [(200.0, 3000.0), (500.0, 8000.0)]

  creates two skopt dimensions:
      max_position_usd_qb__0  ∈ [200,  3000]   (for XRP)
      max_position_usd_qb__1  ∈ [500,  8000]   (for SOL)

  At each trial the GP suggests independent values for both, and they are
  packed back into a single list  [v_xrp, v_sol]  before the strategy is
  instantiated — exactly as if you had written it by hand.
"""

# import logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s  %(levelname)-8s  %(message)s",
#     datefmt="%H:%M:%S",
# )

from mmbt import (
    AssetConfig,
    BayesianOptimizer,
    GTQBStrategy,
    WalkForwardCV,
)


# ── objective ─────────────────────────────────────────────────────────────────

def objective(result) -> float:
    """Maximise trading volume weighted by squared (1 + Return)."""
    ret = result.metrics.get("Return", 0.0)
    vol = result.metrics.get("DailyTradingVolume", 0.0)
    return vol * (2.0 + ret) ** 2 if ret > -1 else - vol * (-2.0 + ret) ** 2


# ── assets ────────────────────────────────────────────────────────────────────

assets = [
    AssetConfig(
        symbol="xrpusd",
        start_date="20260308_00",
        end_date="20260309_00",
        tick_size=0.00001,
        lot_size=0.1,
        roi_lb=0.0,
        roi_ub=100.0,
        maker_fee=0.0005,
        taker_fee=0.001
    )
]

# ── optimizer ─────────────────────────────────────────────────────────────────

opt = BayesianOptimizer(
    strategy_class=GTQBStrategy,
    assets=assets,
    book_size=1500.0,
    objective=objective,

    fixed=dict(
        interval=1_000_000_000,      # 1 s — shared scalar
        order_qty_usd=100.0,         # shared scalar

        # QB
        max_position_usd_qb=1000,

        # GT component disabled for both assets
        max_position_usd_gt=[0.0, 0.0],
        skew_gt=[0.0, 0.0],
        grid_num_gt=[10, 10],
        grid_interval_usd_gt=[0.0001, 0.01],
        half_spread_usd_gt=[0.001, 0.05],
    ),

    variable=dict(
        # ── format 1: scalar range ────────────────────────────────────────
        # One GP dimension.  The same optimised value is passed to BOTH
        # assets (GTQBStrategy broadcasts it via _arr(v)).
        # Use this when you believe the two assets share a natural scale
        # for a parameter and want fewer search dimensions.
        grid_num_qb=(1, 10),          # Integer — same grid depth for both

        # ── format 2: fixed list ──────────────────────────────────────────
        # Passed unchanged every trial.  Useful for params you have already
        # tuned manually or that have a hard asset-specific constraint.
        # (nothing here in this example — shown for illustration)
        # my_fixed_param=[42.0, 99.0],

        # ── format 3: per-asset range ─────────────────────────────────────
        # One independent GP dimension per asset.  Reassembled into a list
        # [v_xrp, v_sol] before the strategy is instantiated each trial.
        # Use this when the two assets operate at very different price /
        # volume scales and need independent tuning.
        qty_threshold=(0.0,  1_000_000.0)
    ),

    cv=WalkForwardCV(n_splits=3, train_ratio=0.5, method="expanding"),
    score_agg="mean",
    evaluate_test=True,
    n_calls=50,
    n_initial=20,
    verbose=True,
)

# ── run ───────────────────────────────────────────────────────────────────────

result = opt.run()
result.save(name="gtqb_xrp_optim")