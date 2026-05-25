from mmbt import GTQBStrategy, BacktestRunner, AssetConfig
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Quiet noisy third-party loggers
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

START = "20260214_23"
END = "20260227_23"

asset_usdcusd = AssetConfig(
    symbol="usdcusd",
    start_date=START,
    end_date=END,
    tick_size=0.00001,
    lot_size=0.1,
    roi_lb=0.0,
    roi_ub=2.0,
    maker_fee=0.0,
    taker_fee=0.0001
)

asset_usdtusd = AssetConfig(
    symbol="usdtusd",
    start_date=START,
    end_date=END,
    tick_size=0.00001,
    lot_size=0.1,
    roi_lb=0.0,
    roi_ub=2.0,
    maker_fee=0.0,
    taker_fee=0.0001
)

strat = GTQBStrategy(
    interval=1_000_000_000,  # 1s
    order_qty_usd=100.0,

    # Per-asset params (index 0 = USDCUSD, index 1 = USDTUSD)
    max_position_usd_qb=[1000.0, 1000.0],
    qty_threshold=[10.0, 10.0],
    grid_num_qb=[2, 2],

    max_position_usd_gt=[1000.0, 1000.0],
    skew_gt=[0.0, 0.0],
    grid_num_gt=[10, 10],
    grid_interval_usd_gt=[0.0001, 0.0001],
    half_spread_usd_gt=[0.001, 0.0001],
)

result = BacktestRunner(
    assets=[asset_usdcusd, asset_usdtusd],
    strategy=strat,
    book_size=4000.0,
).run()

result.save("gtqb_usdc_usdt", params=strat.params_dict)
