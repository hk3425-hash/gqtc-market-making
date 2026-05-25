from mmbt import AvellanedaStoikovStrategy, BacktestRunner, AssetConfig
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Quiet noisy third-party loggers
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

START = "20260308_16"
END = "20260308_18"

asset_xrp = AssetConfig(
    symbol="xrpusd",
    start_date=START,
    end_date=END,
    tick_size=0.00001,
    lot_size=0.1,
    roi_lb=0.0,
    roi_ub=100.0,
    maker_fee=0.0005,
    taker_fee=0.001
)

strat = AvellanedaStoikovStrategy(
    interval=100_000_000,  # 1ms
    gamma=0.0001,
    k=2000.0,
    horizon=300.0,
    order_qty_usd=100.0,
    max_position_usd=100_000.0,
    min_half_spread=0.00001
)

result = BacktestRunner(
    assets=[asset_xrp],
    strategy=strat,
    book_size=3000.0,
).run()

result.save("as_xrp", params=strat.params_dict)
