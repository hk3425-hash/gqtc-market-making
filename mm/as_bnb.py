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

START = "20260412_05"
END = "20260412_13"

asset_bnb = AssetConfig(
    symbol="bnbusd",
    start_date=START,
    end_date=END,
    roi_lb=0.0,
    roi_ub=100_000.0,
    maker_fee=0.0005,
    taker_fee=0.001
)

strat = AvellanedaStoikovStrategy(
    interval=100_000_000,  # 100ms
    gamma=0.00001,
    k=2000.0,
    horizon=300.0,
    order_qty_usd=1000.0,
    max_position_usd=1_000_000.0,
    min_half_spread=0.001
)

result = BacktestRunner(
    assets=[asset_bnb],
    strategy=strat,
    book_size=10_000_000.0,
).run()

result.save("as_bnb", params=strat.params_dict)

# zoom = result.time_range_ns("05:05", "05:15")
# result.plot(time_range=zoom)
