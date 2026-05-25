"""mmbt v1.1 — clean, no-numba market-making backtest framework"""

from .runner import BacktestRunner, BacktestResult
from .asset import AssetConfig
from .metrics import DEFAULT_METRICS, MetricsRegistry
from .optimizer import (
    BayesianOptimizer,
    WalkForwardCV,
    KFoldCV,
    TrainTestSplit,
    OptimizationResult,
    save_optimization,
)
from .algorithms import (
    OBIStrategy, GTQBStrategy, AvellanedaStoikovStrategy,
    AlphaSignal, OBISignal,
)

__all__ = [
    # runner
    "BacktestRunner", "BacktestResult",
    # config
    "AssetConfig",
    # metrics
    "DEFAULT_METRICS", "MetricsRegistry",
    # optimizer
    "BayesianOptimizer",
    "WalkForwardCV", "KFoldCV", "TrainTestSplit",
    "OptimizationResult", "save_optimization",
    # strategies
    "OBIStrategy", "GTQBStrategy", "AvellanedaStoikovStrategy",
    # signals
    "AlphaSignal", "OBISignal",
]
__version__ = "1.1.0"
