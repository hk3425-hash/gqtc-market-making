"""
mmbt.optimizer — Bayesian hyperparameter optimizer with cross-validation.

Public API
----------
Classes
    BayesianOptimizer     Main optimizer (wraps scikit-optimize GP)
    WalkForwardCV         Walk-forward time-series CV splitter
    KFoldCV               Purged K-Fold CV splitter
    TrainTestSplit        Simple single train/test cut
    CVSplit               Dataclass returned by splitters
    OptimizationResult    Return value of BayesianOptimizer.run()
    Trial                 Single trial record (params + scores + fold details)
    FoldResult            Single fold record within a CV trial

Functions
    save_optimization     Persist an OptimizationResult to results_optimizer/

Quick start
-----------
    from mmbt.optimizer import BayesianOptimizer, WalkForwardCV
    from mmbt import AvellanedaStoikovStrategy, AssetConfig

    opt = BayesianOptimizer(
        strategy_class = AvellanedaStoikovStrategy,
        assets         = [AssetConfig("solusd", "20260101", "20260201",
                                      tick_size=0.01, lot_size=0.01)],
        book_size      = 2000.0,
        fixed = dict(
            order_qty_usd    = 100.0,
            max_position_usd = 1000.0,
            interval         = 100_000_000,
        ),
        variable = dict(
            gamma   = (0.001, 0.1,   "log-uniform"),
            k       = (5.0,   200.0, "log-uniform"),
            horizon = (60.0,  600.0),
        ),
        cv        = WalkForwardCV(n_splits=5, train_ratio=0.6),
        n_calls   = 40,
        n_initial = 15,
    )

    result = opt.run()
    result.save()                    # writes to results_optimizer/
    print(result.best_params)
    df = result.to_dataframe()
"""

from .bayesian import BayesianOptimizer
from .cv import WalkForwardCV, KFoldCV, TrainTestSplit, CVSplit
from .results import Trial, FoldResult, OptimizationResult, save_optimization

__all__ = [
    # optimizer
    "BayesianOptimizer",
    # CV splitters
    "WalkForwardCV",
    "KFoldCV",
    "TrainTestSplit",
    "CVSplit",
    # result dataclasses
    "Trial",
    "FoldResult",
    "OptimizationResult",
    "save_optimization",
]
