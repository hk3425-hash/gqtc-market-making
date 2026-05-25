"""
optimizer/bayesian.py — Bayesian hyperparameter optimizer with optional CV.

Quick start
-----------
    from mmbt.optimizer import BayesianOptimizer, WalkForwardCV

    opt = BayesianOptimizer(
        strategy_class = AvellanedaStoikovStrategy,
        assets         = [AssetConfig("solusd", "20260101", "20260201", ...)],
        book_size      = 2000.0,

        # Fixed params — same dict style as variable
        fixed = dict(
            order_qty_usd    = 100.0,
            max_position_usd = 1000.0,
            interval         = 100_000_000,
        ),

        # Variable params — (low, high) or (low, high, prior)
        # prior: "uniform" (default) or "log-uniform"
        variable = dict(
            gamma   = (0.001, 0.1,   "log-uniform"),
            k       = (5.0,   200.0, "log-uniform"),
            horizon = (60.0,  600.0),          # defaults to "uniform"
        ),

        # Optional cross-validation
        cv          = WalkForwardCV(n_splits=5, train_ratio=0.6),
        score_agg   = "mean",   # or "min" for conservative, or callable
        evaluate_test = True,   # run test folds for out-of-sample reporting

        n_calls   = 40,
        n_initial = 15,
    )

    opt_result = opt.run()
    opt_result.save()                        # → results_optimizer/
    print(opt_result.best_params)
    df = opt_result.to_dataframe()
"""

from __future__ import annotations

import logging
import time
from typing import Callable, List, Literal, Optional, Tuple, Type

import numpy as np

from .cv import CVSplit, WalkForwardCV, KFoldCV, TrainTestSplit
from .results import FoldResult, OptimizationResult, Trial, save_optimization

log = logging.getLogger(__name__)


# ── parameter space parser ────────────────────────────────────────────────────

def _build_skopt_space(variable: dict):
    """
    Parse variable dict into (names, skopt_dimensions).

    Accepted spec formats per parameter:
        (lo, hi)             — Real uniform  (or Integer if both are int)
        (lo, hi, "uniform")  — Real uniform
        (lo, hi, "log-uniform") — Real log-uniform
    """
    try:
        from skopt.space import Real, Integer
    except ImportError:
        raise ImportError(
            "scikit-optimize is required for BayesianOptimizer: "
            "pip install scikit-optimize"
        )

    names: List[str] = []
    dims: List = []

    for name, spec in variable.items():
        if not isinstance(spec, (list, tuple)) or len(spec) not in (2, 3):
            raise ValueError(
                f"Variable '{name}': spec must be (lo, hi) or (lo, hi, prior), "
                f"got {spec!r}"
            )

        lo, hi = spec[0], spec[1]
        prior = spec[2] if len(spec) == 3 else "uniform"

        if prior not in ("uniform", "log-uniform"):
            raise ValueError(
                f"Variable '{name}': prior must be 'uniform' or 'log-uniform', "
                f"got {prior!r}"
            )
        if lo >= hi:
            raise ValueError(
                f"Variable '{name}': lo ({lo}) must be < hi ({hi})"
            )

        names.append(name)
        if isinstance(lo, int) and isinstance(hi, int):
            dims.append(Integer(lo, hi, name=name))
        else:
            dims.append(Real(float(lo), float(hi), prior=prior, name=name))

    return names, dims


# ── main optimizer ────────────────────────────────────────────────────────────

class BayesianOptimizer:
    """
    Gaussian-process Bayesian optimizer wrapping scikit-optimize.

    Supports optional time-series cross-validation (WalkForwardCV, KFoldCV,
    or TrainTestSplit) so that the objective function is an aggregate of
    multiple train folds rather than a single full-period backtest.

    Parameters
    ----------
    strategy_class  : un-instantiated BaseStrategy subclass
    assets          : list of AssetConfig objects defining the full date range
    book_size       : capital for metric normalisation
    fixed           : dict of parameters that do NOT vary during optimisation.
                      Same dict style as ``variable``.
                          dict(interval=100_000_000, order_qty_usd=100.0)
    variable        : dict of parameters to optimise. Each value is a tuple:
                          (lo, hi)                 — uniform search
                          (lo, hi, "log-uniform")  — log-uniform search
                      Example:
                          dict(
                              gamma   = (0.001, 0.1, "log-uniform"),
                              k       = (5.0, 200.0, "log-uniform"),
                              horizon = (60.0, 600.0),
                          )
    objective       : callable(BacktestResult) → float to maximise
                      Default: Sharpe ratio  (r.metrics.get("SR"))
    cv              : cross-validator or None.
                      - None → single full-period train (no CV)
                      - WalkForwardCV(...)
                      - KFoldCV(...)
                      - TrainTestSplit(...)
    score_agg       : how to combine per-fold train scores into one number
                      "mean"     — arithmetic mean (default, balanced)
                      "min"      — worst fold (conservative / robust)
                      callable   — e.g. lambda scores: np.percentile(scores, 25)
    evaluate_test   : if True and cv is given, also run the test splits
                      for out-of-sample reporting (does NOT affect the score
                      seen by the optimizer)
    n_calls         : total number of optimizer evaluations
    n_initial       : random seed evaluations before GP model is used
    verbose         : print per-trial one-liner to stdout in addition to logging
    show_progress   : display a live tqdm progress bar
    metrics         : tuple of hftbacktest metric classes
                      (default: mmbt DEFAULT_METRICS)
    name            : base name used when saving results
    """

    def __init__(
            self,
            strategy_class: Type,
            assets: list,
            book_size: float,
            fixed: dict,
            variable: dict,
            objective: Optional[Callable] = None,
            cv=None,
            score_agg: str | Callable = "mean",
            evaluate_test: bool = True,
            n_calls: int = 30,
            n_initial: int = 10,
            verbose: bool = False,
            show_progress: bool = True,
            metrics: Optional[tuple] = None,
            name: Optional[str] = None,
    ) -> None:
        self.strategy_class = strategy_class
        self.assets = assets
        self.book_size = book_size
        self.fixed = dict(fixed)
        self.variable = dict(variable)
        self.objective = objective or (lambda r: r.metrics.get("SR", float("-inf")))
        self.cv = cv
        self.score_agg = score_agg
        self.evaluate_test = evaluate_test
        self.n_calls = n_calls
        self.n_initial = n_initial
        self.verbose = verbose
        self.show_progress = show_progress
        self._metrics = metrics
        self.name = name or strategy_class.__name__

        # Populated during run()
        self._trials: List[Trial] = []

    # ── public API ────────────────────────────────────────────────────────

    def run(self) -> OptimizationResult:
        """
        Execute the optimization loop and return a complete OptimizationResult.

        The result can be saved immediately:
            result = opt.run()
            result.save()
        """
        try:
            from skopt import gp_minimize
        except ImportError:
            raise ImportError("pip install scikit-optimize")

        # Import the existing progress bar (lives in parent package)
        try:
            from ..progress import OptimizerProgressBar
        except ImportError:
            from mmbt.progress import OptimizerProgressBar

        names, dims = _build_skopt_space(self.variable)
        cv_name = repr(self.cv) if self.cv else "None (full period)"

        self._log_header(cv_name, names)

        # ── progress bar ──────────────────────────────────────────────────
        bar: Optional[OptimizerProgressBar] = None
        if self.show_progress:
            bar = OptimizerProgressBar(
                n_calls=self.n_calls,
                n_initial=self.n_initial,
                strategy_name=self.strategy_class.__name__,
            )
            bar.start()

        self._trials.clear()
        counter = [0]
        t_opt_start = time.perf_counter()

        # ── skopt objective (minimise negative score) ─────────────────────
        def _obj(values: list) -> float:
            no = counter[0]
            counter[0] += 1
            params = {**self.fixed, **{n: v for n, v in zip(names, values)}}
            t0 = time.perf_counter()

            log.info(
                "[Trial %3d/%d]  params=%s",
                no, self.n_calls - 1,
                "  ".join(
                    f"{n}={v:.4g}" if isinstance(v, float) else f"{n}={v}"
                    for n, v in zip(names, values)
                ),
            )

            try:
                trial = self._run_trial(no, params)
                trial.elapsed_s = time.perf_counter() - t0
                self._trials.append(trial)

                if bar:
                    bar.trial_done(no, trial.score, failed=False)

                test_str = (
                    f"{trial.test_score:.4f}"
                    if trial.test_score is not None
                    else "N/A"
                )
                log.info(
                    "[Trial %3d/%d]  score=%.4f  test=%s  elapsed=%.1fs",
                    no, self.n_calls - 1,
                    trial.score, test_str, trial.elapsed_s,
                )

                if self.verbose:
                    var_str = "  ".join(
                        f"{n}={v:.4g}" if isinstance(v, float) else f"{n}={v}"
                        for n, v in zip(names, values)
                    )
                    log.debug(
                        "  [%3d]  score=%+.4f  test=%s  %s  (%.1fs)",
                        no, trial.score, test_str, var_str, trial.elapsed_s,
                    )

                return -trial.score  # skopt minimises

            except Exception as exc:
                elapsed = time.perf_counter() - t0
                err_trial = Trial(
                    no=no, params=params, score=float("-inf"),
                    elapsed_s=elapsed, failed=True, error=str(exc),
                )
                self._trials.append(err_trial)
                if bar:
                    bar.trial_done(no, float("-inf"), failed=True)
                log.warning("[Trial %3d/%d]  FAILED: %s", no, self.n_calls - 1, exc)
                return float("inf")

        gp_minimize(
            func=_obj,
            dimensions=dims,
            n_calls=self.n_calls,
            n_initial_points=self.n_initial,
            random_state=42,
            verbose=False,
        )

        total_elapsed = time.perf_counter() - t_opt_start
        good = [t for t in self._trials if not t.failed]

        if not good:
            raise RuntimeError("All optimization trials failed.")

        best = max(good, key=lambda t: t.score)

        if bar:
            bar.finish(best_score=best.score)

        self._log_footer(best, names, total_elapsed)

        return OptimizationResult(
            best_params=best.params,
            best_score=best.score,
            best_test_score=best.test_score,
            trials=list(self._trials),
            strategy_name=self.strategy_class.__name__,
            cv_name=cv_name,
            elapsed_s=total_elapsed,
            variable_names=names,
        )

    # ── trial execution ───────────────────────────────────────────────────

    def _run_trial(self, no: int, params: dict) -> Trial:
        """
        Run one optimizer trial.

        Without CV: single backtest over the full asset date range.
        With CV   : one backtest per fold; scores are aggregated.
        """
        if self.cv is None:
            bt = self._backtest(params, self.assets)
            score = self.objective(bt)
            return Trial(no=no, params=params, score=score, metrics=bt.metrics)

        # ── CV mode ───────────────────────────────────────────────────────
        splits = self.cv.split(
            start_date=self.assets[0].start_date,
            end_date=self.assets[0].end_date,
        )
        fold_results: List[FoldResult] = []
        train_scores: List[float] = []

        for split in splits:
            log.debug(
                "  [Fold %d]  train [%s → %s]  test [%s → %s]",
                split.fold, split.train_start, split.train_end,
                split.test_start, split.test_end,
            )
            t_fold = time.perf_counter()

            try:
                # Train backtest
                train_bt = self._backtest(params, self._make_assets(split.train_start, split.train_end))
                train_score = self.objective(train_bt)

                # Test backtest (optional — for reporting only)
                test_score = None
                test_metrics = {}
                if self.evaluate_test:
                    test_bt = self._backtest(params, self._make_assets(split.test_start, split.test_end))
                    test_score = self.objective(test_bt)
                    test_metrics = test_bt.metrics

                fold_elapsed = time.perf_counter() - t_fold
                fr = FoldResult(
                    fold=split.fold,
                    train_start=split.train_start,
                    train_end=split.train_end,
                    test_start=split.test_start,
                    test_end=split.test_end,
                    train_score=train_score,
                    test_score=test_score,
                    train_metrics=train_bt.metrics,
                    test_metrics=test_metrics,
                    elapsed_s=fold_elapsed,
                )
                fold_results.append(fr)
                train_scores.append(train_score)

                log.debug(
                    "  [Fold %d]  train=%.4f  test=%s  elapsed=%.1fs",
                    split.fold, train_score,
                    f"{test_score:.4f}" if test_score is not None else "N/A",
                    fold_elapsed,
                )

            except Exception as exc:
                fold_elapsed = time.perf_counter() - t_fold
                fold_results.append(FoldResult(
                    fold=split.fold,
                    train_start=split.train_start, train_end=split.train_end,
                    test_start=split.test_start, test_end=split.test_end,
                    train_score=float("-inf"),
                    elapsed_s=fold_elapsed, failed=True, error=str(exc),
                ))
                train_scores.append(float("-inf"))
                log.warning("  [Fold %d]  FAILED: %s", split.fold, exc)

        agg_score = self._aggregate(train_scores)

        valid_test = [
            fr.test_score for fr in fold_results
            if fr.test_score is not None and fr.test_score == fr.test_score
        ]
        agg_test = float(np.mean(valid_test)) if valid_test else None

        # Average train metrics across folds
        metric_keys = list(dict.fromkeys(k for fr in fold_results for k in fr.train_metrics))
        agg_metrics: dict = {}
        for k in metric_keys:
            vals = [
                fr.train_metrics[k] for fr in fold_results
                if k in fr.train_metrics and isinstance(fr.train_metrics[k], (int, float))
            ]
            if vals:
                agg_metrics[k] = float(np.mean(vals))

        return Trial(
            no=no, params=params, score=agg_score,
            test_score=agg_test, fold_results=fold_results,
            metrics=agg_metrics,
        )

    def _aggregate(self, scores: List[float]) -> float:
        valid = [s for s in scores if s != float("-inf") and s == s]
        if not valid:
            return float("-inf")
        if callable(self.score_agg):
            return float(self.score_agg(valid))
        if self.score_agg == "min":
            return float(np.min(valid))
        return float(np.mean(valid))  # default: "mean"

    # ── asset / backtest helpers ──────────────────────────────────────────

    def _make_assets(self, start_date: str, end_date: str) -> list:
        """Clone the asset list with overridden start/end dates."""
        try:
            from ..asset import AssetConfig
        except ImportError:
            from mmbt.asset import AssetConfig

        result = []
        for asset in self.assets:
            new = AssetConfig(
                symbol=asset.symbol,
                start_date=start_date,
                end_date=end_date,
                tick_size=asset.tick_size,
                lot_size=asset.lot_size,
                maker_fee=asset.maker_fee,
                taker_fee=asset.taker_fee,
                latency_ns=asset.latency_ns,
                roi_lb=asset.roi_lb,
                roi_ub=asset.roi_ub,
                last_trades_capacity=asset.last_trades_capacity,
                data_dir=asset.data_dir,
            )
            result.append(new)
        return result

    def _backtest(self, params: dict, assets: list):
        """Instantiate strategy and run a single BacktestRunner.

        Runner and hftbacktest logs are suppressed to WARNING for the
        duration of the call so only optimizer-level logs reach the console.
        """
        try:
            from ..runner import BacktestRunner
        except ImportError:
            from mmbt.runner import BacktestRunner

        # Silence noisy sub-loggers while the optimizer is running.
        # We target the three main sources:
        #   mmbt.runner        — BacktestRunner INFO lines
        #   mmbt               — root mmbt logger
        #   hftbacktest        — underlying C-extension logger (if any)
        _muted = ["mmbt.runner", "mmbt", "hftbacktest"]
        _saved = {name: logging.getLogger(name).level for name in _muted}
        for name in _muted:
            logging.getLogger(name).setLevel(logging.WARNING)

        try:
            strat = self.strategy_class(**params)
            kw: dict = dict(
                assets=assets,
                strategy=strat,
                book_size=self.book_size,
                show_progress=False,
            )
            if self._metrics is not None:
                kw["metrics"] = self._metrics
            return BacktestRunner(**kw).run()
        finally:
            # Always restore — even if the backtest raises
            for name, level in _saved.items():
                logging.getLogger(name).setLevel(level)

    # ── convenience: save shortcut ────────────────────────────────────────

    def save(
            self,
            result: OptimizationResult,
            name: Optional[str] = None,
            base_dir: str = "results_optimizer",
            dpi: int = 150,
    ) -> "Path":
        return save_optimization(
            result, name=name or self.name, base_dir=base_dir, dpi=dpi
        )

    # ── logging ───────────────────────────────────────────────────────────

    def _log_header(self, cv_name: str, names: List[str]) -> None:
        sep = "=" * 60
        log.info(sep)
        log.info("OPTIMIZATION START")
        log.info("  Strategy    : %s", self.strategy_class.__name__)
        log.info("  CV          : %s", cv_name)
        log.info("  n_calls     : %d  (n_initial=%d)", self.n_calls, self.n_initial)
        log.info("  score_agg   : %s", self.score_agg)
        log.info("  eval_test   : %s", self.evaluate_test)
        log.info("")
        log.info("  Fixed params:")
        for k, v in self.fixed.items():
            val_str = f"{v:.4g}" if isinstance(v, float) else str(v)
            log.info("    %-28s %s", k, val_str)
        log.info("")
        log.info("  Variable params:")
        for k, spec in self.variable.items():
            log.info("    %-28s %s", k, spec)
        log.info(sep)

    def _log_footer(self, best: Trial, names: List[str], elapsed: float) -> None:
        sep = "=" * 60
        log.info(sep)
        log.info("OPTIMIZATION COMPLETE  runtime=%.1fs", elapsed)
        log.info(
            "Best trial #%d  train=%.6f%s",
            best.no, best.score,
            f"  test={best.test_score:.6f}" if best.test_score is not None else "",
        )
        log.info("Best parameters:")
        for k in names:
            v = best.params.get(k)
            log.info("  %-28s %s", k, f"{v:.6g}" if isinstance(v, float) else str(v))
        log.info(sep)