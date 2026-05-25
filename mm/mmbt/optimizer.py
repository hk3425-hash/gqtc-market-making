"""
optimizer.py — Bayesian hyperparameter optimizer with live progress bar.

Requires: pip install scikit-optimize

Usage
-----
    from mmbt.optimizer import BayesianOptimizer

    opt = BayesianOptimizer(
        strategy_class = AvellanedaStoikovStrategy,
        assets         = [AssetConfig("solusd", ...)],
        book_size      = 2000.0,
        fixed = dict(
            order_qty_usd    = 100.0,
            max_position_usd = 1000.0,
            interval         = 100_000_000,
        ),
        variable = {
            "gamma":   (0.001, 0.1,   "log-uniform"),
            "k":       (5.0,   200.0, "log-uniform"),
            "horizon": (60.0,  600.0, "uniform"),
        },
        objective = lambda r: r.metrics.get("SR", float("-inf")),
        n_calls   = 30,
        n_initial = 10,
    )

    best_params, best_score = opt.run()
    opt.plot_convergence()
    df = opt.to_dataframe()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np


@dataclass
class Trial:
    no: int
    params: dict
    score: float
    metrics: dict
    elapsed: float
    failed: bool = False
    error: str = ""


class BayesianOptimizer:

    def __init__(
            self,
            strategy_class: Type,
            assets: list,
            book_size: float,
            fixed: dict,
            variable: Dict[str, Tuple],
            objective: Callable = None,
            n_calls: int = 30,
            n_initial: int = 10,
            verbose: bool = False,  # per-trial text log (off by default; bar replaces it)
            show_progress: bool = True,  # in-place progress bar
            metrics: tuple = None,
    ) -> None:
        self.strategy_class = strategy_class
        self.assets = assets
        self.book_size = book_size
        self.fixed = dict(fixed)
        self.variable = dict(variable)
        self.objective = objective or (lambda r: r.metrics.get("SR", float("-inf")))
        self.n_calls = n_calls
        self.n_initial = n_initial
        self.verbose = verbose
        self.show_progress = show_progress
        self._metrics = metrics
        self.trials: List[Trial] = []

    def run(self) -> Tuple[dict, float]:
        try:
            from skopt import gp_minimize
            from skopt.space import Real
        except ImportError:
            raise ImportError("pip install scikit-optimize")

        from .progress import OptimizerProgressBar

        names = list(self.variable.keys())
        space = [Real(lo, hi, prior=pr, name=n)
                 for n, (lo, hi, pr) in self.variable.items()]

        bar: Optional[OptimizerProgressBar] = None
        if self.show_progress:
            bar = OptimizerProgressBar(
                n_calls=self.n_calls,
                n_initial=self.n_initial,
                strategy_name=self.strategy_class.__name__,
            )
            bar.start()

        counter = [0]

        def _obj(values):
            no = counter[0];
            counter[0] += 1
            params = {**self.fixed, **{n: v for n, v in zip(names, values)}}
            t0 = time.perf_counter()
            try:
                result = self._trial(params)
                score = self.objective(result)
                elapsed = time.perf_counter() - t0
                self.trials.append(Trial(no, params, score, result.metrics, elapsed))
                if bar:
                    bar.trial_done(no, score, failed=False)
                if self.verbose:
                    var_str = "  ".join(f"{k}={v:.4g}" for k, v in zip(names, values))
                    print(f"\n  [{no:3d}]  score={score:>10.4f}  {var_str}  ({elapsed:.1f}s)")
            except Exception as e:
                elapsed = time.perf_counter() - t0
                self.trials.append(Trial(no, params, float("-inf"), {}, elapsed, True, str(e)))
                if bar:
                    bar.trial_done(no, float("-inf"), failed=True)
                if self.verbose:
                    print(f"\n  [{no:3d}]  FAILED: {e}")
                return float("inf")
            return -score

        gp_minimize(func=_obj, dimensions=space, n_calls=self.n_calls,
                    n_initial_points=self.n_initial, random_state=42, verbose=False)

        good = [t for t in self.trials if not t.failed]
        if not good:
            raise RuntimeError("All trials failed.")

        best = max(good, key=lambda t: t.score)

        if bar:
            bar.finish(best_score=best.score)

        if self.verbose:
            print(f"  Best trial #{best.no}  score={best.score:.6f}")
            for k in names:
                print(f"    {k:<28} {best.params[k]:.6g}")

        return best.params, best.score

    def _trial(self, params: dict):
        from .runner import BacktestRunner
        strat = self.strategy_class(**params)
        kw = dict(
            assets=self.assets,
            strategy=strat,
            book_size=self.book_size,
            show_progress=False,  # suppress inner bar during optimisation
        )
        if self._metrics:
            kw["metrics"] = self._metrics
        return BacktestRunner(**kw).run()

    def top_n(self, n=5) -> List[Trial]:
        return sorted([t for t in self.trials if not t.failed],
                      key=lambda t: t.score, reverse=True)[:n]

    def to_dataframe(self):
        import pandas as pd
        rows = []
        for t in self.trials:
            row = {"trial": t.no, "score": t.score, "elapsed_s": t.elapsed, "failed": t.failed}
            row.update(t.params)
            row.update({f"metric_{k}": v for k, v in t.metrics.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def plot_convergence(self, show=True):
        import matplotlib.pyplot as plt
        try:
            from .charts import _C, _apply_dark
        except ImportError:
            from mmbt_clean.charts import _C, _apply_dark

        scores = [t.score for t in self.trials if not t.failed]
        best = np.maximum.accumulate(scores)

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor(_C["bg"])
        _apply_dark(ax, "Bayesian Optimisation — Convergence")
        ax.plot(scores, color=_C["muted"], lw=1.0, alpha=0.6, label="Trial score")
        ax.plot(best, color=_C["green"], lw=2.0, label="Best so far")
        ax.axvline(self.n_initial - 1, color=_C["yellow"], lw=1.0, linestyle="--",
                   label=f"End random ({self.n_initial})")
        ax.set_xlabel("Trial");
        ax.set_ylabel("Score")
        ax.legend(fontsize=8, facecolor=_C["surface"], labelcolor=_C["text"])
        plt.tight_layout()
        if show:
            plt.show()
        return fig
