from __future__ import annotations

import logging

from tqdm import tqdm
from typing import Optional

log = logging.getLogger(__name__)


class BacktestProgressBar:
    """
    Tqdm-based progress bar for backtest steps.
    """

    def __init__(self, total_steps: int, desc: str = "Backtesting") -> None:
        self.total_steps = max(total_steps, 1)
        self.desc = desc
        self.pbar: Optional[tqdm] = None

    def start(self) -> None:
        self.pbar = tqdm(
            total=self.total_steps,
            desc=f"  {self.desc}",
            unit="step",
            leave=True,
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

    def update(self, step: int) -> None:
        if self.pbar:
            delta = step - self.pbar.n
            self.pbar.update(delta)

    def finish(self) -> None:
        if self.pbar:
            self.pbar.close()
            # Tqdm already moves to a new line on close — no print needed.


class OptimizerProgressBar:
    """
    Tqdm-based progress bar for Bayesian Optimizer trials.
    """

    def __init__(self, n_calls: int, n_initial: int, strategy_name: str = "") -> None:
        self.n_calls = n_calls
        self.n_initial = n_initial
        self.strategy_name = strategy_name
        self.pbar: Optional[tqdm] = None
        self._best = float("-inf")

    def start(self) -> None:
        desc = f"Opt: {self.strategy_name}" if self.strategy_name else "Optimizer"
        self.pbar = tqdm(
            total=self.n_calls,
            desc=f"  {desc}",
            unit="trial",
            ncols=100,
        )
        self.pbar.set_postfix({"phase": "seed", "best": "N/A"})

    def trial_done(self, trial_no: int, score: float, failed: bool = False) -> None:
        if self.pbar:
            if not failed and score > self._best:
                self._best = score

            phase = "seed" if trial_no < self.n_initial else "GP"
            self.pbar.set_postfix({
                "phase": phase,
                "last": "FAIL" if failed else f"{score:+.4f}",
                "best": f"{self._best:+.4f}",
            })
            self.pbar.update(1)

    def finish(self, best_score: float | None = None) -> None:
        if self.pbar:
            final_best = best_score if best_score is not None else self._best
            self.pbar.set_postfix({"phase": "done", "best": f"{final_best:+.4f}"})
            self.pbar.close()
            # Log the final result instead of printing it.
            log.info("Optimization complete — best score: %.6f", final_best)