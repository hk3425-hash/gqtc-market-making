"""
optimizer/results.py — Trial / result dataclasses and disk saver.

Directory layout written by save_optimization()
-----------------------------------------------
results_optimizer/
  {name}_{YYYYMMDD_HHMMSS}/
      ├── reports/
      │     ├── summary.txt          ← human-readable overview
      │     ├── best_params.json     ← best parameter set
      │     └── trials.csv           ← all trials (params + scores + metrics)
      ├── plots/
      │     ├── convergence.png      ← train (and test) score over trials
      │     ├── param_importance.png ← |correlation| of each variable with train score
      │     └── param_scatter.png    ← parameter vs train score
      └── folds/                     ← only if CV was used
            fold_0_summary.txt
            fold_1_summary.txt
            ...
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Dark theme palette (mirrors charts.py)
_BG = "#0f1117"
_SURFACE = "#1a1d27"
_TEXT = "#e0e0e0"
_MUTED = "#666677"
_BLUE = "#4a9eff"
_GREEN = "#2ecc71"
_RED = "#e74c3c"
_YELLOW = "#f39c12"
_BORDER = "#2a2d3a"


# ── dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    """Outcome of one train/test fold within a trial."""
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_score: float
    test_score: Optional[float] = None
    train_metrics: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)
    elapsed_s: float = 0.0
    failed: bool = False
    error: str = ""


@dataclass
class Trial:
    """One optimizer trial (possibly aggregated over CV folds)."""
    no: int
    params: dict
    score: float  # CV-aggregated train score (what the optimizer maximizes)
    test_score: Optional[float] = None  # CV-aggregated out-of-sample score
    fold_results: List[FoldResult] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    elapsed_s: float = 0.0
    failed: bool = False
    error: str = ""


@dataclass
class OptimizationResult:
    """Everything produced by a completed BayesianOptimizer.run() call."""
    best_params: dict
    best_score: float
    best_test_score: Optional[float]
    trials: List[Trial]
    strategy_name: str
    cv_name: str
    elapsed_s: float
    variable_names: List[str]

    # ── convenience ───────────────────────────────────────────────────────

    @property
    def best_trial(self) -> Optional[Trial]:
        good = [t for t in self.trials if not t.failed]
        return max(good, key=lambda t: t.score) if good else None

    def top_n(self, n: int = 5) -> List[Trial]:
        """Return the n best non-failed trials, sorted by score descending."""
        good = [t for t in self.trials if not t.failed]
        return sorted(good, key=lambda t: t.score, reverse=True)[:n]

    def to_dataframe(self):
        """Return a pandas DataFrame with one row per trial."""
        import pandas as pd
        rows = []
        for t in self.trials:
            row = {
                "trial": t.no,
                "score": t.score,
                "test_score": t.test_score,
                "elapsed_s": t.elapsed_s,
                "failed": t.failed,
                "error": t.error,
            }
            row.update(t.params)
            row.update({f"metric_{k}": v for k, v in t.metrics.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def save(
            self,
            name: Optional[str] = None,
            base_dir: str | Path = "results_optimizer",
            dpi: int = 150,
    ) -> Path:
        return save_optimization(self, name=name, base_dir=base_dir, dpi=dpi)


# ── saver ─────────────────────────────────────────────────────────────────────

def save_optimization(
        result: OptimizationResult,
        name: Optional[str] = None,
        base_dir: str | Path = "results_optimizer",
        dpi: int = 150,
) -> Path:
    """
    Persist all optimization artefacts to disk and return the run directory.
    """
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = name or result.strategy_name
    run_dir = Path(base_dir) / f"{run_name}_{ts_str}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── subdirectories ────────────────────────────────────────────────────
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(exist_ok=True)

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # ── text / data (reports) ─────────────────────────────────────────────
    (reports_dir / "summary.txt").write_text(_build_summary(result), encoding="utf-8")
    logger.info("Summary written: reports/summary.txt")

    _json_dump(result.best_params, reports_dir / "best_params.json")
    logger.info("Best params written: reports/best_params.json")

    _trials_csv(result.trials, reports_dir / "trials.csv")
    logger.info("Trials CSV written: reports/trials.csv  (%d rows)", len(result.trials))

    # ── plots ─────────────────────────────────────────────────────────────
    _save_convergence(result, plots_dir / "convergence.png", dpi)
    logger.info("Convergence plot written: plots/convergence.png")

    if len(result.variable_names) > 1:
        _save_param_importance(result, plots_dir / "param_importance.png", dpi)
        logger.info("Param importance plot written: plots/param_importance.png")

    if len(result.variable_names) >= 2:
        _save_param_scatter(result, plots_dir / "param_scatter.png", dpi)
        logger.info("Param scatter plots written: plots/param_scatter.png")

    # ── per-fold details (best trial only) ────────────────────────────────
    best = result.best_trial
    if best and best.fold_results:
        folds_dir = run_dir / "folds"
        folds_dir.mkdir(exist_ok=True)
        for fr in best.fold_results:
            p = folds_dir / f"fold_{fr.fold}_summary.txt"
            p.write_text(_build_fold_summary(fr), encoding="utf-8")
        logger.info("Fold summaries written: folds/  (%d folds)", len(best.fold_results))

    logger.info("Optimization results saved to: %s", run_dir)
    return run_dir


# ── text builders ─────────────────────────────────────────────────────────────

def _build_summary(result: OptimizationResult) -> str:
    good = [t for t in result.trials if not t.failed]
    failed = len(result.trials) - len(good)
    best = result.best_trial

    lines = [
        "=" * 70,
        "OPTIMIZATION SUMMARY",
        "=" * 70,
        f"Strategy      : {result.strategy_name}",
        f"CV Method     : {result.cv_name}",
        f"Trials        : {len(result.trials)}  ({failed} failed, {len(good)} successful)",
        f"Runtime       : {result.elapsed_s:.1f}s",
        "",
        "BEST RESULT",
        "-" * 70,
        f"  Train score : {result.best_score:.6f}",
    ]

    if result.best_test_score is not None:
        lines.append(f"  Test  score : {result.best_test_score:.6f}")

    if best:
        lines += ["", "BEST PARAMETERS", "-" * 70]
        for k, v in best.params.items():
            val_str = f"{v:.6g}" if isinstance(v, float) else str(v)
            lines.append(f"  {k:<32} {val_str}")

        if best.metrics:
            lines += ["", "BEST TRIAL — AGGREGATED METRICS", "-" * 70]
            for k, v in best.metrics.items():
                val_str = f"{v:.6f}" if isinstance(v, float) else str(v)
                lines.append(f"  {k:<32} {val_str}")

        if best.fold_results:
            lines += ["", "FOLD BREAKDOWN (best trial)", "-" * 70]
            lines.append(f"  {'Fold':<6}  {'Train':>10}  {'Test':>10}  Period")
            lines.append("  " + "-" * 55)
            for fr in best.fold_results:
                ts_str = f"{fr.test_score:>10.4f}" if fr.test_score is not None else f"{'N/A':>10}"
                status = "FAIL" if fr.failed else ""
                lines.append(
                    f"  {fr.fold:<6}  {fr.train_score:>10.4f}  {ts_str}  "
                    f"{fr.train_start} → {fr.test_end}  {status}"
                )

    if good:
        scores = [t.score for t in good]
        lines += [
            "",
            "SCORE DISTRIBUTION (train, successful trials)",
            "-" * 70,
            f"  mean   {float(np.mean(scores)):>12.6f}",
            f"  std    {float(np.std(scores)):>12.6f}",
            f"  min    {float(np.min(scores)):>12.6f}",
            f"  max    {float(np.max(scores)):>12.6f}",
        ]

    lines.append("=" * 70)
    return "\n".join(lines)


def _build_fold_summary(fr: FoldResult) -> str:
    lines = [
        "=" * 60,
        f"FOLD {fr.fold} SUMMARY",
        "=" * 60,
        f"Train : {fr.train_start} → {fr.train_end}",
        f"Test  : {fr.test_start} → {fr.test_end}",
        f"Elapsed : {fr.elapsed_s:.1f}s",
        "",
        f"Train score : {fr.train_score:.6f}",
    ]
    if fr.test_score is not None:
        lines.append(f"Test  score : {fr.test_score:.6f}")
    if fr.failed:
        lines += ["", f"FAILED: {fr.error}"]
        return "\n".join(lines)

    if fr.train_metrics:
        lines += ["", "TRAIN METRICS", "-" * 40]
        for k, v in fr.train_metrics.items():
            val_str = f"{v:.6f}" if isinstance(v, float) else str(v)
            lines.append(f"  {k:<30} {val_str}")
    if fr.test_metrics:
        lines += ["", "TEST METRICS", "-" * 40]
        for k, v in fr.test_metrics.items():
            val_str = f"{v:.6f}" if isinstance(v, float) else str(v)
            lines.append(f"  {k:<30} {val_str}")
    return "\n".join(lines)


# ── CSV / JSON helpers ────────────────────────────────────────────────────────

def _trials_csv(trials: List[Trial], path: Path) -> None:
    if not trials:
        return
    all_param_keys = list(dict.fromkeys(k for t in trials for k in t.params))
    all_metric_keys = list(dict.fromkeys(k for t in trials for k in t.metrics))
    fieldnames = [
        "trial", "score", "test_score", "elapsed_s", "failed", "error",
        *all_param_keys,
        *[f"metric_{k}" for k in all_metric_keys],
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for t in trials:
            row: dict = {
                "trial": t.no, "score": t.score, "test_score": t.test_score,
                "elapsed_s": t.elapsed_s, "failed": t.failed, "error": t.error,
            }
            row.update(t.params)
            row.update({f"metric_{k}": v for k, v in t.metrics.items()})
            w.writerow(row)


def _json_dump(d: dict, path: Path) -> None:
    safe: dict = {}
    for k, v in d.items():
        try:
            json.dumps(v)
            safe[k] = v
        except (TypeError, ValueError):
            safe[k] = str(v)
    path.write_text(json.dumps(safe, indent=2), encoding="utf-8")


# ── plot helpers ──────────────────────────────────────────────────────────────

def _ax_dark(ax, title: str = "") -> None:
    """Apply dark theme to a matplotlib Axes."""
    ax.set_facecolor(_SURFACE)
    if title:
        ax.set_title(title, color=_TEXT, fontsize=11, pad=8)
    ax.tick_params(colors=_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(_BORDER)
    ax.xaxis.label.set_color(_MUTED)
    ax.yaxis.label.set_color(_MUTED)


def _save_convergence(result: OptimizationResult, path: Path, dpi: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        good = [t for t in result.trials if not t.failed]
        if not good:
            return

        has_test = any(t.test_score is not None for t in good)
        ncols = 2 if has_test else 1
        fig, axes = plt.subplots(1, ncols, figsize=(13 if has_test else 8, 4.5), squeeze=False)
        fig.patch.set_facecolor(_BG)

        # ── train convergence ─────────────────────────────────────────────
        ax = axes[0, 0]
        _ax_dark(ax, "Convergence — Train Score")

        all_scores = [t.score if not t.failed else float("nan") for t in result.trials]
        xs = list(range(len(all_scores)))
        best_so_far = []
        cur_best = float("-inf")
        for s in all_scores:
            if s == s and s != float("-inf"):  # not nan, not -inf
                cur_best = max(cur_best, s)
            best_so_far.append(cur_best if cur_best > float("-inf") else float("nan"))

        ax.scatter(xs, all_scores, color=_BLUE, s=18, alpha=0.55, zorder=3, label="Trial")
        ax.plot(xs, best_so_far, color=_GREEN, lw=2.0, zorder=4, label="Best so far")
        if result.trials:
            ax.axvline(
                min(len(result.trials) - 1, 9),  # rough n_initial marker
                color=_YELLOW, lw=1.0, ls="--", alpha=0.6, label="~end random"
            )
        ax.set_xlabel("Trial")
        ax.set_ylabel("Score")
        ax.legend(fontsize=8, facecolor=_SURFACE, labelcolor=_TEXT, framealpha=0.8)

        # ── test scores ───────────────────────────────────────────────────
        if has_test:
            ax2 = axes[0, 1]
            _ax_dark(ax2, "Out-of-Sample — Test Score")
            test_scores = [
                t.test_score if (t.test_score is not None and not t.failed) else float("nan")
                for t in result.trials
            ]
            ax2.scatter(xs, test_scores, color=_RED, s=18, alpha=0.55, zorder=3, label="Test score")
            best_test = []
            cur = float("-inf")
            for s in test_scores:
                if s == s and s != float("-inf"):
                    cur = max(cur, s)
                best_test.append(cur if cur > float("-inf") else float("nan"))
            ax2.plot(xs, best_test, color=_GREEN, lw=2.0, label="Best so far")
            ax2.set_xlabel("Trial")
            ax2.set_ylabel("Score")
            ax2.legend(fontsize=8, facecolor=_SURFACE, labelcolor=_TEXT, framealpha=0.8)

        plt.tight_layout(pad=1.5)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=_BG)
        plt.close(fig)

    except Exception as exc:
        logger.warning("Convergence plot failed: %s", exc)


def _save_param_importance(result: OptimizationResult, path: Path, dpi: int) -> None:
    """Bar chart: |Pearson correlation| of each variable with train score."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        good = [t for t in result.trials if not t.failed]
        if len(good) < 5:
            return

        names = result.variable_names
        scores = np.array([t.score for t in good], dtype=float)
        importances: List[float] = []

        for name in names:
            vals = np.array([t.params.get(name, np.nan) for t in good], dtype=float)
            mask = np.isfinite(vals) & np.isfinite(scores)
            if mask.sum() < 3:
                importances.append(0.0)
                continue
            corr = np.corrcoef(vals[mask], scores[mask])[0, 1]
            importances.append(abs(float(corr)) if corr == corr else 0.0)

        fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.4), 4.5))
        fig.patch.set_facecolor(_BG)
        _ax_dark(ax, "|Pearson r| with Train Score — Parameter Importance")

        colors = [_BLUE if imp >= 0.1 else _MUTED for imp in importances]
        bars = ax.bar(names, importances, color=colors, alpha=0.85, width=0.6)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("|Pearson r|")
        ax.axhline(0.3, color=_YELLOW, lw=0.8, ls="--", alpha=0.5, label="0.3 threshold")
        ax.legend(fontsize=8, facecolor=_SURFACE, labelcolor=_TEXT, framealpha=0.8)

        for bar, imp in zip(bars, importances):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                f"{imp:.2f}", ha="center", va="bottom",
                color=_TEXT, fontsize=9, fontweight="bold",
            )

        plt.tight_layout(pad=1.5)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=_BG)
        plt.close(fig)

    except Exception as exc:
        logger.warning("Param importance plot failed: %s", exc)


def _save_param_scatter(result: OptimizationResult, path: Path, dpi: int) -> None:
    """Scatter plot of each variable vs train score."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        good = [t for t in result.trials if not t.failed]
        names = result.variable_names
        if len(good) < 5 or not names:
            return

        ncols = min(3, len(names))
        nrows = (len(names) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5 * ncols, 4 * nrows),
                                 squeeze=False)
        fig.patch.set_facecolor(_BG)
        scores = np.array([t.score for t in good], dtype=float)

        for idx, name in enumerate(names):
            r, c = divmod(idx, ncols)
            ax = axes[r][c]
            _ax_dark(ax, name)
            vals = np.array([t.params.get(name, np.nan) for t in good], dtype=float)
            sc = ax.scatter(vals, scores, c=scores, cmap="plasma",
                            s=22, alpha=0.7, zorder=3)
            ax.set_xlabel(name)
            ax.set_ylabel("Score")
            plt.colorbar(sc, ax=ax, pad=0.02).ax.tick_params(colors=_MUTED, labelsize=7)

        # Hide unused subplots
        for idx in range(len(names), nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r][c].set_visible(False)

        fig.suptitle("Parameter vs Train Score", color=_TEXT, fontsize=12, y=1.01)
        plt.tight_layout(pad=1.5)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=_BG)
        plt.close(fig)

    except Exception as exc:
        logger.warning("Param scatter plot failed: %s", exc)
