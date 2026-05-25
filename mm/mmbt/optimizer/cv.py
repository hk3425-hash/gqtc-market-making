"""
optimizer/cv.py — Time-series cross-validation splitters.

Three strategies are provided:

    WalkForwardCV   : expanding or sliding train window, n test folds
    KFoldCV         : purged k-fold (train = all folds before test fold)
    TrainTestSplit  : single train / test cut

All splitters expose a ``split(start_date, end_date) → List[CVSplit]`` method.

Date strings follow the mmbt convention:
    "YYYYMMDD"      — full day (00h)
    "YYYYMMDD_H"    — specific hour, no leading zero
    "YYYYMMDD_HH"   — specific hour, with leading zero

Example
-------
    cv = WalkForwardCV(n_splits=5, train_ratio=0.6, method="expanding")
    for split in cv.split("20260101", "20260201"):
        print(split)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Literal, Optional


# ── date helpers ──────────────────────────────────────────────────────────────

def _parse_dt(s: str) -> datetime:
    s = s.strip()
    if "_" in s:
        date_part, hour_part = s.split("_", 1)
        return datetime.strptime(date_part, "%Y%m%d").replace(hour=int(hour_part))
    return datetime.strptime(s, "%Y%m%d")


def _fmt_dt(dt: datetime) -> str:
    if dt.hour != 0:
        return f"{dt.strftime('%Y%m%d')}_{dt.hour}"
    return dt.strftime("%Y%m%d")


# ── split dataclass ───────────────────────────────────────────────────────────

@dataclass
class CVSplit:
    """One train / test fold."""
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str

    def __str__(self) -> str:
        return (
            f"Fold {self.fold:>2}: "
            f"train [{self.train_start} → {self.train_end}] | "
            f"test  [{self.test_start} → {self.test_end}]"
        )


# ── splitters ─────────────────────────────────────────────────────────────────

class WalkForwardCV:
    """
    Walk-forward cross-validation.

    The full date range is divided into a fixed-size training window plus
    ``n_splits`` equal test windows that advance chronologically.

    Parameters
    ----------
    n_splits    : number of test folds (and therefore optimizer evaluations per trial)
    train_ratio : fraction of the total period used for training
                  (ignored when method='sliding' and window_days is set)
    window_days : fixed training window length in days (method='sliding' only)
    method      : 'expanding' — train window grows with each fold (classic walk-forward)
                  'sliding'   — train window is fixed-length and shifts forward
    gap_hours   : purge buffer between train end and test start (avoids lookahead leakage)

    Example  (method='expanding', n_splits=3, train_ratio=0.5)
    -------
    Total:   |─────────────────────────────────────────────|
    Fold 0:  train [────────────────────] test [──────────]
    Fold 1:  train [──────────────────────────] test [────]
    Fold 2:  train [────────────────────────────────] test []
    """

    def __init__(
            self,
            n_splits: int = 5,
            train_ratio: float = 0.6,
            window_days: Optional[float] = None,
            method: Literal["expanding", "sliding"] = "expanding",
            gap_hours: int = 0,
    ) -> None:
        if not 0 < train_ratio < 1:
            raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
        if n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {n_splits}")
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.window_days = window_days
        self.method = method
        self.gap_hours = gap_hours

    def split(self, start_date: str, end_date: str) -> List[CVSplit]:
        start = _parse_dt(start_date)
        end = _parse_dt(end_date)
        total_h = (end - start).total_seconds() / 3600

        # Each test fold gets an equal slice of the remaining period
        test_h = total_h * (1.0 - self.train_ratio) / self.n_splits

        if self.method == "sliding" and self.window_days is not None:
            base_train_h = self.window_days * 24.0
        else:
            base_train_h = total_h * self.train_ratio

        splits: List[CVSplit] = []
        for i in range(self.n_splits):
            # How far from `start` does this test fold begin?
            test_offset_h = base_train_h + i * test_h

            if self.method == "expanding":
                tr_start = start
                tr_end = start + timedelta(hours=test_offset_h)
            else:  # sliding
                tr_end = start + timedelta(hours=test_offset_h)
                tr_start = max(start, tr_end - timedelta(hours=base_train_h))

            te_start = tr_end + timedelta(hours=self.gap_hours)
            te_end = te_start + timedelta(hours=test_h)

            if te_start >= end:
                break
            te_end = min(te_end, end)

            splits.append(CVSplit(
                fold=i,
                train_start=_fmt_dt(tr_start),
                train_end=_fmt_dt(tr_end),
                test_start=_fmt_dt(te_start),
                test_end=_fmt_dt(te_end),
            ))

        return splits

    def __repr__(self) -> str:
        return (
            f"WalkForwardCV(n_splits={self.n_splits}, "
            f"train_ratio={self.train_ratio}, method='{self.method}', "
            f"gap_hours={self.gap_hours})"
        )


class KFoldCV:
    """
    Purged K-Fold cross-validation for time series.

    The period is divided into ``n_splits`` equal folds. For fold *i*, the
    training set is all folds strictly *before* fold *i* (folds after are
    excluded to prevent future lookahead). This gives ``n_splits - 1`` usable
    train/test pairs.

    Parameters
    ----------
    n_splits  : total number of equal-length folds (must be >= 2)
    gap_hours : purge gap between the last training fold and the test fold

    Example  (n_splits=4)
    -------
    Folds:  | A | B | C | D |
    Pair 0: train=A  test=B
    Pair 1: train=A+B  test=C
    Pair 2: train=A+B+C  test=D
    """

    def __init__(self, n_splits: int = 5, gap_hours: int = 0) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        self.n_splits = n_splits
        self.gap_hours = gap_hours

    def split(self, start_date: str, end_date: str) -> List[CVSplit]:
        start = _parse_dt(start_date)
        end = _parse_dt(end_date)
        total_h = (end - start).total_seconds() / 3600
        fold_h = total_h / self.n_splits

        # Pre-compute fold boundaries
        boundaries = [
            start + timedelta(hours=i * fold_h)
            for i in range(self.n_splits + 1)
        ]
        boundaries[-1] = end  # snap to exact end

        splits: List[CVSplit] = []
        for i in range(1, self.n_splits):  # fold 0 has no training data before it
            tr_start = boundaries[0]
            tr_end = boundaries[i]
            te_start = tr_end + timedelta(hours=self.gap_hours)
            te_end = boundaries[i + 1]

            if te_start >= end:
                break

            splits.append(CVSplit(
                fold=i - 1,
                train_start=_fmt_dt(tr_start),
                train_end=_fmt_dt(tr_end),
                test_start=_fmt_dt(te_start),
                test_end=_fmt_dt(te_end),
            ))

        return splits

    def __repr__(self) -> str:
        return f"KFoldCV(n_splits={self.n_splits}, gap_hours={self.gap_hours})"


class TrainTestSplit:
    """
    Simple single train / test split.

    Parameters
    ----------
    train_ratio : fraction of the period used for training (e.g. 0.7)
    gap_hours   : purge gap between train end and test start

    This is the simplest baseline — equivalent to WalkForwardCV(n_splits=1).
    """

    def __init__(self, train_ratio: float = 0.7, gap_hours: int = 0) -> None:
        if not 0 < train_ratio < 1:
            raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
        self.train_ratio = train_ratio
        self.gap_hours = gap_hours

    def split(self, start_date: str, end_date: str) -> List[CVSplit]:
        start = _parse_dt(start_date)
        end = _parse_dt(end_date)
        total_h = (end - start).total_seconds() / 3600
        train_h = total_h * self.train_ratio

        tr_end = start + timedelta(hours=train_h)
        te_start = tr_end + timedelta(hours=self.gap_hours)

        return [
            CVSplit(
                fold=0,
                train_start=_fmt_dt(start),
                train_end=_fmt_dt(tr_end),
                test_start=_fmt_dt(te_start),
                test_end=_fmt_dt(end),
            )
        ]

    def __repr__(self) -> str:
        return f"TrainTestSplit(train_ratio={self.train_ratio}, gap_hours={self.gap_hours})"
