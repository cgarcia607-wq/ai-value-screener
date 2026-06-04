"""Walk-forward / expanding-window cross-validation with embargo.

See docs/design/walk_forward_cv.md for the full design — embargo
methodology, vintage handling, and the rationale for the per-fold
vs per-row vintage simplification.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    """A single walk-forward fold.

    All date fields are inclusive except embargo_end, which equals
    test_start (the embargo period is half-open: [embargo_start,
    embargo_end), equivalent to (train_end, test_start)).

    vintage_date is the prediction-time anchor for vintage-aware data
    fetches and equals test_start by design. See docs/design/walk_forward_cv.md.

    train_indices and test_indices are integer positions in the input
    DatetimeIndex / DataFrame the consumer passed to split(). The
    consumer is responsible for slicing its own input by these
    indices.
    """

    fold_id: int
    train_start: dt.date
    train_end: dt.date
    embargo_start: dt.date
    embargo_end: dt.date
    test_start: dt.date
    test_end: dt.date
    vintage_date: dt.date
    train_indices: np.ndarray = field(repr=False)
    test_indices: np.ndarray = field(repr=False)

    @property
    def n_train(self) -> int:
        return len(self.train_indices)

    @property
    def n_test(self) -> int:
        return len(self.test_indices)

    def __post_init__(self) -> None:
        # Date ordering — the documented monotonic chain.
        if not (
            self.train_start
            <= self.train_end
            < self.embargo_start
            <= self.embargo_end
            <= self.test_start
            <= self.test_end
        ):
            raise ValueError(
                f"Fold {self.fold_id}: date ordering violated. Required "
                f"train_start <= train_end < embargo_start <= embargo_end "
                f"<= test_start <= test_end, got "
                f"train=[{self.train_start}, {self.train_end}], "
                f"embargo=[{self.embargo_start}, {self.embargo_end}), "
                f"test=[{self.test_start}, {self.test_end}]."
            )
        # Vintage = test_start by design.
        if self.vintage_date != self.test_start:
            raise ValueError(
                f"Fold {self.fold_id}: vintage_date ({self.vintage_date}) "
                f"!= test_start ({self.test_start}). The harness sets "
                f"vintage_date = test_start; if you're seeing this you "
                f"likely constructed a Fold by hand instead of via "
                f"WalkForwardCV.split()."
            )
        # embargo_end == test_start (the embargo abuts the test window).
        if self.embargo_end != self.test_start:
            raise ValueError(
                f"Fold {self.fold_id}: embargo_end ({self.embargo_end}) "
                f"!= test_start ({self.test_start}). The embargo half-open "
                f"convention requires these to match."
            )
        # Index non-emptiness — empty folds are dropped by split(),
        # so any Fold that exists must have both windows populated.
        if self.n_train == 0:
            raise ValueError(f"Fold {self.fold_id}: train_indices is empty.")
        if self.n_test == 0:
            raise ValueError(f"Fold {self.fold_id}: test_indices is empty.")
        # No row-index overlap.
        overlap = np.intersect1d(self.train_indices, self.test_indices)
        if overlap.size > 0:
            raise ValueError(
                f"Fold {self.fold_id}: train and test indices overlap on "
                f"{overlap.size} positions (e.g., {overlap[:3].tolist()}). "
                f"This indicates the input DatetimeIndex passed to split() "
                f"had duplicate dates that fell on opposite sides of the "
                f"embargo, or the splitter is broken."
            )


class WalkForwardCV:
    """Walk-forward / expanding-window CV with embargo.

    See docs/design/walk_forward_cv.md for the full design.

    Args:
        train_period: minimum training window length in months. For
            expanding=True (default), this is the floor; the train
            window grows by `step` months each fold. For
            expanding=False, this is the fixed sliding-window length.
        test_period: test window length in months. Default 12 —
            matches the screener's 12-month forward-return label.
        embargo: gap between train_end and test_start, in months.
            Must be >= the label horizon. Default 12 for the
            screener; set to 1 for the regime classifier (concurrent
            labels). See the design doc for the look-ahead-bias
            argument.
        step: months to advance per fold. Default 12 —
            non-overlapping test sets.
        expanding: True (default) for expanding window; False for
            sliding window of fixed length `train_period`.
        strict_period: True (default) drops the final fold if its
            test window would extend past the data end. False yields
            the final fold with a truncated test window; aggregation
            metrics across folds become non-comparable in that case.

    A construction-time WARNING is logged when step < test_period
    (overlapping test sets produce correlated fold scores; see the
    design doc).
    """

    def __init__(
        self,
        train_period: int,
        test_period: int = 12,
        embargo: int = 12,
        step: int = 12,
        expanding: bool = True,
        strict_period: bool = True,
    ):
        if train_period < 1:
            raise ValueError(f"train_period must be >= 1, got {train_period}")
        if test_period < 1:
            raise ValueError(f"test_period must be >= 1, got {test_period}")
        if embargo < 0:
            raise ValueError(f"embargo must be >= 0, got {embargo}")
        if step < 1:
            raise ValueError(f"step must be >= 1, got {step}")
        if embargo > step:
            raise ValueError(
                f"embargo ({embargo}) > step ({step}): consecutive folds "
                f"would overlap in time, violating walk-forward semantics. "
                f"Either reduce embargo or increase step."
            )

        if step < test_period:
            logger.warning(
                "WalkForwardCV configured with step=%d < test_period=%d. "
                "Consecutive folds will have overlapping test sets — fold "
                "scores are NOT independent samples of model performance. "
                "Aggregation methods that assume independence (simple "
                "averaging, t-tests on fold scores, naive bootstrap) will "
                "produce misleading results. Use block-bootstrap or "
                "fold-correlation-aware aggregation. See "
                "docs/design/walk_forward_cv.md.",
                step,
                test_period,
            )

        self.train_period = train_period
        self.test_period = test_period
        self.embargo = embargo
        self.step = step
        self.expanding = expanding
        self.strict_period = strict_period

    def split(self, dates: pd.DatetimeIndex) -> Iterator[Fold]:
        """Yield Fold objects with full date metadata and row indices.

        Not yet implemented — phase 1 commit 2 lands this.
        """
        raise NotImplementedError("split() lands in the next commit")

    def summary(self, dates: pd.DatetimeIndex) -> str:
        """Return an ASCII table of the fold structure.

        Not yet implemented — phase 1 commit 3 lands this.
        """
        raise NotImplementedError("summary() lands in commit 3")
