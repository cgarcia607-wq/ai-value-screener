"""Tests for src/validation/walk_forward.py.

See docs/design/walk_forward_cv.md for the spec these tests pin down.
"""

import datetime as dt

import numpy as np
import pytest

from src.validation.walk_forward import Fold, WalkForwardCV


# ---------- Fold dataclass invariants -------------------------------------


def _ok_fold_kwargs(**overrides) -> dict:
    """Default kwargs for a known-good Fold; tests can override fields."""
    base = dict(
        fold_id=0,
        train_start=dt.date(2014, 1, 1),
        train_end=dt.date(2016, 12, 31),
        embargo_start=dt.date(2017, 1, 1),
        embargo_end=dt.date(2018, 1, 1),
        test_start=dt.date(2018, 1, 1),
        test_end=dt.date(2018, 12, 31),
        vintage_date=dt.date(2018, 1, 1),
        train_indices=np.array([0, 1, 2, 3]),
        test_indices=np.array([4, 5]),
    )
    base.update(overrides)
    return base


def test_fold_constructs_with_valid_inputs():
    fold = Fold(**_ok_fold_kwargs())
    assert fold.fold_id == 0
    assert fold.n_train == 4
    assert fold.n_test == 2


def test_fold_vintage_must_equal_test_start():
    with pytest.raises(ValueError, match="vintage_date"):
        Fold(**_ok_fold_kwargs(vintage_date=dt.date(2017, 6, 1)))


def test_fold_date_ordering_rejected_when_train_start_after_train_end():
    """train_start must come before train_end."""
    with pytest.raises(ValueError, match="date ordering"):
        Fold(
            **_ok_fold_kwargs(
                # train_start after the default train_end
                train_start=dt.date(2017, 6, 1),
            )
        )


def test_fold_date_ordering_rejected_when_test_start_after_test_end():
    """test_start must come before test_end."""
    with pytest.raises(ValueError, match="date ordering"):
        Fold(
            **_ok_fold_kwargs(
                # test_end before the default test_start
                test_end=dt.date(2017, 6, 1),
            )
        )


def test_fold_date_ordering_rejected_when_train_end_after_embargo_start():
    with pytest.raises(ValueError, match="date ordering"):
        Fold(
            **_ok_fold_kwargs(
                train_end=dt.date(2017, 6, 1),
                embargo_start=dt.date(2017, 1, 1),
            )
        )


def test_fold_empty_train_rejected():
    with pytest.raises(ValueError, match="train_indices is empty"):
        Fold(**_ok_fold_kwargs(train_indices=np.array([], dtype=int)))


def test_fold_empty_test_rejected():
    with pytest.raises(ValueError, match="test_indices is empty"):
        Fold(**_ok_fold_kwargs(test_indices=np.array([], dtype=int)))


def test_fold_overlapping_indices_rejected():
    """train and test indices must be disjoint — the embargo guarantees
    they reference different time periods, so any overlap indicates
    duplicate input dates on opposite sides of the embargo or a broken
    splitter."""
    with pytest.raises(ValueError, match="overlap"):
        Fold(
            **_ok_fold_kwargs(
                train_indices=np.array([0, 1, 2, 4]),
                test_indices=np.array([4, 5]),
            )
        )


def test_fold_embargo_end_must_equal_test_start():
    with pytest.raises(ValueError, match="embargo_end"):
        Fold(
            **_ok_fold_kwargs(
                # embargo_end leaves a gap before test_start
                embargo_end=dt.date(2017, 12, 1),
                # test_start stays 2018-01-01
            )
        )


def test_fold_is_frozen():
    fold = Fold(**_ok_fold_kwargs())
    with pytest.raises((AttributeError, Exception)):
        fold.fold_id = 99  # type: ignore[misc]


# ---------- WalkForwardCV constructor -------------------------------------


def test_constructor_accepts_documented_defaults():
    """Default args are the screener defaults: 12mo test, 12mo embargo,
    12mo step, expanding, strict."""
    cv = WalkForwardCV(train_period=36)
    assert cv.train_period == 36
    assert cv.test_period == 12
    assert cv.embargo == 12
    assert cv.step == 12
    assert cv.expanding is True
    assert cv.strict_period is True


def test_constructor_regime_classifier_config():
    """Regime classifier uses embargo=1 (concurrent labels)."""
    cv = WalkForwardCV(train_period=60, embargo=1)
    assert cv.embargo == 1


def test_constructor_rejects_invalid_train_period():
    with pytest.raises(ValueError, match="train_period"):
        WalkForwardCV(train_period=0)
    with pytest.raises(ValueError, match="train_period"):
        WalkForwardCV(train_period=-5)


def test_constructor_rejects_invalid_test_period():
    with pytest.raises(ValueError, match="test_period"):
        WalkForwardCV(train_period=36, test_period=0)


def test_constructor_rejects_negative_embargo():
    with pytest.raises(ValueError, match="embargo"):
        WalkForwardCV(train_period=36, embargo=-1)


def test_constructor_accepts_zero_embargo():
    """embargo=0 is valid (no embargo) — used when the label is fully
    concurrent and there's no leakage risk."""
    cv = WalkForwardCV(train_period=36, embargo=0)
    assert cv.embargo == 0


def test_constructor_rejects_invalid_step():
    with pytest.raises(ValueError, match="step"):
        WalkForwardCV(train_period=36, step=0)


def test_constructor_rejects_embargo_greater_than_step():
    """embargo > step would make consecutive folds overlap in time,
    violating walk-forward semantics."""
    with pytest.raises(ValueError, match=r"embargo.*>.*step"):
        WalkForwardCV(train_period=36, embargo=12, step=6)


def test_constructor_warns_on_step_less_than_test_period(caplog):
    """Overlapping test sets produce correlated fold scores; the
    constructor should warn but still succeed."""
    with caplog.at_level("WARNING", logger="src.validation.walk_forward"):
        cv = WalkForwardCV(train_period=36, test_period=12, step=6, embargo=6)
    assert cv.step == 6
    warnings = [
        r for r in caplog.records
        if r.levelname == "WARNING" and "step" in r.message
    ]
    assert len(warnings) == 1
    msg = warnings[0].message
    assert "overlapping test sets" in msg.lower()
    assert "step=6" in msg
    assert "test_period=12" in msg
    # Names at least one of the aggregation gotchas explicitly.
    assert (
        "averaging" in msg.lower()
        or "t-test" in msg.lower()
        or "bootstrap" in msg.lower()
    )


def test_constructor_no_warning_when_step_equals_test_period(caplog):
    """The documented defaults (step=test_period=12) must not warn."""
    with caplog.at_level("WARNING", logger="src.validation.walk_forward"):
        WalkForwardCV(train_period=36)
    warnings = [
        r for r in caplog.records
        if r.levelname == "WARNING" and "overlapping" in r.message.lower()
    ]
    assert len(warnings) == 0


def test_constructor_no_warning_when_step_greater_than_test_period(caplog):
    """step > test_period leaves gaps between test sets — also fine, no warning."""
    with caplog.at_level("WARNING", logger="src.validation.walk_forward"):
        WalkForwardCV(train_period=36, test_period=12, step=18, embargo=12)
    warnings = [
        r for r in caplog.records
        if r.levelname == "WARNING" and "overlapping" in r.message.lower()
    ]
    assert len(warnings) == 0
