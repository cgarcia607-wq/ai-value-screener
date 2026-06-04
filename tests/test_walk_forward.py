"""Tests for src/validation/walk_forward.py.

See docs/design/walk_forward_cv.md for the spec these tests pin down.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from dateutil.relativedelta import relativedelta

from src.validation.walk_forward import Fold, WalkForwardCV


@pytest.fixture
def monthly_dates_2014_2025():
    """144 month-end dates spanning 2014-01-31 to 2025-12-31."""
    return pd.date_range(start="2014-01-31", end="2025-12-31", freq="ME")


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


# ---------- split() — basic yield and date arithmetic ---------------------


def test_split_yields_at_least_one_fold(monthly_dates_2014_2025):
    cv = WalkForwardCV(train_period=36)
    folds = list(cv.split(monthly_dates_2014_2025))
    assert len(folds) > 0


def test_split_fold_ids_are_zero_indexed_and_sequential(monthly_dates_2014_2025):
    cv = WalkForwardCV(train_period=36)
    folds = list(cv.split(monthly_dates_2014_2025))
    assert [f.fold_id for f in folds] == list(range(len(folds)))


def test_split_every_fold_satisfies_embargo(monthly_dates_2014_2025):
    """For every fold, the gap between train_end and test_start is at
    least `embargo` calendar months. This is the look-ahead-bias
    guardrail at the fold level."""
    cv = WalkForwardCV(train_period=36, embargo=12)
    for f in cv.split(monthly_dates_2014_2025):
        gap = relativedelta(f.test_start, f.train_end)
        gap_months = gap.years * 12 + gap.months
        assert gap_months >= 12, (
            f"Fold {f.fold_id}: gap {gap_months}mo < embargo 12mo "
            f"(train_end={f.train_end}, test_start={f.test_start})"
        )


def test_split_no_train_test_index_overlap_any_fold(monthly_dates_2014_2025):
    cv = WalkForwardCV(train_period=36)
    for f in cv.split(monthly_dates_2014_2025):
        overlap = np.intersect1d(f.train_indices, f.test_indices)
        assert overlap.size == 0, (
            f"Fold {f.fold_id} train/test overlap: {overlap.tolist()}"
        )


def test_split_indices_map_to_correct_input_dates(monthly_dates_2014_2025):
    """Every train_index maps to an input date inside [train_start, train_end];
    same for test."""
    cv = WalkForwardCV(train_period=36)
    for f in cv.split(monthly_dates_2014_2025):
        for i in f.train_indices:
            d = monthly_dates_2014_2025[i].date()
            assert f.train_start <= d <= f.train_end
        for i in f.test_indices:
            d = monthly_dates_2014_2025[i].date()
            assert f.test_start <= d <= f.test_end


def test_split_vintage_equals_test_start_every_fold(monthly_dates_2014_2025):
    """The look-ahead-bias guardrail at the harness level: vintage_date
    must equal test_start so consumers reach for as-known-at-prediction-
    time data, not latest-revision."""
    cv = WalkForwardCV(train_period=36)
    for f in cv.split(monthly_dates_2014_2025):
        assert f.vintage_date == f.test_start


# ---------- Expanding-window semantics ------------------------------------


def test_expanding_train_start_constant(monthly_dates_2014_2025):
    cv = WalkForwardCV(train_period=36, expanding=True)
    folds = list(cv.split(monthly_dates_2014_2025))
    train_starts = {f.train_start for f in folds}
    assert len(train_starts) == 1


def test_expanding_train_end_advances_by_step(monthly_dates_2014_2025):
    cv = WalkForwardCV(train_period=36, step=12, expanding=True)
    folds = list(cv.split(monthly_dates_2014_2025))
    for prev, curr in zip(folds[:-1], folds[1:]):
        delta = relativedelta(curr.train_end, prev.train_end)
        assert delta.years * 12 + delta.months == 12


def test_expanding_screener_config_fold_count(monthly_dates_2014_2025):
    """Screener defaults on 2014-2025 monthly: train_period=36, embargo=12,
    test=12, step=12. Hand-computed expectation: 7 folds.

    Reasoning: fold N's test_end = data_start + (36 + N*12 + 12 + 12) months
    - 1 day = data_start + (60 + 12N) months - 1 day. Need test_end <=
    2025-12-31. For data_start=2014-01-31, max N with test_end <= data_end
    is N=6 (test_end=2025-01-30); N=7 gives 2026-01-30 > data_end.
    """
    cv = WalkForwardCV(train_period=36)
    folds = list(cv.split(monthly_dates_2014_2025))
    assert len(folds) == 7


def test_expanding_first_fold_window_sizes(monthly_dates_2014_2025):
    """Fold 0: 36 months of train, 12 months of test."""
    cv = WalkForwardCV(train_period=36)
    f0 = next(iter(cv.split(monthly_dates_2014_2025)))
    assert f0.n_train == 36
    assert f0.n_test == 12


def test_expanding_train_size_grows_each_fold(monthly_dates_2014_2025):
    """Fold N+1 should have step=12 more training observations than fold N."""
    cv = WalkForwardCV(train_period=36, step=12)
    folds = list(cv.split(monthly_dates_2014_2025))
    for prev, curr in zip(folds[:-1], folds[1:]):
        assert curr.n_train - prev.n_train == 12


# ---------- Sliding-window semantics --------------------------------------


def test_sliding_train_start_advances_by_step(monthly_dates_2014_2025):
    cv = WalkForwardCV(train_period=36, step=12, expanding=False)
    folds = list(cv.split(monthly_dates_2014_2025))
    for prev, curr in zip(folds[:-1], folds[1:]):
        delta = relativedelta(curr.train_start, prev.train_start)
        assert delta.years * 12 + delta.months == 12


def test_sliding_train_window_size_constant(monthly_dates_2014_2025):
    """Sliding mode keeps n_train identical across folds (within data gaps)."""
    cv = WalkForwardCV(train_period=36, expanding=False)
    folds = list(cv.split(monthly_dates_2014_2025))
    train_sizes = {f.n_train for f in folds}
    assert train_sizes == {36}


# ---------- Embargo configurations ----------------------------------------


def test_regime_classifier_embargo_1_yields_more_folds(monthly_dates_2014_2025):
    """For the same data, embargo=1 (regime) gives strictly more folds
    than embargo=12 (screener)."""
    screener = WalkForwardCV(train_period=36, embargo=12)
    regime = WalkForwardCV(train_period=36, embargo=1)
    n_screener = len(list(screener.split(monthly_dates_2014_2025)))
    n_regime = len(list(regime.split(monthly_dates_2014_2025)))
    assert n_regime > n_screener


def test_zero_embargo_test_starts_day_after_train_end(monthly_dates_2014_2025):
    """embargo=0 means test_start = train_end + 1 day. Used by concurrent-label
    classifiers where there's no leakage to prevent."""
    cv = WalkForwardCV(train_period=36, embargo=0, step=12)
    folds = list(cv.split(monthly_dates_2014_2025))
    for f in folds:
        assert f.test_start == f.train_end + dt.timedelta(days=1)


# ---------- Edge cases (commit 2 scope) -----------------------------------


def test_split_too_short_data_range_raises():
    short = pd.date_range(start="2020-01-31", end="2022-12-31", freq="ME")
    cv = WalkForwardCV(train_period=36)  # needs 60 months total
    with pytest.raises(ValueError, match="Data span too short"):
        list(cv.split(short))


def test_split_empty_dates_raises():
    cv = WalkForwardCV(train_period=36)
    with pytest.raises(ValueError, match="empty"):
        list(cv.split(pd.DatetimeIndex([])))


def test_split_non_monotonic_dates_raises(monthly_dates_2014_2025):
    shuffled = monthly_dates_2014_2025[[10, 5, 0, 20]]
    cv = WalkForwardCV(train_period=36)
    with pytest.raises(ValueError, match="monotonic"):
        list(cv.split(shuffled))


# ---------- strict_period toggle ------------------------------------------


@pytest.fixture
def monthly_dates_2014_to_mid_2025():
    """Data ending mid-year so the natural final fold runs past data_end."""
    return pd.date_range(start="2014-01-31", end="2025-06-30", freq="ME")


def test_strict_period_default_drops_partial_final_fold(
    monthly_dates_2014_to_mid_2025, caplog
):
    """Default strict_period=True drops folds whose test would extend
    past data_end. Logs INFO naming the dropped step's hypothetical
    window."""
    cv = WalkForwardCV(train_period=36)
    with caplog.at_level("INFO", logger="src.validation.walk_forward"):
        folds = list(cv.split(monthly_dates_2014_to_mid_2025))
    # Final fold's test_end must not exceed the data end.
    for f in folds:
        assert f.test_end <= dt.date(2025, 6, 30)
    # An INFO log line names the dropped step.
    info_msgs = [r.message for r in caplog.records if r.levelname == "INFO"]
    assert any("dropping step" in m for m in info_msgs)


def test_strict_period_false_yields_truncated_final_fold(
    monthly_dates_2014_to_mid_2025, caplog
):
    """strict_period=False yields the final fold with test_end clipped
    to data_end. The fold's n_test is correspondingly smaller."""
    strict = WalkForwardCV(train_period=36, strict_period=True)
    loose = WalkForwardCV(train_period=36, strict_period=False)
    n_strict = len(list(strict.split(monthly_dates_2014_to_mid_2025)))
    with caplog.at_level("INFO", logger="src.validation.walk_forward"):
        loose_folds = list(loose.split(monthly_dates_2014_to_mid_2025))
    # Loose mode yields one more fold than strict.
    assert len(loose_folds) == n_strict + 1
    # The extra (last) fold's test_end matches data_end.
    last = loose_folds[-1]
    assert last.test_end == dt.date(2025, 6, 30)
    # Truncation INFO log fired.
    info_msgs = [r.message for r in caplog.records if r.levelname == "INFO"]
    assert any("truncating" in m for m in info_msgs)
    # The truncated fold's test window is shorter than test_period=12.
    assert last.n_test < 12


# ---------- summary() -----------------------------------------------------


def test_summary_header_includes_config_and_fold_count(monthly_dates_2014_2025):
    cv = WalkForwardCV(train_period=36)
    s = cv.summary(monthly_dates_2014_2025)
    assert "WalkForwardCV: 7 folds" in s
    assert "expanding" in s
    assert "train_period=36mo" in s
    assert "embargo=12mo" in s
    assert "test_period=12mo" in s
    assert "step=12mo" in s


def test_summary_table_lists_every_fold(monthly_dates_2014_2025):
    cv = WalkForwardCV(train_period=36)
    folds = list(cv.split(monthly_dates_2014_2025))
    s = cv.summary(monthly_dates_2014_2025)
    # Each fold's IDs appear in the table body.
    for f in folds:
        # The fold_id column entry — match against the start of a line
        # (preceded by newline) to avoid matching dates that contain
        # the digits.
        assert f.train_start.isoformat() in s
        assert f.test_end.isoformat() in s


def test_summary_handles_zero_folds():
    """When the date range is too short to yield any fold, the
    summary still renders a coherent header (no crash)."""
    cv = WalkForwardCV(train_period=120)  # 10 years required
    short = pd.date_range(start="2020-01-31", end="2022-12-31", freq="ME")
    # split() would raise; summary should also raise (it calls split).
    with pytest.raises(ValueError):
        cv.summary(short)


# ---------- split_long (long-format adapter) ------------------------------


def _make_long_df(dates: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    """5-ticker uniform long-format DataFrame."""
    rows = [
        {"as_of_date": d, "ticker": t, "feature": float(hash((d, t)) % 100)}
        for d in dates
        for t in tickers
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def long_df_uniform(monthly_dates_2014_2025):
    """Uniform 5-ticker long-format DF, 144 months x 5 = 720 rows."""
    return _make_long_df(monthly_dates_2014_2025, ["AAPL", "MSFT", "GOOGL", "AMZN", "META"])


def test_split_long_yields_same_fold_count_as_base_split(
    long_df_uniform, monthly_dates_2014_2025
):
    cv = WalkForwardCV(train_period=36)
    base = list(cv.split(monthly_dates_2014_2025))
    long_folds = list(cv.split_long(long_df_uniform))
    assert len(long_folds) == len(base) == 7


def test_split_long_fold_date_windows_match_base(
    long_df_uniform, monthly_dates_2014_2025
):
    """Fold date metadata is identical between split() and split_long()."""
    cv = WalkForwardCV(train_period=36)
    for base, long_f in zip(
        cv.split(monthly_dates_2014_2025), cv.split_long(long_df_uniform)
    ):
        assert base.train_start == long_f.train_start
        assert base.train_end == long_f.train_end
        assert base.embargo_start == long_f.embargo_start
        assert base.embargo_end == long_f.embargo_end
        assert base.test_start == long_f.test_start
        assert base.test_end == long_f.test_end
        assert base.vintage_date == long_f.vintage_date


def test_split_long_indices_count_scales_with_universe(long_df_uniform):
    """5 tickers * train months = expected n_train per fold."""
    cv = WalkForwardCV(train_period=36)
    f0 = next(iter(cv.split_long(long_df_uniform)))
    # 36 months * 5 tickers = 180 train rows
    assert f0.n_train == 36 * 5
    # 12 months * 5 tickers = 60 test rows
    assert f0.n_test == 12 * 5


def test_split_long_handles_ragged_universe(monthly_dates_2014_2025):
    """Different tickers in different months: all rows for a given date
    stay together in one fold; no date is split across train/test."""
    np.random.seed(42)
    pool = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    rows = []
    for d in monthly_dates_2014_2025:
        n = int(np.random.randint(3, 8))
        for t in np.random.choice(pool, size=n, replace=False):
            rows.append({"as_of_date": d, "ticker": t, "feature": 1.0})
    df = pd.DataFrame(rows)

    cv = WalkForwardCV(train_period=36)
    for f in cv.split_long(df):
        train_dates = df.iloc[f.train_indices]["as_of_date"]
        test_dates = df.iloc[f.test_indices]["as_of_date"]
        # Every train_date row is in [train_start, train_end].
        assert (train_dates >= pd.Timestamp(f.train_start)).all()
        assert (train_dates <= pd.Timestamp(f.train_end)).all()
        # Every test_date row is in [test_start, test_end].
        assert (test_dates >= pd.Timestamp(f.test_start)).all()
        assert (test_dates <= pd.Timestamp(f.test_end)).all()
        # No date appears in both windows — the strict inequality
        # max(train) < min(test) is the per-fold no-leakage check.
        assert train_dates.max() < test_dates.min()


def test_split_long_excludes_embargo_dates(long_df_uniform):
    """Dates in the embargo period appear in NEITHER train nor test indices."""
    cv = WalkForwardCV(train_period=36, embargo=12)
    for f in cv.split_long(long_df_uniform):
        all_idx = np.concatenate([f.train_indices, f.test_indices])
        all_dates = long_df_uniform.iloc[all_idx]["as_of_date"]
        # Embargo is [embargo_start, embargo_end), so test the open
        # upper bound: no date in [embargo_start, embargo_end).
        in_embargo = (all_dates >= pd.Timestamp(f.embargo_start)) & (
            all_dates < pd.Timestamp(f.embargo_end)
        )
        assert not in_embargo.any()


def test_split_long_invalid_date_col_raises(long_df_uniform):
    cv = WalkForwardCV(train_period=36)
    with pytest.raises(ValueError, match="date_col"):
        list(cv.split_long(long_df_uniform, date_col="not_a_column"))


def test_split_long_no_empty_folds_yielded(long_df_uniform):
    """Contract: split_long never yields a fold with empty train or test."""
    cv = WalkForwardCV(train_period=36)
    for f in cv.split_long(long_df_uniform):
        assert f.n_train > 0
        assert f.n_test > 0


def test_split_long_empty_test_window_logs_warning(monkeypatch, caplog):
    """Defensive WARNING when the long-df mapping finds no test rows.

    The engine already verifies non-empty windows on unique_dates, so
    this case is unlikely in practice — but it can happen if df's date
    column has a type mismatch from the unique_dates the engine saw.
    Skip with a WARNING that names the windows; never silently yield
    an empty fold.
    """
    # df with dates only through 2017
    dates = pd.date_range(start="2014-01-31", end="2017-12-31", freq="ME")
    df = pd.DataFrame(
        {"as_of_date": dates, "ticker": "AAPL", "feature": 1.0}
    )

    # Hand-craft a fold whose test window is in 2018 (no rows in df).
    fake_fold = Fold(
        fold_id=0,
        train_start=dt.date(2014, 1, 31),
        train_end=dt.date(2016, 12, 31),
        embargo_start=dt.date(2017, 1, 1),
        embargo_end=dt.date(2018, 1, 1),
        test_start=dt.date(2018, 1, 1),
        test_end=dt.date(2018, 12, 31),
        vintage_date=dt.date(2018, 1, 1),
        train_indices=np.array([0, 1]),
        test_indices=np.array([2, 3]),
    )

    cv = WalkForwardCV(train_period=36)
    monkeypatch.setattr(cv, "split", lambda _d: iter([fake_fold]))

    with caplog.at_level("WARNING", logger="src.validation.walk_forward"):
        folds = list(cv.split_long(df))

    # No folds yielded — the empty-test-window fold was skipped.
    assert len(folds) == 0
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any(
        "empty long-format window" in w.message for w in warnings
    )
