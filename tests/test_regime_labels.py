"""Tests for src/feature_engineering/regime_labels.py.

Unit tests use synthetic feature rows to exercise each rule branch
and the cascade ordering. Integration tests against real FRED data
live in a separate slow-marked test file (added in a follow-up
commit).
"""

import pandas as pd
import pytest

from src.feature_engineering.regime_labels import (
    CONTRACTION_NFCI_THRESHOLD,
    CONTRACTION_UNRATE_ELEVATION,
    LATE_CYCLE_NFCI_MAX,
    LATE_CYCLE_UNRATE_ABOVE_MIN_MAX,
    RECOVERY_UNRATE_ELEVATION,
    REGIMES,
    _label_row,
    compute_labels,
)


# ---------- Module constants ----------------------------------------------


def test_regimes_constant_locked_at_four_states():
    """The 4-state taxonomy is the design lock; tests guard regressions
    of either the count or the canonical ordering."""
    assert REGIMES == ["Expansion", "Late-cycle", "Contraction", "Recovery"]


def test_unemployment_thresholds_form_clean_ladder():
    """Late-cycle's 0.7pp cutoff must sit strictly below Recovery's
    1.5pp elevation. Without this ordering, the rules could overlap
    and the "ambiguous zone" between 0.7 and 1.5 collapses."""
    assert LATE_CYCLE_UNRATE_ABOVE_MIN_MAX < RECOVERY_UNRATE_ELEVATION


def test_contraction_and_recovery_share_elevation_threshold():
    """+1.5pp is the single 'elevated unemployment' boundary used by
    BOTH rules; the direction of change distinguishes which fires."""
    assert CONTRACTION_UNRATE_ELEVATION == RECOVERY_UNRATE_ELEVATION


def test_nfci_thresholds_match_across_rules():
    """The NFCICREDIT 0.5 boundary should be the same on both sides:
    > 0.5 triggers Contraction's credit-stress arm, < 0.5 is required
    for Late-cycle to fire (credit must be benign)."""
    assert CONTRACTION_NFCI_THRESHOLD == LATE_CYCLE_NFCI_MAX


# ---------- _label_row rule cascade ---------------------------------------


def _baseline_row(**overrides) -> pd.Series:
    """Build a baseline 'Expansion' feature row; tests can override fields."""
    base = {
        "unrate": 4.0,
        "unrate_3mo_change": -0.1,
        "unrate_6mo_change": -0.2,
        "unrate_12mo_min": 4.0,
        "unrate_24mo_min": 4.0,
        "unrate_above_24mo_min": 0.0,
        "t10y3m": 1.5,
        "nfcicredit": -0.1,
    }
    base.update(overrides)
    return pd.Series(base)


def test_label_row_returns_contraction_on_credit_stress_arm():
    """Unemployment spike + high NFCICREDIT -> Contraction (first arm)."""
    row = _baseline_row(
        unrate_3mo_change=0.8,
        nfcicredit=0.8,  # > 0.5
    )
    assert _label_row(row) == "Contraction"


def test_label_row_returns_contraction_on_elevated_unemployment_arm():
    """Unemployment spike + elevated level -> Contraction (second arm).

    This is the COVID May-June 2020 case: dUnr_3mo still very positive
    but NFCICREDIT had stabilized below 0.5. The OR clause keeps the
    label Contraction through the high-unemployment plateau."""
    row = _baseline_row(
        unrate=6.0,
        unrate_3mo_change=0.8,
        unrate_12mo_min=4.0,  # elevation = 2.0 > 1.5
        nfcicredit=0.2,        # NOT > 0.5
    )
    assert _label_row(row) == "Contraction"


def test_label_row_returns_recovery():
    """Elevated unrate vs 24mo min + falling 6mo trend -> Recovery."""
    row = _baseline_row(
        unrate=6.0,
        unrate_3mo_change=-0.3,
        unrate_6mo_change=-0.5,
        unrate_24mo_min=4.0,
        unrate_above_24mo_min=2.0,  # > 1.5
    )
    assert _label_row(row) == "Recovery"


def test_label_row_returns_late_cycle():
    """Flat/inverted curve + cycle-low unrate + benign credit."""
    row = _baseline_row(
        unrate=3.7,
        unrate_24mo_min=3.5,
        unrate_above_24mo_min=0.2,  # < 0.7
        t10y3m=-0.5,                 # < 0.25
        nfcicredit=-0.1,             # < 0.5
    )
    assert _label_row(row) == "Late-cycle"


def test_label_row_returns_expansion_as_default():
    """Benign feature vector -> Expansion (the cascade fall-through)."""
    row = _baseline_row()  # all defaults benign
    assert _label_row(row) == "Expansion"


# ---------- Boundary conditions -------------------------------------------


def test_late_cycle_t10y3m_boundary_is_strict_less_than():
    """T10Y3M exactly at 0.25 -> Expansion. Just below -> Late-cycle."""
    base = dict(
        unrate=3.7,
        unrate_24mo_min=3.5,
        unrate_above_24mo_min=0.2,
        nfcicredit=-0.1,
    )
    assert _label_row(_baseline_row(**base, t10y3m=0.25)) == "Expansion"
    assert _label_row(_baseline_row(**base, t10y3m=0.24)) == "Late-cycle"


def test_late_cycle_unrate_boundary_is_strict_less_than():
    """unrate_above_24mo_min exactly at 0.7 -> Expansion."""
    base = dict(
        unrate=4.2,
        unrate_24mo_min=3.5,
        t10y3m=-0.5,
        nfcicredit=-0.1,
    )
    assert (
        _label_row(_baseline_row(**base, unrate_above_24mo_min=0.7))
        == "Expansion"
    )
    assert (
        _label_row(_baseline_row(**base, unrate_above_24mo_min=0.69))
        == "Late-cycle"
    )


def test_contraction_unrate_3mo_change_boundary_is_strict_greater():
    """unrate_3mo_change exactly at 0.5 -> doesn't trigger; > 0.5 does."""
    base = dict(unrate=4.5, nfcicredit=0.8)
    # At 0.5: not Contraction.
    assert (
        _label_row(_baseline_row(**base, unrate_3mo_change=0.5))
        != "Contraction"
    )
    # Just above 0.5: triggers (credit-stress arm).
    assert (
        _label_row(_baseline_row(**base, unrate_3mo_change=0.51))
        == "Contraction"
    )


def test_recovery_6mo_change_boundary_is_strict_less_than_zero():
    """unrate_6mo_change exactly at 0.0 -> doesn't trigger Recovery."""
    base = dict(unrate=6.0, unrate_24mo_min=4.0, unrate_above_24mo_min=2.0)
    assert _label_row(_baseline_row(**base, unrate_6mo_change=0.0)) != "Recovery"
    assert _label_row(_baseline_row(**base, unrate_6mo_change=-0.01)) == "Recovery"


# ---------- Cascade ordering ----------------------------------------------


def test_cascade_contraction_beats_recovery_on_double_match():
    """A row matching both Contraction and Recovery -> Contraction.

    Possible during a sudden post-recovery shock where unemployment
    is rising hard (Contraction) but has been falling on average over
    6 months (Recovery's trend window). Contraction wins; this
    matches the design's "first match in cascade" semantics."""
    row = _baseline_row(
        unrate=6.0,
        unrate_3mo_change=0.8,        # Contraction trigger
        unrate_6mo_change=-0.2,        # Recovery trigger
        unrate_12mo_min=4.0,           # elevation = 2.0 (Contraction arm 2)
        unrate_24mo_min=4.0,
        unrate_above_24mo_min=2.0,     # Recovery elevation OK
    )
    assert _label_row(row) == "Contraction"


def test_cascade_recovery_beats_late_cycle_on_double_match():
    """A row matching Recovery and Late-cycle -> Recovery."""
    row = _baseline_row(
        unrate=6.0,
        unrate_6mo_change=-0.3,
        unrate_24mo_min=4.0,
        unrate_above_24mo_min=2.0,     # Recovery trigger; > 0.7 so Late-cycle fails anyway
        t10y3m=-0.5,                    # Late-cycle curve trigger
        nfcicredit=-0.1,
    )
    assert _label_row(row) == "Recovery"


# ---------- compute_labels public API -------------------------------------


def _synthetic_fred_matrix(n_months: int = 36) -> pd.DataFrame:
    """Build a benign synthetic FRED matrix for shape/contract tests."""
    dates = pd.date_range(start="2014-01-31", periods=n_months, freq="ME")
    return pd.DataFrame(
        {
            "UNRATE": [4.0] * n_months,
            "T10Y3M": [1.5] * n_months,
            "NFCICREDIT": [-0.1] * n_months,
        },
        index=dates,
    )


def test_compute_labels_returns_series_of_valid_regime_values():
    matrix = _synthetic_fred_matrix(36)
    labels = compute_labels(matrix)
    assert isinstance(labels, pd.Series)
    assert (labels.isin(REGIMES)).all()
    assert not labels.isna().any()


def test_compute_labels_drops_leading_rows_with_insufficient_rolling_history():
    """The first 6 rows lack a valid unrate_6mo_change; they're dropped."""
    matrix = _synthetic_fred_matrix(24)
    labels = compute_labels(matrix)
    assert len(labels) < len(matrix)
    # The retained rows are contiguous from the end (we shift, not interleave).
    assert labels.index[-1] == matrix.index[-1]


def test_compute_labels_benign_synthetic_data_yields_expansion():
    """All values benign -> every row labels Expansion."""
    matrix = _synthetic_fred_matrix(36)
    labels = compute_labels(matrix)
    assert (labels == "Expansion").all()


def test_compute_labels_raises_on_missing_required_column():
    dates = pd.date_range(start="2014-01-31", periods=12, freq="ME")
    matrix = pd.DataFrame(
        {"UNRATE": [4.0] * 12, "T10Y3M": [1.5] * 12},  # NFCICREDIT missing
        index=dates,
    )
    with pytest.raises(ValueError, match="NFCICREDIT"):
        compute_labels(matrix)


def test_compute_labels_raises_on_non_datetime_index():
    matrix = pd.DataFrame(
        {"UNRATE": [4.0], "T10Y3M": [1.5], "NFCICREDIT": [-0.1]},
        index=[0],
    )
    with pytest.raises(ValueError, match="DatetimeIndex"):
        compute_labels(matrix)
