"""Unit tests for src/feature_engineering/macro_features.py.

Fast/synthetic only — no FRED API. Covers build_v2_features column
inventory, delta correctness, z-score trailing-only constraint,
one-hot regime agreement, and v1 walk-forward backward-compatibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.macro_features import build_v2_features
from src.feature_engineering.regime_labels import REGIMES, compute_labels
from src.models.regime_classifier import RegimeClassifier, run_walk_forward
from src.validation.walk_forward import WalkForwardCV


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_v2_matrix(n_months: int = 40) -> pd.DataFrame:
    """Synthetic FRED-like matrix with every column required by build_v2_features."""
    idx = pd.date_range("2018-01-31", periods=n_months, freq="ME")
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "UNRATE": 4.0 + rng.normal(0, 0.1, n_months),
            "T10Y3M": 1.0 + rng.normal(0, 0.3, n_months),
            "BAA10Y": 2.0 + rng.normal(0, 0.1, n_months),
            "FEDFUNDS": 2.0 + rng.normal(0, 0.2, n_months),
            "INDPRO": 100.0 + rng.normal(0, 1.0, n_months),
            "PAYEMS": 150_000.0 + rng.normal(0, 100.0, n_months),
            "ICSA": 200_000.0 + rng.normal(0, 5_000.0, n_months),
            "VIXCLS": 15.0 + rng.normal(0, 2.0, n_months),
        },
        index=idx,
    )


def _make_regime_matrix(n_months: int = 72, switch_at: int = 24) -> pd.DataFrame:
    """Minimal FRED-like matrix suitable for run_walk_forward (compute_labels compatible)."""
    idx = pd.date_range("2015-01-31", periods=n_months, freq="ME")
    rng = np.random.default_rng(0)
    t10y3m = np.concatenate(
        [
            np.full(switch_at, 1.0) + rng.normal(0, 0.03, switch_at),
            np.full(n_months - switch_at, -0.3)
            + rng.normal(0, 0.03, n_months - switch_at),
        ]
    )
    return pd.DataFrame(
        {
            "UNRATE": 4.0 + rng.normal(0, 0.03, n_months),
            "T10Y3M": t10y3m,
            "BAA10Y": 2.0 + rng.normal(0, 0.05, n_months),
            "DGS10": 2.5 + rng.normal(0, 0.1, n_months),
            "FEDFUNDS": 1.5 + rng.normal(0, 0.1, n_months),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_v2_features_adds_expected_columns():
    """Output contains all original columns plus exactly the 17 specified new ones."""
    matrix = _make_v2_matrix(n_months=40)
    out = build_v2_features(matrix)

    original_cols = set(matrix.columns)
    new_cols = set(out.columns) - original_cols

    expected_new: set[str] = set()
    for col in ("UNRATE", "FEDFUNDS", "INDPRO", "PAYEMS", "ICSA"):
        expected_new.add(f"{col}_chg_3mo")
        expected_new.add(f"{col}_chg_12mo")
    for col in ("T10Y3M", "BAA10Y", "VIXCLS"):
        expected_new.add(f"{col}_z12")
    for regime in REGIMES:
        expected_new.add(f"regime_now_{regime}")

    assert new_cols == expected_new, (
        f"Extra columns: {new_cols - expected_new}\n"
        f"Missing columns: {expected_new - new_cols}"
    )
    assert len(new_cols) == 17


def test_v2_deltas_are_correct():
    """On a clean ramp UNRATE series chg_3mo[i] == x[i] - x[i-3] exactly."""
    n = 20
    matrix = _make_v2_matrix(n_months=n)
    matrix = matrix.copy()
    matrix["UNRATE"] = np.arange(n, dtype=float)

    out = build_v2_features(matrix)

    # Leading rows before the window fills must be NaN.
    assert out["UNRATE_chg_3mo"].iloc[:3].isna().all()
    assert out["UNRATE_chg_12mo"].iloc[:12].isna().all()

    # Every valid row: chg_3mo == 3.0, chg_12mo == 12.0 (constant ramp).
    for i in range(3, n):
        assert out["UNRATE_chg_3mo"].iloc[i] == pytest.approx(float(i) - float(i - 3))
    for i in range(12, n):
        assert out["UNRATE_chg_12mo"].iloc[i] == pytest.approx(float(i) - float(i - 12))


def test_v2_zscore_uses_trailing_window_only():
    """z-score at time t is identical regardless of what happens at t+k > t."""
    n = 30
    matrix = _make_v2_matrix(n_months=n)

    # Reference point after the 12-row warm-up.
    t_idx = 20
    t = matrix.index[t_idx]

    features1 = build_v2_features(matrix)

    # Drastic level-shift on all future rows.
    matrix2 = matrix.copy()
    future = matrix2.index[t_idx + 1 :]
    for col in ("T10Y3M", "BAA10Y", "VIXCLS"):
        matrix2.loc[future, col] = 9_999.0
    features2 = build_v2_features(matrix2)

    for col in ("T10Y3M_z12", "BAA10Y_z12", "VIXCLS_z12"):
        assert features1.loc[t, col] == pytest.approx(features2.loc[t, col]), (
            f"{col} at t={t.date()} changed when future rows were modified — "
            "rolling window is not purely trailing"
        )


def test_v2_regime_onehot_matches_compute_labels():
    """regime_now_{REGIME} columns agree with compute_labels row-by-row,
    and exactly one column equals 1 for every labeled row."""
    matrix = _make_v2_matrix(n_months=40)
    out = build_v2_features(matrix)
    labels = compute_labels(matrix)

    regime_cols = [f"regime_now_{r}" for r in REGIMES]

    for date, label in labels.items():
        for regime in REGIMES:
            col = f"regime_now_{regime}"
            expected = 1.0 if label == regime else 0.0
            assert out.loc[date, col] == pytest.approx(expected), (
                f"Mismatch at {date.date()}: {col}={out.loc[date, col]}, "
                f"expected {expected} (true label={label!r})"
            )

    # Exactly one hot per labeled row.
    labeled = out.loc[labels.index, regime_cols]
    row_sums = labeled.sum(axis=1)
    assert np.allclose(row_sums.to_numpy(), 1.0), (
        f"Rows not summing to 1: {row_sums[~np.isclose(row_sums, 1.0)]}"
    )


def test_run_walk_forward_default_is_unchanged(tmp_path, monkeypatch):
    """feature_builder=None produces identical fold prep to omitting the argument
    entirely — v1 results must remain reproducible."""
    matrix = _make_regime_matrix(n_months=72, switch_at=24)
    cv = WalkForwardCV(train_period=36, test_period=12, embargo=3, step=12)

    captured_none: list[pd.DataFrame] = []
    captured_omit: list[pd.DataFrame] = []
    original_fit = RegimeClassifier.fit

    def _capture_none(self, X: pd.DataFrame, y: pd.Series) -> RegimeClassifier:
        captured_none.append(X.copy())
        return original_fit(self, X, y)

    def _capture_omit(self, X: pd.DataFrame, y: pd.Series) -> RegimeClassifier:
        captured_omit.append(X.copy())
        return original_fit(self, X, y)

    monkeypatch.setattr(RegimeClassifier, "fit", _capture_none)
    run_walk_forward(cv=cv, features=matrix, output_dir=tmp_path / "none", feature_builder=None)

    monkeypatch.setattr(RegimeClassifier, "fit", _capture_omit)
    run_walk_forward(cv=cv, features=matrix, output_dir=tmp_path / "omit")

    assert len(captured_none) > 0, "RegimeClassifier.fit was never called"
    assert len(captured_none) == len(captured_omit), (
        f"Call count differs: feature_builder=None → {len(captured_none)}, "
        f"default → {len(captured_omit)}"
    )
    for i, (a, b) in enumerate(zip(captured_none, captured_omit)):
        pd.testing.assert_frame_equal(
            a, b, check_exact=True, obj=f"X_train at RegimeClassifier.fit call #{i}"
        )
