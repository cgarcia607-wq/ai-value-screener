"""Unit tests for src/models/regime_classifier.py.

Fast tests only — synthetic data, no FRED API. Covers both the
phase-1 RegimeClassifier wrapper and the phase-2 walk-forward
orchestrator (T+3 target construction, fold-level NaN filtering,
artifact layout, adoption-decision logic).
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.regime_labels import REGIMES, compute_labels
from src.models.regime_classifier import (
    LABEL_HORIZON_MONTHS,
    RegimeClassifier,
    _apply_adoption_decision,
    _build_t_plus_3_targets,
    run_walk_forward,
)
from src.validation.walk_forward import WalkForwardCV


def _make_synthetic_dataset(
    n_per_class: int = 40, n_features: int = 6, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Balanced 4-class blob data with one informative dimension per class.

    Balanced because CalibratedClassifierCV(cv=3) needs every class
    present in every inner fold; with 40 per class StratifiedKFold
    gives ~13 per (class, fold) which is enough for isotonic fit.
    """
    rng = np.random.default_rng(seed)
    blocks = []
    labels: list[str] = []
    for k, regime in enumerate(REGIMES):
        center = np.zeros(n_features)
        center[k % n_features] = 2.5
        blocks.append(rng.normal(loc=center, scale=1.0, size=(n_per_class, n_features)))
        labels.extend([regime] * n_per_class)
    X = pd.DataFrame(
        np.vstack(blocks),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series(labels, name="regime")
    perm = rng.permutation(len(X))
    X = X.iloc[perm].reset_index(drop=True)
    y = y.iloc[perm].reset_index(drop=True)
    return X, y


@pytest.fixture
def synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    return _make_synthetic_dataset()


def test_classifier_fit_predict_returns_valid_labels(synthetic_data):
    X, y = synthetic_data
    clf = RegimeClassifier(model_type="logistic").fit(X, y)
    preds = clf.predict(X)

    assert isinstance(preds, pd.Series)
    assert len(preds) == len(X)
    assert not preds.isna().any()
    assert set(preds.unique()) <= set(REGIMES)


def test_classifier_predict_proba_sums_to_one(synthetic_data):
    X, y = synthetic_data
    clf = RegimeClassifier(model_type="logistic").fit(X, y)
    proba = clf.predict_proba(X)

    row_sums = proba.sum(axis=1).to_numpy()
    np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-6)


def test_classifier_predict_proba_columns_are_REGIMES_in_order(synthetic_data):
    X, y = synthetic_data
    clf = RegimeClassifier(model_type="logistic").fit(X, y)
    proba = clf.predict_proba(X)

    assert list(proba.columns) == list(REGIMES)


def test_classifier_save_load_roundtrip(synthetic_data, tmp_path):
    X, y = synthetic_data
    clf = RegimeClassifier(model_type="logistic").fit(X, y)
    proba_before = clf.predict_proba(X)

    path = tmp_path / "nested" / "regime_clf.joblib"
    clf.save(path)
    assert path.exists()

    loaded = RegimeClassifier.load(path)
    proba_after = loaded.predict_proba(X)

    pd.testing.assert_frame_equal(proba_before, proba_after)


def test_logistic_vs_xgboost_models_accept_same_inputs(synthetic_data):
    X, y = synthetic_data
    for model_type in ("logistic", "xgboost"):
        clf = RegimeClassifier(model_type=model_type).fit(X, y)
        preds = clf.predict(X)
        proba = clf.predict_proba(X)
        assert len(preds) == len(X)
        assert proba.shape == (len(X), len(REGIMES))
        assert list(proba.columns) == list(REGIMES)


# ---------------------------------------------------------------------------
# Phase 2: walk-forward orchestrator
# ---------------------------------------------------------------------------


def _make_synthetic_regime_features(
    n_months: int = 72, switch_at: int = 24, seed: int = 0
) -> pd.DataFrame:
    """Construct synthetic FRED-like monthly features that produce a
    deterministic mix of Expansion (first segment) and Late-cycle
    (second segment) regimes via compute_labels.

    Stable, low unemployment keeps the Contraction and Recovery rules
    from firing; T10Y3M flips from steeply positive to inverted at
    `switch_at` so the labeling rules transition from Expansion to
    Late-cycle deterministically.
    """
    idx = pd.date_range("2015-01-31", periods=n_months, freq="ME")
    rng = np.random.default_rng(seed)

    unrate = 4.0 + rng.normal(0, 0.03, n_months)
    t10y3m = np.concatenate(
        [
            np.full(switch_at, 1.0) + rng.normal(0, 0.03, switch_at),
            np.full(n_months - switch_at, -0.3)
            + rng.normal(0, 0.03, n_months - switch_at),
        ]
    )
    nfcicredit = -0.2 + rng.normal(0, 0.03, n_months)

    return pd.DataFrame(
        {
            "UNRATE": unrate,
            "T10Y3M": t10y3m,
            "NFCICREDIT": nfcicredit,
            # BAA10Y ≈ 2.0 — well below the 3.0 Contraction/Late-cycle threshold
            # so the synthetic series stays in Expansion/Late-cycle territory.
            "BAA10Y": 2.0 + rng.normal(0, 0.05, n_months),
            "DGS10": 2.5 + rng.normal(0, 0.1, n_months),
            "FEDFUNDS": 1.5 + rng.normal(0, 0.1, n_months),
        },
        index=idx,
    )


def test_walk_forward_shifts_labels_by_3_months():
    """For every date d in labels_t.index whose +3 sibling is also
    present, the T+3 target at d must equal labels_t at d+3 — the
    classifier's training pair is (features[t], labels[t+3]), never
    (features[t], labels[t])."""
    features = _make_synthetic_regime_features(n_months=36, switch_at=18)
    labels_t = compute_labels(features)
    targets = _build_t_plus_3_targets(features)

    # The synthetic switch_at=18 + 3-month shift puts at least one
    # Expansion target and one Late-cycle target in the comparison
    # window, so this isn't a trivially-satisfied assertion.
    assert set(labels_t.unique()) == {"Expansion", "Late-cycle"}

    label_index = labels_t.index
    for i in range(len(label_index) - LABEL_HORIZON_MONTHS):
        date_t = label_index[i]
        date_t_plus_3 = label_index[i + LABEL_HORIZON_MONTHS]
        assert targets.loc[date_t] == labels_t.loc[date_t_plus_3], (
            f"Targets at {date_t.date()} should equal labels_t at "
            f"{date_t_plus_3.date()}; got "
            f"{targets.loc[date_t]!r} vs {labels_t.loc[date_t_plus_3]!r}"
        )


def test_walk_forward_drops_trailing_3_rows():
    """The last LABEL_HORIZON_MONTHS rows of features have NaN T+3
    targets by shift construction, and the orchestrator's NaN filter
    must keep them out of every fold's train and test slice."""
    features = _make_synthetic_regime_features(n_months=51, switch_at=20)
    targets = _build_t_plus_3_targets(features)

    n = len(features)
    trailing_positions = {n - 3, n - 2, n - 1}

    # The trailing-3 NaN is the load-bearing claim of the test.
    assert targets.iloc[-LABEL_HORIZON_MONTHS:].isna().all()
    # The row immediately before the trailing block has a valid label —
    # we are not accidentally dropping more than the spec says.
    assert not pd.isna(targets.iloc[n - LABEL_HORIZON_MONTHS - 1])

    cv = WalkForwardCV(
        train_period=24, test_period=12, embargo=3, step=12
    )
    folds = list(cv.split(features.index))
    assert len(folds) >= 1

    # With 51 months and this config, fold 1's test ends at the data
    # end, so the trailing-3 positions ARE in test_indices pre-filter.
    # That's what makes the test meaningful — the filter has to remove
    # them, not just rely on them being out of range.
    last_fold = folds[-1]
    assert trailing_positions.issubset(set(last_fold.test_indices.tolist())), (
        "Test precondition: trailing positions must be in the last "
        "fold's pre-filter test_indices, otherwise the assertion below "
        "passes vacuously."
    )

    for fold in folds:
        train_mask = targets.iloc[fold.train_indices].notna().to_numpy()
        test_mask = targets.iloc[fold.test_indices].notna().to_numpy()
        train_valid = set(fold.train_indices[train_mask].tolist())
        test_valid = set(fold.test_indices[test_mask].tolist())
        assert trailing_positions.isdisjoint(train_valid), (
            f"Fold {fold.fold_id}: trailing positions leaked into "
            f"train_valid"
        )
        assert trailing_positions.isdisjoint(test_valid), (
            f"Fold {fold.fold_id}: trailing positions leaked into "
            f"test_valid"
        )


def test_run_walk_forward_writes_expected_files(tmp_path):
    features = _make_synthetic_regime_features(n_months=72, switch_at=24)
    cv = WalkForwardCV(
        train_period=36, test_period=12, embargo=3, step=12
    )

    decision = run_walk_forward(
        cv=cv, features=features, output_dir=tmp_path
    )

    run_subdirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(run_subdirs) == 1, (
        f"Expected exactly one timestamped run dir under tmp_path, "
        f"got {[p.name for p in run_subdirs]}"
    )
    run_dir = run_subdirs[0]

    for fname in (
        "fold_0_lr.joblib",
        "fold_0_xgb.joblib",
        "fold_0_metrics.json",
        "aggregated_metrics.json",
        "adoption_decision.json",
    ):
        assert (run_dir / fname).exists(), f"Missing artifact: {fname}"

    assert decision["winner"] in ("logistic", "xgboost")
    assert "lr_macro_f1" in decision
    assert "xgb_macro_f1" in decision
    assert "reason" in decision


def test_adoption_decision_prefers_lr_on_tie():
    """Lift inside ±0.02 → LR wins the tiebreaker for interpretability."""
    lr = [0.50, 0.55, 0.52]
    xgb = [0.51, 0.56, 0.52]  # avg lift ≈ 0.0067, inside ±0.02
    decision = _apply_adoption_decision(lr, xgb)

    assert decision["winner"] == "logistic"
    assert abs(decision["xgb_macro_f1"] - decision["lr_macro_f1"]) < 0.02


def test_adoption_decision_picks_xgboost_on_clear_win():
    """+0.05pp macro-F1 lift exactly hits the adoption bar → XGB wins."""
    lr = [0.50, 0.50, 0.50]
    xgb = [0.55, 0.55, 0.55]  # avg lift == 0.05, no catastrophic fold
    decision = _apply_adoption_decision(lr, xgb)

    assert decision["winner"] == "xgboost"
    assert decision["xgb_macro_f1"] - decision["lr_macro_f1"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# NaN imputation in run_walk_forward
# ---------------------------------------------------------------------------


def test_walk_forward_imputes_nan_with_train_fold_mean(tmp_path, monkeypatch):
    """Column with NaN in its head rows is filled with the training-fold mean.

    LATE_COL is constructed so:
      - train-fold mean (rows 24..35 ≈ 1.0, row 23 is NaN)  ≈ 1.0
      - full-matrix mean (rows 24..35 ≈ 1.0 plus rows 36..71 ≈ 100.0) ≈ 75.0

    The two differ by >10, so the assertion is non-vacuous.
    """
    n = 72
    features = _make_synthetic_regime_features(n_months=n, switch_at=24)

    rng = np.random.default_rng(99)
    late_col = np.full(n, np.nan)
    late_col[24:36] = rng.normal(1.0, 0.001, 12)    # non-NaN only in training fold
    late_col[36:] = rng.normal(100.0, 0.001, n - 36)
    features = features.copy()
    features["LATE_COL"] = late_col

    captured_X_train: list[pd.DataFrame] = []
    original_fit = RegimeClassifier.fit

    def capturing_fit(self, X, y):
        captured_X_train.append(X.copy())
        return original_fit(self, X, y)

    monkeypatch.setattr(RegimeClassifier, "fit", capturing_fit)

    cv = WalkForwardCV(train_period=36, test_period=12, embargo=3, step=12)
    run_walk_forward(cv=cv, features=features, output_dir=tmp_path)

    assert captured_X_train, "RegimeClassifier.fit was never called"
    X_first = captured_X_train[0]   # fold 0, logistic

    # No NaN must reach the classifier.
    assert not X_first.isna().any().any(), "NaN reached the classifier after imputation"

    # Identify which rows in the fold's training slice had NaN in LATE_COL
    # before imputation (using the original features as the reference).
    original_late_col_in_fold = features.loc[X_first.index, "LATE_COL"]
    nan_mask = original_late_col_in_fold.isna()
    assert nan_mask.any(), (
        "Test precondition: at least one training-fold row must be NaN in LATE_COL"
    )

    # Train-fold mean = mean of the non-NaN training rows.
    expected_train_mean = float(original_late_col_in_fold.mean())  # pandas skips NaN
    full_matrix_mean = float(features["LATE_COL"].mean())
    assert abs(full_matrix_mean - expected_train_mean) > 10, (
        "Test precondition: train-fold mean and full-matrix mean must differ by >10"
    )

    # Every cell that was NaN before imputation must now equal the train-fold mean.
    np.testing.assert_allclose(
        X_first.loc[nan_mask, "LATE_COL"].to_numpy(),
        np.full(nan_mask.sum(), expected_train_mean),
        atol=1e-9,
    )


def test_walk_forward_drops_all_nan_train_column(tmp_path, monkeypatch, caplog):
    """Column entirely NaN within the training window is dropped from both
    train and test matrices, and a WARNING naming the column is logged.

    GHOST_COL is all-NaN for every month so every fold's training window
    has zero usable signal — it must be absent from X_train on every .fit()
    call and a WARNING must be emitted.  If GHOST_COL were left in X_test
    but absent from X_train, sklearn raises a feature-count mismatch on
    predict — so run_walk_forward completing without error also proves it
    was dropped from X_test.
    """
    n = 72
    features = _make_synthetic_regime_features(n_months=n, switch_at=24)

    features = features.copy()
    features["GHOST_COL"] = np.nan   # all-NaN in every fold's training window

    captured_X_train: list[pd.DataFrame] = []
    original_fit = RegimeClassifier.fit

    def capturing_fit(self, X, y):
        captured_X_train.append(X.copy())
        return original_fit(self, X, y)

    monkeypatch.setattr(RegimeClassifier, "fit", capturing_fit)

    cv = WalkForwardCV(train_period=36, test_period=12, embargo=3, step=12)
    with caplog.at_level(logging.WARNING, logger="src.models.regime_classifier"):
        run_walk_forward(cv=cv, features=features, output_dir=tmp_path)

    assert captured_X_train, "RegimeClassifier.fit was never called"

    # GHOST_COL must be absent from every X_train passed to .fit().
    for i, X in enumerate(captured_X_train):
        assert "GHOST_COL" not in X.columns, (
            f"GHOST_COL was not dropped from X_train (captured call #{i})"
        )

    # At least one WARNING must name GHOST_COL.
    warning_messages = [
        r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
    ]
    assert any("GHOST_COL" in msg for msg in warning_messages), (
        f"Expected WARNING mentioning GHOST_COL; captured warnings: {warning_messages}"
    )
