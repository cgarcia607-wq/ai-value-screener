"""Phase 1 unit tests for src/models/regime_classifier.py.

Fast tests only — synthetic 4-class data, no FRED API. Phase 2 will
add walk-forward orchestrator tests and integration tests against
real FRED data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.regime_labels import REGIMES
from src.models.regime_classifier import RegimeClassifier


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
