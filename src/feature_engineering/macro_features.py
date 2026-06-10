"""v2 feature engineering for the regime forecaster.

Adds momentum features (3-month / 12-month deltas), trailing z-scores,
and a current-regime one-hot prior to the FRED feature matrix.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.feature_engineering.regime_labels import REGIMES, compute_labels

logger = logging.getLogger(__name__)

_DELTA_COLS: tuple[str, ...] = ("UNRATE", "FEDFUNDS", "INDPRO", "PAYEMS", "ICSA")
_ZSCORE_COLS: tuple[str, ...] = ("T10Y3M", "BAA10Y", "VIXCLS")
_DELTA_WINDOWS: tuple[int, int] = (3, 12)
_ZSCORE_WINDOW: int = 12


def build_v2_features(matrix: pd.DataFrame) -> pd.DataFrame:
    """Augment the FRED feature matrix with v2 momentum features and the
    current-regime prior. Returns a NEW DataFrame: all original columns plus:
    - {col}_chg_3mo and {col}_chg_12mo for UNRATE, FEDFUNDS, INDPRO, PAYEMS, ICSA
      (simple difference: x[t] - x[t-k])
    - {col}_z12 for T10Y3M, BAA10Y, VIXCLS
      (trailing 12-month z-score: (x[t] - rolling12_mean) / rolling12_std;
       rolling stats use ONLY data up to and including t — no centered windows)
    - regime_now_{REGIME} : 4 one-hot columns from compute_labels(matrix)
      applied at time t. This is the current-regime prior — deterministic
      from features at t, zero leakage.
    Leading rows where rolling windows lack history will contain NaN —
    leave them; the orchestrator's existing dropna/imputation handles them.
    """
    out = matrix.copy()

    for col in _DELTA_COLS:
        if col not in matrix.columns:
            logger.warning(
                "build_v2_features: column %s not found — skipping its delta features",
                col,
            )
            continue
        for k in _DELTA_WINDOWS:
            out[f"{col}_chg_{k}mo"] = matrix[col] - matrix[col].shift(k)

    for col in _ZSCORE_COLS:
        if col not in matrix.columns:
            logger.warning(
                "build_v2_features: column %s not found — skipping its z-score",
                col,
            )
            continue
        roll = matrix[col].rolling(window=_ZSCORE_WINDOW)
        out[f"{col}_z12"] = (matrix[col] - roll.mean()) / roll.std()

    # Current-regime prior: regime label at t (not t+3), one-hot encoded.
    # compute_labels drops leading rows where its rolling windows are NaN;
    # reindex to the full matrix index so unlabeled rows carry NaN.
    labels = compute_labels(matrix).reindex(matrix.index)
    valid = labels.notna()
    for regime in REGIMES:
        col_name = f"regime_now_{regime}"
        values = pd.Series(np.nan, index=matrix.index, dtype=float)
        values[valid] = (labels[valid] == regime).astype(float)
        out[col_name] = values

    return out
