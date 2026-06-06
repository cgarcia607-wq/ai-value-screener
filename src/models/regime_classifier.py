"""Calibrated 4-class market regime classifier.

See docs/design/regime_classifier.md for the full design including
the T+3 forward-looking framing.

Phase 1: the `RegimeClassifier` model wrapper (fit / predict /
predict_proba / save / load). predict_proba returns calibrated
probabilities with columns in the canonical REGIMES order regardless
of the order the underlying estimator's classes_ attribute uses.

Phase 2: `run_walk_forward` orchestrator. Walks the WalkForwardCV
folds, fits LR + XGBoost on each, records per-fold and aggregated
metrics, and applies the design doc's adoption rules to pick the
production model.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.feature_engineering.regime_labels import REGIMES, compute_labels
from src.validation.walk_forward import WalkForwardCV

logger = logging.getLogger(__name__)


ModelType = Literal["logistic", "xgboost"]


LABEL_HORIZON_MONTHS: int = 3
"""The T+3 forward-looking prediction horizon — features at time t
target the regime label at t+3. See docs/design/regime_classifier.md
for the rationale (Estrella & Mishkin 3-6 month yield-curve horizon
literature; matches quarterly rebalance cadence)."""


# Adoption thresholds from docs/design/regime_classifier.md §"Adoption
# threshold for XGBoost". Kept as module-level constants so the
# orchestrator and the adoption-decision tests reference the same
# numbers.
XGB_ADOPTION_MIN_LIFT: float = 0.05
"""Average macro-F1 lift required for XGBoost to win outright."""
XGB_TIEBREAKER_BAND: float = 0.02
"""If |XGB avg macro-F1 − LR avg macro-F1| < this, prefer LR for
interpretability."""
XGB_CATASTROPHIC_DEFICIT: float = 0.20
"""Any single fold where XGB macro-F1 is more than this below LR
disqualifies XGB outright."""


class RegimeClassifier:
    """Calibrated multi-class regime classifier.

    Two variants — the LR baseline and the XGBoost challenger from
    the design doc. Both are wrapped in CalibratedClassifierCV with
    isotonic calibration and inner cv=3 (chosen for stability on the
    ~120-row early walk-forward folds; see the design doc).

    The StandardScaler is fit on the training fold and stored on
    self so predict / predict_proba apply the same transform at
    inference time.
    """

    def __init__(self, model_type: ModelType = "logistic") -> None:
        if model_type not in ("logistic", "xgboost"):
            raise ValueError(
                f"model_type must be 'logistic' or 'xgboost', got {model_type!r}"
            )
        self.model_type: ModelType = model_type
        self.scaler_: StandardScaler | None = None
        self.classifier_: CalibratedClassifierCV | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RegimeClassifier":
        """Fit scaler + calibrated classifier on (X, y). Returns self.

        Sample weights come from inverse class frequencies on the
        training fold. For the XGBoost variant they're the only
        class-balancing mechanism (XGBClassifier has no class_weight
        argument); for LR they compound with class_weight='balanced'
        on the base estimator, matching the design doc's walk-forward
        pseudocode.

        Labels are encoded to integer codes using REGIMES as the
        canonical 0..3 mapping. XGBClassifier with explicit num_class
        won't accept string labels; encoding at the wrapper level
        keeps both variants on the same code path and makes the
        predict_proba column order trivially match REGIMES.
        """
        unknown = set(pd.unique(y)) - set(REGIMES)
        if unknown:
            raise ValueError(
                f"y contains labels not in REGIMES: {sorted(unknown)}. "
                f"REGIMES = {REGIMES}."
            )

        scaler = StandardScaler().fit(X.to_numpy())
        X_scaled = scaler.transform(X.to_numpy())

        y_encoded = _encode_labels(y)
        sample_weight = _inverse_frequency_weights(y)
        base = _build_base_model(self.model_type)
        calibrated = CalibratedClassifierCV(base, method="isotonic", cv=3)
        calibrated.fit(X_scaled, y_encoded, sample_weight=sample_weight)

        self.scaler_ = scaler
        self.classifier_ = calibrated
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calibrated probabilities. Columns = REGIMES in canonical order.

        Any regime the classifier never saw during fit appears as a
        zero column so the output shape is always (n_rows, 4).
        """
        self._check_fitted()
        assert self.scaler_ is not None and self.classifier_ is not None
        X_scaled = self.scaler_.transform(X.to_numpy())
        raw = self.classifier_.predict_proba(X_scaled)

        seen_codes = list(self.classifier_.classes_)
        out = np.zeros((raw.shape[0], len(REGIMES)), dtype=float)
        for code in seen_codes:
            out[:, int(code)] = raw[:, seen_codes.index(code)]
        return pd.DataFrame(out, index=X.index, columns=list(REGIMES))

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Hard regime labels — argmax of calibrated probabilities."""
        proba = self.predict_proba(X)
        argmax = proba.to_numpy().argmax(axis=1)
        labels = np.array(REGIMES, dtype=object)[argmax]
        return pd.Series(labels, index=X.index, dtype=object)

    def save(self, path: Path) -> None:
        """Pickle the whole object (scaler + calibrated classifier)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "RegimeClassifier":
        return joblib.load(Path(path))

    def _check_fitted(self) -> None:
        if self.classifier_ is None or self.scaler_ is None:
            raise RuntimeError(
                "RegimeClassifier method called before fit(); "
                "scaler / classifier are not initialized."
            )


def _build_base_model(model_type: ModelType):
    if model_type == "logistic":
        return LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
    return XGBClassifier(
        objective="multi:softprob",
        num_class=len(REGIMES),
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        tree_method="hist",
    )


def _encode_labels(y: pd.Series) -> np.ndarray:
    code_of = {regime: idx for idx, regime in enumerate(REGIMES)}
    return y.map(code_of).to_numpy(dtype=np.int64)


def _inverse_frequency_weights(y: pd.Series) -> np.ndarray:
    counts = y.value_counts()
    n = len(y)
    n_classes = len(counts)
    return y.map(lambda v: n / (n_classes * counts[v])).to_numpy(dtype=float)


# --- Phase 2: walk-forward orchestrator -------------------------------------


def _build_t_plus_3_targets(features: pd.DataFrame) -> pd.Series:
    """Construct the T+LABEL_HORIZON_MONTHS regime targets aligned to
    features.index.

    `compute_labels` drops the leading rows where its rolling-window
    derivations are NaN, so the labels series is a proper subset of
    `features.index`. After shifting forward by 3 months within the
    labels series, we reindex back to features so every position in
    features maps to either a valid T+3 label or NaN:

    - leading rows where compute_labels couldn't derive a label → NaN
    - trailing LABEL_HORIZON_MONTHS rows where T+3 hasn't happened
      yet (the shift puts NaN here by construction) → NaN
    """
    labels_t = compute_labels(features)
    return labels_t.shift(-LABEL_HORIZON_MONTHS).reindex(features.index)


def _evaluate(
    clf: RegimeClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    """Compute per-fold discrimination + calibration metrics for one model.

    Returns macro-F1, per-class F1, multi-class Brier (mean of the
    per-class one-vs-rest Brier scores), and the confusion matrix
    in REGIMES order. Missing classes in y_test fall through cleanly:
    f1_score(zero_division=0) → 0, brier_score_loss with pos_label=1
    handles all-zero y safely.
    """
    y_pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)

    macro_f1 = float(
        f1_score(y_test, y_pred, average="macro", zero_division=0)
    )
    per_class_f1_array = f1_score(
        y_test, y_pred, labels=list(REGIMES), average=None, zero_division=0
    )
    per_class_f1 = {
        regime: float(per_class_f1_array[i]) for i, regime in enumerate(REGIMES)
    }

    y_test_arr = y_test.to_numpy()
    brier_per_class = {}
    for regime in REGIMES:
        y_bin = (y_test_arr == regime).astype(int)
        p = proba[regime].to_numpy()
        brier_per_class[regime] = float(
            brier_score_loss(y_bin, p, pos_label=1)
        )
    brier = float(np.mean(list(brier_per_class.values())))

    cm = confusion_matrix(
        y_test, y_pred, labels=list(REGIMES)
    ).tolist()

    return {
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "brier": brier,
        "brier_per_class": brier_per_class,
        "confusion_matrix": {"labels": list(REGIMES), "matrix": cm},
    }


def _apply_adoption_decision(
    lr_macro_f1_per_fold: list[float],
    xgb_macro_f1_per_fold: list[float],
) -> dict:
    """Apply the design doc's LR-vs-XGBoost adoption rules.

    Rules (in evaluation order):
    1. If any single fold has XGB macro-F1 more than
       XGB_CATASTROPHIC_DEFICIT below LR → LR wins (XGB disqualified).
    2. If |XGB avg − LR avg| < XGB_TIEBREAKER_BAND → LR wins (the
       simpler model gets the tie).
    3. If XGB avg − LR avg >= XGB_ADOPTION_MIN_LIFT → XGB wins.
    4. Otherwise → LR wins (XGB didn't clear the adoption bar).
    """
    if (
        len(lr_macro_f1_per_fold) != len(xgb_macro_f1_per_fold)
        or not lr_macro_f1_per_fold
    ):
        raise ValueError(
            "Adoption decision requires matching, non-empty per-fold "
            f"lists; got len(lr)={len(lr_macro_f1_per_fold)}, "
            f"len(xgb)={len(xgb_macro_f1_per_fold)}."
        )

    lr_avg = float(np.mean(lr_macro_f1_per_fold))
    xgb_avg = float(np.mean(xgb_macro_f1_per_fold))
    lift = xgb_avg - lr_avg

    worst_fold_deficit = max(
        lr - xgb
        for lr, xgb in zip(
            lr_macro_f1_per_fold, xgb_macro_f1_per_fold, strict=True
        )
    )

    if worst_fold_deficit > XGB_CATASTROPHIC_DEFICIT:
        reason = (
            f"XGBoost disqualified: worst-fold deficit "
            f"{worst_fold_deficit:.4f} exceeds catastrophic-failure "
            f"threshold {XGB_CATASTROPHIC_DEFICIT:.2f}."
        )
        winner = "logistic"
    elif abs(lift) < XGB_TIEBREAKER_BAND:
        reason = (
            f"Within tiebreaker band (|lift|={abs(lift):.4f} < "
            f"{XGB_TIEBREAKER_BAND:.2f}); prefer LR for interpretability."
        )
        winner = "logistic"
    elif lift >= XGB_ADOPTION_MIN_LIFT:
        reason = (
            f"XGBoost macro-F1 lift {lift:.4f} clears adoption bar "
            f"{XGB_ADOPTION_MIN_LIFT:.2f}."
        )
        winner = "xgboost"
    else:
        reason = (
            f"XGBoost lift {lift:.4f} below adoption bar "
            f"{XGB_ADOPTION_MIN_LIFT:.2f}; keep LR."
        )
        winner = "logistic"

    return {
        "winner": winner,
        "lr_macro_f1": lr_avg,
        "xgb_macro_f1": xgb_avg,
        "reason": reason,
    }


def run_walk_forward(
    cv: WalkForwardCV,
    features: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """Walk-forward training orchestrator for the regime classifier.

    For each fold from `cv.split(features.index)`:
      1. Construct T+3 targets from compute_labels(features).
      2. Drop rows where the T+3 target is NaN (leading
         compute_labels-dropped rows and trailing shift-induced rows).
      3. Fit both RegimeClassifier variants ("logistic" and "xgboost")
         on the fold's train slice. The wrapper handles its own
         StandardScaler internally.
      4. Compute macro-F1, per-class F1, multi-class Brier, and the
         confusion matrix on the fold's test slice.
      5. Save fold_<N>_lr.joblib, fold_<N>_xgb.joblib, and
         fold_<N>_metrics.json (both models' metrics in one file).

    After all folds:
      - Write aggregated_metrics.json (per-fold macro-F1 + averages
        for both models).
      - Apply the LR-vs-XGBoost adoption rules and write
        adoption_decision.json.

    All artifacts land under output_dir / <timestamp> so multiple
    runs coexist for comparison.

    Args:
        cv: WalkForwardCV. Per the design doc the embargo must equal
            LABEL_HORIZON_MONTHS so training-row labels (which look 3
            months forward) can't overlap with test-row features.
        features: monthly FRED feature matrix from
            get_features_matrix(...). Must contain the rule columns
            consumed by compute_labels (UNRATE, T10Y3M, NFCICREDIT).
        output_dir: parent directory. A timestamped subdir is created
            inside it for this run.

    Returns:
        The adoption_decision dict — also written to
        adoption_decision.json on disk.

    Raises:
        ValueError: cv yields no folds, or every fold's train/test
            window collapses to empty after dropping NaN-label rows.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("run_walk_forward: writing artifacts to %s", run_dir)

    targets = _build_t_plus_3_targets(features)

    folds = list(cv.split(features.index))
    if not folds:
        raise ValueError(
            "WalkForwardCV produced no folds for the input features.index; "
            "extend the data range or reduce the CV periods."
        )

    per_fold_lr_f1: list[float] = []
    per_fold_xgb_f1: list[float] = []
    fold_records: list[dict] = []

    for fold in folds:
        train_mask = targets.iloc[fold.train_indices].notna().to_numpy()
        test_mask = targets.iloc[fold.test_indices].notna().to_numpy()
        train_valid = fold.train_indices[train_mask]
        test_valid = fold.test_indices[test_mask]

        if len(train_valid) == 0 or len(test_valid) == 0:
            logger.warning(
                "Fold %d collapsed after NaN-target filtering "
                "(n_train=%d, n_test=%d); skipping.",
                fold.fold_id, len(train_valid), len(test_valid),
            )
            continue

        X_train = features.iloc[train_valid]
        y_train = targets.iloc[train_valid]
        X_test = features.iloc[test_valid]
        y_test = targets.iloc[test_valid]

        lr = RegimeClassifier(model_type="logistic").fit(X_train, y_train)
        xgb = RegimeClassifier(model_type="xgboost").fit(X_train, y_train)

        lr_metrics = _evaluate(lr, X_test, y_test)
        xgb_metrics = _evaluate(xgb, X_test, y_test)

        logger.info(
            "Fold %d (train=[%s, %s] n=%d, test=[%s, %s] n=%d): "
            "LR macro-F1=%.4f, XGB macro-F1=%.4f",
            fold.fold_id,
            fold.train_start, fold.train_end, len(train_valid),
            fold.test_start, fold.test_end, len(test_valid),
            lr_metrics["macro_f1"], xgb_metrics["macro_f1"],
        )

        lr.save(run_dir / f"fold_{fold.fold_id}_lr.joblib")
        xgb.save(run_dir / f"fold_{fold.fold_id}_xgb.joblib")

        fold_record = {
            "fold_id": fold.fold_id,
            "train_start": fold.train_start.isoformat(),
            "train_end": fold.train_end.isoformat(),
            "test_start": fold.test_start.isoformat(),
            "test_end": fold.test_end.isoformat(),
            "n_train": int(len(train_valid)),
            "n_test": int(len(test_valid)),
            "logistic": lr_metrics,
            "xgboost": xgb_metrics,
        }
        with (run_dir / f"fold_{fold.fold_id}_metrics.json").open("w") as f:
            json.dump(fold_record, f, indent=2)

        per_fold_lr_f1.append(lr_metrics["macro_f1"])
        per_fold_xgb_f1.append(xgb_metrics["macro_f1"])
        fold_records.append(fold_record)

    if not per_fold_lr_f1:
        raise ValueError(
            "All folds collapsed after NaN-target filtering — no model was "
            "fit. Check the feature matrix length vs the CV configuration."
        )

    lr_avg = float(np.mean(per_fold_lr_f1))
    xgb_avg = float(np.mean(per_fold_xgb_f1))
    aggregated = {
        "n_folds": len(per_fold_lr_f1),
        "logistic": {
            "avg_macro_f1": lr_avg,
            "per_fold_macro_f1": per_fold_lr_f1,
        },
        "xgboost": {
            "avg_macro_f1": xgb_avg,
            "per_fold_macro_f1": per_fold_xgb_f1,
        },
    }
    with (run_dir / "aggregated_metrics.json").open("w") as f:
        json.dump(aggregated, f, indent=2)

    decision = _apply_adoption_decision(per_fold_lr_f1, per_fold_xgb_f1)
    with (run_dir / "adoption_decision.json").open("w") as f:
        json.dump(decision, f, indent=2)

    logger.info(
        "Adoption decision: winner=%s (LR avg macro-F1=%.4f, "
        "XGB avg macro-F1=%.4f). %s",
        decision["winner"],
        decision["lr_macro_f1"],
        decision["xgb_macro_f1"],
        decision["reason"],
    )

    return decision
