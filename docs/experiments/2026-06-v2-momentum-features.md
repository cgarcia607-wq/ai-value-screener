# Experiment: v2 regime forecaster — momentum features + current-regime prior

**Status:** Pre-registered, not yet run. Committed BEFORE feature code exists.
**Date:** 2026-06-09

## Baseline (measured, locked)
Persistence baseline (predict regime[T+3] = regime[T], no model):
- Avg macro-F1 across 14 folds: **0.7575**
- Transition folds (8-11) mean: **0.5231**

v1 LR (levels-only, for reference): avg 0.5709, transition folds 0.1590.
v1 lost to persistence on every fold. The bar is persistence, not v1.

## Hypothesis
v1 failed because (a) levels-only features carry no transition signal and
(b) the model had to rediscover persistence from 20 macro series.
v2 adds:
- current rules-regime at time T as a categorical feature (deterministic,
  computable at prediction time — zero leakage)
- 3mo and 12mo deltas: UNRATE, FEDFUNDS, INDPRO, PAYEMS, ICSA
- trailing 12mo z-scores: T10Y3M, BAA10Y, VIXCLS

## Metric & config
macro-F1. Identical 14-fold walk-forward (train_period=150, embargo=3,
test_period=12, step=12). Identical seeds. LR, same hyperparameters as v1.

## Success criteria (BOTH required)
1. v2 avg macro-F1 > 0.7575 (beats persistence overall)
2. v2 transition-fold (8-11) mean > 0.5231 (beats persistence at the turns)

## Failure
Anything else — including beating v1 but not persistence. That outcome is
reported as: "the model cannot beat naive persistence at T+3 on monthly
macro data."

## Stopping rule
ONE iteration. On failure: ship the rules-based nowcaster as the product,
report the forecaster as an honest negative result with the persistence
comparison front and center. No v3 feature set, no model swap, no
threshold adjustment.

## Falsifier
If v2's transition-fold mean lands within ±0.05 of v1's 0.159, momentum
features carry no T+3 transition signal at monthly frequency and the
hypothesis was wrong.

---

## RESULT (2026-06-09, run 20260609_202535)

v2 LR per-fold macro-F1: [0.20, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.45, 0.14, 0.00, 0.26, 0.52, 1.00, 0.40]

- v2 LR average:           **0.6415** (required > 0.7575) — FAIL
- v2 LR transition (8-11): **0.2315** (required > 0.5231) — FAIL

**VERDICT: FAILURE.** Stopping rule binds: ship the rules-based nowcaster
as the product; report the forecaster as a negative result; no v3.

Falsifier not triggered (0.2315 vs v1's 0.159 exceeds ±0.05): momentum
features carry some T+3 signal, but not enough to beat naive persistence
even with the current-regime prior provided as a feature. Conclusion:
T+3 regime-turn prediction from monthly macro aggregates with linear
models is at or below the persistence frontier on this data.

Note: XGBoost averaged 0.6758 (> LR's 0.6415) but below the +0.05
adoption bar, and still below persistence — the verdict is unchanged
under either model.
