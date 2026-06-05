# Design: Market regime classifier

**Modules**: `src/feature_engineering/regime_labels.py`,
`src/models/regime_classifier.py`
**Status**: Proposed, pre-implementation
**Owner**: Chris Garcia
**Last updated**: 2026-06-05

## Purpose

Classify the macro/market environment **3 months in the future** into
one of four regimes: **Expansion / Late-cycle / Contraction /
Recovery**. The model takes features at time T and predicts the
regime label at T+3. Per [CLAUDE.md](../../CLAUDE.md), the regime
classifier is a downstream input to the stock screener's
conditioning features and is the primary user-facing element of the
regime dashboard page — which shows a forward-looking regime
forecast, not a nowcast.

The model consumes the FRED feature matrix (see
[fred_client.md](fred_client.md)) and uses
[walk_forward_cv.md](walk_forward_cv.md) for validation, with
embargo = 3 months matching the label horizon.

## Scope: two files, one model

- `regime_labels.py`: deterministic rule that maps a feature row to a
  regime label. Pure function, no model. Importable by both the
  classifier and the validation tests.
- `regime_classifier.py`: ML wrapper. Takes features, predicts a
  regime distribution. Two model variants (LR baseline + XGBoost),
  calibrated probabilities, walk-forward training harness.

## Ground truth labels — the rule set

CLAUDE.md mandates rules-based labels using yield curve, unemployment
trend, and credit spreads, validated against NBER USREC. The proposal:

### Label features (computed in `regime_labels.py`)

These are derived from `get_features_matrix(...)` columns. The
derivations are deterministic and pure.

| feature | source | derivation |
|---|---|---|
| `unrate` | `UNRATE` | direct |
| `unrate_3mo_change` | `UNRATE` | `UNRATE(t) − UNRATE(t−3)` |
| `unrate_24mo_min` | `UNRATE` | trailing 24-month min |
| `unrate_above_min` | derived | `unrate − unrate_24mo_min` (pp) |
| `unrate_12mo_max` | `UNRATE` | trailing 12-month max |
| `t10y3m` | `T10Y3M` | direct (Fed's preferred recession spread) |
| `nfcicredit` | `NFCICREDIT` | direct (positive = tighter credit) |

USREC is **deliberately excluded from the rules**, even though it's
in the FRED inventory. Rationale below.

### The four rules (evaluated top to bottom; first match wins)

```
1. Contraction:
   unrate_3mo_change > 0.5  AND  nfcicredit > 0.5
   (Sahm-rule-flavored stress detection; real-time-available)

2. Recovery:
   unrate_12mo_max > 6.0  AND  unrate_3mo_change < 0
   (coming off elevated unemployment with falling trend)

3. Late-cycle:
   t10y3m < 0  AND  unrate_above_min < 0.5  AND  nfcicredit < 0.5
   (yield curve inverted, unemployment near cycle low,
    credit not yet stressed — the classic late-cycle signature)

4. Expansion:
   (default — none of the above)
```

These thresholds are starting points and **will be tuned** against
historical episodes (open question §1). The rules are evaluated as
a cascade so a single regime label results per date.

### Why USREC is not in the rules (the publication-lag problem)

USREC is the NBER recession indicator. NBER announces recession start
and end dates retrospectively, often **6-12 months after the fact**.
At any real-time date T, USREC(T) reflects what NBER currently
believes about T, which may have been backfilled long after T.

If we used USREC in the labeling rules:
- Training labels for historical dates would benefit from full
  retroactive knowledge — fine for ground-truth labeling.
- Test-fold labels at vintage `fold.test_start` would be **missing**
  for recent dates where NBER hadn't yet announced. The COVID
  recession is a clean example: NBER announced June 2020 that
  recession started Feb 2020; at vintage 2020-03-01, USREC(Feb 2020)
  was 0, not 1.

Using a feature that has different availability between training and
test is the kind of subtle look-ahead bias that's hard to detect.
Sticking to features with consistent vintage availability (UNRATE,
T10Y3M, NFCICREDIT all publish within weeks of their reference month
with stable revision behavior) keeps the labels honest.

USREC is still used for **validation** — we expect Contraction
labels to overlap NBER recession periods historically. Disagreement
is informative: if our rules call Contraction in a period NBER never
confirmed a recession, the rules may be too sensitive (and vice
versa). See Test plan.

### Alternative taxonomies considered

- **NBER 2-state** (Expansion vs Recession): too coarse for the
  screener's conditioning needs. The dashboard would be one bit.
- **Bridgewater All Weather** (growth × inflation 4-quadrant):
  feature-derivative, not phase-of-cycle. Different abstraction;
  good model in its own right but doesn't match CLAUDE.md's lock.
- **6-state with early/mid/late expansion**: more granular but no
  consensus on thresholds; ~420 monthly observations can't support
  it.
- **HMM with latent regime states**: covered under §Label generation
  strategy below.

The 4-state cycle decomposition (E/L/C/R) is what CLAUDE.md locks
and what I propose to ship. The thresholds within each rule are
the open question, not the taxonomy itself.

## Label generation strategy — forward-looking T+3 prediction

The original CLAUDE.md framing had concurrent labels (model predicts
regime at time T from features at time T). With deterministic
rules-based labels, that reduces the classifier to a rule
approximator — it can't beat the rules and adds no real value.

**Revised approach**: labels are rules-based at time t, but the
prediction task is **shifted forward by 3 months**.

- `regime_label[t]` = rules applied to features at time t (unchanged
  from prior design).
- Training pair = `(features[t], regime_label[t+3])`.
- The classifier learns "features-now → regime-in-three-months."
- The dashboard reports a regime **forecast** rather than nowcast.

This transforms the model from function-approximator to leading-
indicator predictor. Where the rules tell us "based on what's
happening now, here's the regime now", the model tells us "based
on what's happening now, here's where the regime is heading three
months out." Genuinely different and genuinely useful.

### Implications

1. **Embargo = 3 months in walk-forward CV** (was 1 in the original
   design). The 3-month label window must not extend into the test
   set. See [walk_forward_cv.md](walk_forward_cv.md).
2. **Trailing 3 months of any window have no valid label** —
   their T+3 hasn't happened yet. Drop them from both train and
   test sets when constructing training pairs.
3. **LR-vs-XGBoost comparison becomes meaningful.** With concurrent
   labels both models converge to the rule function and differ only
   in calibration. With forward-looking labels, the model has to
   actually learn how current features predict future regime
   transitions — XGBoost's ability to capture nonlinear interactions
   is testable.

### Why T+3, not T+1 or T+12?

- T+1 (one-month-ahead) is close to concurrent for sticky regimes
  and gives little forecast lift.
- T+12 (one-year-ahead) is too noisy — too many regime transitions
  can intervene; the macro features at time T mostly don't carry
  information that far out.
- T+3 is the established window in the Fed regime-prediction
  literature (Estrella & Mishkin's yield-curve work targets 3-6
  month horizons) and matches the natural decision cadence of an
  asset allocator. Quarterly rebalances care about "what's the
  regime next quarter."

### Code path

```
fred = get_features_matrix(start, end, vintage_date)
labels_t = regime_labels(fred)                       # rules at time t
y = labels_t.shift(-3)                                # the T+3 target
X = fred
train_mask = y.notna() & (idx in fold.train_indices)  # drop trailing 3
X_train, y_train = X[train_mask], y[train_mask]
test_mask = y.notna() & (idx in fold.test_indices)
X_test, y_test = X[test_mask], y[test_mask]
classifier.fit(X_train, y_train)
proba = classifier.predict_proba(X_test)
```

`regime_labels(features)` returns labels at time t — pure rules,
deterministic. The T+3 shift happens in the classifier's training
orchestrator, not in `regime_labels.py` itself. Clean separation:
`regime_labels.py` knows nothing about the prediction horizon; it
just maps features → label. The orchestrator chooses how to align
training pairs.

`regime_labels` lives in `feature_engineering/`; the classifier
lives in `models/`. Clean separation.

## Diagnostic phase — review labels before training

Before any model code is written, a diagnostic script
`scripts/regime_label_diagnostic.py` applies the proposed rules to
real FRED data over 2014-2025 and prints the regime label per
month. The owner reviews the output for plausibility:

- Does the COVID recession (Feb-Apr 2020) classify as Contraction
  (or Recovery in the rebound)?
- Does the 2018-2019 yield curve inversion produce Late-cycle?
- Does the post-COVID 2020-Q3+ period progress from Recovery to
  Expansion as the economy recovers?

If the labels look wrong, the rules get adjusted before the
classifier is built. **The labels are the ground truth; if they
look wrong, no model can fix them.** This is the same pattern as
the IVV discovery phase that surfaced the data-source mismatch
before code was written.

The script applies the rule logic inline (or imports from
`regime_labels.py` once written) and is gitignored output-wise but
the script itself is committed for reproducibility. See the
script's docstring for run instructions and expected output.

## Feature set for the model

All 20 features from `get_features_matrix()` are candidates. The
model sees them all by default; feature pruning is a tunable.

USREC is excluded from the model inputs (in addition to from the
rules) because:
- It's the validation reference; using it as a feature would be
  circular.
- Its publication lag (§above) makes it inconsistently available
  across vintages.

So model features = 20 from FRED, `USREC` excluded. Concretely:
`exclude_targets=True` (the `get_features_matrix` default) does
exactly the right thing already.

### Transformations

v1 is deliberately simple:

- **All features standardized** (per-feature z-score) using
  *training-fold-only* statistics. LR is scale-sensitive; XGBoost is
  scale-invariant but standardization makes feature importances
  comparable.
- **No level-vs-change transformations in v1.** Many features are
  naturally interpretable as levels (yield curve, credit spreads,
  VIX); the model is allowed to learn from levels directly.
- **Missing values forward-filled** by the FRED client's `ffill`
  resample policy. Any remaining NaN (e.g., for series that started
  after the fold's `train_start`) is imputed with the training-fold
  mean.

v2 may add:
- Month-over-month changes for trend signals
- Trailing 12-month z-scores (relative to recent history)
- Inflation as yoy % change of CPI, not CPI level

These are deferred until v1 baseline is measured.

## Model architecture

### Baseline: multinomial logistic regression

```
sklearn.linear_model.LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    C=1.0,                       # L2 inverse regularization strength
    max_iter=1000,
    class_weight='balanced',     # offsets the heavy Expansion class
    random_state=42,
)
```

Why these defaults:
- L2 (no L1) — feature count is small (20), no need for sparsity;
  L2 stabilizes coefficients without zeroing them.
- `class_weight='balanced'` — Expansion is ~70%+ of observations
  historically; without weighting, the model would over-predict
  Expansion. Balanced weighting equalizes per-class loss
  contribution.
- `C=1.0` — standard starting point; tuned via inner CV if needed
  (likely not for 4-class on 20 features).
- `random_state=42` — every random seed in this project is 42 for
  reproducibility.

### Challenger: XGBoost

```
xgboost.XGBClassifier(
    objective='multi:softprob',
    num_class=4,
    n_estimators=100,
    max_depth=3,                 # SHALLOW — 420 obs total, easy to overfit
    learning_rate=0.1,
    reg_alpha=1.0,               # L1
    reg_lambda=1.0,              # L2
    random_state=42,
    use_label_encoder=False,
    tree_method='hist',
)
```

Why these defaults:
- `max_depth=3` — at ~120 training observations per fold (early
  folds), depth-6 trees overfit catastrophically. Shallow is forced
  by sample size, not preference.
- `n_estimators=100` — gradient boosting works fine with 50-200
  trees on small data; 100 is the boring choice.
- `reg_alpha + reg_lambda` — both L1 and L2 for an aggressive
  regularization stance against the tiny-data overfitting risk.
- No class_weight argument in XGBClassifier — use `sample_weight`
  computed from inverse class frequencies on the training fold.

### Adoption threshold for XGBoost

XGBoost has to clear a bar to be adopted over LR:

- **macro-F1 lift ≥ 5 percentage points** averaged across walk-forward
  folds. A 1-2pp lift could be noise.
- **Tiebreaker for close calls**: if LR and XGBoost are within ±2pp
  on macro-F1, prefer LR for interpretability. The model only ships
  XGBoost when it clearly wins (5pp+ lift); ties go to the simpler
  model.
- **No fold where XGBoost test accuracy is more than 20pp below LR
  test accuracy.** Catastrophic single-fold failures are
  disqualifying.
- **Calibration not measurably worse** (Brier score within 10% of
  LR's Brier).

If XGBoost fails these, **ship LR**. Document the decision in the
final evaluation report.

CLAUDE.md is explicit: "If XGBoost doesn't beat it meaningfully,
use the simpler model."

## Calibration

Per CLAUDE.md, user-facing probabilities must be calibrated via
`CalibratedClassifierCV(method='isotonic')`. The fit pattern within
a walk-forward fold:

```python
base = LogisticRegression(...)  # or XGBClassifier
calibrated = CalibratedClassifierCV(base, method='isotonic', cv=3)
calibrated.fit(X_train, y_train, sample_weight=...)
proba = calibrated.predict_proba(X_test)  # calibrated probabilities
```

Inner cv=3 (not 5) because early walk-forward folds have ~120
training rows; 5-fold inner CV gives ~24-row holdouts which are too
small for stable isotonic fit. 3-fold gives ~40-row holdouts.

For folds with sufficient data (later expanding-window folds), cv=5
could be used; v1 fixes cv=3 for consistency. Open question §4.

## Evaluation metrics

The right metric set for an imbalanced 4-class problem with
calibrated probabilities:

### Discrimination (which class did the model predict?)

- **Confusion matrix** — full 4×4. Visualizable; shows which
  regime-pairs the model confuses.
- **Per-class precision, recall, F1**.
- **Macro-averaged F1** — equal weight per class. Critical for
  imbalanced problems where the model could get 70% accuracy by
  always predicting Expansion.

### Probability quality

- **Brier score** (multi-class) — mean squared error of predicted
  probability vs one-hot true label. Lower is better. Measures
  calibration AND discrimination simultaneously.
- **Reliability diagram** — plotted per regime; visualizes whether
  predicted p(class) matches empirical frequency of that class.

### Gating thresholds

| metric | baseline LR floor | XGB adoption bar |
|---|---|---|
| macro-F1 | > 0.40 | ≥ LR + 0.05 |
| per-class recall, all 4 classes | > 0.20 | no worse than LR |
| Brier score (multi-class) | <= 0.35 | within 10% of LR |

A model that fails the LR floor doesn't ship at all. A model whose
predictions are systematically poorly calibrated (Brier > 0.35)
doesn't ship either; calibration is the whole point of using
`CalibratedClassifierCV`.

### NBER overlap sanity check (validation, not training metric)

For every period where `USREC == 1` (NBER recession), at least 50%
of months should be labeled Contraction by our rules. If the rules
miss the majority of NBER recessions, the rules are wrong, not the
classifier. This is a label-quality check, computed once per
training run.

## Walk-forward integration

Per fold from `WalkForwardCV(train_period, embargo=3, ...)`:

1. Fetch features at the fold's vintage:
   ```python
   matrix = get_features_matrix(
       start=fold.train_start,
       end=fold.test_end,
       vintage_date=fold.vintage_date,
   )
   ```
2. Generate labels at time t for all rows, then shift to T+3 targets:
   ```python
   labels_t = regime_labels(matrix)
   y_t_plus_3 = labels_t.shift(-3)   # NaN in trailing 3 rows
   ```
3. Split into train and test by fold indices, dropping rows where the
   shifted label is NaN (the trailing 3 months of each window):
   ```python
   train_idx = fold.train_indices
   test_idx = fold.test_indices
   train_valid = train_idx[y_t_plus_3.iloc[train_idx].notna().to_numpy()]
   test_valid = test_idx[y_t_plus_3.iloc[test_idx].notna().to_numpy()]
   X_train = matrix.iloc[train_valid]
   y_train = y_t_plus_3.iloc[train_valid]
   X_test = matrix.iloc[test_valid]
   y_test = y_t_plus_3.iloc[test_valid]
   ```
4. Standardize on train, apply to test:
   ```python
   scaler = StandardScaler().fit(X_train)
   X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
   ```
5. Fit baseline LR with calibration:
   ```python
   lr = CalibratedClassifierCV(LogisticRegression(...), method='isotonic', cv=3)
   lr.fit(X_train_s, y_train, sample_weight=...)
   ```
6. Fit XGBoost with calibration:
   ```python
   xgb = CalibratedClassifierCV(XGBClassifier(...), method='isotonic', cv=3)
   xgb.fit(X_train_s, y_train, sample_weight=...)
   ```
7. Predict on test and record metrics per model. Note: y_test is the
   T+3 forward label, so a correct prediction means the model
   anticipated where the regime would be 3 months out.

**Nothing persists between folds.** Each fold is independent —
fresh scaler, fresh classifier, fresh calibration. Aggregation
happens at the report layer, not during training.

Embargo is **3 months** per the forward-looking label horizon. The
embargo must be at least as long as the label-computation window
to prevent training-row labels (which look 3 months forward) from
overlapping with test-row features. CLAUDE.md updated accordingly.

## Outputs and persistence

```
models/regime_classifier/<run_timestamp>/
  config.json                  # all hyperparameters + walk-forward config
  fold_<N>_lr.joblib           # fitted LR (calibrated) per fold
  fold_<N>_xgb.joblib          # fitted XGB (calibrated) per fold
  fold_<N>_scaler.joblib       # StandardScaler per fold
  fold_<N>_metrics.json        # per-fold metrics for both models
  fold_<N>_shap_xgb.parquet    # SHAP values for test predictions, XGB only
  aggregated_metrics.json      # macro-averaged across folds
  confusion_matrix.parquet     # mean confusion matrix
  reliability.parquet          # calibration data for plotting
  adoption_decision.json       # which model won, with metric comparison
  notes.md                     # any anomalies the run surfaced
```

All under `models/` which is gitignored. Run-timestamped so multiple
runs coexist for comparison.

## Public API

### `regime_labels.py`

```python
REGIMES: list[str] = ["Expansion", "Late-cycle", "Contraction", "Recovery"]
LABEL_HORIZON_MONTHS: int = 3   # the T+3 prediction horizon


def regime_labels(features: pd.DataFrame) -> pd.Series:
    """Apply the rules to produce a regime label PER ROW at time t.

    This function knows nothing about the prediction horizon — it
    just maps features at time t to a regime label at time t. The
    T+3 shift that creates training pairs lives in the classifier
    orchestrator, not here.

    Args:
        features: DataFrame from get_features_matrix(). Must contain
            UNRATE, T10Y3M, NFCICREDIT columns. Other columns
            ignored.

    Returns:
        pd.Series indexed identically to `features`, values in
        REGIMES.

    Raises:
        ValueError: required columns missing, or NaN in any of the
            three rule features after the fold's standard imputation
            should have already handled them.
    """


def validate_against_nber(
    labels: pd.Series, usrec: pd.Series, min_overlap: float = 0.5
) -> dict:
    """Sanity-check labels against NBER recession dating.

    For each NBER recession period (contiguous USREC==1 run), compute
    the fraction of months our rules labeled Contraction. Returns
    {recession_id, start, end, overlap_fraction}. Caller raises if
    any overlap < min_overlap.
    """
```

### `regime_classifier.py`

```python
class RegimeClassifier:
    """Calibrated multi-class regime classifier.

    Two model variants:
        model_type='logistic' -> CalibratedClassifierCV(LogisticRegression)
        model_type='xgboost'  -> CalibratedClassifierCV(XGBClassifier)

    Both produce calibrated probabilities matching REGIMES order.
    """

    def __init__(self, model_type: Literal["logistic", "xgboost"] = "logistic"):
        ...

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RegimeClassifier":
        """Fit standardizer + calibrated classifier. Returns self."""

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Hard regime labels (the argmax of predict_proba)."""

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calibrated probabilities. Columns = REGIMES in canonical order."""

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> "RegimeClassifier": ...


def run_walk_forward(
    cv: WalkForwardCV,
    features: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """Train both LR and XGB across all CV folds, save artifacts,
    return the adoption decision dict.

    Loud orchestrator — logs each fold's metrics, the final
    aggregation, and the adoption decision. The dashboard reads
    output_dir to display the production model + its diagnostics.
    """
```

## Test plan

### Unit tests (fast, mocked or synthetic)

`tests/test_regime_labels.py` (label rules, no T+3 shift — that
shift is tested in the classifier orchestrator):
- `test_regime_labels_returns_valid_label_per_row` — output values
  in `REGIMES`, no NaN.
- `test_regime_labels_deterministic` — same input → same output
  (pure function).
- `test_contraction_rule_triggers_on_unemployment_spike` —
  synthetic row with `unrate_3mo_change=1.0, nfcicredit=1.0` →
  Contraction.
- `test_recovery_rule_triggers_off_high_unemployment_peak` —
  synthetic row with `unrate_12mo_max=8.0, unrate_3mo_change=-0.2`
  → Recovery.
- `test_late_cycle_rule_triggers_on_inversion` — synthetic row
  with `t10y3m=-0.2, unrate_above_min=0.1, nfcicredit=0.0` →
  Late-cycle.
- `test_expansion_rule_is_default` — synthetic row with benign
  values → Expansion.
- `test_rule_cascade_order_is_contraction_recovery_latecycle_expansion`
  — pathological row that matches multiple → Contraction wins.
- `test_regime_labels_missing_columns_raises` — missing UNRATE →
  ValueError.
- `test_validate_against_nber_flags_low_overlap` — synthetic case
  where rules miss an NBER recession → returns overlap < 0.5 for
  that period.

`tests/test_regime_classifier.py`:
- `test_classifier_fit_predict_returns_valid_labels` — fits on
  synthetic 4-class data, predict() returns valid REGIMES values.
- `test_classifier_predict_proba_sums_to_one` — every row's
  predicted probabilities sum to 1.
- `test_classifier_predict_proba_columns_are_REGIMES_in_order` —
  column order is canonical.
- `test_classifier_save_load_roundtrip` — saved model loads and
  produces identical predictions.
- `test_logistic_vs_xgboost_models_accept_same_inputs` — both
  model types fit + predict on the same X, y.
- `test_walk_forward_orchestrator_shifts_labels_by_3_months` —
  for a known toy dataset, verify the training-pair construction
  pairs `features[t]` with `labels[t+3]`, not `labels[t]`.
- `test_walk_forward_orchestrator_drops_trailing_3_rows` — the
  last 3 rows of any window have NaN T+3 labels and must not
  appear in the training or test sets. Synthetic test asserts the
  exact dropped row count.

### Integration tests (slow, real FRED data; `@pytest.mark.slow`)

- `test_real_data_label_overlap_with_nber` — fetch FRED 2014-2024,
  generate labels, validate against USREC. Asserts every NBER
  recession period has ≥50% Contraction overlap from our rules.
- `test_real_data_baseline_lr_achieves_floor_metrics` — run walk-
  forward with LR, assert macro-F1 > 0.40 averaged across folds.
- `test_real_data_xgboost_doesnt_overfit_catastrophically` — for
  every fold, XGB train accuracy − test accuracy < 0.30.
- `test_run_walk_forward_writes_all_artifacts` — invokes the
  orchestrator, asserts output_dir contains expected files.

The integration tests require `FRED_API_KEY` (already required by
the FRED client's slow tests). They skip cleanly without it.

## Open questions — resolved

All seven questions resolved by the owner before implementation. Kept
here as a record of decisions made.

1. **Rule thresholds — tune lightly in-design.** Before the
   classifier is built, `scripts/regime_label_diagnostic.py` applies
   the proposed rules to historical FRED data and prints the regime
   label per month for 2014-2025. Owner reviews against (a) COVID
   recession (Feb-Apr 2020 → Contraction or Recovery), (b) 2018-2019
   yield curve inversion → Late-cycle, (c) post-COVID Q3+ → Recovery
   transitioning to Expansion. Thresholds get adjusted if labels look
   wrong; locked if labels look sensible.
2. **Forward-looking labels — confirmed T+3.** See the rewritten
   Label generation strategy section above. The original concurrent
   framing was rejected as a rule-reproducer.
3. **HMM-style regime persistence — skip for v1.** Add as phase 2
   if classifier output is too noisy after release.
4. **Inner CV folds for CalibratedClassifierCV — confirmed cv=3.**
   Stability on small folds beats marginal calibration precision.
5. **Class weighting — confirmed `balanced`.** No SMOTE.
6. **Transition states — confirmed no 5th label.** Calibrated
   probabilities convey transition uncertainty cleanly.
7. **XGBoost adoption bar — confirmed +5pp macro-F1 with ±2pp
   tiebreaker for LR.** Only adopt XGBoost when it clearly wins;
   ties go to the simpler model for interpretability.

## References

- [CLAUDE.md](../../CLAUDE.md) — methodology lock for the regime classifier.
- [fred_client.md](fred_client.md) — feature matrix and vintage handling.
- [walk_forward_cv.md](walk_forward_cv.md) — CV harness consumed by the orchestrator.
- Sahm, C. (2019). "Direct Stimulus Payments to Individuals." —
  inspiration for the unemployment-spike rule (Sahm Rule trigger).
- Estrella, A. & Mishkin, F. (1998). "Predicting U.S. Recessions:
  Financial Variables as Leading Indicators." — yield-curve-based
  recession prediction, antecedent of T10Y3M as a regime feature.
- Niemira, M. & Klein, P. (1994). *Forecasting Financial and
  Economic Cycles.* — the textbook 4-state cycle decomposition.
