# Design: Walk-forward / expanding-window cross-validation harness

**Module**: `src/validation/walk_forward.py`
**Status**: Proposed, pre-implementation
**Owner**: Chris Garcia
**Last updated**: 2026-06-04

## Purpose

Provide the validation harness for both the stock screener and the
regime classifier. Implements expanding-window walk-forward
cross-validation with an embargo between training and test windows.

This is the *only* validation primitive permitted by [CLAUDE.md](../../CLAUDE.md)
for time-series work in this repo — `train_test_split` and k-fold are
explicitly forbidden. The harness exists to make the correct thing the
easy thing.

## Scope: one harness or two?

The two model use cases look superficially different but share a single
date-axis splitting engine.

| | Stock screener | Regime classifier |
|---|---|---|
| Row layout | Long: `(as_of_date, ticker)` | Wide: monthly macro vector |
| Rows per date | ~500 (per S&P 500) | 1 |
| Label horizon | 12 months forward | Concurrent (monthly) |
| Splitting unit | Date | Month |
| Mapping to rows | All rows whose date ∈ window | Direct row indices |

The differences are about how date windows map to row indices. The
*date-window arithmetic* — when does fold N's train end, where does
the embargo run, when does the test start — is identical.

**Decision**: one harness, one date-based engine. The splitter operates
on a sorted `pd.DatetimeIndex` (the unique dates present in the input)
and yields fold metadata expressed in dates. Two thin consumers — one
for long-format data, one for wide — map the date windows to row
indices. The screener consumer is a `.split_long(df, date_col)`
helper; the regime consumer is `.split_wide(df)` (or the sklearn-
compatible `.split(X, y, groups=None)` shim that expects the
DatetimeIndex on `X`).

This avoids duplicating the embargo-arithmetic logic, which is exactly
the part that's easy to get subtly wrong.

## Walk-forward configuration

### Expanding vs sliding window

**Default: expanding**. Locked by CLAUDE.md ("walk-forward /
expanding-window CV") and consistent with the small-sample reality —
~144 monthly observations from 2014, ~420 from 1990. Discarding early
training observations in a sliding window costs sample efficiency we
can't afford.

**Sliding supported as an option** for the regime classifier's optional
sensitivity analysis (does the model's performance change if we cap
training history at 10 years?). Constructor flag: `expanding=True`.

### Embargo period — flagging a CLAUDE.md inconsistency

CLAUDE.md says "1-month embargo between train and test." This is
correct for the regime classifier (labels are concurrent — month M's
regime is determined from month M's macro features) but **wrong for
the stock screener** (labels are 12-month forward returns).

Why 1mo is wrong for the screener:

- Train row at date `T` has label = 12-month forward return =
  function of prices in `(T, T+12mo]`.
- Test row at `T' = T+1mo` has features known at `T'`.
- The train rows in the last 11 months before `T` carry labels whose
  return windows overlap the test period.
- A model that learns "feature pattern X correlates with the realized
  return windows of late-train rows" implicitly trains on test-period
  outcomes.

**The embargo should be at least the label horizon.** For a 12-month
forward label, that means a 12-month embargo. Anything shorter leaks.

**Default per use case**:

- Stock screener: **embargo = 12 months** (matches the label horizon).
- Regime classifier: embargo = 1 month (matches CLAUDE.md, label is
  concurrent).

The harness exposes `embargo` as a constructor parameter; defaults can
be set per use case at the caller site. The CLAUDE.md inconsistency is
flagged in the open questions section and proposed for resolution.

### Test horizon per fold

**Default: 12 months.** Matches the label horizon for the screener
(each test period covers one full forward-label cycle, evaluated
non-overlappingly). Also a reasonable cadence for the regime
classifier — 12 months gives ~12 regime predictions per fold to
evaluate against the rules-based labels.

Shorter test horizons (e.g., 3 months) give more folds with overlapping
test predictions; this is statistically fine but harder to summarize.

### Step size

**Default: 12 months.** With 12mo test horizons, this means non-
overlapping test sets — each calendar period is tested exactly once.
Cleaner evaluation than overlapping (which would require careful
aggregation across folds when computing metrics like Sharpe).

Configurable via `step` parameter; sub-12mo steps are allowed but the
caller is responsible for handling overlap during evaluation.

### `step < test_period` — overlapping test sets warning

When `step < test_period`, consecutive folds have overlapping test
windows: fold N's test data includes some dates that are also in
fold N+1's test data. The arrangement is mathematically valid (every
fold is still walk-forward-correct in isolation) but **fold scores
become correlated** — they share input rows, so they are not
independent samples of model performance.

Aggregation methods that assume independence will produce misleading
results:

- Simple averaging of fold scores understates the true variance.
- t-tests or confidence intervals computed across folds are wrong by
  approximately the overlap-induced correlation factor.
- Bootstrap resampling of folds (e.g., for fold-CV uncertainty) needs
  block-bootstrap, not simple resample.

The constructor accepts `step < test_period` but **logs a clear
WARNING at construction time** so the caller knows what they're
signing up for. The warning fires once per `WalkForwardCV()` instance
(not once per fold) and includes the exact `step` and `test_period`
values plus the aggregation caveats above.

Conservative default `step == test_period` avoids this entirely; the
warning never fires for the documented defaults.

### Putting it together: expected fold count

Given the screener config (train_period unbounded for expanding,
embargo 12mo, test 12mo, step 12mo) over 2014-01 to 2025-12:

- Fold 0: train 2014-01 → ?, test 2015-?
- The earliest viable fold needs at least ~3 years of training data
  (sample-size floor), so first train_end ≈ 2016-12.
- Then: train_end = 2016-12, embargo through 2017-12, test 2018-2018.
- Subsequent folds advance train_end by step=12mo.
- Final fold's test_end must be ≤ data_end (2025-12) ⇒ last train_end ≈ 2023-12.
- Total folds: ~8.

Eight folds is on the low side but reasonable for the data we have.
The regime classifier with embargo=1mo and ~420 monthly observations
from 1990 yields ~30 folds — much healthier.

## Vintage handling

The harness produces fold metadata; it does not fetch data. The
consumer of each fold is responsible for fetching features with the
appropriate vintage.

**Per-fold vintage = `test_start`** is the documented v1 default.
Matches the prior owner decision in [fred_client.md](fred_client.md)
("for a training fold, the vintage date is the fold's prediction date,
not the fold's training-end date"). The fold object exposes
`vintage_date: dt.date = test_start` directly so consumers don't have
to re-derive it.

**Known limitation: per-fold not per-row.** The stock screener's
test period spans 12 months; ideally each test row at `as_of_date = T'`
would use vintage `T'`, but the v1 harness uses a single
`vintage_date = test_start` for the whole fold. This means test rows
in months 6-12 of the test period are using ~6-month-old vintage data.
For macro features the revision-bias from this is small; for
fundamentals it could matter more.

**Per-row vintage is a phase-2 upgrade dependent on the fundamentals
client.** When `src/data_sources/fundamentals_client.py` is built
(Sharadar SF1 or equivalent), it will need vintage support analogous
to FRED's `realtime_start`/`realtime_end`. The per-row vintage logic
belongs there — fold metadata gains a `vintage_dates: pd.Series`
field mapping each test row to its specific as-of date, and the
fundamentals client honors that per-row. The walk-forward harness
just passes the dates through; it has no reason to know whether the
consumer is using per-fold or per-row vintage. So the v1 harness API
won't need to change to support phase 2 — only the
`Fold.vintage_date` scalar gets supplemented with `Fold.vintage_dates`
optional series. No breakage to v1 consumers.

The fold metadata includes `vintage_date` and consumers are encouraged
to fetch features using it explicitly via, e.g.,
`get_features_matrix(..., vintage_date=fold.vintage_date)`.

## Split semantics

Each fold yields a `Fold` object (dataclass) with these fields:

| field | type | meaning |
|---|---|---|
| `fold_id` | int | zero-indexed |
| `train_start` | dt.date | inclusive |
| `train_end` | dt.date | inclusive |
| `embargo_start` | dt.date | = train_end + 1 day |
| `embargo_end` | dt.date | exclusive (= test_start) |
| `test_start` | dt.date | inclusive |
| `test_end` | dt.date | inclusive |
| `vintage_date` | dt.date | = test_start |
| `train_indices` | np.ndarray[int] | row positions in the input |
| `test_indices` | np.ndarray[int] | row positions in the input |
| `n_train` | int | len(train_indices) |
| `n_test` | int | len(test_indices) |

**Long-format mapping** (stock screener): given a DataFrame with a
`date` column, `train_indices = np.where(df.date.between(train_start, train_end))[0]`,
likewise for test.

**Wide-format mapping** (regime classifier): given a DataFrame with a
DatetimeIndex, the same `.between` arithmetic on the index.

Invariants asserted on every fold (in `__post_init__`):

- `train_start ≤ train_end < embargo_start ≤ embargo_end ≤ test_start ≤ test_end`
- `(test_start - train_end).days ≥ embargo_days` (embargo respected)
- `len(set(train_indices) & set(test_indices)) == 0` (no row overlap)
- `n_train > 0 and n_test > 0` (both windows non-empty, otherwise the
  fold isn't yielded — see Edge cases)

## sklearn compatibility

**Decision: native rich interface + thin sklearn shim.**

The primary `WalkForwardCV.split()` method yields rich `Fold` objects.
For interoperability with `cross_val_score`, `GridSearchCV`, etc., a
companion method `split_indices(X, y=None, groups=None)` matches the
sklearn `BaseCrossValidator.split` signature and yields
`(train_indices, test_indices)` tuples — losing the metadata but
gaining sklearn ecosystem access.

**Why not inherit from `BaseCrossValidator`**:

- sklearn's `split(X, y, groups)` discards everything except indices.
  We need the `vintage_date` to flow through; the sklearn signature
  cannot carry it.
- The screener's actual evaluation is custom (Sharpe, decile spread,
  hit rate, max DD, transaction cost-aware backtest) and will not use
  `cross_val_score`. The sklearn surface is needed mainly for
  hyperparameter tuning of the inner classifier, which is a narrow
  use case.

So the inheritance buys little. The shim method covers the realistic
sklearn use cases without coupling us to the base class.

## Public API

```python
@dataclass(frozen=True)
class Fold:
    """A single walk-forward fold. See split semantics in design doc."""
    fold_id: int
    train_start: dt.date
    train_end: dt.date
    embargo_start: dt.date
    embargo_end: dt.date
    test_start: dt.date
    test_end: dt.date
    vintage_date: dt.date
    train_indices: np.ndarray
    test_indices: np.ndarray

    @property
    def n_train(self) -> int: ...

    @property
    def n_test(self) -> int: ...

    def __post_init__(self) -> None:
        """Assert the date-ordering and embargo invariants."""


class WalkForwardCV:
    """Walk-forward / expanding-window CV with embargo.

    Args:
        train_period: minimum training window length in months. For
            expanding=True, this is the floor; the train window grows
            each fold. For expanding=False, this is the fixed window.
        test_period: test window length in months. Default 12.
        embargo: gap between train_end and test_start, in months.
            Must be >= the label horizon. Default 12 for the screener
            (12-month forward labels); set to 1 for the regime
            classifier (concurrent labels).
        step: months to advance per fold. Default 12 — non-overlapping
            test sets.
        expanding: if True, train window grows each fold (default).
            If False, sliding window of length train_period.
    """

    def __init__(
        self,
        train_period: int,
        test_period: int = 12,
        embargo: int = 12,
        step: int = 12,
        expanding: bool = True,
    ): ...

    def split(self, dates: pd.DatetimeIndex) -> Iterator[Fold]:
        """Primary interface. Yields Fold objects with full metadata.

        `dates` is the unique sorted DatetimeIndex of observations in
        the input (the date axis along which we walk forward). For
        long-format input, pass `df["date"].drop_duplicates().sort_values()`
        and use `.split_long()` to expand to row indices.
        """

    def split_long(
        self, df: pd.DataFrame, date_col: str = "as_of_date"
    ) -> Iterator[Fold]:
        """Long-format helper. Each fold's train_indices and
        test_indices include every row whose date falls in the
        respective window."""

    def split_indices(
        self, X, y=None, groups=None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """sklearn-compatible shim. Yields (train_idx, test_idx).
        Vintage_date is dropped — use `split()` if you need it."""

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """sklearn-compatible: total fold count for the given input."""

    def summary(self, dates: pd.DatetimeIndex) -> str:
        """ASCII table of fold structure. See Visualization."""
```

## Edge cases

**Data range too short for `train_period`**: raise `ValueError` with
a message stating the required vs available span. Hard fail —
silently dropping folds would mask a configuration mistake.

**Final fold's `test_end` exceeds available data**: the fold is
dropped (not yielded), and a single INFO log line names the dropped
fold's hypothetical window. This is the expected end-of-data
condition; raising would be hostile.

**Gaps in the input date index** (e.g., missing months): the engine
operates on dates, not row positions, so gaps don't cause an error.
Folds whose train or test window happens to fall entirely in a gap
end up with `n_train == 0` or `n_test == 0` and are dropped (with
INFO log). Folds straddling gaps just include fewer rows.

**Embargo larger than test_period or step**: configuration error,
raise `ValueError`. Embargo > step means consecutive folds would
overlap in time (test_N starts before train_N+1 ends), violating the
walk-forward semantics.

**`expanding=False` with `train_period > data_span`**: raise
`ValueError` — no folds possible with a sliding window larger than
the data.

**Non-monotonic input dates**: raise `ValueError`. The walk-forward
semantics require time ordering; we don't auto-sort because doing so
silently could hide a data-pipeline bug.

**Single observation per date vs many** (regime vs screener): not an
edge case in the harness itself — the date engine doesn't care. Only
matters in the consumer's index mapping.

## Validation against known good behavior

`tests/test_walk_forward.py`. Unit tests use synthetic monthly date
ranges; no fixtures from FRED or SP500 needed.

### Fold arithmetic invariants

- `test_embargo_respected` — for every fold, `(test_start - train_end).days >= embargo_months * 28` (28-day floor for any month).
- `test_no_train_test_index_overlap` — `set(train_indices) ∩ set(test_indices) == ∅`.
- `test_vintage_equals_test_start` — `fold.vintage_date == fold.test_start` for every fold.
- `test_date_ordering` — every fold's date fields satisfy the documented monotonic chain.
- `test_fold_id_is_zero_indexed_and_sequential` — fold_id values are `range(n_folds)`.

### Configuration → fold count

- `test_expanding_fold_count_matches_expected` — 2014-01 to 2025-12 monthly index, train_period=36, embargo=12, test=12, step=12 → exactly 8 folds. Known calculation, easy to verify by hand.
- `test_sliding_fold_count_matches_expected` — same input, expanding=False, train_period=60 → fewer folds; assert exact count.
- `test_step_smaller_than_test_creates_overlapping_test_periods` — step=6, test=12 → consecutive test periods overlap by 6mo. Verify mathematically.

### Window growth semantics

- `test_expanding_train_start_constant` — every fold's `train_start` equals the first fold's `train_start`.
- `test_expanding_train_end_advances_by_step` — `fold[i+1].train_end == fold[i].train_end + step_months`.
- `test_sliding_train_start_advances_by_step` — `fold[i+1].train_start == fold[i].train_start + step_months`.
- `test_sliding_train_window_length_constant` — `(fold.train_end - fold.train_start).days` is constant across folds (within month-length variance).

### Long-format mapping

- `test_split_long_includes_all_rows_in_window` — synthetic long-format DataFrame with 5 tickers per month. Assert fold's train_indices contains exactly 5 × train_months rows.
- `test_split_long_test_indices_disjoint_from_train` — set arithmetic on the long-format mapping.
- `test_split_long_date_col_argument` — default is `as_of_date`; works with custom column name too.

### Edge cases

- `test_too_short_data_range_raises_clear_error` — data span < `train_period + embargo + test_period` → ValueError with both spans in the message.
- `test_final_fold_truncated_drops_fold` — input ends mid-test-window; last fold is dropped (not yielded with truncated test) and an INFO log records the drop.
- `test_non_monotonic_dates_raises` — input dates out of order → ValueError.
- `test_embargo_exceeds_step_raises_config_error` — ValueError on construction time (not at split time).
- `test_zero_train_or_test_indices_fold_dropped` — fold whose window happens to be entirely in a data gap is dropped with INFO log.

### sklearn shim

- `test_split_indices_matches_split_indices_field` — for the same input, `split_indices` yields the same `(train_idx, test_idx)` pairs as `[(f.train_indices, f.test_indices) for f in split()]`.
- `test_get_n_splits_matches_actual_fold_count` — `get_n_splits(X)` matches `len(list(split(X)))`.
- (Optional) `test_works_with_sklearn_cross_val_score` — `cross_val_score(SomeClassifier(), X, y, cv=WalkForwardCV(...))` runs without TypeError. Skipped in CI to avoid sklearn dep coupling at the test level; left for an integration test.

## Visualization helper

**Decision: include.** ASCII summary table only — no matplotlib, no
plotly. The benefit is low-cost diagnostic ("did I configure 8 folds
or 80?") and self-documenting test failures (`assert (...), splitter.summary(dates)`).

Proposed output:

```
WalkForwardCV: 8 folds, expanding, train_period=36mo, embargo=12mo, test=12mo, step=12mo

Fold | Train start | Train end  | Embargo end | Test start | Test end   | n_train | n_test
   0 | 2014-01-31  | 2017-01-31 | 2018-01-31  | 2018-01-31 | 2018-12-31 |    1825 |    500
   1 | 2014-01-31  | 2018-01-31 | 2019-01-31  | 2019-01-31 | 2019-12-31 |    2325 |    500
   ...
   7 | 2014-01-31  | 2024-01-31 | 2025-01-31  | 2025-01-31 | 2025-12-31 |    5825 |    500
```

Implementation is ~30 lines of f-string formatting. Worth the tiny
cost for the substantial debugging benefit.

ASCII timeline diagram (the "T T T E T T" style I mocked in the open
questions) is *not* included — adds visual noise without informational
value over the table.

## Open questions

1. **Embargo for the stock screener — push back on CLAUDE.md.** The
   "1-month embargo" lock in CLAUDE.md is correct for the regime
   classifier but wrong for the screener's 12-month forward labels.
   Proposing the embargo become a per-use-case parameter with these
   defaults: 12 months for the screener, 1 month for the regime
   classifier. I'd also propose updating CLAUDE.md to reflect this
   (the methodology section was written before this design pass).
   Confirm? If you actually want 1-month for the screener (e.g.,
   you accept some leakage to get more folds), say so explicitly.

2. **Default test horizon = 12 months.** Confirm? Alternatives are 6
   (more folds, overlapping) or 24 (fewer folds, longer evaluation
   periods). I think 12 is the right default for both use cases.

3. **Default step = 12 months.** Confirm? Smaller steps give more
   folds but overlapping test sets, which complicates evaluation
   aggregation.

4. **sklearn compatibility via shim, not inheritance.** Confirm? The
   shim's only loss vs. inheritance is `cv=WalkForwardCV(...)` in
   sklearn API calls, which the shim still supports — it's just that
   we don't get the `isinstance(cv, BaseCrossValidator)` check
   somewhere downstream. I'm not aware of anywhere that matters for
   our use case.

5. **Per-fold vintage vs per-row vintage.** v1 uses per-fold
   (`vintage_date = test_start`). For the screener's 12-month test
   window, this means test rows 6-12 months in use 6-month-old
   vintage. Acceptable simplification, or do we need per-row vintage
   in v1? Per-row vintage is much more complex (vintage_date as a
   row attribute that the consumer threads into per-row data fetches),
   and the fundamentals client doesn't exist yet — pushing this to
   when we build that client seems right.

6. **Minimum training period as a floor for expanding.** For
   `expanding=True`, do we want a `min_train_period` parameter that
   ensures fold 0 has at least N months of training data? Useful to
   avoid degenerate "train on 3 months, test on 12" folds at the
   start of the data range. Proposing `min_train_period =
   train_period` (i.e., always at least train_period months before
   the first test). Defaults are sensible; this is more about API
   shape — should it be a separate param or just baked in?

7. **Final-fold truncation policy.** When the data range ends
   mid-test-window, the current proposal is "drop the fold." The
   alternative is "yield with a shortened test window." Drop is
   conservative; shorten gives one more fold of evaluation. Default
   to drop, with a `strict_period: bool = True` parameter to allow
   shorten if the caller wants it?

## References

- [CLAUDE.md](../../CLAUDE.md) — methodology lock for walk-forward CV.
- [fred_client.md](fred_client.md) — vintage handling and the
  per-fold vintage convention.
- [sp500_constituents.md](sp500_constituents.md) — the long-format
  input shape the screener splitter consumes.
- [Prado, *Advances in Financial Machine Learning*](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) —
  chapter 7 (embargoing) is the textbook reference for the embargo
  policy proposed here. The CLAUDE.md 1-month default predates this
  discussion.
