# Design: Cached FRED API wrapper for macro/credit series

**Module**: `src/data_sources/fred_client.py`
**Status**: Proposed, pre-implementation
**Owner**: Chris Garcia
**Last updated**: 2026-06-02

## Purpose

Provide a cached, vintage-aware wrapper around the FRED API for the macro
and credit series the regime classifier and stock screener consume.

The regime classifier needs ~20 years of monthly macro cross-sections to
fit four-state regime labels. The stock screener may use a subset of the
same series as conditioning features. Both must avoid look-ahead bias:
training data must use *as-of-date* values, not latest-revision values,
because macro series like UNRATE and PAYEMS get revised months after
first publication.

## Source

[FRED — Federal Reserve Economic Data](https://fred.stlouisfed.org/),
maintained by the St. Louis Fed. Free API with [registration](https://fred.stlouisfed.org/docs/api/api_key.html);
rate limit ~120 requests/minute, no daily cap. Stable infrastructure
since 2008; series IDs and the API have been backward-compatible for
years.

Access via `fredapi==0.5.2` (already in [requirements.txt](../../requirements.txt)).
The library handles auth, query construction, and DataFrame parsing —
thin enough that swapping for raw `requests` later would be trivial if
its abandonment ever became a concern.

## Authentication

API key loaded from `FRED_API_KEY` environment variable. Loaded via
`python-dotenv` (already a dependency) on first call to any function
that hits FRED. Loading is **lazy**, not at module import, so the
module remains importable without a key (important for unit tests that
mock the API).

`validate_api_key()` is a public helper that explicitly checks key
presence and, if requested, makes a minimal API call (e.g.,
`get_series("UNRATE", limit=1)`) to confirm the key is accepted.

If `FRED_API_KEY` is missing or empty, raise `EnvironmentError` with
exact message:

```
EnvironmentError: FRED_API_KEY is missing or empty. Copy .env.example
to .env and fill in your key. Get a key at
https://fred.stlouisfed.org/docs/api/api_key.html. See
docs/design/fred_client.md for setup details.
```

If the key is present but FRED rejects it (HTTP 400/403 with
"API key" in the body), raise `ValueError` with a message that
distinguishes "missing" from "invalid" — the latter usually means
the key was rotated or typo'd.

## Series inventory

**20 features + USREC ground truth.** Sized to cover the four-regime
taxonomy without overfitting on macro signals — at ~400 monthly
observations spanning 1990-present, even 20 features is aggressive
and will need regularization.

### Yield curve (rate spreads — recession leading indicator)

| Series | Name | Freq | Units | Why |
|---|---|---|---|---|
| `DGS10` | 10-Year Treasury Constant Maturity Rate | D | % | Long-end rate benchmark |
| `DGS2` | 2-Year Treasury Constant Maturity Rate | D | % | Short-end rate |
| `T10Y2Y` | 10Y - 2Y Treasury Spread | D | % | Classic recession indicator (inverted = recession warning) |
| `T10Y3M` | 10Y - 3M Treasury Spread | D | % | Fed's preferred curve metric per the literature |

### Credit (risk premium — recession indicator)

| Series | Name | Freq | Units | Why |
|---|---|---|---|---|
| `NFCICREDIT` | Chicago Fed NFCI Credit Subindex | W | index | Credit stress proxy with full 1971-present coverage |
| `BAA10Y` | Moody's Baa Corp Yield − 10Y Treasury | D | % | Investment-grade credit spread |

**Why not ICE BofA HY OAS?** The first version of this design used
`BAMLH0A0HYM2` (ICE BofA US High Yield OAS) as the HY credit-stress
signal. In late 2024 ICE Data Indices restricted FRED's public access
to historical ICE BofA data, truncating `BAMLH0A0HYM2` and every
sibling `BAML*` series to a rolling ~3-year window — useless for a
1990-present training corpus. `NFCICREDIT` (Chicago Fed NFCI Credit
Subindex) replaces it: full 1971-present coverage, Fed-native
maintenance (no licensing-disclosure risk), and the Chicago Fed
explicitly weights HY spreads into the subindex construction. Sign
convention is preserved — positive readings mean tighter than average
credit, the same direction as a widening OAS.

### Activity / Labor (coincident + leading)

| Series | Name | Freq | Units | Why |
|---|---|---|---|---|
| `UNRATE` | Unemployment Rate | M | % | Coincident; Sahm Rule trigger for recessions |
| `ICSA` | Initial Jobless Claims | W | count | High-frequency leading indicator |
| `PAYEMS` | Total Nonfarm Payrolls | M | thousand | Coincident; primary growth signal |
| `INDPRO` | Industrial Production Index | M | index | Coincident manufacturing-sector signal |
| `HOUST` | Housing Starts | M | thousand SAAR | Early-cycle leading indicator |

### Rates / Money

| Series | Name | Freq | Units | Why |
|---|---|---|---|---|
| `FEDFUNDS` | Effective Federal Funds Rate | M | % | Fed policy stance |
| `M2SL` | M2 Money Stock | M | $billion | Liquidity / monetary aggregate |

### Inflation

| Series | Name | Freq | Units | Why |
|---|---|---|---|---|
| `CPIAUCSL` | CPI All Urban Consumers | M | index | Headline inflation |
| `CPILFESL` | Core CPI | M | index | Inflation ex food/energy |
| `PCEPILFE` | Core PCE Price Index | M | index | Fed's preferred inflation measure |

### Market internals

| Series | Name | Freq | Units | Why |
|---|---|---|---|---|
| `VIXCLS` | CBOE Volatility Index | D | level | Fear gauge / option-implied vol |
| `DTWEXBGS` | Trade-Weighted Broad Dollar Index | D | index | Dollar strength, regime input |
| `DCOILWTICO` | WTI Crude Oil Spot Price | D | USD/bbl | Commodity / inflation transmission, geopolitical risk |
| `UMCSENT` | Michigan Consumer Sentiment | M | index | Survey-based leading indicator |

### Ground truth (not a feature — for label construction only)

| Series | Name | Freq | Units | Why |
|---|---|---|---|---|
| `USREC` | NBER-based Recession Indicator | M | 0/1 | Validation against rules-based regime labels per CLAUDE.md |

> ⚠️ **USREC is target data — never include in the feature matrix.
> Accidental inclusion would cause label leakage in the regime classifier.**

`USREC` is fetched here for convenience and consumed by
`regime_labels.py`, which is the only module allowed to read it into
model inputs. `get_features_matrix()` excludes USREC from its default
column set; a test asserts the exclusion holds.

All sources are St. Louis Fed (FRED) or upstream FRED-republished
(BLS, BEA, NBER, ICE, Moody's, Census, Michigan).

`REGIME_SERIES: dict[str, dict]` in the module holds the full inventory
with frequency, units, source, and a docstring describing usage. Single
source of truth, importable by `macro_features.py` and tests.

## Vintage handling

FRED supports point-in-time queries via `realtime_start` and
`realtime_end` URL params. When specified, FRED returns observations
as they were *known* on that date — i.e., excludes later revisions.

**Policy**:

- **Training queries**: vintage-aware. Pass `vintage_date=<as_of>` so
  the returned series reflects only what was published on or before
  that date. This is critical for series like UNRATE and PAYEMS, which
  get revised for months after first publication. Walk-forward CV
  folds set `vintage_date` to the fold's training cutoff.
- **Inference queries**: latest. `vintage_date=None` (default) returns
  the current revised values, which is what a present-day prediction
  uses.

**API mechanics**:

- Latest: `fred.get_series(series_id, observation_start=..., observation_end=...)`.
  No realtime params; returns latest revisions only.
- Vintage: `fred.get_series(series_id, observation_start=..., observation_end=...,
  realtime_start=vintage_date, realtime_end=vintage_date)`. **Both
  realtime params must be the same date.**

**Why both, not just `realtime_end`.** FRED's realtime semantics are
subtle:

- Setting only `realtime_end=X` returns the *cumulative* set of
  observation/revision pairs whose revision date ≤ X. For a frequently
  revised series like UNRATE this returns multiple rows per observation
  date (one per revision), which is the full history-of-revisions view,
  not a point-in-time snapshot.
- Setting both `realtime_start=X` and `realtime_end=X` returns only
  observation/revision pairs whose realtime period *contains* X — i.e.,
  the single revision of each observation that was in effect on that
  exact date. This is the "as known on date X" snapshot we want.

We deliberately do **not** use fredapi's higher-level
`get_series_as_of_date()` because its semantics differ subtly across
versions and depend on filtering an all-releases query in Python; the
both-params-equal direct call goes straight to FRED's authoritative
behavior.

**Cost**: vintage queries are slightly slower because FRED has to look
up the appropriate revision. Caching by `(series_id, vintage_date)` is
essential for walk-forward CV, which would otherwise re-fetch the same
vintage hundreds of times.

### Screener vintage policy

Vintage-when-training, latest-for-inference (parallel to the regime
model).

**Important**: for a training fold, the vintage date is the **fold's
prediction date**, not the fold's training-end date. This reflects
what data was actually available to a real-time decision-maker *on
the day the prediction was made*. A model trained through
2018-12-31 to predict 2019-01-31 outperformance must use macro
features as known on 2019-01-31 (the prediction date), not as known
on 2018-12-31 (the training-data cutoff). The vintage is tied to the
inference moment, not the training corpus.

### Missing series at vintage

Some FRED series start after 1990 (e.g., TIPS-related series begin
~1999; BAMLH0A0HYM2 begins 1997; T10Y3M begins 1982). For early
vintage queries, the series may have zero observations on or before
the requested date.

**Policy**: return an empty `pd.Series` and log at WARNING. Do **not**
raise. This is an expected condition during walk-forward CV across
early training periods — the regime model will see NaN columns for
those series in those folds and `macro_features.py` will handle the
imputation strategy.

A mocked unit test asserts the empty-series path does not raise and
that the warning is logged with the series_id and vintage_date in the
message.

**Pushback**: should we also support `realtime_start` for a range query
that returns all known vintages? No. The screener only needs as-of-date
snapshots; collecting full vintage history would balloon storage with
no model benefit. If we ever want to analyze revision behavior, we'll
add it then.

## Frequency harmonization

Series mix daily (D), weekly (W), and monthly (M). The regime model and
the screener's macro conditioning both work in monthly cross-sections.

**Default policy: as-of-month-end forward-fill.** Within each calendar
month, take the most recent observation whose date is ≤ the last
business day of the month. For monthly series this is just the month's
value; for higher-frequency series it's the latest reading available
by month-end.

Rationale:
- Matches what a real-time trader would have seen at month-end.
- Symmetric with `members_on(date)` semantics in
  [sp500_membership.py](../../src/data_sources/sp500_membership.py) —
  "the latest known value as of date".
- Period averages (e.g., monthly average of daily DGS10) drift
  slightly from any single end-of-month observation and are less
  defensible for predictive features.

**Per-series override**: `REGIME_SERIES[series_id]` can declare
`"resample": "mean"` for cases where average makes more sense
(e.g., `FEDFUNDS` effective rate is most naturally interpreted as a
monthly average, though the M-frequency FRED series already does this
internally — included as an example of where override would be useful
if we ever switch to `DFF` daily).

The default for all 20 series above is `"resample": "last"` (as-of).

## Caching strategy

Per-series parquet, **not** a single frozen committed file like
[sp500_constituents.md](sp500_constituents.md). The argument:

- FRED is St. Louis Fed infrastructure — high uptime, free, stable
  API for 15+ years. The single-maintainer takedown risk that
  motivated the change-log frozen copy does not apply.
- Vintage data is *deterministic per (series_id, vintage_date)*: once
  cached, the cache file's contents are immutable. A teammate
  recreating the cache from the API will get bit-identical results.
- Latest data is intentionally fluid; freezing it would defeat the
  purpose.

So: cache to disk for speed and offline work, but do not commit. The
cache is regenerable from the API at any time.

### Layout

```
data/raw/fred/
  observations/
    <series_id>__latest.parquet                   # mutable, TTL 24h
    <series_id>__vintage_<YYYY-MM-DD>.parquet     # immutable, cache forever
  metadata/
    series_info_<series_id>.json                  # title, units, notes
  api_log.jsonl                                   # one line per API call, for rate-limit forensics
```

All under `data/raw/fred/`, all gitignored (covered by the existing
`data/raw/*` exclude — no override needed, in contrast with
`data/raw/sp500_change_log/`).

### TTL policy

- `__latest` files: TTL 24 hours. Refresh on next read after expiry.
  Matches CLAUDE.md's "Default TTL 24 hours for daily data."
- `__vintage_<date>` files: no TTL. Vintage queries are deterministic;
  the cached file is the ground truth.
- `metadata/` JSON: TTL 7 days. Series metadata changes rarely.

### Cache invalidation

A cache file is considered fresh only when **both** invariants hold:

1. **TTL** — `__latest` files within 24 hours; `__vintage_<date>`
   files are immutable, no TTL.
2. **Range coverage** — the cached parquet's index range must cover
   the request's `[start, end]` window. If the requested `start` is
   earlier than the cached minimum date, or the requested `end` is
   later than the cached maximum date, the cache is stale and
   `get_series` refetches the full requested range.

The range-coverage invariant matters because the cache is keyed by
`(series_id, vintage_date)` only — not by date range. Without the
coverage check, an earlier narrow-range fetch (e.g., UNRATE
2020-01 to 2020-12 from a test fixture or one-off REPL) would
populate the cache, and a later wider request (e.g., 2014 to 2025)
would hit the cache within TTL and silently return a Series missing
all data outside the original range. The forward-filling in
`get_features_matrix` would then propagate stale boundary values
across all later month-ends, looking superficially right but being
silently wrong.

This was a real bug, caught only because the regime label
diagnostic eyeballed the output and noticed UNRATE pinned at 6.7%
from 2020-12 through 2025-12. Regression test:
`test_get_series_cache_invalidated_when_narrower_than_request`.

If `__latest` is stale and the network is unreachable, log WARNING
and return the stale cached value (offline batch-degradation
pattern). This applies to TTL-stale caches only — range-stale
caches must always refetch, since serving a known-incomplete range
is a correctness violation, not a freshness inconvenience.

If `__vintage` files would somehow disagree across machines
(shouldn't happen — FRED's vintage data is immutable, by
definition), a future audit-mode flag could hash-compare across
re-fetches. Out of scope for v1.

## Public API

```python
def validate_api_key(probe: bool = False) -> None:
    """Confirm FRED_API_KEY is set. If probe=True, make a minimal API
    call to confirm the key is accepted by FRED.

    Raises EnvironmentError if the key is missing/empty.
    Raises ValueError if the key is present but rejected (probe=True).
    """


def get_series(
    series_id: str,
    start: dt.date | None = None,
    end: dt.date | None = None,
    vintage_date: dt.date | None = None,
) -> pd.Series:
    """Fetch a single FRED series.

    start, end: observation date range. None = full available history.
    vintage_date: if set, return values as known on that date
        (vintage-aware, for training). If None, return latest revisions
        (for inference).

    Cached: latest with TTL 24h, vintages forever.

    Raises ValueError for unknown series_id, EnvironmentError for
    missing API key. Empty range returns empty Series (logged INFO,
    not an error).
    """


def get_features_matrix(
    series_ids: list[str] | None = None,
    start: dt.date = dt.date(1990, 1, 1),
    end: dt.date | None = None,
    frequency: str = "M",
    vintage_date: dt.date | None = None,
) -> pd.DataFrame:
    """Fetch multiple series and align to a common frequency.

    series_ids: default is all entries in REGIME_SERIES *except* USREC
        (USREC is ground truth, not a feature).
    frequency: 'M' (month-end) is the only supported value in v1.
        Other frequencies can be added later if needed.
    vintage_date: applied uniformly to every series.

    Per-series resampling rule comes from REGIME_SERIES[id]["resample"];
    default is "last" (as-of-month-end forward-fill).

    Returns DataFrame indexed by month-end DatetimeIndex with one column
    per series_id. Missing values are forward-filled within reason
    (warn if a series has > 6 months of missing data; that may indicate
    a series that doesn't cover the requested range).
    """


def list_series() -> pd.DataFrame:
    """Return REGIME_SERIES as a DataFrame for inspection — id, name,
    frequency, units, source, description. Convenience helper."""
```

`REGIME_SERIES` is a module-level dict (the 20-entry inventory above),
importable by `macro_features.py` and tests.

USREC is in `REGIME_SERIES` so the inventory is complete, but
`get_features_matrix` excludes it by default to prevent label leakage.
`regime_labels.py` reads USREC via `get_series("USREC")` directly.

## Error handling

CLAUDE.md's "no silent except" rule applies. Per-class policy:

- **Missing API key** (`EnvironmentError`): hard fail at first API call.
  Message points at `.env.example`.
- **Rejected API key** (FRED returns 400/403 with "API key" body):
  raise `ValueError` distinguishing it from the missing case.
- **Unknown series ID** (FRED returns 400 with "series does not
  exist"): hard fail. This is a code bug — the series ID is wrong.
- **Empty observation range** (series exists but no data in
  `[start, end]`): not an error. Return an empty Series, log at INFO.
- **Series doesn't exist at vintage** (series start date is after
  `vintage_date`): not an error. Return an empty Series, log at
  WARNING — expected during walk-forward CV across early training
  periods. The WARNING includes both `series_id` and `vintage_date`
  so the log is greppable.
- **Rate limit (HTTP 429)**: exponential backoff with jitter — sleeps
  of 4s, 16s, 64s, then hard fail. Log at WARNING on each retry. Our
  expected request rate is well under the 120/min limit, so hitting
  this means something pathological (e.g., a tight loop bypassing the
  cache).
- **Network errors (timeout, DNS, connection reset)**: retry once after
  4 seconds, then hard fail.
- **Cache file corruption** (parquet read fails): log WARNING, delete
  the corrupted file, re-fetch from API.

Bulk operations (`get_features_matrix`) follow the documented batch-
degradation pattern: a single series failure logs at WARNING and is
recorded in the returned DataFrame as an all-NaN column; > 25% of
requested series failing aborts the call with a hard error.

## Validation

Known-value spot checks for integration tests. Each chosen because the
value is well-documented historically and would not survive a parsing
or unit bug:

- **T10Y2Y on 2019-08-14** ≈ -0.04% — the famous yield curve inversion
  flagged as a recession warning, widely covered. Assert value < 0.
- **UNRATE for April 2020** = 14.7% — the COVID unemployment spike,
  largest single-month jump in BLS history. Assert |value - 14.7| < 0.1.
- **USREC** = 1 for months in [2020-02, 2020-04] inclusive — NBER's
  COVID recession dates. Assert exactly.
- **VIXCLS on 2020-03-16** ≈ 82.69 — VIX intraday all-time closing
  high during the COVID crash. Assert |value - 82.69| < 0.5.

Each assertion fails if FRED is unreachable, the series ID is wrong,
parsing strips a digit, or unit handling is broken. Tight enough to
catch real bugs, loose enough to survive future revisions to the
historical record (which are rare for these series but possible).

## Test plan

`tests/test_fred_client.py`.

### Unit tests (mocked FRED responses; fast; CI)

- `test_validate_api_key_missing_raises_with_message` — exact message
  text per the spec above.
- `test_validate_api_key_present_no_probe_passes` — env var set, no
  network call attempted.
- `test_get_series_latest_passes_correct_params` — verify the
  underlying call uses `observation_start`/`observation_end` but not
  `realtime_end`.
- `test_get_series_with_vintage_sets_both_realtime_params` — verify
  vintage query passes both `realtime_start=vintage_date` and
  `realtime_end=vintage_date`. Critical for correct point-in-time
  semantics; `realtime_end`-only would return revision history.
- `test_get_series_unknown_id_raises_clear_error` — FRED 400 response
  is translated to ValueError with the series ID in the message.
- `test_get_series_empty_range_returns_empty` — INFO-logged, not raised.
- `test_get_series_missing_at_vintage_returns_empty_with_warning` —
  empty fredapi response for a vintage-aware query returns an empty
  Series and logs WARNING containing both `series_id` and
  `vintage_date`. Does not raise.
- `test_get_series_rate_limit_retries_with_backoff` — mock returns 429
  twice then 200; verify backoff sleeps and final value returned.
- `test_get_series_rate_limit_persistent_failure_raises` — three 429s
  in a row → hard fail.
- `test_cache_hit_skips_api_call` — second call to identical
  `(series_id, vintage_date)` does not touch the network.
- `test_cache_latest_expires_after_24h` — monkeypatch the cache file's
  mtime to >24h ago; next read triggers a fetch.
- `test_cache_vintage_never_expires` — monkeypatch mtime to 30 days ago;
  cache is still used.
- `test_cache_offline_returns_stale_with_warning` — stale cache +
  network failure returns stale data and logs WARNING.
- `test_get_features_matrix_aligns_to_month_end` — daily DGS10 +
  monthly UNRATE both produce one value per month-end business day.
- `test_get_features_matrix_default_excludes_usrec` — USREC is in
  REGIME_SERIES but not in the default columns.
- `test_get_features_matrix_partial_failure_logs_and_continues` —
  one series failing yields an all-NaN column and logs WARNING.
- `test_get_features_matrix_majority_failure_raises` — >25% failing
  triggers hard abort.

### Integration tests (real FRED API, `@pytest.mark.slow`, skipped in CI)

- `test_real_api_t10y2y_inverted_august_2019` — fetch T10Y2Y, assert
  2019-08-14 value < 0.
- `test_real_api_unrate_covid_spike` — fetch UNRATE, assert
  April 2020 ≈ 14.7%.
- `test_real_api_usrec_covid_recession` — fetch USREC, assert =1 for
  Feb-Apr 2020.
- `test_real_api_vixcls_covid_peak` — fetch VIXCLS, assert
  2020-03-16 ≈ 82.69.
- `test_real_api_vintage_query_2018_12_31` — fetch UNRATE with
  vintage 2018-12-31, assert the August 2018 value matches FRED's
  archived as-of value (not the current revised number).
- `test_real_api_features_matrix_full_inventory` — fetch the full
  feature matrix for 2015-2024, assert shape and that no column is
  entirely NaN.

## Open questions

1. **Exact series list — resolved.** Final inventory is 20 features
   plus USREC ground truth, with `DCOILWTICO` (WTI crude) added to
   the market-internals category for commodity / inflation
   transmission signal. International series and additional
   commodity-complex names (gold, etc.) skipped per owner decision.

2. **Vintage policy for the screener.** I've defaulted to "regime
   model always uses vintage; screener uses vintage for training,
   latest for inference." Confirm? The alternative is
   "screener uses latest everywhere" — defensible because the
   screener's main features come from fundamentals (which we already
   handle vintage-correctly via Sharadar SF1) and macro is just
   conditioning. Either is reasonable; vintage-everywhere is more
   conservative.

3. **Pre-materialize the features matrix?** I've defaulted to
   compute-on-demand with per-series caching. The alternative is
   to write a `sp500_membership.parquet`-style processed file. My
   recommendation: don't pre-materialize the matrix because vintage
   queries make it parameterized by `vintage_date`; pre-materializing
   would need ~one parquet per CV fold, which is a lot. Per-series
   caching gives us 90% of the speedup without the explosion.

4. **`USREC` in the inventory vs separate.** I've put it in
   `REGIME_SERIES` but excluded it from the default features matrix.
   Alternative: split it out entirely so it's impossible to
   accidentally include. My take: comment-and-default-exclude is
   enough — a separate `LABELS` dict feels like ceremony, and the
   `default series_ids excludes USREC` line plus a test that asserts
   it's excluded gives the same guarantee with less surface area.

5. **`fredapi` dependency.** Already in `requirements.txt`. Anything
   wrong with using it vs raw `requests`? My take: keep it. The
   library is thin enough to swap if it goes unmaintained, and we
   get DataFrame parsing for free.

6. **Cache size.** ~20 series × (1 latest + ~10 vintage dates from CV
   folds) × monthly resolution × 35 years = ~250 small parquet files
   totaling a few MB. Negligible.

7. **FRED API key in CI?** Unit tests should not need it (they mock).
   Integration tests need it but are marked `slow` and skipped in CI.
   No CI secret required for v1. If we later want to run integration
   tests in CI, add `FRED_API_KEY` as a GitHub Actions secret and
   gate the workflow on it.

## References

- [CLAUDE.md](../../CLAUDE.md) — project conventions and methodology.
- [sp500_constituents.md](sp500_constituents.md) — sibling design doc;
  caching and error-handling patterns are deliberately parallel.
- [FRED API docs](https://fred.stlouisfed.org/docs/api/fred/).
- [fredapi on PyPI](https://pypi.org/project/fredapi/).
