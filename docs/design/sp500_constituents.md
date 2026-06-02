# Design: Point-in-time S&P 500 constituents from a dated change log

**Module**: `src/data_sources/sp500_membership.py`
**Status**: Approved, pre-implementation
**Owner**: Chris Garcia
**Last updated**: 2026-06-02
**Supersedes**: [ivv_constituents.md.superseded](ivv_constituents.md.superseded)
**Discovery audit trail**: [ivv_discovery_findings.md](ivv_discovery_findings.md)

## Purpose

Produce a point-in-time, survivorship-bias-free record of S&P 500 membership
for use as the training universe of the stock screener (see
[CLAUDE.md](../../CLAUDE.md)). Output is a long-format parquet keyed by
`(as_of_date, ticker_normalized)` covering monthly cross-sections from
2014 to present.

The original design attempted to scrape iShares IVV holdings archived by
the Wayback Machine. Discovery proved that approach non-viable
([ivv_discovery_findings.md](ivv_discovery_findings.md)). This design
replaces it with **reconstruction from a dated change log**: take a known
roster and walk through dated add/remove events to derive membership on
any historical date.

## Source

[`fja05680/sp500`](https://github.com/fja05680/sp500) — 854⭐, 190 forks,
MIT licensed, maintained since 2019, updated approximately every two
months. Selected after evaluation of three alternatives
([hanshof/sp500_constituents](https://github.com/hanshof/sp500_constituents),
[datasets/s-and-p-500-companies](https://github.com/datasets/s-and-p-500-companies),
[Ate329/top-us-stock-tickers](https://github.com/Ate329/top-us-stock-tickers));
see commit history for the comparison.

**Upstream sources cited by maintainer**: Andreas Clenow's *Trading
Evolved* dataset (1996-2019 backbone) plus Wikipedia's "List of S&P 500
companies" page and its "Selected changes" table, supplemented by
manual research into S&P press releases for dates Wikipedia omits.

**Pinned file**: `S&P 500 Historical Components & Changes(01-17-2026).csv`,
5.5 MB, 2705 rows spanning 1996-01-02 to 2026-01-14. Two columns:
`date`, `tickers` (comma-separated). Rows are one per change event for
2019+, weekly-ish snapshots for 1996-2018.

We also reference the smaller `sp500_changes_since_2019.csv` (111 rows,
`date, add, remove`) for spot-validation against the 2019+ changelog;
it is derivable from the main file and is not committed separately.

## Frozen-copy reproducibility model

The committed CSV at `data/raw/sp500_change_log/` is the **source of
truth**. Reproducibility does not depend on upstream availability.

- The committed copy is authoritative. Reruns produce identical results
  bit-for-bit.
- An accompanying `README.md` records the source URL, retrieval date,
  SHA256 hash, and pinned filename.
- A separate freshness check (`--check-upstream` flag) compares the
  committed SHA256 against the latest upstream file and reports whether
  upstream has changed. **It never overwrites the frozen copy
  automatically.** Bumping the frozen copy is a deliberate human action,
  ideally accompanied by re-running the validation suite against the
  known-events list to confirm the new version still passes.
- This isolates us from upstream regressions, deletions, or repo
  takedowns. It also makes our results auditable: anyone with the repo
  can reproduce the exact training universe used.

## Reconstruction algorithm

Given the main CSV (one row per change date with full membership as a
ticker list), membership on any date is computed by:

```
def members_on(date):
    # find the most recent row whose date <= target
    row = max((r for r in csv if r.date <= date), key=lambda r: r.date)
    return set(row.tickers.split(','))
```

This is correct because the source file stores the full roster on every
change event (or weekly-ish snapshot for 2014-2018). Walking back to the
nearest prior anchor gives the roster as it was *at that anchor*. Between
anchors, membership is constant by construction of the file.

**Validation queries** (e.g., "was SIVB a member on 2023-03-10?") are
exact lookups against the resulting set. No interpolation needed.

## Derived-event-date convention (LOAD-BEARING)

For 2014–2018, the source file contains weekly-ish snapshots rather than
explicit change events. When we need to enumerate add/remove events in
that era (for validation or for a future "stocks added in the last 90
days" feature), we derive them by diffing consecutive snapshot rows.

**Convention: assign the LATER snapshot's date as the effective date of
any derived add or remove.** Example: if a 2016-05-10 snapshot contains
ticker `X` and a 2016-05-17 snapshot does not, the derived removal event
is dated **2016-05-17**.

**Rationale**: this is the conservative choice for look-ahead bias. It
may claim a ticker remained in the universe for up to ~7 extra days
after its actual S&P-effective removal, which adds harmless noise that
the 12-month forward return label easily absorbs. The alternative
(assigning the earlier date + 1 day) would occasionally claim removal
*before* it happened, which is a real look-ahead leakage and is harder
to detect than membership noise. The cost of being early is silent
contamination of training data; the cost of being late is a small bias
toward including soon-to-be-removed names — visible, bounded, and not
threatening to model credibility.

**Code requirement**: wherever this convention is applied in code, leave
an inline comment of the form:

```python
# Option A: derived removal dated to the LATER snapshot. Conservative —
# never claims removal before it actually happened. See
# docs/design/sp500_constituents.md for rationale.
```

This convention is automatically applied to membership-on-date queries
too: walking back to the nearest prior anchor inherits the same "stays
in until the next snapshot proves otherwise" property. No extra logic
needed for the query path — only the event enumeration path.

## Output schema

Long format, one row per `(as_of_date, ticker_normalized)`.

**Materialization policy**: write one row per
`(month-end-business-day, ticker)` for trading months from 2014-01 to
the most recent month covered by the frozen copy. Daily materialization
is wasteful; the screener rebalances monthly. Other dates are derived
on demand via `members_on(date)` without writing to disk.

### `data/processed/sp500_membership.parquet`

| column | dtype | nullable | notes |
|---|---|---|---|
| `as_of_date` | date32 | no | month-end business day for this row |
| `ticker` | str (dict) | no | as-published, before normalization |
| `ticker_normalized` | str (dict) | no | post-normalization for joins (see Edge cases) |
| `source_version` | str | no | SHA256[:12] of the frozen CSV — provenance |

**Primary key**: `(as_of_date, ticker_normalized)`. Asserted at write time.

**Why dict-encoded strings**: tickers repeat heavily across months;
dict encoding gives ~10× compression and near-instant `==` filtering.

**What is *not* in this output**: weights, shares, market value, sector,
issuer name. The change-log source does not provide these. Sector
information comes from the fundamentals provider (Sharadar SF1) where
needed. The screener's cross-sectional ranks are computed from
fundamentals, not from index weights — losing weight data is harmless
for our use case. See the superseded IVV design for the original
broader schema if you ever need to reintroduce these fields from another
source.

## Caching

There is no scrape, so the caching model is much simpler than the
original IVV design.

```
data/raw/sp500_change_log/
  README.md                                              # source, hash, retrieval date
  S&P 500 Historical Components & Changes(01-17-2026).csv # frozen, committed

data/processed/
  sp500_membership.parquet                               # derived, regenerable

data/raw/sp500_change_log/upstream_check.json            # gitignored
                                                         # last freshness-check result
```

The frozen CSV is committed via a `.gitignore` override
(`!data/raw/sp500_change_log/**`). The processed parquet is gitignored
like all of `data/processed/`. The upstream-check artifact is also
gitignored — it changes with every check and has no reproducibility
value.

The processed parquet is rebuilt whenever the frozen CSV's hash changes.
Otherwise it is read directly. Build time is fast (~1 second for ~140
monthly cross-sections × ~500 tickers).

## Validation

Three classes of check, same structure as the superseded design.

### Known constituent events (hard assertions)

All dates use the **S&P-effective removal date** convention, not the
news-salient event date. The two are different for failed banks: SVB
was closed by FDIC on March 10, 2023 (Friday), but S&P announced index
removal effective March 15. We use March 15 because that is the date
that determines whether SVB was in our training universe on a given
date.

- **SIVB**: present in membership on 2023-03-14; absent on 2023-03-15.
- **FRC**: present on 2023-05-03; absent on 2023-05-04.
- **TSLA**: absent on 2020-12-18; present on 2020-12-21.
- **META**: ticker present from 2022-06-09 onward; **FB** present
  through 2022-06-08, absent thereafter.

### Sanity checks (warnings)

- Each materialized cross-section has 495-510 tickers.
- AAPL and MSFT present in every monthly cross-section from 2014.
- No ticker repeats within a single cross-section.

### Cross-check

- Month-over-month turnover ≤ 2% of constituents — higher suggests a
  source-file regression.
- 2014-06-30 cross-section returns 498 tickers with AAPL ✓, MSFT ✓,
  FB ✓, LEH ✗ (defunct 2008), META ✗ (didn't exist as ticker yet).

## Edge cases

**Ticker renames** (FB→META 2022-06-09, etc.). The source already
records these correctly — `META` appears starting 2022-06-09 and `FB`
disappears the same day. `ticker_normalized` always reflects the
current ticker; `ticker` preserves the as-published symbol for the row's
`as_of_date`. The known-rename mapping is small and hand-maintained:

```python
RENAMES: dict[str, list[tuple[date, str]]] = {
    "FB": [(date(2022, 6, 9), "META")],
    # ... (full list in code, validated against source)
}
```

**Dual-class shares** (GOOG and GOOGL, FOXA and FOX, NWSA and NWS).
Source includes both as separate tickers. Both are preserved as
separate rows. Never collapsed.

**Index size != 500**. The actual S&P 500 has held 500-505 stocks at
various points due to dual-class and transition timing. Source captures
this accurately. We assert 495-510, not exactly 500.

**Inference path** (screening today's S&P 500). Same code path —
membership on today's date via the same `members_on()` function against
the same frozen CSV. The frozen copy may lag the live index by up to
~2 months between updates; this is irrelevant for 12-month forward
return prediction.

**Pre-2014 data**. Available in the source but out of scope per project
decision (sparse, possibly biased). Module raises if called with a
date before 2014-01-01.

**Add/drop event precision**. Deferred. The screener doesn't need
exact add/drop dates. If we later add features like "stocks added in
last 90 days outperform," we'll use the derived-event enumeration path
with the Option A convention documented above. Acknowledged in code
comments; no infrastructure built for it now.

## Error handling

Strict, since the source is local. CLAUDE.md's clarification of the
no-silent-except rule still applies, but the batch-degradation patterns
are not needed here — there is no flaky network.

- **Frozen CSV missing or unreadable**: hard fail with a clear error
  pointing at `data/raw/sp500_change_log/README.md` for retrieval
  instructions.
- **Frozen CSV hash mismatch with README**: hard fail. Indicates the
  file was modified outside the documented retrieval flow.
- **Parse failure on the CSV** (malformed row, missing columns): hard
  fail. Indicates a code bug or an upstream format change requiring
  human attention.
- **Validation failures** (any known-event assertion fails): hard fail.
  Either the source regressed or the code is broken; both require
  immediate attention.
- **Freshness check failure** (cannot reach upstream): log at WARNING,
  return last-known check result, never block. Stale freshness data is
  acceptable; the frozen copy is the source of truth.

## Test plan

`tests/test_sp500_membership.py`.

### Unit tests (fast, run in CI, no network)

- `test_members_on_known_date` — 2014-06-30 returns 498 tickers
  including AAPL, MSFT, FB; excludes LEH, META.
- `test_members_on_before_2014_raises` — date < 2014-01-01 raises.
- `test_members_on_returns_nearest_prior_anchor` — date between
  snapshots returns the prior snapshot's roster.
- `test_known_event_svb_effective_date` — SIVB ∈ membership(2023-03-14),
  SIVB ∉ membership(2023-03-15). Comment cites the convention rationale.
- `test_known_event_frc_effective_date` — FRC ∈ membership(2023-05-03),
  FRC ∉ membership(2023-05-04).
- `test_known_event_tsla_addition` — TSLA ∉ membership(2020-12-18),
  TSLA ∈ membership(2020-12-21).
- `test_known_event_fb_meta_rename` — FB ∈ membership(2022-06-08);
  META ∈ membership(2022-06-09); FB ∉ membership(2022-06-09).
- `test_universe_size_sanity` — five random monthly cross-sections all
  have 495-510 tickers.
- `test_no_duplicate_tickers_within_section` — any cross-section
  asserts unique `ticker_normalized`.
- `test_normalize_ticker_handles_known_renames` — FB→META mapping
  applied per `as_of_date`.
- `test_normalize_ticker_preserves_dual_class` — GOOG and GOOGL both
  preserved as separate identifiers.
- `test_frozen_csv_hash_matches_readme` — committed SHA256 matches the
  hash recorded in `data/raw/sp500_change_log/README.md`.
- `test_derived_event_uses_later_snapshot_date` — synthetic snapshot
  pair where `X` disappears between dates `t1` and `t2` produces a
  derived removal event dated `t2`, not `t1+1`. Comment cites the
  convention rationale.
- `test_materialized_parquet_primary_key_unique` — built parquet
  satisfies `(as_of_date, ticker_normalized)` uniqueness.

### Integration tests (`@pytest.mark.slow`, skipped in CI)

- `test_upstream_freshness_check_returns_status` — actual network call
  to fja05680, verifies the freshness-check code path works end-to-end.

## References

- [CLAUDE.md](../../CLAUDE.md) — project conventions and locked methodology.
- [ivv_discovery_findings.md](ivv_discovery_findings.md) — discovery
  audit trail for the abandoned IVV-Wayback approach.
- [ivv_constituents.md.superseded](ivv_constituents.md.superseded) —
  original design document, preserved for historical context.
- [fja05680/sp500](https://github.com/fja05680/sp500) — upstream source.
