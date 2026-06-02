# Design: Point-in-time S&P 500 constituents from iShares IVV

**Module**: `src/data_sources/ivv_constituents.py`
**Status**: Approved, pre-discovery
**Owner**: Chris Garcia
**Last updated**: 2026-06-02

## Purpose

Produce a point-in-time, survivorship-bias-free record of S&P 500 membership
for use as the training universe of the stock screener (see
[CLAUDE.md](../../CLAUDE.md)). Output is a long-format parquet keyed by
`(as_of_date, ticker)` covering monthly snapshots from 2014 to present.

Why this matters: the legacy v1 pipeline used `pd.read_html` against the
current Wikipedia roster, which encodes survivorship bias — failed companies
(Lehman, Bear Stearns, SVB, FRC) were silently dropped from the training
universe, inflating measured outperformance. This module exists to eliminate
that bias.

## Source strategy

**Primary**: iShares Core S&P 500 ETF (IVV) holdings, archived by the
Wayback Machine.

- Product page: `https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf`
- AJAX download (CSV): `https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf/1467271812596.ajax?fileType=csv&fileName=IVV_holdings&dataType=fund`

Both forms have been archived intermittently. Discovery (§Discovery phase)
will determine which is more reliably available before committing.

- **Plan A**: parse the rendered HTML holdings table.
- **Plan B**: download the archived CSV directly.

Plan B is cleaner if available — CSV avoids HTML parsing fragility — but
relies on the Wayback Machine having crawled the AJAX endpoint. Decide
after discovery, not before.

**Coverage**: training restricted to **2014-present** per project decision.
Pre-2014 Wayback coverage of IVV is too sparse for reliable monthly
cross-sections. Documented as a known limitation in the README, not a flaw.

**Inference path**: also uses Wayback (most recent available snapshot).
Single code path, no source drift between training and inference. Being
1-2 days behind on constituent membership is irrelevant for a 12-month
forward return prediction.

## Snapshot frequency

Monthly, aligned to month-end.

- Walk-forward CV rebalances monthly; finer granularity buys no signal.
- Each `YYYY-MM` selects the snapshot closest to the last business day of
  the month.
- If no snapshot exists in the calendar month, expand ±15 days. If still
  none, mark month as missing in the manifest and continue.

### Two date fields

Both stored, distinct in meaning:

- `snapshot_date` — the Wayback capture date. Provenance.
- `as_of_date` — the holdings effective date, parsed from the file header
  ("Holdings as of [date]"). **This is the join key for fundamentals and
  prices.** iShares typically publishes T-1 holdings, so `as_of_date` is
  usually `snapshot_date - 1 business day`, but it is parsed explicitly,
  never derived.

If header parsing fails, the snapshot is **hard-failed and dropped** —
not back-computed from `snapshot_date`. Guessing the as-of date silently
risks reintroducing look-ahead bias on any snapshot where iShares published
T+0 data. The as_of_date is load-bearing for bias-free joins; we never
guess it.

## Wayback API mechanics

**Discovery: CDX API.** Single call returns the full snapshot inventory:

```
https://web.archive.org/cdx/search/cdx
  ?url=ishares.com/us/products/239726/*
  &from=20140101&to=<today>
  &output=json&filter=statuscode:200&collapse=timestamp:6
```

`collapse=timestamp:6` deduplicates to one snapshot per month.

**Retrieval: direct fetch** with the `id_` flag to skip Wayback's injected
toolbar HTML:

```
https://web.archive.org/web/<timestamp>id_/<original_url>
```

**Why not Memento or Availability API**: both return one snapshot per call.
CDX returns the full inventory in one request — better for cache planning
and rate-limit budget.

### Rate limiting

- 2 req/sec ceiling (well under Wayback's published ~15 req/min limit).
- Retry on 429/503/504 with exponential backoff: 4s, 16s, 64s, then give up.
- Failed snapshots recorded to manifest (see §Error handling).
- `requests.Session` for connection reuse.
- User-Agent: `ai-value-screener/0.1 (<contact email>)` — contact email
  to be specified by owner before first run.

### Expected runtime

~140 snapshots (12 years × ~1/month) × ~0.5s + retry overhead ≈ 2-3
minutes for a cold full pull. Incremental runs (new month only) ≈ 1-2
seconds.

## Output schema

Long format, one row per `(as_of_date, ticker)`. Wide would be a sparse
7000-column matrix with constant churn.

### Consolidated: `data/raw/ivv_constituents/constituents.parquet`

| column | dtype | nullable | notes |
|---|---|---|---|
| `as_of_date` | date32 | no | holdings effective date — primary join key |
| `snapshot_date` | date32 | no | Wayback capture date — provenance |
| `ticker` | str (dict) | no | as-published, pre-normalization |
| `ticker_normalized` | str (dict) | no | post-normalization for joins |
| `name` | str | no | issuer name |
| `sector` | str (dict) | yes | GICS, null pre-2017 |
| `weight` | float64 | no | percent of fund (0-100) |
| `shares` | int64 | yes | null if not parseable |
| `market_value` | float64 | yes | USD |
| `source_url` | str | no | exact Wayback URL — full provenance |

**Primary key**: `(as_of_date, ticker_normalized)`. Asserted at write time.

**Why dict-encoded strings**: tickers and sectors repeat heavily across
snapshots; dict encoding gives ~10× compression and near-instant `==`
filtering downstream.

### Per-snapshot: `data/raw/ivv_constituents/snapshots/<YYYY-MM-DD>.parquet`

Same schema, single snapshot. Enables incremental rebuilds without
re-fetching from Wayback if the consolidated file is corrupted or the
parser changes.

## Caching layout

```
data/raw/ivv_constituents/
  cdx_index.json           # cached CDX response, TTL 7 days
  snapshots/
    2014-03-31.parquet
    2014-04-30.parquet
    ...
    2026-05-31.parquet
  raw_html/                # gzipped raw archive responses
    20140331123456.html.gz
    ...
  manifest.json            # one entry per attempted snapshot
  constituents.parquet     # consolidated long table
```

### Re-scrape avoidance

- `cdx_index.json` refreshed if older than 7 days. Diff against `manifest.json`
  to fetch only missing snapshots.
- Raw HTML/CSV cached unconditionally — Wayback responses for fixed
  `(timestamp, url)` are immutable, so cache hits are always correct.
- Parsed per-snapshot parquets are derived. Deleting them triggers re-parse
  from cached raw files without re-fetching from Wayback.
- `constituents.parquet` is rebuilt from per-snapshot files at the end of
  each run.

The fetch / parse / consolidate split means parser bugs can be fixed and
re-run without burning Wayback bandwidth.

## Validation

Three classes of check. Runs as pytest assertions and as a CLI `--validate`
flag that prints a report.

### Known constituent events (hard assertions)

The IVV snapshot tells us *membership at a point in time*, not the exact
add/drop date. Assertions are framed as "present in snapshot X, absent
in snapshot Y," not "removed on date Z":

- SVB Financial Group (SIVB): present Feb 2023, absent April 2023.
- First Republic Bank (FRC): present March 2023, absent May 2023.
- Tesla (TSLA): absent November 2020, present January 2021.
- Meta (META) ticker present from June 2022 onward; Facebook (FB)
  ticker present before then.

### Sanity checks (warnings)

- Each snapshot has 495-510 holdings (allows transitions, cash, futures).
- Weights sum to 99-101%.
- No ticker repeats within a snapshot.
- AAPL, MSFT present in every snapshot from 2014.

### Cross-snapshot consistency

- Month-over-month turnover ≤ 2% of constituents — anything higher
  suggests parser bug or coverage gap.
- No "phantom drops" where a ticker disappears for one month and
  reappears.

## Edge cases

**Ticker symbol changes**: hard-coded mapping
`TICKER_RENAMES: dict[str, list[tuple[date, str]]]` covering major renames
since 2014 (FB→META 2022-06-09, etc.). `ticker_normalized` always reflects
the *current* ticker as of `as_of_date`. Original `ticker` preserved.

**Dual-class shares**: GOOG and GOOGL are two separate IVV holdings —
both kept as separate rows. Same for FOXA/FOX, NWSA/NWS. Never collapsed.

**Spinoffs/mergers**: handled implicitly — sequence of snapshots preserves
them. No special logic needed.

**Missing snapshots near month boundaries**: window expansion ±15 days,
then mark missing.

**File format changes**: parser locates the header row dynamically.
Pre-2017 files lack `Sector` — stored as null, not a parser failure.

**Non-equity holdings**: filter to `Asset Class == "Equity"`, discard
cash and futures.

**Add/drop event precision**: deferred. The screener doesn't need exact
add/drop dates. If we later add features like "stocks added in last 90
days outperform," we'll source S&P announcements separately. Acknowledged
in code comments; no infrastructure built for it now.

## Error handling

CLAUDE.md prohibits silent except blocks. This module respects that rule
while degrading gracefully on individual snapshot failures during bulk
pulls. (CLAUDE.md updated to clarify the rule applies to *silent* failures,
not logged-and-tracked failures with manifests and thresholds.)

Policy:

1. **Per-snapshot fetch failures**: log at WARNING (snapshot date, URL,
   error), persist to `manifest.json` with `status: failed` and the error,
   continue. Hard-failing on first transient 503 would block all progress.
2. **Aggregate failure threshold**: if more than 10% of attempted
   snapshots in a single run fail, abort with a hard error. Single blips
   are expected; systemic failure (Wayback down, URL structure changed)
   must halt.
3. **Parse failures on a fetched snapshot**: hard-fail. Successful fetch
   but failed parse indicates a code bug, not data quality noise.
4. **`as_of_date` header parse failure**: hard-fail the snapshot, log
   loudly, mark in manifest. **Never back-compute from `snapshot_date`.**
   The as_of_date is load-bearing for bias-free joins; guessing it
   silently risks reintroducing look-ahead bias.
5. **Validation failures** (SVB in April 2023 snapshot, etc.): hard-fail.
   Indicates fundamental brokenness.
6. **Inference path** (single-snapshot fetch for current screening): no
   recovery, fail loudly on any error. Bulk-vs-single distinction matters.

Net: nothing fails silently. Logged-and-tracked failures with a
documented threshold are appropriate for batch operations.

## Discovery phase (before full implementation)

**Scout before storming the beach.** Don't run a 140-snapshot pull
without knowing whether the parser works on the file formats in the wild.
Discovery runs as a separate step, results reviewed, then full
implementation proceeds.

Discovery steps:

1. **CDX inventory**. One CDX query for the product page URL prefix from
   2014-01-01 to today. Report:
   - Total snapshot count.
   - Snapshots per year (table).
   - Years with gaps > 60 days.
   - Whether the `.ajax?fileType=csv` endpoint appears in the index at all.
2. **Plan A vs Plan B determination**. Based on (1), decide:
   - If `.ajax?fileType=csv` is well-archived → Plan B (CSV).
   - Otherwise → Plan A (HTML).
   - If both work → Plan B, with Plan A as fallback for snapshots where
     CSV is missing.
3. **Three-sample fetch**. Manually fetch one snapshot from each of:
   2015, 2020, 2024. For each:
   - Confirm HTTP 200 from Wayback.
   - Save raw response to `data/raw/ivv_constituents/raw_html/`.
   - Attempt to parse with a draft parser.
   - Report: parse success/failure, row count, columns present,
     `as_of_date` extracted, sanity-check holdings count.
4. **Coverage report**. Written to `data/raw/ivv_constituents/discovery_report.md`:
   - CDX inventory table.
   - Plan A/B recommendation with evidence.
   - Three-sample parse results.
   - Estimated runtime for full pull.
   - Known issues and recommended next steps.

**STOP after discovery.** Report findings to owner. No full pull, no
final parser code, no consolidated parquet until discovery is reviewed.

## Test plan

`tests/test_ivv_constituents.py`.

### Unit tests (mocked HTTP, fast, run in CI)

- `test_parse_holdings_modern_format` — post-2018 fixture parses correctly.
- `test_parse_holdings_pre_2017_format` — pre-2017 fixture (no Sector)
  parses correctly.
- `test_parse_holdings_extracts_as_of_date` — header date parsed.
- `test_parse_holdings_as_of_date_failure_hard_fails` — missing/malformed
  header raises, never falls back to snapshot_date.
- `test_parse_holdings_filters_non_equity` — cash/futures dropped.
- `test_parse_holdings_handles_missing_columns` — gracefully nulls.
- `test_cdx_query_builds_correct_url` — URL construction is right.
- `test_cdx_response_parsing_handles_empty_result` — empty inventory
  yields empty list, not error.
- `test_normalize_ticker_handles_known_renames` — FB→META mapping
  applied per as_of_date.
- `test_normalize_ticker_preserves_dual_class` — GOOG and GOOGL both
  preserved.
- `test_consolidate_snapshots_dedupes_correctly` — `(as_of_date, ticker)`
  uniqueness enforced.
- `test_consolidate_snapshots_primary_key_violation_raises` — duplicate
  key raises.
- `test_rate_limit_backoff_schedule` — 4s, 16s, 64s sequence.
- `test_manifest_tracks_failed_snapshots` — failures recorded with
  status and error.
- `test_aggregate_failure_threshold_aborts` — >10% failures triggers
  hard abort.

### Integration tests (real Wayback, `@pytest.mark.slow`, skipped in CI)

- `test_fetch_real_snapshot_smoke` — one known-good snapshot end-to-end.
- `test_known_event_svb_removal` — SIVB Feb 2023 vs April 2023.
- `test_known_event_frc_removal` — FRC March 2023 vs May 2023.
- `test_known_event_tsla_addition` — TSLA Nov 2020 vs Jan 2021.
- `test_known_event_fb_meta_rename` — FB May 2022, META July 2022.
- `test_universe_size_sanity` — 5 random snapshots all have 495-510
  holdings.
- `test_weights_sum_sanity` — 5 random snapshots have weights summing
  to 99-101%.

Fixtures for unit tests live in `tests/fixtures/ivv/` — real archived
responses, captured once and checked in (~50KB each).

## Open items (resolved)

| # | Question | Resolution |
|---|---|---|
| 10.1 | Pre-2014 coverage | Restrict training to 2014+, document as known limitation |
| 10.2 | Add/drop event precision | Defer — out of scope |
| 10.3 | Error handling pushback | Accepted, CLAUDE.md clarified |
| 10.4 | Inference path | Wayback for inference too |
| 10.5 | Include weights/shares/sector | Yes — all fields |
| 10.6 | User-Agent contact email | Owner to specify before first run |
| 10.7 | iShares historical downloads | Wayback is primary source |

## References

- [CLAUDE.md](../../CLAUDE.md) — project conventions and locked methodology.
- iShares IVV product page: `https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf`
- Wayback Machine CDX API: `https://github.com/internetarchive/wayback/tree/master/wayback-cdx-server`
- S&P 500 historical constituent changes (for validation events): public press releases.
