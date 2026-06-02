# Frozen S&P 500 change log

This directory contains a frozen copy of the upstream change-log CSV used
to reconstruct point-in-time S&P 500 membership. The committed file is
the **source of truth** for reproducibility — reruns produce identical
results regardless of upstream availability or changes.

See [docs/design/sp500_constituents.md](../../../docs/design/sp500_constituents.md)
for the full design and the reconstruction algorithm.

## Pinned file

| field | value |
|---|---|
| Filename | `S&P 500 Historical Components & Changes(01-17-2026).csv` |
| Size | 5,499,082 bytes |
| Lines | 2,706 |
| Coverage | 1996-01-02 through 2026-01-14 |
| Schema | `date, tickers` (tickers is a comma-separated list) |
| SHA256 | `711bf3b5f21e25ad76456a19df7632eab6543cf3b2b9c51ab210bb012054a7f6` |

## Provenance

| field | value |
|---|---|
| Source repository | [fja05680/sp500](https://github.com/fja05680/sp500) |
| Source URL | `https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes(01-17-2026).csv` |
| Retrieval date | 2026-06-02 |
| Retrieval method | `curl -sSL` |
| Upstream license | MIT |
| Upstream maintainer | Frank Anderson (fja05680) |
| Upstream sources cited | Andreas Clenow's *Trading Evolved* dataset (1996–2019 backbone) + Wikipedia's S&P 500 page and "Selected changes" table, supplemented by maintainer's manual research into S&P press releases |

## Updating this frozen copy

The frozen copy should be updated **deliberately**, not automatically.
The process:

1. Run the freshness check: `python -m src.data_sources.sp500_membership --check-upstream`.
   This compares the committed SHA256 against the latest upstream file
   and reports whether upstream has changed. It does not modify anything.
2. If upstream has changed and you want to adopt the new version:
   a. Download the new file and place it at this path. The filename
      typically includes a date suffix — keep the upstream filename
      exactly.
   b. Update the table above with new size, lines, SHA256, and
      retrieval date.
   c. Re-run the full validation suite
      (`pytest tests/test_sp500_membership.py`). All known-event
      assertions must still pass. If any fail, investigate before
      committing — the source may have regressed.
   d. Commit the new file, the updated README, and any test-fixture
      updates as a single conventional commit
      (`chore: bump frozen sp500 change log to <date>`).
3. If upstream has changed but you do NOT want to adopt the new
   version, no action needed. The frozen copy remains authoritative.

## Why this is committed despite `data/` being gitignored

The repository-wide `.gitignore` excludes all of `data/`. This directory
overrides that with `!data/raw/sp500_change_log/**` in the project
`.gitignore`. The exception is justified because:

- The file is ~5.5 MB (under the GitHub soft limit of 50 MB for normal
  files).
- It is not generated data — it is an external dataset we depend on
  for reproducibility.
- Committing it eliminates a class of "works on my machine" failure
  modes and protects against upstream takedowns or deletions.
- Future updates are tracked in git history, giving a clear audit
  trail of what data version produced which results.
