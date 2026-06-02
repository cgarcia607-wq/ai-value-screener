# IVV constituents discovery findings

**Date**: 2026-06-02
**Status**: Historical audit trail. The approach investigated here was abandoned. See [sp500_constituents.md](sp500_constituents.md) for the design we adopted.

## TL;DR

**The IVV-via-Wayback approach in the original design ([ivv_constituents.md.superseded](ivv_constituents.md.superseded)) cannot be implemented.** The full ~500-holdings table is, and always has been, JavaScript-rendered from an AJAX endpoint that the Wayback Machine never archived. The product page snapshots only ever contained a server-rendered **top-10 preview**, and even that preview disappeared after 2015. Neither HTML scraping nor CSV download is viable on Wayback for IVV constituents.

## Evidence

### CDX inventory: HTML product page

Query: `url=ishares.com/us/products/239726/ishares-core-sp-500-etf`, monthly dedup (`collapse=timestamp:6`), 2014–2026.

- **124 monthly-deduped snapshots** spanning 2014-04 through 2026-04.
- 6–12 snapshots per year, near-monthly cadence — suitable for the original design on this dimension. The problem turned out to be *what's in the snapshots*, not how many.

| year | snapshots | year | snapshots |
|---|---|---|---|
| 2014 | 6 | 2021 | 9 |
| 2015 | 9 | 2022 | 11 |
| 2016 | 10 | 2023 | 9 |
| 2017 | 9 | 2024 | 9 |
| 2018 | 12 | 2025 | 12 |
| 2019 | 12 | 2026 | 4 (YTD) |
| 2020 | 12 | | |

### CDX inventory: AJAX CSV endpoint

Query: `url=ishares.com/us/products/239726/ishares-core-sp-500-etf/1467271812596.ajax`, 2014–2026.

- **2 archived snapshots total** (Feb 2021, Aug 2025).
- Both are 2.5–3 KB `text/html` responses — far too small to be 500-row CSVs (which would be 30–50 KB minimum).
- Fetching the Aug 2025 snapshot returned a 7 KB HTML widget fragment beginning with `<h2>Holdings</h2>` — i.e., the empty widget shell, not data. Crawlers hit the URL without JavaScript and got the unrendered widget.

Probed five alternative `.ajax` endpoint IDs found in the prefix inventory (`1395165510754`, `1467271812594/595/606`, `1521942788811`): zero or near-zero archives, all <1.5 KB error/widget responses. **No AJAX endpoint anywhere in Wayback contains real holdings data.**

A wider CDX query with `filter=mimetype:.*(csv|xls|excel|octet|plain).*` under product 239726 returned **zero results**. Wayback never crawled a machine-readable holdings file for IVV.

### Three-sample parse

Fetched the snapshot nearest mid-year for 2015, 2020, 2024:

| year | size | AAPL inline | colTicker rows | parseable holdings |
|---|---|---|---|---|
| 2015 | 573 KB | 1 | 10 | **top 10 only** |
| 2020 | 1.4 MB | 0 | 0 | **none** |
| 2024 | 1.7 MB | 0 | 0 | **none** |

Probing further: 2016 and 2017 snapshots contain one `AAPL` mention each (in scripts/metadata, not a holdings table). 2018 and 2019 contain zero.

**The "good" era (2015 and earlier) was never actually good.** Inspecting the 2015 HTML shows the table is hardcoded to 10 rows:

```html
<table class="display" id="topHoldingsTable" data-length="10">
```

It is a preview widget. The full holdings have always been fetched client-side.

### SPY (SSGA) as a backup source

Spot-checked SSGA's SPY holdings xlsx URL on Wayback:

- `holdings-daily-us-en-spy.xlsx`: **0 archives**.
- Wider SSGA CDX query for any URL containing both "spy" and "holdings": **empty**.

SSGA gates downloads behind a click-through that crawlers don't traverse. Wayback has nothing useful here either.

## What this means for the project

The discovery phase did exactly what it was designed to do: it surfaced a fundamental data-availability problem before ~500 lines of scraper code were written against a source that doesn't exist. The pivot was from scraping a third-party fund-holdings page to **reconstructing point-in-time membership from a dated change log** committed to the repo as a frozen reproducible artifact.

The new design is in [sp500_constituents.md](sp500_constituents.md). The methodology lock in [CLAUDE.md](../../CLAUDE.md) was updated to reference change-log reconstruction with frozen-copy reproducibility, validated against known index events using S&P-effective removal dates.

This document is preserved as the audit trail. Any future reader wondering "why isn't this an IVV scraper?" should land here.

## Probe artifacts (local, gitignored)

The raw CDX dumps and fetched HTML samples were saved during discovery to `data/raw/ivv_constituents/` and are gitignored. They are not required for any downstream work and can be deleted at will.

| file | purpose |
|---|---|
| `data/raw/ivv_constituents/discovery/cdx_html.json` | CDX inventory of HTML product page (124 entries) |
| `data/raw/ivv_constituents/discovery/cdx_ajax.json` | CDX inventory of primary AJAX endpoint (2 entries) |
| `data/raw/ivv_constituents/discovery/cdx_all_urls.json` | All URLs under product 239726 (479 entries) |
| `data/raw/ivv_constituents/discovery/cdx_spy_xlsx.json` | SSGA SPY xlsx CDX search (empty) |
| `data/raw/ivv_constituents/discovery/cdx_spy_wide.json` | SSGA SPY-holdings prefix search (empty) |
| `data/raw/ivv_constituents/discovery/probe_ajax_2025.bin` | Aug 2025 AJAX response (7 KB widget fragment) |
| `data/raw/ivv_constituents/raw_html/*.html` | Seven sample HTML snapshots (2015, 2016, 2017, 2018, 2019, 2020, 2024) |
