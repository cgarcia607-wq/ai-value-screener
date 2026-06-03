"""Point-in-time S&P 500 membership via change-log reconstruction.

See docs/design/sp500_constituents.md for the full design.

When bumping the frozen CSV, update FROZEN_CSV_FILENAME, EXPECTED_SHA256,
and data/raw/sp500_change_log/README.md together. The hash-consistency
test enforces all three stay in sync.
"""

from __future__ import annotations

import calendar
import csv
import datetime as dt
import hashlib
import json
import logging
from functools import lru_cache
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = _REPO_ROOT / "data" / "raw" / "sp500_change_log"
FROZEN_CSV_FILENAME = "S&P 500 Historical Components & Changes(01-17-2026).csv"
FROZEN_CSV_PATH = FROZEN_DIR / FROZEN_CSV_FILENAME
EXPECTED_SHA256 = "711bf3b5f21e25ad76456a19df7632eab6543cf3b2b9c51ab210bb012054a7f6"

UPSTREAM_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes(01-17-2026).csv"
)
UPSTREAM_CHECK_PATH = FROZEN_DIR / "upstream_check.json"

PROCESSED_PARQUET_PATH = (
    _REPO_ROOT / "data" / "processed" / "sp500_membership.parquet"
)

MIN_SUPPORTED_DATE = dt.date(2014, 1, 1)

# Ticker renames as published by the source. Keyed by the *original*
# ticker; value is a chronological list of (effective_date, new_ticker)
# entries so multi-hop chains stay representable in the future.
#
# Dual-class shares (GOOG/GOOGL, FOXA/FOX, NWSA/NWS) are NOT renames
# and must never be collapsed — both classes remain separate tickers.
RENAMES: dict[str, list[tuple[dt.date, str]]] = {
    "FB": [(dt.date(2022, 6, 9), "META")],
}


def members_on(as_of_date: dt.date) -> set[str]:
    """Return S&P 500 membership on `as_of_date`.

    Uses the most recent prior anchor in the frozen change-log CSV.
    Raises ValueError for dates before 2014-01-01.
    """
    if as_of_date < MIN_SUPPORTED_DATE:
        raise ValueError(
            f"as_of_date {as_of_date.isoformat()} is before the supported "
            f"training window (2014-01-01). Pre-2014 source data has sparse "
            f"coverage and is excluded per project decision. See "
            f"docs/design/sp500_constituents.md."
        )
    rows = _load_frozen_csv()
    for row_date, tickers in reversed(rows):
        if row_date <= as_of_date:
            return set(tickers)
    raise RuntimeError(
        f"No anchor row at or before {as_of_date.isoformat()}. Source CSV "
        f"may be corrupted; first row is {rows[0][0].isoformat()}."
    )


def _normalize_ticker(ticker: str, as_of_date: dt.date) -> str:
    """Apply ticker renames in RENAMES based on as_of_date.

    Walks RENAMES[ticker] history in reverse and returns the latest
    rename whose effective_date is <= as_of_date. Returns the original
    ticker if no rename applies (including for tickers not in RENAMES).
    """
    history = RENAMES.get(ticker)
    if not history:
        return ticker
    for effective_date, new_ticker in reversed(history):
        if as_of_date >= effective_date:
            return new_ticker
    return ticker


def _month_end_business_days(start: dt.date, end: dt.date) -> list[dt.date]:
    """Last weekday of each calendar month falling within [start, end]."""
    result: list[dt.date] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        last_day = calendar.monthrange(year, month)[1]
        d = dt.date(year, month, last_day)
        while d.weekday() >= 5:  # Sat=5, Sun=6
            d -= dt.timedelta(days=1)
        if start <= d <= end:
            result.append(d)
        month += 1
        if month > 12:
            year, month = year + 1, 1
    return result


def build_membership_table(start: dt.date, end: dt.date) -> pa.Table:
    """Build a long-format pyarrow Table of (as_of_date, ticker) rows.

    Materializes one row per (month-end-business-day, ticker) for months
    in [start, end]. The `ticker` and `ticker_normalized` columns are
    explicitly dictionary-encoded at construction — see the Output
    schema section of docs/design/sp500_constituents.md for why we avoid
    pandas' implicit category dtype here.

    Asserts primary key (as_of_date, ticker_normalized) uniqueness before
    returning.
    """
    dates = _month_end_business_days(start, end)
    source_version = EXPECTED_SHA256[:12]

    as_of: list[dt.date] = []
    tickers: list[str] = []
    normalized: list[str] = []
    for d in dates:
        for t in sorted(members_on(d)):
            as_of.append(d)
            tickers.append(t)
            normalized.append(_normalize_ticker(t, d))

    pairs = set(zip(as_of, normalized))
    if len(pairs) != len(as_of):
        raise ValueError(
            f"Primary key (as_of_date, ticker_normalized) is not unique: "
            f"{len(as_of) - len(pairs)} duplicate rows in built table."
        )

    return pa.table(
        {
            "as_of_date": pa.array(as_of, type=pa.date32()),
            "ticker": pa.array(tickers, type=pa.string()).dictionary_encode(),
            "ticker_normalized": pa.array(
                normalized, type=pa.string()
            ).dictionary_encode(),
            "source_version": pa.array(
                [source_version] * len(as_of), type=pa.string()
            ).dictionary_encode(),
        }
    )


def load_membership_table() -> pa.Table:
    """Load the materialized membership parquet, building if missing or stale.

    Reads `data/processed/sp500_membership.parquet` if it exists and its
    `source_version` column matches `EXPECTED_SHA256[:12]` (the current
    frozen CSV). Otherwise rebuilds over [MIN_SUPPORTED_DATE,
    last_anchor_date] via build_membership_table and writes the parquet.
    """
    expected_version = EXPECTED_SHA256[:12]
    if PROCESSED_PARQUET_PATH.exists():
        existing = pq.read_table(PROCESSED_PARQUET_PATH)
        if len(existing) > 0:
            existing_version = existing["source_version"][0].as_py()
            if existing_version == expected_version:
                return existing
            logger.info(
                "Membership parquet source_version %s != current %s; "
                "rebuilding.",
                existing_version,
                expected_version,
            )
    rows = _load_frozen_csv()
    table = build_membership_table(MIN_SUPPORTED_DATE, rows[-1][0])
    PROCESSED_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, PROCESSED_PARQUET_PATH)
    return table


def check_upstream_freshness() -> dict:
    """Compare frozen CSV SHA256 against the latest upstream version.

    Persists result to data/raw/sp500_change_log/upstream_check.json using
    the schema in docs/design/sp500_constituents.md. Never modifies the
    frozen CSV.

    Network failures log at WARNING and return the last-known result from
    disk if one exists, or a fresh dict with matches=None and an "error"
    field if not.
    """
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    headers = {
        "User-Agent": (
            "ai-value-screener "
            "(+https://github.com/cgarcia607-wq/ai-value-screener)"
        ),
    }
    try:
        resp = requests.get(UPSTREAM_URL, headers=headers, timeout=30)
        resp.raise_for_status()
        content = resp.content
    except requests.exceptions.RequestException as e:
        logger.warning(
            "Upstream freshness check failed: %s. Returning last-known result.",
            e,
        )
        if UPSTREAM_CHECK_PATH.exists():
            return json.loads(UPSTREAM_CHECK_PATH.read_text())
        return {
            "checked_at": now,
            "upstream_sha256": None,
            "frozen_sha256": EXPECTED_SHA256,
            "matches": None,
            "upstream_url": UPSTREAM_URL,
            "error": str(e),
        }

    upstream_sha = hashlib.sha256(content).hexdigest()
    result = {
        "checked_at": now,
        "upstream_sha256": upstream_sha,
        "frozen_sha256": EXPECTED_SHA256,
        "matches": upstream_sha == EXPECTED_SHA256,
        "upstream_url": UPSTREAM_URL,
    }
    UPSTREAM_CHECK_PATH.write_text(json.dumps(result, indent=2) + "\n")
    return result


@lru_cache(maxsize=1)
def _load_frozen_csv() -> tuple[tuple[dt.date, frozenset[str]], ...]:
    """Load and parse the frozen change-log CSV.

    Verifies SHA256 against EXPECTED_SHA256 at first call. Returns rows
    sorted by date ascending. Cached for the process lifetime — the
    frozen CSV is immutable by design.
    """
    if not FROZEN_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Frozen CSV missing at {FROZEN_CSV_PATH}. See "
            f"{FROZEN_DIR / 'README.md'} for retrieval instructions."
        )
    actual = _sha256_of_file(FROZEN_CSV_PATH)
    if actual != EXPECTED_SHA256:
        raise ValueError(
            f"Frozen CSV hash mismatch. Expected {EXPECTED_SHA256}, got "
            f"{actual}. The file at {FROZEN_CSV_PATH} was modified outside "
            f"the documented retrieval flow. See {FROZEN_DIR / 'README.md'} "
            f"for the update process."
        )
    rows: list[tuple[dt.date, frozenset[str]]] = []
    with FROZEN_CSV_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row_date = dt.date.fromisoformat(r["date"])
            tickers = frozenset(r["tickers"].split(","))
            rows.append((row_date, tickers))
    # Algorithmic invariant: source rows must be strictly increasing by
    # date. members_on() walks the rows in reverse to find the nearest
    # prior anchor; a future bisect-based optimization will rely on the
    # same property. Silently sorting would mask upstream regressions, so
    # we verify and hard-fail instead.
    for i in range(1, len(rows)):
        if rows[i][0] <= rows[i - 1][0]:
            raise ValueError(
                f"Frozen CSV is not strictly monotonic-increasing by date: "
                f"row {i} date {rows[i][0]} <= row {i - 1} date "
                f"{rows[i - 1][0]}. The source file is corrupted or upstream "
                f"has regressed. See docs/design/sp500_constituents.md."
            )
    return tuple(rows)


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
