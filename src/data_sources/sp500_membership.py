"""Point-in-time S&P 500 membership via change-log reconstruction.

See docs/design/sp500_constituents.md for the full design.

When bumping the frozen CSV, update FROZEN_CSV_FILENAME, EXPECTED_SHA256,
and data/raw/sp500_change_log/README.md together. The hash-consistency
test enforces all three stay in sync.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import logging
from functools import lru_cache
from pathlib import Path

import pyarrow as pa
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


def build_membership_table(start: dt.date, end: dt.date) -> pa.Table:
    """Build a long-format pyarrow Table of (as_of_date, ticker) rows. Phase 2."""
    raise NotImplementedError


def load_membership_table() -> pa.Table:
    """Load the materialized membership parquet, building if missing. Phase 2."""
    raise NotImplementedError


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
