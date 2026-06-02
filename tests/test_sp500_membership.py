"""Tests for src/data_sources/sp500_membership.py.

Validation events use S&P-effective removal dates (the date the index
roster changed), not news-salient bank-failure dates. See
docs/design/sp500_constituents.md for the rationale.
"""

import datetime as dt
import hashlib
import re
from pathlib import Path

import pytest

from src.data_sources.sp500_membership import (
    EXPECTED_SHA256,
    FROZEN_CSV_PATH,
    FROZEN_DIR,
    members_on,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def test_frozen_csv_hash_consistency():
    """File hash, README hash, and EXPECTED_SHA256 constant must all match."""
    actual = _sha256(FROZEN_CSV_PATH)
    readme = (FROZEN_DIR / "README.md").read_text()
    match = re.search(r"SHA256\s*\|\s*`([a-f0-9]{64})`", readme)
    assert match is not None, "Could not parse SHA256 line from README.md"
    readme_sha = match.group(1)
    assert actual == EXPECTED_SHA256 == readme_sha


def test_members_on_before_2014_raises():
    with pytest.raises(ValueError) as exc:
        members_on(dt.date(2013, 12, 31))
    msg = str(exc.value)
    assert "2013-12-31" in msg
    assert "supported training window (2014-01-01)" in msg
    assert "Pre-2014" in msg
    assert "docs/design/sp500_constituents.md" in msg


def test_members_on_returns_nearest_prior_anchor():
    # Both 2024-04-04 and 2024-04-05 fall between change events
    # 2024-04-03 (SOLV/GEV added) and 2024-05-08 (VST added). Both should
    # resolve to the same anchor and return identical membership sets.
    set1 = members_on(dt.date(2024, 4, 4))
    set2 = members_on(dt.date(2024, 4, 5))
    assert set1 == set2


def test_members_on_known_date_2014_06_30():
    """Cross-check values from the design doc."""
    members = members_on(dt.date(2014, 6, 30))
    assert len(members) == 498
    assert "AAPL" in members
    assert "MSFT" in members
    assert "FB" in members  # Facebook before its 2022 rename to META
    assert "LEH" not in members  # Lehman, defunct 2008
    assert "META" not in members  # Ticker did not exist until 2022


def test_known_event_svb_effective_date():
    # SIVB was in the index through 2023-03-14 and removed effective
    # 2023-03-15. This is the S&P-effective removal date, NOT the FDIC
    # closure date (2023-03-10) or trading-halt date (2023-03-13). See
    # docs/design/sp500_constituents.md "Validation" section.
    assert "SIVB" in members_on(dt.date(2023, 3, 14))
    assert "SIVB" not in members_on(dt.date(2023, 3, 15))


def test_known_event_frc_effective_date():
    # FRC was removed effective 2023-05-04, the trading day after the
    # FDIC seizure on 2023-05-01. We use the S&P-effective date, not
    # the seizure date — see docs/design/sp500_constituents.md.
    assert "FRC" in members_on(dt.date(2023, 5, 3))
    assert "FRC" not in members_on(dt.date(2023, 5, 4))


def test_known_event_tsla_addition():
    # Tesla was added to the S&P 500 effective 2020-12-21 open.
    assert "TSLA" not in members_on(dt.date(2020, 12, 18))
    assert "TSLA" in members_on(dt.date(2020, 12, 21))


def test_known_event_fb_meta_rename():
    # The FB->META ticker change was atomic on 2022-06-09.
    #
    # This test asserts only ticker-level membership. Continuity of the
    # underlying security across the rename (so returns can be chained
    # across the June 8/9 boundary) is the returns module's
    # responsibility, not membership's.
    assert "FB" in members_on(dt.date(2022, 6, 8))
    assert "META" in members_on(dt.date(2022, 6, 9))
    assert "FB" not in members_on(dt.date(2022, 6, 9))
