"""Tests for src/data_sources/fred_client.py.

See docs/design/fred_client.md for the spec these tests pin down.
Unit tests mock the Fred client; integration tests are marked
@pytest.mark.slow and skipped in CI.
"""

import datetime as dt
import os
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.data_sources.fred_client import (
    LATEST_TTL,
    REGIME_SERIES,
    _MISSING_KEY_MESSAGE,
    _reset_client_for_testing,
    get_features_matrix,
    get_series,
    list_series,
    validate_api_key,
)


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch, tmp_path):
    """Reset the lazy Fred client between tests so env changes take effect.

    Sets FRED_API_KEY to an empty string by default (rather than deleting
    it) because load_dotenv() runs inside _get_fred_client() and would
    re-inject any value present in .env. Empty-string + load_dotenv's
    default override=False keeps the empty value sticky, matching the
    "missing key" condition we want to test against.

    Also redirects CACHE_DIR to a per-test tmp_path so cache writes
    during tests don't pollute data/raw/fred/.
    """
    _reset_client_for_testing()
    monkeypatch.setenv("FRED_API_KEY", "")
    monkeypatch.setattr(
        "src.data_sources.fred_client.CACHE_DIR", tmp_path / "fred_cache"
    )
    yield
    _reset_client_for_testing()


def _install_mock_client(monkeypatch, fred_mock: MagicMock) -> None:
    """Helper: install a MagicMock as the cached Fred client, bypassing auth."""
    monkeypatch.setenv("FRED_API_KEY", "fake-key-for-test")
    monkeypatch.setattr("src.data_sources.fred_client._fred_client", fred_mock)


def _make_series(values: dict[str, float]) -> pd.Series:
    """Build a pd.Series with DatetimeIndex from {ISO date: value} dict."""
    return pd.Series(
        list(values.values()),
        index=pd.to_datetime(list(values.keys())),
    )


def test_regime_series_inventory_shape():
    """20 features + USREC ground truth = 21 entries; only USREC is_target."""
    assert len(REGIME_SERIES) == 21
    targets = [sid for sid, m in REGIME_SERIES.items() if m["is_target"]]
    assert targets == ["USREC"]
    # Every entry has the required metadata fields.
    required = {"name", "category", "frequency", "units", "source", "is_target", "resample"}
    for sid, meta in REGIME_SERIES.items():
        assert required <= set(meta.keys()), f"{sid} missing fields"


def test_regime_series_default_resample_is_ffill():
    """Default resample for all series is 'ffill' per phase-2 contract."""
    for sid, meta in REGIME_SERIES.items():
        assert meta["resample"] == "ffill", (
            f"{sid} has resample={meta['resample']!r}; expected 'ffill' as "
            f"the documented default. Per-series overrides should be made "
            f"deliberately, not by default."
        )


def test_list_series_returns_full_inventory():
    df = list_series()
    assert len(df) == len(REGIME_SERIES)
    # Index name and required columns present.
    assert df.index.name == "series_id"
    expected_cols = {"name", "category", "frequency", "units", "source", "is_target", "resample"}
    assert expected_cols <= set(df.columns)
    # USREC is in the unfiltered listing.
    assert "USREC" in df.index


def test_list_series_filters_by_category():
    df = list_series(category="yield_curve")
    expected_ids = {sid for sid, m in REGIME_SERIES.items() if m["category"] == "yield_curve"}
    assert set(df.index) == expected_ids
    # No mixing — none of the returned rows are from other categories.
    assert (df["category"] == "yield_curve").all()


def test_list_series_unknown_category_returns_empty():
    df = list_series(category="not_a_category")
    assert len(df) == 0


def test_validate_api_key_missing_raises_with_clear_message():
    """Unset FRED_API_KEY -> EnvironmentError with the documented message."""
    with pytest.raises(EnvironmentError) as exc:
        validate_api_key()
    msg = str(exc.value)
    assert msg == _MISSING_KEY_MESSAGE
    # Sanity-check the message contains the key cues a user needs.
    assert "FRED_API_KEY" in msg
    assert ".env.example" in msg
    assert "fred.stlouisfed.org" in msg


def test_validate_api_key_empty_string_raises(monkeypatch):
    """Empty-string FRED_API_KEY is treated identically to missing."""
    monkeypatch.setenv("FRED_API_KEY", "")
    with pytest.raises(EnvironmentError) as exc:
        validate_api_key()
    assert str(exc.value) == _MISSING_KEY_MESSAGE


def test_validate_api_key_whitespace_only_raises(monkeypatch):
    """Whitespace-only FRED_API_KEY is treated as missing (we strip)."""
    monkeypatch.setenv("FRED_API_KEY", "   ")
    with pytest.raises(EnvironmentError):
        validate_api_key()


def test_validate_api_key_present_no_probe_passes(monkeypatch):
    """Key present, probe=False -> no network call, no raise."""
    monkeypatch.setenv("FRED_API_KEY", "fake-key-for-test")
    # No mock of Fred class needed because validate_api_key(probe=False)
    # only instantiates the Fred client (which doesn't hit the network)
    # and returns.
    validate_api_key(probe=False)


def test_validate_api_key_probe_rejected_raises_value_error(monkeypatch):
    """Probe + Fred rejects key -> ValueError, not EnvironmentError."""
    monkeypatch.setenv("FRED_API_KEY", "fake-bad-key")

    # Mock the Fred class so its get_series_info raises an "api key" error
    fake_client = MagicMock()
    fake_client.get_series_info.side_effect = ValueError("Bad Request. The API key is invalid.")
    monkeypatch.setattr("src.data_sources.fred_client._fred_client", fake_client)

    with pytest.raises(ValueError) as exc:
        validate_api_key(probe=True)
    assert "rejected by FRED" in str(exc.value)


def test_validate_api_key_probe_non_auth_error_propagates(monkeypatch):
    """A non-auth error during probe should bubble up unchanged."""
    monkeypatch.setenv("FRED_API_KEY", "fake-key")

    fake_client = MagicMock()
    fake_client.get_series_info.side_effect = RuntimeError("Network unreachable")
    monkeypatch.setattr("src.data_sources.fred_client._fred_client", fake_client)

    with pytest.raises(RuntimeError, match="Network unreachable"):
        validate_api_key(probe=True)


# ---------- get_series -----------------------------------------------------


def test_get_series_latest_passes_no_realtime_params(monkeypatch):
    """Latest queries (vintage_date=None) must not pass realtime_*."""
    fred = MagicMock()
    fred.get_series.return_value = _make_series({"2024-01-01": 4.0})
    _install_mock_client(monkeypatch, fred)

    get_series("UNRATE", start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 31))

    fred.get_series.assert_called_once()
    kwargs = fred.get_series.call_args.kwargs
    assert kwargs.get("observation_start") == "2024-01-01"
    assert kwargs.get("observation_end") == "2024-01-31"
    assert "realtime_start" not in kwargs
    assert "realtime_end" not in kwargs


def test_get_series_with_vintage_sets_both_realtime_params(monkeypatch):
    """Vintage queries set realtime_start AND realtime_end to the SAME date.

    Setting only realtime_end returns cumulative revision history, which
    is not the point-in-time snapshot we need to prevent look-ahead bias.
    See docs/design/fred_client.md "Vintage handling".
    """
    fred = MagicMock()
    fred.get_series.return_value = _make_series({"2018-01-01": 4.1})
    _install_mock_client(monkeypatch, fred)

    get_series(
        "UNRATE",
        start=dt.date(2018, 1, 1),
        end=dt.date(2018, 12, 31),
        vintage_date=dt.date(2018, 12, 31),
    )

    kwargs = fred.get_series.call_args.kwargs
    assert kwargs.get("realtime_start") == "2018-12-31"
    assert kwargs.get("realtime_end") == "2018-12-31"
    assert kwargs["realtime_start"] == kwargs["realtime_end"]


def test_get_series_unknown_id_raises_clear_error(monkeypatch):
    """FRED 'does not exist' translates to ValueError naming the bad ID."""
    fred = MagicMock()
    fred.get_series.side_effect = ValueError(
        "Bad Request. The series does not exist."
    )
    _install_mock_client(monkeypatch, fred)

    with pytest.raises(ValueError) as exc:
        get_series("NOTAREAL")
    assert "NOTAREAL" in str(exc.value)
    assert "fred.stlouisfed.org/series/NOTAREAL" in str(exc.value)


def test_get_series_empty_range_returns_empty_no_raise(monkeypatch, caplog):
    """Empty observation range returns empty Series and logs INFO, not WARNING."""
    fred = MagicMock()
    fred.get_series.return_value = pd.Series(dtype="float64")
    _install_mock_client(monkeypatch, fred)

    with caplog.at_level("INFO", logger="src.data_sources.fred_client"):
        result = get_series("UNRATE")
    assert result.empty
    info_records = [r for r in caplog.records if r.levelname == "INFO"]
    warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("no observations" in r.message for r in info_records)
    assert not warning_records


def test_get_series_empty_rangeindex_does_not_crash(monkeypatch):
    """Regression: empty Series with RangeIndex must not raise.

    fredapi returns an empty Series with a default RangeIndex (int64) —
    not an empty DatetimeIndex — when FRED has no observations in the
    requested range. The original implementation ran
    `series[series.index >= pd.Timestamp(start)]` unconditionally,
    crashing with TypeError because RangeIndex(int64) >= Timestamp is
    not defined. Short-circuiting empty before the filter fixes it.

    Triggered in practice by BAMLH0A0HYM2 after ICE Data Indices
    restricted FRED's historical access in late 2024. See
    docs/design/fred_client.md credit-section note.
    """
    fred = MagicMock()
    # Empty Series with default RangeIndex — the fredapi return shape
    # we observed for ICE BofA series with truncated history.
    fred.get_series.return_value = pd.Series(dtype="object")
    _install_mock_client(monkeypatch, fred)

    # No raise. Returns empty.
    result = get_series(
        "BAMLH0A0HYM2",
        start=dt.date(2020, 1, 1),
        end=dt.date(2020, 12, 31),
    )
    assert result.empty


def test_get_series_missing_at_vintage_returns_empty_with_warning(monkeypatch, caplog):
    """Empty vintage response returns empty Series and logs WARNING
    naming series_id and vintage_date. Does not raise.

    Models walk-forward CV requesting a series that hadn't started yet at
    the fold's vintage (e.g., BAMLH0A0HYM2 before 1997).
    """
    fred = MagicMock()
    fred.get_series.return_value = pd.Series(dtype="float64")
    _install_mock_client(monkeypatch, fred)

    with caplog.at_level("WARNING", logger="src.data_sources.fred_client"):
        result = get_series("BAMLH0A0HYM2", vintage_date=dt.date(1995, 6, 30))
    assert result.empty
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "BAMLH0A0HYM2" in warnings[0].message
    assert "1995-06-30" in warnings[0].message


def test_get_series_caches_latest_then_skips_api(monkeypatch):
    """Second call to the same latest query should not hit FRED."""
    fred = MagicMock()
    fred.get_series.return_value = _make_series({"2024-06-30": 4.0})
    _install_mock_client(monkeypatch, fred)

    get_series("UNRATE")
    get_series("UNRATE")

    assert fred.get_series.call_count == 1


def test_get_series_latest_cache_expires_after_ttl(monkeypatch):
    """Latest cache older than LATEST_TTL triggers a fresh fetch."""
    fred = MagicMock()
    fred.get_series.return_value = _make_series({"2024-06-30": 4.0})
    _install_mock_client(monkeypatch, fred)

    get_series("UNRATE")
    # Backdate the cache file's mtime to >TTL ago.
    from src.data_sources.fred_client import _cache_path

    path = _cache_path("UNRATE", None)
    stale = (
        dt.datetime.now(dt.timezone.utc) - LATEST_TTL - dt.timedelta(hours=1)
    ).timestamp()
    os.utime(path, (stale, stale))

    get_series("UNRATE")
    assert fred.get_series.call_count == 2


def test_get_series_cache_invalidated_when_narrower_than_request(monkeypatch):
    """Cache populated by a narrow-range fetch must NOT be served for a
    wider-range request — even if within TTL.

    Regression for the silent data corruption surfaced by the regime
    label diagnostic: UNRATE was cached with data only through 2020-12
    (from an earlier slow integration test), and a later
    get_features_matrix(start=2014, end=2025) hit that cache (within
    24h TTL) and got back a Series missing all post-2020 data. The
    matrix's ffill resample then propagated stale December-2020
    values across all later month-ends, looking superficially right
    but being silently wrong.
    """
    narrow = _make_series({"2020-01-15": 4.0, "2020-12-15": 6.7})
    wide = _make_series({
        "2014-01-15": 6.6,
        "2020-01-15": 4.0,
        "2020-12-15": 6.7,
        "2025-12-15": 4.2,
    })

    fred = MagicMock()
    fred.get_series.side_effect = [narrow, wide]
    _install_mock_client(monkeypatch, fred)

    # First call: narrow range -> populates cache with narrow data.
    first = get_series(
        "UNRATE",
        start=dt.date(2020, 1, 1),
        end=dt.date(2020, 12, 31),
    )
    assert first.index.min().year == 2020
    assert fred.get_series.call_count == 1

    # Second call: wider range. The cache covers only 2020 but we want
    # 2014-2025. _cache_is_fresh must reject the cache and trigger
    # refetch.
    second = get_series(
        "UNRATE",
        start=dt.date(2014, 1, 1),
        end=dt.date(2025, 12, 31),
    )
    assert fred.get_series.call_count == 2, (
        "Cache should have been invalidated due to insufficient range "
        "coverage; expected a second refetch."
    )
    # Result includes pre-2020 data (the bug would have returned only
    # 2020 data forward-filled).
    assert second.index.min().year == 2014


def test_get_series_cache_serves_request_within_cached_range(monkeypatch):
    """Inverse of the regression: when the cache covers the requested
    range (cache 2014-2025, request 2020), serve from cache."""
    wide = _make_series({
        "2014-01-15": 1.0,
        "2020-06-15": 2.0,
        "2025-12-15": 3.0,
    })
    fred = MagicMock()
    fred.get_series.return_value = wide
    _install_mock_client(monkeypatch, fred)

    # First: wide-range fetch populates cache.
    get_series(
        "UNRATE", start=dt.date(2014, 1, 1), end=dt.date(2025, 12, 31)
    )
    # Second: narrower range fully inside the cached range.
    get_series(
        "UNRATE", start=dt.date(2020, 1, 1), end=dt.date(2020, 12, 31)
    )

    # Cache hit on the second call -> still only one underlying fetch.
    assert fred.get_series.call_count == 1


def test_get_series_cache_invalidated_when_start_before_cached(monkeypatch):
    """Cache covers 2020-2025; request for 2014-2025 must refetch."""
    later = _make_series({"2020-01-15": 1.0, "2025-12-15": 2.0})
    earlier_plus_later = _make_series({
        "2014-01-15": 0.5,
        "2020-01-15": 1.0,
        "2025-12-15": 2.0,
    })
    fred = MagicMock()
    fred.get_series.side_effect = [later, earlier_plus_later]
    _install_mock_client(monkeypatch, fred)

    get_series("UNRATE", start=dt.date(2020, 1, 1), end=dt.date(2025, 12, 31))
    get_series("UNRATE", start=dt.date(2014, 1, 1), end=dt.date(2025, 12, 31))

    assert fred.get_series.call_count == 2


def test_get_series_cache_invalidated_when_end_after_cached(monkeypatch):
    """Cache covers 2014-2020; request for 2014-2025 must refetch."""
    earlier = _make_series({"2014-01-15": 0.5, "2020-12-15": 1.5})
    full = _make_series({
        "2014-01-15": 0.5,
        "2020-12-15": 1.5,
        "2025-12-15": 2.5,
    })
    fred = MagicMock()
    fred.get_series.side_effect = [earlier, full]
    _install_mock_client(monkeypatch, fred)

    get_series("UNRATE", start=dt.date(2014, 1, 1), end=dt.date(2020, 12, 31))
    get_series("UNRATE", start=dt.date(2014, 1, 1), end=dt.date(2025, 12, 31))

    assert fred.get_series.call_count == 2


def test_get_series_vintage_cache_never_expires(monkeypatch):
    """Vintage cache is used regardless of file age."""
    fred = MagicMock()
    fred.get_series.return_value = _make_series({"2018-06-01": 3.9})
    _install_mock_client(monkeypatch, fred)

    vintage = dt.date(2018, 12, 31)
    get_series("UNRATE", vintage_date=vintage)

    # Make the cache ancient — way past any TTL we'd ever pick.
    from src.data_sources.fred_client import _cache_path

    path = _cache_path("UNRATE", vintage)
    ancient = (
        dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=365)
    ).timestamp()
    os.utime(path, (ancient, ancient))

    get_series("UNRATE", vintage_date=vintage)
    assert fred.get_series.call_count == 1


# ---------- get_features_matrix -------------------------------------------


def _install_mock_get_series(monkeypatch, fn):
    """Replace module-level get_series with `fn`. Caller controls behavior."""
    monkeypatch.setattr("src.data_sources.fred_client.get_series", fn)


def test_features_matrix_excludes_usrec_by_default(monkeypatch):
    """Label-leakage guardrail: USREC must be absent from default columns."""

    def fake(series_id, start=None, end=None, vintage_date=None):
        return _make_series({"2024-01-15": 1.0, "2024-12-15": 1.0})

    _install_mock_get_series(monkeypatch, fake)

    matrix = get_features_matrix(
        start=dt.date(2024, 1, 1), end=dt.date(2024, 12, 31)
    )
    assert "USREC" not in matrix.columns


def test_features_matrix_includes_usrec_when_exclude_targets_false(monkeypatch):
    """Override exclude_targets=False to include USREC (for label construction)."""

    def fake(series_id, start=None, end=None, vintage_date=None):
        return _make_series({"2024-01-15": 1.0, "2024-12-15": 1.0})

    _install_mock_get_series(monkeypatch, fake)

    matrix = get_features_matrix(
        start=dt.date(2024, 1, 1),
        end=dt.date(2024, 12, 31),
        exclude_targets=False,
    )
    assert "USREC" in matrix.columns


def test_features_matrix_passes_vintage_to_each_get_series_call(monkeypatch):
    """Look-ahead-bias guardrail: every underlying call gets vintage_date."""
    calls: list[dict] = []

    def fake(series_id, start=None, end=None, vintage_date=None):
        calls.append({"series_id": series_id, "vintage_date": vintage_date})
        return _make_series({"2024-01-15": 1.0})

    _install_mock_get_series(monkeypatch, fake)

    vintage = dt.date(2024, 6, 30)
    get_features_matrix(
        series_ids=["DGS10", "UNRATE", "PAYEMS"],
        start=dt.date(2024, 1, 1),
        end=dt.date(2024, 1, 31),
        vintage_date=vintage,
    )
    assert len(calls) == 3
    for c in calls:
        assert c["vintage_date"] == vintage, (
            f"Call for {c['series_id']} did not receive vintage_date"
        )


def test_features_matrix_aligns_daily_to_month_end(monkeypatch):
    """Daily series resolves to one row per month-end."""
    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
    daily = pd.Series(range(len(dates)), index=dates, dtype="float64")

    def fake(series_id, start=None, end=None, vintage_date=None):
        return daily if series_id == "DGS10" else pd.Series(dtype="float64")

    _install_mock_get_series(monkeypatch, fake)

    matrix = get_features_matrix(
        series_ids=["DGS10"],
        start=dt.date(2024, 1, 1),
        end=dt.date(2024, 12, 31),
    )
    # 12 calendar month-ends in 2024.
    assert len(matrix) == 12
    # The Jan 31 value is the daily series' value on Jan 31.
    assert matrix.loc["2024-01-31", "DGS10"] == daily.loc["2024-01-31"]
    # The Dec 31 value is the daily series' value on Dec 31.
    assert matrix.loc["2024-12-31", "DGS10"] == daily.loc["2024-12-31"]


def test_features_matrix_aligns_weekly_to_month_end(monkeypatch):
    """Weekly series resolves to one row per month-end (last weekly obs)."""
    weekly_dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="W-FRI")
    weekly = pd.Series(range(len(weekly_dates)), index=weekly_dates, dtype="float64")

    def fake(series_id, start=None, end=None, vintage_date=None):
        return weekly

    _install_mock_get_series(monkeypatch, fake)

    matrix = get_features_matrix(
        series_ids=["ICSA"],
        start=dt.date(2024, 1, 1),
        end=dt.date(2024, 12, 31),
    )
    assert len(matrix) == 12
    # Jan 31 should reflect the last Friday in January 2024 (Jan 26).
    last_fri_jan = weekly[weekly.index <= "2024-01-31"].iloc[-1]
    assert matrix.loc["2024-01-31", "ICSA"] == last_fri_jan


def test_features_matrix_aligns_monthly_to_month_end(monkeypatch):
    """Monthly series is passed through, one obs per month, no ffill needed."""
    # FRED publishes monthly series dated to the first of the month.
    monthly_dates = pd.date_range(start="2024-01-01", end="2024-12-01", freq="MS")
    monthly = pd.Series(range(len(monthly_dates)), index=monthly_dates, dtype="float64")

    def fake(series_id, start=None, end=None, vintage_date=None):
        return monthly

    _install_mock_get_series(monkeypatch, fake)

    matrix = get_features_matrix(
        series_ids=["UNRATE"],
        start=dt.date(2024, 1, 1),
        end=dt.date(2024, 12, 31),
    )
    assert len(matrix) == 12
    # Each month's value comes through.
    assert matrix.loc["2024-01-31", "UNRATE"] == 0
    assert matrix.loc["2024-12-31", "UNRATE"] == 11


def test_features_matrix_per_series_override_uses_mean(monkeypatch):
    """REGIME_SERIES[id]['resample']='mean' switches to within-month average."""
    # Construct a January series with two distinct halves so mean is non-trivial.
    dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="D")
    values = [10.0] * 15 + [20.0] * (len(dates) - 15)
    series = pd.Series(values, index=dates, dtype="float64")

    def fake(series_id, start=None, end=None, vintage_date=None):
        return series

    _install_mock_get_series(monkeypatch, fake)
    monkeypatch.setitem(REGIME_SERIES["DGS10"], "resample", "mean")

    matrix = get_features_matrix(
        series_ids=["DGS10"],
        start=dt.date(2024, 1, 1),
        end=dt.date(2024, 1, 31),
    )
    # Mean of the 31-day window; clearly distinct from "last" (which would be 20.0).
    expected = series.mean()
    assert abs(matrix.loc["2024-01-31", "DGS10"] - expected) < 1e-6
    assert matrix.loc["2024-01-31", "DGS10"] != 20.0


def test_features_matrix_partial_failure_warns_and_continues(monkeypatch, caplog):
    """3 of 20 features failing (15% < 25% threshold) returns a 17-col matrix.

    Failed series are absent from the result and a single summary WARNING
    names all of them. Below-threshold partial failures must not abort.
    """
    failing = {"BAA10Y", "VIXCLS", "HOUST"}

    def fake(series_id, start=None, end=None, vintage_date=None):
        if series_id in failing:
            raise ValueError(f"Mocked FRED failure for {series_id}")
        return _make_series({"2024-01-15": 1.0, "2024-12-15": 1.0})

    _install_mock_get_series(monkeypatch, fake)

    with caplog.at_level("WARNING", logger="src.data_sources.fred_client"):
        matrix = get_features_matrix(
            start=dt.date(2024, 1, 1), end=dt.date(2024, 12, 31)
        )

    # 20 features - 3 failed = 17 columns.
    assert matrix.shape[1] == 17
    for sid in failing:
        assert sid not in matrix.columns

    # Summary WARNING names all three failed series.
    warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
    summary = [m for m in warnings if "feature-matrix returning" in m]
    assert len(summary) == 1
    for sid in failing:
        assert sid in summary[0]


def test_features_matrix_logs_full_traceback_on_code_bug(monkeypatch, caplog):
    """A code-level exception (TypeError, etc.) must log with full traceback.

    Without exc_info, code bugs in our wrapper masquerade as soft
    missing-data conditions in the per-series WARNING log. The
    BAMLH0A0HYM2 RangeIndex bug lived for weeks because its one-line
    WARNING ("'>=' not supported between numpy.ndarray and Timestamp")
    looked superficially similar to a FRED data issue. The traceback
    makes the distinction unambiguous.
    """

    def fake(series_id, start=None, end=None, vintage_date=None):
        if series_id == "DGS10":
            # Simulate a code-bug exception. Realistic shape: the kind
            # of TypeError that arose from the original BAMLH0A0HYM2 bug.
            raise TypeError(
                "'>=' not supported between instances of "
                "'numpy.ndarray' and 'Timestamp'"
            )
        return _make_series({"2024-01-15": 1.0})

    _install_mock_get_series(monkeypatch, fake)

    with caplog.at_level("WARNING", logger="src.data_sources.fred_client"):
        get_features_matrix(
            start=dt.date(2024, 1, 1), end=dt.date(2024, 12, 31)
        )

    # The per-series WARNING (distinct from the summary WARNING that
    # lists missing series) must include the traceback. caplog's
    # LogRecord.exc_info is non-None iff exc_info=True was passed.
    per_series_warnings = [
        r for r in caplog.records
        if r.levelname == "WARNING" and "raised an exception" in r.message
    ]
    assert len(per_series_warnings) == 1
    record = per_series_warnings[0]
    assert "DGS10" in record.message
    assert record.exc_info is not None, (
        "Per-series WARNING for an exception must include exc_info so "
        "the traceback is logged. Without it, code bugs hide as soft "
        "missing-data conditions."
    )
    exc_type, _, _ = record.exc_info
    assert exc_type is TypeError
    # The message itself should also be unambiguous about this being
    # an exception path, not a missing-data path.
    assert "not a soft missing-data condition" in record.message


def test_features_matrix_majority_failure_raises(monkeypatch):
    """6 of 20 features failing (30% >= 25% threshold) aborts with RuntimeError.

    The error message must name every failed series so the caller can
    diagnose the systemic issue (FRED outage? API key revoked?).
    """
    failing = {"BAA10Y", "VIXCLS", "HOUST", "DGS10", "UNRATE", "PAYEMS"}

    def fake(series_id, start=None, end=None, vintage_date=None):
        if series_id in failing:
            raise ValueError(f"Mocked FRED failure for {series_id}")
        return _make_series({"2024-01-15": 1.0})

    _install_mock_get_series(monkeypatch, fake)

    with pytest.raises(RuntimeError) as exc:
        get_features_matrix(
            start=dt.date(2024, 1, 1), end=dt.date(2024, 12, 31)
        )

    msg = str(exc.value)
    assert "6 of 20" in msg
    assert "30%" in msg
    assert "25% threshold" in msg
    for sid in failing:
        assert sid in msg


def test_features_matrix_smoke_2020(monkeypatch):
    """Shape (12, 20), every expected feature column present, no NaN rows."""
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    daily = pd.Series(range(len(dates)), index=dates, dtype="float64")

    def fake(series_id, start=None, end=None, vintage_date=None):
        return daily

    _install_mock_get_series(monkeypatch, fake)

    matrix = get_features_matrix(
        start=dt.date(2020, 1, 1), end=dt.date(2020, 12, 31)
    )
    assert matrix.shape == (12, 20)
    expected_cols = {
        sid for sid, meta in REGIME_SERIES.items() if not meta["is_target"]
    }
    assert set(matrix.columns) == expected_cols
    assert not matrix.isna().any().any()


# ---------- Integration (real FRED API, slow) -----------------------------


@pytest.mark.slow
def test_real_api_anchor_values(monkeypatch):
    """End-to-end FRED call against known historical anchor values.

    Requires FRED_API_KEY in .env. Skipped (not failed) if missing —
    this test exists to prove the wrapper works against real FRED, not
    to gate CI on a credential the host may not have.

    Anchors chosen from the design doc validation section:
      - T10Y2Y inverted in August 2019 (pre-COVID recession signal)
      - UNRATE = 14.7% in April 2020 (COVID spike, largest single-month
        BLS jump on record)
    """
    from dotenv import load_dotenv

    # override=True is required: the autouse _reset_state fixture sets
    # FRED_API_KEY="" before every test for unit-test isolation, and
    # load_dotenv defaults to override=False (it won't replace an already
    # present env var). Without override we'd always read "" and skip
    # even when .env has a real key.
    load_dotenv(override=True)
    real_key = os.environ.get("FRED_API_KEY", "").strip()
    if not real_key:
        pytest.skip("FRED_API_KEY not set in .env")
    monkeypatch.setenv("FRED_API_KEY", real_key)
    _reset_client_for_testing()

    t10y2y = get_series(
        "T10Y2Y", start=dt.date(2019, 8, 1), end=dt.date(2019, 8, 31)
    )
    assert not t10y2y.empty
    assert (t10y2y < 0).any(), (
        "T10Y2Y should be inverted on at least one day in August 2019"
    )

    # Latest-revision UNRATE for April 2020.
    # BLS revises early-release unemployment figures as more data
    # arrives. Initial release (May 2020) was 14.7%; current revised
    # value is 14.8% as of 2026. Tolerance accommodates further
    # revisions; precision is asserted by the vintage query below.
    unrate_latest = get_series(
        "UNRATE", start=dt.date(2020, 4, 1), end=dt.date(2020, 4, 30)
    )
    assert not unrate_latest.empty
    assert abs(unrate_latest.iloc[0] - 14.7) < 0.2, (
        f"UNRATE for April 2020 expected ~14.7% (or a small revision "
        f"thereof), got {unrate_latest.iloc[0]}"
    )

    # Vintage-aware UNRATE for April 2020 as known on 2020-05-15.
    # This is the LOAD-BEARING look-ahead-bias guardrail at the
    # integration level. A walk-forward CV fold ending in mid-2020
    # must see UNRATE as it was published then (14.7%), not as it
    # was later revised (14.8%). If this assertion ever fails, the
    # vintage code is silently corrupted and all regime-model
    # training is invalid.
    #
    # 2020-05-15 is chosen because it's ~1 month after the May 8
    # initial release, long enough for the first-release value to
    # be the dominant revision but before any subsequent benchmark
    # revisions that would have moved it.
    unrate_vintage = get_series(
        "UNRATE",
        start=dt.date(2020, 4, 1),
        end=dt.date(2020, 4, 30),
        vintage_date=dt.date(2020, 5, 15),
    )
    assert not unrate_vintage.empty
    assert abs(unrate_vintage.iloc[0] - 14.7) < 0.05, (
        f"UNRATE for April 2020 as-of vintage 2020-05-15 expected "
        f"~14.7% (first release), got {unrate_vintage.iloc[0]}. "
        f"If this fails, the vintage query is returning the LATEST "
        f"revision instead of the point-in-time value — this is the "
        f"look-ahead-bias guardrail and is load-bearing."
    )

    # NFCICREDIT (Chicago Fed Credit Subindex) replaces the historical
    # ICE BofA HY OAS series after ICE restricted FRED's access in
    # late 2024. This anchor verifies the replacement actually behaves
    # like a credit-stress index: it must rise sharply during the COVID
    # credit shock of March-April 2020. The series was -0.10 in early
    # January 2020 (looser-than-average credit), peaked at +0.63 on
    # 2020-04-10, and stayed elevated through May. The threshold > 0.5
    # is well above any normal-times reading and unambiguous about
    # which direction the series moved.
    nfci_credit = get_series(
        "NFCICREDIT", start=dt.date(2020, 3, 1), end=dt.date(2020, 5, 1)
    )
    assert not nfci_credit.empty
    assert nfci_credit.max() > 0.5, (
        f"NFCICREDIT max during COVID stress window (Mar-May 2020) "
        f"expected > 0.5; got {nfci_credit.max():.3f}. The series "
        f"should be visibly elevated during the worst credit-stress "
        f"event of the test window. If this fails the replacement "
        f"series may have been further restricted or relocated."
    )
