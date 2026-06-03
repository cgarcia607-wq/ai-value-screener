"""Tests for src/data_sources/fred_client.py.

See docs/design/fred_client.md for the spec these tests pin down.
Unit tests mock the Fred client; integration tests are marked
@pytest.mark.slow and skipped in CI.
"""

from unittest.mock import MagicMock

import pytest

from src.data_sources.fred_client import (
    REGIME_SERIES,
    _MISSING_KEY_MESSAGE,
    _reset_client_for_testing,
    validate_api_key,
)


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Reset the lazy Fred client between tests so env changes take effect.

    Sets FRED_API_KEY to an empty string by default (rather than deleting
    it) because load_dotenv() runs inside _get_fred_client() and would
    re-inject any value present in .env. Empty-string + load_dotenv's
    default override=False keeps the empty value sticky, matching the
    "missing key" condition we want to test against.
    """
    _reset_client_for_testing()
    monkeypatch.setenv("FRED_API_KEY", "")
    yield
    _reset_client_for_testing()


def test_regime_series_inventory_shape():
    """20 features + USREC ground truth = 21 entries; only USREC is_target."""
    assert len(REGIME_SERIES) == 21
    targets = [sid for sid, m in REGIME_SERIES.items() if m["is_target"]]
    assert targets == ["USREC"]
    # Every entry has the required metadata fields.
    required = {"name", "category", "frequency", "units", "source", "is_target", "resample"}
    for sid, meta in REGIME_SERIES.items():
        assert required <= set(meta.keys()), f"{sid} missing fields"


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
