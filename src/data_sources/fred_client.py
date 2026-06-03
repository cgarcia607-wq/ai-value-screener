"""Cached FRED API wrapper for macro/credit series.

See docs/design/fred_client.md for the full design — vintage handling,
caching policy, USREC label-leakage warning, and the series inventory
rationale.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = _REPO_ROOT / "data" / "raw" / "fred" / "observations"
LATEST_TTL = dt.timedelta(hours=24)

# Series inventory. Keyed by FRED series ID. `is_target=True` marks
# USREC as ground truth — NEVER include it in the feature matrix.
# `resample` is the per-series rule for monthly harmonization;
# "last" means as-of-month-end forward-fill (the default).
#
# See docs/design/fred_client.md for the full rationale per series.
REGIME_SERIES: dict[str, dict] = {
    # --- Yield curve --------------------------------------------------
    "DGS10": {
        "name": "10-Year Treasury Constant Maturity Rate",
        "category": "yield_curve",
        "frequency": "D", "units": "%",
        "source": "Board of Governors via FRED",
        "is_target": False, "resample": "last",
    },
    "DGS2": {
        "name": "2-Year Treasury Constant Maturity Rate",
        "category": "yield_curve",
        "frequency": "D", "units": "%",
        "source": "Board of Governors via FRED",
        "is_target": False, "resample": "last",
    },
    "T10Y2Y": {
        "name": "10-Year minus 2-Year Treasury Spread",
        "category": "yield_curve",
        "frequency": "D", "units": "%",
        "source": "FRED (computed from DGS10/DGS2)",
        "is_target": False, "resample": "last",
    },
    "T10Y3M": {
        "name": "10-Year minus 3-Month Treasury Spread",
        "category": "yield_curve",
        "frequency": "D", "units": "%",
        "source": "FRED (computed)",
        "is_target": False, "resample": "last",
    },
    # --- Credit -------------------------------------------------------
    "BAMLH0A0HYM2": {
        "name": "ICE BofA US High Yield Index OAS",
        "category": "credit",
        "frequency": "D", "units": "%",
        "source": "ICE Data Indices via FRED",
        "is_target": False, "resample": "last",
    },
    "BAA10Y": {
        "name": "Moody's Baa Corporate Bond Yield minus 10Y Treasury",
        "category": "credit",
        "frequency": "D", "units": "%",
        "source": "Moody's / FRED",
        "is_target": False, "resample": "last",
    },
    # --- Activity / Labor --------------------------------------------
    "UNRATE": {
        "name": "Unemployment Rate",
        "category": "activity",
        "frequency": "M", "units": "%",
        "source": "BLS via FRED",
        "is_target": False, "resample": "last",
    },
    "ICSA": {
        "name": "Initial Jobless Claims",
        "category": "activity",
        "frequency": "W", "units": "count",
        "source": "DOL via FRED",
        "is_target": False, "resample": "last",
    },
    "PAYEMS": {
        "name": "Total Nonfarm Payrolls",
        "category": "activity",
        "frequency": "M", "units": "thousand",
        "source": "BLS via FRED",
        "is_target": False, "resample": "last",
    },
    "INDPRO": {
        "name": "Industrial Production Index",
        "category": "activity",
        "frequency": "M", "units": "index",
        "source": "Board of Governors via FRED",
        "is_target": False, "resample": "last",
    },
    "HOUST": {
        "name": "Housing Starts",
        "category": "activity",
        "frequency": "M", "units": "thousand SAAR",
        "source": "Census Bureau via FRED",
        "is_target": False, "resample": "last",
    },
    # --- Rates / Money -----------------------------------------------
    "FEDFUNDS": {
        "name": "Effective Federal Funds Rate (monthly)",
        "category": "rates",
        "frequency": "M", "units": "%",
        "source": "Board of Governors via FRED",
        "is_target": False, "resample": "last",
    },
    "M2SL": {
        "name": "M2 Money Stock",
        "category": "rates",
        "frequency": "M", "units": "$ billion",
        "source": "Board of Governors via FRED",
        "is_target": False, "resample": "last",
    },
    # --- Inflation ---------------------------------------------------
    "CPIAUCSL": {
        "name": "Consumer Price Index, All Urban Consumers",
        "category": "inflation",
        "frequency": "M", "units": "index",
        "source": "BLS via FRED",
        "is_target": False, "resample": "last",
    },
    "CPILFESL": {
        "name": "Core CPI (ex food and energy)",
        "category": "inflation",
        "frequency": "M", "units": "index",
        "source": "BLS via FRED",
        "is_target": False, "resample": "last",
    },
    "PCEPILFE": {
        "name": "Core PCE Price Index",
        "category": "inflation",
        "frequency": "M", "units": "index",
        "source": "BEA via FRED",
        "is_target": False, "resample": "last",
    },
    # --- Market internals --------------------------------------------
    "VIXCLS": {
        "name": "CBOE Volatility Index (VIX)",
        "category": "market_internals",
        "frequency": "D", "units": "level",
        "source": "CBOE via FRED",
        "is_target": False, "resample": "last",
    },
    "DTWEXBGS": {
        "name": "Trade-Weighted Broad Dollar Index",
        "category": "market_internals",
        "frequency": "D", "units": "index",
        "source": "Board of Governors via FRED",
        "is_target": False, "resample": "last",
    },
    "DCOILWTICO": {
        "name": "WTI Crude Oil Spot Price",
        "category": "market_internals",
        "frequency": "D", "units": "USD/bbl",
        "source": "EIA via FRED",
        "is_target": False, "resample": "last",
    },
    "UMCSENT": {
        "name": "University of Michigan Consumer Sentiment",
        "category": "market_internals",
        "frequency": "M", "units": "index",
        "source": "University of Michigan via FRED",
        "is_target": False, "resample": "last",
    },
    # --- Ground truth (label, NOT a feature) -------------------------
    # USREC is the NBER-based recession indicator. It is TARGET DATA
    # for the regime classifier and must NEVER be included in the
    # feature matrix. get_features_matrix() excludes it by default
    # via the is_target flag.
    "USREC": {
        "name": "NBER-based Recession Indicator",
        "category": "ground_truth",
        "frequency": "M", "units": "0/1",
        "source": "NBER via FRED",
        "is_target": True, "resample": "last",
    },
}

_MISSING_KEY_MESSAGE = (
    "FRED_API_KEY is missing or empty. Copy .env.example to .env and "
    "fill in your key. Get a key at "
    "https://fred.stlouisfed.org/docs/api/api_key.html. See "
    "docs/design/fred_client.md for setup details."
)

_fred_client: Fred | None = None


def _get_fred_client() -> Fred:
    """Lazy-load the Fred client. Reads FRED_API_KEY from env on first call."""
    global _fred_client
    if _fred_client is None:
        load_dotenv()
        key = os.environ.get("FRED_API_KEY", "").strip()
        if not key:
            raise EnvironmentError(_MISSING_KEY_MESSAGE)
        _fred_client = Fred(api_key=key)
    return _fred_client


def _reset_client_for_testing() -> None:
    """Test-only: clear the cached Fred client so the env can be re-read."""
    global _fred_client
    _fred_client = None


def _cache_path(series_id: str, vintage_date: dt.date | None) -> Path:
    suffix = f"vintage_{vintage_date.isoformat()}" if vintage_date else "latest"
    return CACHE_DIR / f"{series_id}__{suffix}.parquet"


def _cache_is_fresh(path: Path, vintage_date: dt.date | None) -> bool:
    """Vintage cache never expires; latest cache expires after LATEST_TTL."""
    if not path.exists():
        return False
    if vintage_date is not None:
        return True
    mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc)
    age = dt.datetime.now(dt.timezone.utc) - mtime
    return age < LATEST_TTL


def _load_cache(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    return df["value"]


def _write_cache(path: Path, series: pd.Series) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_frame("value").to_parquet(path)


def _fetch_from_fred(
    series_id: str,
    start: dt.date | None,
    end: dt.date | None,
    vintage_date: dt.date | None,
) -> pd.Series:
    """Build FRED kwargs and delegate to fredapi. Translates errors."""
    client = _get_fred_client()
    kwargs: dict[str, str] = {}
    if start is not None:
        kwargs["observation_start"] = start.isoformat()
    if end is not None:
        kwargs["observation_end"] = end.isoformat()
    if vintage_date is not None:
        # BOTH realtime params must equal vintage_date to get the
        # "as known on date X" point-in-time snapshot. realtime_end alone
        # returns cumulative revision history (multiple rows per
        # observation). See docs/design/fred_client.md "Vintage handling".
        v = vintage_date.isoformat()
        kwargs["realtime_start"] = v
        kwargs["realtime_end"] = v
    try:
        return client.get_series(series_id, **kwargs)
    except Exception as e:
        msg = str(e).lower()
        if "does not exist" in msg or "bad request" in msg:
            raise ValueError(
                f"FRED series '{series_id}' not found. Check the series ID "
                f"in REGIME_SERIES or at "
                f"https://fred.stlouisfed.org/series/{series_id}."
            ) from e
        raise


def get_series(
    series_id: str,
    start: dt.date | None = None,
    end: dt.date | None = None,
    vintage_date: dt.date | None = None,
) -> pd.Series:
    """Fetch a single FRED series with optional vintage-aware semantics.

    Args:
        series_id: FRED series identifier (e.g., "UNRATE").
        start, end: observation date bounds. None = full history.
        vintage_date: if set, return values as known on that date
            (vintage-aware, for training). If None, return latest
            revisions (for inference).

    Returns:
        pd.Series indexed by observation date. Empty Series if the
        series has no observations at the requested vintage / range —
        does not raise (see error-handling section of design doc).

    Raises:
        EnvironmentError: if FRED_API_KEY is missing.
        ValueError: if FRED rejects the series_id.
    """
    cache_path = _cache_path(series_id, vintage_date)
    if _cache_is_fresh(cache_path, vintage_date):
        series = _load_cache(cache_path)
    else:
        series = _fetch_from_fred(series_id, start, end, vintage_date)
        _write_cache(cache_path, series)

    # Defensive filter — cache may have a wider range than the current call.
    if start is not None:
        series = series[series.index >= pd.Timestamp(start)]
    if end is not None:
        series = series[series.index <= pd.Timestamp(end)]

    if series.empty:
        if vintage_date is not None:
            logger.warning(
                "FRED series %s has no observations at vintage %s. "
                "Expected if the series starts after that date.",
                series_id,
                vintage_date.isoformat(),
            )
        else:
            logger.info(
                "FRED series %s returned no observations in range [%s, %s].",
                series_id,
                start,
                end,
            )
    return series


def validate_api_key(probe: bool = False) -> None:
    """Confirm FRED_API_KEY is set; optionally probe FRED to verify acceptance.

    Args:
        probe: if True, make a minimal API call (series metadata for UNRATE)
            to confirm FRED accepts the key. Default False — env presence only.

    Raises:
        EnvironmentError: if FRED_API_KEY is missing or empty.
        ValueError: if probe=True and FRED rejects the key (HTTP 400/403
            with "api key" in the response body).
    """
    client = _get_fred_client()
    if not probe:
        return
    try:
        client.get_series_info("UNRATE")
    except Exception as e:
        msg = str(e).lower()
        if "api key" in msg or "401" in msg or "403" in msg:
            raise ValueError(
                f"FRED_API_KEY is present but rejected by FRED: {e}. "
                f"Check the key value; it may have been rotated or "
                f"contain a typo."
            ) from e
        raise
