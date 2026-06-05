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
# `resample` is the per-series rule for monthly harmonization:
#   "ffill" (default): take last observation <= month-end, forward-fill
#       across gaps. Matches as-of-date trader semantics.
#   "last": last observation in the month, NaN if none. Stricter than
#       ffill — does not carry values across empty months.
#   "mean": within-month average. For series where the period-average
#       interpretation is preferred (e.g., monthly mean of a daily rate).
#
# See docs/design/fred_client.md for the full rationale per series.
REGIME_SERIES: dict[str, dict] = {
    # --- Yield curve --------------------------------------------------
    "DGS10": {
        "name": "10-Year Treasury Constant Maturity Rate",
        "category": "yield_curve",
        "frequency": "D", "units": "%",
        "source": "Board of Governors via FRED",
        "is_target": False, "resample": "ffill",
    },
    "DGS2": {
        "name": "2-Year Treasury Constant Maturity Rate",
        "category": "yield_curve",
        "frequency": "D", "units": "%",
        "source": "Board of Governors via FRED",
        "is_target": False, "resample": "ffill",
    },
    "T10Y2Y": {
        "name": "10-Year minus 2-Year Treasury Spread",
        "category": "yield_curve",
        "frequency": "D", "units": "%",
        "source": "FRED (computed from DGS10/DGS2)",
        "is_target": False, "resample": "ffill",
    },
    "T10Y3M": {
        "name": "10-Year minus 3-Month Treasury Spread",
        "category": "yield_curve",
        "frequency": "D", "units": "%",
        "source": "FRED (computed)",
        "is_target": False, "resample": "ffill",
    },
    # --- Credit -------------------------------------------------------
    # NOTE: BAMLH0A0HYM2 (ICE BofA US High Yield OAS) was the original
    # HY-spread series here. ICE Data Indices restricted public access
    # to FRED's historical ICE BofA data in late 2024, truncating that
    # series (and all sibling BAML* series) to a rolling ~3-year window.
    # NFCICREDIT (Chicago Fed NFCI Credit Subindex) replaces it: full
    # 1971-present coverage, Fed-native maintenance, and directly
    # captures credit stress (positive = tighter than average credit,
    # negative = looser). See docs/design/fred_client.md.
    "NFCICREDIT": {
        "name": "Chicago Fed NFCI Credit Subindex",
        "category": "credit",
        "frequency": "W", "units": "index",
        "source": "Federal Reserve Bank of Chicago via FRED",
        "is_target": False, "resample": "ffill",
    },
    "BAA10Y": {
        "name": "Moody's Baa Corporate Bond Yield minus 10Y Treasury",
        "category": "credit",
        "frequency": "D", "units": "%",
        "source": "Moody's / FRED",
        "is_target": False, "resample": "ffill",
    },
    # --- Activity / Labor --------------------------------------------
    "UNRATE": {
        "name": "Unemployment Rate",
        "category": "activity",
        "frequency": "M", "units": "%",
        "source": "BLS via FRED",
        "is_target": False, "resample": "ffill",
    },
    "ICSA": {
        "name": "Initial Jobless Claims",
        "category": "activity",
        "frequency": "W", "units": "count",
        "source": "DOL via FRED",
        "is_target": False, "resample": "ffill",
    },
    "PAYEMS": {
        "name": "Total Nonfarm Payrolls",
        "category": "activity",
        "frequency": "M", "units": "thousand",
        "source": "BLS via FRED",
        "is_target": False, "resample": "ffill",
    },
    "INDPRO": {
        "name": "Industrial Production Index",
        "category": "activity",
        "frequency": "M", "units": "index",
        "source": "Board of Governors via FRED",
        "is_target": False, "resample": "ffill",
    },
    "HOUST": {
        "name": "Housing Starts",
        "category": "activity",
        "frequency": "M", "units": "thousand SAAR",
        "source": "Census Bureau via FRED",
        "is_target": False, "resample": "ffill",
    },
    # --- Rates / Money -----------------------------------------------
    "FEDFUNDS": {
        "name": "Effective Federal Funds Rate (monthly)",
        "category": "rates",
        "frequency": "M", "units": "%",
        "source": "Board of Governors via FRED",
        "is_target": False, "resample": "ffill",
    },
    "M2SL": {
        "name": "M2 Money Stock",
        "category": "rates",
        "frequency": "M", "units": "$ billion",
        "source": "Board of Governors via FRED",
        "is_target": False, "resample": "ffill",
    },
    # --- Inflation ---------------------------------------------------
    "CPIAUCSL": {
        "name": "Consumer Price Index, All Urban Consumers",
        "category": "inflation",
        "frequency": "M", "units": "index",
        "source": "BLS via FRED",
        "is_target": False, "resample": "ffill",
    },
    "CPILFESL": {
        "name": "Core CPI (ex food and energy)",
        "category": "inflation",
        "frequency": "M", "units": "index",
        "source": "BLS via FRED",
        "is_target": False, "resample": "ffill",
    },
    "PCEPILFE": {
        "name": "Core PCE Price Index",
        "category": "inflation",
        "frequency": "M", "units": "index",
        "source": "BEA via FRED",
        "is_target": False, "resample": "ffill",
    },
    # --- Market internals --------------------------------------------
    "VIXCLS": {
        "name": "CBOE Volatility Index (VIX)",
        "category": "market_internals",
        "frequency": "D", "units": "level",
        "source": "CBOE via FRED",
        "is_target": False, "resample": "ffill",
    },
    "DTWEXBGS": {
        "name": "Trade-Weighted Broad Dollar Index",
        "category": "market_internals",
        "frequency": "D", "units": "index",
        "source": "Board of Governors via FRED",
        "is_target": False, "resample": "ffill",
    },
    "DCOILWTICO": {
        "name": "WTI Crude Oil Spot Price",
        "category": "market_internals",
        "frequency": "D", "units": "USD/bbl",
        "source": "EIA via FRED",
        "is_target": False, "resample": "ffill",
    },
    "UMCSENT": {
        "name": "University of Michigan Consumer Sentiment",
        "category": "market_internals",
        "frequency": "M", "units": "index",
        "source": "University of Michigan via FRED",
        "is_target": False, "resample": "ffill",
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
        "is_target": True, "resample": "ffill",
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


def _cache_is_fresh(
    path: Path,
    vintage_date: dt.date | None,
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> bool:
    """Decide whether the cached parquet at `path` can serve the request.

    Two invariants must hold:

    1. **TTL** (latest caches only) — files older than LATEST_TTL are
       stale. Vintage caches are immutable, no TTL.
    2. **Range coverage** — the cached data must cover the requested
       [start, end] window. Without this check, a cache populated by an
       earlier narrow-range fetch (e.g., UNRATE 2020-01 to 2020-12)
       would be served for a later wider request (e.g., 2014-2025), and
       the resulting Series would silently lack history outside the
       cached range. The forward-filling in `get_features_matrix` would
       then propagate stale values across all later month-ends, looking
       superficially correct but being silently wrong. See
       docs/design/fred_client.md "Cache range-coverage invariant".

    The range check reads the parquet to inspect the index extent. This
    is a small file (~few KB) and the load is cheap; the subsequent
    `_load_cache` in the calling code does a second read, accepted as
    cleaner than threading the loaded DataFrame through.
    """
    if not path.exists():
        return False
    if vintage_date is None:
        mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc)
        age = dt.datetime.now(dt.timezone.utc) - mtime
        if age >= LATEST_TTL:
            return False
    if start is None and end is None:
        return True
    try:
        df = pd.read_parquet(path)
    except Exception:
        # Corrupt cache file; treat as stale, force refetch.
        return False
    if len(df) == 0:
        # An empty cache can only satisfy a range-less request. For an
        # explicit range, we have no evidence the empty result is right
        # for that range — refetch.
        return False
    cached_min = df.index.min().date()
    cached_max = df.index.max().date()
    if start is not None and cached_min > start:
        return False
    if end is not None and cached_max < end:
        return False
    return True


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
    if _cache_is_fresh(cache_path, vintage_date, start, end):
        series = _load_cache(cache_path)
    else:
        series = _fetch_from_fred(series_id, start, end, vintage_date)
        _write_cache(cache_path, series)

    # Short-circuit empty results BEFORE the defensive filter. fredapi
    # returns empty Series with a default RangeIndex (int64), not an
    # empty DatetimeIndex, so `series.index >= pd.Timestamp(start)` would
    # raise TypeError ('>=' not supported between numpy.ndarray and
    # Timestamp). The filter is only meaningful when the index is real.
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

    # Defensive filter — cache may have a wider range than the current call.
    if start is not None:
        series = series[series.index >= pd.Timestamp(start)]
    if end is not None:
        series = series[series.index <= pd.Timestamp(end)]

    return series


def _resample_to_month_end(
    series: pd.Series,
    month_ends: pd.DatetimeIndex,
    method: str,
) -> pd.Series:
    """Align a series to a month-end DatetimeIndex via the given method.

    Methods:
        "ffill": resample('ME').last(), then ffill across gaps after
            reindexing to `month_ends`. Default for as-of semantics.
        "last": resample('ME').last(), no ffill. Strict — empty months
            remain NaN.
        "mean": resample('ME').mean(). Period-average semantics.
    """
    if method not in {"ffill", "last", "mean"}:
        raise ValueError(
            f"Unknown resample method '{method}'. Expected 'ffill', "
            f"'mean', or 'last'."
        )
    if series.empty:
        return pd.Series(index=month_ends, dtype="float64")

    if method == "mean":
        resampled = series.resample("ME").mean()
    else:
        resampled = series.resample("ME").last()

    aligned = resampled.reindex(month_ends)
    if method == "ffill":
        aligned = aligned.ffill()
    return aligned


def get_features_matrix(
    series_ids: list[str] | None = None,
    start: dt.date = dt.date(1990, 1, 1),
    end: dt.date | None = None,
    frequency: str = "M",
    vintage_date: dt.date | None = None,
    exclude_targets: bool = True,
) -> pd.DataFrame:
    """Fetch multiple FRED series and align to a common monthly cross-section.

    Args:
        series_ids: list of FRED IDs to fetch. None = all REGIME_SERIES.
            USREC is excluded by default via `exclude_targets`; pass
            `exclude_targets=False` to include it (e.g., for label
            construction in `regime_labels.py`).
        start, end: date range. `end=None` means today.
        frequency: only 'M' (month-end) is supported in v1.
        vintage_date: passed through to every underlying get_series
            call. None = latest revisions; date = point-in-time
            "as known on date X" snapshot.
        exclude_targets: if True (default), drop any series with
            is_target=True from the result. This is the label-leakage
            guardrail.

    Returns:
        DataFrame indexed by month-end DatetimeIndex with one column
        per requested series. Per-series resampling uses
        REGIME_SERIES[id]["resample"].
    """
    if frequency != "M":
        raise NotImplementedError(
            f"frequency={frequency!r} not yet supported. Only 'M' "
            f"(month-end) is implemented in v1. See "
            f"docs/design/fred_client.md."
        )

    if series_ids is None:
        series_ids = list(REGIME_SERIES.keys())
    if exclude_targets:
        series_ids = [
            s for s in series_ids
            if not REGIME_SERIES.get(s, {}).get("is_target", False)
        ]

    end_date = end or dt.date.today()
    month_ends = pd.date_range(
        start=pd.Timestamp(start), end=pd.Timestamp(end_date), freq="ME"
    )

    # Per-series fetch with batch-degradation behavior. Individual
    # failures are collected; <25% failure rate logs a WARNING and
    # continues with the available series, >=25% aborts. See the
    # error-handling section of docs/design/fred_client.md.
    #
    # Any exception that lands here is by construction NOT a soft
    # "FRED returned empty" condition — those are absorbed inside
    # get_series and return an empty Series without raising. So
    # anything we catch is either a FRED-side API failure or a
    # code-level bug in our wrapper. We log the full traceback
    # (exc_info=True) so the two are immediately distinguishable: a
    # FRED HTTP error vs. a TypeError in our own code reads very
    # differently in the traceback. Without exc_info, the original
    # BAMLH0A0HYM2 RangeIndex bug masqueraded as a missing-data
    # condition for weeks.
    fetched: dict[str, pd.Series] = {}
    failures: dict[str, str] = {}
    for sid in series_ids:
        try:
            fetched[sid] = get_series(
                sid, start=start, end=end_date, vintage_date=vintage_date
            )
        except Exception as e:
            logger.warning(
                "FRED series %s raised an exception during fetch "
                "(this is not a soft missing-data condition — those "
                "do not raise). Full traceback follows:",
                sid,
                exc_info=True,
            )
            failures[sid] = repr(e)

    total = len(series_ids)
    if total > 0 and len(failures) / total >= 0.25:
        failed_names = sorted(failures.keys())
        raise RuntimeError(
            f"FRED feature-matrix fetch aborted: {len(failures)} of {total} "
            f"series ({len(failures) / total:.0%}) failed, exceeding the "
            f"25% threshold. Failed series: {failed_names}. See WARNING "
            f"logs for per-series errors."
        )
    if failures:
        logger.warning(
            "FRED feature-matrix returning with %d of %d series missing: %s. "
            "Affected columns will be absent from the result.",
            len(failures),
            total,
            sorted(failures.keys()),
        )

    matrix = pd.DataFrame(index=month_ends, dtype="float64")
    for sid, series in fetched.items():
        method = REGIME_SERIES.get(sid, {}).get("resample", "ffill")
        matrix[sid] = _resample_to_month_end(series, month_ends, method)
    return matrix


def list_series(category: str | None = None) -> pd.DataFrame:
    """Return REGIME_SERIES as a DataFrame for inspection.

    Args:
        category: optional filter, e.g., "yield_curve". Returns rows
            whose category matches exactly. None = all series.

    Returns:
        DataFrame indexed by series_id with columns: name, category,
        frequency, units, source, is_target, resample.
    """
    df = pd.DataFrame.from_dict(REGIME_SERIES, orient="index")
    df.index.name = "series_id"
    if category is not None:
        df = df[df["category"] == category]
    return df


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
