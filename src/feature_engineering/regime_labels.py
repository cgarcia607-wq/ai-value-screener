"""Regime labeling rules for the four-state cycle taxonomy.

See docs/design/regime_classifier.md for the full design including
the T+3 forward-looking framing — this module knows nothing about
the prediction horizon. It just maps features at time t to a regime
label at time t. The T+3 shift that creates training pairs happens
in the classifier orchestrator, not here.

Rule thresholds were locked after running
scripts/regime_label_diagnostic.py against real FRED data 2014-2025
and reviewing the resulting label sequence with the owner. Changes
to any constant in this file require re-running the diagnostic and
sanity-checking the output against known historical episodes (COVID
recession, 2018-2019 yield curve inversion, post-COVID recovery).
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


REGIMES: list[str] = ["Expansion", "Late-cycle", "Contraction", "Recovery"]
"""The 4-state cycle taxonomy. Canonical order; classifier
predict_proba columns appear in this order."""


# --- Threshold constants ----------------------------------------------------
# Numbers locked after the regime_label_diagnostic.py review pass. The
# unemployment elevation ladder is the key structural decision:
#
#   unrate_above_24mo_min < 0.7   -> "fine"        => Late-cycle-eligible
#   0.7 - 1.5                     -> "ambiguous"   (classifier expresses
#                                                   uncertainty between
#                                                   Late-cycle and Recovery)
#   >= 1.5                        -> "elevated"    => Recovery or
#                                                   Contraction depending
#                                                   on direction of change

# -- Contraction --
CONTRACTION_UNRATE_3MO_CHANGE_MIN: float = 0.5
"""3-month unemployment rise (pp) required to trigger Contraction.
0.5pp over 3 months is a real spike — historical normal-times
fluctuations are under 0.3pp."""

CONTRACTION_NFCI_THRESHOLD: float = 0.5
"""Chicago Fed NFCICREDIT level (positive = tighter credit) at or
above which Contraction fires on the credit-stress arm. NFCICREDIT
in normal times ranges -0.3 to +0.3; > 0.5 is meaningful stress.
COVID March-April 2020 peaked at 0.554 and 0.600."""

CONTRACTION_UNRATE_ELEVATION: float = 1.5
"""Unemployment above 12-month minimum (pp) at which Contraction
fires on the elevated-level arm. Matches RECOVERY_UNRATE_ELEVATION
so the elevated-unemployment zone has a single boundary; the
direction of change (rising vs falling unemployment trend)
distinguishes which regime applies. 12-month lookback (vs
Recovery's 24-month) detects new shocks against the recent floor.

This OR-clause exists so Contraction stays labeled through a
high-unemployment plateau after credit has stabilized — e.g.,
May-June 2020 when COVID unemployment was 11-13% but NFCICREDIT
had already declined below the stress threshold."""

# -- Recovery --
RECOVERY_UNRATE_ELEVATION: float = 1.5
"""Unemployment above 24-month minimum (pp) above which Recovery
fires. Same value as CONTRACTION_UNRATE_ELEVATION — the +1.5pp
threshold is the single "elevated unemployment" boundary. The
24-month lookback (longer than Contraction's 12) measures elevation
vs the pre-shock baseline; after a long expansion that gets
interrupted, the 24mo min preserves the pre-shock low even months
after the shock starts."""

RECOVERY_UNRATE_6MO_CHANGE_MAX: float = 0.0
"""Recovery requires unemployment trending DOWN over 6 months.
6-month smoothing dampens the month-to-month noise that caused
flicker in earlier rule iterations (an earlier version used a
3-month window and produced spurious Recovery flips in
late-expansion months with mildly negative unrate changes)."""

# -- Late-cycle --
LATE_CYCLE_T10Y3M_MAX: float = 0.25
"""10Y-3M Treasury spread (Fed's preferred recession-prediction
spread) at or below which the curve is considered flat-or-inverted.
Locked at 0.25 after inspecting Nov 2018-Aug 2019 (genuine
flattening into inversion) and Apr-Jul 2022 (rate-hike-era
flattening from very steep). 0.25 cleanly includes the 2019
episode but excludes the 2022 May-June months (1.69, 1.26) where
the curve was still meaningfully positive."""

LATE_CYCLE_UNRATE_ABOVE_MIN_MAX: float = 0.7
"""Unemployment above 24-month minimum (pp) below which Late-cycle
fires. Sits below RECOVERY_UNRATE_ELEVATION (1.5) to form the
clean ladder documented at the top of the constants section.
Bumped from 0.5 to 0.7 to capture the 2024-2025 inverted-curve
era — unemployment had drifted 0.5-0.7pp off the Apr 2023 cycle
low, which is "labor market basically OK, past peak" rather than
"unemployment elevated"."""

LATE_CYCLE_NFCI_MAX: float = 0.5
"""NFCICREDIT level below which Late-cycle fires. Matches
CONTRACTION_NFCI_THRESHOLD — the same NFCICREDIT boundary
separates 'credit normal' from 'credit stressed' in both
directions of the rule cascade."""


_REQUIRED_COLUMNS: tuple[str, ...] = ("UNRATE", "T10Y3M", "NFCICREDIT")


def compute_labels(features_df: pd.DataFrame) -> pd.Series:
    """Apply the rule cascade to produce a regime label per row.

    Args:
        features_df: FRED feature matrix as returned by
            `get_features_matrix(...)`. Must contain UNRATE, T10Y3M,
            and NFCICREDIT columns. Index must be a DatetimeIndex.
            Other columns (DGS10, FEDFUNDS, etc.) are ignored.

    Returns:
        pd.Series indexed identically to `features_df` rows for which
        the derived rolling-window features have sufficient history.
        Rows where unrate_3mo_change, unrate_6mo_change, or other
        rolling derivations are NaN are dropped from the output.
        Values are members of `REGIMES`.

    Raises:
        ValueError: required column missing, or non-DatetimeIndex input.
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in features_df.columns]
    if missing:
        raise ValueError(
            f"compute_labels requires columns {list(_REQUIRED_COLUMNS)}; "
            f"missing {missing}. Got columns: {list(features_df.columns)}"
        )
    if not isinstance(features_df.index, pd.DatetimeIndex):
        raise ValueError(
            f"compute_labels requires a DatetimeIndex; got "
            f"{type(features_df.index).__name__}"
        )

    derived = _derive_label_features(features_df).dropna()
    return derived.apply(_label_row, axis=1)


def _derive_label_features(matrix: pd.DataFrame) -> pd.DataFrame:
    """Derive rolling-window features required by the labeling rules.

    Public-ish (single underscore): the diagnostic script imports this
    to expose intermediate features (e.g., unrate_6mo_change) in its
    output, which would otherwise have to be recomputed.
    """
    df = pd.DataFrame(index=matrix.index)
    df["unrate"] = matrix["UNRATE"]
    df["unrate_3mo_change"] = df["unrate"] - df["unrate"].shift(3)
    df["unrate_6mo_change"] = df["unrate"] - df["unrate"].shift(6)
    df["unrate_12mo_min"] = df["unrate"].rolling(window=12, min_periods=1).min()
    df["unrate_24mo_min"] = df["unrate"].rolling(window=24, min_periods=1).min()
    df["unrate_above_24mo_min"] = df["unrate"] - df["unrate_24mo_min"]
    df["t10y3m"] = matrix["T10Y3M"]
    df["nfcicredit"] = matrix["NFCICREDIT"]
    return df


def validate_against_nber(
    labels: pd.Series,
    usrec: pd.Series,
    min_overlap: float = 0.5,
) -> list[dict]:
    """Sanity-check our regime labels against NBER recession dating.

    For each NBER recession period (a contiguous run of USREC == 1),
    compute the fraction of months in that period that our rules
    labeled "Contraction". The function returns a per-recession
    report; the caller decides what to do with low overlap (e.g.,
    assert in a test, log a WARNING in a diagnostic). Per the design
    doc this function deliberately does not raise on low overlap —
    "Caller raises if any overlap < min_overlap" is the caller's job.

    Args:
        labels: regime labels per month, DatetimeIndex. Values are
            members of REGIMES.
        usrec: NBER recession indicator (FRED USREC), DatetimeIndex.
            NaN is treated as 0 (not in recession); NaN never
            propagates into the report.
        min_overlap: threshold for the `passes_min` flag. Default
            0.5 matches the design doc's "≥50%" NBER overlap sanity
            check.

    Returns:
        List of dicts, ordered by recession start date, with
        recession_id assigned 1-based in that order. Each dict has:

          - recession_id: int, 1-based sequence number
          - start: pd.Timestamp, recession start (trimmed if needed)
          - end: pd.Timestamp, recession end (trimmed if needed)
          - overlap_fraction: float in [0, 1]
          - passes_min: bool, overlap_fraction >= min_overlap
          - trimmed_to_labels_coverage: True if either boundary was
            trimmed to fit within labels' index range; key is absent
            when the recession fell entirely inside labels coverage

        Empty USREC (no recessions) → empty list, no raise.

    Raises:
        ValueError: labels.index and usrec.index have empty
            intersection — nothing comparable.
    """
    if len(labels.index.intersection(usrec.index)) == 0:
        raise ValueError(
            "validate_against_nber: labels.index and usrec.index have no "
            f"overlap; nothing to validate. labels covers "
            f"[{labels.index.min()} … {labels.index.max()}], usrec covers "
            f"[{usrec.index.min()} … {usrec.index.max()}]."
        )

    in_recession = (usrec.fillna(0) == 1).to_numpy(dtype=bool)
    if not in_recession.any():
        return []

    usrec_index = usrec.index
    labels_min = labels.index.min()
    labels_max = labels.index.max()

    results: list[dict] = []
    run_start: int | None = None

    def _emit(start_pos: int, end_pos: int) -> None:
        rec_start = usrec_index[start_pos]
        rec_end = usrec_index[end_pos]
        trimmed_start = max(rec_start, labels_min)
        trimmed_end = min(rec_end, labels_max)
        was_trimmed = (trimmed_start != rec_start) or (
            trimmed_end != rec_end
        )

        if trimmed_start > trimmed_end:
            overlap = 0.0
        else:
            window = labels.loc[trimmed_start:trimmed_end]
            overlap = (
                float((window == "Contraction").mean())
                if len(window) > 0
                else 0.0
            )

        record: dict = {
            "recession_id": len(results) + 1,
            "start": trimmed_start,
            "end": trimmed_end,
            "overlap_fraction": overlap,
            "passes_min": overlap >= min_overlap,
        }
        if was_trimmed:
            record["trimmed_to_labels_coverage"] = True
        results.append(record)

    for i, flag in enumerate(in_recession):
        if flag and run_start is None:
            run_start = i
        elif not flag and run_start is not None:
            _emit(run_start, i - 1)
            run_start = None

    if run_start is not None:
        _emit(run_start, len(in_recession) - 1)

    return results


def _label_row(row: pd.Series) -> str:
    """Apply the four-rule cascade to a single derived feature row.

    Cascade order is load-bearing: Contraction beats Recovery beats
    Late-cycle beats Expansion. A row that matches multiple rules
    gets the first match.
    """
    # 1. Contraction: 3-month unemployment spike combined with EITHER
    #    credit stress OR elevated unemployment level.
    if row["unrate_3mo_change"] > CONTRACTION_UNRATE_3MO_CHANGE_MIN and (
        row["nfcicredit"] > CONTRACTION_NFCI_THRESHOLD
        or row["unrate"]
        > row["unrate_12mo_min"] + CONTRACTION_UNRATE_ELEVATION
    ):
        return "Contraction"

    # 2. Recovery: still elevated vs 24-month low, trending down
    #    over 6 months.
    if (
        row["unrate"] > row["unrate_24mo_min"] + RECOVERY_UNRATE_ELEVATION
        and row["unrate_6mo_change"] < RECOVERY_UNRATE_6MO_CHANGE_MAX
    ):
        return "Recovery"

    # 3. Late-cycle: yield curve flat or inverted, unemployment near
    #    cycle low, credit not yet stressed.
    if (
        row["t10y3m"] < LATE_CYCLE_T10Y3M_MAX
        and row["unrate_above_24mo_min"] < LATE_CYCLE_UNRATE_ABOVE_MIN_MAX
        and row["nfcicredit"] < LATE_CYCLE_NFCI_MAX
    ):
        return "Late-cycle"

    # 4. Default: Expansion.
    return "Expansion"
