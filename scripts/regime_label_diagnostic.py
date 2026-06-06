"""Diagnostic: apply the regime label rules to real FRED data.

Run from the repo root:

    PYTHONPATH=. python scripts/regime_label_diagnostic.py

Fetches the FRED feature matrix for 2014-01-01 → 2024-12-31 at the
2024-12-31 vintage, applies compute_labels() per
src.feature_engineering.regime_labels, and prints:

  1. The full per-month timeline (date | unrate | t10y3m | nfcicredit
     | regime).
  2. Three summary blocks listing every Contraction, Recovery, and
     Late-cycle month.
  3. The value_counts() of all four regimes.

Per the design doc (docs/design/regime_classifier.md §"Diagnostic
phase"), this script is a one-off human-review tool — owner reads the
output for plausibility against known historical episodes (COVID
recession, 2018-2019 inversion, post-COVID recovery) and adjusts the
rule thresholds in regime_labels.py if anything looks wrong. The
labels are the ground truth; if they look wrong, no model can fix
them.

Operational chatter (FRED fetch, cache hits/misses) goes to logging.
The diagnostic output itself goes to print().
"""

from __future__ import annotations

import datetime as dt
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_sources.fred_client import get_features_matrix  # noqa: E402
from src.feature_engineering.regime_labels import (  # noqa: E402
    REGIMES,
    compute_labels,
)

logger = logging.getLogger(__name__)


START = dt.date(2014, 1, 1)
END = dt.date(2024, 12, 31)
VINTAGE = dt.date(2024, 12, 31)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logger.info(
        "Fetching FRED feature matrix start=%s end=%s vintage=%s",
        START, END, VINTAGE,
    )
    matrix = get_features_matrix(start=START, end=END, vintage_date=VINTAGE)
    logger.info("Fetched %d monthly observations", len(matrix))

    labels = compute_labels(matrix)
    # compute_labels drops rows whose rolling-window derivations were
    # NaN (typically the leading rows of the window). Align the
    # rule-input columns to the surviving index for the timeline.
    aligned = matrix.loc[labels.index, ["UNRATE", "T10Y3M", "NFCICREDIT"]]

    # 1. Full per-month timeline.
    print("=== Regime label timeline ({} → {}) ===".format(
        labels.index[0].date(), labels.index[-1].date()
    ))
    print()
    print(f"{'date':>10}  {'unrate':>7}  {'t10y3m':>7}  "
          f"{'nfcicredit':>10}  {'regime':<12}")
    print("-" * 54)
    for ts, regime in labels.items():
        row = aligned.loc[ts]
        print(
            f"{ts.date().isoformat():>10}  "
            f"{row['UNRATE']:>7.2f}  "
            f"{row['T10Y3M']:>+7.2f}  "
            f"{row['NFCICREDIT']:>+10.3f}  "
            f"{regime:<12}"
        )

    # 2. Per-regime month listings (Contraction, Recovery, Late-cycle).
    for regime in ("Contraction", "Recovery", "Late-cycle"):
        months = labels.index[labels == regime]
        print()
        print(f"=== {regime} months ({len(months)}) ===")
        if len(months) == 0:
            print("  (none)")
            continue
        for ts in months:
            print(f"  {ts.date().isoformat()}")

    # 3. value_counts() of all four regimes (in REGIMES order, with
    #    zero rows shown so missing regimes are obvious).
    print()
    print("=== Regime value counts ===")
    counts = labels.value_counts()
    for regime in REGIMES:
        n = int(counts.get(regime, 0))
        pct = 100.0 * n / len(labels) if len(labels) else 0.0
        print(f"  {regime:<12}  {n:>4}  ({pct:>5.1f}%)")
    print(f"  {'TOTAL':<12}  {len(labels):>4}")


if __name__ == "__main__":
    main()
