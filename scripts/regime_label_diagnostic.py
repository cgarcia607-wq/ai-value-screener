"""Diagnostic: apply the regime label rules to real FRED data.

Run from anywhere:
    python scripts/regime_label_diagnostic.py

Prints the regime label per month for 2014-01 to 2025-12, plus three
sanity-check sections (COVID recession, 2018-19 yield curve inversion,
post-COVID recovery), an NBER USREC overlap check, and a transitions-
per-year coherence metric.

Uses the rule logic from src.feature_engineering.regime_labels. The
diagnostic exists to surface what those rules produce on real data so
threshold revisions can be reviewed before locking. After locking, the
diagnostic remains as a regression-detection tool: re-run it any time
the rule constants change and confirm the output still passes the
sanity checks.
"""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.data_sources.fred_client import get_features_matrix, get_series  # noqa: E402
from src.feature_engineering.regime_labels import (  # noqa: E402
    _derive_label_features as derive_label_features,
    compute_labels,
)


def main() -> None:
    print("Fetching FRED data 2014-01 to 2025-12...")
    matrix = get_features_matrix(
        start=dt.date(2014, 1, 1), end=dt.date(2025, 12, 31)
    )
    usrec = get_series(
        "USREC", start=dt.date(2014, 1, 1), end=dt.date(2025, 12, 31)
    )
    print(f"Loaded {len(matrix)} monthly observations")
    print()

    labels = compute_labels(matrix)
    label_features = derive_label_features(matrix).loc[labels.index]

    display = pd.DataFrame(
        {
            "label": labels,
            "unrate": label_features["unrate"],
            "d_unrate": label_features["unrate_3mo_change"],
            "t10y3m": label_features["t10y3m"],
            "nfcic": label_features["nfcicredit"],
        }
    )

    # Align USREC (monthly, month-start dated) to display index (month-end
    # business days). Both refer to the same calendar month either way.
    usrec_ME = usrec.resample("ME").last()
    display["usrec"] = (
        usrec_ME.reindex(display.index, method="nearest").astype("Int64")
    )

    # --- Full sequence ----------------------------------------------------
    print("=== Full label sequence ===")
    print()
    print(
        f"{'Date':>10} | {'Label':>11} | {'unrate':>6} | {'dUnr':>5} | "
        f"{'t10y3m':>6} | {'nfcic':>6} | {'USREC':>5}"
    )
    print("-" * 75)
    for date, row in display.iterrows():
        print(
            f"{date.date().isoformat():>10} | {row['label']:>11} | "
            f"{row['unrate']:>5.1f}% | {row['d_unrate']:>+5.2f} | "
            f"{row['t10y3m']:>+6.2f} | {row['nfcic']:>+6.3f} | "
            f"{int(row['usrec']):>5}"
        )

    # --- Distribution -----------------------------------------------------
    print()
    print("=== Label distribution ===")
    counts = display["label"].value_counts()
    total = len(display)
    for label in ("Expansion", "Late-cycle", "Contraction", "Recovery"):
        count = int(counts.get(label, 0))
        pct = 100.0 * count / total
        print(f"  {label:>11}: {count:>3} months ({pct:>4.1f}%)")

    # --- Sanity check 1: COVID --------------------------------------------
    print()
    print("=== Sanity check 1: COVID recession (Feb-Aug 2020) ===")
    try:
        covid = display.loc["2020-02-01":"2020-08-31"]
        for date, row in covid.iterrows():
            usrec_status = "REC" if row["usrec"] == 1 else "exp"
            print(
                f"  {date.date().isoformat()}: {row['label']:>11} "
                f"(USREC={usrec_status}, dUnr={row['d_unrate']:+.2f}, "
                f"nfcic={row['nfcic']:+.3f})"
            )
    except KeyError:
        print("  (no data in COVID window)")

    # --- Sanity check 2: yield curve inversion ----------------------------
    print()
    print("=== Sanity check 2: Yield curve inversion (mid-2018 to mid-2019) ===")
    try:
        inv_window = display.loc["2018-06-01":"2019-12-31"]
        for date, row in inv_window.iterrows():
            inv_flag = "INV" if row["t10y3m"] < 0 else "   "
            above_min = (
                row["unrate"] - label_features["unrate_24mo_min"].loc[date]
            )
            print(
                f"  {date.date().isoformat()}: {row['label']:>11} "
                f"(t10y3m={row['t10y3m']:+.2f} {inv_flag}, "
                f"unrate_above_min={above_min:.2f})"
            )
    except KeyError:
        print("  (no data in inversion window)")

    # --- Sanity check 3: post-COVID recovery ------------------------------
    print()
    print("=== Sanity check 3: Post-COVID recovery (mid-2020 to end-2021) ===")
    try:
        rec_window = display.loc["2020-06-01":"2021-12-31"]
        for date, row in rec_window.iterrows():
            d6 = label_features["unrate_6mo_change"].loc[date]
            print(
                f"  {date.date().isoformat()}: {row['label']:>11} "
                f"(unrate={row['unrate']:.1f}%, dUnr_3mo={row['d_unrate']:+.2f}, "
                f"dUnr_6mo={d6:+.2f})"
            )
    except KeyError:
        print("  (no data in recovery window)")

    # --- Transitions per year ---------------------------------------------
    print()
    print("=== Regime transitions per calendar year (coherence metric) ===")
    print("    Healthy: 2-4 transitions/year. Higher = label flicker.")
    transitions_by_year: dict[int, int] = {}
    prev_label = None
    for date, row in display.iterrows():
        year = date.year
        transitions_by_year.setdefault(year, 0)
        if prev_label is not None and row["label"] != prev_label:
            transitions_by_year[year] += 1
        prev_label = row["label"]
    for year in sorted(transitions_by_year):
        print(f"  {year}: {transitions_by_year[year]} transitions")
    avg = sum(transitions_by_year.values()) / len(transitions_by_year)
    print(f"  Average: {avg:.1f} transitions/year")

    # --- NBER overlap validation ------------------------------------------
    print()
    print("=== Validation against NBER USREC ===")
    contraction_mask = display["label"] == "Contraction"
    usrec_mask = display["usrec"] == 1

    contraction_dates = display[contraction_mask].index
    usrec_dates = display[usrec_mask].index

    print(f"  Months labeled Contraction: {len(contraction_dates)}")
    if len(contraction_dates) > 0:
        print(
            "    "
            + ", ".join(d.date().isoformat() for d in contraction_dates)
        )
    print(f"  Months with NBER USREC=1:   {len(usrec_dates)}")
    if len(usrec_dates) > 0:
        print("    " + ", ".join(d.date().isoformat() for d in usrec_dates))

    overlap = set(contraction_dates) & set(usrec_dates)
    if len(usrec_dates) > 0:
        overlap_pct = 100.0 * len(overlap) / len(usrec_dates)
        marker = "OK" if overlap_pct >= 50 else "FAIL"
        print(
            f"  Overlap (Contraction during USREC=1): "
            f"{len(overlap)}/{len(usrec_dates)} = {overlap_pct:.0f}%  [{marker}]"
        )
    else:
        print("  (no NBER recession in window — nothing to validate)")


if __name__ == "__main__":
    main()
