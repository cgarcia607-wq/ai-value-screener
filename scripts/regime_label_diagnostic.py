"""Diagnostic: apply proposed regime label rules to real FRED data.

Run from anywhere:
    python scripts/regime_label_diagnostic.py

Prints the regime label per month for 2014-01 to 2025-12 (the period
covered by the current FRED inventory), plus three sanity-check
sections:
  - COVID recession (Feb-Apr 2020)
  - Yield curve inversion (2018-2019)
  - Post-COVID recovery (mid-2020 to end-2021)

Validates against NBER USREC: every NBER recession period should have
>= 50% Contraction overlap from the rules.

The rule logic lives inline here (not in src/feature_engineering/
regime_labels.py yet) because the rule thresholds are still under
review. After owner approval of the label sequence, the rules get
factored into the module.
"""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.data_sources.fred_client import get_features_matrix, get_series  # noqa: E402


def derive_label_features(matrix: pd.DataFrame) -> pd.DataFrame:
    """Derive the rule features from raw FRED columns."""
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


def regime_label(row: pd.Series) -> str:
    """Apply the four-rule cascade; return one of E/L/C/R.

    The "+1.5 still-elevated unemployment" threshold appears in BOTH
    Contraction (12-month lookback) and Recovery (24-month lookback).
    Same threshold on both sides means the elevated-unemployment zone
    is a single band; the direction of change (rising vs falling)
    distinguishes which regime applies. Contraction uses the shorter
    12-month lookback because new shocks are detected against recent
    floors; Recovery uses 24-month lookback because "still elevated"
    is measured against the pre-shock baseline that may pre-date the
    current shock by a year+.
    """
    # 1. Contraction: 3-month unemployment spike combined with EITHER
    #    credit stress OR elevated unemployment level. The credit-stress
    #    clause catches the initial shock; the elevation clause keeps
    #    Contraction labeled through the high-unemployment plateau
    #    after credit has stabilized (e.g., May-June 2020).
    if row["unrate_3mo_change"] > 0.5 and (
        row["nfcicredit"] > 0.5
        or row["unrate"] > row["unrate_12mo_min"] + 1.5
    ):
        return "Contraction"

    # 2. Recovery: still elevated vs 24-month low, trending down over
    #    6 months. The 6-month smoothing prevents flicker from month-
    #    to-month noise. The 24-month lookback for the "still elevated"
    #    reference avoids false-Recovery during normal expansion when
    #    unemployment briefly ticks up and then declines.
    if (
        row["unrate"] > row["unrate_24mo_min"] + 1.5
        and row["unrate_6mo_change"] < 0
    ):
        return "Recovery"

    # 3. Late-cycle: yield curve flat or inverted (t10y3m < 0.25,
    #    threshold locked after inspecting Nov 2018-Aug 2019 and
    #    Apr-Jul 2022 episodes), unemployment near cycle low (within
    #    0.7pp of 24-month min), credit not yet stressed.
    #
    #    The 0.7pp unemployment threshold sits below Recovery's +1.5
    #    elevation threshold, forming a clean ladder:
    #      <= 0.7pp from 24mo min  -> "labor market basically OK,
    #                                  past peak" => Late-cycle-eligible
    #      0.7-1.5pp                -> ambiguous (classifier expresses
    #                                  uncertainty between Late-cycle
    #                                  and Recovery)
    #      >= 1.5pp                -> "still elevated" => Recovery
    #                                  (or Contraction if also spiking)
    #
    #    Bumped from 0.5 to 0.7 because the 2024-2025 era had a deeply
    #    inverted curve but unrate had drifted ~0.5pp off cycle low —
    #    structurally Late-cycle, but the old 0.5 cutoff was too tight
    #    to capture it.
    if (
        row["t10y3m"] < 0.25
        and row["unrate_above_24mo_min"] < 0.7
        and row["nfcicredit"] < 0.5
    ):
        return "Late-cycle"

    # 4. Default: Expansion.
    return "Expansion"


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

    label_features = derive_label_features(matrix).dropna()
    labels = label_features.apply(regime_label, axis=1)

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
