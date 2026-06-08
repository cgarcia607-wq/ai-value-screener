Project handoff: AI Value Screener v2.0
Identity and working style
You are continuing collaboration with CG (Chris Garcia, GitHub cgarcia607-wq), a finance-background builder shipping a portfolio-grade ML project in 2026. Repo: github.com/cgarcia607-wq/ai-value-screener.
CG's preferred working style with me:

Honest pushback. If their instruction is methodologically wrong, say so with reasoning.
Design-doc-first, phased implementation. Major work runs: design doc → review → phase 1 → REPL verification → phase 2 → REPL verification. No code without a plan; no merge without a REPL spot-check against real data.
Catch bugs in markdown, not in code. Pause before implementation when subtleties arise.
Prose-first responses. Minimal headers and bullets when conversational; structured formatting only when there's a list of distinct items or decisions to track.
Specific, actionable next steps. End each turn with a clear prompt to paste into Claude Code.

The collaboration pattern: I'm the design partner / methodology reviewer; Claude Code is the implementer. CG runs Claude Code in their terminal and pastes results to me for review. I refuse code reviews until CG verifies against real data via REPL.
Project goal
Take CG's working v1.0 stock screener (github.com/cgarcia607-wq/ml-value-screener — survivorship-biased, look-ahead-biased, no proper validation) and rebuild it methodologically clean for portfolio purposes. The framing: "show I am an AI expert and part of the 1% club with AI."
The 1% framing has two layers:

Methodology rigor — point-in-time data, walk-forward CV, calibration, explainability. Catch and engineer through real-world data problems.
AI-native capability (v3.0 plans) — LLM features (earnings call sentiment via Claude API), agentic research memos, RAG over SEC filings. The infrastructure being built now is foundation for these.

Locked methodology decisions
Read CLAUDE.md in the repo for the authoritative list. Highlights:

Point-in-time S&P 500 constituents (no Wikipedia roster, no IVV scrape — uses fja05680/sp500 change log, frozen and hash-pinned in repo)
Vintage-aware FRED data (no current-snapshot fundamentals for training; macro features queried with realtime_start=realtime_end=vintage_date)
Walk-forward CV with embargo = label horizon (12 months for screener, 3 months for regime classifier — was 1, corrected when CG built the embargo specification)
Cross-sectional rank/z-score features, not raw values
Calibrated probabilities via CalibratedClassifierCV (isotonic) for any user-facing output
SHAP for explainability, not feature_importances_
Logistic regression baseline alongside XGBoost; ship simpler unless XGBoost wins by ≥5pp macro-F1 with ±2pp tiebreaker preferring LR
Regime classifier predicts T+3 forward labels (refinement from concurrent — concurrent would have made the model a function-approximator of the rules; T+3 makes it a real forecaster)
NFCICREDIT is being retired from the labeling rules (NOT the feature matrix) — see "Findings that reshaped the plan" below. Replacement is BAA10Y with re-derived thresholds. CLAUDE.md and regime_classifier.md must be updated when the swap is implemented, with the reasoning recorded.

Foundation phase: COMPLETE
Three modules built, ~140 tests passing, real-data verified via REPL at every milestone:

src/data_sources/sp500_membership.py — point-in-time S&P 500 constituents reconstructed from frozen change-log CSV. 72,521 rows × 4 columns, 144 month-ends 2014-2025. Pyarrow dict-encoded ticker columns. 19 unit tests + 1 slow integration test.
src/data_sources/fred_client.py — 20 macro/credit features + USREC as ground truth, organized in 7 categories. Vintage-aware queries via realtime_start=realtime_end=vintage_date. Per-series parquet caching with range-coverage invariant (cache invalidated if cached range narrower than requested — bug discovered and fixed). 35 unit tests + 1 slow integration with 4 anchor assertions (T10Y2Y inversion 2019, UNRATE latest and vintage, NFCICREDIT COVID stress). Per-series "vintaged" bool added this session (see supporting work below).
src/validation/walk_forward.py — calendar-month-anchored fold engine with embargo, expanding/sliding window, sklearn shim. 58 tests. Verified against real membership data: 8 folds for screener config, 0 embargo violations.

Regime classifier phase: COMPLETE (both phases)

src/feature_engineering/regime_labels.py — DONE. Deterministic rules-based label function. 4 regimes (Expansion / Late-cycle / Contraction / Recovery). 26 tests including 6 slow integration tests pinning known regime windows (COVID Contraction, 2018-2019 Late-cycle, post-COVID Recovery, 2024 inverted-curve Late-cycle). Threshold constants surfaced as module constants with cross-consistency tests guarding the 0.5/0.7/1.5 ladder. validate_against_nber() added this session.
src/models/regime_classifier.py — COMPLETE. Phase 1 and Phase 2 both built and real-data verified via REPL against FRED 2014-2024.

Phase 1: RegimeClassifier class wrapping StandardScaler + CalibratedClassifierCV(isotonic, cv=3) over LogisticRegression or XGBClassifier. Label encoding to integers via REGIMES as canonical 0..3 mapping (required because XGBClassifier(num_class=4) rejects string labels; also guarantees predict_proba column order is deterministic). Methods: fit / predict / predict_proba / save / load. 5 unit tests.
Phase 2: run_walk_forward() orchestrator. T+3 target construction, per-fold scaler/classifier/calibration, macro-F1 + per-class F1 + multi-class Brier + confusion matrix, adoption decision (XGB adopted only on +5pp macro-F1, ±2pp tiebreaker to LR, catastrophic-fold disqualification). 5 unit tests. Total 10 tests in test_regime_classifier.py.

scripts/regime_label_diagnostic.py — COMPLETE. Per-month label timeline over a date range for human review. Committed, gitignored output.

Note: the classifier code is correct but the regime classifier CANNOT be credibly trained on 2014-2024 data (see "Findings" below). The code is the right code; the inputs need to change.

Supporting work completed this session

Type guards added to get_features_matrix (fred_client.py): rejects string start/end/vintage_date with a clean TypeError naming the offending argument, instead of crashing deep in _fetch_from_fred. 3 tests.
validate_against_nber() added to regime_labels.py: checks each NBER recession period (contiguous USREC==1) for >=min_overlap fraction of Contraction labels. Returns list of per-recession dicts; does not raise (caller decides). Handles empty USREC, NaN, index mismatch, recession-extends-past-labels trimming. 7 tests.
Vintaging fix in fred_client.py: added per-series "vintaged" bool to REGIME_SERIES. Non-revised/market-observed/NBER series (DGS10, DGS2, T10Y2Y, T10Y3M, BAA10Y, NFCICREDIT, FEDFUNDS, VIXCLS, DTWEXBGS, DCOILWTICO, USREC) fall back to current-values query when a vintage is requested, logging a WARNING (current == point-in-time for never-restated series). Revised series (UNRATE, ICSA, PAYEMS, INDPRO, HOUST, M2SL, CPIAUCSL, CPILFESL, PCEPILFE, UMCSENT) keep realtime queries. Unknown series default to vintaged=True. 4 tests. Discovered because daily market series and USREC have no ALFRED vintage archive and were raising "series not found" on vintaged deep-history queries.

Findings that reshaped the plan
These are the load-bearing discoveries from this session, in order of discovery. Each is the kind of real-world data problem that belongs in the README portfolio narrative.

(a) DATA-SCOPE FINDING — 2014-2024 is not enough regime variety for training. Over that 11-year window the regime distribution is ~59% Expansion / 29% Late-cycle / 9% Recovery / 3% Contraction (4 Contraction months, 11 Recovery months out of 132). Walk-forward CV produces too few folds and folds whose train windows have never seen the minority class the test window contains. Verified: a real walk-forward run gave Fold-0 macro-F1 0.14 and a degenerate Fold-1 macro-F1 1.00 (single-class test window) — meaningless. Root cause is insufficient regime variety in 11 years, not a model bug. The classifier code is correct; the label distribution requires at least three recession episodes to produce defensible folds.

(b) yfinance FUNDAMENTALS DEAD-END — the stock screener cannot have an honest MVP on free data. yfinance returns only ~4-5 annual fundamental periods, all current-snapshot with fiscal-period-end dates and NO filing dates / NO vintage. Verified across AAPL/MSFT/JPM/XOM/WMT. This is fatal on two counts: (1) ~4 annual snapshots minus the unlabeled tail minus a 12-month embargo leaves ~1 usable train/test split — not walk-forward CV; (2) no point-in-time info means the look-ahead bias cannot even be quarantined honestly. The screener CANNOT have an honest MVP until Sharadar SF1 is available.

(c) NFCICREDIT LOOK-AHEAD IN LABELING RULES — NFCICREDIT (Chicago Fed National Financial Conditions Credit subindex) is backfilled to the 1970s but was only constructed and published from approximately 2010. Pre-2010 values were not knowable in real time. NFCICREDIT appears in BOTH the Contraction rule and the Late-cycle rule. Verified that the Contraction credit-arm is the deciding factor for the 1990 and early-2008 Contraction months — so this bias materially affects pre-2010 labels. NFCICREDIT must be removed from the rules (it may remain a model feature, since the model learns from it at inference time, not rule-time).

(d) CONTRACTION RULE IS A PEAK DETECTOR, NOT A RECESSION PERIOD DETECTOR — The Contraction rule requires unrate_3mo_change > 0.5, which only fires during the steepest 1-3 months of an unemployment climb. Verified over 1988-2024: 1990-91 recession got 1 Contraction month out of 13 NBER months; 2001 got 1 of 14; 2008-09 got 11 of 23; 2020 got 4 of 10. The rule catches recession peaks, not recession periods. As designed, it FAILS the >=50% NBER-overlap check for 1990-91 and 2001. This is a feature, not a bug — but the NBER-overlap validation expectation must be redefined accordingly (see Decision below).

(e) BAA10Y CANNOT SEE THE 1990-91 RECESSION — BAA10Y (corporate-Treasury credit spread, honest back to 1986, the intended NFCICREDIT replacement in the Contraction rule) stayed at calm-year levels (~2.0–2.35 percentage points) through the entire 1990-91 recession, indistinguishable from calm 2017. The 1990-91 episode was a labor-driven, oil-shock recession, not a credit-stress recession. No BAA10Y threshold can catch it. This is the direct reason the training window is restricted to 2000-2024 (three recessions with clear credit-spread signatures: 2001, 2008-09, 2020).

Decision and current plan

The stock screener is DEFERRED. User declined to pay for Sharadar SF1 (~$70/mo via Nasdaq Data Link) at this time. A stock_screener_mvp.md design doc has NOT been written — it is a future task for Sharadar day. No implementation work on the screener until that subscription exists.
The REGIME CLASSIFIER is the MVP, on free FRED data. All current implementation effort is here.
Chosen approach is "Path 2 (lighter fix)" — swap the look-ahead-biased NFCICREDIT out of the rules, re-derive thresholds for its honest replacement (BAA10Y), and restrict the training window to 2000-2024.

The three concrete steps of Path 2:

Step 1 (NEXT): Swap NFCICREDIT → BAA10Y in the Contraction and Late-cycle rules. Thresholds must be re-derived for BAA10Y's units (percentage spread, not a z-score index): Contraction credit path around BAA10Y >= 3.0; Late-cycle credit condition re-derived similarly. Start with a BAA10Y data diagnostic to set the thresholds, then update the rule code and tests, then re-run the label diagnostic over 2000-2024.

Step 2: Redefine the NBER-overlap validation expectation. Contraction = the ACUTE phase of a downturn, NOT the entire NBER recession. The validate_against_nber check changes from ">=50% of NBER recession months are Contraction" to "every NBER recession contains >=1 Contraction month, and Contraction months fall within NBER recessions." Document this definition explicitly in both code and docs.

Step 3: Restrict training/label window to 2000-2024. Three clean recessions. 1990s excluded because BAA10Y cannot detect the 1990-91 non-credit recession (finding (e)).

When Step 1 is implemented, CLAUDE.md and regime_classifier.md both name NFCICREDIT in the rules and must be updated with the swap rationale recorded. This is a methodology-lock change.

Bug discoveries documented (the README narrative)
Each is a one-paragraph story showing engineering judgment. In order of discovery:

Survivorship bias in v1.0 (Wikipedia roster) → pivot to fja05680 change log. The original screener used the current Wikipedia S&P 500 table, silently including stocks that joined the index after the training period. Portfolio-grade work requires point-in-time constituents; rebuilt against a frozen, hash-pinned change log with validation against known addition/removal events.

ICE data truncation (BAMLH series limited to rolling 3-year window after late-2024 licensing change) → replaced with NFCICREDIT (Chicago Fed). FRED dropped the ICE high-yield spread series to a rolling 3-year window due to a licensing renegotiation. Discovered when the FRED client returned a shorter-than-expected history. Replaced with the Chicago Fed National Financial Conditions Credit subindex, which has a long free history and similar credit-stress signal.

BLS revision drift (UNRATE April 2020 revised 14.7 → 14.8 over time) → vintage assertion tightens to original publication value. The integration test asserted the at-vintage UNRATE value against the number available today, which had been revised upward. Demonstrates vintage-aware testing: the assertion must be against the value published at the original release date, not the current revised figure.

Calendar boundary off-by-one in walk-forward CV (business-day-adjusted month-ends caused non-uniform folds on real data, invisible against synthetic month-ends) → calendar-month anchor fix. The fold engine computed fold boundaries using business-day-adjusted month-ends, which produced folds of inconsistent length when the pandas offset fell on a weekend. Invisible in unit tests using synthetic uniform dates; caught only when real membership data exposed the irregular spacing.

Cache range-blindness in FRED client (long-range query served by previously-cached narrow-range fetch returned forward-filled garbage) → range-coverage invariant. The client read from cache whenever any cache file existed, regardless of whether the cached date range covered the requested range. A cached 2020-2024 file would silently serve a 2014-2024 request, forward-filling the missing 2014-2019 data. Fixed by checking that the cached range fully covers the requested range before serving from cache.

Rules-based label flickering (initial regime rules produced month-to-month flips; revised with 6-month smoothing in Recovery, OR-clause in Contraction, threshold tightening in Late-cycle) → stable regime sequences. The first rule draft produced regimes that flickered between adjacent classifications month-to-month, inconsistent with the slow-moving nature of macro cycles. Resolved by adding hysteresis to Recovery exit (6-month smoothing), broadening the Contraction OR-clause, and tightening Late-cycle thresholds.

NFCICREDIT vintaging failure (daily market series and USREC have no ALFRED vintage archive) → per-series "vintaged" flag with current-value fallback. When the regime classifier attempted vintage-aware queries for non-revised series (yield spreads, VIX, FEDFUNDS, USREC), ALFRED returned "series not found" because those series have no vintage archive — they are never restated, so the concept of a vintage doesn't apply. Fixed by adding a per-series vintaged bool in REGIME_SERIES: un-vintaged series fall back to current-values queries (legitimate because current == point-in-time for market-observed or never-restated series), logging a WARNING for traceability.

NFCICREDIT look-ahead bias in labeling rules (constructed from ~2010, backfilled to 1970s) → must be swapped for BAA10Y. NFCICREDIT was chosen as the credit-stress signal because BAMLH was truncated (see above). Discovered this session via REPL: NFCICREDIT was not publicly available pre-2010, but ALFRED backfills it to 1970. Using it in the labeling rules silently injects knowledge that was not available in real time for decades of training data. BAA10Y (Moody's corporate-Treasury spread, available since 1986 with no look-ahead issue) is the replacement. Thresholds must be re-derived in BAA10Y's native units.

Contraction rule is a peak detector, not a period detector (unrate_3mo_change > 0.5 catches only the steepest months) → redefine NBER-overlap expectation from >=50% of recession months to >=1 Contraction month per recession within-NBER-bounds. The unemployment-spike threshold correctly identifies the acute onset of labor-market stress, but NBER recessions last 14-23 months while the threshold fires for only 1-11 of them. Rather than changing the rule (which would dilute its real-time signal), the validation expectation changes: Contraction is redefined as the acute phase, and the sanity check becomes "every NBER recession contains at least one Contraction month, and Contraction months do not appear in non-recession periods."

BAA10Y cannot detect the 1990-91 non-credit recession (spread ~2.0–2.35pp, indistinguishable from calm years) → training window restricted to 2000-2024. The 1990-91 recession was oil-shock and labor-driven; BAA10Y barely moved. No credit-spread threshold distinguishes it from calm 2017. Excluding the 1990s is the methodologically honest choice: better to openly restrict the training window than to ship a model that silently fails on a whole regime type. Documented explicitly so the limitation is part of the portfolio narrative, not hidden.

Where we are right now (end of this session)
The classifier code (regime_classifier.py + run_walk_forward) is complete and tested. The labeling rules have a known look-ahead bug (NFCICREDIT) and a known scope limitation (2014-2024 is not enough regime variety). Both are understood and documented. The stock screener is deferred pending Sharadar access.

Next action: implement Path 2 Step 1 — swap NFCICREDIT out of the Contraction and Late-cycle rules, replacing with BAA10Y at re-derived thresholds. Start with a data diagnostic over 2000-2024 BAA10Y to set the threshold, then update compute_labels() in regime_labels.py, update affected unit tests, re-run the label diagnostic, and run validate_against_nber over 2000-2024 with the new ">=1 Contraction month per recession" expectation. The Late-cycle rule's NFCICREDIT dependency must be handled in the same swap (flagged this session but not yet changed).
Tooling notes

Local: Apple Silicon Mac Studio, Python 3.11.15 venv at repo root
Repo conventions: ruff formatting, type hints on public functions, logging not print, parquet not CSV, conventional commits
Ad-hoc verification scripts live in scripts/ and run with PYTHONPATH=. python scripts/<name>.py
Test markers: fast (default, run in CI) and slow (real network/API, skip in CI, run locally with pytest -m slow)
.env holds FRED_API_KEY; loaded via load_dotenv() at module import time in both production code and test files (the latter is critical — pytest.mark.skipif evaluates at collection time before fixtures run)

Open questions awaiting CG decisions
One open question: BAA10Y threshold values for the Contraction and Late-cycle rules. The thresholds must be derived from a data diagnostic over 2000-2024 before the rule code is changed. CG has not yet seen the diagnostic output; the threshold will be proposed after the diagnostic is run and reviewed.
Working principles I uphold for CG

I always recommend REPL checks against real data after each phase, not just CI green
I push back when their instruction is methodologically wrong (the embargo=12 correction is the canonical example)
I name when I made an error (the series-count miscount episode, the .date() bug in REPL snippets)
I keep the README narrative arc in mind — each bug discovery is a portfolio artifact, not just a problem to solve
