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
NFCICREDIT swap DONE (2026-06): NFCICREDIT removed from labeling rules (NOT the feature matrix — it may remain an inference-time model input). Replaced with BAA10Y at threshold 3.0pp in both the Contraction credit arm and the Late-cycle calm ceiling. CLAUDE.md and regime_classifier.md updated with the swap rationale. Threshold 3.0 was the cleanest separation point across all three in-scope recessions (2001, 2008-09, 2020).

Foundation phase: COMPLETE

Three modules built, ~140 tests passing, real-data verified via REPL at every milestone:

src/data_sources/sp500_membership.py — point-in-time S&P 500 constituents reconstructed from frozen change-log CSV. 72,521 rows × 4 columns, 144 month-ends 2014-2025. Pyarrow dict-encoded ticker columns. 19 unit tests + 1 slow integration test.
src/data_sources/fred_client.py — 20 macro/credit features + USREC as ground truth, organized in 7 categories. Vintage-aware queries via realtime_start=realtime_end=vintage_date. Per-series parquet caching with range-coverage invariant (cache invalidated if cached range narrower than requested — bug discovered and fixed). 35 unit tests + 1 slow integration with 4 anchor assertions (T10Y2Y inversion 2019, UNRATE latest and vintage, NFCICREDIT COVID stress). Per-series "vintaged" bool added this session (see supporting work below).
src/validation/walk_forward.py — calendar-month-anchored fold engine with embargo, expanding/sliding window, sklearn shim. 58 tests. Verified against real membership data: 8 folds for screener config, 0 embargo violations.

Regime classifier phase: COMPLETE (all phases, including Path 2)

src/feature_engineering/regime_labels.py — DONE. Deterministic rules-based label function. 4 regimes (Expansion / Late-cycle / Contraction / Recovery). BAA10Y replaces NFCICREDIT in both the Contraction credit arm and Late-cycle calm ceiling at threshold 3.0pp. 33 tests including slow real-data tests. New test test_real_data_every_nber_recession_has_acute_contraction encodes the acute-phase expectation (≥1 Contraction month per recession, within-NBER-bounds, 3-month post-trough allowance). validate_against_nber() added in prior session.
src/models/regime_classifier.py — COMPLETE. Phase 1 and Phase 2 both built and real-data verified via REPL. Training window 2000-2024 (300 months). First full training run executed; findings documented in "Findings that reshaped the plan" (f)-(i) below.
src/feature_engineering/macro_features.py — COMPLETE. v2 momentum features built (3mo/12mo deltas for UNRATE, FEDFUNDS, INDPRO, PAYEMS, ICSA; 12mo trailing z-scores for T10Y3M, BAA10Y, VIXCLS; current-regime one-hot prior). run_walk_forward() integrated with feature_builder parameter. 5 leakage and correctness tests.

scripts/regime_label_diagnostic.py — COMPLETE. Per-month label timeline over a date range for human review. Committed, gitignored output.

Path 2 — COMPLETED

NFCICREDIT → BAA10Y swap implemented in regime_labels.py at threshold 3.0pp for both the Contraction credit arm and the Late-cycle calm ceiling. All NFCICREDIT references removed from rule logic, required columns, derived features, and tests; only the historical-rationale docstring mentions it. 33 tests green including slow real-data tests.

2000-2024 labels validated against NBER: every recession contains Contraction months — 2001 (2 months, 25% of NBER episode), 2008-09 (11 months, 61%), 2020 (2 months, 100%). Only strays are 2020-05 and 2020-06, post-trough plateau at unemployment 11–13%, acceptable by design (Contraction captures the acute onset phase; extreme labor-market stress persisting post-trough is within the documented scope). Label distribution over 300 months: 199 Expansion / 65 Late-cycle / 19 Recovery / 17 Contraction.

CLAUDE.md and design doc updated with the methodology revision rationale. New slow integration test test_real_data_every_nber_recession_has_acute_contraction encodes the acute-phase expectation with a 3-month post-trough allowance.

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

(f) FIRST TRAINING RUN SHOWED THE MODEL LEARNS PERSISTENCE, NOT TURNS — The first full training run (train_period=150, embargo=3, test_period=12, step=12; 14 folds) produced LR avg macro-F1 0.5709 and XGB disqualified for catastrophic fold deficit (avg 0.5455, lost to LR on transition folds by more than the adoption threshold). The dominant pattern: perfect scores (1.0) on single-regime test windows where one regime fills the whole test window, near-zero on transition folds (folds 8-11 mean macro-F1 0.159). The model had learned class priors and regime persistence, not the signal needed to call a turn. This directly prompted the computation of a non-model persistence baseline (finding g).

(g) PERSISTENCE BASELINE BEATS TRAINED MODELS ON EVERY FOLD — Computing predict regime[T+3] = regime[T] (no model, no features, no training) yielded avg macro-F1 0.7575 and transition-fold mean 0.5231 — strictly dominating the best trained model on every single fold. This reframed the entire evaluation: the bar is persistence, not v1. Every future experiment must clear 0.7575 overall and 0.5231 on folds 8-11 to be considered a genuine improvement. Any result that beats v1 but not persistence is still a failure.

(h) TWO IMPLEMENTATION BUGS DISCOVERED DURING TRAINING — (1) WalkForwardCV class-count crash: the 2001 recession is underrepresented in early expanding windows (as few as 2 Contraction and 1 Recovery month in a 2000-2002 train window), triggering a sklearn exception when the calibration fold lacks enough class examples. Solved by setting train_period=150 so every train window includes adequate recession variety. (2) DTWEXBGS all-NaN crash: the broad-dollar-index series starts in 2006, leaving 6 years of NaN for 2000-2005 rows in training folds. The imputer failed when an entire column was all-NaN. Fixed with all-NaN-column drop plus a WARNING log, then per-fold training-mean imputation for partially-missing columns (commit 0b39f52).

(i) QUASI-SEPARATION DIAGNOSIS — LogisticRegression emitted RuntimeWarnings about overflow in matmul on minority-class folds. Initially suspected as a data pipeline bug. Diagnosed as quasi-separation: with class_weight='balanced' and ~8-example minority classes, the solver pushes coefficients toward ±∞ trying to fit those few samples perfectly. Column diagnostics showed no degenerate inputs (max |scaled| feature value 4.6–13.9, zero non-finite values). The warnings do not affect macro-F1, but minority-class coefficients are numerically unstable. Noted and documented, not fixed: reducing class_weight or adding stronger L2 would trade a cosmetic warning for further minority-class recall loss — a bad trade when T+3 transition prediction is already failing.

Process upgrades adopted

Pre-registration discipline: experiments are committed to docs/experiments/ with hypothesis, metric, success criteria, stopping rule, and falsifier BEFORE any code is written. First instance: docs/experiments/2026-06-v2-momentum-features.md. This prevents outcome-dependent framing and metric-chasing at machine speed.

Pre-commit ritual adopted: ruff check . && PYTHONPATH=. pytest -m "not slow". A stray unused import left over from the NFCICREDIT refactor broke CI for 4 consecutive pushes (runs #24-27, fixed in commit 1725964). The lesson: CI is feedback after the fact; the linter must run locally before every push.

Planned: adversarial review by a fresh Claude instance once the README exists; a scripts/reproduce.py asserting all committed numbers; an experiment runner that refuses to execute without a committed pre-registration file.

v2 experiment — RUN AND FAILED (verdict recorded in docs/experiments/2026-06-v2-momentum-features.md)

v2 added momentum features (3mo/12mo deltas for UNRATE, FEDFUNDS, INDPRO, PAYEMS, ICSA; 12mo trailing z-scores for T10Y3M, BAA10Y, VIXCLS) plus a current-regime one-hot prior, built in src/feature_engineering/macro_features.py with run_walk_forward() integration and 5 leakage/correctness tests. Run ID 20260609_202535.

Results: v2 LR avg macro-F1 0.6415 (needed >0.7575) — FAIL; v2 transition folds (8-11) mean 0.2315 (needed >0.5231) — FAIL. XGBoost averaged 0.6758, also below persistence. Falsifier not triggered (0.2315 vs v1's 0.159 exceeds ±0.05): momentum features carry some T+3 signal but not enough to beat naive persistence even with the current-regime prior as a feature.

STOPPING RULE IS BINDING: no v3 feature set, no model swap, no threshold adjustment. The forecaster is an honest negative result.

Decision and current plan

THE PRODUCT IS THE RULES-BASED NOWCASTER: validated against NBER over three recessions, honest point-in-time inputs, shipped via the Streamlit dashboard. The T+3 forecaster ships alongside it as a documented negative result, with the persistence comparison front and center.

Remaining work, in order:

(1) Streamlit dashboard for the nowcaster. Build with self-verifying loop: build → headless launch → verify renders with real FRED data → iterate to green. The dashboard shows current regime (rules-based), regime timeline chart, key contributing features, and as a separate panel the T+3 forecaster output with persistence comparison and honest Limitations.

(2) scripts/reproduce.py asserting all committed numbers (0.7575 persistence baseline, 0.6415 v2 macro-F1, NBER overlaps: ≥1 Contraction month per recession for 2001, 2008-09, 2020). Wire into CI so the numbers are machine-verified, not just human-stated.

(3) README with Limitations section drafted FIRST, then the bug-discovery narrative (~14 entries), then results. Draft Limitations before any achievements text — the honest shape of this project is a rules-based nowcaster with a machine-learning negative result, not a predictive model.

(4) Adversarial review by a fresh Claude instance once the README exists.

(5) Stock screener remains deferred to Sharadar day (~$70/mo via Nasdaq Data Link, declined for now).

LOOP POLICY (record verbatim): a loop is safe when its success criterion is a fixed objective spec (tests pass, lint clean, app renders). A loop whose criterion is a statistical metric on the research data is forbidden — metric-chasing loops are p-hacking at machine speed.

Bug discoveries documented (the README narrative)

Each is a one-paragraph story showing engineering judgment. In order of discovery:

Survivorship bias in v1.0 (Wikipedia roster) → pivot to fja05680 change log. The original screener used the current Wikipedia S&P 500 table, silently including stocks that joined the index after the training period. Portfolio-grade work requires point-in-time constituents; rebuilt against a frozen, hash-pinned change log with validation against known addition/removal events.

ICE data truncation (BAMLH series limited to rolling 3-year window after late-2024 licensing change) → replaced with NFCICREDIT (Chicago Fed). FRED dropped the ICE high-yield spread series to a rolling 3-year window due to a licensing renegotiation. Discovered when the FRED client returned a shorter-than-expected history. Replaced with the Chicago Fed National Financial Conditions Credit subindex, which has a long free history and similar credit-stress signal.

BLS revision drift (UNRATE April 2020 revised 14.7 → 14.8 over time) → vintage assertion tightens to original publication value. The integration test asserted the at-vintage UNRATE value against the number available today, which had been revised upward. Demonstrates vintage-aware testing: the assertion must be against the value published at the original release date, not the current revised figure.

Calendar boundary off-by-one in walk-forward CV (business-day-adjusted month-ends caused non-uniform folds on real data, invisible against synthetic month-ends) → calendar-month anchor fix. The fold engine computed fold boundaries using business-day-adjusted month-ends, which produced folds of inconsistent length when the pandas offset fell on a weekend. Invisible in unit tests using synthetic uniform dates; caught only when real membership data exposed the irregular spacing.

Cache range-blindness in FRED client (long-range query served by previously-cached narrow-range fetch returned forward-filled garbage) → range-coverage invariant. The client read from cache whenever any cache file existed, regardless of whether the cached date range covered the requested range. A cached 2020-2024 file would silently serve a 2014-2024 request, forward-filling the missing 2014-2019 data. Fixed by checking that the cached range fully covers the requested range before serving from cache.

Rules-based label flickering (initial regime rules produced month-to-month flips; revised with 6-month smoothing in Recovery, OR-clause in Contraction, threshold tightening in Late-cycle) → stable regime sequences. The first rule draft produced regimes that flickered between adjacent classifications month-to-month, inconsistent with the slow-moving nature of macro cycles. Resolved by adding hysteresis to Recovery exit (6-month smoothing), broadening the Contraction OR-clause, and tightening Late-cycle thresholds.

NFCICREDIT vintaging failure (daily market series and USREC have no ALFRED vintage archive) → per-series "vintaged" flag with current-value fallback. When the regime classifier attempted vintage-aware queries for non-revised series (yield spreads, VIX, FEDFUNDS, USREC), ALFRED returned "series not found" because those series have no vintage archive — they are never restated, so the concept of a vintage doesn't apply. Fixed by adding a per-series vintaged bool in REGIME_SERIES: un-vintaged series fall back to current-values queries (legitimate because current == point-in-time for market-observed or never-restated series), logging a WARNING for traceability.

NFCICREDIT look-ahead bias in labeling rules (constructed from ~2010, backfilled to 1970s) → swapped for BAA10Y at threshold 3.0pp. NFCICREDIT was chosen as the credit-stress signal because BAMLH was truncated (see above). Discovered via REPL: NFCICREDIT was not publicly available pre-2010, but ALFRED backfills it to 1970. Using it in the labeling rules silently injects knowledge that was not available in real time for decades of training data. BAA10Y (Moody's corporate-Treasury spread, available since 1986 with no look-ahead issue) is the replacement; threshold 3.0pp separates clean recession credit stress from expansions across 2001, 2008-09, and 2020.

Contraction rule is a peak detector, not a period detector (unrate_3mo_change > 0.5 catches only the steepest months) → redefine NBER-overlap expectation from >=50% of recession months to >=1 Contraction month per recession within-NBER-bounds. The unemployment-spike threshold correctly identifies the acute onset of labor-market stress, but NBER recessions last 14-23 months while the threshold fires for only 1-11 of them. Rather than changing the rule (which would dilute its real-time signal), the validation expectation changes: Contraction is redefined as the acute phase, and the sanity check becomes "every NBER recession contains at least one Contraction month, and Contraction months do not appear in non-recession periods."

BAA10Y cannot detect the 1990-91 non-credit recession (spread ~2.0–2.35pp, indistinguishable from calm years) → training window restricted to 2000-2024. The 1990-91 recession was oil-shock and labor-driven; BAA10Y barely moved. No credit-spread threshold distinguishes it from calm 2017. Excluding the 1990s is the methodologically honest choice: better to openly restrict the training window than to ship a model that silently fails on a whole regime type. Documented explicitly so the limitation is part of the portfolio narrative, not hidden.

Persistence baseline beats all trained models (T+3 monthly regime prediction, 14 folds, 0.7575 avg macro-F1 with no model) → reframed evaluation bar from v1 LR to naive persistence. After the first full training run showed LR avg macro-F1 0.5709 and XGB even lower, computing the no-model baseline revealed the trained models were learning persistence less efficiently than just predicting persistence directly. The discipline of computing a non-model baseline before declaring a trained model an improvement is now standard for this project. Every future model result is reported relative to persistence, not relative to the prior model version. The v2 experiment (0.6415 avg, 0.2315 on transition folds) was measured and reported against this bar.

WalkForwardCV crash on early minority-class folds and DTWEXBGS all-NaN columns → train_period=150 minimum and per-fold training-mean imputation. Two bugs surfaced during the first training run: (1) early expanding windows over 2000-2001 sometimes contained only 2 Contraction and 1 Recovery month — too few for sklearn's calibration CV step, which raised a class-count exception. Setting train_period=150 ensures every train window contains enough recession variety. (2) DTWEXBGS (broad dollar index) starts in 2006, leaving 6 years of NaN for 2000-2005 training rows. The imputer failed when entire columns were all-NaN in early folds. Fixed with all-NaN-column drop plus a WARNING log, then per-fold training-mean imputation for partially-missing columns (commit 0b39f52). The two fixes were committed together since both blocked the same training run.

Quasi-separation in LogisticRegression with balanced class weights on 8-example minority folds → noted and documented, not fixed. Overflow RuntimeWarnings in scipy's LR solver appeared during transition folds, initially suspected as a data pipeline bug. Diagnosed as quasi-separation: class_weight='balanced' amplifies minority-class sample weights to ~10x the majority, and with ~8 minority examples the solver pushes coefficients toward ±∞ to achieve near-perfect separation. Column diagnostics showed no degenerate inputs (max |scaled| feature value 4.6–13.9, zero non-finite values). The warnings do not affect macro-F1, but minority-class coefficients are numerically unstable. Fixing it by reducing class_weight or adding stronger L2 would trade a cosmetic warning for further minority-class recall degradation — a bad trade when transition-fold prediction is already the core failure mode.

CI lint break (stray unused import from NFCICREDIT refactor broke runs #24-27 for 4 consecutive pushes) → pre-commit ritual added. After the NFCICREDIT-to-BAA10Y swap removed a feature from the code that a module had been importing, the now-unused import was left behind (fixed in commit 1725964). Four red CI runs elapsed before the fix. The lesson: ruff check is the fastest possible feedback loop and must run locally before every push, not only in CI. Pre-commit ritual is now: ruff check . && PYTHONPATH=. pytest -m "not slow".

Tooling notes

Local: Apple Silicon Mac Studio, Python 3.11.15 venv at repo root
Repo conventions: ruff formatting, type hints on public functions, logging not print, parquet not CSV, conventional commits
Ad-hoc verification scripts live in scripts/ and run with PYTHONPATH=. python scripts/<name>.py
Test markers: fast (default, run in CI) and slow (real network/API, skip in CI, run locally with pytest -m slow)
.env holds FRED_API_KEY; loaded via load_dotenv() at module import time in both production code and test files (the latter is critical — pytest.mark.skipif evaluates at collection time before fixtures run)

Working principles I uphold for CG

I always recommend REPL checks against real data after each phase, not just CI green
I push back when their instruction is methodologically wrong (the embargo=12 correction is the canonical example)
I name when I made an error (the series-count miscount episode, the .date() bug in REPL snippets)
I keep the README narrative arc in mind — each bug discovery is a portfolio artifact, not just a problem to solve
