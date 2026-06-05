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

Foundation phase: COMPLETE
Three modules built, ~140 tests passing, real-data verified via REPL at every milestone:

src/data_sources/sp500_membership.py — point-in-time S&P 500 constituents reconstructed from frozen change-log CSV. 72,521 rows × 4 columns, 144 month-ends 2014-2025. Pyarrow dict-encoded ticker columns. 19 unit tests + 1 slow integration test.
src/data_sources/fred_client.py — 20 macro/credit features + USREC as ground truth, organized in 7 categories. Vintage-aware queries via realtime_start=realtime_end=vintage_date. Per-series parquet caching with range-coverage invariant (cache invalidated if cached range narrower than requested — bug discovered and fixed). 35 unit tests + 1 slow integration with 4 anchor assertions (T10Y2Y inversion 2019, UNRATE latest and vintage, NFCICREDIT COVID stress).
src/validation/walk_forward.py — calendar-month-anchored fold engine with embargo, expanding/sliding window, sklearn shim. 58 tests. Verified against real membership data: 8 folds for screener config, 0 embargo violations.

Regime classifier phase: PARTIAL

src/feature_engineering/regime_labels.py — DONE. Deterministic rules-based label function. 4 regimes (Expansion / Late-cycle / Contraction / Recovery). 26 tests including 6 slow integration tests pinning known regime windows (COVID Contraction, 2018-2019 Late-cycle, post-COVID Recovery, 2024 inverted-curve Late-cycle). Threshold constants surfaced as module constants with cross-consistency tests guarding the 0.5/0.7/1.5 ladder.
src/models/regime_classifier.py — NOT STARTED. Design doc exists at docs/design/regime_classifier.md and specifies:

Label task: predict regime at T+3 given features at T
Models: LR baseline (class_weight='balanced', L2) + XGBoost challenger (max_depth=3 — aggressively shallow for ~120 training rows in early folds)
Calibration: CalibratedClassifierCV(method='isotonic', cv=3)
Walk-forward integration: embargo=3, vintage_date threaded into every get_features_matrix call
Drop trailing 3 months (no valid T+3 label yet) from every train and test fold
Adoption gate: XGBoost only if +5pp macro-F1 over LR, ±2pp tiebreaker → LR



Bug discoveries documented (the README narrative)
Each is a one-paragraph story showing engineering judgment. In order of discovery:

Survivorship bias in v1.0 (Wikipedia roster) → pivot to fja05680 change log
ICE data truncation (BAMLH series limited to rolling 3-year window after late-2024 licensing change) → replaced with NFCICREDIT (Chicago Fed)
BLS revision drift (UNRATE April 2020 revised 14.7 → 14.8 over time) → vintage assertion tightens to original publication value
Calendar boundary off-by-one in walk-forward CV (business-day-adjusted month-ends caused non-uniform folds on real data, invisible against synthetic month-ends) → calendar-month anchor fix
Cache range-blindness in FRED client (long-range query served by previously-cached narrow-range fetch returned forward-filled garbage) → range-coverage invariant
Rules-based label flickering (initial regime rules produced month-to-month flips; revised with 6-month smoothing in Recovery, OR-clause in Contraction, threshold tightening in Late-cycle)

Where we are right now (last conversation turn)
Regime labels approved and locked. regime_labels.py is the stable ground-truth generator. 132 fast + 8 slow tests passing. CI green.
Next: regime classifier implementation. Design doc is approved (modulo the embargo=3 update which was completed). Need to plan phased implementation per the established pattern. CG has not yet been given the phase 1 implementation prompt.
Tooling notes

Local: Apple Silicon Mac Studio, Python 3.11.15 venv at repo root
Repo conventions: ruff formatting, type hints on public functions, logging not print, parquet not CSV, conventional commits
Ad-hoc verification scripts live in scripts/ and run with PYTHONPATH=. python scripts/<name>.py
Test markers: fast (default, run in CI) and slow (real network/API, skip in CI, run locally with pytest -m slow)
.env holds FRED_API_KEY; loaded via load_dotenv() at module import time in both production code and test files (the latter is critical — pytest.mark.skipif evaluates at collection time before fixtures run)

Open questions awaiting CG decisions
None right now — the next move is the phase 1 implementation prompt for the regime classifier.
Working principles I uphold for CG

I always recommend REPL checks against real data after each phase, not just CI green
I push back when their instruction is methodologically wrong (the embargo=12 correction is the canonical example)
I name when I made an error (the series-count miscount episode, the .date() bug in REPL snippets)
I keep the README narrative arc in mind — each bug discovery is a portfolio artifact, not just a problem to solve
