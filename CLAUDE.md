# CLAUDE.md

Project context for Claude Code. Read this before doing any work in this repo.

## Project Goal

End-to-end ML pipeline for equity analysis, run locally on Apple Silicon. Two
models share infrastructure:

1. **Stock screener** — XGBoost classifier predicting whether an S&P 500 stock
   will outperform the index over the next 12 months, given fundamentals.
2. **Market regime classifier** — XGBoost (or HMM) classifier assigning the
   current market to one of four regimes (Expansion / Late-cycle / Contraction
   / Recovery) given macro and credit features.

Both surface through a single Streamlit dashboard.

## Architecture

```
src/
  data_sources/
    ivv_constituents.py   # Point-in-time S&P 500 membership from iShares IVV
                          # holdings (Wayback Machine archive).
    fred_client.py        # Cached FRED API wrapper for macro/credit series.
    market_client.py      # yfinance wrappers — PRICES ONLY, not fundamentals.
    shiller_client.py     # CAPE ratio from shillerdata.com.
    fundamentals_client.py # Point-in-time fundamentals (TBD: Sharadar or
                          # equivalent — yfinance is NOT acceptable here).
  feature_engineering/
    stock_features.py     # Cross-sectional ranks/z-scores per date.
    macro_features.py     # Resampling, lags, transformations for regime model.
    regime_labels.py      # Ground-truth regime labels (rules-based + NBER).
  models/
    stock_model.py        # Training + inference for the screener.
    regime_model.py       # Training + inference for the regime classifier.
  validation/
    walk_forward.py       # Expanding-window CV with embargo period.
  dashboard.py            # Streamlit app — both models, two pages.

data/                     # Gitignored. Parquet preferred over CSV.
  raw/
  processed/
models/                   # Gitignored. Serialized model artifacts.
tests/                    # pytest, run on every push via GitHub Actions.
```

## Methodology Decisions (LOCKED — do not relitigate without explicit ask)

### Stock screener
- **Survivorship bias**: training data MUST use point-in-time S&P 500
  constituents from the IVV scraper. Never `pd.read_html` the current Wikipedia
  roster for training data.
- **Look-ahead bias**: fundamentals MUST be as-of the prediction date, never
  the current snapshot. yfinance is acceptable for prices, NEVER for
  fundamentals in training.
- **Validation**: walk-forward / expanding-window CV with a 1-month embargo
  between train and test. NEVER use `train_test_split` or k-fold on time
  series data.
- **Features**: convert raw fundamentals to cross-sectional ranks or z-scores
  within each date, optionally sector-neutralized. Raw P/E etc. does not
  generalize across regimes.
- **Calibration**: wrap final model in `CalibratedClassifierCV` (isotonic) on
  a held-out fold. The Streamlit confidence gauge must show calibrated
  probabilities.
- **Explainability**: SHAP values, not XGBoost `feature_importances_`.
  Per-prediction SHAP waterfall in the dashboard.
- **Evaluation**: include a backtest harness — top-quintile long, bottom-quintile
  short, monthly rebalance, 15bps transaction costs. Report Sharpe, max
  drawdown, hit rate, decile spread.

### Regime classifier
- **Not** a binary drawdown predictor. The taxonomy is 4 regimes, not 2.
- **Ground truth**: rules-based labels using yield curve, unemployment trend,
  credit spreads, validated against NBER recession dates (FRED series USREC).
  Document the rules clearly in `regime_labels.py`.
- **Baseline**: always train a logistic regression baseline alongside XGBoost.
  If XGBoost doesn't beat it meaningfully, use the simpler model.
- **Validation**: same walk-forward CV as the stock model.
- **Sample size discipline**: ~35 years of monthly macro data is ~420 obs.
  Be skeptical of complex models. Regularize aggressively.

## Data Sources

| Data | Source | Cost | Notes |
|------|--------|------|-------|
| S&P 500 constituents (point-in-time) | iShares IVV via Wayback Machine | Free | Build the scraper. Verify against known events (SVB removal Mar 2023). |
| Macro/credit series | FRED API (`fredapi`) | Free | Primary source for ~70% of regime features. |
| Equity prices | yfinance | Free | Acceptable here. Cache aggressively. |
| Shiller CAPE | shillerdata.com / GitHub mirror | Free | Monthly. |
| Point-in-time fundamentals | TBD — Sharadar SF1 (~$50/mo) preferred | Paid | DO NOT use yfinance for this. |
| Market breadth | Computed from IVV constituents | Free | Reuses constituents work. |

## Conventions

- **Python**: 3.11+, isolated `venv` at repo root.
- **Formatting**: `ruff format` + `ruff check`. No black, no isort separately.
- **Types**: type hints on all public functions. `mypy --strict` is the goal,
  not yet enforced.
- **Tests**: `pytest`. Every data client and feature transformer needs at
  least a smoke test. Target functions get unit tests.
- **Logging**: `logging` module, not `print`. Module-level logger:
  `logger = logging.getLogger(__name__)`.
- **Data files**: parquet, not CSV. Faster, typed, smaller.
- **Caching**: every API client caches to `data/raw/` with a timestamp.
  Default TTL 24 hours for daily data, 1 hour for intraday.
- **Secrets**: `.env` file, gitignored. `python-dotenv` to load. FRED API key
  goes there.
- **Commits**: conventional commits (`feat:`, `fix:`, `refactor:`, `docs:`,
  `test:`). One logical change per commit.

## Do Not Do

- Do not use `yfinance` for fundamentals in training data. It returns current
  snapshots, which causes look-ahead bias.
- Do not use `train_test_split` or `KFold` on time series. Walk-forward only.
- Do not use `pd.read_html` on Wikipedia for the constituent list outside of
  the current-snapshot inference path. Training data uses the IVV scraper.
- Do not predict binary "crash / no crash". Predict regime distribution or
  forward volatility magnitude.
- Do not use raw `feature_importances_` from XGBoost in user-facing output.
  SHAP only.
- Do not show uncalibrated `predict_proba` outputs as "confidence" in the
  dashboard. Calibrate first.
- Do not commit `data/`, `models/`, `venv/`, `.env`, or `.DS_Store`. Check
  `.gitignore` before adding files.
- Do not add Tableau or FastAPI scaffolding. Streamlit + Plotly is sufficient.
- Do not silently catch exceptions in data clients. Log and re-raise, or fail
  loudly. Silent failures in financial data pipelines are dangerous.

## Current State (as of handoff)

- This repo is a clean rebuild. v1.0 of the stock screener lives at
  github.com/cgarcia607-wq/ml-value-screener for reference only — it has
  survivorship bias and uses current-snapshot fundamentals, and is not being
  ported here. Build fresh against the methodology above.
- No tests yet.
- No CI yet.
- No CLAUDE.md hierarchy in subdirectories yet.

## Next Up

1. Build `src/data_sources/ivv_constituents.py` — Wayback Machine scraper for
   point-in-time S&P 500 membership. Validate against known events.
2. Build `src/data_sources/fred_client.py` — cached FRED wrapper for the
   macro/credit series listed in design notes.
3. Build `src/validation/walk_forward.py` — expanding-window CV with embargo.
4. Refactor the stock model to use 1+2+3.
5. Build the regime classifier on top of 2+3.

Plan before executing on any of these. Use plan mode.
