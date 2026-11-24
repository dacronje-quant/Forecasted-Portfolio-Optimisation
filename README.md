# Forecasted Portfolio Optimisation

Actionable monthly machine-learning stock selection, **walk-forward optimal strategy blending**, and **Avanza / Nordnet cost-aware execution** for **US + Swedish equities**, with leak-free backtesting and an interactive dashboard.

Every month it:

1. validates live Yahoo Finance OHLCV & USD/SEK FX data without incomplete-month corruption;
2. evaluates diverse *(model × optimiser)* strategies strictly out-of-sample (adjusted for multiple testing via the **Deflated Sharpe Ratio**) to blend the **most optimal strategy** walk-forward;
3. builds a long-only portfolio with single-stock, **country**, and **sector** caps, weekly Ledoit-Wolf covariance, dynamic OOS IC calibration, and L1 turnover penalties;
4. generates a **cash-safe SELL → BUY order ticket** tailored for **Avanza / Nordnet** (courtage + minimum SEK fee + 25 bps FX fee), supporting both **Smart DCA** (deploying your monthly 10 000 SEK deposit first to minimise churn) and **Full Rebalance**.

## Quick start (Windows / macOS / Linux)

Requires Python 3.10+.

```bash
git clone https://github.com/dacronje-quant/Forecasted-Portfolio-Optimisation.git
cd Forecasted-Portfolio-Optimisation
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

streamlit run app/dashboard.py
```

The dashboard opens directly on the **Monthly Action Plan** tab. Pick **Simulated demo** in the sidebar to try it offline.

## Monthly Workflow (CLI)

```bash
# 1. One-time setup: create your persistent portfolio state file
fpo init-portfolio --file my_portfolio.json --broker avanza_mini --cash 0
# (or import existing holdings: --from-csv my_holdings.csv)

# 2. Every month: run the fast optimal strategy engine with your 10 000 SEK deposit
fpo decide --portfolio my_portfolio.json --deposit 10000 --broker avanza_mini --mode smart_dca

# Add --apply after executing the trades at Avanza/Nordnet to update my_portfolio.json & trade log
fpo decide --portfolio my_portfolio.json --deposit 10000 --broker avanza_mini --mode smart_dca --apply

# Multi-asset Funds & UCITS ETFs tactical core (0%-fee Swedish fund proxies + optional active stock satellite)
fpo tactical --budget 550000 --broker nordnet_mini --exec-mode mutual_fund --with-satellite 150000

# Benchmark the walk-forward meta-strategy selector (Smooth Residual Mom, LightGBM XENDCG, HERC, NCO)
fpo meta-backtest --broker avanza_mini

# Run a single-strategy backtest
fpo backtest --model lgbm_rank --optimizer herc --broker avanza_mini --n-holdings 15
```

`fpo decide` prints the **Strategy Health** light (`GREEN` / `YELLOW` / `RED`), top-line **Verdict** (`DEPLOY CASH ONLY` / `REBALANCE` / `HOLD`), ordered **SELL → BUY** ticket with exact SEK fees and plain-English reasons, and saves:
- `outputs/decision_YYYY-MM.md` — full monthly Markdown brief
- `outputs/orders_YYYY-MM.csv` — execution ticket CSV

Broker presets: `avanza_mini`, `avanza_small`, `avanza_medium`, `nordnet_mini`, `nordnet_small`, `nordnet_normal`, or `custom`.

## The dashboard

| Tab | What you get |
|---|---|
| **Monthly Action Plan** | Strategy Health badge, top-line verdict, Avanza/Nordnet fee estimates, Smart DCA (10 000 SEK default) or Full Rebalance, inline holdings editor + `portfolio.json` import/export, ordered SELL → BUY ticket, Markdown brief download, OOS feature importance |
| **Optimal Strategy (Meta)** | Walk-forward candidate leaderboard (Smooth Residual Mom, LightGBM XENDCG, Ridge, Ensemble × HERC/NCO/HRP) with **Deflated Sharpe Ratio**, dynamic strategy weights, sub-period stability |
| **Funds & ETFs (Core)** | Multi-asset tactical allocation (DAA + HERC, Causal Jump-Regime NCO, All-Weather HERC) across 0%-fee Swedish mutual fund proxies & PRIIPs UCITS ETFs (Global, Sweden, EM, Small Cap, Gold, Bonds, Cash) |
| **Performance** | Growth of 1 SEK vs benchmarks (incl. total-return `XACT-OMXS30.ST`), drawdowns, full metrics table, calendar-year & monthly heatmap |
| **Is it real?** | t-test & block-bootstrap, Probabilistic & Deflated Sharpe Ratio, 500 random portfolios, rank-IC, quintile spread, rolling excess & Sharpe |
| **Holdings** | Country & sector allocation over time, most-held stocks, turnover and cost drag, historical portfolio snapshot |
| **Features** | Univariate predictive power of every price, OHLCV, residual-momentum, Frog-in-the-Pan, and sector/country-neutral feature |
| **Method** | Full methodology, statistical safeguards, and known limitations |

## Validation

```bash
pytest -v
```

Automated tests in [tests/test_pipeline.py](tests/test_pipeline.py), [tests/test_actionable.py](tests/test_actionable.py) and [tests/test_modern_tactical.py](tests/test_modern_tactical.py) verify:
- strict no-look-ahead across features, walk-forward training, and meta-strategy weighting;
- incomplete-month protection (running on the 1st of a month never corrupts momentum or training labels);
- simultaneous single-stock, country, and sector cap feasibility even under concentrated top scores;
- smooth L1 turnover-penalized mean-variance & weekly covariance estimation;
- Avanza/Nordnet fee calculation, cash-safe SELL-then-BUY sizing, Smart DCA cash deployment, and `portfolio.json` persistence;
- Deflated Sharpe Ratio multiple-testing monotonicity and live data-hygiene alerts;
- Smooth Residual Momentum, LightGBM learning-to-rank, HERC/NCO constraint feasibility, ensemble z-score reuse in the meta search, and the multi-asset Funds & ETFs tactical engine.

## Project layout

```
src/fpo/
  universe.py    US + Swedish tickers, UCITS ETF universe + Swedish fund proxies, sector map, benchmarks
  config.py      BacktestConfig & Avanza/Nordnet BrokerFeeSchedule presets
  data.py        OHLCV Yahoo Finance download + incremental parquet cache, USD/EUR→SEK FX, hygiene validator
  features.py    Trading-day lookback, OHLCV, residual momentum & Frog-in-the-Pan features (no look-ahead)
  models.py      Ridge, LightGBM ranker, GBM, Smooth Residual Momentum, ensemble, OOS importance
  optimizer.py   Feasible country/sector selection, weekly cov, L1 turnover MV, HRP, HERC, NCO, dynamic IC, regime overlay
  backtest.py    Walk-forward monthly backtest with dynamic IC, broker fees, overlays, shared precomputed diagnostics
  meta.py        Walk-forward optimal strategy selection/blending, DSR leaderboard, on-disk meta cache
  tactical.py    Multi-asset Funds & ETFs core: DAA + HERC, causal Regime-NCO, All-Weather HERC, meta-selection
  portfolio.py   Persistent portfolio.json state, cash-safe Smart DCA / Full Rebalance execution engine
  metrics.py     Performance metrics, Deflated Sharpe Ratio, sub-period stability, significance tests
  recommend.py   Live recommendations, optimal monthly decision builder, Markdown brief renderer
  cli.py         `fpo decide` / `fpo tactical` / `fpo meta-backtest` / `fpo backtest` / `fpo recommend` / `fpo init-portfolio`
app/dashboard.py Streamlit dashboard
tests/           pytest suite (pipeline, actionable decision, modern strategy & tactical tests)
```

*Research and decision-support tool, not licensed investment advice.*

## License

GPL-3.0
