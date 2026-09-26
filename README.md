# Forecasted Portfolio Optimisation

Monthly machine-learning stock selection and portfolio optimisation for **US + Swedish equities**,
with leak-free walk-forward backtesting against the index funds you could buy instead, and an
interactive dashboard to explore the results.

Every month it:

1. scores ~115 large-cap stocks (≈70 US, ≈45 Swedish) with a machine-learning model trained only on the past;
2. builds an optimised long-only portfolio (mean-variance, minimum variance, HRP, ...);
3. tells you **exactly what to buy**: shares per stock for your budget, in SEK.

The backtest asks: *would this have beaten an S&P 500, MSCI World or OMXS30 index fund, after trading
costs and currency effects, and is the difference more than luck?*

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

The dashboard opens in your browser. The first run downloads ~20 years of prices from Yahoo Finance
(cached for the day in `data_cache/`). Pick **Simulated demo** in the sidebar to try it offline.

### Command line

```bash
# Walk-forward backtest, prints metrics + significance tests, saves CSVs to outputs/
fpo backtest --model ensemble --optimizer mean_variance --n-holdings 15

# This month's portfolio for 50 000 SEK
fpo recommend --budget 50000

# ...and the trades needed from what you already own (CSV with columns ticker,shares)
fpo recommend --budget 250000 --holdings my_holdings.csv
```

Run `fpo backtest --help` for every option (markets, currency, costs, weight caps, ...).

## The dashboard

| Tab | What you get |
|---|---|
| **Performance** | Growth of 1 SEK vs benchmarks (log/linear), drawdowns, full metrics table, calendar-year and monthly heatmap |
| **Is it real?** | t-test and bootstrap of excess returns, probabilistic Sharpe, percentile vs 500 random portfolios, monthly rank-IC, quintile spread, rolling excess return and Sharpe |
| **Holdings** | Country allocation over time, most-held stocks, turnover and cost drag, portfolio at any past date with each stock's contribution |
| **This month's portfolio** | Stocks, weights and whole-share quantities for your budget, CSV download, trade list from your current holdings, model explanation (feature importance) |
| **Features** | Predictive power of each input on its own |
| **Compare models** | Same settings, different models: does ML beat simple momentum? |
| **Method** | How it works and its limitations |

Everything in the sidebar (model, optimiser, number of stocks, weight caps, country cap, costs,
currency, markets, backtest period) can be changed and re-run.

## Method

**Features** (from past prices only, converted to cross-sectional percentile ranks): momentum over
1/3/6/12 months and 12-minus-1, 1-week reversal, 1/3/12-month and downside volatility, skewness,
6-month max drawdown, distance to the 52-week high, trend vs the 200-day average, beta and idiosyncratic
volatility vs the universe, share of positive months, and country.

**Models** predict next month's cross-sectional return rank:
`gbm` (histogram gradient boosting), `random_forest`, `ridge`, `mlp` (neural network, a nod to this repo's
origins in the `nnfor` R package), `ensemble` (average of all four), and `momentum` (non-ML baseline).

**Walk-forward training**: at each month-end *t*, the model is trained only on months whose outcome was
known before *t* (expanding or rolling window, optional embargo), refit every *k* months.

**Optimisers**: `mean_variance` (expected alpha = IC × volatility × score, per Grinold & Kahn, with a
Ledoit-Wolf shrunk covariance), `min_variance`, `hrp` (hierarchical risk parity), `inverse_vol`, `score`,
`equal`. All long-only, with max/min position size, an optional max country weight, and a hold buffer that
keeps existing positions while they still rank well, to limit turnover.

**Costs**: a per-trade cost (default 15 bps) plus an FX fee (default 10 bps) on foreign-currency trades,
applied to actual turnover from the drifted portfolio. Returns are measured in SEK (or USD), so currency
moves are included.

**Benchmarks**: S&P 500 (SPY), MSCI World (URTH), OMXS30 (^OMX), a 50/50 SPY/OMXS30 blend, and an
equal-weight portfolio of the same universe.

### Validation

`tests/test_pipeline.py` checks, among other things, that:

- changing future prices never changes today's features (no look-ahead);
- training dates are strictly before each prediction date;
- on simulated data with **no** predictable signal, the model finds no edge (catches leakage);
- on simulated data **with** a signal, the model finds it and beats the equal-weight universe;
- every optimiser returns valid long-only weights within the caps.

```bash
pytest
```

### Limitations: read before investing

- **Survivorship bias.** The universe is *today's* large caps, so companies that collapsed are missing.
  This flatters any strategy on this universe, so compare against the **equal-weight universe** and the
  **random portfolios**, not only the index.
- **^OMX is a price index** (no dividends, ≈3–4 %/yr), which understates an OMXS30 fund. SPY and URTH include dividends.
- Yahoo Finance data can have errors. Taxes (ISK/KF), minimum brokerage fees and bid/ask on small orders are simplified.
- Trying many settings and choosing the best one overfits. Prefer settings that hold up across models and periods.

*Research tool, not investment advice.*

## Project layout

```
src/fpo/
  universe.py    US + Swedish tickers, benchmarks
  data.py        Yahoo Finance download + cache, FX conversion, simulated market
  features.py    feature engineering (no look-ahead)
  models.py      ML models, walk-forward training, IC, feature importance
  optimizer.py   holding selection + portfolio optimisers
  backtest.py    monthly walk-forward backtest with costs, benchmarks, diagnostics
  metrics.py     performance metrics and significance tests
  recommend.py   this month's portfolio and whole-share allocation
  cli.py         `fpo backtest` / `fpo recommend`
app/dashboard.py Streamlit dashboard
tests/           pytest suite
```

## License

GPL-3.0
