### How the strategy works

**Every month-end, for each stock in the universe (≈70 US + ≈45 Swedish large caps):**

1. **Features** are computed from prices up to that day only: momentum (1, 3, 6, 12 and 12-minus-1 months),
   short-term reversal, volatility (1/3/12 months, downside), skewness, drawdown, distance from the 52-week high,
   trend vs the 200-day average, beta and idiosyncratic volatility, consistency of positive months, and country.
   Each feature is converted to a cross-sectional percentile rank so it is comparable across time and regimes.
2. **Model**: a machine-learning model predicts each stock's *relative* return rank for the next month.
   Options: gradient boosting, random forest, ridge regression, a small neural network (a nod to this repo's
   origins in `nnfor`), an ensemble of all four, or a plain 12-1 momentum rule as a non-ML baseline.
3. **Portfolio**: the top-ranked stocks are selected (existing holdings are kept while they remain within
   *N × hold buffer*, to cut trading), then weighted by the chosen optimiser. Mean-variance uses Grinold's
   rule *expected alpha = IC × volatility × score* with a Ledoit-Wolf shrunk covariance matrix, under
   long-only, max-weight and optional country caps.
4. **Trading costs** are charged on every trade (commission + spread), plus an FX fee on foreign-currency stocks.
5. Returns are measured in your base currency (SEK by default), so USD/SEK moves are included, as they would be for you.

### Why the backtest can be trusted (and where it can't)

- **Walk-forward, no look-ahead**: at date *t* the model is trained only on months whose outcome was known
  before *t*. Features use only past prices. Automated tests verify that changing future prices never changes
  today's features, and that on pure-noise data the model finds no edge.
- **Honest benchmarks**: S&P 500, MSCI World and OMXS30 (the index funds you could buy instead), a 50/50 blend,
  and an **equal-weight portfolio of the same universe**, which isolates stock-picking skill from the universe choice.
- **Significance tests**: t-test and block-bootstrap of excess returns, probabilistic Sharpe ratio, rank-IC
  t-stats, quintile spreads, and a comparison with 500 random portfolios drawn from the same universe.

**Known limitations**

- **Survivorship bias**: the universe is *today's* large caps. Companies that collapsed or were delisted are
  missing, which flatters every strategy on this universe, including the equal-weight benchmark. Judge the
  strategy against the equal-weight universe and the random portfolios, not only against the index.
- **OMXS30 (^OMX) is a price index**: it excludes dividends (≈3–4 %/yr), so it understates what an OMXS30 fund
  earns. SPY and URTH are dividend-adjusted.
- Yahoo Finance data can contain errors and gaps. Costs, taxes (ISK/KF), and minimum brokerage fees are approximations.
- **Multiple testing**: trying many settings and picking the best inflates results. Prefer settings that work
  across models and sub-periods.

*This is a research tool, not investment advice.*
