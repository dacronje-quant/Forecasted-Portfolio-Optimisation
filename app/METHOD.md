### How the strategy & monthly decision engine work

**Every month, for each stock in the universe (≈70 US + ≈45 Swedish large caps):**

1. **Data hygiene & completed month-ends**: live quotes are checked for staleness, missing FX, and extreme single-day moves (>40%, potential unadjusted splits). Completed calendar month-ends are strictly separated from live mid-month / 1st-of-month snapshots so partial months never corrupt momentum or training labels.
2. **Features**: computed strictly from past OHLCV up to date *t* using exact trading-day lookbacks (21/63/126/252 days): momentum (1/3/6/12m and 12-minus-1), **sector-relative** and **country-relative** momentum, short-term reversal, volatility (1/3/12m, downside, **Parkinson high-low range**), skewness, 6m drawdown, distance from 52-week high, trend vs 200d SMA, beta & idiosyncratic vol, consistency of positive months, **dollar-volume trend**, **Amihud illiquidity**, and country. All features are converted to cross-sectional percentile ranks.
3. **Walk-forward models**: Ridge regression, Histogram Gradient Boosting, Random Forest, a regularized Neural Network (MLP), an Ensemble of all four, and a plain 12-1 momentum baseline. Training samples use optional exponential time-decay weighting favouring more recent market regimes.
4. **Optimal Meta-Strategy (walk-forward)**: instead of hand-picking the best backtest in hindsight, the meta-engine evaluates diverse *(model × optimiser)* candidates strictly on trailing out-of-sample net returns, adjusts for multiple testing using the **Deflated Sharpe Ratio** (Bailey & López de Prado), and blends the top validated strategies walk-forward.
5. **Constrained, turnover-aware optimisation**: enforces long-only, single-stock, **country**, and **sector** caps simultaneously; uses **weekly** Ledoit-Wolf shrunk covariance (avoiding US/SE timezone close-time correlation bias); dynamically calibrates Grinold's IC from trailing OOS rank-IC; and penalises L1 turnover directly inside mean-variance (1.5× higher on foreign-currency trades).
6. **Avanza / Nordnet execution & Smart DCA**: models tiered Swedish retail brokerage fees (courtage bps + minimum fee in SEK + **25 bps FX spread** + bid/ask spread). In **Smart DCA** mode, existing holdings inside the hold band are kept without trimming and your monthly deposit (e.g. 10 000 SEK) is deployed into the most underweight top targets first.

### Why the backtest can be trusted (and where it can't)

- **Walk-forward, no look-ahead**: at date *t*, models train only on completed months before *t*, dynamic IC uses only realised past IC `< t`, and meta-strategy weights use only candidate returns realised `<= t`. Automated tests verify zero look-ahead across features, models, and meta-strategy selection.
- **Honest benchmarks**: S&P 500 (`SPY`), MSCI World (`URTH`), **total-return OMXS30 (`XACT-OMXS30.ST`)**, price-only `^OMX`, a 50/50 US+SE blend, and an **equal-weight universe** benchmark.
- **Significance & multiple-testing tests**: t-test and block-bootstrap of excess returns, Probabilistic & **Deflated Sharpe Ratios**, sub-period stability, rank-IC t-stats, quintile spreads, and 500 random portfolios.

**Known limitations**

- **Survivorship bias**: the universe is *today's* large caps. Always judge skill against the **equal-weight universe** and **random portfolios**, not only the index.
- Yahoo Finance data can occasionally have gaps or delayed split adjustments — always review the **Data Hygiene Alerts** before placing live orders.
- Swedish tax nuances (ISK schablonbeskattning vs KF foreign dividend withholding tax recovery) are not explicitly modeled in net returns.

*This is a research and decision-support tool, not licensed investment advice.*
