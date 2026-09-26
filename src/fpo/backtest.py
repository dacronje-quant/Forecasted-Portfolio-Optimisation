"""Walk-forward monthly backtest.

At every month end t:
  1. features are computed from prices up to t only;
  2. the model (trained only on labels realised before t) scores each stock;
  3. the optimiser builds target weights;
  4. trading costs are charged on turnover versus the drifted previous portfolio;
  5. the portfolio is held until the next month end.

Returns are labelled by the END of the holding month.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fpo.config import BacktestConfig
from fpo.data import MarketData, month_end_prices
from fpo.features import build_features
from fpo.models import feature_ic_table, rank_ic, walk_forward_predict
from fpo.optimizer import optimise_weights
from fpo.universe import BENCHMARKS, ticker_currency

log = logging.getLogger(__name__)

EW_UNIVERSE = "Equal-weight universe"
BLEND = "50/50 S&P 500 + OMXS30"


@dataclass
class BacktestResult:
    config: BacktestConfig
    returns: pd.Series                 # net monthly returns (after costs)
    gross_returns: pd.Series           # before costs
    benchmarks: pd.DataFrame           # monthly benchmark returns, same dates
    weights: pd.DataFrame              # target weights at each rebalance date (rows) x ticker
    turnover: pd.Series                # one-way turnover per rebalance
    costs: pd.Series                   # cost drag per rebalance (fraction of NAV)
    scores: pd.Series                  # out-of-sample model scores (date, ticker)
    ic: pd.Series                      # monthly rank IC of scores
    quantile_returns: pd.DataFrame     # mean next-month return by score quintile
    random_returns: pd.DataFrame       # monthly returns of random portfolios (frictionless)
    feature_ic: pd.DataFrame           # univariate feature diagnostics
    features: pd.DataFrame = field(repr=False, default=None)
    names: dict = field(default_factory=dict)
    base_currency: str = "SEK"
    source: str = "yahoo"

    @property
    def rebalance_dates(self) -> pd.DatetimeIndex:
        return self.weights.index

    def equity_curves(self) -> pd.DataFrame:
        df = pd.concat([self.returns.rename("Strategy"), self.benchmarks], axis=1).fillna(0.0)
        return (1 + df).cumprod()


def _period_returns(monthly: pd.DataFrame) -> pd.DataFrame:
    """Return from month end t to the next month end, indexed by t."""
    return monthly.shift(-1) / monthly - 1.0


def run_backtest(data: MarketData, cfg: BacktestConfig | None = None,
                 features: pd.DataFrame | None = None, progress=None) -> BacktestResult:
    cfg = (cfg or BacktestConfig()).validate()
    prices = data.prices
    if progress:
        progress(0.05, "Building features")
    feats = features if features is not None else build_features(prices)
    monthly = month_end_prices(prices)
    fwd = _period_returns(monthly)
    daily_ret = prices.pct_change(fill_method=None)

    feat_dates = feats.index.get_level_values("date").unique().sort_values()
    # The last month end has no following month -> can't be evaluated
    eval_dates = feat_dates[(feat_dates >= pd.Timestamp(cfg.backtest_start)) & (feat_dates < monthly.index[-1])]
    if len(eval_dates) < 3:
        raise ValueError("Backtest period too short; move backtest_start earlier or download more history")

    if progress:
        progress(0.15, f"Training {cfg.model} model walk-forward")
    scores, _ = walk_forward_predict(feats, cfg, eval_dates)
    score_dates = scores.index.get_level_values("date").unique().sort_values()

    if progress:
        progress(0.7, "Simulating portfolio")
    weights_hist, gross, net, turn, costs, period_end = {}, [], [], [], [], []
    prev_w = pd.Series(dtype=float)  # drifted weights just before rebalance
    base = data.base_currency
    for t in score_dates:
        s_t = scores.xs(t, level="date")
        hist = daily_ret.loc[:t]
        w = optimise_weights(s_t, hist, cfg, current=list(prev_w[prev_w > 0].index))
        all_t = w.index.union(prev_w.index)
        trade = (w.reindex(all_t, fill_value=0) - prev_w.reindex(all_t, fill_value=0)).abs()
        to = trade.sum() / 2 if len(prev_w) else 1.0
        if not len(prev_w):
            trade = w.copy()
        fx_trade = sum(v for k, v in trade.items() if ticker_currency(k) != base)
        cost = (trade.sum() * cfg.cost_bps + fx_trade * cfg.fx_cost_bps) / 1e4

        r = fwd.loc[t, w.index].fillna(0.0)
        g = float((w * r).sum())
        n_ = (1 + g) * (1 - cost) - 1
        prev_w = w * (1 + r) / (1 + g) if (1 + g) > 0 else w
        weights_hist[t] = w
        gross.append(g); net.append(n_); turn.append(to); costs.append(cost)
        nxt = monthly.index[monthly.index.get_loc(t) + 1]
        period_end.append(nxt)

    idx = pd.DatetimeIndex(period_end, name="date")
    returns = pd.Series(net, index=idx, name="Strategy")
    gross_returns = pd.Series(gross, index=idx, name="Strategy (gross)")
    weights = pd.DataFrame(weights_hist).T.fillna(0.0).sort_index()
    weights.index.name = "rebalance_date"

    # ---- Benchmarks (same holding periods) ----
    bm_monthly = month_end_prices(data.benchmarks)
    bm_fwd = _period_returns(bm_monthly).reindex(score_dates)
    bench = pd.DataFrame(index=score_dates)
    for tk, label in BENCHMARKS.items():
        if tk in bm_fwd.columns:
            bench[label] = bm_fwd[tk].values
    fr = feats["fwd_ret"]
    ew = fr[fr.index.get_level_values("date").isin(score_dates)].groupby(level="date").mean()
    bench[EW_UNIVERSE] = ew.reindex(score_dates).values
    if "SPY" in bm_fwd.columns and "^OMX" in bm_fwd.columns:
        bench[BLEND] = 0.5 * bm_fwd["SPY"].values + 0.5 * bm_fwd["^OMX"].values
    bench.index = idx

    # ---- Diagnostics ----
    if progress:
        progress(0.85, "Computing diagnostics")
    ic = rank_ic(scores, fr)
    ic.index = pd.DatetimeIndex(ic.index)
    q = pd.concat([scores.rename("s"), fr.rename("r")], axis=1, join="inner").dropna()
    q["quintile"] = q.groupby(level="date")["s"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
    quint = q.groupby([q.index.get_level_values("date"), "quintile"])["r"].mean().unstack()
    quint.columns = [f"Q{int(c)}" for c in quint.columns]

    rand = _random_portfolios(fr, score_dates, cfg.n_holdings, cfg.n_random_portfolios, cfg.random_state)
    rand.index = idx

    if progress:
        progress(1.0, "Done")
    return BacktestResult(
        config=cfg, returns=returns, gross_returns=gross_returns, benchmarks=bench,
        weights=weights, turnover=pd.Series(turn, index=score_dates, name="turnover"),
        costs=pd.Series(costs, index=score_dates, name="cost"), scores=scores, ic=ic,
        quantile_returns=quint, random_returns=rand, feature_ic=feature_ic_table(feats),
        features=feats, names=data.names, base_currency=data.base_currency, source=data.source,
    )


def _random_portfolios(fwd_ret: pd.Series, dates: pd.DatetimeIndex, n: int, k: int,
                       seed: int) -> pd.DataFrame:
    """Equal-weight portfolios of n stocks drawn at random from the same eligible universe."""
    rng = np.random.default_rng(seed)
    out = np.zeros((len(dates), k))
    for i, t in enumerate(dates):
        r = fwd_ret.xs(t, level="date").fillna(0.0).values
        m = len(r)
        nn = min(n, m)
        picks = np.argsort(rng.random((k, m)), axis=1)[:, :nn]
        out[i] = r[picks].mean(axis=1)
    return pd.DataFrame(out, index=dates, columns=[f"rand_{j}" for j in range(k)])
