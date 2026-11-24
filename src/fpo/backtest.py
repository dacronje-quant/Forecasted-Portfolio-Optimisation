"""Walk-forward monthly backtest.

At every completed month end t:
  1. features are computed from prices/OHLCV up to t only;
  2. the model (trained only on labels realised before t) scores each stock;
  3. dynamic IC is calibrated strictly from past realised OOS rank-IC (< t);
  4. the optimiser builds constraint-respecting, turnover-aware target weights;
  5. optional regime / defensive overlay scales equity exposure;
  6. trading costs (broker commission + FX + spread) are charged on turnover;
  7. the portfolio is held until the next completed month end.

Returns are labelled by the END of the holding month.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fpo.config import BacktestConfig
from fpo.data import MarketData, month_end_prices
from fpo.features import build_features_from_data
from fpo.models import feature_ic_table, rank_ic, walk_forward_predict
from fpo.optimizer import (
    apply_caps, compute_regime_exposure, estimate_dynamic_ic, optimise_weights,
)
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
    weights: pd.DataFrame              # target equity weights at each rebalance date (rows) x ticker
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
    exposure: pd.Series | None = None          # equity exposure fraction (1.0 when regime_overlay=False)
    effective_ic: pd.Series | None = None      # trailing calibrated IC used at each rebalance

    @property
    def rebalance_dates(self) -> pd.DatetimeIndex:
        return self.weights.index

    def equity_curves(self) -> pd.DataFrame:
        df = pd.concat([self.returns.rename("Strategy"), self.benchmarks], axis=1).fillna(0.0)
        return (1 + df).cumprod()


def _period_returns(monthly: pd.DataFrame) -> pd.DataFrame:
    """Return from completed month end t to the next completed month end, indexed by t."""
    return monthly.shift(-1) / monthly - 1.0


def _build_benchmarks(data: MarketData, feats: pd.DataFrame, score_dates: pd.DatetimeIndex,
                      period_index: pd.DatetimeIndex) -> pd.DataFrame:
    bm_monthly = month_end_prices(data.benchmarks, completed_only=True)
    bm_fwd = _period_returns(bm_monthly).reindex(score_dates)
    bench = pd.DataFrame(index=score_dates)
    for tk, label in BENCHMARKS.items():
        if tk in bm_fwd.columns and bm_fwd[tk].notna().any():
            bench[label] = bm_fwd[tk].values
    fr = feats["fwd_ret"]
    ew = fr[fr.index.get_level_values("date").isin(score_dates)].groupby(level="date").mean()
    bench[EW_UNIVERSE] = ew.reindex(score_dates).values
    # Prefer total-return XACT OMXS30 over price-only ^OMX for the 50/50 blend when available
    se_col = "XACT-OMXS30.ST" if ("XACT-OMXS30.ST" in bm_fwd.columns and bm_fwd["XACT-OMXS30.ST"].notna().all()) else "^OMX"
    if "SPY" in bm_fwd.columns and se_col in bm_fwd.columns:
        bench[BLEND] = 0.5 * bm_fwd["SPY"].values + 0.5 * bm_fwd[se_col].values
    bench.index = period_index
    return bench


def run_backtest(data: MarketData, cfg: BacktestConfig | None = None,
                 features: pd.DataFrame | None = None,
                 precomputed_scores: pd.Series | None = None,
                 progress=None) -> BacktestResult:
    cfg = (cfg or BacktestConfig()).validate()
    prices = data.prices
    if progress:
        progress(0.05, "Building features")
    feats = features if features is not None else build_features_from_data(data)
    monthly = month_end_prices(prices, completed_only=True)
    fwd = _period_returns(monthly)
    daily_ret = prices.pct_change(fill_method=None)

    feat_dates = feats.index.get_level_values("date").unique().sort_values()
    # Strictly completed month ends that have a subsequent completed month end
    eval_dates = feat_dates[(feat_dates >= pd.Timestamp(cfg.backtest_start)) & (feat_dates < monthly.index[-1])]
    if len(eval_dates) < 3:
        raise ValueError("Backtest period too short; move backtest_start earlier or download more history")

    if precomputed_scores is not None:
        scores = precomputed_scores[precomputed_scores.index.get_level_values("date").isin(eval_dates)].sort_index()
    else:
        if progress:
            progress(0.15, f"Training {cfg.model} model walk-forward")
        scores, _ = walk_forward_predict(feats, cfg, eval_dates)
    score_dates = scores.index.get_level_values("date").unique().sort_values()

    # Precompute per-date realised rank IC so we can query strictly past (< t) IC leak-free
    fr = feats["fwd_ret"]
    ic_full = rank_ic(scores, fr)
    ic_full.index = pd.DatetimeIndex(ic_full.index)

    if progress:
        progress(0.7, "Simulating portfolio")
    fee_sched = cfg.fee_schedule()
    weights_hist, gross, net, turn, costs, period_end = {}, [], [], [], [], []
    exposures, eff_ics = [], []
    prev_w = pd.Series(dtype=float)  # drifted equity weights just before rebalance
    prev_exp = 0.0                   # previous equity exposure
    base = data.base_currency
    rf_monthly = cfg.rf_annual / 12.0

    for t in score_dates:
        s_t = scores.xs(t, level="date")
        hist = daily_ret.loc[:t]

        # Strictly past realised OOS IC (date s < t means s's forward return ended at or before t)
        past_ic = ic_full[ic_full.index < t]
        if cfg.use_dynamic_ic:
            eff_ic = estimate_dynamic_ic(past_ic, prior=cfg.ic_prior, lookback=cfg.ic_lookback_months)
        else:
            eff_ic = cfg.assumed_ic
        eff_ics.append(eff_ic)

        w = optimise_weights(
            s_t, hist, cfg,
            current=list(prev_w[prev_w > 0].index) if len(prev_w) else None,
            prev_weights=prev_w if len(prev_w) else None,
            effective_ic=eff_ic,
        )

        # Core/defensive fallback when trailing OOS IC is weak (after enough history)
        if cfg.core_fallback and len(past_ic) >= 18 and eff_ic < 0.015 and len(w) > 1:
            conf = float(np.clip(eff_ic / 0.015, 0.0, 1.0))
            vol = hist[w.index].iloc[-cfg.cov_lookback_days:].std().fillna(0.02).values * np.sqrt(252.0)
            w_def = apply_caps(1.0 / np.maximum(vol, 1e-4), list(w.index),
                               cfg.max_weight, cfg.max_country_weight, cfg.max_sector_weight)
            w_blend = conf * w.values + (1.0 - conf) * w_def
            w = pd.Series(
                apply_caps(w_blend, list(w.index), cfg.max_weight, cfg.max_country_weight, cfg.max_sector_weight),
                index=w.index,
            )

        # Optional volatility-targeting + trend regime overlay
        exp_t = compute_regime_exposure(data.benchmarks, t, target_vol=cfg.target_vol) if cfg.regime_overlay else 1.0
        exposures.append(exp_t)

        # Effective portfolio weights including exposure scaling for accurate turnover & costs
        w_eff = w * exp_t
        prev_eff = prev_w * prev_exp if len(prev_w) else pd.Series(dtype=float)
        all_t = w_eff.index.union(prev_eff.index)
        trade = (w_eff.reindex(all_t, fill_value=0.0) - prev_eff.reindex(all_t, fill_value=0.0)).abs()
        to = float(trade.sum() / 2.0) if len(prev_eff) else float(exp_t)
        if not len(prev_eff):
            trade = w_eff.copy()

        cost = sum(float(v) * fee_sched.marginal_rate(ticker_currency(k), base) for k, v in trade.items())

        r = fwd.loc[t, w.index].fillna(0.0)
        eq_ret = float((w * r).sum())
        g = exp_t * eq_ret + (1.0 - exp_t) * rf_monthly
        n_ = (1.0 + g) * (1.0 - cost) - 1.0
        prev_w = w * (1.0 + r) / (1.0 + eq_ret) if (1.0 + eq_ret) > 0 else w
        prev_exp = exp_t * (1.0 + eq_ret) / (1.0 + g) if (1.0 + g) > 0 else exp_t

        weights_hist[t] = w
        gross.append(g); net.append(n_); turn.append(to); costs.append(cost)
        nxt = monthly.index[monthly.index.get_loc(t) + 1]
        period_end.append(nxt)

    idx = pd.DatetimeIndex(period_end, name="date")
    returns = pd.Series(net, index=idx, name="Strategy")
    gross_returns = pd.Series(gross, index=idx, name="Strategy (gross)")
    weights = pd.DataFrame(weights_hist).T.fillna(0.0).sort_index()
    weights.index.name = "rebalance_date"

    bench = _build_benchmarks(data, feats, score_dates, idx)

    if progress:
        progress(0.85, "Computing diagnostics")
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
        costs=pd.Series(costs, index=score_dates, name="cost"), scores=scores, ic=ic_full,
        quantile_returns=quint, random_returns=rand, feature_ic=feature_ic_table(feats),
        features=feats, names=data.names, base_currency=data.base_currency, source=data.source,
        exposure=pd.Series(exposures, index=score_dates, name="exposure"),
        effective_ic=pd.Series(eff_ics, index=score_dates, name="effective_ic"),
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
