"""Walk-forward optimal strategy selection and meta-strategy blending.

Evaluates a curated set of diverse (model, optimizer) strategies strictly
out-of-sample on trailing realised net returns, adjusts for multiple testing
via the Deflated Sharpe Ratio, and blends (or selects) the optimal strategies
at every monthly rebalance without look-ahead bias.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from fpo import metrics as M
from fpo.backtest import BacktestResult, EW_UNIVERSE, _build_benchmarks, _period_returns, _random_portfolios, run_backtest
from fpo.config import BacktestConfig
from fpo.data import MarketData, month_end_prices
from fpo.features import build_features_from_data
from fpo.models import feature_ic_table, rank_ic, walk_forward_predict
from fpo.optimizer import apply_caps, compute_regime_exposure, estimate_dynamic_ic
from fpo.universe import ticker_currency

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StrategyCandidate:
    name: str
    model: str
    optimizer: str


DEFAULT_CANDIDATES: tuple[StrategyCandidate, ...] = (
    StrategyCandidate("Momentum + HRP", "momentum", "hrp"),
    StrategyCandidate("Ridge + Mean-Var", "ridge", "mean_variance"),
    StrategyCandidate("Ridge + HRP", "ridge", "hrp"),
    StrategyCandidate("GBM + Mean-Var", "gbm", "mean_variance"),
    StrategyCandidate("GBM + Min-Var", "gbm", "min_variance"),
    StrategyCandidate("Ensemble + Mean-Var", "ensemble", "mean_variance"),
    StrategyCandidate("Ensemble + HRP", "ensemble", "hrp"),
)

# Faster subset useful for interactive previews or quick tests
FAST_CANDIDATES: tuple[StrategyCandidate, ...] = (
    StrategyCandidate("Momentum + HRP", "momentum", "hrp"),
    StrategyCandidate("Ridge + Mean-Var", "ridge", "mean_variance"),
    StrategyCandidate("Ridge + HRP", "ridge", "hrp"),
    StrategyCandidate("GBM + Mean-Var", "gbm", "mean_variance"),
    StrategyCandidate("GBM + Min-Var", "gbm", "min_variance"),
)


@dataclass
class StrategyHealth:
    status: str                 # 'GREEN', 'YELLOW', 'RED'
    headline: str
    trailing_ic: float
    trailing_sharpe: float
    deflated_sharpe: float
    current_drawdown: float
    recommended_exposure: float
    core_fallback_active: bool
    notes: list[str]


@dataclass
class MetaBacktestResult:
    meta_result: BacktestResult
    candidates: dict[str, BacktestResult]
    candidate_specs: tuple[StrategyCandidate, ...]
    leaderboard: pd.DataFrame
    strategy_weights: pd.DataFrame          # rows = rebalance_date, cols = candidate names
    current_strategy_weights: pd.Series     # latest walk-forward weights for live decision
    health: StrategyHealth
    selection_mode: str = "blend"
    lookback_months: int = 36


def _softmax(x: np.ndarray, temperature: float = 0.5) -> np.ndarray:
    z = (x - np.nanmax(x)) / max(temperature, 1e-6)
    e = np.exp(np.clip(z, -30.0, 30.0))
    s = e.sum()
    return e / s if s > 0 else np.full_like(x, 1.0 / len(x))


def compute_meta_weights_at(as_of: pd.Timestamp,
                            candidate_returns: dict[str, pd.Series],
                            ew_returns: pd.Series,
                            lookback_months: int = 36,
                            min_eval_months: int = 12,
                            selection_mode: str = "blend",
                            rf_annual: float = 0.0,
                            temperature: float = 0.5) -> pd.Series:
    """Compute leak-free strategy weights at `as_of` using ONLY returns realised on or before `as_of`.

    Because monthly returns are indexed by period END date, filtering `r.loc[:as_of]`
    strictly excludes the upcoming month starting at `as_of`.
    """
    names = list(candidate_returns)
    k = len(names)
    if k == 0:
        return pd.Series(dtype=float)
    equal = pd.Series(1.0 / k, index=names)

    scores = []
    dsrs = []
    ew_past = ew_returns.loc[:as_of].dropna().iloc[-lookback_months:]
    for name in names:
        r_past = candidate_returns[name].loc[:as_of].dropna().iloc[-lookback_months:]
        if len(r_past) < min_eval_months:
            return equal
        sr = M.sharpe(r_past, rf=rf_annual)
        ir = M.information_ratio(r_past, ew_past.reindex(r_past.index))
        cal = M.calmar(r_past)
        sr = 0.0 if np.isnan(sr) else float(np.clip(sr, -2.0, 3.0))
        ir = 0.0 if np.isnan(ir) else float(np.clip(ir, -2.0, 3.0))
        cal = 0.0 if np.isnan(cal) else float(np.clip(cal, -2.0, 3.0))
        scores.append(0.50 * sr + 0.35 * ir + 0.15 * cal)
        dsr = M.deflated_sharpe(r_past, n_trials=k, rf=rf_annual)
        dsrs.append(0.5 if np.isnan(dsr) else float(dsr))

    scores_arr = np.asarray(scores, dtype=float)
    if selection_mode == "winner":
        w = np.zeros(k)
        w[int(np.argmax(scores_arr))] = 1.0
        return pd.Series(w, index=names)

    raw_p = _softmax(scores_arr, temperature=temperature)
    # Shrink harder toward equal weight when no strategy clears DSR confidence hurdle
    max_dsr = max(dsrs) if dsrs else 0.5
    shrink = 0.25 if max_dsr >= 0.55 else 0.65
    blended = (1.0 - shrink) * raw_p + shrink * (1.0 / k)
    blended = blended / blended.sum()
    return pd.Series(blended, index=names)


def assess_strategy_health(meta_res: BacktestResult, n_trials: int,
                           recommended_exposure: float = 1.0,
                           rf_annual: float = 0.0) -> StrategyHealth:
    """Summarize live strategy health as a GREEN / YELLOW / RED traffic light."""
    r = meta_res.returns.dropna()
    ic = meta_res.ic.dropna()
    trail_ic = float(ic.iloc[-24:].mean()) if len(ic) >= 6 else float(ic.mean() if len(ic) else 0.0)
    trail_r = r.iloc[-36:] if len(r) >= 12 else r
    trail_sr = float(M.sharpe(trail_r, rf=rf_annual)) if len(trail_r) >= 6 else 0.0
    if np.isnan(trail_sr):
        trail_sr = 0.0
    dsr = float(M.deflated_sharpe(r, n_trials=max(1, n_trials), rf=rf_annual))
    if np.isnan(dsr):
        dsr = 0.5
    cur_dd = float(M.drawdown(r).iloc[-1]) if len(r) else 0.0

    notes = []
    core_fallback_active = False
    if trail_ic <= 0.0:
        notes.append(f"Trailing 24m rank-IC is weak ({trail_ic:+.3f}); scaling back active conviction.")
        core_fallback_active = True
    elif trail_ic < 0.015:
        notes.append(f"Trailing 24m rank-IC is modest ({trail_ic:+.3f}); defensive blending advised.")
    else:
        notes.append(f"Trailing 24m rank-IC is healthy ({trail_ic:+.3f}).")

    if cur_dd <= -0.25:
        notes.append(f"Strategy is in a deep drawdown ({cur_dd:.1%}).")
    elif cur_dd <= -0.15:
        notes.append(f"Strategy is in a moderate drawdown ({cur_dd:.1%}).")

    if dsr < 0.50:
        notes.append(f"Deflated Sharpe Ratio ({dsr:.0%}) is below 50% after adjusting for {n_trials} strategies tested.")
    else:
        notes.append(f"Deflated Sharpe Ratio ({dsr:.0%}) passes multiple-testing adjustment ({n_trials} candidates).")

    if recommended_exposure < 0.99:
        notes.append(f"Regime overlay recommends {recommended_exposure:.0%} equity exposure ({1 - recommended_exposure:.0%} defensive/cash).")

    if trail_ic <= 0.0 or cur_dd <= -0.25 or trail_sr < 0.0:
        status = "RED"
        headline = "Cautious — weak trailing signal or deep drawdown; lean defensive / core ETF."
    elif trail_ic < 0.015 or cur_dd <= -0.15 or dsr < 0.55 or trail_sr < 0.35:
        status = "YELLOW"
        headline = "Moderate conviction — stick to Smart DCA / low turnover and respect caps."
    else:
        status = "GREEN"
        headline = "Healthy OOS edge — proceed with optimal meta-strategy allocation."

    return StrategyHealth(
        status=status, headline=headline, trailing_ic=trail_ic, trailing_sharpe=trail_sr,
        deflated_sharpe=dsr, current_drawdown=cur_dd, recommended_exposure=recommended_exposure,
        core_fallback_active=core_fallback_active, notes=notes,
    )


def run_meta_backtest(data: MarketData, base_cfg: BacktestConfig | None = None,
                      candidates: tuple[StrategyCandidate, ...] = DEFAULT_CANDIDATES,
                      features: pd.DataFrame | None = None,
                      lookback_months: int = 36,
                      min_eval_months: int = 12,
                      selection_mode: str = "blend",
                      progress=None) -> MetaBacktestResult:
    """Run all candidate strategies and simulate a leak-free walk-forward meta-strategy."""
    base_cfg = (base_cfg or BacktestConfig()).validate()
    if selection_mode not in ("blend", "winner"):
        raise ValueError("selection_mode must be 'blend' or 'winner'")

    if progress:
        progress(0.05, "Building shared features")
    feats = features if features is not None else build_features_from_data(data)
    monthly = month_end_prices(data.prices, completed_only=True)
    fwd = _period_returns(monthly)
    feat_dates = feats.index.get_level_values("date").unique().sort_values()
    eval_dates = feat_dates[(feat_dates >= pd.Timestamp(base_cfg.backtest_start)) & (feat_dates < monthly.index[-1])]
    if len(eval_dates) < 3:
        raise ValueError("Backtest period too short for meta-strategy evaluation")

    # 1. Train each unique forecasting model ONCE walk-forward
    unique_models = list(dict.fromkeys(c.model for c in candidates))
    model_scores: dict[str, pd.Series] = {}
    for idx_m, m_name in enumerate(unique_models):
        if progress:
            progress(0.10 + 0.45 * (idx_m / max(1, len(unique_models))), f"Training {m_name} walk-forward")
        m_cfg = replace(base_cfg, model=m_name)
        sc, _ = walk_forward_predict(feats, m_cfg, eval_dates)
        model_scores[m_name] = sc

    # 2. Simulate every (model, optimizer) candidate using cached model scores
    cand_results: dict[str, BacktestResult] = {}
    for idx_c, cand in enumerate(candidates):
        if progress:
            progress(0.55 + 0.25 * (idx_c / max(1, len(candidates))), f"Backtesting {cand.name}")
        c_cfg = replace(base_cfg, model=cand.model, optimizer=cand.optimizer,
                        n_random_portfolios=min(base_cfg.n_random_portfolios, 100))
        cand_results[cand.name] = run_backtest(
            data, c_cfg, features=feats, precomputed_scores=model_scores[cand.model]
        )

    # Common rebalance dates across all candidates
    common_dates = None
    for res in cand_results.values():
        d = res.rebalance_dates
        common_dates = d if common_dates is None else common_dates.intersection(d)
    common_dates = common_dates.sort_values()
    if len(common_dates) < 3:
        raise ValueError("Not enough overlapping rebalance dates across candidates")

    # 3. Walk-forward meta-strategy simulation (strictly leak-free)
    if progress:
        progress(0.82, "Simulating walk-forward meta-strategy")
    first_cand = next(iter(cand_results.values()))
    ew_series = first_cand.benchmarks[EW_UNIVERSE]
    cand_ret_map = {name: res.returns for name, res in cand_results.items()}

    fee_sched = base_cfg.fee_schedule()
    base = data.base_currency
    rf_monthly = base_cfg.rf_annual / 12.0

    meta_weights_hist: dict[pd.Timestamp, pd.Series] = {}
    strat_w_hist: dict[pd.Timestamp, pd.Series] = {}
    meta_scores_parts: list[pd.Series] = []
    gross, net, turn, costs, period_end, exposures, eff_ics = [], [], [], [], [], [], []
    prev_w = pd.Series(dtype=float)
    prev_exp = 0.0

    for t in common_dates:
        sw = compute_meta_weights_at(
            t, cand_ret_map, ew_series,
            lookback_months=lookback_months, min_eval_months=min_eval_months,
            selection_mode=selection_mode, rf_annual=base_cfg.rf_annual,
        )
        strat_w_hist[t] = sw

        # Blend candidate equity weights and candidate z-scored alpha signals
        blended_w = pd.Series(dtype=float)
        blended_score_parts = []
        for cand in candidates:
            pw = float(sw[cand.name])
            if pw <= 1e-9:
                continue
            cw = cand_results[cand.name].weights.loc[t]
            cw = cw[cw > 1e-9]
            blended_w = blended_w.add(cw * pw, fill_value=0.0)
            s_t = model_scores[cand.model].xs(t, level="date")
            z_t = (s_t - s_t.mean()) / (s_t.std() + 1e-12)
            blended_score_parts.append(z_t * pw)

        # Enforce top-n_holdings & caps cleanly on the blended portfolio
        blended_w = blended_w[blended_w > 1e-6].sort_values(ascending=False)
        if len(blended_w) > base_cfg.n_holdings:
            blended_w = blended_w.iloc[:base_cfg.n_holdings]
        w_arr = apply_caps(
            blended_w.values, list(blended_w.index),
            base_cfg.max_weight, base_cfg.max_country_weight, base_cfg.max_sector_weight,
        )
        w = pd.Series(w_arr, index=blended_w.index)
        if base_cfg.min_weight > 0 and (w < base_cfg.min_weight).any() and (w >= base_cfg.min_weight).sum() >= 1:
            w = w[w >= base_cfg.min_weight]
            w = pd.Series(
                apply_caps(w.values, list(w.index), base_cfg.max_weight,
                           base_cfg.max_country_weight, base_cfg.max_sector_weight),
                index=w.index,
            )
        w = w / w.sum()

        # Combined z-score for IC diagnostics
        comb_s = sum(blended_score_parts)
        comb_s.index = pd.MultiIndex.from_product([[t], comb_s.index], names=["date", "ticker"])
        meta_scores_parts.append(comb_s)

        exp_t = compute_regime_exposure(data.benchmarks, t, target_vol=base_cfg.target_vol) if base_cfg.regime_overlay else 1.0
        exposures.append(exp_t)

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

        meta_weights_hist[t] = w
        gross.append(g); net.append(n_); turn.append(to); costs.append(cost)
        nxt = monthly.index[monthly.index.get_loc(t) + 1]
        period_end.append(nxt)

    idx = pd.DatetimeIndex(period_end, name="date")
    meta_returns = pd.Series(net, index=idx, name="Strategy")
    meta_gross = pd.Series(gross, index=idx, name="Strategy (gross)")
    meta_weights = pd.DataFrame(meta_weights_hist).T.fillna(0.0).sort_index()
    meta_weights.index.name = "rebalance_date"
    strat_w_df = pd.DataFrame(strat_w_hist).T.sort_index()
    strat_w_df.index.name = "rebalance_date"

    meta_scores = pd.concat(meta_scores_parts).sort_index()
    fr = feats["fwd_ret"]
    ic = rank_ic(meta_scores, fr)
    ic.index = pd.DatetimeIndex(ic.index)

    for t in common_dates:
        past_ic = ic[ic.index < t]
        eff_ics.append(estimate_dynamic_ic(past_ic, prior=base_cfg.ic_prior, lookback=base_cfg.ic_lookback_months))

    q = pd.concat([meta_scores.rename("s"), fr.rename("r")], axis=1, join="inner").dropna()
    q["quintile"] = q.groupby(level="date")["s"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
    quint = q.groupby([q.index.get_level_values("date"), "quintile"])["r"].mean().unstack()
    quint.columns = [f"Q{int(c)}" for c in quint.columns]

    bench = _build_benchmarks(data, feats, common_dates, idx)
    rand = _random_portfolios(fr, common_dates, base_cfg.n_holdings,
                              base_cfg.n_random_portfolios, base_cfg.random_state)
    rand.index = idx

    meta_res = BacktestResult(
        config=base_cfg, returns=meta_returns, gross_returns=meta_gross, benchmarks=bench,
        weights=meta_weights, turnover=pd.Series(turn, index=common_dates, name="turnover"),
        costs=pd.Series(costs, index=common_dates, name="cost"), scores=meta_scores, ic=ic,
        quantile_returns=quint, random_returns=rand, feature_ic=feature_ic_table(feats),
        features=feats, names=data.names, base_currency=data.base_currency, source=data.source,
        exposure=pd.Series(exposures, index=common_dates, name="exposure"),
        effective_ic=pd.Series(eff_ics, index=common_dates, name="effective_ic"),
    )

    # Latest strategy weights using ALL realised returns up through the latest completed month
    latest_obs = meta_returns.index.max()
    current_sw = compute_meta_weights_at(
        latest_obs, cand_ret_map, ew_series,
        lookback_months=lookback_months, min_eval_months=min_eval_months,
        selection_mode=selection_mode, rf_annual=base_cfg.rf_annual,
    )

    # Build candidate leaderboard with Deflated Sharpe Ratio
    n_trials = len(candidates)
    ew_bench = bench[EW_UNIVERSE]
    lb_rows = []
    for label, res_obj, m_col, o_col in (
        [("Meta-Strategy (Walk-Forward)", meta_res, f"meta ({selection_mode})", "blended")]
        + [(c.name, cand_results[c.name], c.model, c.optimizer) for c in candidates]
    ):
        r_aligned = res_obj.returns.reindex(idx).dropna()
        s = M.summary(r_aligned, ew_bench.reindex(r_aligned.index), rf=base_cfg.rf_annual)
        dsr = M.deflated_sharpe(r_aligned, n_trials=n_trials, rf=base_cfg.rf_annual)
        lb_rows.append({
            "Strategy": label,
            "Model": m_col,
            "Optimizer": o_col,
            "CAGR": s["CAGR"],
            "Volatility": s["Volatility"],
            "Sharpe": s["Sharpe"],
            "Deflated Sharpe": dsr,
            "Max drawdown": s["Max drawdown"],
            "Excess CAGR (vs EW)": s["Excess CAGR"],
            "Info ratio": s["Information ratio"],
            "Mean IC": float(res_obj.ic.mean()) if len(res_obj.ic) else np.nan,
            "Turnover": float(res_obj.turnover.iloc[1:].mean()) if len(res_obj.turnover) > 1 else float(res_obj.turnover.mean()),
            "Annual cost": float(res_obj.costs.iloc[1:].mean() * 12) if len(res_obj.costs) > 1 else 0.0,
            "Current weight": float(current_sw.get(label, np.nan)),
        })
    leaderboard = pd.DataFrame(lb_rows).set_index("Strategy")

    live_exp = (
        compute_regime_exposure(data.benchmarks, pd.Timestamp(data.prices.index.max()), target_vol=base_cfg.target_vol)
        if base_cfg.regime_overlay else 1.0
    )
    health = assess_strategy_health(meta_res, n_trials=n_trials, recommended_exposure=live_exp,
                                    rf_annual=base_cfg.rf_annual)

    if progress:
        progress(1.0, "Done")
    return MetaBacktestResult(
        meta_result=meta_res, candidates=cand_results, candidate_specs=candidates,
        leaderboard=leaderboard, strategy_weights=strat_w_df,
        current_strategy_weights=current_sw, health=health,
        selection_mode=selection_mode, lookback_months=lookback_months,
    )
