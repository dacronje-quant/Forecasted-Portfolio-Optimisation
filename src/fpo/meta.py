"""Walk-forward optimal strategy selection and meta-strategy blending.

Evaluates a curated, fast modern set of (model, optimizer) strategies strictly
out-of-sample on trailing realised net returns, adjusts for multiple testing
via the Deflated Sharpe Ratio, and blends (or selects) the optimal strategies
at every monthly rebalance without look-ahead bias.

Performance optimizations:
- Zero-redundancy ensemble scoring (reuses member walk-forward z-scores directly).
- Precomputes weekly returns, feature IC, and random portfolios ONCE across all candidates.
- Optional on-disk result cache keyed by config + latest completed month-end.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from fpo import metrics as M
from fpo.backtest import (
    BacktestResult, EW_UNIVERSE, _build_benchmarks, _period_returns, _random_portfolios, run_backtest,
)
from fpo.config import BacktestConfig
from fpo.data import DEFAULT_CACHE_DIR, MarketData, month_end_prices
from fpo.features import build_features_from_data
from fpo.models import EnsembleModel, feature_ic_table, rank_ic, walk_forward_predict
from fpo.optimizer import (
    apply_caps, compute_regime_exposure, estimate_dynamic_ic, precompute_weekly_returns,
)
from fpo.universe import ticker_currency

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StrategyCandidate:
    name: str
    model: str
    optimizer: str


# Fast, modern low-turnover lineup (Smooth Residual Mom, LightGBM XENDCG, Ridge, HERC, NCO, HRP)
DEFAULT_CANDIDATES: tuple[StrategyCandidate, ...] = (
    StrategyCandidate("Smooth Residual Mom + HERC", "smooth_mom", "herc"),
    StrategyCandidate("Smooth Residual Mom + NCO", "smooth_mom", "nco"),
    StrategyCandidate("Momentum + HRP", "momentum", "hrp"),
    StrategyCandidate("LightGBM XENDCG + HERC", "lgbm_rank", "herc"),
    StrategyCandidate("LightGBM XENDCG + NCO", "lgbm_rank", "nco"),
    StrategyCandidate("Ridge + HERC", "ridge", "herc"),
    StrategyCandidate("Modern Ensemble + NCO", "ensemble", "nco"),
)

# Ultra-fast 5-strategy subset (~3-5s total cold run!)
FAST_CANDIDATES: tuple[StrategyCandidate, ...] = (
    StrategyCandidate("Smooth Residual Mom + HERC", "smooth_mom", "herc"),
    StrategyCandidate("Momentum + HRP", "momentum", "hrp"),
    StrategyCandidate("LightGBM XENDCG + HERC", "lgbm_rank", "herc"),
    StrategyCandidate("LightGBM XENDCG + NCO", "lgbm_rank", "nco"),
    StrategyCandidate("Ridge + HERC", "ridge", "herc"),
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


def _zscore_by_date(s: pd.Series) -> pd.Series:
    g = s.groupby(level="date")
    return (s - g.transform("mean")) / (g.transform("std").fillna(1.0) + 1e-12)


def compute_meta_weights_at(as_of: pd.Timestamp,
                            candidate_returns: dict[str, pd.Series],
                            ew_returns: pd.Series,
                            lookback_months: int = 36,
                            min_eval_months: int = 12,
                            selection_mode: str = "blend",
                            rf_annual: float = 0.0,
                            temperature: float = 0.5) -> pd.Series:
    """Compute leak-free strategy weights at `as_of` using ONLY returns realised on or before `as_of`."""
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


def _meta_cache_path(cache_dir: Path, data: MarketData, base_cfg: BacktestConfig,
                     candidates: tuple[StrategyCandidate, ...], last_month: pd.Timestamp,
                     lookback_months: int, selection_mode: str) -> Path:
    cfg_d = base_cfg.to_dict()
    cfg_d.pop("extra", None)
    payload = {
        "cfg": cfg_d,
        "cands": [(c.name, c.model, c.optimizer) for c in candidates],
        "last_month": str(last_month.date()),
        "source": data.source,
        "n_tickers": len(data.tickers),
        "lookback": lookback_months,
        "sel_mode": selection_mode,
    }
    h = hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:14]
    return cache_dir / f"meta_{h}.pkl"


def run_meta_backtest(data: MarketData, base_cfg: BacktestConfig | None = None,
                      candidates: tuple[StrategyCandidate, ...] = DEFAULT_CANDIDATES,
                      features: pd.DataFrame | None = None,
                      lookback_months: int = 36,
                      min_eval_months: int = 12,
                      selection_mode: str = "blend",
                      cache_dir: Path | str | None = None,
                      progress=None) -> MetaBacktestResult:
    """Run all candidate strategies and simulate a leak-free walk-forward meta-strategy."""
    base_cfg = (base_cfg or BacktestConfig()).validate()
    if selection_mode not in ("blend", "winner"):
        raise ValueError("selection_mode must be 'blend' or 'winner'")

    monthly = month_end_prices(data.prices, completed_only=True)
    if len(monthly) < 4:
        raise ValueError("Not enough completed months for meta-strategy evaluation")

    # Fast disk cache hit when evaluating live data on an unchanged completed month-end
    cpath = None
    if cache_dir is not None and data.source == "yahoo":
        cdir = Path(cache_dir)
        cdir.mkdir(parents=True, exist_ok=True)
        cpath = _meta_cache_path(cdir, data, base_cfg, candidates, monthly.index[-1],
                                 lookback_months, selection_mode)
        if cpath.exists():
            try:
                log.info("Loading cached meta-backtest %s", cpath.name)
                cached_res: MetaBacktestResult = pd.read_pickle(cpath)
                if progress:
                    progress(1.0, "Loaded from cache")
                return cached_res
            except Exception:
                pass

    if progress:
        progress(0.05, "Building shared features & weekly returns")
    feats = features if features is not None else build_features_from_data(data)
    fwd = _period_returns(monthly)
    daily_ret = data.prices.pct_change(fill_method=None)
    wk_ret = precompute_weekly_returns(daily_ret) if base_cfg.cov_freq == "weekly" else None
    shared_feat_ic = feature_ic_table(feats)

    feat_dates = feats.index.get_level_values("date").unique().sort_values()
    eval_dates = feat_dates[(feat_dates >= pd.Timestamp(base_cfg.backtest_start)) & (feat_dates < monthly.index[-1])]
    if len(eval_dates) < 3:
        raise ValueError("Backtest period too short for meta-strategy evaluation")

    # 1. Train each unique base model ONCE; compose 'ensemble' with ZERO redundant training!
    requested_models = list(dict.fromkeys(c.model for c in candidates))
    need_ensemble = "ensemble" in requested_models
    base_models = [m for m in requested_models if m != "ensemble"]
    if need_ensemble:
        for member in EnsembleModel.MEMBER_NAMES:
            if member not in base_models:
                base_models.append(member)

    model_scores: dict[str, pd.Series] = {}
    for idx_m, m_name in enumerate(base_models):
        if progress:
            progress(0.10 + 0.40 * (idx_m / max(1, len(base_models))), f"Training {m_name} walk-forward")
        # Use semi-annual refit (>= 6m) for tree models inside meta-search unless user set longer
        retrain = max(base_cfg.retrain_every, 6) if m_name in ("gbm", "lgbm_rank", "random_forest") else base_cfg.retrain_every
        m_cfg = replace(base_cfg, model=m_name, retrain_every=retrain)
        sc, _ = walk_forward_predict(feats, m_cfg, eval_dates)
        model_scores[m_name] = sc

    if need_ensemble:
        member_z = [_zscore_by_date(model_scores[m]) for m in EnsembleModel.MEMBER_NAMES]
        common_idx = member_z[0].index
        for mz in member_z[1:]:
            common_idx = common_idx.intersection(mz.index)
        ens_s = sum(mz.reindex(common_idx) for mz in member_z) / float(len(member_z))
        model_scores["ensemble"] = ens_s.rename("score").sort_index()

    # Precompute shared random portfolios once for fast candidate diagnostics
    fr = feats["fwd_ret"]
    first_scores = model_scores[candidates[0].model]
    first_dates = first_scores.index.get_level_values("date").unique().sort_values()
    shared_rand = _random_portfolios(fr, first_dates, base_cfg.n_holdings,
                                     min(base_cfg.n_random_portfolios, 100), base_cfg.random_state)

    # 2. Simulate every (model, optimizer) candidate using cached model scores & weekly returns
    cand_results: dict[str, BacktestResult] = {}
    for idx_c, cand in enumerate(candidates):
        if progress:
            progress(0.52 + 0.30 * (idx_c / max(1, len(candidates))), f"Backtesting {cand.name}")
        c_cfg = replace(base_cfg, model=cand.model, optimizer=cand.optimizer,
                        n_random_portfolios=min(base_cfg.n_random_portfolios, 100))
        cand_results[cand.name] = run_backtest(
            data, c_cfg, features=feats, precomputed_scores=model_scores[cand.model],
            weekly_returns=wk_ret, precomputed_feature_ic=shared_feat_ic,
            precomputed_random=shared_rand,
        )

    common_dates = None
    for res in cand_results.values():
        d = res.rebalance_dates
        common_dates = d if common_dates is None else common_dates.intersection(d)
    common_dates = common_dates.sort_values()
    if len(common_dates) < 3:
        raise ValueError("Not enough overlapping rebalance dates across candidates")

    # 3. Walk-forward meta-strategy simulation (strictly leak-free)
    if progress:
        progress(0.84, "Simulating walk-forward meta-strategy")
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
        quantile_returns=quint, random_returns=rand, feature_ic=shared_feat_ic,
        features=feats, names=data.names, base_currency=data.base_currency, source=data.source,
        exposure=pd.Series(exposures, index=common_dates, name="exposure"),
        effective_ic=pd.Series(eff_ics, index=common_dates, name="effective_ic"),
    )

    latest_obs = meta_returns.index.max()
    current_sw = compute_meta_weights_at(
        latest_obs, cand_ret_map, ew_series,
        lookback_months=lookback_months, min_eval_months=min_eval_months,
        selection_mode=selection_mode, rf_annual=base_cfg.rf_annual,
    )

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

    out = MetaBacktestResult(
        meta_result=meta_res, candidates=cand_results, candidate_specs=candidates,
        leaderboard=leaderboard, strategy_weights=strat_w_df,
        current_strategy_weights=current_sw, health=health,
        selection_mode=selection_mode, lookback_months=lookback_months,
    )
    if cpath is not None:
        try:
            pd.to_pickle(out, cpath)
        except Exception:
            pass
    if progress:
        progress(1.0, "Done")
    return out
