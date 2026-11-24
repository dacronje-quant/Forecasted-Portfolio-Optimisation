"""Multi-asset tactical allocation engine for Nordnet/Avanza UCITS ETFs & 0%-fee Swedish mutual funds.

Implements three modern, low-turnover multi-asset strategies strictly leak-free:
  1. DAA  — Defensive Asset Allocation (Keller & Keuning canary crash protection + HERC)
  2. Regime-NCO — Causal persistent 3-state regime switching + Nested Clustered Optimization
  3. All-Weather HERC — Multi-asset Hierarchical Equal Risk Contribution across equities, gold, bonds & cash

All signals are computed strictly from past prices <= t and evaluated net of realistic
Nordnet/Avanza costs (0% courtage/FX on Swedish mutual fund proxies, full broker+FX fees on
exchange-traded ETCs like Xetra-Gold, plus ongoing TER).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fpo import metrics as M
from fpo.backtest import _period_returns
from fpo.config import BacktestConfig
from fpo.data import MarketData, month_end_prices
from fpo.meta import compute_meta_weights_at
from fpo.optimizer import (
    apply_caps, herc_weights, nco_weights, precompute_weekly_returns, shrunk_covariance,
)
from fpo.universe import ETF_UNIVERSE, etf_fund_proxy, ticker_currency

TACTICAL_STRATEGIES = ("daa", "regime_nco", "all_weather_herc")

TACTICAL_LABELS = {
    "daa": "Defensive Asset Allocation (DAA + HERC)",
    "regime_nco": "Causal Regime-Switching NCO",
    "all_weather_herc": "All-Weather Multi-Asset HERC",
}


@dataclass
class TacticalStrategyResult:
    name: str
    returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    regimes: pd.Series | None = None


@dataclass
class TacticalMetaResult:
    meta_returns: pd.Series
    meta_weights: pd.DataFrame
    meta_turnover: pd.Series
    meta_costs: pd.Series
    benchmarks: pd.DataFrame
    strategies: dict[str, TacticalStrategyResult]
    leaderboard: pd.DataFrame
    strategy_weights: pd.DataFrame
    current_strategy_weights: pd.Series
    live_as_of: pd.Timestamp
    live_regime: str               # 'BULL', 'SIDEWAYS', 'BEAR'
    live_canary_bad: int
    live_table: pd.DataFrame       # Ticker, Name, Nordnet/Avanza Fund Proxy, Asset Class, Weight, Amount_SEK
    execution_mode: str = "mutual_fund"
    base_currency: str = "SEK"
    notes: list[str] = field(default_factory=list)


def _momentum_13612w(prices_up_to_t: pd.DataFrame) -> pd.Series:
    """Keller & Keuning 13612W weighted momentum using exact trading-day lookbacks (strictly <= t)."""
    n = len(prices_up_to_t)
    if n < 253:
        return pd.Series(dtype=float)
    p0 = prices_up_to_t.iloc[-1]
    r1 = p0 / prices_up_to_t.iloc[-22] - 1.0
    r3 = p0 / prices_up_to_t.iloc[-64] - 1.0
    r6 = p0 / prices_up_to_t.iloc[-127] - 1.0
    r12 = p0 / prices_up_to_t.iloc[-253] - 1.0
    score = (12.0 * r1 + 4.0 * r3 + 2.0 * r6 + 1.0 * r12) / 19.0
    return score.dropna()


def _detect_causal_regime(prices_up_to_t: pd.DataFrame, prev_regime: str = "BULL") -> str:
    """Strictly causal 3-state regime detector with persistence hysteresis (Bull / Sideways / Bear)."""
    # Use global equity proxy if available, else average of offensive equities
    ref_col = next((c for c in ("EUNL.DE", "SXR8.DE", "XACT-SVERIGE.ST") if c in prices_up_to_t.columns and prices_up_to_t[c].dropna().shape[0] >= 200), None)
    if ref_col is None:
        return "BULL"
    px = prices_up_to_t[ref_col].dropna()
    if len(px) < 200:
        return "BULL"
    rets = px.pct_change(fill_method=None).dropna()
    vol63 = float(rets.iloc[-63:].std() * np.sqrt(252.0)) if len(rets) >= 63 else 0.14
    sma200 = float(px.iloc[-200:].mean())
    trend = float(px.iloc[-1] / sma200 - 1.0)
    peak252 = float(px.iloc[-min(252, len(px)):].max())
    dd = float(px.iloc[-1] / peak252 - 1.0)

    # Persistence hysteresis prevents noisy monthly whipsawing
    bear_enter = (trend < -0.03) or (dd <= -0.14) or (vol63 >= 0.24)
    bear_exit = (trend > 0.01) and (dd > -0.09) and (vol63 < 0.20)
    bull_enter = (trend > 0.02) and (dd > -0.07) and (vol63 < 0.17)
    bull_exit = (trend < -0.005) or (dd <= -0.10) or (vol63 >= 0.20)

    if prev_regime == "BEAR":
        if bear_exit:
            return "BULL" if bull_enter else "SIDEWAYS"
        return "BEAR"
    if prev_regime == "BULL":
        if bear_enter:
            return "BEAR"
        if bull_exit:
            return "SIDEWAYS"
        return "BULL"
    # SIDEWAYS
    if bear_enter:
        return "BEAR"
    if bull_enter:
        return "BULL"
    return "SIDEWAYS"


def _daa_weights(prices_up_to_t: pd.DataFrame, wk_ret_up_to_t: pd.DataFrame,
                 max_weight: float = 0.40, n_offensive: int = 4) -> tuple[pd.Series, int]:
    """Defensive Asset Allocation (Keller & Keuning) with HERC offensive weighting."""
    avail = [c for c in prices_up_to_t.columns if prices_up_to_t[c].notna().sum() >= 253 and pd.notna(prices_up_to_t[c].iloc[-1])]
    if len(avail) < 2:
        return pd.Series(1.0 / max(1, len(avail)), index=avail), 0

    mom = _momentum_13612w(prices_up_to_t[avail])
    cash_tk = "IB01.L" if "IB01.L" in mom.index else None
    cash_mom = float(mom[cash_tk]) if cash_tk is not None else 0.0
    excess_mom = mom - cash_mom

    canaries = [t for t in avail if ETF_UNIVERSE.get(t, {}).get("canary", False) and t in excess_mom.index]
    n_bad = int(sum(float(excess_mom[t]) <= 0.0 for t in canaries))
    if not canaries:
        def_frac = 0.0
    elif n_bad == 0:
        def_frac = 0.0
    elif n_bad == 1:
        def_frac = 0.50
    else:
        def_frac = 1.0

    off_pool = [t for t in avail if ETF_UNIVERSE.get(t, {}).get("role") in ("offensive", "both") and t in excess_mom.index]
    def_pool = [t for t in avail if ETF_UNIVERSE.get(t, {}).get("role") in ("defensive", "both") and t in excess_mom.index]

    w_out = pd.Series(0.0, index=avail, dtype=float)
    daily_r = prices_up_to_t[avail].pct_change(fill_method=None).iloc[-252:]

    if def_frac < 1.0 - 1e-9 and off_pool:
        off_ranked = excess_mom[off_pool].sort_values(ascending=False)
        pos_off = off_ranked[off_ranked > 0]
        chosen_off = list((pos_off if len(pos_off) >= 2 else off_ranked).index[:min(n_offensive, len(off_ranked))])
        if len(chosen_off) == 1:
            w_off = np.array([1.0])
        else:
            wk_sub = wk_ret_up_to_t[chosen_off].iloc[-52:] if wk_ret_up_to_t is not None else None
            cov = shrunk_covariance(daily_r[chosen_off], freq="weekly", weekly_returns=wk_sub)
            w_off = herc_weights(cov, returns=daily_r[chosen_off])
            w_off = apply_caps(w_off, chosen_off, max_weight=max(max_weight, 1.0 / len(chosen_off)))
        for t, wt in zip(chosen_off, w_off):
            w_out[t] += (1.0 - def_frac) * float(wt)

    if def_frac > 1e-9 and def_pool:
        def_ranked = mom[def_pool].sort_values(ascending=False)
        chosen_def = list(def_ranked.index[:min(2, len(def_ranked))])
        if len(chosen_def) == 1:
            w_def = np.array([1.0])
        else:
            vol = daily_r[chosen_def].std().fillna(0.01).values * np.sqrt(252.0)
            w_def = (1.0 / np.maximum(vol, 1e-4))
            w_def /= w_def.sum()
        for t, wt in zip(chosen_def, w_def):
            w_out[t] += def_frac * float(wt)

    w_out = w_out[w_out > 1e-6]
    return w_out / w_out.sum(), n_bad


def _regime_nco_weights(prices_up_to_t: pd.DataFrame, wk_ret_up_to_t: pd.DataFrame,
                        regime: str, max_weight: float = 0.40) -> pd.Series:
    """Regime-conditioned Nested Clustered Optimization across offensive & defensive sleeves."""
    avail = [c for c in prices_up_to_t.columns if prices_up_to_t[c].notna().sum() >= 253 and pd.notna(prices_up_to_t[c].iloc[-1])]
    if len(avail) < 2:
        return pd.Series(1.0 / max(1, len(avail)), index=avail)

    mom = _momentum_13612w(prices_up_to_t[avail])
    daily_r = prices_up_to_t[avail].pct_change(fill_method=None).iloc[-252:]
    off_pool = [t for t in avail if ETF_UNIVERSE.get(t, {}).get("role") in ("offensive", "both") and t in mom.index]
    def_pool = [t for t in avail if ETF_UNIVERSE.get(t, {}).get("role") in ("defensive", "both") and t in mom.index]

    eq_budget = {"BULL": 0.90, "SIDEWAYS": 0.65, "BEAR": 0.25}.get(regime, 0.70)
    def_budget = 1.0 - eq_budget
    w_out = pd.Series(0.0, index=avail, dtype=float)

    if eq_budget > 0 and off_pool:
        chosen_off = list(mom[off_pool].sort_values(ascending=False).index[:min(5, len(off_pool))])
        wk_sub = wk_ret_up_to_t[chosen_off].iloc[-52:] if wk_ret_up_to_t is not None else None
        cov = shrunk_covariance(daily_r[chosen_off], freq="weekly", weekly_returns=wk_sub)
        vol = np.sqrt(np.maximum(np.diag(cov.values), 1e-12))
        z = (mom[chosen_off] - mom[chosen_off].mean()) / (mom[chosen_off].std() + 1e-12)
        mu = 0.04 * vol * z.values
        w_off = nco_weights(cov, mu=mu, risk_aversion=4.0 if regime == "BULL" else 7.0)
        w_off = apply_caps(w_off, chosen_off, max_weight=max(max_weight, 1.0 / len(chosen_off)))
        for t, wt in zip(chosen_off, w_off):
            w_out[t] += eq_budget * float(wt)

    if def_budget > 0 and def_pool:
        wk_sub = wk_ret_up_to_t[def_pool].iloc[-52:] if wk_ret_up_to_t is not None else None
        cov_d = shrunk_covariance(daily_r[def_pool], freq="weekly", weekly_returns=wk_sub)
        w_def = herc_weights(cov_d, returns=daily_r[def_pool])
        for t, wt in zip(def_pool, w_def):
            w_out[t] += def_budget * float(wt)

    w_out = w_out[w_out > 1e-6]
    return w_out / w_out.sum()


def _all_weather_herc_weights(prices_up_to_t: pd.DataFrame, wk_ret_up_to_t: pd.DataFrame,
                              max_weight: float = 0.35) -> pd.Series:
    """Multi-asset All-Weather HERC across core global/Swedish equities, gold, bonds & cash."""
    core_pref = ["EUNL.DE", "XACT-SVERIGE.ST", "IS3N.DE", "IUSN.DE", "4GLD.DE", "EUNA.DE", "IB01.L"]
    avail = [c for c in core_pref if c in prices_up_to_t.columns and prices_up_to_t[c].notna().sum() >= 253 and pd.notna(prices_up_to_t[c].iloc[-1])]
    if len(avail) < 2:
        avail = [c for c in prices_up_to_t.columns if prices_up_to_t[c].notna().sum() >= 253 and pd.notna(prices_up_to_t[c].iloc[-1])]
    if len(avail) <= 1:
        return pd.Series(1.0, index=avail)
    daily_r = prices_up_to_t[avail].pct_change(fill_method=None).iloc[-252:]
    wk_sub = wk_ret_up_to_t[avail].iloc[-52:] if wk_ret_up_to_t is not None else None
    cov = shrunk_covariance(daily_r, freq="weekly", weekly_returns=wk_sub)
    w = herc_weights(cov, returns=daily_r)
    w = apply_caps(w, avail, max_weight=max(max_weight, 1.0 / len(avail)))
    return pd.Series(w, index=avail)


def _tactical_trade_cost(trade: pd.Series, cfg: BacktestConfig, execution_mode: str) -> float:
    """Compute realistic one-way trade cost for funds (0% on Swedish fund proxies) vs ETFs."""
    fee_sched = cfg.fee_schedule()
    base = cfg.base_currency
    cost = 0.0
    for tk, wt in trade.items():
        if wt <= 1e-9:
            continue
        meta = ETF_UNIVERSE.get(tk, {})
        if execution_mode == "mutual_fund" and meta.get("fund_zero_fee", False):
            # 0% courtage & 0% FX fee on Swedish SEK mutual funds on Nordnet/Avanza
            continue
        cost += float(wt) * fee_sched.marginal_rate(ticker_currency(tk), base)
    return float(cost)


def _monthly_ter_drag(w: pd.Series) -> float:
    return sum(float(wt) * float(ETF_UNIVERSE.get(tk, {}).get("ter_bps", 15.0)) / 1e4 / 12.0
               for tk, wt in w.items())


def run_tactical_backtest(
    etf_data: MarketData,
    cfg: BacktestConfig | None = None,
    backtest_start: str | None = None,
    execution_mode: str = "mutual_fund",
    lookback_months: int = 24,
    selection_mode: str = "blend",
    budget_base: float = 550_000.0,
) -> TacticalMetaResult:
    """Walk-forward backtest & live decision across multi-asset Fund/UCITS ETF strategies."""
    cfg = (cfg or BacktestConfig(broker_preset="nordnet_mini")).validate()
    if execution_mode not in ("mutual_fund", "etf"):
        raise ValueError("execution_mode must be 'mutual_fund' or 'etf'")

    prices = etf_data.prices.sort_index()
    monthly = month_end_prices(prices, completed_only=True)
    fwd = _period_returns(monthly)
    daily_ret = prices.pct_change(fill_method=None)
    wk_ret = precompute_weekly_returns(daily_ret)

    # First valid date needs >= 253 daily obs (~1 year)
    first_valid_day = prices.index[min(253, len(prices) - 1)]
    start_ts = pd.Timestamp(backtest_start) if backtest_start else first_valid_day
    eval_dates = monthly.index[(monthly.index >= max(start_ts, first_valid_day)) & (monthly.index < monthly.index[-1])]
    if len(eval_dates) < 3:
        raise ValueError("Not enough ETF history for tactical backtest (need at least ~15 months)")

    # Build monthly benchmarks aligned to eval_dates
    bm_m = month_end_prices(etf_data.benchmarks, completed_only=True)
    bm_fwd = _period_returns(bm_m).reindex(eval_dates)
    period_ends = pd.DatetimeIndex([monthly.index[monthly.index.get_loc(t) + 1] for t in eval_dates], name="date")
    bench = pd.DataFrame(index=period_ends)
    for col, label in [("URTH", "MSCI World (URTH)"), ("SPY", "S&P 500 (SPY)"), ("XACT-OMXS30.ST", "OMXS30 Total Return (XACT)")]:
        if col in bm_fwd.columns and bm_fwd[col].notna().any():
            bench[label] = bm_fwd[col].values
    if "EUNL.DE" in fwd.columns:
        bench["Global Index (EUNL)"] = fwd.loc[eval_dates, "EUNL.DE"].values
    if "EUNL.DE" in fwd.columns and "EUNA.DE" in fwd.columns:
        bench["60/40 Global/Bonds"] = (0.6 * fwd.loc[eval_dates, "EUNL.DE"] + 0.4 * fwd.loc[eval_dates, "EUNA.DE"]).values

    # Simulate each tactical strategy walk-forward
    strat_results: dict[str, TacticalStrategyResult] = {}
    for s_key in TACTICAL_STRATEGIES:
        w_hist, gross, net, turn, costs_l, reg_l = {}, [], [], [], [], []
        prev_w = pd.Series(dtype=float)
        prev_reg = "BULL"
        for t in eval_dates:
            p_t = prices.loc[:t]
            wk_t = wk_ret.loc[:t]
            reg_t = _detect_causal_regime(p_t, prev_regime=prev_reg)
            reg_l.append(reg_t)

            if s_key == "daa":
                w, _ = _daa_weights(p_t, wk_t)
            elif s_key == "regime_nco":
                w_cand = _regime_nco_weights(p_t, wk_t, regime=reg_t)
                # Adaptive drift rule: rebalance on regime shift or L1 drift > 10%
                if len(prev_w) and reg_t == prev_reg:
                    all_idx = w_cand.index.union(prev_w.index)
                    drift = float((w_cand.reindex(all_idx, fill_value=0.0) - prev_w.reindex(all_idx, fill_value=0.0)).abs().sum() / 2.0)
                    w = prev_w.copy() if drift < 0.08 else w_cand
                else:
                    w = w_cand
            else:
                w_cand = _all_weather_herc_weights(p_t, wk_t)
                if len(prev_w):
                    all_idx = w_cand.index.union(prev_w.index)
                    drift = float((w_cand.reindex(all_idx, fill_value=0.0) - prev_w.reindex(all_idx, fill_value=0.0)).abs().sum() / 2.0)
                    w = prev_w.copy() if drift < 0.05 else w_cand
                else:
                    w = w_cand

            prev_reg = reg_t
            all_t = w.index.union(prev_w.index)
            trade = (w.reindex(all_t, fill_value=0.0) - prev_w.reindex(all_t, fill_value=0.0)).abs()
            to = float(trade.sum() / 2.0) if len(prev_w) else 1.0
            if not len(prev_w):
                trade = w.copy()
            cost = _tactical_trade_cost(trade, cfg, execution_mode) + _monthly_ter_drag(w)

            r = fwd.loc[t, w.index].fillna(0.0)
            g = float((w * r).sum())
            n_ = (1.0 + g) * (1.0 - cost) - 1.0
            prev_w = w * (1.0 + r) / (1.0 + g) if (1.0 + g) > 0 else w

            w_hist[t] = w
            gross.append(g); net.append(n_); turn.append(to); costs_l.append(cost)

        strat_results[s_key] = TacticalStrategyResult(
            name=TACTICAL_LABELS[s_key],
            returns=pd.Series(net, index=period_ends, name=s_key),
            gross_returns=pd.Series(gross, index=period_ends, name=f"{s_key}_gross"),
            weights=pd.DataFrame(w_hist).T.fillna(0.0).sort_index(),
            turnover=pd.Series(turn, index=eval_dates, name="turnover"),
            costs=pd.Series(costs_l, index=eval_dates, name="cost"),
            regimes=pd.Series(reg_l, index=eval_dates, name="regime"),
        )

    # Walk-forward optimal tactical meta-strategy
    ref_bm = bench.iloc[:, 0] if not bench.empty else pd.Series(0.0, index=period_ends)
    cand_ret_map = {TACTICAL_LABELS[k]: v.returns for k, v in strat_results.items()}
    key_by_label = {TACTICAL_LABELS[k]: k for k in TACTICAL_STRATEGIES}

    meta_w_hist, sw_hist, m_net, m_turn, m_costs = {}, {}, [], [], []
    prev_w = pd.Series(dtype=float)
    for t in eval_dates:
        sw = compute_meta_weights_at(
            t, cand_ret_map, ref_bm, lookback_months=lookback_months,
            min_eval_months=min(12, max(6, len(eval_dates) // 4)),
            selection_mode=selection_mode, rf_annual=cfg.rf_annual,
        )
        sw_hist[t] = sw
        w = pd.Series(dtype=float)
        for lbl, pw in sw.items():
            if pw <= 1e-9:
                continue
            cw = strat_results[key_by_label[lbl]].weights.loc[t]
            w = w.add(cw[cw > 1e-9] * float(pw), fill_value=0.0)
        w = w[w > 1e-4]
        w = w / w.sum()

        all_t = w.index.union(prev_w.index)
        trade = (w.reindex(all_t, fill_value=0.0) - prev_w.reindex(all_t, fill_value=0.0)).abs()
        to = float(trade.sum() / 2.0) if len(prev_w) else 1.0
        if not len(prev_w):
            trade = w.copy()
        cost = _tactical_trade_cost(trade, cfg, execution_mode) + _monthly_ter_drag(w)
        r = fwd.loc[t, w.index].fillna(0.0)
        g = float((w * r).sum())
        n_ = (1.0 + g) * (1.0 - cost) - 1.0
        prev_w = w * (1.0 + r) / (1.0 + g) if (1.0 + g) > 0 else w

        meta_w_hist[t] = w
        m_net.append(n_); m_turn.append(to); m_costs.append(cost)

    meta_ret = pd.Series(m_net, index=period_ends, name="Tactical Meta (Funds/ETFs)")
    meta_w_df = pd.DataFrame(meta_w_hist).T.fillna(0.0).sort_index()
    sw_df = pd.DataFrame(sw_hist).T.sort_index()
    latest_obs = period_ends[-1]
    current_sw = compute_meta_weights_at(
        latest_obs, cand_ret_map, ref_bm, lookback_months=lookback_months,
        min_eval_months=min(12, max(6, len(eval_dates) // 4)),
        selection_mode=selection_mode, rf_annual=cfg.rf_annual,
    )

    # Live allocation at latest available market date (strictly <= latest close)
    live_date = pd.Timestamp(prices.index.max())
    prev_reg_live = str(strat_results["regime_nco"].regimes.iloc[-1]) if strat_results["regime_nco"].regimes is not None else "BULL"
    live_regime = _detect_causal_regime(prices, prev_regime=prev_reg_live)
    w_daa_live, live_bad = _daa_weights(prices, wk_ret)
    w_nco_live = _regime_nco_weights(prices, wk_ret, regime=live_regime)
    w_aw_live = _all_weather_herc_weights(prices, wk_ret)
    live_map = {
        TACTICAL_LABELS["daa"]: w_daa_live,
        TACTICAL_LABELS["regime_nco"]: w_nco_live,
        TACTICAL_LABELS["all_weather_herc"]: w_aw_live,
    }
    w_live = pd.Series(dtype=float)
    for lbl, pw in current_sw.items():
        if pw > 1e-9:
            w_live = w_live.add(live_map[lbl] * float(pw), fill_value=0.0)
    w_live = w_live[w_live > 0.01].sort_values(ascending=False)
    w_live = w_live / w_live.sum()

    mom_live = _momentum_13612w(prices)
    live_rows = []
    for tk, wt in w_live.items():
        info = ETF_UNIVERSE.get(tk, {})
        live_rows.append({
            "ticker": tk,
            "name": info.get("name", tk),
            "fund_proxy": etf_fund_proxy(tk),
            "asset_class": info.get("asset_class", "Other"),
            "zero_fee_fund": bool(info.get("fund_zero_fee", False)),
            "weight": float(wt),
            f"target_{cfg.base_currency}": float(wt * budget_base),
            "mom_13612w": float(mom_live.get(tk, np.nan)),
            "ter_bps": float(info.get("ter_bps", 15.0)),
        })
    live_table = pd.DataFrame(live_rows).set_index("ticker")

    # Leaderboard
    n_trials = len(TACTICAL_STRATEGIES)
    lb_rows = []
    for lbl, r_s, to_s, c_s in (
        [("Tactical Meta (Funds/ETFs)", meta_ret, pd.Series(m_turn), pd.Series(m_costs))]
        + [(v.name, v.returns, v.turnover, v.costs) for v in strat_results.values()]
    ):
        s = M.summary(r_s, ref_bm.reindex(r_s.index), rf=cfg.rf_annual)
        lb_rows.append({
            "Strategy": lbl,
            "CAGR": s["CAGR"],
            "Volatility": s["Volatility"],
            "Sharpe": s["Sharpe"],
            "Deflated Sharpe": M.deflated_sharpe(r_s, n_trials=n_trials, rf=cfg.rf_annual),
            "Max drawdown": s["Max drawdown"],
            "Calmar": s["Calmar"],
            "Turnover": float(to_s.iloc[1:].mean()) if len(to_s) > 1 else float(to_s.mean()),
            "Annual cost": float(c_s.iloc[1:].mean() * 12.0) if len(c_s) > 1 else 0.0,
            "Current weight": float(current_sw.get(lbl, np.nan)),
        })
    leaderboard = pd.DataFrame(lb_rows).set_index("Strategy")

    notes = [
        f"Causal regime detector: {live_regime} | DAA canary alerts: {live_bad}/3 negative momentum.",
        f"Cost model: {'0% courtage/FX on Swedish mutual fund proxies + TER' if execution_mode == 'mutual_fund' else cfg.fee_schedule().label + ' + TER'}.",
    ]
    return TacticalMetaResult(
        meta_returns=meta_ret, meta_weights=meta_w_df,
        meta_turnover=pd.Series(m_turn, index=eval_dates, name="turnover"),
        meta_costs=pd.Series(m_costs, index=eval_dates, name="cost"),
        benchmarks=bench, strategies=strat_results, leaderboard=leaderboard,
        strategy_weights=sw_df, current_strategy_weights=current_sw,
        live_as_of=live_date, live_regime=live_regime, live_canary_bad=live_bad,
        live_table=live_table, execution_mode=execution_mode,
        base_currency=cfg.base_currency, notes=notes,
    )


def render_tactical_brief(tactical: TacticalMetaResult, core_budget: float,
                          satellite_brief: str | None = None) -> str:
    """Render a Markdown report for the multi-asset Fund/ETF core (and optional Core-Satellite brief)."""
    cur = tactical.base_currency
    lines = [
        f"# Multi-Asset Fund & UCITS ETF Allocation — {tactical.live_as_of:%B %Y}",
        "",
        f"- **As of:** {tactical.live_as_of:%Y-%m-%d} | **Regime:** `{tactical.live_regime}` | **DAA Canary Alerts:** `{tactical.live_canary_bad}/3`",
        f"- **Core Budget:** {core_budget:,.0f} {cur} | **Execution Mode:** `{tactical.execution_mode}`",
    ]
    for n in tactical.notes:
        lines.append(f"- {n}")
    lines.extend(["", "## Target Core Fund / UCITS ETF Allocation", ""])
    lines.append(f"| Proxy ETF | Nordnet / Avanza 0%-Fee Fund Equivalent | Asset Class | Weight | Target ({cur}) | 13612W Mom | 0% Fee Fund? |")
    lines.append("|---|---|---|---:|---:|---:|:---:|")
    for tk, r in tactical.live_table.iterrows():
        mom_s = f"{r['mom_13612w']:+.1%}" if pd.notna(r["mom_13612w"]) else "–"
        zf = "Yes" if r["zero_fee_fund"] else "ETF/ETC"
        lines.append(
            f"| `{tk}` | {r['fund_proxy']} | {r['asset_class']} | {r['weight']:.1%} | "
            f"{r['weight'] * core_budget:,.0f} | {mom_s} | {zf} |"
        )
    lines.extend(["", "## Walk-Forward Multi-Asset Strategy Leaderboard", ""])
    lines.append("| Strategy | CAGR | Vol | Sharpe | Deflated Sharpe | Max DD | Calmar | Turnover | Annual Cost | Live Wt |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, r in tactical.leaderboard.iterrows():
        lw = f"{r['Current weight']:.0%}" if pd.notna(r["Current weight"]) else "–"
        lines.append(
            f"| {name} | {r['CAGR']:.1%} | {r['Volatility']:.1%} | {r['Sharpe']:.2f} | "
            f"{r['Deflated Sharpe']:.0%} | {r['Max drawdown']:.1%} | {r['Calmar']:.2f} | "
            f"{r['Turnover']:.0%} | {r['Annual cost']:.2%} | {lw} |"
        )
    if satellite_brief:
        lines.extend(["", "---", "", "## Active Single-Stock Satellite Sleeve", "", satellite_brief])
    return "\n".join(lines)
