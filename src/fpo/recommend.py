"""Live recommendation and monthly actionable portfolio decision engine."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from fpo.config import BacktestConfig
from fpo.data import DataWarning, MarketData, validate_market_data
from fpo.features import FEATURES, build_live_features, cross_sectional_ranks
from fpo.meta import (
    DEFAULT_CANDIDATES, MetaBacktestResult, StrategyCandidate, StrategyHealth,
    assess_strategy_health, run_meta_backtest,
)
from fpo.models import EnsembleModel, fit_full_model, oos_permutation_importance
from fpo.optimizer import (
    apply_caps, compute_regime_exposure, estimate_dynamic_ic, optimise_weights,
)
from fpo.portfolio import ExecutionPlan, Holding, PortfolioState, build_execution_plan
from fpo.universe import ticker_country, ticker_currency, ticker_sector


@dataclass
class Recommendation:
    as_of: pd.Timestamp
    price_date: pd.Timestamp
    table: pd.DataFrame            # one row per holding
    all_scores: pd.DataFrame       # every eligible stock with its score and rank
    cash_left: float
    budget: float
    base_currency: str
    importance: pd.Series | None = None
    trades: pd.DataFrame | None = None
    data_warnings: list[DataWarning] = field(default_factory=list)
    execution_plan: ExecutionPlan | None = None


@dataclass
class MonthlyDecision:
    as_of: pd.Timestamp
    price_date: pd.Timestamp
    strategy_label: str
    strategy_weights: pd.Series
    health: StrategyHealth
    data_warnings: list[DataWarning]
    target_table: pd.DataFrame
    all_scores: pd.DataFrame
    plan: ExecutionPlan
    meta_result: MetaBacktestResult | None = None
    importance: pd.Series | None = None


def allocate_whole_shares(weights: pd.Series, prices: pd.Series, budget: float) -> dict[str, int]:
    """Round target amounts down to whole shares, then greedily spend leftover cash on the
    holdings furthest below target, as long as each extra share brings the holding closer
    to its target (i.e. its shortfall is at least half a share) and fits in the budget."""
    px = prices.astype(float)
    target = weights * budget
    shares = np.floor(target / px).fillna(0).astype(int)
    cash = budget - float((shares * px).sum())
    for _ in range(10_000):
        gap = target - shares * px                    # shortfall vs target, in money
        ok = (px <= cash + 1e-9) & (gap >= 0.5 * px)
        if not ok.any():
            break
        best = (gap[ok] / budget).idxmax()
        shares[best] += 1
        cash -= px[best]
    return {k: int(v) for k, v in shares.items()}


def _clean_holdings_map(current_holdings: dict[str, float] | None) -> dict[str, float]:
    if not current_holdings:
        return {}
    return {str(k).strip(): float(v) for k, v in current_holdings.items() if float(v) > 1e-9}


def _current_weights_from_holdings(holdings: dict[str, float], base_px: pd.Series) -> pd.Series | None:
    if not holdings:
        return None
    vals = {}
    for t, sh in holdings.items():
        px = base_px.get(t, np.nan)
        if pd.notna(px) and px > 0 and sh > 0:
            vals[t] = sh * float(px)
    tot = sum(vals.values())
    if tot <= 0:
        return None
    return pd.Series({t: v / tot for t, v in vals.items()}, dtype=float)


def recommend_portfolio(data: MarketData, cfg: BacktestConfig | None = None, budget: float = 100_000,
                        features: pd.DataFrame | None = None,
                        current_holdings: dict[str, float] | None = None,
                        compute_importance: bool = False,
                        as_of: pd.Timestamp | None = None,
                        effective_ic: float | None = None) -> Recommendation:
    """Train on all completed history and build the live portfolio for `as_of` (latest close).

    budget: total amount to invest, in the base currency.
    current_holdings: optional {ticker: shares} you already own -> feeds hold_buffer,
                      turnover penalty, and produces a trade list & ExecutionPlan.
    """
    cfg = (cfg or BacktestConfig()).validate()
    hist_feats, live_feats, live_date = build_live_features(data, as_of=as_of, completed_feats=features)
    model, _, _ = fit_full_model(hist_feats, cfg, live_date)

    X_now = cross_sectional_ranks(live_feats).xs(live_date, level="date")
    uncertainty = None
    if isinstance(model, EnsembleModel):
        pred, unc = model.predict_with_uncertainty(X_now[FEATURES])
        scores = pd.Series(pred, index=X_now.index)
        uncertainty = pd.Series(unc, index=X_now.index)
    else:
        scores = pd.Series(np.asarray(model.predict(X_now[FEATURES]), dtype=float), index=X_now.index)

    price_date = data.prices.index[data.prices.index <= live_date][-1]
    local_px = data.local_prices.loc[:price_date].ffill().iloc[-1]
    base_px = data.prices.loc[:price_date].ffill().iloc[-1]
    fx_now = float(data.fx.loc[:price_date].ffill().iloc[-1])

    clean_h = _clean_holdings_map(current_holdings)
    held = list(clean_h) if clean_h else None
    prev_w = _current_weights_from_holdings(clean_h, base_px)

    hist = data.prices.pct_change(fill_method=None).loc[:live_date]
    w = optimise_weights(scores, hist, cfg, current=held, prev_weights=prev_w, effective_ic=effective_ic)

    shares_map = allocate_whole_shares(w, base_px[w.index], budget)
    live_raw = live_feats.xs(live_date, level="date")
    rows = []
    for t, wt in w.items():
        amount = budget * wt
        px_b = float(base_px[t])
        shares = shares_map[t]
        row = {
            "ticker": t, "name": data.names.get(t, t), "country": ticker_country(t),
            "sector": ticker_sector(t), "currency": ticker_currency(t),
            "weight": wt, "score": float(scores[t]),
            "price_local": float(local_px[t]), f"price_{data.base_currency}": px_b,
            f"target_{data.base_currency}": amount, "shares": shares,
            f"invested_{data.base_currency}": shares * px_b,
            "mom_12_1": float(live_raw.loc[t, "mom_12_1"]),
            "vol_3m": float(live_raw.loc[t, "vol_3m"]),
        }
        if uncertainty is not None:
            row["disagreement"] = float(uncertainty[t])
        rows.append(row)
    table = pd.DataFrame(rows).set_index("ticker")
    cash_left = budget - float(table[f"invested_{data.base_currency}"].sum())

    all_scores = pd.DataFrame({"score": scores, "rank": scores.rank(ascending=False).astype(int)})
    all_scores["name"] = [data.names.get(t, t) for t in all_scores.index]
    all_scores["country"] = [ticker_country(t) for t in all_scores.index]
    all_scores["sector"] = [ticker_sector(t) for t in all_scores.index]
    all_scores["mom_12_1"] = live_raw["mom_12_1"].reindex(all_scores.index)
    all_scores["vol_3m"] = live_raw["vol_3m"].reindex(all_scores.index)
    if uncertainty is not None:
        all_scores["disagreement"] = uncertainty.reindex(all_scores.index)
    all_scores["selected"] = all_scores.index.isin(w.index)
    all_scores = all_scores.sort_values("rank")

    trades = None
    if clean_h:
        cur = pd.Series(clean_h, dtype=float)
        tgt = table["shares"].astype(float)
        idx = cur.index.union(tgt.index)
        trades = pd.DataFrame({"current_shares": cur.reindex(idx, fill_value=0.0),
                               "target_shares": tgt.reindex(idx, fill_value=0.0)})
        trades["trade_shares"] = trades["target_shares"] - trades["current_shares"]
        trades["action"] = np.where(trades["trade_shares"] > 0, "BUY",
                                    np.where(trades["trade_shares"] < 0, "SELL", "HOLD"))

    # Also build a full fee-aware ExecutionPlan
    existing_val = sum(sh * float(base_px[t]) for t, sh in clean_h.items() if t in base_px.index and pd.notna(base_px[t]))
    implied_cash = max(0.0, budget - existing_val)
    state = PortfolioState(
        broker_preset=cfg.broker_preset,
        base_currency=data.base_currency,
        cash_sek=implied_cash if data.base_currency == "SEK" else 0.0,
        cash_usd=implied_cash if data.base_currency == "USD" else 0.0,
        holdings={t: Holding(sh) for t, sh in clean_h.items()},
    )
    plan = build_execution_plan(
        target_weights=w, all_scores=all_scores, local_prices=local_px, base_prices=base_px,
        fx_rate=fx_now, state=state, cfg=cfg, as_of=live_date, price_date=price_date,
        deposit_base=0.0, exposure=1.0,
        effective_ic=cfg.assumed_ic if effective_ic is None else effective_ic,
        names=data.names,
    )

    importance = None
    if compute_importance and cfg.model != "momentum":
        importance = oos_permutation_importance(hist_feats, cfg)

    check_tickers = list(dict.fromkeys(list(w.index) + list(clean_h)))
    warnings_out = validate_market_data(data, tickers=check_tickers)

    return Recommendation(
        as_of=live_date, price_date=price_date, table=table, all_scores=all_scores,
        cash_left=cash_left, budget=budget, base_currency=data.base_currency,
        importance=importance, trades=trades, data_warnings=warnings_out,
        execution_plan=plan,
    )


def recommend_optimal_portfolio(
    data: MarketData,
    cfg: BacktestConfig | None = None,
    state: PortfolioState | None = None,
    deposit_base: float = 10_000.0,
    features: pd.DataFrame | None = None,
    meta_result: MetaBacktestResult | None = None,
    candidates: tuple[StrategyCandidate, ...] = DEFAULT_CANDIDATES,
    selection_mode: str = "blend",
    lookback_months: int = 36,
    use_meta: bool = True,
    compute_importance: bool = False,
    as_of: pd.Timestamp | None = None,
    progress=None,
) -> MonthlyDecision:
    """Produce a complete, actionable monthly portfolio decision.

    When `use_meta=True` (default), uses walk-forward meta-strategy selection/blending
    to choose the optimal combination of models & optimizers, calibrates dynamic IC,
    evaluates strategy health (GREEN/YELLOW/RED), and builds a cash-safe Avanza/Nordnet
    execution ticket incorporating `state` and `deposit_base`.
    """
    cfg = (cfg or BacktestConfig()).validate()
    if state is None:
        state = PortfolioState(broker_preset=cfg.broker_preset, base_currency=cfg.base_currency)

    hist_feats, live_feats, live_date = build_live_features(data, as_of=as_of, completed_feats=features)
    price_date = data.prices.index[data.prices.index <= live_date][-1]
    local_px = data.local_prices.loc[:price_date].ffill().iloc[-1]
    base_px = data.prices.loc[:price_date].ffill().iloc[-1]
    fx_now = float(data.fx.loc[:price_date].ffill().iloc[-1])

    clean_h = state.held_shares()
    held = list(clean_h) if clean_h else None
    prev_w = _current_weights_from_holdings(clean_h, base_px)
    hist_rets = data.prices.pct_change(fill_method=None).loc[:live_date]
    X_now = cross_sectional_ranks(live_feats).xs(live_date, level="date")
    live_raw = live_feats.xs(live_date, level="date")

    if use_meta:
        meta = meta_result
        if meta is None:
            meta = run_meta_backtest(
                data, cfg, candidates=candidates, features=hist_feats,
                lookback_months=lookback_months, selection_mode=selection_mode,
                progress=progress,
            )
        sw = meta.current_strategy_weights
        active_cands = [c for c in meta.candidate_specs if float(sw.get(c.name, 0.0)) > 1e-4]
        unique_models = list(dict.fromkeys(c.model for c in active_cands))

        # Train each active model ONCE on all completed history strictly before live_date
        live_scores_by_model: dict[str, pd.Series] = {}
        for m_name in unique_models:
            m_cfg = replace(cfg, model=m_name)
            model, _, _ = fit_full_model(hist_feats, m_cfg, live_date)
            pred = np.asarray(model.predict(X_now[FEATURES]), dtype=float)
            live_scores_by_model[m_name] = pd.Series(pred, index=X_now.index)

        blended_w = pd.Series(dtype=float)
        blended_z = pd.Series(0.0, index=X_now.index, dtype=float)
        for c in active_cands:
            pw = float(sw[c.name])
            sc = live_scores_by_model[c.model]
            z = (sc - sc.mean()) / (sc.std() + 1e-12)
            blended_z = blended_z + z * pw

            cand_ic = meta.candidates[c.name].ic
            eff_ic_c = estimate_dynamic_ic(cand_ic, prior=cfg.ic_prior, lookback=cfg.ic_lookback_months) if cfg.use_dynamic_ic else cfg.assumed_ic
            c_cfg = replace(cfg, model=c.model, optimizer=c.optimizer)
            cw = optimise_weights(sc, hist_rets, c_cfg, current=held, prev_weights=prev_w, effective_ic=eff_ic_c)
            blended_w = blended_w.add(cw * pw, fill_value=0.0)

        blended_w = blended_w[blended_w > 1e-6].sort_values(ascending=False)
        if len(blended_w) > cfg.n_holdings:
            blended_w = blended_w.iloc[:cfg.n_holdings]
        w_arr = apply_caps(
            blended_w.values, list(blended_w.index),
            cfg.max_weight, cfg.max_country_weight, cfg.max_sector_weight,
        )
        w = pd.Series(w_arr, index=blended_w.index)
        if cfg.min_weight > 0 and (w < cfg.min_weight).any() and (w >= cfg.min_weight).sum() >= 1:
            w = w[w >= cfg.min_weight]
            w = pd.Series(
                apply_caps(w.values, list(w.index), cfg.max_weight, cfg.max_country_weight, cfg.max_sector_weight),
                index=w.index,
            )
        w = w / w.sum()

        health = meta.health
        eff_ic = estimate_dynamic_ic(meta.meta_result.ic, prior=cfg.ic_prior, lookback=cfg.ic_lookback_months) if cfg.use_dynamic_ic else cfg.assumed_ic
        exposure = health.recommended_exposure
        scores = blended_z
        strategy_label = f"Optimal Meta-Strategy ({selection_mode})"
        strategy_weights = sw
        imp_model_name = max(active_cands, key=lambda c: float(sw[c.name])).model
    else:
        from fpo.backtest import run_backtest
        bt = run_backtest(data, cfg, features=hist_feats)
        model, _, _ = fit_full_model(hist_feats, cfg, live_date)
        scores = pd.Series(np.asarray(model.predict(X_now[FEATURES]), dtype=float), index=X_now.index)
        eff_ic = estimate_dynamic_ic(bt.ic, prior=cfg.ic_prior, lookback=cfg.ic_lookback_months) if cfg.use_dynamic_ic else cfg.assumed_ic
        w = optimise_weights(scores, hist_rets, cfg, current=held, prev_weights=prev_w, effective_ic=eff_ic)
        exposure = compute_regime_exposure(data.benchmarks, live_date, target_vol=cfg.target_vol) if cfg.regime_overlay else 1.0
        health = assess_strategy_health(bt, n_trials=1, recommended_exposure=exposure, rf_annual=cfg.rf_annual)
        meta = None
        strategy_label = f"{cfg.model} + {cfg.optimizer}"
        strategy_weights = pd.Series({strategy_label: 1.0})
        imp_model_name = cfg.model

    # Defensive blend when core_fallback is active and trailing IC is weak
    if cfg.core_fallback and health.core_fallback_active and len(w) > 1:
        vol = hist_rets[w.index].iloc[-cfg.cov_lookback_days:].std().fillna(0.02).values * np.sqrt(252.0)
        w_def = apply_caps(1.0 / np.maximum(vol, 1e-4), list(w.index),
                           cfg.max_weight, cfg.max_country_weight, cfg.max_sector_weight)
        w = pd.Series(
            apply_caps(0.35 * w.values + 0.65 * w_def, list(w.index),
                       cfg.max_weight, cfg.max_country_weight, cfg.max_sector_weight),
            index=w.index,
        )
        w = w / w.sum()

    all_scores = pd.DataFrame({"score": scores, "rank": scores.rank(ascending=False).astype(int)})
    all_scores["name"] = [data.names.get(t, t) for t in all_scores.index]
    all_scores["country"] = [ticker_country(t) for t in all_scores.index]
    all_scores["sector"] = [ticker_sector(t) for t in all_scores.index]
    all_scores["mom_12_1"] = live_raw["mom_12_1"].reindex(all_scores.index)
    all_scores["vol_3m"] = live_raw["vol_3m"].reindex(all_scores.index)
    all_scores["selected"] = all_scores.index.isin(w.index)
    all_scores = all_scores.sort_values("rank")

    plan = build_execution_plan(
        target_weights=w, all_scores=all_scores, local_prices=local_px, base_prices=base_px,
        fx_rate=fx_now, state=state, cfg=cfg, as_of=live_date, price_date=price_date,
        deposit_base=deposit_base, exposure=exposure, effective_ic=eff_ic, names=data.names,
    )

    rows = []
    for t, wt in w.items():
        rows.append({
            "ticker": t, "name": data.names.get(t, t), "country": ticker_country(t),
            "sector": ticker_sector(t), "currency": ticker_currency(t),
            "weight": float(wt), "rank": int(all_scores.loc[t, "rank"]),
            "score": float(scores[t]), "price_local": float(local_px[t]),
            f"price_{data.base_currency}": float(base_px[t]),
            "mom_12_1": float(live_raw.loc[t, "mom_12_1"]),
            "vol_3m": float(live_raw.loc[t, "vol_3m"]),
        })
    target_table = pd.DataFrame(rows).set_index("ticker")

    check_tickers = list(dict.fromkeys(list(w.index) + list(clean_h)))
    data_warnings = validate_market_data(data, tickers=check_tickers)

    importance = None
    if compute_importance and imp_model_name != "momentum":
        importance = oos_permutation_importance(hist_feats, replace(cfg, model=imp_model_name))

    return MonthlyDecision(
        as_of=live_date, price_date=price_date, strategy_label=strategy_label,
        strategy_weights=strategy_weights, health=health, data_warnings=data_warnings,
        target_table=target_table, all_scores=all_scores, plan=plan,
        meta_result=meta, importance=importance,
    )


def render_monthly_brief(decision: MonthlyDecision) -> str:
    """Render a clean, actionable Markdown monthly decision report."""
    p = decision.plan
    h = decision.health
    cur = p.base_currency
    badge = {"GREEN": "🟢 GREEN", "YELLOW": "🟡 YELLOW", "RED": "🔴 RED"}.get(h.status, h.status)

    lines = [
        f"# Monthly Portfolio Decision — {decision.as_of:%B %Y}",
        "",
        f"- **Signal date:** {decision.as_of:%Y-%m-%d} (prices as of {decision.price_date:%Y-%m-%d})",
        f"- **Strategy:** {decision.strategy_label}",
        f"- **Broker / Mode:** {p.broker_schedule.label} · `{p.rebalance_mode}`",
        f"- **Verdict:** **`{p.verdict}`** — {p.verdict_reason}",
        f"- **Strategy Health:** **{badge}** — {h.headline}",
        "",
        "## Portfolio & Cash Summary",
        "",
        f"| Metric | Value ({cur}) |",
        "|---|---:|",
        f"| Existing holdings (mark-to-market) | {p.existing_equity_value:,.0f} |",
        f"| Cash + new deposit ({p.deposit_amount:,.0f} new) | {p.starting_cash:,.0f} |",
        f"| **Total portfolio value** | **{p.total_portfolio_value:,.0f}** |",
        f"| Target equity exposure | {p.exposure:.0%} |",
        f"| Gross SELL proceeds | {p.sell_proceeds_gross:,.0f} |",
        f"| Planned BUY notional | {p.buy_notional:,.0f} |",
        f"| Estimated brokerage + FX + spread fees | {p.total_fees:,.0f} ({p.total_fees / max(p.total_portfolio_value, 1.0):.2%}) |",
        f"| Projected cash remaining | {p.projected_cash_left:,.0f} |",
        "",
    ]

    if decision.data_warnings:
        lines.append("## Data Hygiene Warnings")
        lines.append("")
        for w in decision.data_warnings:
            lines.append(f"- **[{w.severity.upper()}] {w.ticker}**: {w.message}")
        lines.append("")

    lines.append("## Optimal Strategy Blend & Health")
    lines.append("")
    lines.append(f"- Trailing 24m OOS Rank-IC: **{h.trailing_ic:+.3f}** | Trailing 36m Sharpe: **{h.trailing_sharpe:.2f}** | Deflated Sharpe: **{h.deflated_sharpe:.0%}** | Current Drawdown: **{h.current_drawdown:.1%}**")
    for note in h.notes:
        lines.append(f"- {note}")
    lines.append("")
    if len(decision.strategy_weights) > 1:
        lines.append("| Strategy | Allocation |")
        lines.append("|---|---:|")
        for sname, sw in decision.strategy_weights.sort_values(ascending=False).items():
            if sw > 1e-4:
                lines.append(f"| {sname} | {sw:.1%} |")
        lines.append("")

    def _fmt_orders(df: pd.DataFrame, title: str) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if df.empty:
            lines.append("_None this month._")
            lines.append("")
            return
        lines.append(f"| Ticker | Name | Sector | Shares | Price | Value ({cur}) | Est. Fee ({cur}) | Post Wt | Reason |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---|")
        for tk, r in df.iterrows():
            sh_txt = f"{r['trade_shares']:+.0f}" if r["action"] in ("BUY", "SELL") else f"{r['current_shares']:.0f}"
            px_txt = f"{r['price_local']:,.2f} {r['currency']}" if pd.notna(r["price_local"]) else "N/A"
            lines.append(
                f"| `{tk}` | {r['name']} | {r['sector']} | {sh_txt} | {px_txt} | "
                f"{r[f'trade_value_{cur}']:,.0f} | {r[f'est_fee_{cur}']:,.0f} | "
                f"{r['post_weight']:.1%} | {r['reason']} |"
            )
        lines.append("")

    _fmt_orders(p.sells, "Step 1 · SELL Orders (execute first)")
    _fmt_orders(p.buys, "Step 2 · BUY Orders (execute after sells clear)")
    _fmt_orders(p.holds, "Step 3 · HOLD / Skipped Micro-Trades")

    lines.append("---")
    lines.append("_Research & decision-support tool, not licensed investment advice. Verify quotes and FX rates at Avanza/Nordnet before placing orders._")
    return "\n".join(lines)
