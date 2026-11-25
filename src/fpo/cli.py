"""Command line interface.

    fpo decide        --portfolio my_portfolio.json --deposit 10000 --broker avanza_mini
    fpo meta-backtest [--broker avanza_mini] [--selection-mode blend] [--synthetic]
    fpo backtest      [--model ensemble] [--optimizer mean_variance] [--synthetic]
    fpo recommend     --budget 50000 [--holdings holdings.csv]
    fpo init-portfolio --out my_portfolio.json [--from-csv holdings.csv]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from fpo import metrics as M
from fpo.backtest import EW_UNIVERSE, run_backtest
from fpo.config import BROKER_PRESETS, MODELS, OPTIMIZERS, REBALANCE_MODES, BacktestConfig
from fpo.data import (
    DEFAULT_CACHE_DIR, load_etf_market_data, load_market_data,
    synthetic_etf_market_data, synthetic_market_data,
)
from fpo.meta import DEFAULT_CANDIDATES, FAST_CANDIDATES, run_meta_backtest
from fpo.portfolio import (
    PortfolioState, apply_execution_plan, load_portfolio, portfolio_from_csv, save_portfolio,
)
from fpo.recommend import recommend_optimal_portfolio, recommend_portfolio, render_monthly_brief
from fpo.tactical import render_tactical_brief, run_tactical_backtest


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--markets", default="both", choices=["us", "se", "both"])
    p.add_argument("--currency", default="SEK", choices=["SEK", "USD"])
    p.add_argument("--start", default="2005-01-01", help="first date of price history")
    p.add_argument("--model", default="ensemble", choices=MODELS)
    p.add_argument("--optimizer", default="mean_variance", choices=OPTIMIZERS)
    p.add_argument("--n-holdings", type=int, default=15)
    p.add_argument("--max-weight", type=float, default=0.12)
    p.add_argument("--max-country-weight", type=float, default=1.0)
    p.add_argument("--max-sector-weight", type=float, default=0.35)
    p.add_argument("--broker", default="custom", choices=list(BROKER_PRESETS),
                   help="broker fee preset (e.g. avanza_mini, nordnet_mini, nordnet_small)")
    p.add_argument("--cost-bps", type=float, default=15.0)
    p.add_argument("--fx-cost-bps", type=float, default=10.0)
    p.add_argument("--regime-overlay", action="store_true",
                   help="enable vol-targeting + 200d trend defensive cash overlay")
    p.add_argument("--rf", type=float, default=0.02, help="annual risk-free rate for Sharpe/Sortino")
    p.add_argument("--synthetic", action="store_true", help="use simulated data (offline demo)")
    p.add_argument("--out", default="outputs", help="folder for outputs")


def _config(a, default_broker: str | None = None) -> BacktestConfig:
    broker = getattr(a, "broker", None) or default_broker or "custom"
    return BacktestConfig(
        markets=a.markets, base_currency=a.currency, start=a.start, model=a.model,
        optimizer=a.optimizer, n_holdings=a.n_holdings, max_weight=a.max_weight,
        max_country_weight=a.max_country_weight, max_sector_weight=a.max_sector_weight,
        broker_preset=broker, cost_bps=a.cost_bps, fx_cost_bps=a.fx_cost_bps,
        regime_overlay=getattr(a, "regime_overlay", False),
        rebalance_mode=getattr(a, "mode", "smart_dca"),
        min_order_sek=getattr(a, "min_order_sek", 1500.0),
        no_trade_band=getattr(a, "no_trade_band", 0.015),
        rf_annual=getattr(a, "rf", 0.02),
        backtest_start=getattr(a, "backtest_start", None) or "2012-01-31",
    )


def _data(a, cfg: BacktestConfig, extra_tickers: list[str] | None = None):
    if a.synthetic:
        return synthetic_market_data(cfg.markets, start=cfg.start, base_currency=cfg.base_currency)
    return load_market_data(cfg.markets, start=cfg.start, base_currency=cfg.base_currency,
                            extra_tickers=extra_tickers)


def cmd_backtest(a) -> None:
    cfg = _config(a)
    data = _data(a, cfg)
    res = run_backtest(data, cfg, progress=lambda p, msg: print(f"[{p:>4.0%}] {msg}"))
    pd.set_option("display.width", 200)
    table = M.comparison_table(res.returns, res.benchmarks, ref=EW_UNIVERSE, rf=cfg.rf_annual)
    print("\n=== Performance (monthly, net of costs, in", cfg.base_currency, ") ===")
    print(table.T[["CAGR", "Volatility", "Sharpe", "Max drawdown", "Total return"]].round(3))
    for b in res.benchmarks.columns:
        t = M.excess_return_test(res.returns, res.benchmarks[b])
        print(f"vs {b:32s} excess t-stat {t['t_stat']:+.2f}  one-sided p {t['p_value']:.3f}  "
              f"P(outperform, bootstrap) {t['p_outperform_bootstrap']:.0%}")
    rp = M.random_portfolio_percentile(res.gross_returns, res.random_returns)
    print(f"\nMean monthly rank IC: {res.ic.mean():.3f} (t={res.ic.mean() / res.ic.std() * len(res.ic) ** .5:.2f})")
    print(f"Strategy CAGR beats {rp['cagr_percentile']:.0%} of {len(rp['random_cagr'])} random "
          f"{cfg.n_holdings}-stock portfolios from the same universe (before costs)")
    print(f"Average one-way turnover per month: {res.turnover.mean():.0%}, "
          f"annual cost drag: {res.costs.mean() * 12:.2%}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat([res.returns, res.benchmarks], axis=1).to_csv(out / "monthly_returns.csv")
    res.weights.to_csv(out / "weights.csv")
    table.to_csv(out / "summary.csv")
    print(f"\nSaved CSVs to {out.resolve()}")


def cmd_meta_backtest(a) -> None:
    cfg = _config(a)
    data = _data(a, cfg)
    cands = FAST_CANDIDATES if a.fast else DEFAULT_CANDIDATES
    meta = run_meta_backtest(
        data, cfg, candidates=cands, lookback_months=a.lookback_months,
        selection_mode=a.selection_mode,
        cache_dir=None if a.synthetic else DEFAULT_CACHE_DIR,
        progress=lambda p, msg: print(f"[{p:>4.0%}] {msg}"),
    )
    pd.set_option("display.width", 220)
    print(f"\n=== Walk-Forward Optimal Strategy Leaderboard ({cfg.base_currency}, {cfg.fee_schedule().label}) ===")
    cols = ["CAGR", "Volatility", "Sharpe", "Deflated Sharpe", "Max drawdown", "Excess CAGR (vs EW)", "Mean IC", "Turnover", "Current weight"]
    print(meta.leaderboard[cols].round(3).to_string())
    print(f"\nStrategy Health: [{meta.health.status}] {meta.health.headline}")
    for note in meta.health.notes:
        print(f"  - {note}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta.leaderboard.to_csv(out / "meta_leaderboard.csv")
    meta.strategy_weights.to_csv(out / "meta_strategy_weights.csv")
    pd.concat([meta.meta_result.returns, meta.meta_result.benchmarks], axis=1).to_csv(out / "meta_returns.csv")
    print(f"\nSaved meta-backtest CSVs to {out.resolve()}")


def cmd_tactical(a) -> None:
    cfg = _config(a)
    if a.synthetic:
        etf_data = synthetic_etf_market_data(start=a.start, base_currency=cfg.base_currency)
    else:
        etf_data = load_etf_market_data(start=a.start, base_currency=cfg.base_currency)

    tactical = run_tactical_backtest(
        etf_data, cfg=cfg, backtest_start=a.backtest_start,
        execution_mode=a.exec_mode, lookback_months=a.lookback_months,
        selection_mode=a.selection_mode, budget_base=a.budget,
    )
    pd.set_option("display.width", 220)
    print(f"\n=== Multi-Asset Fund & UCITS ETF Leaderboard ({cfg.base_currency}, mode={a.exec_mode}) ===")
    print(tactical.leaderboard.round(3).to_string())
    print(f"\nLive Regime: {tactical.live_regime} | DAA Canary Alerts: {tactical.live_canary_bad}/3 | As of {tactical.live_as_of:%Y-%m-%d}")
    print(f"\nTarget Core Allocation ({a.budget:,.0f} {cfg.base_currency}):")
    cols = ["fund_proxy", "asset_class", "weight", f"target_{cfg.base_currency}", "mom_13612w", "zero_fee_fund"]
    print(tactical.live_table[cols].to_string(float_format=lambda x: f"{x:,.3f}"))

    sat_brief = None
    if a.with_satellite > 0:
        stock_data = _data(a, cfg)
        st = PortfolioState(broker_preset=cfg.broker_preset, base_currency=cfg.base_currency)
        dec = recommend_optimal_portfolio(
            stock_data, cfg, state=st, deposit_base=a.with_satellite,
            candidates=FAST_CANDIDATES if a.fast else DEFAULT_CANDIDATES,
            cache_dir=None if a.synthetic else DEFAULT_CACHE_DIR,
        )
        sat_brief = render_monthly_brief(dec)
        print(f"\nActive Satellite Sleeve ({a.with_satellite:,.0f} {cfg.base_currency}): "
              f"Verdict={dec.plan.verdict} | Buy={dec.plan.buy_notional:,.0f} {cfg.base_currency} | Fees={dec.plan.total_fees:,.0f}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    brief_path = out / f"tactical_{tactical.live_as_of:%Y-%m}.md"
    alloc_path = out / f"tactical_allocation_{tactical.live_as_of:%Y-%m}.csv"
    brief_path.write_text(render_tactical_brief(tactical, a.budget, satellite_brief=sat_brief), encoding="utf-8")
    tactical.live_table.to_csv(alloc_path)
    print(f"\nSaved tactical brief      -> {brief_path.resolve()}")
    print(f"Saved tactical allocation -> {alloc_path.resolve()}")


def cmd_decide(a) -> None:
    cfg = _config(a)
    state: PortfolioState
    if a.portfolio and Path(a.portfolio).exists():
        state = load_portfolio(a.portfolio)
        if a.broker != "avanza_mini" or state.broker_preset == "custom":
            state.broker_preset = a.broker
        else:
            cfg.broker_preset = state.broker_preset
    else:
        state = PortfolioState(broker_preset=cfg.broker_preset, base_currency=cfg.base_currency,
                               cash_sek=a.cash if cfg.base_currency == "SEK" else 0.0,
                               cash_usd=a.cash if cfg.base_currency == "USD" else 0.0)

    extra = list(state.held_shares()) if state.held_shares() else None
    data = _data(a, cfg, extra_tickers=extra)
    cands = FAST_CANDIDATES if a.fast else DEFAULT_CANDIDATES
    decision = recommend_optimal_portfolio(
        data, cfg, state=state, deposit_base=a.deposit,
        candidates=cands, selection_mode=a.selection_mode,
        lookback_months=a.lookback_months, use_meta=(a.strategy == "optimal"),
        cache_dir=None if a.synthetic else DEFAULT_CACHE_DIR,
        progress=lambda p, msg: print(f"[{p:>4.0%}] {msg}"),
    )

    brief = render_monthly_brief(decision)
    p = decision.plan
    cur = p.base_currency
    print(f"\n=== Monthly Decision for {decision.as_of:%B %Y} (prices as of {decision.price_date:%Y-%m-%d}) ===")
    print(f"Strategy : {decision.strategy_label}")
    print(f"Health   : [{decision.health.status}] {decision.health.headline}")
    print(f"Verdict  : [{p.verdict}] {p.verdict_reason}")
    print(f"Portfolio: {p.existing_equity_value:,.0f} equity + {p.starting_cash:,.0f} cash/deposit = "
          f"{p.total_portfolio_value:,.0f} {cur} | Est. fees: {p.total_fees:,.0f} {cur} | "
          f"Cash left: {p.projected_cash_left:,.0f} {cur}")

    if decision.data_warnings:
        print("\nData warnings:")
        for w in decision.data_warnings:
            print(f"  [{w.severity.upper()}] {w.ticker}: {w.message}")

    active_orders = p.orders[p.orders["action"].isin(["SELL", "BUY"])]
    if not active_orders.empty:
        print("\nOrdered Execution Ticket (execute SELLs first, then BUYs):")
        cols = ["action", "name", "sector", "trade_shares", "price_local", "currency",
                f"trade_value_{cur}", f"est_fee_{cur}", "post_weight", "reason"]
        print(active_orders[cols].to_string(float_format=lambda x: f"{x:,.2f}"))
    else:
        print("\nNo BUY or SELL orders recommended this month.")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    brief_path = out / f"decision_{decision.as_of:%Y-%m}.md"
    orders_path = out / f"orders_{decision.as_of:%Y-%m}.csv"
    brief_path.write_text(brief, encoding="utf-8")
    p.orders.to_csv(orders_path)
    print(f"\nSaved brief  -> {brief_path.resolve()}")
    print(f"Saved orders -> {orders_path.resolve()}")

    if a.apply:
        if not a.portfolio:
            print("Warning: --apply ignored because --portfolio path was not provided.")
        else:
            updated = apply_execution_plan(state, p)
            save_portfolio(updated, a.portfolio)
            print(f"Updated portfolio state saved to {Path(a.portfolio).resolve()}")


def cmd_recommend(a) -> None:
    cfg = _config(a)
    holdings = None
    extra = None
    if a.holdings:
        st = load_portfolio(a.holdings)
        holdings = st.held_shares()
        extra = list(holdings)
    data = _data(a, cfg, extra_tickers=extra)
    rec = recommend_portfolio(data, cfg, budget=a.budget, current_holdings=holdings)
    cur = rec.base_currency
    print(f"\nPortfolio for {rec.as_of:%B %Y} (prices as of {rec.price_date:%Y-%m-%d}), "
          f"budget {rec.budget:,.0f} {cur}\n")
    if rec.data_warnings:
        print("Data warnings:")
        for w in rec.data_warnings:
            print(f"  [{w.severity.upper()}] {w.ticker}: {w.message}")
        print()
    cols = ["name", "country", "sector", "weight", "price_local", "currency", "shares", f"invested_{cur}"]
    t = rec.table[cols].copy()
    t["weight"] = (t["weight"] * 100).round(1).astype(str) + "%"
    print(t.to_string(float_format=lambda x: f"{x:,.2f}"))
    print(f"\nUninvested cash: {rec.cash_left:,.0f} {cur}")
    if rec.trades is not None:
        print("\nTrades vs current holdings:")
        print(rec.trades[rec.trades["action"] != "HOLD"].to_string())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec.table.to_csv(out / f"portfolio_{rec.as_of:%Y-%m}.csv")
    print(f"\nSaved to {out.resolve() / f'portfolio_{rec.as_of:%Y-%m}.csv'}")


def cmd_init_portfolio(a) -> None:
    if a.from_csv:
        st = portfolio_from_csv(a.from_csv, broker_preset=a.broker, base_currency=a.currency, cash_sek=a.cash)
    else:
        st = PortfolioState(broker_preset=a.broker, base_currency=a.currency,
                            cash_sek=a.cash if a.currency == "SEK" else 0.0,
                            cash_usd=a.cash if a.currency == "USD" else 0.0)
    saved = save_portfolio(st, a.file)
    print(f"Created portfolio state file at {saved.resolve()} ({len(st.holdings)} holdings, {a.broker})")


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(prog="fpo", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Flagship monthly decision command
    d = sub.add_parser("decide", help="generate this month's optimal portfolio action plan & order ticket")
    _add_common(d)
    d.set_defaults(broker="avanza_mini")
    d.add_argument("--portfolio", help="path to portfolio.json (or legacy holdings.csv)")
    d.add_argument("--cash", type=float, default=0.0, help="existing uninvested cash if no portfolio file")
    d.add_argument("--deposit", type=float, default=10_000.0, help="new monthly contribution (base currency)")
    d.add_argument("--mode", default="smart_dca", choices=REBALANCE_MODES,
                   help="smart_dca (deploy cash first, avoid forced sells) or full rebalance")
    d.add_argument("--strategy", default="optimal", choices=["optimal", "single"],
                   help="optimal (walk-forward meta-strategy) or single (--model/--optimizer)")
    d.add_argument("--selection-mode", default="blend", choices=["blend", "winner"])
    d.add_argument("--lookback-months", type=int, default=36)
    d.add_argument("--min-order-sek", type=float, default=1500.0)
    d.add_argument("--no-trade-band", type=float, default=0.015)
    d.add_argument("--backtest-start", default="2012-01-31")
    d.add_argument("--fast", action="store_true", help="use faster 5-strategy candidate set")
    d.add_argument("--apply", action="store_true", help="record orders and update --portfolio file")
    d.set_defaults(func=cmd_decide)

    # Multi-asset Funds & UCITS ETFs tactical command
    t = sub.add_parser("tactical", help="multi-asset tactical allocation across Nordnet/Avanza funds & UCITS ETFs")
    _add_common(t)
    t.set_defaults(broker="nordnet_mini", start="2018-01-01")
    t.add_argument("--budget", type=float, default=550_000.0, help="core fund/ETF budget (base currency)")
    t.add_argument("--exec-mode", default="mutual_fund", choices=["mutual_fund", "etf"],
                   help="mutual_fund (0% fee on Swedish fund proxies) or etf (broker+FX fees)")
    t.add_argument("--selection-mode", default="blend", choices=["blend", "winner"])
    t.add_argument("--lookback-months", type=int, default=24)
    t.add_argument("--backtest-start", default=None)
    t.add_argument("--with-satellite", type=float, default=0.0,
                   help="optional active single-stock satellite budget (base currency) to combine")
    t.add_argument("--fast", action="store_true")
    t.set_defaults(func=cmd_tactical)

    # Walk-forward meta-strategy backtest
    m = sub.add_parser("meta-backtest", help="walk-forward backtest of optimal meta-strategy selection")
    _add_common(m)
    m.set_defaults(broker="avanza_mini")
    m.add_argument("--backtest-start", default="2012-01-31")
    m.add_argument("--selection-mode", default="blend", choices=["blend", "winner"])
    m.add_argument("--lookback-months", type=int, default=36)
    m.add_argument("--fast", action="store_true", help="use faster 5-strategy candidate set")
    m.set_defaults(func=cmd_meta_backtest)

    # Single-strategy backtest
    b = sub.add_parser("backtest", help="walk-forward backtest vs index benchmarks")
    _add_common(b)
    b.add_argument("--backtest-start", default="2012-01-31")
    b.set_defaults(func=cmd_backtest)

    # Simple single-strategy recommend
    r = sub.add_parser("recommend", help="portfolio to buy this month (single strategy)")
    _add_common(r)
    r.add_argument("--budget", type=float, default=100_000, help="amount to invest (base currency)")
    r.add_argument("--holdings", help="CSV (ticker,shares) or portfolio.json of what you own now")
    r.set_defaults(func=cmd_recommend)

    # Initialize portfolio.json
    ip = sub.add_parser("init-portfolio", help="create a persistent portfolio.json state file")
    ip.add_argument("--file", default="my_portfolio.json", help="output JSON path")
    ip.add_argument("--from-csv", help="optional existing ticker,shares CSV to import")
    ip.add_argument("--broker", default="avanza_mini", choices=list(BROKER_PRESETS))
    ip.add_argument("--currency", default="SEK", choices=["SEK", "USD"])
    ip.add_argument("--cash", type=float, default=0.0, help="starting uninvested cash")
    ip.set_defaults(func=cmd_init_portfolio)

    a = parser.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
