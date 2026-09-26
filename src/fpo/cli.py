"""Command line interface.

    fpo backtest  [--model ensemble] [--optimizer mean_variance] [--synthetic]
    fpo recommend --budget 50000 [--holdings holdings.csv]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from fpo import metrics as M
from fpo.backtest import EW_UNIVERSE, run_backtest
from fpo.config import MODELS, OPTIMIZERS, BacktestConfig
from fpo.data import load_market_data, month_end_prices, synthetic_market_data
from fpo.recommend import recommend_portfolio


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--markets", default="both", choices=["us", "se", "both"])
    p.add_argument("--currency", default="SEK", choices=["SEK", "USD"])
    p.add_argument("--start", default="2005-01-01", help="first date of price history")
    p.add_argument("--model", default="ensemble", choices=MODELS)
    p.add_argument("--optimizer", default="mean_variance", choices=OPTIMIZERS)
    p.add_argument("--n-holdings", type=int, default=15)
    p.add_argument("--max-weight", type=float, default=0.12)
    p.add_argument("--max-country-weight", type=float, default=1.0)
    p.add_argument("--cost-bps", type=float, default=15.0)
    p.add_argument("--synthetic", action="store_true", help="use simulated data (offline demo)")
    p.add_argument("--out", default="outputs", help="folder for CSV outputs")


def _config(a) -> BacktestConfig:
    return BacktestConfig(
        markets=a.markets, base_currency=a.currency, start=a.start, model=a.model,
        optimizer=a.optimizer, n_holdings=a.n_holdings, max_weight=a.max_weight,
        max_country_weight=a.max_country_weight, cost_bps=a.cost_bps,
        backtest_start=getattr(a, "backtest_start", "2012-01-31"),
    )


def _data(a, cfg):
    if a.synthetic:
        return synthetic_market_data(cfg.markets, start=cfg.start, base_currency=cfg.base_currency)
    return load_market_data(cfg.markets, start=cfg.start, base_currency=cfg.base_currency)


def _data_report(data, out: Path) -> None:
    """Save data-cleaning corrections and show the largest monthly moves as a sanity check."""
    out.mkdir(parents=True, exist_ok=True)
    q = data.quality
    if q is not None and len(q):
        q.to_csv(out / "data_quality.csv", index=False)
        print(f"\nData cleaning: {len(q)} corrections to Yahoo prices (saved to data_quality.csv)")
        print(q.groupby("issue").size().to_string())
    m = month_end_prices(data.prices).pct_change(fill_method=None).stack()
    big = m.abs().sort_values(ascending=False).head(8).index
    print("\nLargest single-stock monthly moves after cleaning (check these look real):")
    for d, t in big:
        print(f"  {t:12s} {d:%Y-%m}  {m[(d, t)]:+.0%}")


def cmd_backtest(a) -> None:
    cfg = _config(a)
    data = _data(a, cfg)
    _data_report(data, Path(a.out))
    res = run_backtest(data, cfg, progress=lambda p, msg: print(f"[{p:>4.0%}] {msg}"))
    pd.set_option("display.width", 200)
    table = M.comparison_table(res.returns, res.benchmarks, ref=EW_UNIVERSE)
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


def cmd_recommend(a) -> None:
    cfg = _config(a)
    data = _data(a, cfg)
    holdings = None
    if a.holdings:
        h = pd.read_csv(a.holdings)
        holdings = dict(zip(h["ticker"], h["shares"]))
    rec = recommend_portfolio(data, cfg, budget=a.budget, current_holdings=holdings)
    cur = rec.base_currency
    print(f"\nPortfolio for {rec.as_of:%B %Y} (prices as of {rec.price_date:%Y-%m-%d}), "
          f"budget {rec.budget:,.0f} {cur}\n")
    cols = ["name", "country", "weight", "price_local", "currency", "shares", f"invested_{cur}"]
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


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(prog="fpo", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest", help="walk-forward backtest vs index benchmarks")
    _add_common(b)
    b.add_argument("--backtest-start", default="2012-01-31")
    b.set_defaults(func=cmd_backtest)
    r = sub.add_parser("recommend", help="portfolio to buy this month")
    _add_common(r)
    r.add_argument("--budget", type=float, default=100_000, help="amount to invest (base currency)")
    r.add_argument("--holdings", help="CSV with columns ticker,shares of what you own now")
    r.set_defaults(func=cmd_recommend)
    a = parser.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
