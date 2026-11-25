"""Tests for modern strategies (LightGBM XENDCG, Smooth Residual Mom, HERC, NCO),
speed optimizations, EUR multi-currency conversion, and multi-asset Funds/UCITS ETFs."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from fpo.backtest import run_backtest
from fpo.cli import main as cli_main
from fpo.config import BacktestConfig
from fpo.data import synthetic_etf_market_data, synthetic_market_data
from fpo.features import build_features_from_data
from fpo.meta import DEFAULT_CANDIDATES, run_meta_backtest
from fpo.optimizer import (
    herc_weights, nco_weights, optimise_weights, precompute_weekly_returns, shrunk_covariance,
)
from fpo.tactical import render_tactical_brief, run_tactical_backtest
from fpo.universe import ticker_country, ticker_currency

warnings.simplefilter("ignore")


@pytest.fixture(scope="module")
def stock_data():
    return synthetic_market_data("both", start="2010-01-01", end="2020-12-31", seed=21)


@pytest.fixture(scope="module")
def stock_feats(stock_data):
    return build_features_from_data(stock_data)


@pytest.fixture(scope="module")
def etf_data():
    return synthetic_etf_market_data(start="2015-01-01", end="2024-12-31", seed=23)


def test_modern_features_no_lookahead(stock_data):
    """resid_mom_12_1 and fip_12_1 must never change when future prices are shocked."""
    t = pd.Timestamp("2017-06-30")
    base = build_features_from_data(stock_data, pd.DatetimeIndex([t]))
    shocked = stock_data.prices.copy()
    shocked.loc[shocked.index > t] *= 1.35
    after = build_features_from_data(
        type(stock_data)(
            prices=shocked, local_prices=stock_data.local_prices, benchmarks=stock_data.benchmarks,
            fx=stock_data.fx, base_currency=stock_data.base_currency, names=stock_data.names,
            source=stock_data.source, high=stock_data.high, low=stock_data.low, volume=stock_data.volume,
        ),
        pd.DatetimeIndex([t]),
    )
    for col in ("resid_mom_12_1", "fip_12_1"):
        assert base[col].notna().any()
        pd.testing.assert_series_equal(base[col], after[col])


@pytest.mark.parametrize("opt", ["herc", "nco"])
def test_herc_and_nco_valid_constrained_weights(stock_data, stock_feats, opt):
    t = stock_feats.index.get_level_values("date").unique()[-5]
    scores = stock_feats.xs(t, level="date")["resid_mom_12_1"]
    rets = stock_data.prices.pct_change(fill_method=None).loc[:t]
    wk = precompute_weekly_returns(rets)
    cfg = BacktestConfig(
        optimizer=opt, n_holdings=15, max_weight=0.12, min_weight=0.02,
        max_country_weight=0.70, max_sector_weight=0.35,
    )
    w = optimise_weights(scores, rets, cfg, weekly_returns=wk)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= 0).all() and w.max() <= 0.12 + 1e-5
    us_w = sum(v for k, v in w.items() if ticker_country(k) == "US")
    assert us_w <= 0.70 + 1e-4


def test_herc_and_nco_direct_functions():
    rng = np.random.default_rng(42)
    r = pd.DataFrame(rng.normal(0.0004, 0.015, (260, 10)), columns=[f"A{i}" for i in range(10)],
                     index=pd.bdate_range("2024-01-01", periods=260))
    cov = shrunk_covariance(r, freq="weekly")
    w_herc = herc_weights(cov, returns=r)
    w_nco = nco_weights(cov, mu=np.linspace(0.05, -0.02, 10), risk_aversion=5.0)
    assert abs(w_herc.sum() - 1.0) < 1e-9 and (w_herc > 0).all()
    assert abs(w_nco.sum() - 1.0) < 1e-9 and (w_nco >= 0).all()


@pytest.mark.parametrize("model_name", ["smooth_mom", "lgbm_rank"])
def test_modern_models_walk_forward_backtest(stock_data, stock_feats, model_name):
    cfg = BacktestConfig(
        model=model_name, optimizer="herc", backtest_start="2016-01-31",
        broker_preset="nordnet_mini", n_random_portfolios=30, retrain_every=6,
    )
    res = run_backtest(stock_data, cfg, features=stock_feats)
    assert len(res.returns) >= 40
    assert np.allclose(res.weights.sum(axis=1), 1.0)
    assert res.returns.notna().all()


def test_modern_meta_backtest_fast_with_ensemble_reuse(stock_data, stock_feats):
    cfg = BacktestConfig(
        backtest_start="2016-01-31", broker_preset="nordnet_mini",
        n_holdings=12, max_country_weight=0.70, max_sector_weight=0.35,
        n_random_portfolios=30, retrain_every=6,
    )
    meta = run_meta_backtest(stock_data, cfg, candidates=DEFAULT_CANDIDATES,
                             features=stock_feats, lookback_months=24)
    assert "Meta-Strategy (Walk-Forward)" in meta.leaderboard.index
    assert "Smooth Residual Mom + HERC" in meta.leaderboard.index
    assert "LightGBM XENDCG + HERC" in meta.leaderboard.index
    assert "Modern Ensemble + NCO" in meta.leaderboard.index
    assert np.allclose(meta.strategy_weights.sum(axis=1), 1.0)


def test_etf_eur_currency_and_tactical_backtest(etf_data):
    assert ticker_currency("EUNL.DE") == "EUR"
    assert ticker_currency("XACT-SVERIGE.ST") == "SEK"
    assert ticker_currency("IB01.L") == "USD"
    assert etf_data.fx_eur is not None

    cfg = BacktestConfig(broker_preset="nordnet_mini", rf_annual=0.025)
    tac_fund = run_tactical_backtest(etf_data, cfg=cfg, execution_mode="mutual_fund", budget_base=550_000.0)
    tac_etf = run_tactical_backtest(etf_data, cfg=cfg, execution_mode="etf", budget_base=550_000.0)

    assert "Tactical Meta (Funds/ETFs)" in tac_fund.leaderboard.index
    assert tac_fund.live_regime in ("BULL", "SIDEWAYS", "BEAR")
    assert abs(tac_fund.live_table["weight"].sum() - 1.0) < 1e-6
    assert "fund_proxy" in tac_fund.live_table.columns
    # 0%-fee Swedish mutual fund mode must have strictly lower annual cost than full ETF fees
    assert tac_fund.meta_costs.mean() <= tac_etf.meta_costs.mean() + 1e-9

    brief = render_tactical_brief(tac_fund, core_budget=550_000.0)
    assert "# Multi-Asset Fund & UCITS ETF Allocation" in brief
    assert tac_fund.live_regime in brief


def test_cli_tactical_smoke(tmp_path):
    outdir = tmp_path / "tac_out"
    cli_main([
        "tactical", "--synthetic", "--start", "2010-01-01",
        "--budget", "550000", "--exec-mode", "mutual_fund",
        "--with-satellite", "150000", "--fast",
        "--out", str(outdir),
    ])
    briefs = list(outdir.glob("tactical_*.md"))
    assert len(briefs) == 1
    assert "Active Single-Stock Satellite Sleeve" in briefs[0].read_text(encoding="utf-8")
    assert len(list(outdir.glob("tactical_allocation_*.csv"))) == 1
