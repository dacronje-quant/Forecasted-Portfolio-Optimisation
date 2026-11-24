"""Tests for live reliability fixes, walk-forward meta-strategy, and Avanza/Nordnet execution."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from fpo import metrics as M
from fpo.cli import main as cli_main
from fpo.config import BROKER_PRESETS, BacktestConfig
from fpo.data import completed_month_ends, month_end_prices, synthetic_market_data, validate_market_data
from fpo.features import FEATURES, build_features_from_data, build_live_features
from fpo.meta import FAST_CANDIDATES, compute_meta_weights_at, run_meta_backtest
from fpo.optimizer import optimise_weights, shrunk_covariance
from fpo.portfolio import (
    Holding, PortfolioState, apply_execution_plan, build_execution_plan, load_portfolio, save_portfolio,
)
from fpo.recommend import recommend_optimal_portfolio, render_monthly_brief
from fpo.universe import ticker_country, ticker_sector

warnings.simplefilter("ignore")


@pytest.fixture(scope="module")
def synth_data():
    return synthetic_market_data("both", start="2008-01-01", end="2020-12-31", seed=11)


@pytest.fixture(scope="module")
def synth_feats(synth_data):
    return build_features_from_data(synth_data)


# --------------------------------------------------------------------------- Phase 1: Reliability fixes
def test_incomplete_month_excluded_and_live_features_clean():
    """Running on Dec 1 (1 trading day into Dec) must NOT treat Dec as a completed month-end
    or corrupt November's forward return or live 1-month momentum."""
    d = synthetic_market_data("us", start="2014-01-01", end="2020-12-01", seed=5)
    done = completed_month_ends(d.prices.index)
    assert done.max() == pd.Timestamp("2020-11-30")
    me = month_end_prices(d.prices, completed_only=True)
    assert me.index.max() == pd.Timestamp("2020-11-30")

    hist_feats, live_feats, live_date = build_live_features(d)
    assert live_date == pd.Timestamp("2020-12-01")
    assert hist_feats.index.get_level_values("date").max() == pd.Timestamp("2020-11-30")
    # Latest completed month (Nov 2020) must have NaN fwd_ret (Dec is not finished!)
    assert hist_feats.xs(pd.Timestamp("2020-11-30"), level="date")["fwd_ret"].isna().all()
    # Live features at Dec 1 must have NaN fwd_ret and realistic 21-day mom_1m (not a 1-day return)
    live_row = live_feats.xs(live_date, level="date")
    assert live_row["fwd_ret"].isna().all()
    one_day = (d.prices.iloc[-1] / d.prices.iloc[-2] - 1.0).reindex(live_row.index)
    assert not np.allclose(live_row["mom_1m"].values, one_day.values, atol=1e-4)
    # OHLCV and sector/country-neutral features should be populated
    for col in ("parkinson_vol_3m", "dollar_vol_trend", "amihud_3m", "mom_12_1_sec_rel", "mom_6m_cty_rel"):
        assert col in FEATURES and live_row[col].notna().any()


def test_country_and_sector_caps_strictly_respected_when_top_scores_concentrated(synth_data):
    """Even when the highest-scoring stocks are 100% US Technology, country and sector caps must hold."""
    tickers = synth_data.tickers
    scores = pd.Series(0.0, index=tickers, dtype=float)
    for t in tickers:
        if ticker_country(t) == "US" and ticker_sector(t) == "Technology":
            scores[t] = 10.0
        elif ticker_country(t) == "US":
            scores[t] = 5.0
        else:
            scores[t] = 1.0

    rets = synth_data.prices.pct_change(fill_method=None)
    cfg = BacktestConfig(
        optimizer="mean_variance", n_holdings=15, max_weight=0.12, min_weight=0.02,
        max_country_weight=0.60, max_sector_weight=0.30,
    )
    w = optimise_weights(scores, rets, cfg)
    assert abs(w.sum() - 1.0) < 1e-6
    assert w.max() <= 0.12 + 1e-5
    us_w = sum(v for k, v in w.items() if ticker_country(k) == "US")
    assert us_w <= 0.60 + 1e-4
    sec_totals = {}
    for k, v in w.items():
        sec_totals[ticker_sector(k)] = sec_totals.get(ticker_sector(k), 0.0) + v
    assert max(sec_totals.values()) <= 0.30 + 1e-4


def test_weekly_covariance_and_l1_turnover_penalty_reduces_turnover(synth_data, synth_feats):
    t = synth_feats.index.get_level_values("date").unique()[-5]
    scores = synth_feats.xs(t, level="date")["mom_12_1"]
    rets = synth_data.prices.pct_change(fill_method=None).loc[:t]

    cov_wk = shrunk_covariance(rets.iloc[-252:, :10], freq="weekly")
    assert cov_wk.shape == (10, 10)
    assert np.all(np.diag(cov_wk.values) > 0)

    # Start from an equal-weighted subset of the top 15
    top15 = list(scores.sort_values(ascending=False).index[:15])
    prev_w = pd.Series(1.0 / len(top15), index=top15)
    # Perturb scores slightly to provoke turnover
    rng = np.random.default_rng(42)
    scores_next = scores + pd.Series(rng.normal(0, 0.15, len(scores)), index=scores.index)

    cfg_free = BacktestConfig(optimizer="mean_variance", n_holdings=15, max_weight=0.15,
                              max_sector_weight=0.40, turnover_penalty=0.0, hold_buffer=1.5)
    cfg_pen = BacktestConfig(optimizer="mean_variance", n_holdings=15, max_weight=0.15,
                             max_sector_weight=0.40, turnover_penalty=0.03, hold_buffer=1.5)
    w_free = optimise_weights(scores_next, rets, cfg_free, prev_weights=prev_w)
    w_pen = optimise_weights(scores_next, rets, cfg_pen, prev_weights=prev_w)

    idx_f = w_free.index.union(prev_w.index)
    idx_p = w_pen.index.union(prev_w.index)
    to_free = (w_free.reindex(idx_f, fill_value=0.0) - prev_w.reindex(idx_f, fill_value=0.0)).abs().sum() / 2
    to_pen = (w_pen.reindex(idx_p, fill_value=0.0) - prev_w.reindex(idx_p, fill_value=0.0)).abs().sum() / 2
    assert to_pen < to_free


def test_data_hygiene_validator_catches_glitches(synth_data):
    d = synthetic_market_data("us", start="2018-01-01", end="2020-12-31", seed=9)
    # Inject stale ticker (>5 bdays NaN at end) and extreme +60% jump
    stale_tk = d.tickers[0]
    jump_tk = d.tickers[1]
    d.local_prices.iloc[-8:, d.local_prices.columns.get_loc(stale_tk)] = np.nan
    d.local_prices.iloc[-2, d.local_prices.columns.get_loc(jump_tk)] *= 1.65

    warns = validate_market_data(d, tickers=[stale_tk, jump_tk, "NONEXISTENT"])
    warn_tickers = {w.ticker for w in warns}
    assert stale_tk in warn_tickers
    assert jump_tk in warn_tickers
    assert "NONEXISTENT" in warn_tickers


# --------------------------------------------------------------------------- Phase 2: Optimal Meta-Strategy & DSR
def test_deflated_sharpe_penalizes_multiple_trials():
    rng = np.random.default_rng(123)
    r = pd.Series(rng.normal(0.012, 0.04, 84), index=pd.date_range("2015-01-31", periods=84, freq="ME"))
    psr = M.probabilistic_sharpe(r, sr_benchmark=0.0)
    dsr1 = M.deflated_sharpe(r, n_trials=1)
    dsr5 = M.deflated_sharpe(r, n_trials=5)
    dsr25 = M.deflated_sharpe(r, n_trials=25)
    dsr100 = M.deflated_sharpe(r, n_trials=100)
    assert dsr1 == pytest.approx(psr, rel=1e-9)
    assert dsr1 > dsr5 > dsr25 > dsr100


def test_meta_strategy_strictly_no_lookahead():
    """Shocking candidate returns AFTER date t must never change meta weights AT date t."""
    dates = pd.date_range("2015-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(7)
    ew = pd.Series(rng.normal(0.008, 0.04, len(dates)), index=dates)
    cands = {
        "A": pd.Series(rng.normal(0.010, 0.04, len(dates)), index=dates),
        "B": pd.Series(rng.normal(0.009, 0.035, len(dates)), index=dates),
        "C": pd.Series(rng.normal(0.007, 0.045, len(dates)), index=dates),
    }
    t = dates[40]
    w_before = compute_meta_weights_at(t, cands, ew, lookback_months=36, selection_mode="blend")

    # Violently shock all returns strictly after t
    cands_shocked = {k: v.copy() for k, v in cands.items()}
    cands_shocked["C"].loc[dates > t] += 0.25
    cands_shocked["A"].loc[dates > t] -= 0.25
    w_after = compute_meta_weights_at(t, cands_shocked, ew, lookback_months=36, selection_mode="blend")
    pd.testing.assert_series_equal(w_before, w_after)


# --------------------------------------------------------------------------- Phase 3: Broker Fees & Execution Engine
def test_broker_fee_schedules_avanza_and_nordnet():
    av = BROKER_PRESETS["avanza_mini"]
    nn = BROKER_PRESETS["nordnet_small"]
    # 5,000 SEK Swedish order on Avanza Mini: max(1, 12.5) + 0 FX + 2.5 spread = 15.0 SEK
    assert av.trade_cost(5_000.0, "SEK", "SEK") == pytest.approx(15.0)
    # 5,000 SEK US order on Avanza Mini: max(10, 12.5) + 12.5 FX + 2.5 spread = 27.5 SEK
    assert av.trade_cost(5_000.0, "USD", "SEK") == pytest.approx(27.5)
    # 5,000 SEK US order on Nordnet Small: max(39, 7.5) + 12.5 FX + 2.5 spread = 54.0 SEK
    assert nn.trade_cost(5_000.0, "USD", "SEK") == pytest.approx(54.0)


def test_smart_dca_deploys_10000_sek_without_forced_sells(synth_data):
    """Smart DCA with 10,000 SEK monthly deposit must not force-sell top-ranked holdings
    and must strictly respect cash safety after Avanza fees."""
    latest = synth_data.prices.index[-1]
    local_px = synth_data.local_prices.loc[latest].dropna()
    base_px = synth_data.prices.loc[latest].dropna()
    fx_now = float(synth_data.fx.loc[latest])

    top = list(base_px.index[:10])
    target_w = pd.Series(1.0 / len(top), index=top)
    all_scores = pd.DataFrame({
        "score": np.linspace(2.0, -2.0, len(base_px)),
        "rank": np.arange(1, len(base_px) + 1),
        "name": [synth_data.names.get(t, t) for t in base_px.index],
        "mom_12_1": 0.15,
    }, index=base_px.index)

    # Already hold first 3 stocks (~5,000 SEK each, well within max_weight) + deposit 10,000 SEK
    state = PortfolioState(
        broker_preset="avanza_mini", base_currency="SEK", cash_sek=0.0,
        holdings={
            t: Holding(shares=max(1, int(5_000.0 // float(base_px[t]))), avg_cost_local=float(local_px[t]))
            for t in top[:3]
        },
    )
    cfg = BacktestConfig(
        broker_preset="avanza_mini", rebalance_mode="smart_dca",
        n_holdings=10, max_weight=0.25, hold_buffer=1.5, min_order_sek=1500.0,
    )
    plan = build_execution_plan(
        target_weights=target_w, all_scores=all_scores, local_prices=local_px, base_prices=base_px,
        fx_rate=fx_now, state=state, cfg=cfg, as_of=latest, price_date=latest, deposit_base=10_000.0,
    )
    assert plan.sells.empty
    assert not plan.buys.empty
    assert plan.verdict == "DEPLOY CASH ONLY"
    assert plan.buy_notional + plan.buy_fees <= 10_000.0 + 1e-6
    assert plan.buy_notional >= 5_000.0  # deploys the majority of the 10,000 SEK deposit
    assert plan.projected_cash_left >= 0.0


def test_full_rebalance_sells_first_and_persists_state(synth_data, tmp_path):
    latest = synth_data.prices.index[-1]
    local_px = synth_data.local_prices.loc[latest].dropna()
    base_px = synth_data.prices.loc[latest].dropna()
    fx_now = float(synth_data.fx.loc[latest])

    top = list(base_px.index[:8])
    bad_stock = base_px.index[-1]  # worst rank -> must be sold!
    target_w = pd.Series(1.0 / len(top), index=top)
    all_scores = pd.DataFrame({
        "score": np.linspace(2.0, -2.0, len(base_px)),
        "rank": np.arange(1, len(base_px) + 1),
        "name": [synth_data.names.get(t, t) for t in base_px.index],
    }, index=base_px.index)

    state = PortfolioState(
        broker_preset="nordnet_mini", base_currency="SEK", cash_sek=2_000.0,
        holdings={bad_stock: Holding(shares=20, avg_cost_local=float(local_px[bad_stock]))},
    )
    cfg = BacktestConfig(broker_preset="nordnet_mini", rebalance_mode="full", n_holdings=8, min_order_sek=1000.0)
    plan = build_execution_plan(
        target_weights=target_w, all_scores=all_scores, local_prices=local_px, base_prices=base_px,
        fx_rate=fx_now, state=state, cfg=cfg, as_of=latest, price_date=latest, deposit_base=10_000.0,
    )
    assert bad_stock in plan.sells.index
    assert plan.verdict == "REBALANCE"
    assert plan.buy_notional + plan.buy_fees <= plan.starting_cash + plan.sell_proceeds_gross - plan.sell_fees + 1e-6

    updated = apply_execution_plan(state, plan)
    assert bad_stock not in updated.holdings
    assert len(updated.trade_log) >= 2
    pfile = tmp_path / "portfolio.json"
    save_portfolio(updated, pfile)
    reloaded = load_portfolio(pfile)
    assert reloaded.broker_preset == "nordnet_mini"
    assert len(reloaded.holdings) == len(updated.holdings)


# --------------------------------------------------------------------------- End-to-End Meta & CLI
def test_meta_backtest_and_optimal_decision_end_to_end(synth_data, synth_feats):
    cfg = BacktestConfig(
        backtest_start="2014-01-31", broker_preset="avanza_mini",
        n_holdings=12, max_country_weight=0.70, max_sector_weight=0.35,
        rebalance_mode="smart_dca", n_random_portfolios=30,
    )
    meta = run_meta_backtest(synth_data, cfg, candidates=FAST_CANDIDATES, features=synth_feats,
                             lookback_months=24, selection_mode="blend")
    assert "Meta-Strategy (Walk-Forward)" in meta.leaderboard.index
    assert "Deflated Sharpe" in meta.leaderboard.columns
    assert np.allclose(meta.strategy_weights.sum(axis=1), 1.0)
    assert meta.health.status in ("GREEN", "YELLOW", "RED")

    decision = recommend_optimal_portfolio(
        synth_data, cfg, state=PortfolioState(broker_preset="avanza_mini"),
        deposit_base=10_000.0, features=synth_feats, meta_result=meta,
        candidates=FAST_CANDIDATES, use_meta=True,
    )
    assert decision.plan.projected_cash_left >= 0.0
    assert decision.plan.buy_notional + decision.plan.buy_fees <= 10_000.0 + 1e-6
    brief = render_monthly_brief(decision)
    assert "# Monthly Portfolio Decision" in brief
    assert decision.plan.verdict in brief


def test_cli_decide_and_meta_backtest_smoke(tmp_path):
    pfile = tmp_path / "my_portfolio.json"
    outdir = tmp_path / "out"
    cli_main(["init-portfolio", "--file", str(pfile), "--broker", "avanza_mini", "--cash", "0"])
    assert pfile.exists()

    cli_main([
        "decide", "--synthetic", "--fast", "--portfolio", str(pfile),
        "--deposit", "10000", "--broker", "avanza_mini", "--mode", "smart_dca",
        "--start", "2012-01-01", "--backtest-start", "2018-01-31",
        "--out", str(outdir), "--apply",
    ])
    md_files = list(outdir.glob("decision_*.md"))
    csv_files = list(outdir.glob("orders_*.csv"))
    assert len(md_files) == 1 and len(csv_files) == 1
    reloaded = load_portfolio(pfile)
    assert len(reloaded.trade_log) >= 1

    cli_main([
        "meta-backtest", "--synthetic", "--fast",
        "--start", "2012-01-01", "--backtest-start", "2018-01-31",
        "--out", str(outdir),
    ])
    assert (outdir / "meta_leaderboard.csv").exists()
    assert (outdir / "meta_strategy_weights.csv").exists()
