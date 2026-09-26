import warnings

import numpy as np
import pandas as pd
import pytest

from fpo import BacktestConfig, recommend_portfolio, run_backtest
from fpo import metrics as M
from fpo.data import synthetic_market_data
from fpo.features import build_features
from fpo.models import training_dates
from fpo.optimizer import apply_caps, hrp_weights, optimise_weights, select_holdings
from fpo.recommend import allocate_whole_shares

warnings.simplefilter("ignore")


@pytest.fixture(scope="module")
def data():
    return synthetic_market_data("both", start="2008-01-01", end="2020-12-31", seed=1)


@pytest.fixture(scope="module")
def feats(data):
    return build_features(data.prices)


# --------------------------------------------------------------------------- look-ahead
def test_features_do_not_use_future_prices(data):
    """Changing prices AFTER date t must not change features AT date t."""
    t = pd.Timestamp("2015-06-30")
    base = build_features(data.prices, pd.DatetimeIndex([t]))
    shocked = data.prices.copy()
    shocked.loc[shocked.index > t] *= np.random.default_rng(0).uniform(0.5, 1.5, shocked.loc[shocked.index > t].shape)
    after = build_features(shocked, pd.DatetimeIndex([t]))
    cols = [c for c in base.columns if c != "fwd_ret"]
    pd.testing.assert_frame_equal(base[cols], after[cols])
    assert not np.allclose(base["fwd_ret"].values, after["fwd_ret"].values)  # label does change


def test_training_dates_strictly_before_prediction():
    dates = pd.date_range("2010-01-31", periods=60, freq="ME")
    cfg = BacktestConfig(embargo_months=1, train_window_months=24)
    t = dates[40]
    td = training_dates(dates, t, cfg)
    assert td.max() < t
    assert td.max() == dates[38]  # embargo drops dates[39]
    assert len(td) == 24


def test_no_signal_no_edge():
    """With zero predictable signal the model's out-of-sample IC must be ~0 (catches leakage)."""
    d = synthetic_market_data("us", start="2006-01-01", end="2018-12-31", seed=3, signal_strength=0.0)
    res = run_backtest(d, BacktestConfig(markets="us", model="ridge", backtest_start="2010-01-31",
                                         n_random_portfolios=50))
    t_stat = res.ic.mean() / (res.ic.std() / np.sqrt(len(res.ic)))
    assert abs(t_stat) < 3


def test_strong_signal_is_found():
    d = synthetic_market_data("us", start="2006-01-01", end="2018-12-31", seed=3, signal_strength=4.0)
    res = run_backtest(d, BacktestConfig(markets="us", model="gbm", backtest_start="2010-01-31",
                                         n_random_portfolios=50))
    assert res.ic.mean() > 0.08
    assert M.cagr(res.returns) > M.cagr(res.benchmarks["Equal-weight universe"])


# --------------------------------------------------------------------------- optimiser
def test_caps_respected():
    tickers = [f"U{i}" for i in range(8)] + [f"S{i}.ST" for i in range(4)]
    w = apply_caps(np.arange(1, 13, dtype=float), tickers, max_weight=0.1, max_country=0.6)
    assert abs(w.sum() - 1) < 1e-9
    assert w.max() <= 0.1 + 1e-9
    assert w[:8].sum() <= 0.6 + 1e-6


@pytest.mark.parametrize("opt", ["mean_variance", "min_variance", "hrp", "inverse_vol", "score", "equal"])
def test_optimisers_valid_weights(data, feats, opt):
    t = feats.index.get_level_values("date").unique()[-10]
    scores = feats.xs(t, level="date")["mom_12_1"]
    rets = data.prices.pct_change(fill_method=None).loc[:t]
    cfg = BacktestConfig(optimizer=opt, n_holdings=10, max_weight=0.15)
    w = optimise_weights(scores, rets, cfg)
    assert len(w) <= 10
    assert abs(w.sum() - 1) < 1e-6
    assert (w >= 0).all() and w.max() <= 0.15 + 1e-6


def test_hold_buffer_keeps_existing():
    s = pd.Series(np.arange(20, 0, -1), index=[f"T{i}" for i in range(20)], dtype=float)
    chosen = select_holdings(s, 5, current=["T6"], buffer=1.5)  # T6 is rank 7 <= 7.5
    assert "T6" in chosen and len(chosen) == 5


def test_hrp_sums_to_one():
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.normal(0, 0.01, (300, 6)))
    w = hrp_weights(r.cov())
    assert abs(w.sum() - 1) < 1e-9 and (w > 0).all()


# --------------------------------------------------------------------------- metrics
def test_metrics_basic():
    r = pd.Series([0.01] * 24, index=pd.date_range("2020-01-31", periods=24, freq="ME"))
    assert M.cagr(r) == pytest.approx(1.01 ** 12 - 1)
    assert M.max_drawdown(r) == 0
    r2 = pd.Series([0.1, -0.5, 0.2])
    assert M.max_drawdown(r2) == pytest.approx(-0.5)


# --------------------------------------------------------------------------- end to end
def test_backtest_end_to_end(data, feats):
    cfg = BacktestConfig(model="ridge", backtest_start="2012-01-31", n_random_portfolios=50)
    res = run_backtest(data, cfg, features=feats)
    assert len(res.returns) > 50
    assert res.returns.index.equals(res.benchmarks.index)
    assert np.allclose(res.weights.sum(axis=1), 1)
    assert (res.costs >= 0).all()
    assert (res.gross_returns >= res.returns - 1e-12).all()
    eq = res.equity_curves()
    assert "Strategy" in eq and eq.notna().all().all()


def test_recommendation(data, feats):
    rec = recommend_portfolio(data, BacktestConfig(model="ridge"), budget=200_000, features=feats,
                              current_holdings={"AAPL": 3})
    assert abs(rec.table["weight"].sum() - 1) < 1e-6
    assert rec.cash_left >= 0
    assert rec.trades is not None and "AAPL" in rec.trades.index


def test_whole_shares_respect_cap():
    w = pd.Series({"A": 0.12, "B": 0.88})
    px = pd.Series({"A": 11_749.0, "B": 100.0})
    sh = allocate_whole_shares(w, px, 50_000, max_weight=0.12)
    assert sh["A"] == 0  # one share would be 23.5% of the budget


def test_clean_prices_spikes_and_splits():
    from fpo.data import clean_prices
    idx = pd.bdate_range("2020-01-01", periods=40)
    good = pd.Series(np.linspace(100, 110, 40), index=idx)
    spiky = good.copy()
    spiky.iloc[10] = 1000.0            # one-day bad print
    spiky.iloc[20:22] = 5.0            # two-day bad print
    split = good.copy()
    split.iloc[:30] *= 10              # unadjusted 10:1 split
    df = pd.DataFrame({"A": spiky, "B": split, "C": good})
    out, rep = clean_prices(df)
    assert out["A"].isna().sum() == 3
    assert out["A"].dropna().pct_change().abs().max() < 0.05
    assert np.allclose(out["B"], good, rtol=1e-9)
    pd.testing.assert_series_equal(out["C"], good, check_names=False)
    assert set(rep["ticker"]) == {"A", "B"}


def test_real_crash_is_kept():
    """A genuine one-way -60% move (no reversal) is not a split ratio and must be kept."""
    from fpo.data import clean_prices
    idx = pd.bdate_range("2020-01-01", periods=20)
    s = pd.Series([100.0] * 10 + [40.0] * 10, index=idx)
    out, rep = clean_prices(pd.DataFrame({"X": s}))
    pd.testing.assert_series_equal(out["X"], s, check_names=False)
    assert rep.empty


def test_whole_share_allocation():
    w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    px = pd.Series({"A": 90.0, "B": 45.0, "C": 700.0})
    sh = allocate_whole_shares(w, px, 1000)
    spent = sum(sh[k] * px[k] for k in sh)
    assert spent <= 1000
    assert sh["C"] == 0  # 700 per share vs a 200 target: buying one would overshoot badly
    for k in sh:  # every holding lands within one share of its target amount
        assert abs(sh[k] * px[k] - w[k] * 1000) <= px[k]
