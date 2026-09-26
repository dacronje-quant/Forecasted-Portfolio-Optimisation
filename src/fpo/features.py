"""Feature engineering on month-end dates.

Every feature for rebalance date t is computed from prices on or before t
only. The label (`fwd_ret`) is the return from t to the next month end and is
used exclusively for training on *past* dates and for evaluation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from fpo.data import month_end_prices
from fpo.universe import ticker_country

FEATURES = [
    "mom_1m", "mom_3m", "mom_6m", "mom_12m", "mom_12_1",
    "vol_1m", "vol_3m", "vol_12m", "downside_vol_3m", "skew_3m",
    "max_dd_6m", "dist_52w_high", "trend_200d", "reversal_1w",
    "beta_12m", "idio_vol_12m", "pos_months_12m", "is_se",
]

FEATURE_DESCRIPTIONS = {
    "mom_1m": "1-month return (short-term reversal)",
    "mom_3m": "3-month return",
    "mom_6m": "6-month return",
    "mom_12m": "12-month return",
    "mom_12_1": "12-month return skipping the last month (classic momentum)",
    "vol_1m": "1-month realised volatility",
    "vol_3m": "3-month realised volatility",
    "vol_12m": "12-month realised volatility",
    "downside_vol_3m": "3-month downside volatility",
    "skew_3m": "3-month skewness of daily returns",
    "max_dd_6m": "Max drawdown over 6 months",
    "dist_52w_high": "Distance from 52-week high",
    "trend_200d": "Price vs 200-day moving average",
    "reversal_1w": "1-week return",
    "beta_12m": "Beta to the equal-weight universe (12m)",
    "idio_vol_12m": "Idiosyncratic volatility (12m)",
    "pos_months_12m": "Share of positive months in last 12",
    "is_se": "Swedish stock (1) vs US (0)",
}

MIN_DAILY_OBS = 200


def _max_drawdown(p: np.ndarray) -> np.ndarray:
    running_max = np.fmax.accumulate(p, axis=0)
    dd = p / running_max - 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmin(dd, axis=0)


def build_features(prices: pd.DataFrame, rebalance_dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Return a long DataFrame indexed by (date, ticker) with raw features and `fwd_ret`.

    prices: daily prices in base currency.
    """
    prices = prices.sort_index()
    monthly = month_end_prices(prices)
    if rebalance_dates is None:
        rebalance_dates = monthly.index
    mret = monthly.pct_change(fill_method=None)
    fwd = monthly.shift(-1) / monthly - 1.0   # label: next-month return

    daily_ret = prices.pct_change(fill_method=None)
    dvals = prices.values
    rvals = daily_ret.values
    didx = prices.index
    cols = prices.columns
    is_se = np.array([1.0 if ticker_country(c) == "SE" else 0.0 for c in cols])

    rows = []
    for t in rebalance_dates:
        pos = didx.searchsorted(t, side="right") - 1
        if pos < 260:
            continue
        mpos = monthly.index.get_loc(t)
        if mpos < 13:
            continue
        p_now = dvals[pos]
        win_p = dvals[pos - 251: pos + 1]
        win_r = rvals[pos - 251: pos + 1]
        n_obs = np.sum(~np.isnan(win_r), axis=0)
        m = monthly.values
        eligible = (~np.isnan(p_now)) & (n_obs >= MIN_DAILY_OBS) & (~np.isnan(m[mpos - 12]))
        if eligible.sum() < 5:
            continue

        r = np.where(np.isnan(win_r), 0.0, win_r)
        ann = np.sqrt(252)
        f = {}
        f["mom_1m"] = m[mpos] / m[mpos - 1] - 1
        f["mom_3m"] = m[mpos] / m[mpos - 3] - 1
        f["mom_6m"] = m[mpos] / m[mpos - 6] - 1
        f["mom_12m"] = m[mpos] / m[mpos - 12] - 1
        f["mom_12_1"] = m[mpos - 1] / m[mpos - 12] - 1
        f["vol_1m"] = r[-21:].std(axis=0) * ann
        f["vol_3m"] = r[-63:].std(axis=0) * ann
        f["vol_12m"] = r.std(axis=0) * ann
        neg = np.minimum(r[-63:], 0.0)
        f["downside_vol_3m"] = np.sqrt((neg ** 2).mean(axis=0)) * ann
        r3 = r[-63:]
        sd3 = r3.std(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns of ineligible stocks
            f["skew_3m"] = ((r3 - r3.mean(axis=0)) ** 3).mean(axis=0) / sd3 ** 3
            f["max_dd_6m"] = _max_drawdown(win_p[-126:])
            f["dist_52w_high"] = p_now / np.nanmax(win_p, axis=0) - 1
            f["trend_200d"] = p_now / np.nanmean(win_p[-200:], axis=0) - 1
            f["reversal_1w"] = p_now / dvals[pos - 5] - 1
            mkt = r[:, eligible].mean(axis=1)
            mkt_c = mkt - mkt.mean()
            beta = ((r - r.mean(axis=0)) * mkt_c[:, None]).sum(axis=0) / (mkt_c ** 2).sum()
            resid = r - beta[None, :] * mkt[:, None]
            f["beta_12m"] = beta
            f["idio_vol_12m"] = resid.std(axis=0) * ann
            f["pos_months_12m"] = (mret.values[mpos - 11: mpos + 1] > 0).mean(axis=0)
        f["is_se"] = is_se

        df = pd.DataFrame(f, index=cols)
        df["fwd_ret"] = fwd.values[mpos]
        df = df[eligible]
        df.index.name = "ticker"
        df["date"] = t
        rows.append(df.reset_index())

    if not rows:
        raise ValueError("Not enough price history to build features")
    out = pd.concat(rows, ignore_index=True).set_index(["date", "ticker"]).sort_index()
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def cross_sectional_ranks(feats: pd.DataFrame, columns: list[str] = FEATURES) -> pd.DataFrame:
    """Map each feature to its per-date percentile rank centred on 0 (range -0.5..0.5).

    Ranking makes features comparable across time (volatility regimes, currency
    moves) and robust to outliers. Missing values become 0 (the median).
    """
    ranked = feats[columns].groupby(level="date").rank(pct=True) - 0.5
    ranked["is_se"] = feats["is_se"] - 0.5  # binary, keep as-is
    return ranked.fillna(0.0)


def make_target(feats: pd.DataFrame, kind: str = "rank") -> pd.Series:
    """Label for supervised learning (per-date cross-sectional)."""
    g = feats["fwd_ret"].groupby(level="date")
    if kind == "rank":
        y = g.rank(pct=True) - 0.5
    elif kind == "excess":
        y = feats["fwd_ret"] - g.transform("median")
        y = y.clip(-0.5, 0.5)
    else:
        raise ValueError(kind)
    return y.where(feats["fwd_ret"].notna())
