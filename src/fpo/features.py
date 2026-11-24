"""Feature engineering on month-end or live rebalance dates.

Every feature for rebalance date t is computed strictly from prices/OHLCV on or
before t using exact trading-day lookbacks (21/63/126/252 days), so features
remain consistent on completed month-ends as well as live mid-month runs.
The label (`fwd_ret`) is the return from completed month-end t to the next
completed month-end and is used exclusively for training on *past* dates and
evaluation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from fpo.data import MarketData, month_end_prices
from fpo.universe import ticker_country, ticker_sector

FEATURES = [
    "mom_1m", "mom_3m", "mom_6m", "mom_12m", "mom_12_1",
    "mom_12_1_sec_rel", "mom_6m_cty_rel",
    "vol_1m", "vol_3m", "vol_12m", "downside_vol_3m", "parkinson_vol_3m",
    "skew_3m", "max_dd_6m", "dist_52w_high", "trend_200d", "reversal_1w",
    "beta_12m", "idio_vol_12m", "pos_months_12m",
    "dollar_vol_trend", "amihud_3m", "is_se",
]

FEATURE_DESCRIPTIONS = {
    "mom_1m": "1-month (21d) return (short-term reversal)",
    "mom_3m": "3-month (63d) return",
    "mom_6m": "6-month (126d) return",
    "mom_12m": "12-month (252d) return",
    "mom_12_1": "12-month return skipping the last month (classic momentum)",
    "mom_12_1_sec_rel": "Sector-relative 12-1 momentum (stock vs sector median)",
    "mom_6m_cty_rel": "Country-relative 6m momentum (stock vs country median)",
    "vol_1m": "1-month realised volatility",
    "vol_3m": "3-month realised volatility",
    "vol_12m": "12-month realised volatility",
    "downside_vol_3m": "3-month downside volatility",
    "parkinson_vol_3m": "3-month Parkinson high-low range volatility",
    "skew_3m": "3-month skewness of daily returns",
    "max_dd_6m": "Max drawdown over 6 months",
    "dist_52w_high": "Distance from 52-week high",
    "trend_200d": "Price vs 200-day moving average",
    "reversal_1w": "1-week (5d) return",
    "beta_12m": "Beta to the equal-weight universe (12m)",
    "idio_vol_12m": "Idiosyncratic volatility (12m)",
    "pos_months_12m": "Share of positive 21-day months in last 12",
    "dollar_vol_trend": "Dollar-volume trend (21d vs 126d liquidity flow)",
    "amihud_3m": "3-month Amihud illiquidity (|return| / dollar volume)",
    "is_se": "Swedish stock (1) vs US (0)",
}

MIN_DAILY_OBS = 200


def _max_drawdown(p: np.ndarray) -> np.ndarray:
    running_max = np.fmax.accumulate(p, axis=0)
    dd = p / running_max - 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmin(dd, axis=0)


def _group_demeaned(values: np.ndarray, groups: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Subtract within-group median among eligible stocks (falls back to overall median)."""
    out = np.full_like(values, np.nan, dtype=float)
    elig_vals = values[eligible]
    overall_med = np.nanmedian(elig_vals) if np.any(np.isfinite(elig_vals)) else 0.0
    for g in np.unique(groups[eligible]):
        m = eligible & (groups == g)
        gv = values[m]
        fin = gv[np.isfinite(gv)]
        med = np.median(fin) if len(fin) >= 2 else overall_med
        out[m] = values[m] - med
    return out


def build_features(prices: pd.DataFrame, rebalance_dates: pd.DatetimeIndex | None = None,
                   *, high: pd.DataFrame | None = None, low: pd.DataFrame | None = None,
                   volume: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a long DataFrame indexed by (date, ticker) with raw features and `fwd_ret`.

    prices: daily prices in base currency.
    high, low, volume: optional aligned OHLCV DataFrames for range-volatility & liquidity features.
    """
    prices = prices.sort_index()
    monthly = month_end_prices(prices, completed_only=True)
    if rebalance_dates is None:
        rebalance_dates = monthly.index
    fwd = monthly.shift(-1) / monthly - 1.0   # label: next completed month-end return

    daily_ret = prices.pct_change(fill_method=None)
    dvals = prices.values
    rvals = daily_ret.values
    didx = prices.index
    cols = prices.columns
    is_se = np.array([1.0 if ticker_country(c) == "SE" else 0.0 for c in cols])
    countries = np.array([ticker_country(c) for c in cols])
    sectors = np.array([ticker_sector(c) for c in cols])

    hvals = high.reindex(index=didx, columns=cols).values if high is not None else None
    lvals = low.reindex(index=didx, columns=cols).values if low is not None else None
    vvals = volume.reindex(index=didx, columns=cols).values if volume is not None else None
    park_const = 4.0 * np.log(2.0)

    rows = []
    for t in rebalance_dates:
        pos = didx.searchsorted(t, side="right") - 1
        if pos < 252:
            continue
        p_now = dvals[pos]
        win_p = dvals[pos - 251: pos + 1]
        win_r = rvals[pos - 251: pos + 1]
        n_obs = np.sum(~np.isnan(win_r), axis=0)
        p_12m = dvals[pos - 252]
        eligible = (~np.isnan(p_now)) & (n_obs >= MIN_DAILY_OBS) & (~np.isnan(p_12m)) & (p_12m > 0)
        if eligible.sum() < 5:
            continue

        r = np.where(np.isnan(win_r), 0.0, win_r)
        ann = np.sqrt(252)
        f = {}
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            p_1m = dvals[pos - 21]
            p_3m = dvals[pos - 63]
            p_6m = dvals[pos - 126]
            f["mom_1m"] = p_now / p_1m - 1.0
            f["mom_3m"] = p_now / p_3m - 1.0
            f["mom_6m"] = p_now / p_6m - 1.0
            f["mom_12m"] = p_now / p_12m - 1.0
            f["mom_12_1"] = p_1m / p_12m - 1.0
            f["mom_12_1_sec_rel"] = _group_demeaned(f["mom_12_1"], sectors, eligible)
            f["mom_6m_cty_rel"] = _group_demeaned(f["mom_6m"], countries, eligible)

            f["vol_1m"] = r[-21:].std(axis=0) * ann
            f["vol_3m"] = r[-63:].std(axis=0) * ann
            f["vol_12m"] = r.std(axis=0) * ann
            neg = np.minimum(r[-63:], 0.0)
            f["downside_vol_3m"] = np.sqrt((neg ** 2).mean(axis=0)) * ann

            if hvals is not None and lvals is not None:
                h3 = hvals[pos - 62: pos + 1]
                l3 = lvals[pos - 62: pos + 1]
                hl = np.where((h3 > 0) & (l3 > 0) & (h3 >= l3), np.log(h3 / l3) ** 2, np.nan)
                f["parkinson_vol_3m"] = np.sqrt(np.nanmean(hl, axis=0) / park_const * 252.0)
            else:
                f["parkinson_vol_3m"] = np.full(len(cols), np.nan)

            r3 = r[-63:]
            sd3 = r3.std(axis=0)
            f["skew_3m"] = ((r3 - r3.mean(axis=0)) ** 3).mean(axis=0) / sd3 ** 3
            f["max_dd_6m"] = _max_drawdown(win_p[-126:])
            f["dist_52w_high"] = p_now / np.nanmax(win_p, axis=0) - 1.0
            f["trend_200d"] = p_now / np.nanmean(win_p[-200:], axis=0) - 1.0
            f["reversal_1w"] = p_now / dvals[pos - 5] - 1.0

            mkt = r[:, eligible].mean(axis=1)
            mkt_c = mkt - mkt.mean()
            denom = (mkt_c ** 2).sum()
            beta = ((r - r.mean(axis=0)) * mkt_c[:, None]).sum(axis=0) / denom if denom > 0 else np.zeros(len(cols))
            resid = r - beta[None, :] * mkt[:, None]
            f["beta_12m"] = beta
            f["idio_vol_12m"] = resid.std(axis=0) * ann

            # 12 non-overlapping 21-day windows up to pos
            block_idx = np.arange(pos - 252, pos + 1, 21)
            block_p = dvals[block_idx]
            block_ret = block_p[1:] / block_p[:-1] - 1.0
            f["pos_months_12m"] = np.nanmean((block_ret > 0).astype(float), axis=0)

            if vvals is not None:
                v6 = vvals[pos - 125: pos + 1]
                p6 = dvals[pos - 125: pos + 1]
                dv = np.where((v6 > 0) & np.isfinite(p6), v6 * p6, np.nan)
                adv_1m = np.nanmean(dv[-21:], axis=0)
                adv_6m = np.nanmean(dv, axis=0)
                f["dollar_vol_trend"] = np.log1p(np.maximum(adv_1m, 0.0)) - np.log1p(np.maximum(adv_6m, 0.0))
                dv3 = dv[-63:]
                illiq = np.abs(r[-63:]) / np.where(dv3 > 0, dv3, np.nan)
                f["amihud_3m"] = np.log1p(np.nanmean(illiq, axis=0) * 1e9)
            else:
                f["dollar_vol_trend"] = np.full(len(cols), np.nan)
                f["amihud_3m"] = np.full(len(cols), np.nan)

        f["is_se"] = is_se

        df = pd.DataFrame(f, index=cols)
        if t in fwd.index:
            df["fwd_ret"] = fwd.loc[t].values
        else:
            df["fwd_ret"] = np.nan
        df = df[eligible]
        df.index.name = "ticker"
        df["date"] = t
        rows.append(df.reset_index())

    if not rows:
        raise ValueError("Not enough price history to build features")
    out = pd.concat(rows, ignore_index=True).set_index(["date", "ticker"]).sort_index()
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def build_features_from_data(data: MarketData, rebalance_dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Build features using prices plus OHLCV columns when present on `MarketData`."""
    return build_features(data.prices, rebalance_dates=rebalance_dates,
                          high=data.high, low=data.low, volume=data.volume)


def build_live_features(data: MarketData, as_of: pd.Timestamp | None = None,
                        completed_feats: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Return `(completed_month_feats, live_feats_at_as_of, resolved_as_of)`.

    Separates completed month-end training data from the live prediction snapshot so
    running mid-month or on the 1st of the month never contaminates historical labels.
    """
    hist_feats = completed_feats if completed_feats is not None else build_features_from_data(data)
    latest_date = pd.Timestamp(data.prices.index.max())
    live_date = min(pd.Timestamp(as_of), latest_date) if as_of is not None else latest_date
    live_feats = build_features_from_data(data, rebalance_dates=pd.DatetimeIndex([live_date]))
    return hist_feats, live_feats, live_date


def cross_sectional_ranks(feats: pd.DataFrame, columns: list[str] = FEATURES) -> pd.DataFrame:
    """Map each feature to its per-date percentile rank centred on 0 (range -0.5..0.5).

    Ranking makes features comparable across time (volatility regimes, currency
    moves) and robust to outliers. Missing values become 0 (the median).
    """
    present = [c for c in columns if c in feats.columns]
    ranked = feats[present].groupby(level="date").rank(pct=True) - 0.5
    if "is_se" in ranked.columns:
        ranked["is_se"] = feats["is_se"] - 0.5  # binary, keep as-is
    for c in columns:
        if c not in ranked.columns:
            ranked[c] = 0.0
    return ranked[columns].fillna(0.0)


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
