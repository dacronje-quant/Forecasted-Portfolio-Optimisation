"""Market data: download from Yahoo Finance (with on-disk cache) or simulate.

All prices are dividend/split adjusted closes. Everything the backtester sees is
converted into a single base currency (SEK by default) using USDSEK, so a
Swedish investor's currency exposure is part of the measured returns.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from fpo.universe import (
    BENCHMARKS, FX_TICKER, all_names, get_universe, ticker_country, ticker_currency,
)

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data_cache")


@dataclass
class MarketData:
    prices: pd.DataFrame          # daily adjusted close, base currency, columns = stocks
    local_prices: pd.DataFrame    # daily adjusted close, trading currency
    benchmarks: pd.DataFrame      # daily benchmark levels, base currency
    fx: pd.Series                 # SEK per USD
    base_currency: str = "SEK"
    names: dict[str, str] = field(default_factory=dict)
    source: str = "yahoo"
    quality: pd.DataFrame | None = None   # data-cleaning corrections applied

    @property
    def tickers(self) -> list[str]:
        return list(self.prices.columns)

    def country(self, ticker: str) -> str:
        return ticker_country(ticker)

    def currency(self, ticker: str) -> str:
        return ticker_currency(ticker)


# --------------------------------------------------------------------------- #
# Yahoo Finance
# --------------------------------------------------------------------------- #
def _cache_path(cache_dir: Path, tickers: list[str], start: str) -> Path:
    key = hashlib.md5((",".join(sorted(tickers)) + start).encode()).hexdigest()[:12]
    # CSV (not pickle) so the cache is portable across pandas/pyarrow versions and machines
    return cache_dir / f"prices_{key}.csv.gz"


def download_prices(tickers: list[str], start: str, cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
                    max_age_hours: float = 20.0) -> pd.DataFrame:
    """Download daily adjusted closes for `tickers` (local currency). Cached on disk."""
    cache_file = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = _cache_path(cache_dir, tickers, start)
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < max_age_hours * 3600:
            log.info("Loading prices from cache %s", cache_file)
            return _read_cache(cache_file)

    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise ImportError("yfinance is required for live data: pip install yfinance") from e

    log.info("Downloading %d tickers from Yahoo Finance...", len(tickers))
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False,
                      group_by="column", threads=True)
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance returned no data (network blocked or rate limited?)")
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:  # single ticker
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    close = close.dropna(how="all", axis=1).sort_index()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    missing = sorted(set(tickers) - set(close.columns))
    if missing:
        log.warning("No data for: %s", ", ".join(missing))
    close = close.astype("float64")
    close.columns = [str(c) for c in close.columns]
    if cache_file is not None:
        close.to_csv(cache_file)
        return _read_cache(cache_file)  # same numbers whether the data came fresh or from cache
    return close


def _read_cache(path: Path) -> pd.DataFrame:
    # round_trip parsing: the default fast parser can be off in the last digit, which is enough
    # to change later rankings and make fresh and cached runs disagree
    return pd.read_csv(path, index_col=0, parse_dates=True, float_precision="round_trip")


def clean_prices(prices: pd.DataFrame, spike: float = 0.40, revert_days: int = 5,
                 split_ratio: float = 3.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repair common Yahoo Finance data errors in daily (local-currency) prices.

    1. Non-positive prices -> removed.
    2. Spikes: a jump of more than `spike` (in log terms, either direction) that reverts to
       within 35% of the jump size within `revert_days` days is a bad print; those prices are removed.
    3. Unadjusted splits: a one-day move of more than `split_ratio`x (or 1/x) that does not
       revert is treated as a missing split adjustment; earlier prices are rescaled.

    Returns (cleaned prices, report of every correction).
    """
    out = prices.copy()
    report = []
    thr = np.log1p(spike)
    split_thr = np.log(split_ratio)
    for col in out.columns:
        s = out[col]
        bad0 = s <= 0
        if bad0.any():
            report += [{"ticker": col, "date": d, "issue": "non-positive price", "move": np.nan}
                       for d in s.index[bad0]]
            s = s.mask(bad0)
        v = s.dropna()
        if len(v) < 3:
            out[col] = s
            continue
        lp = np.log(v.values)
        idx = v.index
        keep = np.ones(len(lp), dtype=bool)
        prev = 0  # index of the last good price
        i = 1
        while i < len(lp):
            jump = lp[i] - lp[prev]
            if abs(jump) > thr:
                end = None
                for j in range(i + 1, min(i + 1 + revert_days, len(lp))):
                    if abs(lp[j] - lp[prev]) < 0.35 * abs(jump):
                        end = j
                        break
                if end is not None:
                    keep[i:end] = False
                    report.append({"ticker": col, "date": idx[i], "issue": f"spike removed ({end - i} day(s))",
                                   "move": float(np.expm1(jump))})
                    prev, i = end, end + 1
                    continue
                if abs(jump) > split_thr:
                    # snap to the nearest whole split ratio (e.g. 10:1) when close, so the day's
                    # genuine price change is preserved; otherwise remove the whole jump
                    k = round(float(np.exp(abs(jump))))
                    adj = np.sign(jump) * np.log(k) if abs(np.exp(abs(jump)) / k - 1) < 0.08 else jump
                    lp[:i] += adj  # rescale history so the level is continuous
                    report.append({"ticker": col, "date": idx[i], "issue": "unadjusted split fixed",
                                   "move": float(np.expm1(jump))})
            prev = i
            i += 1
        clean = pd.Series(np.where(keep, np.exp(lp), np.nan), index=idx)
        out[col] = clean.reindex(s.index)
    rep = pd.DataFrame(report, columns=["ticker", "date", "issue", "move"])
    return out, rep


def _to_base(local: pd.DataFrame, fx: pd.Series, base_currency: str) -> pd.DataFrame:
    """Convert local-currency prices to the base currency. fx = SEK per USD."""
    out = local.copy()
    for t in out.columns:
        cur = ticker_currency(t)
        if cur == base_currency:
            continue
        if cur == "USD" and base_currency == "SEK":
            out[t] = out[t] * fx
        elif cur == "SEK" and base_currency == "USD":
            out[t] = out[t] / fx
    return out


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    # Different exchange holidays: forward-fill short gaps only (never before the first price
    # and not across long suspensions / delistings).
    return df.ffill(limit=5)


def load_market_data(markets: str = "both", start: str = "2005-01-01", base_currency: str = "SEK",
                     cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
                     extra_tickers: list[str] | None = None) -> MarketData:
    """Download universe + benchmarks + FX from Yahoo and convert to base currency."""
    stocks = list(get_universe(markets))
    if extra_tickers:
        stocks += [t for t in extra_tickers if t not in stocks]
    everything = stocks + list(BENCHMARKS) + [FX_TICKER]
    raw = download_prices(everything, start, cache_dir)

    if FX_TICKER not in raw.columns:
        raise RuntimeError("Could not download USDSEK exchange rate")
    # Business-day calendar (union of US + SE trading days)
    idx = raw.index[raw.index.dayofweek < 5]
    raw = raw.loc[idx]
    raw, quality = clean_prices(raw)
    if len(quality):
        log.warning("Data cleaning: %d corrections (see data_quality report)", len(quality))
    fx = raw[FX_TICKER].ffill().bfill()

    local = _clean(raw[[t for t in stocks if t in raw.columns]])
    bench_local = _clean(raw[[t for t in BENCHMARKS if t in raw.columns]])
    prices = _to_base(local, fx, base_currency)
    bench = _to_base(bench_local, fx, base_currency)
    return MarketData(prices=prices, local_prices=local, benchmarks=bench, fx=fx,
                      base_currency=base_currency, names=all_names(), source="yahoo", quality=quality)


# --------------------------------------------------------------------------- #
# Synthetic data (offline demo / tests)
# --------------------------------------------------------------------------- #
def synthetic_market_data(markets: str = "both", start: str = "2005-01-01", end: str = "2025-12-31",
                          base_currency: str = "SEK", seed: int = 7,
                          signal_strength: float = 1.0) -> MarketData:
    """Simulate a realistic-ish market with a *known, weak* predictable component.

    Each stock has a slowly-varying latent expected return (AR(1) at monthly
    frequency). Momentum-type features can partially recover it, so a working
    ML pipeline should find a small edge — and with signal_strength=0 it should
    find none (a useful sanity check against look-ahead bias).
    """
    rng = np.random.default_rng(seed)
    universe = get_universe(markets)
    tickers = list(universe)
    n = len(tickers)
    dates = pd.bdate_range(start, end)
    T = len(dates)
    is_se = np.array([ticker_country(t) == "SE" for t in tickers])

    # Factors (daily)
    us_mkt = rng.normal(0.0004, 0.011, T)
    se_mkt = 0.6 * us_mkt + rng.normal(0.0002, 0.009, T)
    fx_ret = rng.normal(0.0, 0.006, T) - 0.25 * us_mkt   # SEK tends to weaken in risk-off
    # Occasional crashes to create realistic drawdowns
    for _ in range(4):
        s = rng.integers(0, T - 60)
        us_mkt[s:s + 40] -= 0.006
        se_mkt[s:s + 40] -= 0.007

    beta = rng.uniform(0.6, 1.4, n)
    idio_vol = rng.uniform(0.012, 0.028, n)

    # Latent monthly alpha, AR(1), switched daily at month boundaries
    months = dates.to_period("M")
    month_codes, month_idx = np.unique(months.asi8, return_inverse=True)
    n_months = len(month_codes)
    alpha_m = np.zeros((n_months, n))
    a = rng.normal(0, 0.01, n)
    for m in range(n_months):
        a = 0.92 * a + rng.normal(0, 0.004, n)
        alpha_m[m] = a
    alpha_d = alpha_m[month_idx] / 21.0 * signal_strength

    mkt = np.where(is_se[None, :], se_mkt[:, None], us_mkt[:, None])
    rets = beta[None, :] * mkt + alpha_d + rng.normal(0, 1, (T, n)) * idio_vol[None, :]
    rets = np.clip(rets, -0.5, 0.5)
    local = pd.DataFrame(100 * np.exp(np.cumsum(np.log1p(rets), axis=0)), index=dates, columns=tickers)

    # Some stocks list late, one gets delisted -> exercises missing-data handling
    for t in rng.choice(tickers, size=max(1, n // 10), replace=False):
        local.loc[: dates[rng.integers(200, T // 2)], t] = np.nan
    dead = rng.choice(tickers)
    local.loc[dates[int(T * 0.8)]:, dead] = np.nan

    fx = pd.Series(7.0 * np.exp(np.cumsum(fx_ret)), index=dates, name=FX_TICKER)
    bench_local = pd.DataFrame({
        "SPY": 100 * np.exp(np.cumsum(np.log1p(us_mkt + 0.00005))),
        "URTH": 100 * np.exp(np.cumsum(np.log1p(0.7 * us_mkt + 0.3 * se_mkt))),
        "^OMX": 100 * np.exp(np.cumsum(np.log1p(se_mkt))),
    }, index=dates)
    prices = _to_base(local, fx, base_currency)
    bench = _to_base(bench_local, fx, base_currency)
    names = {**universe, **BENCHMARKS}
    return MarketData(prices=prices, local_prices=local, benchmarks=bench, fx=fx,
                      base_currency=base_currency, names=names, source="synthetic")


def month_end_prices(daily: pd.DataFrame) -> pd.DataFrame:
    """Last available price in each calendar month (indexed by month-end timestamp)."""
    return daily.resample("ME").last()
