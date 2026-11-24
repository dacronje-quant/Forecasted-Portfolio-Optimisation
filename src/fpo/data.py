"""Market data: download OHLCV from Yahoo Finance (with fast incremental parquet cache), validate hygiene, or simulate.

All prices are dividend/split adjusted closes. Everything the backtester sees is
converted into a single base currency (SEK by default) using USDSEK and EURSEK,
so a Swedish investor's currency exposure across US, European UCITS ETFs, and
Swedish equities is part of the measured returns.
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
    BENCHMARKS, ETF_UNIVERSE, EUR_FX_TICKER, FX_TICKER, all_names, get_universe,
    ticker_country, ticker_currency, ticker_sector,
)

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data_cache")


@dataclass
class DataWarning:
    severity: str       # 'warning' or 'error'
    ticker: str
    message: str


@dataclass
class MarketData:
    prices: pd.DataFrame          # daily adjusted close, base currency, columns = assets
    local_prices: pd.DataFrame    # daily adjusted close, trading currency
    benchmarks: pd.DataFrame      # daily benchmark levels, base currency
    fx: pd.Series                 # SEK per USD
    base_currency: str = "SEK"
    names: dict[str, str] = field(default_factory=dict)
    source: str = "yahoo"
    high: pd.DataFrame | None = None     # daily adjusted high, local currency
    low: pd.DataFrame | None = None      # daily adjusted low, local currency
    volume: pd.DataFrame | None = None   # daily share volume
    fx_eur: pd.Series | None = None      # SEK per EUR (for European UCITS ETFs)

    @property
    def tickers(self) -> list[str]:
        return list(self.prices.columns)

    def country(self, ticker: str) -> str:
        return ticker_country(ticker)

    def currency(self, ticker: str) -> str:
        return ticker_currency(ticker)

    def sector(self, ticker: str) -> str:
        return ticker_sector(ticker)


# --------------------------------------------------------------------------- #
# Month-end calendar & data hygiene
# --------------------------------------------------------------------------- #
def _last_weekday_of_month(month_end: pd.Timestamp) -> pd.Timestamp:
    """Last Mon-Fri date of the calendar month containing `month_end`."""
    d = month_end.normalize()
    if d.day_of_week == 5:  # Saturday
        return d - pd.Timedelta(days=1)
    if d.day_of_week == 6:  # Sunday
        return d - pd.Timedelta(days=2)
    return d


def completed_month_ends(daily_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return calendar month-end timestamps that are fully completed in `daily_index`.

    Prevents an unfinished current month (e.g. running on Dec 1 or Nov 24) from
    being treated as a completed month-end and corrupting momentum or forward-return labels.
    """
    if len(daily_index) == 0:
        return pd.DatetimeIndex([])
    idx = pd.DatetimeIndex(daily_index).sort_values()
    last_obs = idx[-1].normalize()
    me = pd.Series(1, index=idx).resample("ME").last().dropna().index
    if len(me) == 0:
        return me
    if last_obs < _last_weekday_of_month(me[-1]):
        me = me[:-1]
    return me


def month_end_prices(daily: pd.DataFrame, completed_only: bool = True) -> pd.DataFrame:
    """Last available price in each calendar month (indexed by month-end timestamp).

    When `completed_only=True` (default), drops an unfinished trailing calendar month
    so mid-month or 1st-of-month runs never distort monthly returns or training labels.
    """
    monthly = daily.resample("ME").last()
    if completed_only and len(daily) > 0:
        done = completed_month_ends(daily.index)
        monthly = monthly.reindex(done)
    return monthly


def validate_market_data(data: MarketData, tickers: list[str] | None = None,
                         max_stale_bdays: int = 3, jump_threshold: float = 0.40) -> list[DataWarning]:
    """Run live data-hygiene checks before generating real trade orders.

    Flags stale prices, missing latest quotes, missing FX, and extreme single-day
    jumps (>40% in the trailing month, which often indicate unadjusted splits or bad ticks).
    """
    warnings_out: list[DataWarning] = []
    if data.prices.empty:
        return [DataWarning("error", "ALL", "Market price table is empty.")]

    latest = data.prices.index.max()
    # FX freshness check
    fx_valid = data.fx.dropna()
    if fx_valid.empty or not np.isfinite(fx_valid.iloc[-1]) or fx_valid.iloc[-1] <= 0:
        warnings_out.append(DataWarning("error", FX_TICKER, "Missing or invalid USD/SEK exchange rate."))
    else:
        fx_lag = len(pd.bdate_range(fx_valid.index[-1], latest)) - 1
        if fx_lag > max_stale_bdays:
            warnings_out.append(DataWarning(
                "warning", FX_TICKER,
                f"USD/SEK rate is stale ({fx_lag} business days old, last {fx_valid.index[-1]:%Y-%m-%d}).",
            ))

    check_tickers = list(tickers) if tickers is not None else list(data.prices.columns)
    recent_rets = data.local_prices.pct_change(fill_method=None).iloc[-21:]
    for t in check_tickers:
        if t not in data.local_prices.columns:
            warnings_out.append(DataWarning("error", t, "Ticker missing from downloaded price data."))
            continue
        s = data.local_prices[t].dropna()
        if s.empty:
            warnings_out.append(DataWarning("error", t, "No valid price history available."))
            continue
        last_t = s.index[-1]
        lag = len(pd.bdate_range(last_t, latest)) - 1
        if lag > max_stale_bdays:
            warnings_out.append(DataWarning(
                "warning", t,
                f"Price is stale ({lag} business days behind {latest:%Y-%m-%d}, last quote {last_t:%Y-%m-%d}).",
            ))
        if t in recent_rets.columns:
            max_move = recent_rets[t].abs().max()
            if pd.notna(max_move) and max_move >= jump_threshold:
                warnings_out.append(DataWarning(
                    "warning", t,
                    f"Extreme single-day move ({max_move:.0%}) in last 21d — verify split/quote before trading.",
                ))
    return warnings_out


# --------------------------------------------------------------------------- #
# Yahoo Finance (with fast incremental parquet cache & SQLite lock retry)
# --------------------------------------------------------------------------- #
def _cache_key(tickers: list[str], start: str, kind: str = "ohlcv") -> str:
    return hashlib.md5((",".join(sorted(tickers)) + "|" + start + "|" + kind).encode()).hexdigest()[:12]


def _extract_field(raw: pd.DataFrame, field_name: str, tickers: list[str]) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        if field_name not in raw.columns.get_level_values(0):
            return pd.DataFrame(index=raw.index)
        df = raw[field_name]
    else:
        if field_name not in raw.columns:
            return pd.DataFrame(index=raw.index)
        df = raw[[field_name]].rename(columns={field_name: tickers[0]})
    df = df.dropna(how="all", axis=1).sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def _yf_fetch(tickers: list[str], start: str) -> dict[str, pd.DataFrame]:
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise ImportError("yfinance is required for live data: pip install yfinance") from e

    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False,
                      group_by="column", threads=True)
    if raw is None or raw.empty:
        raw = yf.download(tickers, start=start, auto_adjust=True, progress=False,
                          group_by="column", threads=False)
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance returned no data (network blocked or rate limited?)")

    out = {k: _extract_field(raw, k, tickers) for k in ("Close", "High", "Low", "Volume")}
    # Retry any tickers dropped by transient SQLite tz-cache lock with threads=False
    missing = sorted(set(tickers) - set(out["Close"].columns))
    if missing and len(missing) <= 15:
        try:
            retry_raw = yf.download(missing, start=start, auto_adjust=True, progress=False,
                                    group_by="column", threads=False)
            if retry_raw is not None and not retry_raw.empty:
                for k in ("Close", "High", "Low", "Volume"):
                    add = _extract_field(retry_raw, k, missing)
                    if not add.empty:
                        out[k] = out[k].combine_first(add).sort_index()
        except Exception:
            pass
    return out


def _save_ohlcv_cache(cache_dir: Path, key: str, ohlcv: dict[str, pd.DataFrame]) -> None:
    sub = cache_dir / f"ohlcv_{key}"
    sub.mkdir(parents=True, exist_ok=True)
    for k, df in ohlcv.items():
        try:
            df.to_parquet(sub / f"{k}.parquet")
        except Exception:
            df.to_pickle(sub / f"{k}.pkl")


def _load_ohlcv_cache(cache_dir: Path, key: str) -> tuple[dict[str, pd.DataFrame] | None, float]:
    sub = cache_dir / f"ohlcv_{key}"
    if sub.exists() and (sub / "Close.parquet").exists():
        mtime = (sub / "Close.parquet").stat().st_mtime
        out = {}
        for k in ("Close", "High", "Low", "Volume"):
            p = sub / f"{k}.parquet"
            out[k] = pd.read_parquet(p) if p.exists() else pd.DataFrame()
        return out, mtime
    # Legacy single-pickle cache support
    legacy = cache_dir / f"ohlcv_{key}.pkl"
    if legacy.exists():
        return pd.read_pickle(legacy), legacy.stat().st_mtime
    return None, 0.0


def download_ohlcv(tickers: list[str], start: str, cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
                   max_age_hours: float = 20.0, incremental_max_days: int = 14) -> dict[str, pd.DataFrame]:
    """Download daily adjusted Close, High, Low, Volume for `tickers` (local currency).

    Uses fast parquet caching with automatic incremental tail refresh (fetching only
    the last ~10 days when cache is < 14 days old, unless a split adjustment is detected).
    """
    key = _cache_key(tickers, start, kind="ohlcv")
    cached = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached, mtime = _load_ohlcv_cache(cache_dir, key)
        if cached is not None and not cached["Close"].empty:
            age_hours = (time.time() - mtime) / 3600.0
            if age_hours < max_age_hours:
                log.info("Loading OHLCV from cache (%s, age %.1fh)", key, age_hours)
                return cached
            # Fast incremental tail update when cache is recent (< 14 days old)
            if age_hours < incremental_max_days * 24.0:
                last_dt = pd.Timestamp(cached["Close"].index.max())
                tail_start = (last_dt - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
                try:
                    log.info("Incremental OHLCV refresh from %s (%d tickers)...", tail_start, len(tickers))
                    tail = _yf_fetch(tickers, tail_start)
                    # Sanity check overlap for retroactive splits/dividend adjustments
                    overlap = cached["Close"].index.intersection(tail["Close"].index)
                    split_detected = False
                    if len(overlap) > 0:
                        common_cols = cached["Close"].columns.intersection(tail["Close"].columns)
                        old_ov = cached["Close"].loc[overlap, common_cols]
                        new_ov = tail["Close"].loc[overlap, common_cols]
                        rel_diff = ((new_ov - old_ov).abs() / old_ov.replace(0, np.nan)).max().max()
                        if pd.notna(rel_diff) and rel_diff > 0.05:
                            split_detected = True
                            log.info("Detected split/adjustment (%.1f%%); performing full refresh.", rel_diff * 100)
                    if not split_detected:
                        merged = {}
                        for k in ("Close", "High", "Low", "Volume"):
                            merged[k] = tail[k].combine_first(cached[k]).sort_index()
                        _save_ohlcv_cache(cache_dir, key, merged)
                        return merged
                except Exception as e:
                    log.warning("Incremental refresh failed (%s); falling back to full download.", e)

    log.info("Downloading %d tickers (OHLCV) from Yahoo Finance...", len(tickers))
    out = _yf_fetch(tickers, start)
    missing = sorted(set(tickers) - set(out["Close"].columns))
    if missing:
        log.warning("No data for: %s", ", ".join(missing))
    if cache_dir is not None:
        _save_ohlcv_cache(cache_dir, key, out)
    return out


def download_prices(tickers: list[str], start: str, cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
                    max_age_hours: float = 20.0) -> pd.DataFrame:
    """Download daily adjusted closes for `tickers` (local currency). Cached on disk."""
    return download_ohlcv(tickers, start, cache_dir, max_age_hours)["Close"]


def _to_base(local: pd.DataFrame, fx_usd: pd.Series, base_currency: str,
             fx_eur: pd.Series | None = None) -> pd.DataFrame:
    """Convert local-currency (SEK/USD/EUR) prices to `base_currency`."""
    out = local.copy()
    for t in out.columns:
        cur = ticker_currency(t)
        if cur == base_currency:
            continue
        if cur == "USD" and base_currency == "SEK":
            out[t] = out[t] * fx_usd
        elif cur == "SEK" and base_currency == "USD":
            out[t] = out[t] / fx_usd
        elif cur == "EUR":
            if fx_eur is None:
                raise RuntimeError(f"EURSEK exchange rate required to convert {t}")
            if base_currency == "SEK":
                out[t] = out[t] * fx_eur
            elif base_currency == "USD":
                out[t] = out[t] * (fx_eur / fx_usd)
    return out


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    # Different exchange holidays: forward-fill short gaps only (never before the first price
    # and not across long suspensions / delistings).
    return df.ffill(limit=5)


def load_market_data(markets: str = "both", start: str = "2005-01-01", base_currency: str = "SEK",
                     cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
                     extra_tickers: list[str] | None = None) -> MarketData:
    """Download universe + benchmarks + FX (OHLCV) from Yahoo and convert to base currency."""
    assets = list(get_universe(markets))
    if extra_tickers:
        assets += [t for t in extra_tickers if t not in assets]
    need_eur = any(ticker_currency(t) == "EUR" for t in assets)
    fx_list = [FX_TICKER] + ([EUR_FX_TICKER] if need_eur else [])
    everything = list(dict.fromkeys(assets + list(BENCHMARKS) + fx_list))
    ohlcv = download_ohlcv(everything, start, cache_dir)
    raw = ohlcv["Close"]

    if FX_TICKER not in raw.columns:
        raise RuntimeError("Could not download USDSEK exchange rate")
    idx = raw.index[raw.index.dayofweek < 5]
    raw = raw.loc[idx]
    fx_usd = raw[FX_TICKER].ffill().bfill()
    fx_eur = raw[EUR_FX_TICKER].ffill().bfill() if (need_eur and EUR_FX_TICKER in raw.columns) else None

    avail_assets = [t for t in assets if t in raw.columns]
    local = _clean(raw[avail_assets])
    bench_local = _clean(raw[[t for t in BENCHMARKS if t in raw.columns]])
    prices = _to_base(local, fx_usd, base_currency, fx_eur=fx_eur)
    bench = _to_base(bench_local, fx_usd, base_currency, fx_eur=fx_eur)

    def _align_opt(df: pd.DataFrame) -> pd.DataFrame | None:
        cols = [t for t in avail_assets if t in df.columns]
        if not cols:
            return None
        return df.reindex(index=idx, columns=avail_assets)

    high = _align_opt(ohlcv["High"])
    low = _align_opt(ohlcv["Low"])
    vol = _align_opt(ohlcv["Volume"])
    if high is not None:
        high = _clean(high)
    if low is not None:
        low = _clean(low)
    if vol is not None:
        vol = vol.fillna(0.0)

    return MarketData(
        prices=prices, local_prices=local, benchmarks=bench, fx=fx_usd, fx_eur=fx_eur,
        base_currency=base_currency, names=all_names(), source="yahoo",
        high=high, low=low, volume=vol,
    )


def load_etf_market_data(start: str = "2018-01-01", base_currency: str = "SEK",
                         cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
                         extra_tickers: list[str] | None = None) -> MarketData:
    """Download the PRIIPs-compliant UCITS ETF & Swedish mutual fund proxy universe."""
    return load_market_data(markets="etf", start=start, base_currency=base_currency,
                            cache_dir=cache_dir, extra_tickers=extra_tickers)


# --------------------------------------------------------------------------- #
# Synthetic data (offline demo / tests)
# --------------------------------------------------------------------------- #
def synthetic_market_data(markets: str = "both", start: str = "2005-01-01", end: str = "2025-12-31",
                          base_currency: str = "SEK", seed: int = 7,
                          signal_strength: float = 1.0) -> MarketData:
    """Simulate a realistic-ish market with a *known, weak* predictable component."""
    if markets.lower() == "etf":
        return synthetic_etf_market_data(start=start, end=end, base_currency=base_currency, seed=seed)

    rng = np.random.default_rng(seed)
    universe = get_universe(markets)
    tickers = list(universe)
    n = len(tickers)
    dates = pd.bdate_range(start, end)
    T = len(dates)
    is_se = np.array([ticker_country(t) == "SE" for t in tickers])

    us_mkt = rng.normal(0.0004, 0.011, T)
    se_mkt = 0.6 * us_mkt + rng.normal(0.0002, 0.009, T)
    fx_ret = rng.normal(0.0, 0.006, T) - 0.25 * us_mkt
    for _ in range(4):
        s = rng.integers(0, max(1, T - 60))
        us_mkt[s:s + 40] -= 0.006
        se_mkt[s:s + 40] -= 0.007

    beta = rng.uniform(0.6, 1.4, n)
    idio_vol = rng.uniform(0.012, 0.028, n)

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
    local_arr = 100 * np.exp(np.cumsum(np.log1p(rets), axis=0))
    local = pd.DataFrame(local_arr, index=dates, columns=tickers)

    intraday = np.abs(rng.normal(0.0, 1.0, (T, n))) * (idio_vol[None, :] * 0.8 + np.abs(rets) * 0.4)
    high = pd.DataFrame(local_arr * (1.0 + intraday), index=dates, columns=tickers)
    low = pd.DataFrame(local_arr / (1.0 + intraday), index=dates, columns=tickers)
    base_vol = rng.uniform(2e5, 5e6, n)
    vol_arr = base_vol[None, :] * np.exp(rng.normal(0.0, 0.35, (T, n)))
    volume = pd.DataFrame(vol_arr, index=dates, columns=tickers)

    for t in rng.choice(tickers, size=max(1, n // 10), replace=False):
        cut = dates[rng.integers(200, max(201, T // 2))]
        local.loc[:cut, t] = np.nan
        high.loc[:cut, t] = np.nan
        low.loc[:cut, t] = np.nan
        volume.loc[:cut, t] = np.nan
    dead = rng.choice(tickers)
    cut_dead = dates[int(T * 0.8)]
    local.loc[cut_dead:, dead] = np.nan
    high.loc[cut_dead:, dead] = np.nan
    low.loc[cut_dead:, dead] = np.nan
    volume.loc[cut_dead:, dead] = np.nan

    fx = pd.Series(9.5 * np.exp(np.cumsum(fx_ret)), index=dates, name=FX_TICKER)
    se_div_daily = 0.035 / 252.0
    bench_local = pd.DataFrame({
        "SPY": 100 * np.exp(np.cumsum(np.log1p(us_mkt + 0.00005))),
        "URTH": 100 * np.exp(np.cumsum(np.log1p(0.7 * us_mkt + 0.3 * se_mkt))),
        "XACT-OMXS30.ST": 100 * np.exp(np.cumsum(np.log1p(se_mkt + se_div_daily))),
        "^OMX": 100 * np.exp(np.cumsum(np.log1p(se_mkt))),
    }, index=dates)
    prices = _to_base(local, fx, base_currency)
    bench = _to_base(bench_local, fx, base_currency)
    names = {**universe, **BENCHMARKS}
    return MarketData(
        prices=prices, local_prices=local, benchmarks=bench, fx=fx,
        base_currency=base_currency, names=names, source="synthetic",
        high=high, low=low, volume=volume,
    )


def synthetic_etf_market_data(start: str = "2015-01-01", end: str = "2025-12-31",
                              base_currency: str = "SEK", seed: int = 19) -> MarketData:
    """Simulate a multi-asset UCITS ETF + Fund universe with distinct bull/bear regimes."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    T = len(dates)

    # Create persistent regimes (bull vs crash) so tactical & defensive assets shine realistically
    eq_us = rng.normal(0.00045, 0.010, T)
    eq_se = 0.65 * eq_us + rng.normal(0.00018, 0.0085, T)
    eq_em = 0.55 * eq_us + rng.normal(0.00015, 0.011, T)
    gold = -0.15 * eq_us + rng.normal(0.00030, 0.0085, T)
    bonds = -0.25 * eq_us + rng.normal(0.00012, 0.0038, T)
    cash = np.full(T, 0.025 / 252.0) + rng.normal(0.0, 0.0004, T)

    # Inject two realistic multi-month bear regimes where equities drop and gold/bonds/cash protect
    for start_frac in (0.35, 0.72):
        s = int(T * start_frac)
        e = min(T, s + 90)
        eq_us[s:e] -= 0.0035
        eq_se[s:e] -= 0.0038
        eq_em[s:e] -= 0.0042
        gold[s:e] += 0.0018
        bonds[s:e] += 0.0008

    fx_usd = pd.Series(9.8 * np.exp(np.cumsum(rng.normal(0.0, 0.005, T) - 0.2 * eq_us)),
                       index=dates, name=FX_TICKER)
    fx_eur = pd.Series(10.8 * np.exp(np.cumsum(rng.normal(0.0, 0.0035, T) - 0.1 * eq_us)),
                       index=dates, name=EUR_FX_TICKER)

    series_map = {
        "EUNL.DE": 0.75 * eq_us + 0.25 * eq_se,
        "SXR8.DE": eq_us,
        "SXRV.DE": 1.2 * eq_us + rng.normal(0.0001, 0.005, T),
        "XACT-SVERIGE.ST": eq_se + 0.03 / 252.0,
        "XACT-OMXS30.ST": eq_se + 0.032 / 252.0,
        "XACT-NORDEN.ST": 0.8 * eq_se + 0.2 * eq_us + 0.02 / 252.0,
        "IS3N.DE": eq_em,
        "IUSN.DE": 0.85 * eq_us + 0.15 * eq_se + rng.normal(0.0, 0.004, T),
        "4GLD.DE": gold,
        "EUNA.DE": bonds,
        "IB01.L": cash,
    }
    local = pd.DataFrame({
        t: 100.0 * np.exp(np.cumsum(np.log1p(np.clip(r, -0.3, 0.3))))
        for t, r in series_map.items() if t in ETF_UNIVERSE
    }, index=dates)
    bench_local = pd.DataFrame({
        "SPY": 100.0 * np.exp(np.cumsum(np.log1p(eq_us))),
        "URTH": 100.0 * np.exp(np.cumsum(np.log1p(0.75 * eq_us + 0.25 * eq_se))),
        "XACT-OMXS30.ST": local["XACT-OMXS30.ST"],
        "^OMX": 100.0 * np.exp(np.cumsum(np.log1p(eq_se))),
    }, index=dates)

    prices = _to_base(local, fx_usd, base_currency, fx_eur=fx_eur)
    bench = _to_base(bench_local, fx_usd, base_currency, fx_eur=fx_eur)
    return MarketData(
        prices=prices, local_prices=local, benchmarks=bench, fx=fx_usd, fx_eur=fx_eur,
        base_currency=base_currency, names=all_names(), source="synthetic",
    )
