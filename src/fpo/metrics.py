"""Performance and statistical-significance metrics for monthly return series."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

PPY = 12  # periods per year


def cagr(r: pd.Series) -> float:
    r = r.dropna()
    if r.empty:
        return np.nan
    growth = (1 + r).prod()
    return growth ** (PPY / len(r)) - 1 if growth > 0 else -1.0


def ann_vol(r: pd.Series) -> float:
    return r.dropna().std() * np.sqrt(PPY)


def sharpe(r: pd.Series, rf: float = 0.0) -> float:
    ex = r.dropna() - rf / PPY
    sd = ex.std()
    return ex.mean() / sd * np.sqrt(PPY) if sd > 0 else np.nan


def sortino(r: pd.Series, rf: float = 0.0) -> float:
    ex = r.dropna() - rf / PPY
    down = np.sqrt((np.minimum(ex, 0) ** 2).mean())
    return ex.mean() / down * np.sqrt(PPY) if down > 0 else np.nan


def drawdown(r: pd.Series) -> pd.Series:
    eq = (1 + r.fillna(0)).cumprod()
    return eq / eq.cummax() - 1


def max_drawdown(r: pd.Series) -> float:
    return drawdown(r).min()


def calmar(r: pd.Series) -> float:
    mdd = max_drawdown(r)
    return cagr(r) / abs(mdd) if mdd < 0 else np.nan


def beta_alpha(r: pd.Series, b: pd.Series) -> tuple[float, float]:
    df = pd.concat([r, b], axis=1).dropna()
    if len(df) < 6:
        return np.nan, np.nan
    slope, intercept, *_ = stats.linregress(df.iloc[:, 1], df.iloc[:, 0])
    return slope, intercept * PPY


def tracking_error(r: pd.Series, b: pd.Series) -> float:
    return (r - b).dropna().std() * np.sqrt(PPY)


def information_ratio(r: pd.Series, b: pd.Series) -> float:
    a = (r - b).dropna()
    return a.mean() / a.std() * np.sqrt(PPY) if a.std() > 0 else np.nan


def probabilistic_sharpe(r: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(true Sharpe > sr_benchmark) accounting for skew/kurtosis (Bailey & López de Prado)."""
    r = r.dropna()
    n = len(r)
    if n < 12 or r.std() == 0:
        return np.nan
    sr = r.mean() / r.std()                 # per-period
    sr_b = sr_benchmark / np.sqrt(PPY)
    g3 = stats.skew(r)
    g4 = stats.kurtosis(r, fisher=False)
    denom = np.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return float(stats.norm.cdf((sr - sr_b) * np.sqrt(n - 1) / denom))


def excess_return_test(r: pd.Series, b: pd.Series) -> dict:
    """t-test of mean monthly excess return plus a block-bootstrap probability of outperformance."""
    df = pd.concat([r.rename("r"), b.rename("b")], axis=1).dropna()
    a = df["r"] - df["b"]
    if len(a) < 12:
        return {"t_stat": np.nan, "p_value": np.nan, "p_outperform_bootstrap": np.nan}
    t, p = stats.ttest_1samp(a, 0.0)
    rng = np.random.default_rng(0)
    n, block, sims = len(a), 6, 2000
    lr = np.log1p(df.values)  # paired log returns: keeps the r/b correlation within each block
    n_blocks = int(np.ceil(n / block))
    wins = 0
    for _ in range(sims):
        starts = rng.integers(0, n - block + 1, n_blocks)
        sample = np.concatenate([lr[s:s + block] for s in starts])[:n]
        wins += sample[:, 0].sum() > sample[:, 1].sum()
    return {"t_stat": float(t), "p_value": float(p / 2 if t > 0 else 1 - p / 2),  # one-sided
            "p_outperform_bootstrap": wins / sims}


def summary(r: pd.Series, bench: pd.Series | None = None) -> dict:
    out = {
        "CAGR": cagr(r), "Volatility": ann_vol(r), "Sharpe": sharpe(r), "Sortino": sortino(r),
        "Max drawdown": max_drawdown(r), "Calmar": calmar(r),
        "Best month": r.max(), "Worst month": r.min(), "% positive months": (r > 0).mean(),
        "Total return": (1 + r.fillna(0)).prod() - 1,
    }
    if bench is not None:
        b = bench.reindex(r.index)
        beta, alpha = beta_alpha(r, b)
        out.update({
            "Excess CAGR": cagr(r) - cagr(b), "Beta": beta, "Alpha (ann.)": alpha,
            "Tracking error": tracking_error(r, b), "Information ratio": information_ratio(r, b),
            "% months beating": ((r - b) > 0).mean(),
        })
    return out


def comparison_table(strategy: pd.Series, benchmarks: pd.DataFrame, ref: str | None = None) -> pd.DataFrame:
    cols = {"Strategy": summary(strategy, benchmarks[ref] if ref else None)}
    for c in benchmarks.columns:
        cols[c] = summary(benchmarks[c].reindex(strategy.index),
                          benchmarks[ref] if ref and c != ref else None)
    return pd.DataFrame(cols)


def rolling_excess(r: pd.Series, b: pd.Series, window: int = 12) -> pd.Series:
    rr = (1 + r).rolling(window).apply(np.prod, raw=True) - 1
    bb = (1 + b).rolling(window).apply(np.prod, raw=True) - 1
    return rr - bb


def rolling_sharpe(r: pd.Series, window: int = 36) -> pd.Series:
    return r.rolling(window).mean() / r.rolling(window).std() * np.sqrt(PPY)


def calendar_year_returns(df: pd.DataFrame) -> pd.DataFrame:
    return (1 + df).groupby(df.index.year).prod() - 1


def random_portfolio_percentile(strategy_gross: pd.Series, random_returns: pd.DataFrame) -> dict:
    """Where the strategy's (frictionless) CAGR/Sharpe fall in the random-portfolio distribution."""
    rc = random_returns.apply(cagr)
    rs = random_returns.apply(sharpe)
    sc, ss = cagr(strategy_gross), sharpe(strategy_gross)
    return {"cagr_percentile": float((rc < sc).mean()), "sharpe_percentile": float((rs < ss).mean()),
            "random_cagr": rc, "random_sharpe": rs, "strategy_cagr": sc, "strategy_sharpe": ss}
