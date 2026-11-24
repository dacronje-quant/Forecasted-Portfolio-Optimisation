"""Performance and statistical-significance metrics for monthly return series."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

PPY = 12  # periods per year
EULER_GAMMA = 0.5772156649015329


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


def probabilistic_sharpe(r: pd.Series, sr_benchmark: float = 0.0, rf: float = 0.0) -> float:
    """P(true Sharpe > sr_benchmark) accounting for skew/kurtosis (Bailey & López de Prado)."""
    ex = r.dropna() - rf / PPY
    n = len(ex)
    if n < 12 or ex.std() == 0:
        return np.nan
    sr = ex.mean() / ex.std()                 # per-period
    sr_b = sr_benchmark / np.sqrt(PPY)
    g3 = stats.skew(ex)
    g4 = stats.kurtosis(ex, fisher=False)
    denom = np.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return float(stats.norm.cdf((sr - sr_b) * np.sqrt(n - 1) / denom))


def deflated_sharpe(r: pd.Series, n_trials: int = 1, sr_benchmark: float = 0.0,
                    rf: float = 0.0) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Adjusts the benchmark Sharpe hurdle upward for the expected maximum Sharpe
    obtained by luck when selecting among `n_trials` candidate strategies.
    Equals `probabilistic_sharpe` when `n_trials <= 1`.
    """
    ex = r.dropna() - rf / PPY
    n = len(ex)
    if n < 12 or ex.std() == 0:
        return np.nan
    if n_trials <= 1:
        return probabilistic_sharpe(r, sr_benchmark=sr_benchmark, rf=rf)

    # Expected maximum of N standard normals (extreme value theorem approximation)
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    emax = (1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2
    sr_null_ann = emax / np.sqrt(max(1, n - 1)) * np.sqrt(PPY)
    effective_bench = max(sr_benchmark, sr_null_ann)
    return probabilistic_sharpe(r, sr_benchmark=effective_bench, rf=rf)


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


def summary(r: pd.Series, bench: pd.Series | None = None, rf: float = 0.0) -> dict:
    out = {
        "CAGR": cagr(r), "Volatility": ann_vol(r), "Sharpe": sharpe(r, rf=rf),
        "Sortino": sortino(r, rf=rf), "Max drawdown": max_drawdown(r), "Calmar": calmar(r),
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


def comparison_table(strategy: pd.Series, benchmarks: pd.DataFrame,
                     ref: str | None = None, rf: float = 0.0) -> pd.DataFrame:
    cols = {"Strategy": summary(strategy, benchmarks[ref] if ref else None, rf=rf)}
    for c in benchmarks.columns:
        cols[c] = summary(benchmarks[c].reindex(strategy.index),
                          benchmarks[ref] if ref and c != ref else None, rf=rf)
    return pd.DataFrame(cols)


def subperiod_summary(r: pd.Series, bench: pd.Series | None = None,
                      rf: float = 0.0, years_per_bucket: int = 3) -> pd.DataFrame:
    """Break down CAGR, Sharpe, Max DD, and excess CAGR across non-overlapping multi-year periods."""
    r = r.dropna()
    if r.empty:
        return pd.DataFrame()
    min_y, max_y = int(r.index.year.min()), int(r.index.year.max())
    rows = {}
    y = min_y
    while y <= max_y:
        end_y = min(max_y, y + years_per_bucket - 1)
        mask = (r.index.year >= y) & (r.index.year <= end_y)
        sub_r = r[mask]
        if len(sub_r) >= 6:
            label = f"{y}–{end_y}" if end_y > y else str(y)
            row = {
                "Months": len(sub_r),
                "CAGR": cagr(sub_r),
                "Sharpe": sharpe(sub_r, rf=rf),
                "Max drawdown": max_drawdown(sub_r),
            }
            if bench is not None:
                sub_b = bench.reindex(sub_r.index).dropna()
                if len(sub_b) >= 6:
                    row["Bench CAGR"] = cagr(sub_b)
                    row["Excess CAGR"] = cagr(sub_r) - cagr(sub_b)
            rows[label] = row
        y = end_y + 1
    return pd.DataFrame(rows).T


def rolling_excess(r: pd.Series, b: pd.Series, window: int = 12) -> pd.Series:
    rr = (1 + r).rolling(window).apply(np.prod, raw=True) - 1
    bb = (1 + b).rolling(window).apply(np.prod, raw=True) - 1
    return rr - bb


def rolling_sharpe(r: pd.Series, window: int = 36, rf: float = 0.0) -> pd.Series:
    ex = r - rf / PPY
    return ex.rolling(window).mean() / ex.rolling(window).std() * np.sqrt(PPY)


def calendar_year_returns(df: pd.DataFrame) -> pd.DataFrame:
    return (1 + df).groupby(df.index.year).prod() - 1


def random_portfolio_percentile(strategy_gross: pd.Series, random_returns: pd.DataFrame) -> dict:
    """Where the strategy's (frictionless) CAGR/Sharpe fall in the random-portfolio distribution."""
    rc = random_returns.apply(cagr)
    rs = random_returns.apply(sharpe)
    sc, ss = cagr(strategy_gross), sharpe(strategy_gross)
    return {"cagr_percentile": float((rc < sc).mean()), "sharpe_percentile": float((rs < ss).mean()),
            "random_cagr": rc, "random_sharpe": rs, "strategy_cagr": sc, "strategy_sharpe": ss}
