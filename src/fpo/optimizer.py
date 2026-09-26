"""Turn forecast scores into long-only portfolio weights."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

from fpo.config import BacktestConfig
from fpo.universe import ticker_country


def select_holdings(scores: pd.Series, n: int, current: list[str] | None = None,
                    buffer: float = 1.0) -> list[str]:
    """Top-n by score, but keep existing holdings still ranked within n*buffer (cuts turnover)."""
    ranked = scores.sort_values(ascending=False)
    order = list(ranked.index)
    n = min(n, len(order))
    keep = []
    if current and buffer > 1.0:
        limit = int(np.ceil(n * buffer))
        top_buffer = set(order[:limit])
        keep = [t for t in order if t in current and t in top_buffer][:n]
    chosen = list(keep)
    for t in order:
        if len(chosen) >= n:
            break
        if t not in chosen:
            chosen.append(t)
    # return in score order
    return [t for t in order if t in set(chosen)]


def shrunk_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Annualised Ledoit-Wolf covariance of daily returns."""
    r = returns.fillna(0.0).values
    if r.shape[0] < 20:
        cov = np.diag(np.full(r.shape[1], 0.3 ** 2))
    else:
        cov = LedoitWolf().fit(r).covariance_ * 252
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def _country_groups(tickers: list[str]) -> dict[str, np.ndarray]:
    c = np.array([ticker_country(t) for t in tickers])
    return {k: (c == k) for k in np.unique(c)}


def apply_caps(w: np.ndarray, tickers: list[str], max_weight: float, max_country: float = 1.0,
               iters: int = 100) -> np.ndarray:
    """Project weights onto {sum=1, 0<=w<=max_weight, country<=max_country} by iterative capping."""
    w = np.clip(np.asarray(w, dtype=float), 0, None)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()
    n = len(w)
    max_weight = max(max_weight, 1.0 / n)
    groups = _country_groups(tickers)
    for _ in range(iters):
        changed = False
        over = w > max_weight + 1e-12
        if over.any():
            excess = (w[over] - max_weight).sum()
            w[over] = max_weight
            free = ~over & (w < max_weight - 1e-12)
            if free.any():
                w[free] += excess * w[free] / w[free].sum() if w[free].sum() > 0 else excess / free.sum()
            changed = True
        if max_country < 1.0 and len(groups) > 1:
            for mask in groups.values():
                tot = w[mask].sum()
                if tot > max_country + 1e-9:
                    others = ~mask
                    excess = tot - max_country
                    w[mask] *= max_country / tot
                    if w[others].sum() > 0:
                        w[others] += excess * w[others] / w[others].sum()
                    else:
                        w[others] += excess / others.sum()
                    changed = True
        if not changed:
            break
    return w / w.sum()


def _solve(objective, n: int, tickers: list[str], cfg: BacktestConfig, x0: np.ndarray | None = None):
    max_w = max(cfg.max_weight, 1.0 / n)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    if cfg.max_country_weight < 1.0:
        for mask in _country_groups(tickers).values():
            cons.append({"type": "ineq", "fun": lambda w, m=mask: cfg.max_country_weight - w[m].sum()})
    x0 = np.full(n, 1.0 / n) if x0 is None else x0
    res = minimize(objective, x0, method="SLSQP", bounds=[(0.0, max_w)] * n, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-10})
    w = res.x if res.success else x0
    return apply_caps(w, tickers, cfg.max_weight, cfg.max_country_weight)


def hrp_weights(cov: pd.DataFrame) -> np.ndarray:
    """Hierarchical Risk Parity (López de Prado, 2016)."""
    c = cov.values
    sd = np.sqrt(np.diag(c))
    corr = np.clip(c / np.outer(sd, sd), -1, 1)
    n = len(sd)
    if n == 1:
        return np.array([1.0])
    dist = np.sqrt(np.clip((1 - corr) / 2.0, 0, None))
    np.fill_diagonal(dist, 0.0)
    order = leaves_list(linkage(squareform(dist, checks=False), method="single"))
    w = np.ones(n)
    clusters = [list(order)]
    while clusters:
        new = []
        for cl in clusters:
            if len(cl) <= 1:
                continue
            half = len(cl) // 2
            a, b = cl[:half], cl[half:]

            def cvar(idx):
                sub = c[np.ix_(idx, idx)]
                ivp = 1 / np.diag(sub)
                ivp /= ivp.sum()
                return ivp @ sub @ ivp

            va, vb = cvar(a), cvar(b)
            alpha = 1 - va / (va + vb)
            w[a] *= alpha
            w[b] *= 1 - alpha
            new += [a, b]
        clusters = new
    return w / w.sum()


def optimise_weights(scores: pd.Series, daily_returns: pd.DataFrame, cfg: BacktestConfig,
                     current: list[str] | None = None) -> pd.Series:
    """Choose holdings and weights for one rebalance date.

    scores: model score per eligible ticker (higher = better expected relative return).
    daily_returns: trailing daily returns (base currency) used for risk estimates.
    """
    scores = scores.dropna()
    if scores.empty:
        return pd.Series(dtype=float)
    chosen = select_holdings(scores, cfg.n_holdings, current, cfg.hold_buffer)
    n = len(chosen)
    rets = daily_returns[chosen].iloc[-cfg.cov_lookback_days:]
    method = cfg.optimizer

    if method == "equal" or n == 1:
        w = np.full(n, 1.0 / n)
        w = apply_caps(w, chosen, cfg.max_weight, cfg.max_country_weight)
    elif method == "score":
        ranks = np.arange(n, 0, -1, dtype=float)  # best gets n, worst gets 1
        w = apply_caps(ranks, chosen, cfg.max_weight, cfg.max_country_weight)
    elif method == "inverse_vol":
        vol = rets.std().fillna(rets.std().mean()).values * np.sqrt(252)
        w = apply_caps(1.0 / np.maximum(vol, 1e-4), chosen, cfg.max_weight, cfg.max_country_weight)
    else:
        cov = shrunk_covariance(rets)
        S = cov.values
        if method == "min_variance":
            w = _solve(lambda x: x @ S @ x, n, chosen, cfg)
        elif method == "hrp":
            w = apply_caps(hrp_weights(cov), chosen, cfg.max_weight, cfg.max_country_weight)
        elif method == "mean_variance":
            # Grinold & Kahn: expected active return = IC * volatility * score z-score.
            z_all = (scores - scores.mean()) / (scores.std() + 1e-12)
            z = z_all[chosen].values
            vol = np.sqrt(np.diag(S))
            mu = cfg.assumed_ic * vol * z
            lam = cfg.risk_aversion
            w = _solve(lambda x: -(mu @ x) + 0.5 * lam * (x @ S @ x), n, chosen, cfg)
        else:
            raise ValueError(f"Unknown optimizer {method!r}")

    out = pd.Series(w, index=chosen)
    out = out[out > 1e-6]
    out = out / out.sum()
    # Drop positions too small to be worth buying, then re-apply caps on what's left
    if cfg.min_weight > 0 and (out < cfg.min_weight).any() and (out >= cfg.min_weight).sum() >= 1:
        out = out[out >= cfg.min_weight]
        out = pd.Series(apply_caps(out.values, list(out.index), cfg.max_weight, cfg.max_country_weight),
                        index=out.index)
    return out / out.sum()
