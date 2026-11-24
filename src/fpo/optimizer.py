"""Turn forecast scores into long-only, constraint-respecting, turnover-aware portfolio weights."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

from fpo.config import BacktestConfig
from fpo.universe import ticker_country, ticker_currency, ticker_sector


def precompute_weekly_returns(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Resample daily returns to Friday-ending weekly compounded returns ONCE for fast backtesting."""
    if daily_returns.empty:
        return daily_returns
    return (1.0 + daily_returns).resample("W-FRI").prod(min_count=1) - 1.0


def _group_masks(tickers: list[str], fn) -> dict[str, np.ndarray]:
    labels = np.array([fn(t) for t in tickers])
    return {k: (labels == k) for k in np.unique(labels)}


def select_holdings(scores: pd.Series, n: int, current: list[str] | None = None,
                    buffer: float = 1.0, max_weight: float = 1.0,
                    max_country: float = 1.0, max_sector: float = 1.0) -> list[str]:
    """Top-n by score, keeping existing holdings within n*buffer while guaranteeing
    country and sector cap feasibility whenever enough diverse stocks exist."""
    ranked = scores.sort_values(ascending=False)
    order = list(ranked.index)
    n = min(n, len(order))
    if n == 0:
        return []

    eff_max_w = max(max_weight, 1.0 / n)
    avail_countries = {ticker_country(t) for t in order}
    avail_sectors = {ticker_sector(t) for t in order}

    # Maximum slots one country/sector may take while guaranteeing mathematical feasibility
    # of simultaneous max_weight + max_country + max_sector constraints.
    if max_country < 1.0 - 1e-9 and len(avail_countries) > 1:
        min_other_cty = int(np.ceil((1.0 - max_country - 1e-9) / eff_max_w))
        max_per_country = max(1, min(n - min_other_cty, int(np.ceil(n * max_country))))
    else:
        max_per_country = n

    if max_sector < 1.0 - 1e-9 and len(avail_sectors) > 1:
        min_other_sec = int(np.ceil((1.0 - max_sector - 1e-9) / eff_max_w))
        # Ensure at least ceil(1 / max_sector) distinct sectors are chosen when n allows
        min_sectors_needed = int(np.ceil(1.0 / max_sector - 1e-9))
        if len(avail_sectors) >= min_sectors_needed and n >= min_sectors_needed:
            by_share = max(1, int(np.floor(n * max_sector)))
            max_per_sector = max(1, min(n - min_other_sec, by_share))
        else:
            max_per_sector = max(1, n - min_other_sec)
    else:
        max_per_sector = n

    cty_counts: dict[str, int] = {}
    sec_counts: dict[str, int] = {}
    chosen: list[str] = []
    chosen_set: set[str] = set()

    def _can_add(t: str) -> bool:
        c = ticker_country(t)
        s = ticker_sector(t)
        return cty_counts.get(c, 0) < max_per_country and sec_counts.get(s, 0) < max_per_sector

    def _add(t: str) -> None:
        chosen.append(t)
        chosen_set.add(t)
        c = ticker_country(t)
        s = ticker_sector(t)
        cty_counts[c] = cty_counts.get(c, 0) + 1
        sec_counts[s] = sec_counts.get(s, 0) + 1

    # 1. Keep eligible existing holdings within the hold buffer
    if current and buffer > 1.0:
        cur_set = set(current)
        limit = int(np.ceil(n * buffer))
        top_buffer = set(order[:limit])
        for t in order:
            if len(chosen) >= n:
                break
            if t in cur_set and t in top_buffer and _can_add(t):
                _add(t)

    # 2. Fill remaining slots with highest-scoring feasible stocks
    for t in order:
        if len(chosen) >= n:
            break
        if t not in chosen_set and _can_add(t):
            _add(t)

    # 3. Fallback if universe lacks enough secondary-country/sector stocks to fill n
    if len(chosen) < n:
        for t in order:
            if len(chosen) >= n:
                break
            if t not in chosen_set:
                _add(t)

    # Return in score order
    return [t for t in order if t in chosen_set]


def shrunk_covariance(returns: pd.DataFrame, freq: str = "weekly",
                      weekly_returns: pd.DataFrame | None = None) -> pd.DataFrame:
    """Annualised Ledoit-Wolf covariance matrix.

    Using weekly (5-day) compounded returns by default eliminates cross-timezone
    close-time asynchronicity and holiday zero-fill correlation bias between US,
    European, and Swedish assets. Pass `weekly_returns` (precomputed once) to avoid
    redundant per-date resampling inside backtest loops.
    """
    if freq == "weekly":
        wk = None
        if weekly_returns is not None and len(weekly_returns) >= 15:
            wk = weekly_returns.dropna(how="all")
        elif len(returns) >= 40:
            wk = ((1.0 + returns).resample("W-FRI").prod(min_count=1) - 1.0).dropna(how="all")
        if wk is not None and len(wk) >= 15:
            r = wk.fillna(0.0).values
            cov = LedoitWolf().fit(r).covariance_ * 52.0
            return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)

    r = returns.fillna(0.0).values
    if r.shape[0] < 20:
        cov = np.diag(np.full(r.shape[1], 0.3 ** 2))
    else:
        cov = LedoitWolf().fit(r).covariance_ * 252.0
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def _group_caps_feasible(groups: dict[str, np.ndarray], cap: float, max_weight: float) -> bool:
    if cap >= 1.0 - 1e-9 or len(groups) <= 1:
        return False
    # Sum of individual group ceilings must be >= 1.0
    total_capacity = sum(min(cap, int(m.sum()) * max_weight) for m in groups.values())
    return total_capacity >= 1.0 - 1e-9


def apply_caps(w: np.ndarray, tickers: list[str], max_weight: float,
               max_country: float = 1.0, max_sector: float = 1.0,
               iters: int = 150) -> np.ndarray:
    """Project weights onto {sum=1, 0<=w<=max_weight, country<=max_country, sector<=max_sector}."""
    w0 = np.clip(np.asarray(w, dtype=float), 0.0, None)
    if w0.sum() <= 0:
        w0 = np.ones_like(w0)
    w = w0 / w0.sum()
    n = len(w)
    max_weight = max(max_weight, 1.0 / n)

    cty_groups = _group_masks(tickers, ticker_country)
    sec_groups = _group_masks(tickers, ticker_sector)
    use_cty = _group_caps_feasible(cty_groups, max_country, max_weight)
    use_sec = _group_caps_feasible(sec_groups, max_sector, max_weight)

    def _cap_groups(vec: np.ndarray, groups: dict[str, np.ndarray], cap: float) -> bool:
        mod = False
        for mask in groups.values():
            tot = vec[mask].sum()
            if tot > cap + 1e-9:
                others = ~mask
                excess = tot - cap
                vec[mask] *= cap / tot
                if vec[others].sum() > 0:
                    vec[others] += excess * vec[others] / vec[others].sum()
                else:
                    vec[others] += excess / others.sum()
                mod = True
        return mod

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
        if use_cty and _cap_groups(w, cty_groups, max_country):
            changed = True
        if use_sec and _cap_groups(w, sec_groups, max_sector):
            changed = True
        if not changed:
            break

    w = w / w.sum()

    # Verify tight feasibility; if iterative capping oscillated between country & sector caps,
    # solve exact Euclidean projection via SLSQP with analytical Jacobians.
    viol = (w.max() > max_weight + 1e-6)
    if use_cty and any(w[m].sum() > max_country + 1e-5 for m in cty_groups.values()):
        viol = True
    if use_sec and any(w[m].sum() > max_sector + 1e-5 for m in sec_groups.values()):
        viol = True
    if viol:
        ones = np.ones(n)
        cons = [{"type": "eq", "fun": lambda x: x.sum() - 1.0, "jac": lambda x: ones}]
        if use_cty:
            for m in cty_groups.values():
                mf = -m.astype(float)
                cons.append({"type": "ineq", "fun": lambda x, mm=m: max_country - x[mm].sum(),
                             "jac": lambda x, j=mf: j})
        if use_sec:
            for m in sec_groups.values():
                mf = -m.astype(float)
                cons.append({"type": "ineq", "fun": lambda x, mm=m: max_sector - x[mm].sum(),
                             "jac": lambda x, j=mf: j})
        target = w0 / w0.sum()
        res = minimize(
            lambda x: 0.5 * float(np.sum((x - target) ** 2)),
            w,
            jac=lambda x: x - target,
            method="SLSQP",
            bounds=[(0.0, max_weight)] * n,
            constraints=cons,
            options={"maxiter": 200, "ftol": 1e-10},
        )
        if res.success and np.all(np.isfinite(res.x)):
            w = np.clip(res.x, 0.0, max_weight)
            w = w / w.sum()
    return w


def _build_constraints(tickers: list[str], cfg: BacktestConfig, max_w: float) -> list[dict]:
    n = len(tickers)
    ones = np.ones(n)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: ones}]
    cty_groups = _group_masks(tickers, ticker_country)
    if _group_caps_feasible(cty_groups, cfg.max_country_weight, max_w):
        for mask in cty_groups.values():
            mf = -mask.astype(float)
            cons.append({"type": "ineq", "fun": lambda w, m=mask: cfg.max_country_weight - w[m].sum(),
                         "jac": lambda w, j=mf: j})
    sec_groups = _group_masks(tickers, ticker_sector)
    if _group_caps_feasible(sec_groups, cfg.max_sector_weight, max_w):
        for mask in sec_groups.values():
            mf = -mask.astype(float)
            cons.append({"type": "ineq", "fun": lambda w, m=mask: cfg.max_sector_weight - w[m].sum(),
                         "jac": lambda w, j=mf: j})
    return cons


def _solve(objective, n: int, tickers: list[str], cfg: BacktestConfig,
           x0: np.ndarray | None = None, jac=None) -> np.ndarray:
    max_w = max(cfg.max_weight, 1.0 / n)
    cons = _build_constraints(tickers, cfg, max_w)
    x0 = np.full(n, 1.0 / n) if x0 is None else x0
    x0 = apply_caps(x0, tickers, cfg.max_weight, cfg.max_country_weight, cfg.max_sector_weight)
    res = minimize(objective, x0, jac=jac, method="SLSQP", bounds=[(0.0, max_w)] * n,
                   constraints=cons, options={"maxiter": 200, "ftol": 1e-8})
    w = res.x if res.success and np.all(np.isfinite(res.x)) else x0
    return apply_caps(w, tickers, cfg.max_weight, cfg.max_country_weight, cfg.max_sector_weight)


def _solve_mv_turnover(mu: np.ndarray, S: np.ndarray, w_prev: np.ndarray, trade_cost: np.ndarray,
                       tickers: list[str], cfg: BacktestConfig) -> np.ndarray:
    """Mean-variance with exact, smooth L1 turnover penalty via split variables w = w_prev + u - d."""
    n = len(mu)
    max_w = max(cfg.max_weight, 1.0 / n)
    lam = cfg.risk_aversion

    def _obj(z: np.ndarray) -> float:
        w = w_prev + z[:n] - z[n:]
        return float(-(mu @ w) + 0.5 * lam * (w @ S @ w) + trade_cost @ (z[:n] + z[n:]))

    def _jac(z: np.ndarray) -> np.ndarray:
        w = w_prev + z[:n] - z[n:]
        gw = -mu + lam * (S @ w)
        return np.concatenate([gw + trade_cost, -gw + trade_cost])

    eq_jac = np.concatenate([np.ones(n), -np.ones(n)])
    eye = np.eye(n)
    j_lb = np.hstack([eye, -eye])
    j_ub = -j_lb

    cons = [
        {"type": "eq", "fun": lambda z: (w_prev + z[:n] - z[n:]).sum() - 1.0, "jac": lambda z: eq_jac},
        {"type": "ineq", "fun": lambda z: w_prev + z[:n] - z[n:], "jac": lambda z: j_lb},
        {"type": "ineq", "fun": lambda z: max_w - (w_prev + z[:n] - z[n:]), "jac": lambda z: j_ub},
    ]
    cty_groups = _group_masks(tickers, ticker_country)
    if _group_caps_feasible(cty_groups, cfg.max_country_weight, max_w):
        for mask in cty_groups.values():
            mf = mask.astype(float)
            jm = np.concatenate([-mf, mf])
            cons.append({"type": "ineq", "fun": lambda z, m=mask: cfg.max_country_weight - (w_prev + z[:n] - z[n:])[m].sum(),
                         "jac": lambda z, j=jm: j})
    sec_groups = _group_masks(tickers, ticker_sector)
    if _group_caps_feasible(sec_groups, cfg.max_sector_weight, max_w):
        for mask in sec_groups.values():
            mf = mask.astype(float)
            jm = np.concatenate([-mf, mf])
            cons.append({"type": "ineq", "fun": lambda z, m=mask: cfg.max_sector_weight - (w_prev + z[:n] - z[n:])[m].sum(),
                         "jac": lambda z, j=jm: j})

    w_init = apply_caps(w_prev if w_prev.sum() > 0 else np.full(n, 1.0 / n),
                        tickers, cfg.max_weight, cfg.max_country_weight, cfg.max_sector_weight)
    diff = w_init - w_prev
    z0 = np.concatenate([np.maximum(diff, 0.0), np.maximum(-diff, 0.0)])
    bounds = [(0.0, max_w)] * n + [(0.0, float(max(max_w, w_prev[i]))) for i in range(n)]

    res = minimize(_obj, z0, jac=_jac, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 150, "ftol": 1e-7})
    if res.success and np.all(np.isfinite(res.x)):
        w = w_prev + res.x[:n] - res.x[n:]
        return apply_caps(w, tickers, cfg.max_weight, cfg.max_country_weight, cfg.max_sector_weight)
    return _solve(lambda x: -(mu @ x) + 0.5 * lam * (x @ S @ x), n, tickers, cfg,
                  jac=lambda x: -mu + lam * (S @ x))


def hrp_weights(cov: pd.DataFrame) -> np.ndarray:
    """Hierarchical Risk Parity (López de Prado, 2016)."""
    c = cov.values
    sd = np.sqrt(np.maximum(np.diag(c), 1e-12))
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
                ivp = 1.0 / np.maximum(np.diag(sub), 1e-12)
                ivp /= ivp.sum()
                return ivp @ sub @ ivp

            va, vb = cvar(a), cvar(b)
            alpha = 1.0 - va / (va + vb) if (va + vb) > 0 else 0.5
            w[a] *= alpha
            w[b] *= 1.0 - alpha
            new += [a, b]
        clusters = new
    return w / w.sum()


def _cluster_labels(cov_arr: np.ndarray, max_clusters: int = 5) -> np.ndarray:
    """Ward hierarchical clustering on correlation distance."""
    n = cov_arr.shape[0]
    if n <= 2:
        return np.arange(1, n + 1)
    sd = np.sqrt(np.maximum(np.diag(cov_arr), 1e-12))
    corr = np.clip(cov_arr / np.outer(sd, sd), -1.0, 1.0)
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method="ward")
    k = max(2, min(max_clusters, int(np.ceil(np.sqrt(n)))))
    return fcluster(Z, t=k, criterion="maxclust")


def herc_weights(cov: pd.DataFrame, returns: pd.DataFrame | None = None) -> np.ndarray:
    """Hierarchical Equal Risk Contribution (Raffinot, 2018).

    Uses Ward linkage clustering to avoid HRP's single-linkage chaining, computes
    cluster downside-adjusted risk (CVaR/variance blend when returns are given),
    and equalizes risk contribution across and within clusters.
    """
    c = cov.values
    n = c.shape[0]
    if n == 1:
        return np.array([1.0])
    if n == 2:
        iv = 1.0 / np.sqrt(np.maximum(np.diag(c), 1e-12))
        return iv / iv.sum()

    labels = _cluster_labels(c, max_clusters=min(6, n))
    unique_cl = np.unique(labels)
    w = np.zeros(n, dtype=float)
    cl_risks = {}
    cl_intra = {}

    r_vals = returns.fillna(0.0).values if returns is not None and len(returns) >= 20 else None
    for cl in unique_cl:
        idx = np.where(labels == cl)[0]
        sub_c = c[np.ix_(idx, idx)]
        # Intra-cluster inverse-variance weights (downside-tilted if returns available)
        if r_vals is not None:
            sub_r = r_vals[:, idx]
            down = np.sqrt(np.mean(np.minimum(sub_r, 0.0) ** 2, axis=0)) * np.sqrt(252.0)
            asset_risk = 0.5 * np.sqrt(np.maximum(np.diag(sub_c), 1e-12)) + 0.5 * np.maximum(down, 1e-4)
        else:
            asset_risk = np.sqrt(np.maximum(np.diag(sub_c), 1e-12))
        iw = 1.0 / np.maximum(asset_risk, 1e-6)
        iw /= iw.sum()
        cl_intra[cl] = (idx, iw)
        # Cluster volatility (equal risk contribution allocates inversely to cluster risk)
        cl_vol = float(np.sqrt(max(float(iw @ sub_c @ iw), 1e-12)))
        if r_vals is not None:
            cl_ret = r_vals[:, idx] @ iw
            q05 = np.quantile(cl_ret, 0.05)
            tail = cl_ret[cl_ret <= q05]
            cvar = abs(float(tail.mean())) * np.sqrt(252.0) if len(tail) > 0 else cl_vol
            cl_risks[cl] = 0.5 * cl_vol + 0.5 * max(cvar, 1e-4)
        else:
            cl_risks[cl] = cl_vol

    inv_cl = {cl: 1.0 / max(rk, 1e-6) for cl, rk in cl_risks.items()}
    tot_inv = sum(inv_cl.values())
    for cl in unique_cl:
        idx, iw = cl_intra[cl]
        w[idx] = (inv_cl[cl] / tot_inv) * iw
    return w / w.sum()


def _mini_mv(S: np.ndarray, mu: np.ndarray | None, lam: float) -> np.ndarray:
    """Fast long-only convex MV / MinVar solver on a small sub-cluster."""
    m = S.shape[0]
    if m == 1:
        return np.array([1.0])
    # Regularize diagonal slightly for numerical conditioning
    S_reg = S + np.eye(m) * 1e-6
    x0 = np.full(m, 1.0 / m)
    ones = np.ones(m)
    cons = [{"type": "eq", "fun": lambda x: x.sum() - 1.0, "jac": lambda x: ones}]
    if mu is None:
        obj = lambda x: float(x @ S_reg @ x)
        jac = lambda x: 2.0 * (S_reg @ x)
    else:
        obj = lambda x: float(-(mu @ x) + 0.5 * lam * (x @ S_reg @ x))
        jac = lambda x: -mu + lam * (S_reg @ x)
    res = minimize(obj, x0, jac=jac, method="SLSQP", bounds=[(0.0, 1.0)] * m,
                   constraints=cons, options={"maxiter": 100, "ftol": 1e-8})
    if res.success and np.all(np.isfinite(res.x)) and res.x.sum() > 0:
        w = np.clip(res.x, 0.0, 1.0)
        return w / w.sum()
    iv = 1.0 / np.maximum(np.diag(S_reg), 1e-8)
    return iv / iv.sum()


def nco_weights(cov: pd.DataFrame, mu: np.ndarray | None = None,
                risk_aversion: float = 5.0) -> np.ndarray:
    """Nested Clustered Optimization (López de Prado, 2019).

    Decouples intra-cluster and inter-cluster optimisation so alpha forecasts can
    be used without Markowitz covariance-inversion blow-ups.
    """
    S = cov.values
    n = S.shape[0]
    if n == 1:
        return np.array([1.0])
    if n <= 3:
        return _mini_mv(S, mu, risk_aversion)

    labels = _cluster_labels(S, max_clusters=min(5, n))
    unique_cl = np.unique(labels)
    k = len(unique_cl)
    W_intra = np.zeros((n, k), dtype=float)
    for j, cl in enumerate(unique_cl):
        idx = np.where(labels == cl)[0]
        sub_S = S[np.ix_(idx, idx)]
        sub_mu = mu[idx] if mu is not None else None
        W_intra[idx, j] = _mini_mv(sub_S, sub_mu, risk_aversion)

    # Reduced k x k cluster covariance and expected returns
    S_red = W_intra.T @ S @ W_intra
    mu_red = (W_intra.T @ mu) if mu is not None else None
    w_inter = _mini_mv(S_red, mu_red, risk_aversion)
    w = W_intra @ w_inter
    w = np.clip(w, 0.0, None)
    return w / w.sum() if w.sum() > 0 else np.full(n, 1.0 / n)


def estimate_dynamic_ic(past_ic: pd.Series | list[float], prior: float = 0.03,
                        lookback: int = 36, prior_weight: float = 12.0) -> float:
    """Empirical-Bayes shrunk estimate of trailing out-of-sample rank-IC, clipped to [0, 0.12]."""
    s = pd.Series(past_ic, dtype=float).dropna()
    if len(s) == 0:
        return float(np.clip(prior, 0.0, 0.12))
    recent = s.iloc[-lookback:]
    n = len(recent)
    shrunk = (n * float(recent.mean()) + prior_weight * prior) / (n + prior_weight)
    return float(np.clip(shrunk, 0.0, 0.12))


def compute_regime_exposure(bench_daily: pd.DataFrame, as_of: pd.Timestamp,
                            target_vol: float = 0.14, min_exposure: float = 0.40) -> float:
    """Compute leak-free equity exposure scalar in [min_exposure, 1.0] from trailing benchmark prices.

    Combines 63-day volatility targeting with a 200-day trend filter. Uses strictly prices <= as_of.
    """
    if bench_daily is None or bench_daily.empty:
        return 1.0
    sub = bench_daily.loc[:as_of].dropna(how="all")
    if len(sub) < 70:
        return 1.0
    if "URTH" in sub.columns and sub["URTH"].dropna().shape[0] >= 70:
        px = sub["URTH"].dropna()
    elif "SPY" in sub.columns and sub["SPY"].dropna().shape[0] >= 70:
        px = sub["SPY"].dropna()
    else:
        rets = sub.pct_change(fill_method=None).mean(axis=1).dropna()
        px = (1.0 + rets).cumprod()

    rets = px.pct_change(fill_method=None).dropna()
    if len(rets) < 63:
        return 1.0
    realized_vol = float(rets.iloc[-63:].std() * np.sqrt(252.0))
    vol_scale = float(np.clip(target_vol / max(realized_vol, 1e-6), min_exposure, 1.0))

    trend_scale = 1.0
    if len(px) >= 200:
        sma200 = float(px.iloc[-200:].mean())
        if float(px.iloc[-1]) < sma200:
            trend_scale = 0.75
    return float(np.clip(vol_scale * trend_scale, min_exposure, 1.0))


def optimise_weights(scores: pd.Series, daily_returns: pd.DataFrame, cfg: BacktestConfig,
                     current: list[str] | None = None,
                     prev_weights: pd.Series | None = None,
                     effective_ic: float | None = None,
                     weekly_returns: pd.DataFrame | None = None) -> pd.Series:
    """Choose holdings and weights for one rebalance date.

    scores: model score per eligible ticker (higher = better expected relative return).
    daily_returns: trailing daily returns (base currency) used for risk estimates.
    prev_weights: optional drifted portfolio weights just before rebalance (enables L1 turnover penalty).
    effective_ic: optional dynamically calibrated Grinold IC (overrides cfg.assumed_ic).
    weekly_returns: optional precomputed weekly returns sliced up to rebalance date for fast covariance.
    """
    scores = scores.dropna()
    if scores.empty:
        return pd.Series(dtype=float)

    held = current
    if held is None and prev_weights is not None:
        held = list(prev_weights[prev_weights > 1e-6].index)

    chosen = select_holdings(
        scores, cfg.n_holdings, current=held, buffer=cfg.hold_buffer,
        max_weight=cfg.max_weight, max_country=cfg.max_country_weight,
        max_sector=cfg.max_sector_weight,
    )
    n = len(chosen)
    rets = daily_returns[chosen].iloc[-cfg.cov_lookback_days:]
    wk_slice = weekly_returns[chosen].iloc[-max(16, cfg.cov_lookback_days // 5):] if weekly_returns is not None else None
    method = cfg.optimizer

    if method == "equal" or n == 1:
        w = np.full(n, 1.0 / n)
        w = apply_caps(w, chosen, cfg.max_weight, cfg.max_country_weight, cfg.max_sector_weight)
    elif method == "score":
        ranks = np.arange(n, 0, -1, dtype=float)
        w = apply_caps(ranks, chosen, cfg.max_weight, cfg.max_country_weight, cfg.max_sector_weight)
    elif method == "inverse_vol":
        vol = rets.std().fillna(rets.std().mean()).values * np.sqrt(252.0)
        w = apply_caps(1.0 / np.maximum(vol, 1e-4), chosen, cfg.max_weight,
                       cfg.max_country_weight, cfg.max_sector_weight)
    else:
        cov = shrunk_covariance(rets, freq=cfg.cov_freq, weekly_returns=wk_slice)
        S = cov.values
        if method == "min_variance":
            w = _solve(lambda x: float(x @ S @ x), n, chosen, cfg, jac=lambda x: 2.0 * (S @ x))
        elif method == "hrp":
            w = apply_caps(hrp_weights(cov), chosen, cfg.max_weight,
                           cfg.max_country_weight, cfg.max_sector_weight)
        elif method == "herc":
            w = apply_caps(herc_weights(cov, returns=rets), chosen, cfg.max_weight,
                           cfg.max_country_weight, cfg.max_sector_weight)
        elif method == "nco":
            z_all = (scores - scores.mean()) / (scores.std() + 1e-12)
            z = z_all[chosen].values
            vol = np.sqrt(np.maximum(np.diag(S), 1e-12))
            ic_val = cfg.assumed_ic if effective_ic is None else float(effective_ic)
            mu = ic_val * vol * z
            w = apply_caps(nco_weights(cov, mu=mu, risk_aversion=cfg.risk_aversion), chosen,
                           cfg.max_weight, cfg.max_country_weight, cfg.max_sector_weight)
        elif method == "mean_variance":
            z_all = (scores - scores.mean()) / (scores.std() + 1e-12)
            z = z_all[chosen].values
            vol = np.sqrt(np.maximum(np.diag(S), 1e-12))
            ic_val = cfg.assumed_ic if effective_ic is None else float(effective_ic)
            mu = ic_val * vol * z
            lam = cfg.risk_aversion
            if prev_weights is not None and len(prev_weights) > 0 and cfg.turnover_penalty > 0:
                w_prev = prev_weights.reindex(chosen, fill_value=0.0).values.astype(float)
                tc = np.array([
                    cfg.turnover_penalty * (1.5 if ticker_currency(t) != cfg.base_currency else 1.0)
                    for t in chosen
                ], dtype=float)
                w = _solve_mv_turnover(mu, S, w_prev, tc, chosen, cfg)
            else:
                w = _solve(lambda x: float(-(mu @ x) + 0.5 * lam * (x @ S @ x)), n, chosen, cfg,
                           jac=lambda x: -mu + lam * (S @ x))
        else:
            raise ValueError(f"Unknown optimizer {method!r}")

    out = pd.Series(w, index=chosen)
    out = out[out > 1e-6]
    out = out / out.sum()
    if cfg.min_weight > 0 and (out < cfg.min_weight).any() and (out >= cfg.min_weight).sum() >= 1:
        out = out[out >= cfg.min_weight]
        out = pd.Series(
            apply_caps(out.values, list(out.index), cfg.max_weight,
                       cfg.max_country_weight, cfg.max_sector_weight),
            index=out.index,
        )
    return out / out.sum()
