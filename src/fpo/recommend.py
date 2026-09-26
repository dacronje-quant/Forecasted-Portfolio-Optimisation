"""Live recommendation: the portfolio to buy this month."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fpo.config import BacktestConfig
from fpo.data import MarketData
from fpo.features import FEATURES, build_features, cross_sectional_ranks
from fpo.models import fit_full_model, permutation_importance_table
from fpo.optimizer import optimise_weights
from fpo.universe import ticker_country, ticker_currency


@dataclass
class Recommendation:
    as_of: pd.Timestamp
    price_date: pd.Timestamp
    table: pd.DataFrame            # one row per holding
    all_scores: pd.DataFrame       # every eligible stock with its score and rank
    cash_left: float
    budget: float
    base_currency: str
    importance: pd.Series | None = None
    trades: pd.DataFrame | None = None


def allocate_whole_shares(weights: pd.Series, prices: pd.Series, budget: float) -> dict[str, int]:
    """Round target amounts down to whole shares, then greedily spend leftover cash on the
    holdings furthest below target, as long as each extra share brings the holding closer
    to its target (i.e. its shortfall is at least half a share) and fits in the budget."""
    px = prices.astype(float)
    target = weights * budget
    shares = np.floor(target / px).fillna(0).astype(int)
    cash = budget - float((shares * px).sum())
    for _ in range(10_000):
        gap = target - shares * px                    # shortfall vs target, in money
        ok = (px <= cash + 1e-9) & (gap >= 0.5 * px)
        if not ok.any():
            break
        best = (gap[ok] / budget).idxmax()
        shares[best] += 1
        cash -= px[best]
    return {k: int(v) for k, v in shares.items()}


def recommend_portfolio(data: MarketData, cfg: BacktestConfig | None = None, budget: float = 100_000,
                        features: pd.DataFrame | None = None,
                        current_holdings: dict[str, float] | None = None,
                        compute_importance: bool = False) -> Recommendation:
    """Train on all history and build the portfolio for the latest available month end.

    budget: amount to invest, in the base currency.
    current_holdings: optional {ticker: shares} you already own -> produces a trade list.
    """
    cfg = (cfg or BacktestConfig()).validate()
    feats = features if features is not None else build_features(data.prices)
    as_of = feats.index.get_level_values("date").max()
    model, X_tr, y_tr = fit_full_model(feats, cfg, as_of)

    X_now = cross_sectional_ranks(feats).xs(as_of, level="date")
    scores = pd.Series(np.asarray(model.predict(X_now[FEATURES]), dtype=float), index=X_now.index)
    hist = data.prices.pct_change(fill_method=None).loc[:as_of]
    held = list(current_holdings) if current_holdings else None
    w = optimise_weights(scores, hist, cfg, current=held)

    price_date = data.prices.index[data.prices.index <= as_of][-1]
    local_px = data.local_prices.loc[:price_date].ffill().iloc[-1]
    base_px = data.prices.loc[:price_date].ffill().iloc[-1]

    shares_map = allocate_whole_shares(w, base_px[w.index], budget)
    rows = []
    for t, wt in w.items():
        amount = budget * wt
        px_b = float(base_px[t])
        shares = shares_map[t]
        rows.append({
            "ticker": t, "name": data.names.get(t, t), "country": ticker_country(t),
            "currency": ticker_currency(t), "weight": wt, "score": scores[t],
            "price_local": float(local_px[t]), f"price_{data.base_currency}": px_b,
            f"target_{data.base_currency}": amount, "shares": shares,
            f"invested_{data.base_currency}": shares * px_b,
            "mom_12_1": feats.loc[(as_of, t), "mom_12_1"], "vol_3m": feats.loc[(as_of, t), "vol_3m"],
        })
    table = pd.DataFrame(rows).set_index("ticker")
    cash_left = budget - table[f"invested_{data.base_currency}"].sum()

    all_scores = pd.DataFrame({"score": scores, "rank": scores.rank(ascending=False).astype(int)})
    all_scores["name"] = [data.names.get(t, t) for t in all_scores.index]
    all_scores["country"] = [ticker_country(t) for t in all_scores.index]
    all_scores["selected"] = all_scores.index.isin(w.index)
    all_scores = all_scores.sort_values("rank")

    trades = None
    if current_holdings:
        cur = pd.Series(current_holdings, dtype=float)
        tgt = table["shares"].astype(float)
        idx = cur.index.union(tgt.index)
        trades = pd.DataFrame({"current_shares": cur.reindex(idx, fill_value=0),
                               "target_shares": tgt.reindex(idx, fill_value=0)})
        trades["trade_shares"] = trades["target_shares"] - trades["current_shares"]
        trades["action"] = np.where(trades["trade_shares"] > 0, "BUY",
                                    np.where(trades["trade_shares"] < 0, "SELL", "HOLD"))

    importance = None
    if compute_importance and cfg.model != "momentum":
        importance = permutation_importance_table(model, X_tr, y_tr)

    return Recommendation(as_of=as_of, price_date=price_date, table=table, all_scores=all_scores,
                          cash_left=cash_left, budget=budget, base_currency=data.base_currency,
                          importance=importance, trades=trades)
