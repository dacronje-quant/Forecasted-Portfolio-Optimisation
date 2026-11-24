"""Persistent portfolio state and cost-aware Avanza/Nordnet execution planning."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fpo.config import BacktestConfig, BrokerFeeSchedule
from fpo.universe import ticker_country, ticker_currency, ticker_sector


@dataclass
class Holding:
    shares: float
    avg_cost_local: float = 0.0


@dataclass
class PortfolioState:
    broker_preset: str = "avanza_mini"
    base_currency: str = "SEK"
    cash_sek: float = 0.0
    cash_usd: float = 0.0
    holdings: dict[str, Holding] = field(default_factory=dict)
    trade_log: list[dict] = field(default_factory=list)

    def held_shares(self) -> dict[str, float]:
        """Return only active holdings with strictly positive share counts."""
        return {t: float(h.shares) for t, h in self.holdings.items() if h.shares > 1e-9}

    def total_cash_base(self, fx_rate: float, base_currency: str | None = None) -> float:
        cur = base_currency or self.base_currency
        if cur == "SEK":
            return float(self.cash_sek + self.cash_usd * fx_rate)
        return float(self.cash_usd + self.cash_sek / max(fx_rate, 1e-6))

    def to_dict(self) -> dict:
        return {
            "broker_preset": self.broker_preset,
            "base_currency": self.base_currency,
            "cash_sek": round(float(self.cash_sek), 2),
            "cash_usd": round(float(self.cash_usd), 2),
            "holdings": {
                t: {"shares": float(h.shares), "avg_cost_local": round(float(h.avg_cost_local), 4)}
                for t, h in self.holdings.items() if h.shares > 1e-9
            },
            "trade_log": list(self.trade_log),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PortfolioState":
        raw_h = d.get("holdings", {})
        holdings: dict[str, Holding] = {}
        for t, v in raw_h.items():
            if isinstance(v, (int, float)):
                if float(v) > 1e-9:
                    holdings[str(t)] = Holding(shares=float(v))
            elif isinstance(v, dict):
                sh = float(v.get("shares", 0.0))
                if sh > 1e-9:
                    holdings[str(t)] = Holding(shares=sh, avg_cost_local=float(v.get("avg_cost_local", 0.0)))
        return cls(
            broker_preset=str(d.get("broker_preset", "avanza_mini")),
            base_currency=str(d.get("base_currency", "SEK")),
            cash_sek=float(d.get("cash_sek", 0.0)),
            cash_usd=float(d.get("cash_usd", 0.0)),
            holdings=holdings,
            trade_log=list(d.get("trade_log", [])),
        )


def load_portfolio(path: Path | str) -> PortfolioState:
    """Load portfolio state from JSON (`portfolio.json`) or legacy CSV (`ticker,shares`)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Portfolio file not found: {p}")
    if p.suffix.lower() == ".csv":
        return portfolio_from_csv(p)
    return PortfolioState.from_dict(json.loads(p.read_text(encoding="utf-8")))


def save_portfolio(state: PortfolioState, path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=False), encoding="utf-8")
    return p


def portfolio_from_csv(path: Path | str, broker_preset: str = "avanza_mini",
                       base_currency: str = "SEK", cash_sek: float = 0.0) -> PortfolioState:
    df = pd.read_csv(path)
    if "ticker" not in df.columns or "shares" not in df.columns:
        raise ValueError("Holdings CSV must contain columns 'ticker' and 'shares'")
    holdings = {}
    for _, row in df.iterrows():
        sh = float(row["shares"])
        if sh > 1e-9:
            avg_c = float(row["avg_cost_local"]) if "avg_cost_local" in df.columns and pd.notna(row["avg_cost_local"]) else 0.0
            holdings[str(row["ticker"]).strip()] = Holding(shares=sh, avg_cost_local=avg_c)
    return PortfolioState(broker_preset=broker_preset, base_currency=base_currency,
                          cash_sek=cash_sek, holdings=holdings)


@dataclass
class ExecutionPlan:
    as_of: pd.Timestamp
    price_date: pd.Timestamp
    base_currency: str
    broker_schedule: BrokerFeeSchedule
    rebalance_mode: str
    verdict: str                      # 'REBALANCE', 'DEPLOY CASH ONLY', 'HOLD'
    verdict_reason: str
    existing_equity_value: float
    starting_cash: float
    deposit_amount: float
    total_portfolio_value: float
    exposure: float
    orders: pd.DataFrame              # ordered SELL -> BUY -> HOLD / SKIP with exact fees & reasons
    sell_proceeds_gross: float
    sell_fees: float
    buy_notional: float
    buy_fees: float
    total_fees: float
    projected_cash_left: float
    expected_monthly_alpha_gain: float
    unpriced_holdings: list[str] = field(default_factory=list)

    @property
    def sells(self) -> pd.DataFrame:
        return self.orders[self.orders["action"] == "SELL"]

    @property
    def buys(self) -> pd.DataFrame:
        return self.orders[self.orders["action"] == "BUY"]

    @property
    def holds(self) -> pd.DataFrame:
        return self.orders[self.orders["action"].isin(["HOLD", "SKIP"])]


def _order_fee(fee_sched: BrokerFeeSchedule, notional: float, ticker: str,
               base_currency: str, fx_rate: float) -> float:
    return fee_sched.trade_cost(notional, ticker_currency(ticker), base_currency, fx_rate)


def build_execution_plan(
    target_weights: pd.Series,
    all_scores: pd.DataFrame,
    local_prices: pd.Series,
    base_prices: pd.Series,
    fx_rate: float,
    state: PortfolioState,
    cfg: BacktestConfig,
    as_of: pd.Timestamp,
    price_date: pd.Timestamp,
    deposit_base: float = 0.0,
    exposure: float = 1.0,
    effective_ic: float = 0.03,
    names: dict[str, str] | None = None,
) -> ExecutionPlan:
    """Build a cash-safe, broker-fee-aware monthly SELL -> BUY execution plan."""
    names = names or {}
    base = cfg.base_currency
    fee_sched = cfg.fee_schedule()
    mode = cfg.rebalance_mode
    min_order_base = cfg.min_order_sek if base == "SEK" else cfg.min_order_sek / max(fx_rate, 1e-6)
    exit_rank = int(np.ceil(cfg.n_holdings * max(cfg.hold_buffer, 1.0)))

    cur_holdings = state.held_shares()
    unpriced: list[str] = []
    cur_val: dict[str, float] = {}
    for t, sh in cur_holdings.items():
        px = base_prices.get(t, np.nan)
        if pd.isna(px) or px <= 0:
            unpriced.append(t)
            cur_val[t] = 0.0
        else:
            cur_val[t] = float(sh * px)

    existing_equity = float(sum(cur_val.values()))
    starting_cash = state.total_cash_base(fx_rate, base) + float(deposit_base)
    total_val = existing_equity + starting_cash
    investable_target = max(0.0, total_val * float(np.clip(exposure, 0.0, 1.0)))

    target_w = target_weights[target_weights > 1e-6].copy()
    if len(target_w) > 0:
        target_w = target_w / target_w.sum()
    target_amt = {t: float(w * investable_target) for t, w in target_w.items()}

    # Helper to extract metadata & plain-English drivers
    def _rank_of(t: str) -> int | None:
        if t in all_scores.index and pd.notna(all_scores.loc[t, "rank"]):
            return int(all_scores.loc[t, "rank"])
        return None

    def _name_of(t: str) -> str:
        if t in all_scores.index and "name" in all_scores.columns:
            return str(all_scores.loc[t, "name"])
        return names.get(t, t)

    # ------------------------------------------------------------------ SELLS
    sell_shares: dict[str, int] = {}
    sell_reasons: dict[str, str] = {}
    skip_reasons: dict[str, str] = {}

    for t, sh in cur_holdings.items():
        if t in unpriced:
            skip_reasons[t] = "Unpriced ticker — verify quote manually before selling"
            continue
        px_b = float(base_prices[t])
        val_now = cur_val[t]
        tgt_val = target_amt.get(t, 0.0)
        rk = _rank_of(t)
        sh_int = int(np.floor(sh + 1e-9))

        if t not in target_amt:
            # Not in target portfolio
            if mode == "smart_dca" and rk is not None and rk <= exit_rank and val_now <= total_val * cfg.max_weight * 1.35:
                skip_reasons[t] = f"Smart DCA: kept (rank #{rk} within #{exit_rank} hold band)"
                continue
            if sh_int > 0:
                sell_shares[t] = sh_int
                if rk is None:
                    sell_reasons[t] = "Exited: outside investable universe"
                else:
                    sell_reasons[t] = f"Exited: rank #{rk} fell below #{exit_rank} hold band"
        else:
            excess = val_now - tgt_val
            drift = excess / max(total_val, 1.0)
            if excess <= 0:
                continue
            if mode == "smart_dca":
                # Only trim if concentration breaches 1.35x max_weight
                if val_now / max(total_val, 1.0) > cfg.max_weight * 1.35:
                    n_sell = int(np.floor(excess / px_b))
                    if n_sell > 0 and n_sell * px_b >= min_order_base:
                        sell_shares[t] = min(sh_int, n_sell)
                        sell_reasons[t] = f"Trimmed: weight {val_now / total_val:.1%} breached concentration cap"
                else:
                    skip_reasons[t] = f"Smart DCA: held without trimming (drift +{drift:.1%})"
            else:
                # Full rebalance mode: respect no-trade band & min order size
                n_sell = int(np.floor(excess / px_b))
                if drift >= cfg.no_trade_band and n_sell > 0 and n_sell * px_b >= min_order_base:
                    sell_shares[t] = min(sh_int, n_sell)
                    sell_reasons[t] = f"Rebalance trim: overweight by +{drift:.1%}"
                elif n_sell > 0:
                    skip_reasons[t] = f"Skipped micro-sell (drift +{drift:.1%}, below band/min order)"

    sell_gross = 0.0
    sell_fees = 0.0
    for t, n_s in sell_shares.items():
        notional = n_s * float(base_prices[t])
        fee = _order_fee(fee_sched, notional, t, base, fx_rate)
        sell_gross += notional
        sell_fees += fee

    net_sell_proceeds = max(0.0, sell_gross - sell_fees)
    available_cash = starting_cash + net_sell_proceeds
    defensive_reserve = max(0.0, (1.0 - float(np.clip(exposure, 0.0, 1.0))) * total_val)
    spendable_before_buf = max(0.0, available_cash - defensive_reserve)
    # Keep a small slippage buffer so live limit/market BUY orders never bounce
    buy_budget_cap = spendable_before_buf * (1.0 - float(np.clip(cfg.slippage_buffer, 0.0, 0.15)))

    # ------------------------------------------------------------------ BUYS
    post_sell_shares = {t: float(cur_holdings.get(t, 0.0)) - float(sell_shares.get(t, 0))
                        for t in set(cur_holdings) | set(target_amt)}
    post_sell_val = {
        t: (post_sell_shares[t] * float(base_prices[t]) if t in base_prices.index and pd.notna(base_prices[t]) else 0.0)
        for t in post_sell_shares
    }

    buy_shares: dict[str, int] = {}
    buy_reasons: dict[str, str] = {}

    # Candidate buy targets sorted by largest dollar shortfall (underweight first)
    shortfalls = {}
    for t, tgt_val in target_amt.items():
        px_b = base_prices.get(t, np.nan)
        if pd.isna(px_b) or px_b <= 0:
            continue
        gap = tgt_val - post_sell_val.get(t, 0.0)
        if gap > 0:
            shortfalls[t] = gap

    def _current_buy_total_cost(bmap: dict[str, int]) -> float:
        tot = 0.0
        for tk, ns in bmap.items():
            if ns <= 0:
                continue
            notional = ns * float(base_prices[tk])
            tot += notional + _order_fee(fee_sched, notional, tk, base, fx_rate)
        return tot

    # For starter / small portfolios where even the largest target position is smaller than
    # min_order_base, scale the effective minimum order down so the deposit actually gets invested.
    max_single_target = max(target_amt.values()) if target_amt else 0.0
    if max_single_target > 0 and max_single_target < min_order_base:
        eff_min_order = max(300.0 if base == "SEK" else 30.0, max_single_target * 0.60)
    else:
        eff_min_order = min_order_base

    # Filter candidates respecting no_trade_band (in full mode) and eff_min_order.
    buy_candidates = sorted(shortfalls.keys(), key=lambda tk: shortfalls[tk], reverse=True)
    if buy_budget_cap > 0 and buy_candidates:
        # Pass 1: allocate proportional/target whole shares starting from most underweight stocks
        for t in buy_candidates:
            px_b = float(base_prices[t])
            gap = shortfalls[t]
            drift = gap / max(total_val, 1.0)
            if mode == "full" and drift < cfg.no_trade_band and existing_equity > 0:
                skip_reasons[t] = f"Skipped micro-buy (drift -{drift:.1%} inside no-trade band)"
                continue
            rem_cash = buy_budget_cap - _current_buy_total_cost(buy_shares)
            if rem_cash < px_b:
                if px_b > gap * 1.5 and cur_holdings.get(t, 0.0) <= 0:
                    skip_reasons[t] = f"Share price ({px_b:,.0f} {base}) exceeds target ({gap:,.0f} {base})"
                continue
            # Desired whole shares up to gap (allow rounding up by 1 share if >= 50% of share price)
            desired_sh = int(np.floor(gap / px_b))
            if desired_sh == 0 and gap >= 0.5 * px_b:
                desired_sh = 1
            if desired_sh <= 0:
                if cur_holdings.get(t, 0.0) <= 0:
                    skip_reasons[t] = f"Share price ({px_b:,.0f} {base}) exceeds target ({gap:,.0f} {base})"
                continue
            # Cap shares to fit remaining cash after exact brokerage + FX fee
            max_fit = desired_sh
            while max_fit > 0:
                test_notional = max_fit * px_b
                test_cost = test_notional + _order_fee(fee_sched, test_notional, t, base, fx_rate)
                if test_cost <= rem_cash + 1e-9:
                    break
                max_fit -= 1
            if max_fit <= 0:
                continue
            order_notional = max_fit * px_b
            if order_notional < eff_min_order and buy_budget_cap >= eff_min_order:
                skip_reasons[t] = f"Skipped micro-buy ({order_notional:,.0f} {base} < min {eff_min_order:,.0f} {base})"
                continue
            buy_shares[t] = max_fit

        # Pass 2: greedily deploy leftover cash into underweight targets (within 0.5 share of target)
        for _ in range(500):
            rem_cash = buy_budget_cap - _current_buy_total_cost(buy_shares)
            best_t = None
            best_gap_ratio = -1.0
            for t in buy_candidates:
                px_b = float(base_prices[t])
                cur_b = buy_shares.get(t, 0)
                new_notional = (cur_b + 1) * px_b
                if cur_b == 0 and new_notional < eff_min_order and buy_budget_cap >= eff_min_order:
                    continue
                if cur_b == 0 and mode == "full" and existing_equity > 0:
                    if shortfalls[t] / max(total_val, 1.0) < cfg.no_trade_band:
                        continue
                gap_left = target_amt[t] - (post_sell_val.get(t, 0.0) + cur_b * px_b)
                if gap_left < 0.5 * px_b:
                    continue
                old_cost = (cur_b * px_b + _order_fee(fee_sched, cur_b * px_b, t, base, fx_rate)) if cur_b > 0 else 0.0
                new_cost = new_notional + _order_fee(fee_sched, new_notional, t, base, fx_rate)
                inc_cost = new_cost - old_cost
                if inc_cost <= rem_cash + 1e-9:
                    ratio = gap_left / max(target_amt[t], 1.0)
                    if ratio > best_gap_ratio:
                        best_gap_ratio = ratio
                        best_t = t
            if best_t is None:
                break
            buy_shares[best_t] = buy_shares.get(best_t, 0) + 1
            skip_reasons.pop(best_t, None)

        # Fallback: if significant cash remains undeployed in Smart DCA (e.g. deposit spread thin
        # across 15 targets so individual gaps < min order), concentrate into top underweight stocks.
        rem_after = buy_budget_cap - _current_buy_total_cost(buy_shares)
        if mode == "smart_dca" and rem_after >= eff_min_order:
            for t in buy_candidates:
                if t in buy_shares:
                    continue
                rem_cash = buy_budget_cap - _current_buy_total_cost(buy_shares)
                if rem_cash < eff_min_order:
                    break
                px_b = float(base_prices[t])
                if px_b > rem_cash:
                    continue
                alloc = min(rem_cash, max(shortfalls[t], eff_min_order))
                n_b = int(np.floor(alloc / px_b))
                while n_b > 0:
                    notional = n_b * px_b
                    if notional + _order_fee(fee_sched, notional, t, base, fx_rate) <= rem_cash + 1e-9:
                        break
                    n_b -= 1
                if n_b > 0 and n_b * px_b >= eff_min_order * 0.8:
                    buy_shares[t] = n_b
                    skip_reasons.pop(t, None)

    buy_notional = 0.0
    buy_fees = 0.0
    for t, n_b in buy_shares.items():
        notional = n_b * float(base_prices[t])
        fee = _order_fee(fee_sched, notional, t, base, fx_rate)
        buy_notional += notional
        buy_fees += fee
        rk = _rank_of(t)
        gap_pct = shortfalls.get(t, 0.0) / max(total_val, 1.0)
        mom_txt = ""
        if t in all_scores.index and "mom_12_1" in all_scores.columns and pd.notna(all_scores.loc[t, "mom_12_1"]):
            mom_txt = f", 12-1m {all_scores.loc[t, 'mom_12_1']:+.0%}"
        rk_txt = f"rank #{rk}" if rk is not None else "selected"
        prefix = "New position" if cur_holdings.get(t, 0.0) <= 0 else "Top up"
        buy_reasons[t] = f"{prefix}: {rk_txt}{mom_txt}, underweight by {gap_pct:.1%}"

    total_fees = sell_fees + buy_fees
    projected_cash = available_cash - (buy_notional + buy_fees)
    # Strict safety invariant
    assert projected_cash >= -1e-6, f"Cash safety violation: projected_cash={projected_cash:.2f}"
    projected_cash = max(0.0, projected_cash)

    # ------------------------------------------------------------------ ORDER TABLE
    all_tickers = list(dict.fromkeys(list(sell_shares) + list(buy_shares) + list(cur_holdings) + list(target_amt)))
    rows = []
    for t in all_tickers:
        cur_sh = float(cur_holdings.get(t, 0.0))
        d_sh = float(buy_shares.get(t, 0) - sell_shares.get(t, 0))
        tgt_sh = cur_sh + d_sh
        px_l = float(local_prices[t]) if t in local_prices.index and pd.notna(local_prices[t]) else np.nan
        px_b = float(base_prices[t]) if t in base_prices.index and pd.notna(base_prices[t]) else np.nan
        trade_notional = abs(d_sh) * px_b if pd.notna(px_b) else 0.0
        est_fee = _order_fee(fee_sched, trade_notional, t, base, fx_rate) if abs(d_sh) > 0 else 0.0
        post_val = tgt_sh * px_b if pd.notna(px_b) else 0.0

        if d_sh < 0:
            action = "SELL"
            order_prio = 0
            reason = sell_reasons.get(t, "Rebalance sell")
        elif d_sh > 0:
            action = "BUY"
            order_prio = 1
            reason = buy_reasons.get(t, "Rebalance buy")
        else:
            if t in skip_reasons:
                action = "SKIP"
                order_prio = 3
                reason = skip_reasons[t]
            elif cur_sh <= 0:
                action = "SKIP"
                order_prio = 3
                reason = "Not purchased (insufficient cash or below min order)"
            else:
                action = "HOLD"
                order_prio = 2
                rk = _rank_of(t)
                reason = f"On target (rank #{rk})" if rk is not None else "Held"

        rows.append({
            "_prio": order_prio,
            "action": action,
            "ticker": t,
            "name": _name_of(t),
            "country": ticker_country(t),
            "sector": ticker_sector(t),
            "currency": ticker_currency(t),
            "rank": _rank_of(t),
            "price_local": px_l,
            f"price_{base}": px_b,
            "current_shares": cur_sh,
            "trade_shares": d_sh,
            "target_shares": tgt_sh,
            f"trade_value_{base}": trade_notional,
            f"est_fee_{base}": est_fee,
            "target_weight": float(target_w.get(t, 0.0) * exposure),
            "post_weight": post_val / max(total_val, 1.0),
            "reason": reason,
        })

    orders = pd.DataFrame(rows)
    if not orders.empty:
        orders = orders.sort_values(["_prio", f"trade_value_{base}"], ascending=[True, False]).drop(columns="_prio")
        orders = orders.set_index("ticker")

    # ------------------------------------------------------------------ VERDICT
    # Estimate expected monthly active alpha improvement (Grinold approximation)
    z_scores = all_scores["score"] if "score" in all_scores.columns else pd.Series(dtype=float)
    z_norm = (z_scores - z_scores.mean()) / (z_scores.std() + 1e-12) if len(z_scores) > 1 else pd.Series(dtype=float)
    w_before = pd.Series({t: cur_val.get(t, 0.0) / max(total_val, 1.0) for t in all_tickers})
    w_after = orders["post_weight"] if not orders.empty else pd.Series(dtype=float)
    common_z = z_norm.reindex(w_after.index).fillna(0.0)
    score_gain = float((w_after * common_z).sum() - (w_before.reindex(w_after.index).fillna(0.0) * common_z).sum())
    expected_monthly_alpha = max(0.0, effective_ic * 0.22 * score_gain / 12.0 * total_val)

    has_sells = len(sell_shares) > 0
    has_buys = len(buy_shares) > 0
    fee_pct = total_fees / max(total_val, 1.0)

    if not has_sells and not has_buys:
        verdict = "HOLD"
        verdict_reason = "No cost-effective trades needed this month — existing positions are within bands."
    elif has_buys and not has_sells:
        verdict = "DEPLOY CASH ONLY"
        verdict_reason = (
            f"Deploy {buy_notional:,.0f} {base} into {len(buy_shares)} underweight top stock(s) "
            f"(est. fees {total_fees:,.0f} {base} / {fee_pct:.2%}) — zero unnecessary sells or FX churn."
        )
    else:
        verdict = "REBALANCE"
        verdict_reason = (
            f"Execute {len(sell_shares)} SELL(s) ({sell_gross:,.0f} {base}) then {len(buy_shares)} BUY(s) "
            f"({buy_notional:,.0f} {base}); est. total fees {total_fees:,.0f} {base} ({fee_pct:.2%} of portfolio)."
        )

    return ExecutionPlan(
        as_of=as_of, price_date=price_date, base_currency=base, broker_schedule=fee_sched,
        rebalance_mode=mode, verdict=verdict, verdict_reason=verdict_reason,
        existing_equity_value=existing_equity, starting_cash=starting_cash,
        deposit_amount=float(deposit_base), total_portfolio_value=total_val,
        exposure=float(exposure), orders=orders, sell_proceeds_gross=sell_gross,
        sell_fees=sell_fees, buy_notional=buy_notional, buy_fees=buy_fees,
        total_fees=total_fees, projected_cash_left=projected_cash,
        expected_monthly_alpha_gain=expected_monthly_alpha, unpriced_holdings=unpriced,
    )


def apply_execution_plan(state: PortfolioState, plan: ExecutionPlan) -> PortfolioState:
    """Apply executed orders from `plan` to `state`, updating holdings, cash, and trade_log."""
    new_holdings = {t: Holding(h.shares, h.avg_cost_local) for t, h in state.holdings.items()}
    trades_logged = []
    base = plan.base_currency

    for t, row in plan.orders.iterrows():
        act = row["action"]
        if act not in ("BUY", "SELL"):
            continue
        d_sh = float(row["trade_shares"])
        px_l = float(row["price_local"])
        cur_h = new_holdings.get(t, Holding(0.0, 0.0))
        if act == "BUY":
            tot_sh = cur_h.shares + d_sh
            avg_c = ((cur_h.shares * cur_h.avg_cost_local) + (d_sh * px_l)) / max(tot_sh, 1e-9)
            new_holdings[t] = Holding(shares=tot_sh, avg_cost_local=avg_c)
        else:
            rem_sh = max(0.0, cur_h.shares + d_sh)
            if rem_sh <= 1e-9:
                new_holdings.pop(t, None)
            else:
                new_holdings[t] = Holding(shares=rem_sh, avg_cost_local=cur_h.avg_cost_local)
        trades_logged.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "as_of": str(plan.as_of.date()),
            "action": act,
            "ticker": t,
            "shares": abs(d_sh),
            "price_local": round(px_l, 4),
            "currency": str(row["currency"]),
            f"value_{base}": round(float(row[f"trade_value_{base}"]), 2),
            f"fee_{base}": round(float(row[f"est_fee_{base}"]), 2),
        })

    new_log = list(state.trade_log) + trades_logged
    cash_sek = plan.projected_cash_left if base == "SEK" else 0.0
    cash_usd = plan.projected_cash_left if base == "USD" else 0.0
    return PortfolioState(
        broker_preset=state.broker_preset, base_currency=base,
        cash_sek=cash_sek, cash_usd=cash_usd,
        holdings=new_holdings, trade_log=new_log,
    )
