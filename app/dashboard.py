"""Interactive dashboard for the Forecasted Portfolio Optimisation decision engine.

Run with:   streamlit run app/dashboard.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fpo import metrics as M  # noqa: E402
from fpo.backtest import BLEND, EW_UNIVERSE, run_backtest  # noqa: E402
from fpo.config import BROKER_PRESETS, MODELS, OPTIMIZERS, REBALANCE_MODES, BacktestConfig  # noqa: E402
from fpo.data import load_market_data, synthetic_market_data  # noqa: E402
from fpo.features import FEATURE_DESCRIPTIONS, build_features_from_data  # noqa: E402
from fpo.meta import DEFAULT_CANDIDATES, FAST_CANDIDATES, run_meta_backtest  # noqa: E402
from fpo.portfolio import (  # noqa: E402
    Holding, PortfolioState, apply_execution_plan,
)
from fpo.recommend import recommend_optimal_portfolio, render_monthly_brief  # noqa: E402
from fpo.universe import ticker_country, ticker_sector  # noqa: E402

st.set_page_config(page_title="Forecasted Portfolio Optimisation", page_icon="📈", layout="wide")

# Validated categorical palette (fixed order; colour follows the entity)
STRATEGY_COLOR = "#2a78d6"
SERIES_COLORS = ["#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BENCH_ORDER = [
    "S&P 500 (SPY)", "MSCI World (URTH)", "OMXS30 Total Return (XACT)",
    "OMX Stockholm 30 (price index)", EW_UNIVERSE, BLEND,
]
BENCH_COLORS = dict(zip(BENCH_ORDER, SERIES_COLORS))
COUNTRY_COLORS = {"US": "#2a78d6", "SE": "#eda100"}
DIVERGING = [[0.0, "#e34948"], [0.5, "#f0efec"], [1.0, "#2a78d6"]]

MODEL_LABELS = {
    "ensemble": "Ensemble (Ridge + GBM + Random Forest + Neural net)",
    "gbm": "Gradient boosting",
    "random_forest": "Random forest",
    "ridge": "Ridge regression (linear)",
    "mlp": "Neural network (MLP)",
    "momentum": "12-1 momentum (no ML baseline)",
}
OPT_LABELS = {
    "mean_variance": "Mean-variance (forecast + shrunk cov + L1 turnover)",
    "min_variance": "Minimum variance",
    "hrp": "Hierarchical risk parity",
    "inverse_vol": "Inverse volatility",
    "score": "Score-weighted",
    "equal": "Equal weight",
}
BROKER_LABELS = {k: v.label for k, v in BROKER_PRESETS.items()}


# --------------------------------------------------------------------------- helpers
def show(fig: go.Figure) -> None:
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True)


def table(df: pd.DataFrame, **kw) -> None:
    try:
        st.dataframe(df, width="stretch", **kw)
    except Exception:
        st.dataframe(df, use_container_width=True, **kw)


def style_fig(fig: go.Figure, height: int = 420, yfmt: str | None = None, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=40 if title else 10, b=10), title=title,
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    if yfmt:
        fig.update_yaxes(tickformat=yfmt)
    return fig


def pct(x, d=1):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def color_for(name: str) -> str:
    return STRATEGY_COLOR if name.startswith("Strategy") or name.startswith("Meta") else BENCH_COLORS.get(name, "#888888")


# --------------------------------------------------------------------------- cached compute
@st.cache_data(show_spinner=False, ttl=6 * 3600)
def get_data(source: str, markets: str, currency: str, start: str, extra_tickers: tuple[str, ...] = ()):
    if source == "synthetic":
        return synthetic_market_data(markets, start=start, base_currency=currency)
    return load_market_data(markets, start=start, base_currency=currency,
                            extra_tickers=list(extra_tickers) if extra_tickers else None)


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def get_features(source: str, markets: str, currency: str, start: str, extra_tickers: tuple[str, ...] = ()):
    return build_features_from_data(get_data(source, markets, currency, start, extra_tickers))


@st.cache_data(show_spinner=False, ttl=6 * 3600, max_entries=40)
def get_backtest(source: str, cfg_items: tuple):
    cfg = BacktestConfig(**dict(cfg_items))
    data = get_data(source, cfg.markets, cfg.base_currency, cfg.start)
    feats = get_features(source, cfg.markets, cfg.base_currency, cfg.start)
    return run_backtest(data, cfg, features=feats)


@st.cache_data(show_spinner=False, ttl=6 * 3600, max_entries=16)
def get_meta_backtest(source: str, cfg_items: tuple, fast: bool, selection_mode: str, lookback_months: int):
    cfg = BacktestConfig(**dict(cfg_items))
    data = get_data(source, cfg.markets, cfg.base_currency, cfg.start)
    feats = get_features(source, cfg.markets, cfg.base_currency, cfg.start)
    cands = FAST_CANDIDATES if fast else DEFAULT_CANDIDATES
    return run_meta_backtest(data, cfg, candidates=cands, features=feats,
                             lookback_months=lookback_months, selection_mode=selection_mode)


def cfg_key(cfg: BacktestConfig) -> tuple:
    d = cfg.to_dict()
    d.pop("extra", None)
    return tuple(sorted(d.items()))


# --------------------------------------------------------------------------- sidebar
st.sidebar.title("⚙️ Settings")
with st.sidebar.form("settings"):
    st.markdown("**Data**")
    source = st.radio("Source", ["yahoo", "synthetic"], horizontal=True,
                      format_func=lambda s: "Yahoo Finance (live)" if s == "yahoo" else "Simulated demo",
                      help="Simulated data works offline and is useful to explore the app.")
    markets = st.selectbox("Markets", ["both", "us", "se"],
                           format_func={"both": "US + Sweden", "us": "US only", "se": "Sweden only"}.get)
    currency = st.selectbox("Measure returns in", ["SEK", "USD"])
    c1, c2 = st.columns(2)
    start = c1.text_input("History from", "2005-01-01")
    bt_start = c2.text_input("Backtest from", "2012-01-31")

    st.markdown("**Broker & Costs (Avanza / Nordnet)**")
    broker_keys = list(BROKER_PRESETS)
    broker_preset = st.selectbox("Broker fee preset", broker_keys,
                                 index=broker_keys.index("avanza_mini"),
                                 format_func=BROKER_LABELS.get)
    c1, c2 = st.columns(2)
    cost = c1.number_input("Custom bps", 0.0, 100.0, 15.0, 1.0, help="Used only when preset = Custom")
    fx_cost = c2.number_input("Custom FX bps", 0.0, 100.0, 25.0, 1.0, help="Used only when preset = Custom")
    rf_annual = st.number_input("Risk-free rate (%/yr)", 0.0, 10.0, 2.0, 0.25) / 100.0

    st.markdown("**Forecasting model (single-strategy view)**")
    model = st.selectbox("Model", MODELS, index=MODELS.index("gbm"), format_func=MODEL_LABELS.get,
                         help="Ensemble is the most robust but slower. Use 'Optimal Meta-Strategy' in the Action Plan for automatic selection.")
    c1, c2 = st.columns(2)
    retrain = c1.number_input("Retrain every (months)", 1, 24, 3)
    window = c2.number_input("Train window (0=expanding)", 0, 240, 0, step=12)
    target = st.selectbox("Target", ["rank", "excess"],
                          format_func={"rank": "Cross-sectional rank", "excess": "Excess return vs median"}.get)
    use_dynamic_ic = st.toggle("Dynamic OOS IC calibration", value=True,
                               help="Calibrates expected alpha from trailing out-of-sample IC instead of a fixed 0.05.")

    st.markdown("**Portfolio & Risk Constraints**")
    optimizer = st.selectbox("Optimiser", OPTIMIZERS, format_func=OPT_LABELS.get)
    n_hold = st.slider("Number of stocks", 5, 40, 15)
    c1, c2 = st.columns(2)
    max_w = c1.slider("Max stock weight", 0.03, 0.40, 0.12, 0.01)
    min_w = c2.slider("Min stock weight", 0.0, 0.05, 0.02, 0.005)
    c1, c2 = st.columns(2)
    max_country = c1.slider("Max country weight", 0.3, 1.0, 0.70, 0.05,
                            help="0.70 keeps at least 30% in the other market.")
    max_sector = c2.slider("Max sector weight", 0.15, 1.0, 0.35, 0.05,
                           help="Prevents accidental over-concentration in a single sector.")
    c1, c2 = st.columns(2)
    risk_aversion = c1.number_input("Risk aversion", 0.5, 50.0, 5.0, 0.5)
    buffer = c2.number_input("Hold buffer", 1.0, 3.0, 1.5, 0.1,
                             help="Keep existing holdings while ranked within N × buffer.")
    turnover_pen = st.number_input("L1 turnover penalty", 0.0, 0.05, 0.0025, 0.0005, format="%.4f",
                                   help="Penalises unnecessary rebalancing inside mean-variance (1.5x higher on USD trades).")
    core_fallback = st.toggle("Defensive fallback when IC <= 0", value=True)
    regime_overlay = st.toggle("Vol-target + 200d trend overlay", value=False,
                               help="Scales equity exposure down to defensive cash during high volatility / downtrends.")

    submitted = st.form_submit_button("▶ Apply & Run", type="primary")

cfg = BacktestConfig(
    markets=markets, base_currency=currency, start=start, backtest_start=bt_start, model=model,
    target=target, train_window_months=int(window), retrain_every=int(retrain), optimizer=optimizer,
    n_holdings=int(n_hold), max_weight=float(max_w), min_weight=float(min_w),
    max_country_weight=float(max_country), max_sector_weight=float(max_sector),
    risk_aversion=float(risk_aversion), hold_buffer=float(buffer),
    turnover_penalty=float(turnover_pen), use_dynamic_ic=bool(use_dynamic_ic),
    core_fallback=bool(core_fallback), regime_overlay=bool(regime_overlay),
    broker_preset=broker_preset, cost_bps=float(cost), fx_cost_bps=float(fx_cost),
    rf_annual=float(rf_annual),
)

if submitted or "cfg" not in st.session_state:
    st.session_state["cfg"] = cfg
    st.session_state["source"] = source
cfg = st.session_state["cfg"]
source = st.session_state["source"]

# --------------------------------------------------------------------------- run base backtest
st.title("📈 Forecasted Portfolio Optimisation")
st.caption(f"{cfg.fee_schedule().label} · {cfg.n_holdings} stocks · "
           f"{'US + Sweden' if cfg.markets == 'both' else cfg.markets.upper()} · returns in {cfg.base_currency} · "
           f"max country {cfg.max_country_weight:.0%} · max sector {cfg.max_sector_weight:.0%}")

try:
    with st.spinner("Loading prices & OHLCV..." if source == "yahoo" else "Simulating market..."):
        data = get_data(source, cfg.markets, cfg.base_currency, cfg.start)
    with st.spinner(f"Running walk-forward backtest ({MODEL_LABELS[cfg.model]})..."):
        res = get_backtest(source, cfg_key(cfg))
except Exception as e:  # pragma: no cover - UI feedback
    st.error(f"Could not run the backtest: {e}")
    if source == "yahoo":
        st.info("If Yahoo Finance is unreachable, switch the data source to **Simulated demo** in the sidebar.")
    st.stop()

if res.source == "synthetic":
    st.warning("You are looking at **simulated** data. Switch to **Yahoo Finance (live)** in the sidebar for real decisions.")

bench_names = list(res.benchmarks.columns)
default_ref = bench_names.index(BLEND) if BLEND in bench_names else 0
ref = st.selectbox("Main benchmark", bench_names, index=default_ref,
                   help="Index fund baseline. 'Equal-weight universe' isolates pure stock-picking skill.")
r = res.returns
b = res.benchmarks[ref]

s_sum = M.summary(r, b, rf=cfg.rf_annual)
b_sum = M.summary(b, rf=cfg.rf_annual)
k = st.columns(6)
k[0].metric("CAGR", pct(s_sum["CAGR"]), f"{(s_sum['CAGR'] - b_sum['CAGR']) * 100:+.1f} pp vs bench")
k[1].metric("Sharpe", f"{s_sum['Sharpe']:.2f}", f"{s_sum['Sharpe'] - b_sum['Sharpe']:+.2f}")
k[2].metric("Max drawdown", pct(s_sum["Max drawdown"]),
            f"{(s_sum['Max drawdown'] - b_sum['Max drawdown']) * 100:+.1f} pp")
k[3].metric("Volatility", pct(s_sum["Volatility"]),
            f"{(s_sum['Volatility'] - b_sum['Volatility']) * 100:+.1f} pp", delta_color="inverse")
k[4].metric("Months beating", pct(s_sum["% months beating"], 0))
test = M.excess_return_test(r, b)
k[5].metric("P(outperform)", pct(test["p_outperform_bootstrap"], 0),
            help="Block-bootstrap probability that cumulative return beats the benchmark.")

tabs = st.tabs([
    "🛒 Monthly Action Plan",
    "🏆 Optimal Strategy (Meta)",
    "📈 Performance",
    "🧪 Is it real?",
    "🧺 Holdings",
    "🔬 Features",
    "ℹ️ Method",
])

# =========================================================================== Tab 0: Monthly Action Plan
with tabs[0]:
    st.markdown("Generate this month's **cash-safe, Avanza/Nordnet-fee-aware execution ticket** "
                "using either the **walk-forward optimal meta-strategy** or your sidebar strategy.")

    c1, c2, c3, c4 = st.columns([1.4, 1.2, 1.0, 1.0])
    use_meta = c1.radio("Strategy engine", ["Optimal Meta-Strategy", "Sidebar Strategy"],
                        horizontal=True) == "Optimal Meta-Strategy"
    rebal_mode = c2.radio("Rebalance mode", list(REBALANCE_MODES), horizontal=True,
                          format_func={"smart_dca": "Smart DCA (cash first)", "full": "Full Rebalance"}.get)
    deposit_amt = c3.number_input(f"New monthly deposit ({cfg.base_currency})", 0.0, 10_000_000.0, 10_000.0, 1_000.0)
    existing_cash = c4.number_input(f"Existing cash ({cfg.base_currency})", 0.0, 10_000_000.0, 0.0, 1_000.0)

    c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.0, 1.2])
    min_ord = c1.number_input("Min order size (SEK)", 0.0, 50_000.0, 1_500.0, 250.0)
    nt_band = c2.number_input("No-trade band (%)", 0.0, 10.0, 1.5, 0.25) / 100.0
    fast_meta = c3.toggle("Fast 5-strategy meta", value=True,
                          help="Uses Momentum/Ridge/GBM across HRP/MV/MinVar (fast). Uncheck to include full Ensemble.")
    want_imp = c4.toggle("Explain model (OOS importance)", value=False)

    # Current holdings loader + inline editor
    with st.expander("Current holdings (upload portfolio.json / CSV or edit inline)", expanded=False):
        up = st.file_uploader("Upload portfolio.json or holdings.csv (ticker,shares)", type=["json", "csv"])
        if "holdings_df" not in st.session_state:
            st.session_state["holdings_df"] = pd.DataFrame({"ticker": pd.Series(dtype=str),
                                                            "shares": pd.Series(dtype=float),
                                                            "avg_cost_local": pd.Series(dtype=float)})
        if up is not None:
            try:
                if up.name.endswith(".json"):
                    parsed = PortfolioState.from_dict(json.loads(up.read().decode("utf-8")))
                    st.session_state["holdings_df"] = pd.DataFrame([
                        {"ticker": t, "shares": h.shares, "avg_cost_local": h.avg_cost_local}
                        for t, h in parsed.holdings.items()
                    ])
                    st.info(f"Loaded {len(parsed.holdings)} holdings from {up.name} (broker preset: {parsed.broker_preset}).")
                else:
                    h_csv = pd.read_csv(up)
                    if "avg_cost_local" not in h_csv.columns:
                        h_csv["avg_cost_local"] = 0.0
                    st.session_state["holdings_df"] = h_csv[["ticker", "shares", "avg_cost_local"]].copy()
                    st.info(f"Loaded {len(h_csv)} rows from {up.name}.")
            except Exception as e:
                st.error(f"Could not parse uploaded holdings file: {e}")

        edited_h = st.data_editor(
            st.session_state["holdings_df"], num_rows="dynamic", use_container_width=True,
            column_config={
                "ticker": st.column_config.TextColumn("Ticker (e.g. INVE-B.ST, MSFT)"),
                "shares": st.column_config.NumberColumn("Shares", min_value=0.0, step=1.0),
                "avg_cost_local": st.column_config.NumberColumn("Avg cost (local)", min_value=0.0, step=1.0),
            },
        )
        st.session_state["holdings_df"] = edited_h

    # Construct PortfolioState from editor
    holdings_dict: dict[str, Holding] = {}
    for _, row in st.session_state["holdings_df"].iterrows():
        tk = str(row.get("ticker", "")).strip()
        sh = float(row.get("shares", 0.0) or 0.0)
        ac = float(row.get("avg_cost_local", 0.0) or 0.0)
        if tk and sh > 1e-9:
            holdings_dict[tk] = Holding(shares=sh, avg_cost_local=ac)

    live_state = PortfolioState(
        broker_preset=cfg.broker_preset,
        base_currency=cfg.base_currency,
        cash_sek=existing_cash if cfg.base_currency == "SEK" else 0.0,
        cash_usd=existing_cash if cfg.base_currency == "USD" else 0.0,
        holdings=holdings_dict,
    )

    live_cfg = replace(cfg, rebalance_mode=rebal_mode, min_order_sek=float(min_ord), no_trade_band=float(nt_band))
    extra_tk = tuple(sorted(t for t in holdings_dict if t not in data.prices.columns))
    live_data = get_data(source, live_cfg.markets, live_cfg.base_currency, live_cfg.start, extra_tk) if extra_tk else data
    live_feats_hist = get_features(source, live_cfg.markets, live_cfg.base_currency, live_cfg.start, extra_tk)

    with st.spinner("Evaluating optimal strategy and building cash-safe Avanza/Nordnet execution ticket..."):
        meta_obj = None
        if use_meta:
            meta_obj = get_meta_backtest(source, cfg_key(live_cfg), fast_meta, "blend", 36)
        decision = recommend_optimal_portfolio(
            live_data, live_cfg, state=live_state, deposit_base=float(deposit_amt),
            features=live_feats_hist, meta_result=meta_obj,
            candidates=FAST_CANDIDATES if fast_meta else DEFAULT_CANDIDATES,
            use_meta=use_meta, compute_importance=want_imp,
        )

    plan = decision.plan
    health = decision.health
    cur = plan.base_currency

    # Data hygiene warnings banner
    if decision.data_warnings:
        warn_lines = "\n".join(f"- **[{w.severity.upper()}] `{w.ticker}`**: {w.message}" for w in decision.data_warnings)
        st.warning(f"**Data Hygiene Alerts — verify before trading:**\n{warn_lines}")

    # Strategy Health + Verdict Banner
    h_col, v_col = st.columns(2)
    with h_col:
        h_msg = f"**Strategy Health ({health.status}):** {health.headline}\n\n" + " · ".join([
            f"24m IC `{health.trailing_ic:+.3f}`",
            f"36m Sharpe `{health.trailing_sharpe:.2f}`",
            f"DSR `{health.deflated_sharpe:.0%}`",
            f"DD `{health.current_drawdown:.1%}`",
        ])
        if health.status == "GREEN":
            st.success(h_msg)
        elif health.status == "YELLOW":
            st.warning(h_msg)
        else:
            st.error(h_msg)
    with v_col:
        st.info(f"**Verdict: `{plan.verdict}`** ({decision.strategy_label})\n\n{plan.verdict_reason}")

    # KPI Row
    m = st.columns(5)
    m[0].metric("Existing holdings", f"{plan.existing_equity_value:,.0f} {cur}")
    m[1].metric("Cash + Deposit", f"{plan.starting_cash:,.0f} {cur}", f"+{plan.deposit_amount:,.0f} new")
    m[2].metric("Total portfolio", f"{plan.total_portfolio_value:,.0f} {cur}", f"Exposure {plan.exposure:.0%}")
    m[3].metric("Est. Broker + FX fees", f"{plan.total_fees:,.0f} {cur}",
                f"{plan.total_fees / max(plan.total_portfolio_value, 1.0):.2%} of NAV", delta_color="inverse")
    m[4].metric("Projected cash left", f"{plan.projected_cash_left:,.0f} {cur}")

    # Ordered Execution Tickets
    st.subheader(f"Execution Ticket — {decision.as_of:%B %Y} (prices as of {decision.price_date:%Y-%m-%d})")
    ord_cols = ["action", "name", "sector", "rank", "current_shares", "trade_shares", "target_shares",
                "price_local", "currency", f"trade_value_{cur}", f"est_fee_{cur}", "post_weight", "reason"]

    if not plan.sells.empty:
        st.markdown("#### Step 1 · SELL Orders (execute first)")
        s_view = plan.sells[ord_cols].copy()
        s_view["post_weight"] = s_view["post_weight"].map(lambda v: f"{v:.1%}")
        table(s_view)

    if not plan.buys.empty:
        st.markdown("#### Step 2 · BUY Orders (execute after sells settle)")
        b_view = plan.buys[ord_cols].copy()
        b_view["post_weight"] = b_view["post_weight"].map(lambda v: f"{v:.1%}")
        table(b_view)
    elif plan.sells.empty:
        st.success("No BUY or SELL orders needed this month.")

    if not plan.holds.empty:
        with st.expander(f"Step 3 · HOLD / Skipped Micro-Trades ({len(plan.holds)} positions)"):
            h_view = plan.holds[ord_cols].copy()
            h_view["post_weight"] = h_view["post_weight"].map(lambda v: f"{v:.1%}")
            table(h_view)

    # Downloads row
    brief_md = render_monthly_brief(decision)
    updated_state = apply_execution_plan(live_state, plan)
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇ Download Monthly Brief (.md)", brief_md.encode("utf-8"),
                       f"decision_{decision.as_of:%Y-%m}.md", "text/markdown")
    d2.download_button("⬇ Download Order Ticket (.csv)", plan.orders.to_csv().encode("utf-8"),
                       f"orders_{decision.as_of:%Y-%m}.csv", "text/csv")
    d3.download_button("⬇ Download Updated portfolio.json",
                       json.dumps(updated_state.to_dict(), indent=2).encode("utf-8"),
                       "my_portfolio.json", "application/json")

    # Target portfolio breakdown (Country & Sector donuts)
    tb = decision.target_table
    c1, c2 = st.columns(2)
    cw = tb.groupby("country")["weight"].sum()
    fig_c = go.Figure(go.Pie(
        labels=[f"{n}" for n in tb["name"]], values=tb["weight"], hole=0.55, sort=False,
        marker=dict(colors=[COUNTRY_COLORS.get(c, "#888") for c in tb["country"]], line=dict(color="white", width=2)),
        textinfo="none", hovertemplate="%{label}: %{value:.1%}<extra></extra>",
    ))
    fig_c.update_layout(showlegend=False, annotations=[dict(
        text="<br>".join(f"{k} {v:.0%}" for k, v in cw.items()), showarrow=False, font_size=15,
    )])
    with c1:
        show(style_fig(fig_c, 320, title="Target Allocation by Stock & Country"))

    sw_sec = tb.groupby("sector")["weight"].sum().sort_values(ascending=False)
    fig_s = go.Figure(go.Bar(x=sw_sec.values, y=sw_sec.index, orientation="h",
                             marker_color=STRATEGY_COLOR, hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig_s.update_yaxes(autorange="reversed")
    with c2:
        show(style_fig(fig_s, 320, ".0%", "Target Allocation by Sector"))

    with st.expander("Full ranked stock universe"):
        a = decision.all_scores.copy()
        a["score"] = a["score"].round(3)
        table(a, height=420)

    if want_imp and decision.importance is not None:
        imp = decision.importance.sort_values()
        fig = go.Figure(go.Bar(x=imp.values, y=[FEATURE_DESCRIPTIONS.get(i, i) for i in imp.index], orientation="h",
                               marker_color=np.where(imp.values >= 0, "#2a78d6", "#c3c2b7"),
                               hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
        show(style_fig(fig, 540, title="Out-of-sample permutation importance (held-out recent 24 months)"))

# =========================================================================== Tab 1: Optimal Strategy (Meta)
with tabs[1]:
    st.markdown("Evaluates diverse **(model × optimiser)** strategies strictly out-of-sample on trailing "
                "realised net returns, adjusts for multiple testing with the **Deflated Sharpe Ratio**, and "
                "blends the top strategies walk-forward.")
    c1, c2, c3 = st.columns(3)
    meta_mode = c1.selectbox("Selection mode", ["blend", "winner"],
                             format_func={"blend": "Softmax Blend (DSR-shrunk)", "winner": "Winner-take-all"}.get)
    meta_lb = c2.slider("Trailing evaluation window (months)", 12, 60, 36, 6)
    meta_fast = c3.toggle("Fast 5-strategy set", value=True, key="meta_tab_fast")

    with st.spinner("Running leak-free walk-forward meta-strategy backtest..."):
        mres = get_meta_backtest(source, cfg_key(cfg), meta_fast, meta_mode, int(meta_lb))

    st.subheader("Walk-Forward Candidate Leaderboard")
    lb = mres.leaderboard.copy()
    pct_cols = ["CAGR", "Volatility", "Deflated Sharpe", "Max drawdown", "Excess CAGR (vs EW)",
                "Turnover", "Annual cost", "Current weight"]
    lb_fmt = lb.copy().astype(object)
    for col in lb.columns:
        if col in pct_cols:
            lb_fmt[col] = [pct(v) for v in lb[col]]
        elif col in ("Sharpe", "Info ratio", "Mean IC"):
            lb_fmt[col] = ["–" if pd.isna(v) else f"{v:.2f}" for v in lb[col]]
    table(lb_fmt)

    fig = go.Figure()
    eq_bm = (1 + mres.meta_result.benchmarks[ref]).cumprod()
    fig.add_scatter(x=eq_bm.index, y=eq_bm, name=ref, line=dict(color="#9a9990", width=1.6, dash="dash"))
    for i, (cname, cres) in enumerate(mres.candidates.items()):
        e = (1 + cres.returns).cumprod()
        fig.add_scatter(x=e.index, y=e, name=cname, line=dict(color=SERIES_COLORS[i % len(SERIES_COLORS)], width=1.4), opacity=0.7)
    eq_meta = (1 + mres.meta_result.returns).cumprod()
    fig.add_scatter(x=eq_meta.index, y=eq_meta, name="Meta-Strategy (Walk-Forward)",
                    line=dict(color=STRATEGY_COLOR, width=3.2))
    fig.update_yaxes(type="log")
    show(style_fig(fig, 440, title=f"Walk-Forward Meta-Strategy vs Candidates (Growth of 1 {cfg.base_currency}, net of costs)"))

    st.subheader("Dynamic Strategy Weights Over Time")
    sw_df = mres.strategy_weights
    fig_sw = go.Figure()
    for i, col in enumerate(sw_df.columns):
        fig_sw.add_scatter(x=sw_df.index, y=sw_df[col], name=col, stackgroup="one",
                           line=dict(width=0.5, color=SERIES_COLORS[i % len(SERIES_COLORS)]),
                           hovertemplate="%{y:.0%}")
    show(style_fig(fig_sw, 300, ".0%", "How the meta-engine shifted weight across strategies walk-forward"))

    st.subheader("Meta-Strategy Sub-Period Stability")
    sp = M.subperiod_summary(mres.meta_result.returns, mres.meta_result.benchmarks[ref], rf=cfg.rf_annual)
    if not sp.empty:
        sp_fmt = sp.copy().astype(object)
        for col in ["CAGR", "Max drawdown", "Bench CAGR", "Excess CAGR"]:
            if col in sp_fmt:
                sp_fmt[col] = [pct(v) for v in sp[col]]
        sp_fmt["Sharpe"] = [f"{v:.2f}" if pd.notna(v) else "–" for v in sp["Sharpe"]]
        sp_fmt["Months"] = sp["Months"].astype(int)
        table(sp_fmt)

# =========================================================================== Tab 2: Performance
with tabs[2]:
    c1, c2 = st.columns([3, 1])
    shown = c2.multiselect("Show benchmarks", bench_names, default=[ref] + ([EW_UNIVERSE] if ref != EW_UNIVERSE else []))
    log_scale = c2.toggle("Log scale", value=True)
    show_gross = c2.toggle("Show before-cost line", value=False)
    eq = res.equity_curves()
    fig = go.Figure()
    for name in shown:
        fig.add_scatter(x=eq.index, y=eq[name], name=name, line=dict(color=color_for(name), width=1.8))
    if show_gross:
        g = (1 + res.gross_returns).cumprod()
        fig.add_scatter(x=g.index, y=g, name="Strategy (before costs)",
                        line=dict(color=STRATEGY_COLOR, width=1.5, dash="dot"))
    fig.add_scatter(x=eq.index, y=eq["Strategy"], name="Strategy", line=dict(color=STRATEGY_COLOR, width=3))
    fig.update_traces(hovertemplate="%{y:.2f}×")
    if log_scale:
        fig.update_yaxes(type="log")
    with c1:
        show(style_fig(fig, 460, title=f"Growth of 1 {res.base_currency} (net of costs)"))

    dd = pd.DataFrame({"Strategy": M.drawdown(r), ref: M.drawdown(b)})
    fig = go.Figure()
    fig.add_scatter(x=dd.index, y=dd[ref], name=ref, line=dict(color=color_for(ref), width=1.5))
    fig.add_scatter(x=dd.index, y=dd["Strategy"], name="Strategy", fill="tozeroy",
                    line=dict(color=STRATEGY_COLOR, width=2), fillcolor="rgba(42,120,214,0.15)")
    fig.update_traces(hovertemplate="%{y:.1%}")
    show(style_fig(fig, 280, ".0%", "Drawdown"))

    st.subheader("Performance summary")
    comp = M.comparison_table(r, res.benchmarks, ref=ref, rf=cfg.rf_annual)
    pct_rows = ["CAGR", "Volatility", "Max drawdown", "Best month", "Worst month", "% positive months",
                "Total return", "Excess CAGR", "Alpha (ann.)", "Tracking error", "% months beating"]
    fmt = comp.copy().astype(object)
    for row in comp.index:
        fmt.loc[row] = [pct(v) if row in pct_rows else ("–" if pd.isna(v) else f"{v:.2f}") for v in comp.loc[row]]
    table(fmt)

    st.subheader("Calendar-year returns")
    yr = M.calendar_year_returns(pd.concat([r.rename("Strategy"), res.benchmarks[[ref]]], axis=1))
    fig = go.Figure()
    fig.add_bar(x=yr.index.astype(str), y=yr[ref], name=ref, marker_color=color_for(ref))
    fig.add_bar(x=yr.index.astype(str), y=yr["Strategy"], name="Strategy", marker_color=STRATEGY_COLOR)
    fig.update_traces(hovertemplate="%{y:.1%}")
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    show(style_fig(fig, 340, ".0%"))

    st.subheader("Monthly returns heatmap (strategy)")
    mm = r.to_frame("r")
    mm["Year"], mm["Month"] = mm.index.year, mm.index.strftime("%b")
    hm = mm.pivot(index="Year", columns="Month", values="r")
    hm = hm[[m for m in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
             if m in hm.columns]]
    lim = float(np.nanmax(np.abs(hm.values)))
    fig = go.Figure(go.Heatmap(z=hm.values, x=hm.columns, y=hm.index.astype(str), colorscale=DIVERGING,
                               zmid=0, zmin=-lim, zmax=lim,
                               text=np.vectorize(lambda v: "" if np.isnan(v) else f"{v:.1%}")(hm.values),
                               texttemplate="%{text}", hovertemplate="%{y} %{x}: %{z:.2%}<extra></extra>",
                               xgap=2, ygap=2, colorbar=dict(tickformat=".0%")))
    fig.update_yaxes(autorange="reversed")
    show(style_fig(fig, 28 * len(hm) + 80))

# =========================================================================== Tab 3: Is it real?
with tabs[3]:
    st.markdown("A backtest can look good by luck, by overfitting, or because the universe itself did well. "
                "These tests separate genuine forecasting skill from those effects.")
    st.subheader("1 · Is the outperformance statistically significant?")
    rows = []
    for bn in bench_names:
        t = M.excess_return_test(r, res.benchmarks[bn])
        rows.append({"Benchmark": bn, "Excess CAGR": pct(M.cagr(r) - M.cagr(res.benchmarks[bn])),
                     "Information ratio": f"{M.information_ratio(r, res.benchmarks[bn]):.2f}",
                     "t-stat (monthly excess)": f"{t['t_stat']:.2f}", "one-sided p-value": f"{t['p_value']:.3f}",
                     "P(outperform) bootstrap": pct(t["p_outperform_bootstrap"], 0)})
    table(pd.DataFrame(rows).set_index("Benchmark"))
    psr = M.probabilistic_sharpe(r, M.sharpe(b, rf=cfg.rf_annual), rf=cfg.rf_annual)
    dsr_7 = M.deflated_sharpe(r, n_trials=len(DEFAULT_CANDIDATES), rf=cfg.rf_annual)
    st.caption(f"Probabilistic Sharpe Ratio vs benchmark: **{pct(psr, 0)}** · "
               f"Deflated Sharpe Ratio (adjusted for {len(DEFAULT_CANDIDATES)} candidate trials): **{pct(dsr_7, 0)}**.")

    st.subheader("2 · Does it beat random stock picking from the same universe?")
    rp = M.random_portfolio_percentile(res.gross_returns, res.random_returns)
    c1, c2 = st.columns([3, 1])
    fig = go.Figure()
    fig.add_histogram(x=rp["random_cagr"], nbinsx=40, marker_color="#86b6ef", name="Random portfolios",
                      marker_line=dict(color="white", width=1), hovertemplate="CAGR %{x:.1%}: %{y}<extra></extra>")
    fig.add_vline(x=rp["strategy_cagr"], line=dict(color=STRATEGY_COLOR, width=3),
                  annotation_text=f"Strategy {rp['strategy_cagr']:.1%}", annotation_position="top")
    fig.update_xaxes(tickformat=".0%", title="CAGR (before costs)")
    with c1:
        show(style_fig(fig, 320))
    c2.metric("Percentile vs random (CAGR)", pct(rp["cagr_percentile"], 0))
    c2.metric("Percentile vs random (Sharpe)", pct(rp["sharpe_percentile"], 0))

    st.subheader("3 · Does the model rank stocks correctly?")
    c1, c2 = st.columns(2)
    ic = res.ic
    fig = go.Figure()
    fig.add_bar(x=ic.index, y=ic, name="Monthly IC",
                marker_color=np.where(ic >= 0, "#2a78d6", "#e34948"), hovertemplate="%{y:.3f}")
    fig.add_scatter(x=ic.index, y=ic.rolling(12).mean(), name="12-month average", line=dict(color="#0b0b0b", width=2))
    ic_t = ic.mean() / (ic.std() / np.sqrt(len(ic))) if len(ic) > 1 and ic.std() > 0 else 0.0
    with c1:
        show(style_fig(fig, 320, title=f"Rank IC: mean {ic.mean():.3f}, t-stat {ic_t:.2f}, positive {(ic > 0).mean():.0%} of months"))
    qm = res.quantile_returns.mean() * 12
    fig = go.Figure(go.Bar(x=qm.index, y=qm.values, marker_color=["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab"][:len(qm)],
                           hovertemplate="%{x}: %{y:.1%}/yr<extra></extra>"))
    with c2:
        show(style_fig(fig, 320, ".0%", "Avg. annualised return by forecast quintile (Q5 = best)"))

    st.subheader("4 · Is performance stable over time?")
    c1, c2 = st.columns(2)
    rx = M.rolling_excess(r, b, 12)
    fig = go.Figure(go.Scatter(x=rx.index, y=rx, fill="tozeroy", line=dict(color=STRATEGY_COLOR, width=1.5),
                               fillcolor="rgba(42,120,214,0.15)", hovertemplate="%{y:.1%}"))
    with c1:
        show(style_fig(fig, 300, ".0%", f"Rolling 12-month excess return vs {ref}"))
    rs = pd.DataFrame({"Strategy": M.rolling_sharpe(r, rf=cfg.rf_annual), ref: M.rolling_sharpe(b, rf=cfg.rf_annual)})
    fig = go.Figure()
    fig.add_scatter(x=rs.index, y=rs[ref], name=ref, line=dict(color=color_for(ref), width=1.5))
    fig.add_scatter(x=rs.index, y=rs["Strategy"], name="Strategy", line=dict(color=STRATEGY_COLOR, width=2.5))
    with c2:
        show(style_fig(fig, 300, title="Rolling 3-year Sharpe ratio"))

# =========================================================================== Tab 4: Holdings
with tabs[4]:
    w = res.weights
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg. monthly turnover", pct(res.turnover.iloc[1:].mean(), 0))
    c2.metric("Annual cost drag", pct(res.costs.iloc[1:].mean() * 12, 2))
    c3.metric("Avg. holding period", f"{1 / max(res.turnover.iloc[1:].mean(), 1e-6):.1f} months")

    country_w = w.T.groupby([ticker_country(t) for t in w.columns]).sum().T
    fig = go.Figure()
    for c in ["US", "SE"]:
        if c in country_w:
            fig.add_scatter(x=country_w.index, y=country_w[c], name={"US": "United States", "SE": "Sweden"}[c],
                            stackgroup="one", line=dict(width=0.5, color=COUNTRY_COLORS[c]),
                            hovertemplate="%{y:.0%}")
    show(style_fig(fig, 260, ".0%", "Country allocation over time"))

    sector_w = w.T.groupby([ticker_sector(t) for t in w.columns]).sum().T
    fig_sec = go.Figure()
    for i, sec in enumerate(sector_w.columns):
        fig_sec.add_scatter(x=sector_w.index, y=sector_w[sec], name=sec, stackgroup="one",
                            line=dict(width=0.5, color=SERIES_COLORS[i % len(SERIES_COLORS)]),
                            hovertemplate="%{y:.0%}")
    show(style_fig(fig_sec, 280, ".0%", "Sector allocation over time"))

    freq = (w > 0).sum().sort_values(ascending=False)
    top = freq.head(20)
    fig = go.Figure(go.Bar(x=top.values, y=[f"{res.names.get(t, t)} ({t})" for t in top.index], orientation="h",
                           marker_color=[COUNTRY_COLORS[ticker_country(t)] for t in top.index],
                           hovertemplate="%{y}: %{x} months<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    show(style_fig(fig, 520, title="Most frequently held stocks (months held; blue = US, yellow = Sweden)"))

    st.subheader("Portfolio at a given date")
    dates = list(w.index)
    d = st.select_slider("Rebalance date", options=dates, value=dates[-1], format_func=lambda x: x.strftime("%Y-%m"))
    wd = w.loc[d][w.loc[d] > 0].sort_values(ascending=False)
    fwd = res.features["fwd_ret"].xs(d, level="date").reindex(wd.index) if res.features is not None else None
    snap = pd.DataFrame({"Name": [res.names.get(t, t) for t in wd.index],
                         "Country": [ticker_country(t) for t in wd.index],
                         "Sector": [ticker_sector(t) for t in wd.index],
                         "Weight": wd.map(lambda v: f"{v:.1%}")}, index=wd.index)
    if fwd is not None:
        snap["Return next month"] = fwd.map(lambda v: pct(v))
        snap["Contribution"] = (wd * fwd.fillna(0)).map(lambda v: pct(v, 2))
    table(snap)

# =========================================================================== Tab 5: Features
with tabs[5]:
    st.markdown("How well does each input predict next month's relative return **on its own**? "
                "(Spearman rank IC across stocks, averaged over completed months.)")
    fi = res.feature_ic.copy().sort_values("mean_ic")
    fig = go.Figure(go.Bar(x=fi["mean_ic"], y=[FEATURE_DESCRIPTIONS.get(i, i) for i in fi.index], orientation="h",
                           marker_color=np.where(fi["mean_ic"] >= 0, "#2a78d6", "#e34948"),
                           customdata=fi[["t_stat", "hit_rate"]].values,
                           hovertemplate="%{y}<br>mean IC %{x:.3f}<br>t-stat %{customdata[0]:.2f}"
                                         "<br>positive %{customdata[1]:.0%} of months<extra></extra>"))
    show(style_fig(fig, 620, title="Mean rank IC per feature (blue = higher value → better future return)"))
    table(fi.sort_values("mean_ic", key=np.abs, ascending=False).round(3))

# =========================================================================== Tab 6: Method
with tabs[6]:
    st.markdown(Path(__file__).with_name("METHOD.md").read_text(encoding="utf-8"))
