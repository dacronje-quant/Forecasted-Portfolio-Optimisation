"""Interactive dashboard for the Forecasted Portfolio Optimisation strategy.

Run with:   streamlit run app/dashboard.py
"""

from __future__ import annotations

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
from fpo.config import MODELS, OPTIMIZERS, BacktestConfig  # noqa: E402
from fpo.data import load_market_data, synthetic_market_data  # noqa: E402
from fpo.features import FEATURE_DESCRIPTIONS, build_features  # noqa: E402
from fpo.recommend import allocate_whole_shares, recommend_portfolio  # noqa: E402
from fpo.universe import ticker_country  # noqa: E402

st.set_page_config(page_title="Forecasted Portfolio Optimisation", page_icon="📈", layout="wide")

# Validated categorical palette (fixed order; colour follows the entity)
STRATEGY_COLOR = "#2a78d6"
SERIES_COLORS = ["#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BENCH_ORDER = ["S&P 500 (SPY)", "MSCI World (URTH)", "OMX Stockholm 30 (price index)",
               EW_UNIVERSE, BLEND]
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
    "mean_variance": "Mean-variance (forecast + shrunk covariance)",
    "min_variance": "Minimum variance",
    "hrp": "Hierarchical risk parity",
    "inverse_vol": "Inverse volatility",
    "score": "Score-weighted",
    "equal": "Equal weight",
}


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
    has_legend = sum(1 for t in fig.data if t.showlegend is not False) > 1
    top = 10 + (34 if title else 0) + (28 if has_legend else 0)
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=top, b=10),
        title=dict(text=title, x=0, xanchor="left", y=1, yanchor="top", yref="container",
                   pad=dict(t=8), font=dict(size=15)) if title else None,
        hovermode="x unified", showlegend=has_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, yref="paper", x=0, font=dict(size=12)),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    if yfmt:
        fig.update_yaxes(tickformat=yfmt)
    return fig


def log_axis(fig: go.Figure, values) -> None:
    """Log y-axis with a few clean 'x-times' labels instead of Plotly's crowded minor ticks."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    lo, hi = (v.min(), v.max()) if len(v) else (0.5, 2)
    ticks = [t for t in (0.25, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100, 200) if lo / 1.3 <= t <= hi * 1.3]
    fig.update_yaxes(type="log", tickvals=ticks, ticktext=[f"{t:g}×" for t in ticks], minor=dict(showgrid=False))


def pct(x, d=1):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def color_for(name: str) -> str:
    return STRATEGY_COLOR if name.startswith("Strategy") else BENCH_COLORS.get(name, "#888888")


# --------------------------------------------------------------------------- cached compute
@st.cache_data(show_spinner=False, ttl=6 * 3600)
def get_data(source: str, markets: str, currency: str, start: str):
    if source == "synthetic":
        return synthetic_market_data(markets, start=start, base_currency=currency)
    return load_market_data(markets, start=start, base_currency=currency)


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def get_features(source: str, markets: str, currency: str, start: str):
    return build_features(get_data(source, markets, currency, start).prices)


@st.cache_data(show_spinner=False, ttl=6 * 3600, max_entries=40)
def get_backtest(source: str, cfg_items: tuple):
    cfg = BacktestConfig(**dict(cfg_items))
    data = get_data(source, cfg.markets, cfg.base_currency, cfg.start)
    feats = get_features(source, cfg.markets, cfg.base_currency, cfg.start)
    return run_backtest(data, cfg, features=feats)


@st.cache_data(show_spinner=False, ttl=6 * 3600, max_entries=20)
def get_recommendation(source: str, cfg_items: tuple, importance: bool):
    cfg = BacktestConfig(**dict(cfg_items))
    data = get_data(source, cfg.markets, cfg.base_currency, cfg.start)
    feats = get_features(source, cfg.markets, cfg.base_currency, cfg.start)
    return recommend_portfolio(data, cfg, budget=100_000, features=feats, compute_importance=importance)


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

    st.markdown("**Forecasting model**")
    model = st.selectbox("Model", MODELS, index=MODELS.index("gbm"), format_func=MODEL_LABELS.get,
                         help="Ensemble is the most robust but ~10x slower to backtest.")
    c1, c2 = st.columns(2)
    retrain = c1.number_input("Retrain every (months)", 1, 24, 3)
    window = c2.number_input("Train window (months, 0=all)", 0, 240, 0, step=12)
    target = st.selectbox("Target", ["rank", "excess"],
                          format_func={"rank": "Cross-sectional rank", "excess": "Excess return vs median"}.get)

    st.markdown("**Portfolio construction**")
    optimizer = st.selectbox("Optimiser", OPTIMIZERS, format_func=OPT_LABELS.get)
    n_hold = st.slider("Number of stocks", 5, 40, 15)
    c1, c2 = st.columns(2)
    max_w = c1.slider("Max weight", 0.03, 0.40, 0.12, 0.01)
    min_w = c2.slider("Min weight", 0.0, 0.05, 0.02, 0.005)
    max_country = st.slider("Max weight in one country", 0.3, 1.0, 1.0, 0.05,
                            help="1.0 = no limit. 0.7 keeps at least 30% in the other market.")
    c1, c2 = st.columns(2)
    risk_aversion = c1.number_input("Risk aversion", 0.5, 50.0, 5.0, 0.5,
                                    help="Mean-variance only. Higher = closer to minimum variance.")
    buffer = c2.number_input("Hold buffer", 1.0, 3.0, 1.5, 0.1,
                             help="Keep a holding while it ranks within N × buffer. Cuts turnover.")

    st.markdown("**Costs**")
    c1, c2 = st.columns(2)
    cost = c1.number_input("Trading cost (bps)", 0.0, 100.0, 15.0, 1.0)
    fx_cost = c2.number_input("FX cost (bps)", 0.0, 100.0, 10.0, 1.0,
                              help="Charged on trades in foreign-currency stocks (e.g. US stocks for SEK investors).")
    submitted = st.form_submit_button("▶ Run backtest", type="primary")

cfg = BacktestConfig(
    markets=markets, base_currency=currency, start=start, backtest_start=bt_start, model=model,
    target=target, train_window_months=int(window), retrain_every=int(retrain), optimizer=optimizer,
    n_holdings=int(n_hold), max_weight=float(max_w), min_weight=float(min_w),
    max_country_weight=float(max_country), risk_aversion=float(risk_aversion), hold_buffer=float(buffer),
    cost_bps=float(cost), fx_cost_bps=float(fx_cost),
)

if submitted or "cfg" not in st.session_state:
    st.session_state["cfg"] = cfg
    st.session_state["source"] = source
cfg = st.session_state["cfg"]
source = st.session_state["source"]

# --------------------------------------------------------------------------- run
st.title("📈 Forecasted Portfolio Optimisation")
st.caption(f"{MODEL_LABELS[cfg.model]} · {OPT_LABELS[cfg.optimizer]} · {cfg.n_holdings} stocks · "
           f"{'US + Sweden' if cfg.markets == 'both' else cfg.markets.upper()} · returns in {cfg.base_currency} · "
           f"monthly rebalancing · costs {cfg.cost_bps:.0f}+{cfg.fx_cost_bps:.0f} bps")

try:
    with st.spinner("Loading prices..." if source == "yahoo" else "Simulating market..."):
        data = get_data(source, cfg.markets, cfg.base_currency, cfg.start)
    with st.spinner(f"Running walk-forward backtest ({MODEL_LABELS[cfg.model]})... "
                    "the first run of a configuration can take a minute or two"):
        res = get_backtest(source, cfg_key(cfg))
except Exception as e:  # pragma: no cover - UI feedback
    st.error(f"Could not run the backtest: {e}")
    if source == "yahoo":
        st.info("If Yahoo Finance is unreachable, switch the data source to **Simulated demo** in the sidebar.")
    st.stop()

if res.source == "synthetic":
    st.warning("You are looking at **simulated** data. Switch to Yahoo Finance in the sidebar for real results.")

bench_names = list(res.benchmarks.columns)
default_ref = bench_names.index(BLEND) if BLEND in bench_names else 0
ref = st.selectbox("Main benchmark", bench_names, index=default_ref,
                   help="The index fund you would otherwise buy. 'Equal-weight universe' measures pure stock-picking skill.")
r = res.returns
b = res.benchmarks[ref]

s_sum = M.summary(r, b)
b_sum = M.summary(b)
k = st.columns(3) + st.columns(3)
k[0].metric("CAGR", pct(s_sum["CAGR"]), f"{(s_sum['CAGR'] - b_sum['CAGR']) * 100:+.1f} pp vs benchmark")
k[1].metric("Sharpe", f"{s_sum['Sharpe']:.2f}", f"{s_sum['Sharpe'] - b_sum['Sharpe']:+.2f}")
k[2].metric("Max drawdown", pct(s_sum["Max drawdown"], 0),
            f"{(s_sum['Max drawdown'] - b_sum['Max drawdown']) * 100:+.1f} pp")
k[3].metric("Volatility", pct(s_sum["Volatility"]),
            f"{(s_sum['Volatility'] - b_sum['Volatility']) * 100:+.1f} pp", delta_color="inverse")
k[4].metric("Months beating benchmark", pct(s_sum["% months beating"], 0),
            help="Share of months in which the strategy's return was higher than the benchmark's.")
test = M.excess_return_test(r, b)
k[5].metric("P(outperform)", pct(test["p_outperform_bootstrap"], 0),
            help="Block-bootstrap probability that the strategy's cumulative return beats the benchmark.")

tabs = st.tabs(["📈 Performance", "🧪 Is it real?", "🧺 Holdings", "🛒 This month's portfolio",
                "🔬 Features", "⚖️ Compare models", "ℹ️ Method"])

# =========================================================================== Performance
with tabs[0]:
    c1, c2, c3 = st.columns([4, 1, 1], vertical_alignment="bottom")
    shown = c1.multiselect("Compare with", bench_names, default=[ref] + ([EW_UNIVERSE] if ref != EW_UNIVERSE else []))
    log_scale = c2.toggle("Log scale", value=True)
    show_gross = c3.toggle("Before costs", value=False, help="Also show the strategy before trading costs.")
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
    style_fig(fig, 460, title=f"Growth of 1 {res.base_currency} (net of costs)")
    if log_scale:
        log_axis(fig, eq[["Strategy"] + shown].values)
    show(fig)

    dd = pd.DataFrame({"Strategy": M.drawdown(r), ref: M.drawdown(b)})
    fig = go.Figure()
    fig.add_scatter(x=dd.index, y=dd[ref], name=ref, line=dict(color=color_for(ref), width=1.5))
    fig.add_scatter(x=dd.index, y=dd["Strategy"], name="Strategy", fill="tozeroy",
                    line=dict(color=STRATEGY_COLOR, width=2), fillcolor="rgba(42,120,214,0.15)")
    fig.update_traces(hovertemplate="%{y:.1%}")
    show(style_fig(fig, 280, ".0%", "Drawdown"))

    st.subheader("Performance summary")
    comp = M.comparison_table(r, res.benchmarks, ref=ref)
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
                               zmid=0, zmin=-lim, zmax=lim, text=np.vectorize(lambda v: "" if np.isnan(v) else f"{v:.1%}")(hm.values),
                               texttemplate="%{text}", hovertemplate="%{y} %{x}: %{z:.2%}<extra></extra>",
                               xgap=2, ygap=2, colorbar=dict(tickformat=".0%")))
    fig.update_yaxes(autorange="reversed")
    show(style_fig(fig, 28 * len(hm) + 80))

# =========================================================================== Is it real?
with tabs[1]:
    st.markdown("A backtest can look good by luck, by overfitting, or because the universe itself did well. "
                "These tests try to separate genuine forecasting skill from those effects.")
    st.subheader("1 · Is the outperformance statistically significant?")
    rows = []
    for bn in bench_names:
        t = M.excess_return_test(r, res.benchmarks[bn])
        rows.append({"Benchmark": bn, "Excess CAGR": pct(M.cagr(r) - M.cagr(res.benchmarks[bn])),
                     "Info ratio": f"{M.information_ratio(r, res.benchmarks[bn]):.2f}",
                     "t-stat": f"{t['t_stat']:.2f}", "p-value": f"{t['p_value']:.3f}",
                     "P(outperform)": pct(t["p_outperform_bootstrap"], 0)})
    table(pd.DataFrame(rows).set_index("Benchmark"))
    st.caption("t-stat and p-value test the average monthly excess return (one-sided). "
               "P(outperform) is a block-bootstrap probability that cumulative return beats the benchmark.")
    psr = M.probabilistic_sharpe(r, M.sharpe(b))
    st.caption(f"Probabilistic Sharpe ratio: **{pct(psr, 0)}** probability that the strategy's true Sharpe exceeds "
               f"the benchmark's ({M.sharpe(b):.2f}), adjusting for track-record length, skew and fat tails. "
               "Rule of thumb: t-stat > 2 and p < 0.05 are needed before trusting an edge.")

    st.subheader("2 · Does it beat random stock picking from the same universe?")
    rp = M.random_portfolio_percentile(res.gross_returns, res.random_returns)
    c1, c2 = st.columns([3, 1])
    fig = go.Figure()
    fig.add_histogram(x=rp["random_cagr"], nbinsx=40, marker_color="#86b6ef", name="Random portfolios",
                      marker_line=dict(color="white", width=1), hovertemplate="CAGR %{x:.1%}: %{y}<extra></extra>")
    fig.add_vline(x=rp["strategy_cagr"], line=dict(color=STRATEGY_COLOR, width=3),
                  annotation_text=f"Strategy {rp['strategy_cagr']:.1%}", annotation_position="top right",
                  annotation_yshift=-4)
    fig.update_xaxes(tickformat=".0%", title="CAGR (before costs)")
    fig.update_layout(showlegend=False)
    with c1:
        show(style_fig(fig, 320, title="Random 15-stock portfolios vs the strategy"))
    c2.metric("Percentile vs random (CAGR)", pct(rp["cagr_percentile"], 0))
    c2.metric("Percentile vs random (Sharpe)", pct(rp["sharpe_percentile"], 0))
    c2.caption(f"{len(rp['random_cagr'])} equal-weight portfolios of {cfg.n_holdings} random stocks, redrawn monthly. "
               "Above 95% suggests genuine selection skill.")

    st.subheader("3 · Does the model rank stocks correctly?")
    c1, c2 = st.columns(2)
    ic = res.ic
    fig = go.Figure()
    fig.add_bar(x=ic.index, y=ic, name="Monthly IC",
                marker_color=np.where(ic >= 0, "#2a78d6", "#e34948"), hovertemplate="%{y:.3f}")
    fig.add_scatter(x=ic.index, y=ic.rolling(12).mean(), name="12-month average", line=dict(color="#0b0b0b", width=2))
    ic_t = ic.mean() / (ic.std() / np.sqrt(len(ic)))
    with c1:
        show(style_fig(fig, 320, title=f"Rank IC: mean {ic.mean():.3f}, t-stat {ic_t:.2f}, positive {(ic > 0).mean():.0%} of months"))
    qm = res.quantile_returns.mean() * 12
    fig = go.Figure(go.Bar(x=qm.index, y=qm.values, marker_color=["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab"][:len(qm)],
                           hovertemplate="%{x}: %{y:.1%}/yr<extra></extra>"))
    with c2:
        show(style_fig(fig, 320, ".0%", "Avg. annualised return by forecast quintile (Q5 = best)"))
    ls = (1 + (res.quantile_returns.iloc[:, -1] - res.quantile_returns.iloc[:, 0])).cumprod()
    fig = go.Figure(go.Scatter(x=ls.index, y=ls, line=dict(color=STRATEGY_COLOR, width=2), name="Q5 − Q1",
                               hovertemplate="%{y:.2f}×"))
    show(style_fig(fig, 260, title="Long best quintile / short worst quintile (cumulative, frictionless)"))
    st.caption("A skilful model shows positive IC most months, rising bars from Q1 to Q5, and a steadily rising long-short line.")

    st.subheader("4 · Is performance stable over time?")
    c1, c2 = st.columns(2)
    rx = M.rolling_excess(r, b, 12)
    fig = go.Figure(go.Scatter(x=rx.index, y=rx, fill="tozeroy", line=dict(color=STRATEGY_COLOR, width=1.5),
                               fillcolor="rgba(42,120,214,0.15)", hovertemplate="%{y:.1%}"))
    with c1:
        show(style_fig(fig, 300, ".0%", f"Rolling 12-month excess return vs {ref}"))
    rs = pd.DataFrame({"Strategy": M.rolling_sharpe(r), ref: M.rolling_sharpe(b)})
    fig = go.Figure()
    fig.add_scatter(x=rs.index, y=rs[ref], name=ref, line=dict(color=color_for(ref), width=1.5))
    fig.add_scatter(x=rs.index, y=rs["Strategy"], name="Strategy", line=dict(color=STRATEGY_COLOR, width=2.5))
    with c2:
        show(style_fig(fig, 300, title="Rolling 3-year Sharpe ratio"))

# =========================================================================== Holdings
with tabs[2]:
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
    show(style_fig(fig, 280, ".0%", "Country allocation over time"))

    freq = (w > 0).sum().sort_values(ascending=False)
    top = freq.head(20)
    fig = go.Figure(go.Bar(x=top.values, y=[f"{res.names.get(t, t)} ({t})" for t in top.index], orientation="h",
                           marker_color=[COUNTRY_COLORS[ticker_country(t)] for t in top.index],
                           hovertemplate="%{y}: %{x} months<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    show(style_fig(fig, 520, title="Most frequently held stocks (months held; blue = US, yellow = Sweden)"))

    fig = go.Figure(go.Bar(x=res.turnover.index, y=res.turnover, marker_color="#6da7ec", hovertemplate="%{y:.0%}"))
    show(style_fig(fig, 240, ".0%", "One-way turnover at each rebalance"))

    st.subheader("Portfolio at a given date")
    dates = list(w.index)
    d = st.select_slider("Rebalance date", options=dates, value=dates[-1], format_func=lambda x: x.strftime("%Y-%m"))
    wd = w.loc[d][w.loc[d] > 0].sort_values(ascending=False)
    fwd = res.features["fwd_ret"].xs(d, level="date").reindex(wd.index) if res.features is not None else None
    snap = pd.DataFrame({"Name": [res.names.get(t, t) for t in wd.index],
                         "Country": [ticker_country(t) for t in wd.index],
                         "Weight": wd.map(lambda v: f"{v:.1%}")}, index=wd.index)
    if fwd is not None:
        snap["Return next month"] = fwd.map(lambda v: pct(v))
        snap["Contribution"] = (wd * fwd.fillna(0)).map(lambda v: pct(v, 2))
    table(snap)

# =========================================================================== This month
with tabs[3]:
    st.markdown("The model is retrained on **all** history, then scores every stock at the latest month-end "
                "and the optimiser builds the portfolio with your settings.")
    c1, c2 = st.columns([1, 2])
    budget = c1.number_input(f"Amount to invest ({cfg.base_currency})", 1_000.0, 100_000_000.0, 50_000.0, 1_000.0)
    want_imp = c1.toggle("Feature importance", value=False, help="Explain which inputs drive the model (slower).")
    holdings_file = c2.file_uploader("Optional: your current holdings (CSV with columns ticker,shares)", type="csv")
    with st.spinner("Training on full history and building this month's portfolio..."):
        rec = get_recommendation(source, cfg_key(cfg), want_imp)
    cur = rec.base_currency
    tb = rec.table.copy()
    sh = allocate_whole_shares(tb["weight"], tb[f"price_{cur}"], budget,
                               max_weight=max(cfg.max_weight, float(tb["weight"].max())))
    tb["shares"] = pd.Series(sh)
    tb["target"] = tb["weight"] * budget
    tb["invested"] = tb["shares"] * tb[f"price_{cur}"]
    cash = budget - tb["invested"].sum()

    st.subheader(f"Portfolio for {rec.as_of:%B %Y}")
    too_pricey = tb.index[tb["shares"] == 0]
    note = (f" {', '.join(tb.loc[too_pricey, 'name'])}: one share costs more than its target amount, so it is "
            "skipped and the money spread over the others. Raise the amount to include it.") if len(too_pricey) else ""
    st.caption(f"Prices as of {rec.price_date:%Y-%m-%d}. Whole shares only; uninvested cash {cash:,.0f} {cur}.{note}")
    c1, c2 = st.columns([2, 1])
    view = pd.DataFrame({
        "Name": tb["name"], "Country": tb["country"], "Target": tb["weight"].map(lambda v: f"{v:.1%}"),
        "Actual": (tb["invested"] / budget).map(lambda v: f"{v:.1%}"),
        "Price": [f"{p:,.2f} {c}" for p, c in zip(tb["price_local"], tb["currency"])],
        "Shares to buy": tb["shares"], f"Amount ({cur})": tb["invested"].map(lambda v: f"{v:,.0f}"),
        "12-1 momentum": tb["mom_12_1"].map(lambda v: pct(v, 0)), "Volatility": tb["vol_3m"].map(lambda v: pct(v, 0)),
    })
    with c1:
        table(view)
        st.download_button("⬇ Download as CSV", tb.to_csv().encode(), f"portfolio_{rec.as_of:%Y-%m}.csv", "text/csv")
    fig = go.Figure(go.Pie(labels=[f"{n}" for n in tb["name"]], values=tb["weight"], hole=0.55, sort=False,
                           marker=dict(colors=[COUNTRY_COLORS[c] for c in tb["country"]], line=dict(color="white", width=2)),
                           textinfo="none", hovertemplate="%{label}: %{value:.1%}<extra></extra>"))
    cw = tb.groupby("country")["weight"].sum()
    fig.update_layout(showlegend=False, annotations=[dict(text="<br>".join(f"{k} {v:.0%}" for k, v in cw.items()),
                                                          showarrow=False, font_size=16)])
    with c2:
        show(style_fig(fig, 340))

    if holdings_file is not None:
        try:
            h = pd.read_csv(holdings_file)
            curh = pd.Series(dict(zip(h["ticker"], h["shares"])), dtype=float)
            idx = curh.index.union(tb.index)
            trades = pd.DataFrame({"Name": [rec.all_scores["name"].get(t, t) for t in idx],
                                   "Own now": curh.reindex(idx, fill_value=0),
                                   "Target": tb["shares"].reindex(idx, fill_value=0).astype(float)})
            trades["Trade"] = trades["Target"] - trades["Own now"]
            trades["Action"] = np.select([trades["Trade"] > 0, trades["Trade"] < 0], ["BUY", "SELL"], "HOLD")
            st.subheader("Trades to reach the target portfolio")
            table(trades.sort_values("Action"))
            st.caption("Target shares are based on the amount above; set it to your total portfolio value "
                       "(existing holdings + new money) for a full rebalance.")
        except Exception as e:
            st.error(f"Could not read holdings file: {e}")

    with st.expander("All stocks ranked by the model"):
        a = rec.all_scores.copy()
        a["score"] = a["score"].round(3)
        table(a, height=420)

    if want_imp and rec.importance is not None:
        imp = rec.importance.sort_values()
        fig = go.Figure(go.Bar(x=imp.values, y=[FEATURE_DESCRIPTIONS.get(i, i) for i in imp.index], orientation="h",
                               marker_color=np.where(imp.values >= 0, "#2a78d6", "#c3c2b7"),
                               hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
        show(style_fig(fig, 520, title="Permutation importance: drop in rank-IC when the feature is shuffled"))

# =========================================================================== Features
with tabs[4]:
    st.markdown("How well does each input predict next month's relative return **on its own**? "
                "(Spearman rank IC across stocks, averaged over months.)")
    fi = res.feature_ic.copy()
    fi = fi.sort_values("mean_ic")
    fig = go.Figure(go.Bar(x=fi["mean_ic"], y=[FEATURE_DESCRIPTIONS.get(i, i) for i in fi.index], orientation="h",
                           marker_color=np.where(fi["mean_ic"] >= 0, "#2a78d6", "#e34948"),
                           customdata=fi[["t_stat", "hit_rate"]].values,
                           hovertemplate="%{y}<br>mean IC %{x:.3f}<br>t-stat %{customdata[0]:.2f}"
                                         "<br>positive %{customdata[1]:.0%} of months<extra></extra>"))
    show(style_fig(fig, 560, title="Mean rank IC per feature (blue = higher value → better future return)"))
    ft = fi.sort_values("mean_ic", key=np.abs, ascending=False)
    table(pd.DataFrame({"Feature": [FEATURE_DESCRIPTIONS.get(i, i) for i in ft.index],
                        "Mean IC": ft["mean_ic"].map(lambda v: f"{v:+.3f}"),
                        "t-stat": ft["t_stat"].map(lambda v: f"{v:.2f}"),
                        "Positive months": ft["hit_rate"].map(lambda v: pct(v, 0))}), hide_index=True)

# =========================================================================== Compare
with tabs[5]:
    st.markdown("Run the **same settings** with different models to see whether machine learning adds anything "
                "over a simple momentum rule. Each run is cached.")
    picks = st.multiselect("Models", MODELS, default=["momentum", "ridge", "gbm"], format_func=MODEL_LABELS.get)
    if st.button("Run comparison", type="primary"):
        st.session_state["compare"] = picks
    picks = st.session_state.get("compare", [])
    if picks:
        results = {}
        prog = st.progress(0.0)
        for i, m in enumerate(picks):
            with st.spinner(f"Backtesting {MODEL_LABELS[m]}..."):
                results[m] = get_backtest(source, cfg_key(replace(cfg, model=m)))
            prog.progress((i + 1) / len(picks))
        prog.empty()
        fig = go.Figure()
        eqb = (1 + b).cumprod()
        fig.add_scatter(x=eqb.index, y=eqb, name=ref, line=dict(color="#9a9990", width=1.5, dash="dash"))
        palette = [STRATEGY_COLOR] + SERIES_COLORS
        for i, (m, rr) in enumerate(results.items()):
            e = (1 + rr.returns).cumprod()
            fig.add_scatter(x=e.index, y=e, name=MODEL_LABELS[m], line=dict(color=palette[i % len(palette)], width=2))
        fig.update_yaxes(type="log")
        show(style_fig(fig, 440, title="Growth of 1 (net of costs)"))
        rows = {}
        for m, rr in results.items():
            s = M.summary(rr.returns, rr.benchmarks[ref])
            rp = M.random_portfolio_percentile(rr.gross_returns, rr.random_returns)
            rows[MODEL_LABELS[m]] = {
                "CAGR": pct(s["CAGR"]), "Sharpe": f"{s['Sharpe']:.2f}", "Max DD": pct(s["Max drawdown"]),
                "Excess CAGR": pct(s["Excess CAGR"]), "Info ratio": f"{s['Information ratio']:.2f}",
                "Mean IC": f"{rr.ic.mean():.3f}", "IC t-stat": f"{rr.ic.mean() / (rr.ic.std() / np.sqrt(len(rr.ic))):.2f}",
                "vs random": pct(rp["cagr_percentile"], 0), "Turnover": pct(rr.turnover.iloc[1:].mean(), 0),
            }
        table(pd.DataFrame(rows).T)
        st.caption("Beware of picking the best of many runs: every extra configuration you try raises the chance "
                   "that the winner is luck. Prefer settings that work across models and periods.")

# =========================================================================== Method
with tabs[6]:
    st.markdown(Path(__file__).with_name("METHOD.md").read_text(encoding="utf-8"))
    q = getattr(data, "quality", None)
    with st.expander(f"Data quality: {0 if q is None else len(q)} corrections applied to Yahoo prices"):
        if q is None or q.empty:
            st.write("No corrections were needed.")
        else:
            st.caption("Bad prints (spikes that reverse within days) are removed; obviously unadjusted "
                       "stock splits are fixed. Everything else is left as downloaded.")
            qq = q.copy()
            qq["name"] = [data.names.get(t, t) for t in qq["ticker"]]
            qq["move"] = qq["move"].map(lambda v: pct(v, 0))
            table(qq[["ticker", "name", "date", "issue", "move"]])
