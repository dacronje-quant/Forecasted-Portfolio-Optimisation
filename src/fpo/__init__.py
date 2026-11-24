"""Forecasted Portfolio Optimisation (fpo).

Fast monthly machine-learning stock selection, walk-forward optimal meta-strategy
blending, multi-asset UCITS ETF / 0%-fee Swedish mutual fund tactical allocation,
and cost-aware Avanza/Nordnet execution.
"""

from fpo.config import BacktestConfig, BrokerFeeSchedule, BROKER_PRESETS
from fpo.backtest import run_backtest, BacktestResult
from fpo.data import load_market_data, load_etf_market_data, synthetic_market_data, synthetic_etf_market_data
from fpo.meta import run_meta_backtest, MetaBacktestResult, DEFAULT_CANDIDATES, FAST_CANDIDATES
from fpo.portfolio import PortfolioState, ExecutionPlan, load_portfolio, save_portfolio
from fpo.recommend import recommend_portfolio, recommend_optimal_portfolio, render_monthly_brief
from fpo.tactical import run_tactical_backtest, TacticalMetaResult, render_tactical_brief

__all__ = [
    "BacktestConfig",
    "BrokerFeeSchedule",
    "BROKER_PRESETS",
    "run_backtest",
    "BacktestResult",
    "load_market_data",
    "load_etf_market_data",
    "synthetic_market_data",
    "synthetic_etf_market_data",
    "run_meta_backtest",
    "MetaBacktestResult",
    "DEFAULT_CANDIDATES",
    "FAST_CANDIDATES",
    "PortfolioState",
    "ExecutionPlan",
    "load_portfolio",
    "save_portfolio",
    "recommend_portfolio",
    "recommend_optimal_portfolio",
    "render_monthly_brief",
    "run_tactical_backtest",
    "TacticalMetaResult",
    "render_tactical_brief",
]
__version__ = "0.3.0"
