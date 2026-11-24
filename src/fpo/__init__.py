"""Forecasted Portfolio Optimisation (fpo).

Monthly machine-learning stock selection, walk-forward optimal meta-strategy
blending, and cost-aware Avanza/Nordnet execution for US + Swedish equities.
"""

from fpo.config import BacktestConfig, BrokerFeeSchedule, BROKER_PRESETS
from fpo.backtest import run_backtest, BacktestResult
from fpo.meta import run_meta_backtest, MetaBacktestResult, DEFAULT_CANDIDATES, FAST_CANDIDATES
from fpo.portfolio import PortfolioState, ExecutionPlan, load_portfolio, save_portfolio
from fpo.recommend import recommend_portfolio, recommend_optimal_portfolio, render_monthly_brief

__all__ = [
    "BacktestConfig",
    "BrokerFeeSchedule",
    "BROKER_PRESETS",
    "run_backtest",
    "BacktestResult",
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
]
__version__ = "0.2.0"
