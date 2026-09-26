"""Forecasted Portfolio Optimisation (fpo).

Monthly machine-learning stock selection and portfolio optimisation for
US + Swedish equities, with leak-free walk-forward backtesting against
index benchmarks.
"""

from fpo.config import BacktestConfig
from fpo.backtest import run_backtest, BacktestResult
from fpo.recommend import recommend_portfolio

__all__ = ["BacktestConfig", "run_backtest", "BacktestResult", "recommend_portfolio"]
__version__ = "0.1.0"
