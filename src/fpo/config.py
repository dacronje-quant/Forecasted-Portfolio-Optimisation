"""Configuration for backtests and live recommendations."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

MODELS = ("ensemble", "gbm", "random_forest", "ridge", "mlp", "momentum")
OPTIMIZERS = ("mean_variance", "min_variance", "hrp", "inverse_vol", "score", "equal")


@dataclass
class BacktestConfig:
    # Universe / data
    markets: str = "both"            # 'us', 'se' or 'both'
    base_currency: str = "SEK"       # returns are measured in this currency
    start: str = "2005-01-01"        # first date of price history to download
    backtest_start: str = "2012-01-31"  # first rebalance date of the backtest

    # Forecasting model
    model: str = "ensemble"          # see MODELS
    target: str = "rank"             # 'rank' (cross-sectional rank) or 'excess' (vs median)
    train_window_months: int = 0     # 0 = expanding window, else rolling window length
    min_train_months: int = 36       # need at least this many labelled months to train
    retrain_every: int = 3           # refit the model every N rebalances
    embargo_months: int = 0          # extra gap between train labels and prediction date
    random_state: int = 42

    # Portfolio construction
    optimizer: str = "mean_variance"  # see OPTIMIZERS
    n_holdings: int = 15
    max_weight: float = 0.12
    min_weight: float = 0.02          # positions smaller than this are dropped
    hold_buffer: float = 1.5          # keep current holdings while ranked within n_holdings*buffer
    risk_aversion: float = 5.0        # mean-variance risk aversion
    assumed_ic: float = 0.05          # Grinold: alpha = IC * vol * z-score
    cov_lookback_days: int = 252
    max_country_weight: float = 1.0   # e.g. 0.7 caps US or SE at 70%

    # Trading frictions
    cost_bps: float = 15.0            # one-way cost per unit of turnover (commission + spread)
    fx_cost_bps: float = 10.0         # extra cost on trades in non-base-currency stocks

    # Statistical tests
    n_random_portfolios: int = 500

    extra: dict = field(default_factory=dict)

    def validate(self) -> "BacktestConfig":
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}")
        if self.target not in ("rank", "excess"):
            raise ValueError("target must be 'rank' or 'excess'")
        if self.base_currency not in ("SEK", "USD"):
            raise ValueError("base_currency must be 'SEK' or 'USD'")
        if self.n_holdings < 1:
            raise ValueError("n_holdings must be >= 1")
        if not 0 < self.max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")
        return self

    def to_dict(self) -> dict:
        return asdict(self)
