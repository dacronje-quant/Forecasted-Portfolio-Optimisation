"""Configuration for backtests, broker fee schedules, and live monthly decisions."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

MODELS = ("ensemble", "lgbm_rank", "smooth_mom", "gbm", "random_forest", "ridge", "mlp", "momentum")
OPTIMIZERS = ("mean_variance", "min_variance", "hrp", "herc", "nco", "inverse_vol", "score", "equal")
REBALANCE_MODES = ("smart_dca", "full")


@dataclass(frozen=True)
class BrokerFeeSchedule:
    """Realistic retail brokerage fee schedule (commission + min fee + FX + spread)."""
    name: str
    label: str
    se_bps: float = 25.0       # % commission on Swedish equities (bps)
    se_min_sek: float = 1.0    # minimum courtage on Swedish trades (SEK)
    us_bps: float = 25.0       # % commission on US equities (bps)
    us_min_sek: float = 10.0   # minimum courtage on US trades (~$1 in SEK)
    fx_bps: float = 25.0       # FX conversion spread on foreign-currency trades (bps)
    spread_bps: float = 5.0    # estimated one-way bid/ask half-spread (bps)

    def trade_cost(self, notional_base: float, ticker_currency: str,
                   base_currency: str = "SEK", fx_rate: float = 10.5) -> float:
        """Exact one-way cost in `base_currency` for a single order of size `notional_base`."""
        if notional_base <= 0:
            return 0.0
        bps = self.se_bps if ticker_currency == "SEK" else self.us_bps
        min_fee_sek = self.se_min_sek if ticker_currency == "SEK" else self.us_min_sek
        min_fee_base = min_fee_sek if base_currency == "SEK" else min_fee_sek / max(fx_rate, 1e-6)
        commission = max(min_fee_base, notional_base * bps / 1e4)
        fx_fee = notional_base * self.fx_bps / 1e4 if ticker_currency != base_currency else 0.0
        spread = notional_base * self.spread_bps / 1e4
        return float(commission + fx_fee + spread)

    def marginal_rate(self, ticker_currency: str, base_currency: str = "SEK") -> float:
        """Proportional one-way trade cost rate (used for L1 turnover penalty & backtest bps)."""
        bps = self.se_bps if ticker_currency == "SEK" else self.us_bps
        fx = self.fx_bps if ticker_currency != base_currency else 0.0
        return (bps + fx + self.spread_bps) / 1e4


BROKER_PRESETS: dict[str, BrokerFeeSchedule] = {
    "avanza_mini": BrokerFeeSchedule(
        name="avanza_mini", label="Avanza Mini (0.25%, min 1/10 SEK, 25 bps FX)",
        se_bps=25.0, se_min_sek=1.0, us_bps=25.0, us_min_sek=10.0, fx_bps=25.0, spread_bps=5.0,
    ),
    "avanza_small": BrokerFeeSchedule(
        name="avanza_small", label="Avanza Small (0.15%, min 39 SEK, 25 bps FX)",
        se_bps=15.0, se_min_sek=39.0, us_bps=15.0, us_min_sek=39.0, fx_bps=25.0, spread_bps=5.0,
    ),
    "avanza_medium": BrokerFeeSchedule(
        name="avanza_medium", label="Avanza Medium (0.069%, min 69 SEK, 25 bps FX)",
        se_bps=6.9, se_min_sek=69.0, us_bps=6.9, us_min_sek=69.0, fx_bps=25.0, spread_bps=5.0,
    ),
    "nordnet_mini": BrokerFeeSchedule(
        name="nordnet_mini", label="Nordnet Mini (0.25%, min 1/10 SEK, 25 bps FX)",
        se_bps=25.0, se_min_sek=1.0, us_bps=25.0, us_min_sek=10.0, fx_bps=25.0, spread_bps=5.0,
    ),
    "nordnet_small": BrokerFeeSchedule(
        name="nordnet_small", label="Nordnet Small (0.15%, min 39 SEK, 25 bps FX)",
        se_bps=15.0, se_min_sek=39.0, us_bps=15.0, us_min_sek=39.0, fx_bps=25.0, spread_bps=5.0,
    ),
    "nordnet_normal": BrokerFeeSchedule(
        name="nordnet_normal", label="Nordnet Normal (0.10%, min 99 SEK, 25 bps FX)",
        se_bps=10.0, se_min_sek=99.0, us_bps=10.0, us_min_sek=99.0, fx_bps=25.0, spread_bps=5.0,
    ),
    "custom": BrokerFeeSchedule(
        name="custom", label="Custom (uses cost_bps / fx_cost_bps, no min fee)",
        se_bps=15.0, se_min_sek=0.0, us_bps=15.0, us_min_sek=0.0, fx_bps=10.0, spread_bps=0.0,
    ),
}


@dataclass
class BacktestConfig:
    # Universe / data
    markets: str = "both"               # 'us', 'se' or 'both'
    base_currency: str = "SEK"          # returns are measured in this currency
    start: str = "2005-01-01"           # first date of price history to download
    backtest_start: str = "2012-01-31"  # first rebalance date of the backtest

    # Forecasting model
    model: str = "ensemble"             # see MODELS
    target: str = "rank"                # 'rank' (cross-sectional rank) or 'excess' (vs median)
    train_window_months: int = 0        # 0 = expanding window, else rolling window length
    min_train_months: int = 36          # need at least this many labelled months to train
    retrain_every: int = 3              # refit the model every N rebalances
    embargo_months: int = 0             # extra gap between train labels and prediction date
    sample_halflife_months: int = 72    # exponential decay half-life for training weights (0 = uniform)
    random_state: int = 42

    # Portfolio construction
    optimizer: str = "mean_variance"    # see OPTIMIZERS
    n_holdings: int = 15
    max_weight: float = 0.12
    min_weight: float = 0.02            # positions smaller than this are dropped
    hold_buffer: float = 1.5            # keep current holdings while ranked within n_holdings*buffer
    risk_aversion: float = 5.0          # mean-variance risk aversion
    assumed_ic: float = 0.05            # fallback Grinold IC when use_dynamic_ic=False
    use_dynamic_ic: bool = True         # calibrate Grinold IC from trailing OOS rank-IC
    ic_prior: float = 0.03              # shrinkage prior for dynamic IC
    ic_lookback_months: int = 36        # trailing months used to estimate OOS IC
    cov_lookback_days: int = 252
    cov_freq: str = "weekly"            # 'weekly' (fixes US/SE timezone bias) or 'daily'
    max_country_weight: float = 1.0     # e.g. 0.7 caps US or SE at 70%
    max_sector_weight: float = 0.35     # e.g. 0.35 caps any single sector at 35%
    turnover_penalty: float = 0.0025    # L1 turnover cost weight in mean-variance (0 = off)

    # Risk / fallback overlays
    core_fallback: bool = True          # blend toward core index / defensive weights when OOS IC <= 0
    regime_overlay: bool = False        # optional vol-targeting + 200d trend defensive overlay
    target_vol: float = 0.14            # annualised target vol when regime_overlay=True

    # Trading frictions & execution
    broker_preset: str = "custom"       # 'avanza_mini', 'nordnet_mini', ..., or 'custom'
    cost_bps: float = 15.0              # one-way cost per unit of turnover (used when broker_preset='custom')
    fx_cost_bps: float = 10.0           # extra cost on foreign-currency trades (when broker_preset='custom')
    no_trade_band: float = 0.015        # ignore weight drifts smaller than 1.5% in live execution
    min_order_sek: float = 1500.0       # skip micro-orders smaller than this (avoids min-fee drag)
    slippage_buffer: float = 0.01       # keep 1% cash buffer so live BUY orders never bounce
    rebalance_mode: str = "smart_dca"   # 'smart_dca' or 'full'
    rf_annual: float = 0.0              # annual risk-free rate for Sharpe / Sortino

    # Statistical tests
    n_random_portfolios: int = 500

    extra: dict = field(default_factory=dict)

    def fee_schedule(self) -> BrokerFeeSchedule:
        """Return the active broker fee schedule."""
        if self.broker_preset == "custom" or self.broker_preset not in BROKER_PRESETS:
            return BrokerFeeSchedule(
                name="custom",
                label=f"Custom ({self.cost_bps:.0f} bps + {self.fx_cost_bps:.0f} bps FX)",
                se_bps=self.cost_bps,
                se_min_sek=0.0,
                us_bps=self.cost_bps,
                us_min_sek=0.0,
                fx_bps=self.fx_cost_bps,
                spread_bps=0.0,
            )
        return BROKER_PRESETS[self.broker_preset]

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
        if not 0 < self.max_country_weight <= 1:
            raise ValueError("max_country_weight must be in (0, 1]")
        if not 0 < self.max_sector_weight <= 1:
            raise ValueError("max_sector_weight must be in (0, 1]")
        if self.cov_freq not in ("weekly", "daily"):
            raise ValueError("cov_freq must be 'weekly' or 'daily'")
        if self.rebalance_mode not in REBALANCE_MODES:
            raise ValueError(f"rebalance_mode must be one of {REBALANCE_MODES}")
        if self.broker_preset not in BROKER_PRESETS:
            raise ValueError(f"broker_preset must be one of {tuple(BROKER_PRESETS)}")
        return self

    def to_dict(self) -> dict:
        return asdict(self)
