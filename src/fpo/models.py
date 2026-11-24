"""Cross-sectional return forecasting models and leak-free walk-forward training.

The models predict *which stocks will do relatively better next month* (a
ranking problem), not the absolute market direction. Scores are later turned
into portfolio weights by the optimiser.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from fpo.config import BacktestConfig
from fpo.features import FEATURES, cross_sectional_ranks, make_target

log = logging.getLogger(__name__)


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-12 else np.zeros_like(x)


def _time_decay_weights(dates: pd.DatetimeIndex, halflife_months: int) -> np.ndarray | None:
    """Exponential sample weights favouring more recent months (None = uniform)."""
    if halflife_months <= 0 or len(dates) == 0:
        return None
    max_d = dates.max()
    age_months = np.maximum(0.0, (max_d - dates).days / 30.4375)
    w = np.power(0.5, age_months / float(halflife_months))
    return np.asarray(w / np.mean(w), dtype=float)


def _fit_with_weights(est, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None,
                      seed: int = 42):
    if sample_weight is not None and not isinstance(est, MomentumModel):
        # HistGradientBoostingRegressor's native weighted quantile binner is ~20x slower in sklearn;
        # deterministic weighted resampling preserves exponential time-decay at full Cython speed
        # and also brings time-decay weighting to MLPRegressor.
        if isinstance(est, (HistGradientBoostingRegressor, MLPRegressor)):
            p = np.asarray(sample_weight, dtype=float)
            p = p / p.sum()
            idx = np.random.default_rng(seed).choice(len(y), size=len(y), replace=True, p=p)
            est.fit(X.iloc[idx] if hasattr(X, "iloc") else X[idx], y[idx])
            return est
        try:
            est.fit(X, y, sample_weight=sample_weight)
            return est
        except TypeError:
            pass
    est.fit(X, y)
    return est


class MomentumModel:
    """Non-ML baseline: classic 12-1 momentum. Useful to see what ML adds."""

    def fit(self, X, y, sample_weight=None):
        return self

    def predict(self, X):
        return np.asarray(X["mom_12_1"], dtype=float)


class EnsembleModel:
    """Average of z-scored predictions from diverse learners (linear, trees, neural net)."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.members = {
            "ridge": make_model("ridge", random_state),
            "gbm": make_model("gbm", random_state),
            "random_forest": make_model("random_forest", random_state),
            "mlp": make_model("mlp", random_state),
        }

    def fit(self, X, y, sample_weight=None):
        for m in self.members.values():
            _fit_with_weights(m, X, y, sample_weight, seed=self.random_state)
        return self

    def predict_members(self, X) -> np.ndarray:
        """Return (n_members, n_samples) array of per-member z-scored predictions."""
        return np.vstack([_zscore(np.asarray(m.predict(X), dtype=float)) for m in self.members.values()])

    def predict_with_uncertainty(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean_score, disagreement_std) across ensemble members."""
        mat = self.predict_members(X)
        return mat.mean(axis=0), mat.std(axis=0)

    def predict(self, X):
        return self.predict_members(X).mean(axis=0)


def make_model(name: str, random_state: int = 42):
    if name == "ridge":
        return Ridge(alpha=10.0)
    if name == "gbm":
        return HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.04, max_depth=3, min_samples_leaf=200,
            l2_regularization=1.0, early_stopping=False, random_state=random_state)
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=100, max_depth=5, min_samples_leaf=100, max_features=0.33,
            max_samples=0.5, n_jobs=-1, random_state=random_state)
    if name == "mlp":
        # Avoid random internal validation splits across time (prevents chronological leakage)
        return MLPRegressor(
            hidden_layer_sizes=(16, 8), alpha=5e-2, learning_rate_init=1e-3, max_iter=120,
            early_stopping=False, random_state=random_state)
    if name == "momentum":
        return MomentumModel()
    if name == "ensemble":
        return EnsembleModel(random_state)
    raise ValueError(f"Unknown model {name!r}")


def training_dates(all_dates: pd.DatetimeIndex, t: pd.Timestamp, cfg: BacktestConfig) -> pd.DatetimeIndex:
    """Feature dates whose labels are fully known at rebalance date t.

    A label for feature date s is the return s -> next completed month end,
    known only at the month end after s. So we can train on dates strictly
    before t (minus an optional embargo). This is the key guard against
    look-ahead bias.
    """
    past = all_dates[all_dates < t]
    if cfg.embargo_months > 0:
        past = past[: max(0, len(past) - cfg.embargo_months)]
    if cfg.train_window_months > 0:
        past = past[-cfg.train_window_months:]
    return past


def walk_forward_predict(feats: pd.DataFrame, cfg: BacktestConfig,
                         rebalance_dates: pd.DatetimeIndex) -> tuple[pd.Series, dict]:
    """Produce out-of-sample scores for every rebalance date.

    Returns (scores indexed by (date, ticker), info dict with the last fitted model
    and optional ensemble disagreement series).
    """
    X_all = cross_sectional_ranks(feats)
    y_all = make_target(feats, cfg.target)
    all_dates = feats.index.get_level_values("date").unique().sort_values()

    scores = []
    uncertainties = []
    model = None
    last_fit = None
    fits = 0
    for i, t in enumerate(rebalance_dates):
        if t not in all_dates:
            continue
        need_fit = model is None or last_fit is None or (i - last_fit) >= cfg.retrain_every
        if need_fit:
            tdates = training_dates(all_dates, t, cfg)
            if len(tdates) < cfg.min_train_months and cfg.model != "momentum":
                continue
            mask = X_all.index.get_level_values("date").isin(tdates)
            X_tr, y_tr = X_all[mask], y_all[mask]
            ok = y_tr.notna().values
            X_ok = X_tr[ok][FEATURES]
            y_ok = y_tr[ok].values
            sw = _time_decay_weights(X_ok.index.get_level_values("date"), cfg.sample_halflife_months)
            model = make_model(cfg.model, cfg.random_state)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _fit_with_weights(model, X_ok, y_ok, sw)
            last_fit = i
            fits += 1
        X_t = X_all.xs(t, level="date", drop_level=False)
        if isinstance(model, EnsembleModel):
            pred, unc = model.predict_with_uncertainty(X_t[FEATURES])
            uncertainties.append(pd.Series(unc, index=X_t.index, name="uncertainty"))
        else:
            pred = np.asarray(model.predict(X_t[FEATURES]), dtype=float)
        scores.append(pd.Series(pred, index=X_t.index, name="score"))

    if not scores:
        raise ValueError("No predictions produced: not enough training history before backtest_start")
    log.info("Walk-forward: %d model fits, %d prediction dates", fits, len(scores))
    info = {"model": model, "n_fits": fits}
    if uncertainties:
        info["uncertainty"] = pd.concat(uncertainties).sort_index()
    return pd.concat(scores).sort_index(), info


def fit_full_model(feats: pd.DataFrame, cfg: BacktestConfig, as_of: pd.Timestamp):
    """Fit on every completed, labelled date strictly before `as_of` (used for live recommendation)."""
    X_all = cross_sectional_ranks(feats)
    y_all = make_target(feats, cfg.target)
    all_dates = feats.index.get_level_values("date").unique().sort_values()
    tdates = training_dates(all_dates, as_of, cfg)
    mask = X_all.index.get_level_values("date").isin(tdates)
    X_tr, y_tr = X_all[mask], y_all[mask]
    ok = y_tr.notna().values
    X_ok = X_tr[ok][FEATURES]
    y_ok = y_tr[ok]
    sw = _time_decay_weights(X_ok.index.get_level_values("date"), cfg.sample_halflife_months)
    model = make_model(cfg.model, cfg.random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _fit_with_weights(model, X_ok, y_ok.values, sw)
    return model, X_ok, y_ok


def permutation_importance_table(model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 3,
                                 max_rows: int = 6000, seed: int = 0) -> pd.Series:
    """Model-agnostic importance: drop in rank-IC when a feature is shuffled."""
    rng = np.random.default_rng(seed)
    if len(X) > max_rows:
        idx = rng.choice(len(X), max_rows, replace=False)
        X, y = X.iloc[idx], y.iloc[idx]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = pd.Series(model.predict(X)).corr(pd.Series(y.values), method="spearman")
        imp = {}
        for col in X.columns:
            drops = []
            for _ in range(n_repeats):
                Xp = X.copy()
                Xp[col] = rng.permutation(Xp[col].values)
                s = pd.Series(model.predict(Xp)).corr(pd.Series(y.values), method="spearman")
                drops.append(base - s)
            imp[col] = float(np.mean(drops))
    return pd.Series(imp).sort_values(ascending=False)


def oos_permutation_importance(feats: pd.DataFrame, cfg: BacktestConfig, val_months: int = 24,
                               n_repeats: int = 3, seed: int = 0) -> pd.Series:
    """Out-of-sample permutation importance evaluated on the most recent `val_months` labelled dates.

    Trains on strictly earlier dates and measures the Spearman rank-IC drop on the held-out
    validation window, avoiding in-sample overfitting distortion.
    """
    X_all = cross_sectional_ranks(feats)
    y_all = make_target(feats, cfg.target)
    labelled_dates = y_all.dropna().index.get_level_values("date").unique().sort_values()
    if len(labelled_dates) < cfg.min_train_months + 6:
        # Fall back to full-sample if history is too short for a clean holdout
        as_of = labelled_dates[-1] + pd.Timedelta(days=1)
        model, X_tr, y_tr = fit_full_model(feats, cfg, as_of)
        return permutation_importance_table(model, X_tr, y_tr, n_repeats=n_repeats, seed=seed)

    split_idx = max(cfg.min_train_months, len(labelled_dates) - val_months)
    train_d = labelled_dates[:split_idx]
    val_d = labelled_dates[split_idx:]
    X_tr = X_all[X_all.index.get_level_values("date").isin(train_d)][FEATURES]
    y_tr = y_all[y_all.index.get_level_values("date").isin(train_d)].dropna()
    X_tr = X_tr.reindex(y_tr.index)

    X_val = X_all[X_all.index.get_level_values("date").isin(val_d)][FEATURES]
    y_val = y_all[y_all.index.get_level_values("date").isin(val_d)].dropna()
    X_val = X_val.reindex(y_val.index)

    sw = _time_decay_weights(X_tr.index.get_level_values("date"), cfg.sample_halflife_months)
    model = make_model(cfg.model, cfg.random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _fit_with_weights(model, X_tr, y_tr.values, sw)
    return permutation_importance_table(model, X_val, y_val, n_repeats=n_repeats, seed=seed)


def rank_ic(scores: pd.Series, fwd_ret: pd.Series) -> pd.Series:
    """Per-date Spearman correlation between score and realised next-month return."""
    df = pd.concat([scores.rename("s"), fwd_ret.rename("r")], axis=1, join="inner").dropna()

    def _corr(g: pd.DataFrame) -> float:
        if len(g) <= 4 or g["s"].nunique() <= 1 or g["r"].nunique() <= 1:
            return np.nan
        return float(g["s"].corr(g["r"], method="spearman"))

    return df.groupby(level="date").apply(_corr).dropna()


def feature_ic_table(feats: pd.DataFrame) -> pd.DataFrame:
    """Univariate predictive power of each raw feature (mean rank-IC and t-stat)."""
    rows = {}
    for f in FEATURES:
        if f not in feats.columns:
            continue
        ic = rank_ic(feats[f], feats["fwd_ret"])
        if len(ic) < 3:
            continue
        rows[f] = {"mean_ic": ic.mean(), "t_stat": ic.mean() / (ic.std() / np.sqrt(len(ic)) + 1e-12),
                   "hit_rate": (ic > 0).mean()}
    return pd.DataFrame(rows).T.sort_values("mean_ic", key=np.abs, ascending=False)
