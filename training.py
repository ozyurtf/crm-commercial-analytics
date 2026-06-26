import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

SEASONAL_PERIODS = 12

def fit_ets(series: pd.Series):
    """Fit additive-trend, additive-seasonal Holt-Winters on a monthly series."""
    model = ExponentialSmoothing(
        series,
        trend="add",
        seasonal="add",
        seasonal_periods=SEASONAL_PERIODS,
        initialization_method="estimated",
    )
    return model.fit()

def ets_forecast(series: pd.Series, horizon: int = 12, ci: float = 0.90,
                 reps: int = 1000, seed: int = 0):
    """
    Fit ETS on `series` and forecast `horizon` months ahead.
    Confidence interval is built by simulation (ETS has no closed-form PI here).
    Returns a DataFrame indexed by future month with: forecast, lower, upper.
    """
    fit = fit_ets(series)
    mean = fit.forecast(horizon)

    sims = fit.simulate(horizon, repetitions=reps, anchor="end",
                        random_state=seed)
    lo = (1 - ci) / 2
    lower = sims.quantile(lo, axis=1)
    upper = sims.quantile(1 - lo, axis=1)

    out = pd.DataFrame({"forecast": mean, "lower": lower.values,
                        "upper": upper.values}, index=mean.index)
    return out.clip(lower=0)     # negative volume/sales are not meaningful

def seasonal_naive(series: pd.Series, horizon: int = 12) -> pd.Series:
    """Baseline forecast: repeat the value from 12 months earlier."""
    last_season = series.iloc[-SEASONAL_PERIODS:].values
    idx = pd.date_range(series.index[-1], periods=horizon + 1, freq="MS")[1:]
    return pd.Series(np.resize(last_season, horizon), index=idx)
