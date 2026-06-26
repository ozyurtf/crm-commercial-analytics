import numpy as np
import pandas as pd
from feature_engineering import train_test_split
from training import fit_ets, seasonal_naive

def mape(actual, forecast) -> float:
    """Mean absolute percentage error (%), ignoring zero-actual months."""
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    mask = a != 0
    return float(np.mean(np.abs((a[mask] - f[mask]) / a[mask])) * 100)


def backtest(series: pd.Series, horizon: int = 12) -> dict:
    """
    Holdout backtest on the final `horizon` months.
    Returns MAPE for ETS and for the seasonal-naive baseline.
    """
    train, test = train_test_split(series, horizon)

    ets_fc = fit_ets(train).forecast(horizon)
    naive_fc = seasonal_naive(train, horizon)

    return {
        "ets_mape": mape(test, ets_fc),
        "naive_mape": mape(test, naive_fc),
        "n_test": int(horizon),
    }
