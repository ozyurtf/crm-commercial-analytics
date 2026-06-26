import pandas as pd
from data_processing import METRICS

NATIONAL = "National"

def build_series(panel: pd.DataFrame, metric: str, geo: str) -> pd.Series:
    """
    One monthly series for a given metric and geography.
    geo == 'National' aggregates across all areas; otherwise filters to that area.
    Returned series has a monthly-start ('MS') frequency with no gaps.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {list(METRICS)}")

    d = panel if geo == NATIONAL else panel[panel["area"] == geo]
    s = (
        d.groupby("month")[metric].sum()
        .sort_index()
        .asfreq("MS")           
    )
    return s.interpolate()     


def all_geographies(panel: pd.DataFrame) -> list:
    """National first, then each area."""
    return [NATIONAL] + sorted(panel["area"].dropna().unique().tolist())


def train_test_split(series: pd.Series, horizon: int = 12):
    """Chronological holdout: last `horizon` months are the test set."""
    return series.iloc[:-horizon], series.iloc[-horizon:]
