from pathlib import Path
import pandas as pd

from data_processing import load_sales, monthly_panel, METRICS
from feature_engineering import build_series, all_geographies
from training import ets_forecast
from evaluation import backtest

HORIZON = 12
OUT_DIR = Path(__file__).resolve().parent / "outputs"


def run():
    OUT_DIR.mkdir(exist_ok=True)

    print("Loading sales and building monthly panel ...")
    panel = monthly_panel(load_sales())
    geos = all_geographies(panel)

    forecast_rows, validation_rows = [], []

    for metric in METRICS:                       
        for geo in geos:                         
            series = build_series(panel, metric, geo)

            # 1) validation
            scores = backtest(series, HORIZON)
            scores.update(geo=geo, metric=metric,
                          beats_baseline=scores["ets_mape"] < scores["naive_mape"])
            validation_rows.append(scores)

            fc = ets_forecast(series, HORIZON)

            for month, value in series.items():
                forecast_rows.append(dict(geo=geo, metric=metric, month=month,
                                          kind="actual", value=value,
                                          lower=None, upper=None))
            for month, row in fc.iterrows():
                forecast_rows.append(dict(geo=geo, metric=metric, month=month,
                                          kind="forecast", value=row["forecast"],
                                          lower=row["lower"], upper=row["upper"]))

            print(f"  {metric:9} | {geo:9} | ETS MAPE {scores['ets_mape']:5.1f}%"
                  f" vs naive {scores['naive_mape']:5.1f}%"
                  f"  {'OK' if scores['beats_baseline'] else 'check'}")

    fc_df = pd.DataFrame(forecast_rows)
    val_df = pd.DataFrame(validation_rows)[
        ["geo", "metric", "ets_mape", "naive_mape", "beats_baseline", "n_test"]
    ]

    fc_df.to_csv(OUT_DIR / "forecast.csv", index=False)
    val_df.to_csv(OUT_DIR / "validation.csv", index=False)

    print(f"\nWrote {OUT_DIR/'forecast.csv'}  ({len(fc_df):,} rows)")
    print(f"Wrote {OUT_DIR/'validation.csv'} ({len(val_df)} rows)")


if __name__ == "__main__":
    run()
