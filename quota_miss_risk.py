from pathlib import Path
import numpy as np
import pandas as pd
from training import fit_ets

DATA = Path(__file__).resolve().parent / "data"
OUT = Path(__file__).resolve().parent / "outputs"
N_SIM = 1000
SEED = 0
HORIZON = 12
FCAST_YEAR = 2026
RISK_THRESHOLD = 0.50          


def load_sales() -> pd.DataFrame:
    df = pd.read_csv(DATA / "crm_sales.csv",
                     usecols=["sales_date", "area", "region_code", "net_sales"])
    df["net_sales"] = pd.to_numeric(df["net_sales"], errors="coerce")
    df["month"] = pd.to_datetime(df["sales_date"]).dt.to_period("M").dt.to_timestamp()
    return df.dropna(subset=["month"])


def monthly_series(df: pd.DataFrame, level_col: str, entity: str) -> pd.Series:
    s = (df[df[level_col] == entity]
         .groupby("month")["net_sales"].sum()
         .sort_index().asfreq("MS"))
    return s.interpolate()


def annual_forecast(series: pd.Series):
    """Return (array of simulated annual-2026 totals, point annual forecast)."""
    fit = fit_ets(series)
    sims = fit.simulate(HORIZON, repetitions=N_SIM, anchor="end", random_state=SEED)
    annual_draws = np.asarray(sims).sum(axis=0)          # sum the 12 months per path
    point = float(fit.forecast(HORIZON).sum())
    return annual_draws, point


def extrapolate_2026_target(tgt: pd.DataFrame, level: str,
                            entity_col: str, entity: str) -> float:
    t = tgt[(tgt["target_level"] == level) & (tgt[entity_col] == entity)]
    yearly = t.groupby("fiscal_year")["annual_net_sales_target"].sum()
    if yearly.size < 2:
        return np.nan
    yrs = yearly.index.values.astype(float)
    vals = yearly.values.astype(float)
    slope, intercept = np.polyfit(yrs, vals, 1)
    return float(slope * FCAST_YEAR + intercept)


def score_entities(df, tgt, level, level_col, entity_col):
    rows = []
    for entity in sorted(df[level_col].dropna().unique()):
        series = monthly_series(df, level_col, entity)
        if series.dropna().size < 24:                    # need >= 2 seasons
            continue
        target = extrapolate_2026_target(tgt, level, entity_col, entity)
        if not np.isfinite(target) or target <= 0:
            continue
        draws, point = annual_forecast(series)
        p_miss = float((draws < target).mean())
        attainment = float(np.median(draws) / target * 100)
        rows.append(dict(
            level=level, entity=entity,
            area=entity if level == "area" else entity[:2],
            region_code=entity if level == "region" else "",
            target_2026=target, forecast_2026=point,
            attainment_pct=attainment,
            expected_shortfall=target - point,   # $ below quota (negative = surplus)
            p_miss=p_miss,
            at_risk=p_miss > RISK_THRESHOLD,
        ))
    return rows


def run():
    OUT.mkdir(exist_ok=True)
    df = load_sales()
    tgt = pd.read_csv(DATA / "targets_data.csv",
                      usecols=["target_level", "fiscal_year", "area",
                               "region_code", "annual_net_sales_target"])
    tgt["annual_net_sales_target"] = pd.to_numeric(
        tgt["annual_net_sales_target"], errors="coerce")
    tgt = tgt.dropna(subset=["target_level", "fiscal_year"])

    print("Scoring areas ...")
    rows = score_entities(df, tgt, "area", "area", "area")
    print("Scoring regions ...")
    rows += score_entities(df, tgt, "region", "region_code", "region_code")

    # p_miss gates "at risk"; expected_shortfall ($) gives the usable ranking
    # because p_miss saturates near 1.0 when the forecast is confidently short.
    risk = pd.DataFrame(rows).sort_values(["level", "expected_shortfall"],
                                          ascending=[True, False])
    risk["risk_rank"] = (risk.groupby("level")["expected_shortfall"]
                         .rank(ascending=False, method="first").astype(int))
    risk.to_csv(OUT / "quota_risk.csv", index=False)

    print(f"\nWrote {OUT/'quota_risk.csv'}  ({len(risk)} rows)\n")
    show = ["level", "entity", "target_2026", "forecast_2026", "attainment_pct",
            "expected_shortfall", "p_miss", "at_risk"]
    print("Top area risks:")
    print(risk[risk.level == "area"][show].head(6).to_string(index=False))
    print("\nTop region risks:")
    print(risk[risk.level == "region"][show].head(10).to_string(index=False))


if __name__ == "__main__":
    run()
