from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
OUT = Path(__file__).resolve().parent / "outputs"
LATEST, PREV = 2025, 2024
MIN_VOL = 20          # min devices in LATEST year to be eligible
TOP_N = 25


def load():
    imp = pd.read_csv(DATA / "crm_implants.csv",
                      usecols=["fiscal_year", "hospital_id", "hospital_name",
                               "area", "product_type", "serial_number"])
    imp["fiscal_year"] = imp["fiscal_year"].astype(int)
    sal = pd.read_csv(DATA / "crm_sales.csv",
                      usecols=["fiscal_year", "hospital_id", "net_sales"])
    sal["fiscal_year"] = sal["fiscal_year"].astype(int)
    sal["net_sales"] = pd.to_numeric(sal["net_sales"], errors="coerce")
    return imp, sal


def hospital_year(imp, sal):
    hy = (imp.groupby(["hospital_id", "fiscal_year"])
          .agg(volume=("serial_number", "size"),
               ll_units=("product_type", lambda s: (s == "LL").sum()))
          .reset_index())
    hy["ll_pen"] = hy["ll_units"] / hy["volume"]
    net = (sal.groupby(["hospital_id", "fiscal_year"])["net_sales"].sum()
           .reset_index())
    return hy.merge(net, on=["hospital_id", "fiscal_year"], how="left")


def area_benchmark(imp):
    b = (imp.groupby(["area", "fiscal_year"])
         .agg(vol=("serial_number", "size"),
              ll=("product_type", lambda s: (s == "LL").sum()))
         .reset_index())
    b["bench_ll_pen"] = b["ll"] / b["vol"]
    return b


def predict_next(years, vals):
    years = np.asarray(years, float)
    vals = np.asarray(vals, float)
    if len(years) < 2 or np.allclose(vals, vals[0]):
        return float(vals[-1])
    slope, intercept = np.polyfit(years, vals, 1)
    return float(slope * (years.max() + 1) + intercept)


def build():
    imp, sal = load()
    hy = hospital_year(imp, sal)
    bench = area_benchmark(imp)
    meta = (imp.sort_values("fiscal_year")
            .groupby("hospital_id")
            .agg(hospital_name=("hospital_name", "last"), area=("area", "last"))
            .reset_index())
    bench_latest = (bench[bench.fiscal_year == LATEST]
                    .set_index("area")["bench_ll_pen"])

    rows = []
    for hid, grp in hy.groupby("hospital_id"):
        g = grp.set_index("fiscal_year").sort_index()
        if LATEST not in g.index or PREV not in g.index:
            continue
        vol_now, vol_prev = g.loc[LATEST, "volume"], g.loc[PREV, "volume"]
        if vol_now < MIN_VOL:
            continue

        ll_now, ll_prev = g.loc[LATEST, "ll_pen"], g.loc[PREV, "ll_pen"]
        net_now, net_prev = g.loc[LATEST, "net_sales"], g.loc[PREV, "net_sales"]
        area = meta.loc[meta.hospital_id == hid, "area"].iloc[0]
        bench_now = float(bench_latest.get(area, np.nan))

        vol_growth = (vol_now - vol_prev) / vol_prev if vol_prev else np.nan
        net_growth = ((net_now - net_prev) / net_prev
                      if (net_prev and np.isfinite(net_prev)) else np.nan)
        positive_now = (np.isfinite(vol_growth) and vol_growth > 0) or \
                       (np.isfinite(net_growth) and net_growth > 0)

        pred_ll = predict_next(g.index.values, g["ll_pen"].values)
        ll_drop_pp = (ll_prev - ll_now) * 100
        bench_gap_pp = (bench_now - ll_now) * 100
        pred_decline = pred_ll < ll_now

        decline_later = (bench_gap_pp > 0) and (ll_drop_pp > 0 or pred_decline)
        if not (positive_now and decline_later):
            continue

        risk = bench_gap_pp + max(ll_drop_pp, 0)

        trend = []
        if np.isfinite(vol_growth):
            trend.append(f"Volume {vol_growth:+.0%}")
        if np.isfinite(net_growth):
            trend.append(f"Net sales {net_growth:+.0%}")
        driver = (f"Leadless {ll_now:.0%} vs area benchmark {bench_now:.0%} "
                  f"({-bench_gap_pp:+.0f}pp gap)")
        if ll_drop_pp > 0:
            driver += f", down {ll_drop_pp:.0f}pp YoY"

        rows.append(dict(
            hospital_id=hid,
            hospital_name=meta.loc[meta.hospital_id == hid, "hospital_name"].iloc[0],
            area=area,
            current_trend=", ".join(trend),
            risk_score=round(risk, 1),
            primary_driver=driver,
            leadless_pen=round(ll_now, 3),
            area_benchmark=round(bench_now, 3),
            leadless_yoy_pp=round(-ll_drop_pp, 1),
            pred_leadless_2026=round(pred_ll, 3),
            vol_growth=round(vol_growth, 3) if np.isfinite(vol_growth) else None,
            net_growth=round(net_growth, 3) if np.isfinite(net_growth) else None,
            volume_2025=int(vol_now),
        ))

    risk = (pd.DataFrame(rows)
            .sort_values("risk_score", ascending=False)
            .head(TOP_N)
            .reset_index(drop=True))
    risk.insert(0, "rank", risk.index + 1)
    return risk


def run():
    OUT.mkdir(exist_ok=True)
    risk = build()
    risk.to_csv(OUT / "high_risk_hospitals.csv", index=False)
    print(f"Wrote {OUT/'high_risk_hospitals.csv'}  ({len(risk)} hospitals)\n")
    show = ["rank", "hospital_name", "area", "current_trend",
            "risk_score", "primary_driver"]
    print(risk[show].head(25).to_string(index=False))


if __name__ == "__main__":
    run()
