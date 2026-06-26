# CRM Commercial Analytics — Take-Home Solution

**Presentation slides:** https://docs.google.com/presentation/d/1hEisli2MrZ6f58F9ZDrub35EM12vgbcb6TySevOgrTs/edit?usp=sharing

---

## How to run

```bash
# 1. create / activate a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt          # or the list under "Requirements"

# 3. place the data files in ./data/
#    data/crm_implants.csv  data/crm_sales.csv  data/targets_data.csv  data/readme.xlsx

# 4. run the models (each writes to ./outputs/)
python inference.py            # Deliverable 3 — 12-month forecast
python quota_miss_risk.py      # Deliverable 4 — quota-miss risk (area & region)
python high_risk_hospitals.py  # Deliverable 5 — top 25 high-risk hospitals

# 5. open the notebook for all visuals (Deliverables 1, 2, 3, 4, 5)
jupyter notebook exploratory_data_analysis.ipynb
```

Run order: the three scripts produce the CSVs in `outputs/`; the notebook reads
those CSVs to render the forecast / risk / hospital charts, so run the scripts
first.

## Requirements

Python 3.10+ and:

```
pandas
numpy
statsmodels      # ETS / Holt-Winters forecasting
plotly           # interactive charts in the notebook
matplotlib       # EDA grids
missingno        # missing-value visualisation
openpyxl         # reads readme.xlsx data dictionary
jupyter
```

## Repository structure

| File | Deliverable | Purpose |
|------|-------------|---------|
| `exploratory_data_analysis.ipynb` | 1, 2 (+ viz for 3–5) | EDA + all executive/forecast/risk visuals (Plotly) |
| `data_processing.py` | 3 | Load sales, build monthly panel |
| `feature_engineering.py` | 3 | Build per-geography monthly series + train/test split |
| `training.py` | 3 | ETS (Holt-Winters) model + seasonal-naive baseline |
| `evaluation.py` | 3 | Holdout backtest, MAPE vs baseline |
| `inference.py` | 3 | Entry point -> `outputs/forecast.csv`, `outputs/validation.csv` |
| `quota_miss_risk.py` | 4 | Quota-miss probability + shortfall -> `outputs/quota_risk.csv` |
| `high_risk_hospitals.py` | 5 | Top-25 "positive now, declining later" -> `outputs/high_risk_hospitals.csv` |

---

## Key assumptions & data decisions

**Data structure**
- `crm_implants.csv` has **no date column — only `fiscal_year`**. Implant volume
  and leadless trends are therefore **yearly**. `crm_sales.csv` is the only
  monthly-dated source.
- **Net sales** (`net_sales = gross − commission − rebate`) is the headline KPI.
- The targets file's `actual_*` / `attainment_pct` columns are **empty**, so all
  attainment is **computed by us** (actual net sales ÷ target via geography join).
- 2025 is treated as a **complete year** (sales run to 2025-12-31).

**Deliverable 3 — forecast**
- Monthly forecast of **net sales** and **volume** (= `actual_units_sold`, since
  implants lack dates), at **national + area** level (region optional).
- Model: **Holt-Winters ETS** (additive trend + 12-month seasonality), validated
  against a **seasonal-naive baseline** via a 12-month holdout (MAPE).
- *Finding:* the series are highly regular, so the seasonal-naive baseline is
  strong; **volume has a 2025 structural break** (units −27% while revenue rose ->
  premium mix shift) — this is the main assumption that would invalidate volume
  forecasts.

**Deliverable 4 — quota-miss risk**
- "At risk" = **P(simulated 2026 annual net sales < 2026 quota)** from ETS
  simulation; **ranked by expected dollar shortfall** because `p_miss` saturates
  near 1.0.
- The 2026 quota is **linearly extrapolated** from each entity's 2020–2025 target
  trend (targets only run through 2025).
- *Finding:* targets outpace plateauing sales -> ~85% attainment across the board;
  worst areas NE/SA/GL/SG, worst regions GP01/WC01/GL03.

**Deliverable 5 — top 25 high-risk hospitals**
- "Positive now" = 2024->2025 **volume OR net revenue up**; "declining later" =
  **leadless penetration below its area benchmark AND falling / forecast down**.
- `risk_score = benchmark_gap_pp + max(leadless_drop_pp, 0)` (percentage points).
- Eligibility filter: **≥ 20 implants in 2025** to avoid tiny-account noise.
- Per-hospital 2026 forecast is a **simple linear trend** (few yearly points).

## Outputs (`./outputs/`)

| File | Contents |
|------|----------|
| `forecast.csv` | Actuals + 12-month forecast (+ interval) per geography × metric |
| `validation.csv` | ETS vs seasonal-naive MAPE per series |
| `quota_risk.csv` | Per area/region: 2026 target, forecast, attainment, p_miss, shortfall, rank |
| `high_risk_hospitals.csv` | Ranked top-25 hospitals with current trend, risk score, primary driver |
