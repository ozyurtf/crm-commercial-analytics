from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
SALES_CSV = DATA_DIR / "crm_sales.csv"
METRICS = {"net_sales": "net_sales", "volume": "actual_units_sold"}


def load_sales(path: Path = SALES_CSV) -> pd.DataFrame:
    """Load the sales file with only the columns we need and a monthly stamp."""
    df = pd.read_csv(
        path,
        usecols=["sales_date", "area", "net_sales", "actual_units_sold"],
    )
    df["net_sales"] = pd.to_numeric(df["net_sales"], errors="coerce")
    df["actual_units_sold"] = pd.to_numeric(df["actual_units_sold"], errors="coerce")
    df["month"] = pd.to_datetime(df["sales_date"]).dt.to_period("M").dt.to_timestamp()
    return df.dropna(subset=["month"])


def monthly_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Tidy monthly panel: one row per (month, area) with both metrics summed."""
    panel = (
        df.groupby(["month", "area"])
        .agg(net_sales=("net_sales", "sum"),
             volume=("actual_units_sold", "sum"))
        .reset_index()
        .sort_values(["area", "month"])
    )
    return panel


def list_areas(df: pd.DataFrame) -> list:
    return sorted(df["area"].dropna().unique().tolist())


if __name__ == "__main__":
    d = load_sales()
    p = monthly_panel(d)
    print("rows:", len(p), "| months:", p["month"].nunique(),
          "| areas:", list_areas(d))
    print(p.head())
