"""Export Delhi-only CPCB rows from master_aqi_daily.csv → data/delhi_city_aqi.csv"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data"
MASTER = ROOT / "AIR_QUALITY_DASHBOARD" / "data" / "master_aqi_daily.csv"
OUT = DATA / "delhi_city_aqi.csv"

if not MASTER.is_file():
    raise SystemExit(f"Missing master file: {MASTER}")

parts = []
for chunk in pd.read_csv(
    MASTER,
    usecols=["date", "city", "index_value", "air_quality_category"],
    chunksize=250_000,
    low_memory=False,
):
    sub = chunk[chunk["city"].astype(str).str.strip().eq("Delhi")]
    if not sub.empty:
        parts.append(sub)

city = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
city = city[["date", "index_value", "air_quality_category"]].sort_values("date")
city.to_csv(OUT, index=False)
print(f"Wrote {len(city):,} rows → {OUT}")
print(city.tail(3))
