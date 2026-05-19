"""
Run once: python generate_data.py
Converts Delhi RAWAVG + master transport sheet → CSVs for the AQMSAT dashboard.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DASHBOARD_DELHI_DATA = ROOT / "AIR_QUALITY_DASHBOARD" / "data" / "delhi_data"
DASHBOARD_DELHI_DATA.mkdir(parents=True, exist_ok=True)

SOURCE_CANDIDATES = [
    # Newest All_States_Output_RAW (Apr 2026 daily) — prefer over older FINAL copy
    ROOT / "DASHBOARD_RAW_DATA/All_States_Output_RAW/Delhi/Delhi/Delhi_RAWAVG.xlsx",
    ROOT / "1.FINAL_DASHBOARD_RAW_DATA/All_States_Output_RAW/Delhi/Delhi/Delhi_RAWAVG.xlsx",
    ROOT / "DASHBOARD_RAW_DATA/RAW_DATA_EXTRACTED/Delhi/Delhi_RAWAVG.xlsx",
]
MASTER_CSV = ROOT / "AIR_QUALITY_DASHBOARD" / "data" / "master_aqi_daily.csv"
VEHICLES_XLSX = DATA_DIR / "Delhi master data.xlsx"
POL_XLSX_GLOB = "1747633531_Statewise_Sales-POL_Consumption_Final.xlsx"
IMD_PDF = DATA_DIR / "LATEST DATA DELHI.pdf"

AQI_BANDS = [
    (0, 50, "Good"),
    (51, 100, "Satisfactory"),
    (101, 200, "Moderate"),
    (201, 300, "Poor"),
    (301, 400, "Very Poor"),
    (401, 500, "Severe"),
]


def _rawavg_max_daily_date(path: Path) -> pd.Timestamp | None:
    """Latest date in PM2.5_Daily — used to pick the newest Delhi_RAWAVG copy."""
    if not path.exists():
        return None
    try:
        xl = pd.ExcelFile(path)
        if "PM2.5_Daily" not in xl.sheet_names:
            return None
        dates = pd.to_datetime(xl.parse("PM2.5_Daily", usecols=["Date"])["Date"], errors="coerce")
        mx = dates.max()
        return mx if pd.notna(mx) else None
    except Exception:
        return None


def _resolve_source() -> Path:
    """Use the candidate whose PM2.5_Daily sheet has the latest dates (not first on disk)."""
    found: list[Path] = [p for p in SOURCE_CANDIDATES if p.exists()]
    if not found:
        raise FileNotFoundError(f"Delhi RAWAVG not found. Tried: {SOURCE_CANDIDATES}")

    best_path = found[0]
    best_date = pd.Timestamp.min
    for path in found:
        mx = _rawavg_max_daily_date(path)
        if mx is not None and mx > best_date:
            best_date = mx
            best_path = path
    return best_path


def _aqi_cat(val: float) -> str:
    if pd.isna(val):
        return "N/A"
    v = float(val)
    for lo, hi, name in AQI_BANDS:
        if lo <= v <= hi:
            return name
    return "Severe"


def _station_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {"State", "City", "Date", "DateTime", "City Average"}]


_DOMINANT_POLLS = ("PM2.5", "PM10", "NO2", "O3", "CO", "SO2")


def _dominant_pollutant_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Per station-day: pollutant with highest daily concentration (for Chart 3)."""
    if daily.empty:
        return pd.DataFrame(columns=["station", "date", "prominent_pollutant"])
    sub = daily[daily["pollutant"].isin(_DOMINANT_POLLS)].copy()
    sub["station_val"] = pd.to_numeric(sub["station_val"], errors="coerce")
    sub = sub.dropna(subset=["station", "date", "station_val"])
    if sub.empty:
        return pd.DataFrame(columns=["station", "date", "prominent_pollutant"])
    idx = sub.groupby(["station", "date"], sort=False)["station_val"].idxmax()
    return (
        sub.loc[idx, ["station", "date", "pollutant"]]
        .rename(columns={"pollutant": "prominent_pollutant"})
        .reset_index(drop=True)
    )


def _enrich_aqi_prominent(aqi: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Station-wise dominant pollutant from daily RAW (not city-only master bulletin)."""
    if aqi.empty:
        return aqi
    dom = _dominant_pollutant_from_daily(daily)
    aqi = aqi.merge(dom, on=["station", "date"], how="left")
    if MASTER_CSV.exists():
        master = pd.read_csv(MASTER_CSV, usecols=["date", "city", "prominent_pollutant"], low_memory=False)
        master["date"] = pd.to_datetime(master["date"], errors="coerce")
        master = master[master["city"].astype(str).str.strip().eq("Delhi")][
            ["date", "prominent_pollutant"]
        ].rename(columns={"prominent_pollutant": "prominent_city"})
        aqi = aqi.merge(master, on="date", how="left")
        aqi["prominent_pollutant"] = aqi["prominent_pollutant"].fillna(aqi["prominent_city"])
        aqi = aqi.drop(columns=["prominent_city"], errors="ignore")
    return aqi


def _melt_station_sheet(
    df: pd.DataFrame,
    *,
    date_col: str,
    value_name: str,
) -> pd.DataFrame:
    stations = _station_cols(df)
    if not stations:
        return pd.DataFrame()
    long_df = df.melt(
        id_vars=["City", date_col],
        value_vars=stations,
        var_name="station",
        value_name=value_name,
    )
    long_df = long_df.rename(columns={"City": "city", date_col: date_col.lower()})
    long_df["city"] = long_df["city"].astype(str).str.strip()
    long_df["station"] = long_df["station"].astype(str).str.strip()
    long_df[value_name] = pd.to_numeric(long_df[value_name], errors="coerce")
    return long_df


def process_delhi(source: Path) -> dict[str, pd.DataFrame]:
    xl = pd.ExcelFile(source)
    daily_frames: list[pd.DataFrame] = []
    monthly_frames: list[pd.DataFrame] = []
    yearly_frames: list[pd.DataFrame] = []
    aqi_frames: list[pd.DataFrame] = []

    for sheet in xl.sheet_names:
        df = xl.parse(sheet)

        if sheet.endswith("_Daily") and not sheet.startswith("AQI"):
            poll = sheet[:-6]
            long_df = _melt_station_sheet(df, date_col="Date", value_name="station_val")
            if long_df.empty:
                continue
            long_df["pollutant"] = poll
            city_avg = df[["Date", "City Average"]].rename(
                columns={"Date": "date", "City Average": "city_avg"}
            )
            city_avg["date"] = pd.to_datetime(city_avg["date"], errors="coerce")
            long_df["date"] = pd.to_datetime(long_df["date"], errors="coerce")
            long_df = long_df.merge(city_avg, on="date", how="left")
            long_df["city_avg"] = pd.to_numeric(long_df["city_avg"], errors="coerce")
            daily_frames.append(long_df[["city", "station", "date", "pollutant", "station_val", "city_avg"]])

        elif sheet.endswith("_Monthly"):
            poll = sheet[:-8]
            long_df = _melt_station_sheet(df, date_col="Date", value_name="station_val")
            if long_df.empty:
                continue
            long_df = long_df.rename(columns={"date": "month"})
            long_df["pollutant"] = poll
            city_avg = df[["Date", "City Average"]].rename(
                columns={"Date": "month", "City Average": "city_avg"}
            )
            long_df = long_df.merge(city_avg, on="month", how="left")
            long_df["city_avg"] = pd.to_numeric(long_df["city_avg"], errors="coerce")
            monthly_frames.append(long_df[["city", "station", "month", "pollutant", "station_val", "city_avg"]])

        elif sheet.endswith("_Yearly"):
            poll = sheet[:-7]
            long_df = _melt_station_sheet(df, date_col="Date", value_name="station_val")
            if long_df.empty:
                continue
            long_df = long_df.rename(columns={"date": "year"})
            long_df["pollutant"] = poll
            long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce")
            city_avg = df[["Date", "City Average"]].rename(
                columns={"Date": "year", "City Average": "city_avg"}
            )
            city_avg["year"] = pd.to_numeric(city_avg["year"], errors="coerce")
            long_df = long_df.merge(city_avg, on="year", how="left")
            long_df["city_avg"] = pd.to_numeric(long_df["city_avg"], errors="coerce")
            yearly_frames.append(long_df[["city", "station", "year", "pollutant", "station_val", "city_avg"]])

        elif sheet == "AQI_Daily":
            long_df = _melt_station_sheet(df, date_col="Date", value_name="index_value")
            if long_df.empty:
                continue
            long_df["date"] = pd.to_datetime(long_df["date"], errors="coerce")
            long_df["air_quality_category"] = long_df["index_value"].map(_aqi_cat)
            aqi_frames.append(long_df[["city", "station", "date", "index_value", "air_quality_category"]])

    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    monthly = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    yearly = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    aqi = pd.concat(aqi_frames, ignore_index=True) if aqi_frames else pd.DataFrame()

    if not aqi.empty and not daily.empty:
        aqi = _enrich_aqi_prominent(aqi, daily)

    return {"daily": daily, "monthly": monthly, "yearly": yearly, "aqi": aqi}


def _find_pol_xlsx() -> Path | None:
    matches = sorted(DATA_DIR.glob(POL_XLSX_GLOB))
    if matches:
        return matches[0]
    matches = sorted(DATA_DIR.glob("*POL*Consumption*.xlsx"))
    return matches[0] if matches else None


def export_pol_consumption() -> None:
    path = _find_pol_xlsx()
    if not path:
        print("  (skip) POL consumption — xlsx not found in data/")
        return

    def _delhi_series(sheet: str) -> tuple[list[str], list[float]]:
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        fy_cols = [str(c).strip() for c in raw.iloc[7, 1:].tolist() if pd.notna(c)]
        row = raw[raw.iloc[:, 0].astype(str).str.upper().eq("DELHI")]
        if row.empty:
            return [], []
        vals = pd.to_numeric(row.iloc[0, 1 : 1 + len(fy_cols)], errors="coerce").tolist()
        return fy_cols, vals

    fy_cols, petrol = _delhi_series("PT_Cons_Statewise MS")
    _, diesel = _delhi_series("PT_Cons_Statewise HSD")
    if not fy_cols:
        print(f"  (skip) POL consumption — Delhi row not found in {path.name}")
        return

    rows: list[dict] = []
    for fy, p, d in zip(fy_cols, petrol, diesel):
        if pd.notna(p):
            rows.append({"fy": fy, "product": "Petrol", "consumption_kt": float(p)})
        if pd.notna(d):
            rows.append({"fy": fy, "product": "Diesel", "consumption_kt": float(d)})

    out = pd.DataFrame(rows)
    out.to_csv(DATA_DIR / "delhi_pol_consumption.csv", index=False)
    print(f"  delhi_pol_consumption.csv  {len(out)} rows  (FY {fy_cols[0]}–{fy_cols[-1]})")


def export_imd_met() -> None:
    if not IMD_PDF.exists():
        print(f"  (skip) IMD met — not found: {IMD_PDF.name}")
        return

    try:
        import pdfplumber
    except ImportError:
        print("  (skip) IMD met — install pdfplumber")
        return

    with pdfplumber.open(IMD_PDF) as doc:
        text = doc.pages[0].extract_text() or ""
        tables = doc.pages[0].extract_tables() or []

    month = "May"
    m_month = re.search(r"Month:\s*(\w+)", text)
    if m_month:
        month = m_month.group(1)

    pdf_date = None
    m_date = re.search(r"(\d{2}/\d{2}/\d{4})", text)
    if m_date:
        pdf_date = pd.to_datetime(m_date.group(1), dayfirst=True, errors="coerce")

    year = int(pdf_date.year) if pd.notna(pdf_date) else 2026

    daily_rows: list[dict] = []
    summary: dict = {"month": month, "year": year, "source": "IMD New Delhi"}

    for table in tables:
        if not table:
            continue
        for i, row in enumerate(table):
            if not row or str(row[0]).strip() != "2026":
                continue
            label = str(row[1]).strip().lower() if len(row) > 1 and row[1] else ""
            if label != "max":
                continue
            min_row = table[i + 1] if i + 1 < len(table) else None
            rf_row = table[i + 2] if i + 2 < len(table) else None
            for day in range(1, 32):
                col = day + 1
                if col >= len(row):
                    break
                tmax = pd.to_numeric(row[col], errors="coerce")
                tmin = (
                    pd.to_numeric(min_row[col], errors="coerce")
                    if min_row and col < len(min_row)
                    else float("nan")
                )
                rf = (
                    pd.to_numeric(rf_row[col], errors="coerce")
                    if rf_row and col < len(rf_row) and str(rf_row[col]).strip() not in {"-", "TR", ""}
                    else float("nan")
                )
                if pd.isna(tmax) and pd.isna(tmin):
                    continue
                daily_rows.append(
                    {
                        "year": year,
                        "month": month,
                        "day": day,
                        "tmax_c": tmax,
                        "tmin_c": tmin,
                        "rainfall_mm": rf,
                    }
                )
            if len(row) > 33:
                summary["avg_tmax_c"] = pd.to_numeric(row[33], errors="coerce")
            if min_row and len(min_row) > 33:
                summary["avg_tmin_c"] = pd.to_numeric(min_row[33], errors="coerce")
            if rf_row and len(rf_row) > 33:
                summary["total_rainfall_mm"] = pd.to_numeric(rf_row[33], errors="coerce")
            break

    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        print("  (skip) IMD met — could not parse 2026 table from PDF")
        return

    daily = daily.sort_values("day")
    if pd.notna(pdf_date):
        summary["as_of"] = pdf_date.strftime("%Y-%m-%d")
        upto = daily[daily["day"] <= pdf_date.day]
    else:
        summary["as_of"] = f"{year}-{month}-latest"
        upto = daily

    latest = upto[upto["tmax_c"].notna()].tail(1)
    if latest.empty:
        latest = upto.tail(1)

    if not latest.empty:
        summary["latest_day"] = int(latest.iloc[0]["day"])
        if pd.notna(latest.iloc[0]["tmax_c"]):
            summary["latest_tmax_c"] = float(latest.iloc[0]["tmax_c"])
        if pd.notna(latest.iloc[0]["tmin_c"]):
            summary["latest_tmin_c"] = float(latest.iloc[0]["tmin_c"])

    daily.to_csv(DATA_DIR / "imd_met_daily.csv", index=False)
    pd.DataFrame([summary]).to_csv(DATA_DIR / "imd_met_summary.csv", index=False)
    print(
        f"  imd_met_daily.csv    {len(daily)} days  "
        f"(latest {summary.get('latest_tmax_c', '—')}°C max)"
    )


def export_vehicles() -> None:
    if not VEHICLES_XLSX.exists():
        print(f"  (skip) vehicles — not found: {VEHICLES_XLSX.name}")
        return
    vehicles = pd.read_excel(VEHICLES_XLSX)
    vehicles.columns = [str(c).strip() for c in vehicles.columns]
    out = DATA_DIR / "delhi_all_data.csv"
    vehicles.to_csv(out, index=False)
    print(f"  delhi_all_data.csv  {len(vehicles):,} rows")


def main() -> None:
    source = _resolve_source()
    _through = _rawavg_max_daily_date(source)
    print(f"Reading {source} …")
    if _through is not None:
        print(f"  (PM2.5_Daily through {_through.strftime('%Y-%m-%d')})")
    data = process_delhi(source)

    for key in ("daily", "aqi"):
        frame = data[key]
        if frame.empty:
            continue
        frame["city"] = frame["city"].astype(str).str.strip()
        frame["station"] = frame["station"].astype(str).str.strip()
        if key == "daily":
            frame["pollutant"] = frame["pollutant"].astype(str).str.strip()
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame = frame.dropna(subset=["date"]).sort_values(["date", "station", "pollutant"])
        else:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame = frame.dropna(subset=["date"]).sort_values(["date", "station"])
        data[key] = frame

    _aq_exports = [
        ("daily", "daily_all.csv", data["daily"]),
        ("aqi", "aqi_daily_all.csv", data["aqi"]),
    ]
    for _label, _fname, _frame in _aq_exports:
        if _frame.empty:
            continue
        for _out_dir in (DATA_DIR, DASHBOARD_DELHI_DATA):
            _frame.to_csv(_out_dir / _fname, index=False)
        print(f"  {_fname:<20} {len(_frame):,} rows  → data/ + AQ dashboard delhi_data/")

    if not data["monthly"].empty:
        for _out_dir in (DATA_DIR, DASHBOARD_DELHI_DATA):
            data["monthly"].to_csv(_out_dir / "monthly_all.csv", index=False)
        print(f"  monthly_all.csv      {len(data['monthly']):,} rows")
    if not data["yearly"].empty:
        for _out_dir in (DATA_DIR, DASHBOARD_DELHI_DATA):
            data["yearly"].to_csv(_out_dir / "yearly_all.csv", index=False)
        print(f"  yearly_all.csv       {len(data['yearly']):,} rows")

    if MASTER_CSV.is_file():
        m = pd.read_csv(
            MASTER_CSV,
            usecols=["date", "city", "index_value", "air_quality_category"],
            low_memory=False,
        )
        city = m[m["city"].astype(str).str.strip().eq("Delhi")].sort_values("date")
        city.to_csv(DATA_DIR / "delhi_city_aqi.csv", index=False)
        print(f"  delhi_city_aqi.csv     {len(city):,} rows  (CPCB city bulletin for MVP gauge)")

    print("\nTransport …")
    export_vehicles()
    print("\nPetrol / Diesel (PPAC) …")
    export_pol_consumption()
    print("\nIMD Meteorology …")
    export_imd_met()
    fund_csv = DATA_DIR / "delhi_ncap_fund.csv"
    if fund_csv.exists():
        print(f"\nNCAP fund …  {fund_csv.name} (edit to update amounts)")
    waste_csv = DATA_DIR / "delhi_waste.csv"
    if waste_csv.exists():
        print(f"  waste …  {waste_csv.name}")
    print(f"\nDone → {DATA_DIR}")


if __name__ == "__main__":
    main()
