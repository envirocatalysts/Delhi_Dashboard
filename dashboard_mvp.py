from __future__ import annotations

import base64
from pathlib import Path
import json
import math
import urllib.request
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

from station_coords import (
    MAP_CENTER,
    STATION_COORDS,
    STATION_PLOT_OFFSETS,
    station_lat_lon,
)


DATA = Path(__file__).resolve().parent / "data"
ROOT = Path(__file__).resolve().parent.parent
MASTER_CSV = ROOT / "AIR_QUALITY_DASHBOARD" / "data" / "master_aqi_daily.csv"
DELHI_CITY_AQI_CSV = DATA / "delhi_city_aqi.csv"
DELHI_CITY = "Delhi"
LOGO_PATH = Path(__file__).resolve().parent / "EC_Logo-35.jpg"
AQI_DASH_URL = "https://envirocatalysts-delhi-aq-dashboard.streamlit.app/"
TRANSPORT_DASH_URL = "https://www.envirocatalysts.com/changing-gears-dashboard"
EV_TARGET = 25.0
NAAQS_PM25 = 60.0
# Punjab-style: fit full NCT outline in panel on first load (do not force high zoom floor).
DELHI_MAP_CENTER = MAP_CENTER
DELHI_MAP_ZOOM = 9.95
DELHI_MAP_PAD = 0.055
AQI_MAP_HEIGHT = 260
TRANSPORT_MAP_HEIGHT = 260
AQI_MAP_WIDTH = 380.0
TRANSPORT_MAP_WIDTH = 520.0

NAAQS        = {"PM2.5": 60,  "PM10": 100, "NO2": 80,  "SO2": 80,  "O3": 100}
NAAQS_ANNUAL = {"PM2.5": 40,  "PM10": 60,  "NO2": 40,  "SO2": 50}
WHO          = {"PM2.5": 15,  "PM10": 45,  "NO2": 25,  "SO2": 40}
DELHI_STATES_GEOJSON_URL = "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson"
DELHI_DISTRICTS_GEOJSON_URL = (
    "https://raw.githubusercontent.com/shklnrj/IndiaStateTopojsonFiles/master/Delhi.geojson"
)
# 2011 census — used to spread city-level registrations across NCT districts for the map.
DELHI_DIST_POP: dict[str, float] = {
    "Central": 582_320,
    "North": 887_978,
    "South": 2_735_369,
    "East": 1_707_432,
    "North East": 2_250_810,
    "South West": 2_292_959,
    "New Delhi": 142_004,
    "North West": 3_656_539,
    "West": 2_543_243,
    "South East": 1_751_013,
    "Shahdara": 1_709_349,
}
DELHI_DIST_POP_TOTAL = float(sum(DELHI_DIST_POP.values()))

AQI_CAT_COLORS = {
    "Good": "#00b050", "Satisfactory": "#92d050", "Moderate": "#ffff00",
    "Poor": "#ffa500", "Very Poor": "#ff0000",    "Severe":   "#7b0000",
    "N/A": "#9aa4b2",
}
PM25_COLOR_MAP = {
    "Good": "#00b050",
    "Satisfactory": "#92d050",
    "Moderate": "#ffff00",
    "Poor": "#ffa500",
    "Very Poor": "#ff0000",
    "Severe": "#7b0000",
    "N/A": "#9aa4b2",
}
AQI_BANDS = [
    ("Good",       0,   50,  "#00b050"),
    ("Satisfactory",51, 100, "#92d050"),
    ("Moderate",  101,  200, "#ffff00"),
    ("Poor",      201,  300, "#ffa500"),
    ("Very poor", 301,  400, "#ff0000"),
    ("Severe",    401,  500, "#7b0000"),
]


def _to_dt(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(series, errors="coerce")


def _aqi_category(aqi_val: float) -> tuple[str, str]:
    if np.isnan(aqi_val):
        return "N/A", "#9aa4b2"
    for name, lo, hi, color in AQI_BANDS:
        if lo <= aqi_val <= hi:
            return name, color
    return "Severe", "#7b0000"


def _pollutant_kv_label(pollutant: str) -> str:
    """Pollutant unit + NAAQS standard in brackets (O3 = 8h avg; others = 24h avg)."""
    std = NAAQS.get(pollutant)
    if std is None:
        return f"{pollutant} µg/m³"
    avg_h = "8h" if pollutant == "O3" else "24h"
    return (
        f'{pollutant} µg/m³<br>'
        f'<span class="poll-naaqs">(NAAQS {avg_h} avg: {std:g})</span>'
    )


def _pm25_category(pm25_val: float) -> str:
    if np.isnan(pm25_val):
        return "N/A"
    if pm25_val <= 30:
        return "Good"
    if pm25_val <= 60:
        return "Satisfactory"
    if pm25_val <= 90:
        return "Moderate"
    if pm25_val <= 120:
        return "Poor"
    if pm25_val <= 250:
        return "Very Poor"
    return "Severe"


@st.cache_data(show_spinner=False)
def load_delhi_boundary() -> list[list[tuple[float, float]]]:
    """Return polygon rings [(lon, lat), ...] for NCT Delhi boundary."""
    rings: list[list[tuple[float, float]]] = []
    try:
        with urllib.request.urlopen(DELHI_STATES_GEOJSON_URL, timeout=20) as resp:
            geo = json.load(resp)
        features = geo.get("features", [])
        delhi_feature = None
        for feat in features:
            props = feat.get("properties", {})
            if str(props.get("NAME_1", "")).strip().lower() == "delhi":
                delhi_feature = feat
                break
        if not delhi_feature:
            return rings
        geom = delhi_feature.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])
        if gtype == "Polygon":
            polys = [coords]
        elif gtype == "MultiPolygon":
            polys = coords
        else:
            return rings
        for poly in polys:
            if not poly:
                continue
            outer_ring = poly[0]
            ring = [(float(pt[0]), float(pt[1])) for pt in outer_ring if len(pt) >= 2]
            if ring:
                rings.append(ring)
    except Exception:
        return []
    return rings


def _point_in_polygon(lon: float, lat: float, ring: list[tuple[float, float]]) -> bool:
    """Ray-cast test; ring vertices are (lon, lat)."""
    inside = False
    n = len(ring)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def _clip_latlon_inside_delhi(
    lat: float,
    lon: float,
    rings: list[list[tuple[float, float]]],
) -> tuple[float, float]:
    """Keep map pins inside NCT Delhi outline (approx coords + offsets can sit in NCR)."""
    if not rings:
        return lat, lon
    ring = rings[0]
    if _point_in_polygon(lon, lat, ring):
        return lat, lon
    clat, clon = MAP_CENTER["lat"], MAP_CENTER["lon"]
    for step in range(1, 26):
        t = step / 25.0
        tlat = lat + (clat - lat) * t
        tlon = lon + (clon - lon) * t
        if _point_in_polygon(tlon, tlat, ring):
            return (
                tlat - (tlat - clat) * 0.015,
                tlon - (tlon - clon) * 0.015,
            )
    return clat, clon


def _station_map_coords(
    station: str,
    rings: list[list[tuple[float, float]]],
) -> tuple[float, float, float, float]:
    """True lat/lon plus plot position clipped to Delhi boundary."""
    lat, lon = station_lat_lon(station)
    dlat, dlon = STATION_PLOT_OFFSETS.get(station, (0.0, 0.0))
    plat = lat + dlat
    plon = lon + dlon
    plat, plon = _clip_latlon_inside_delhi(plat, plon, rings)
    return lat, lon, plat, plon


@st.cache_data(show_spinner=False)
def load_delhi_district_geojson() -> dict:
    """NCT Delhi revenue districts for transport choropleth (like Punjab districts map)."""
    try:
        with urllib.request.urlopen(DELHI_DISTRICTS_GEOJSON_URL, timeout=20) as resp:
            return json.load(resp)
    except Exception:
        return {"type": "FeatureCollection", "features": []}


def _geojson_bounds(geojson: dict, pad: float = 0.055) -> dict[str, float] | None:
    """Fit map view to geojson extent (Punjab-style centred state view)."""
    features = geojson.get("features", [])
    if not features:
        return None
    lats: list[float] = []
    lons: list[float] = []

    def _walk(coords) -> None:
        if not coords:
            return
        if isinstance(coords[0], (int, float)):
            lons.append(float(coords[0]))
            lats.append(float(coords[1]))
        else:
            for part in coords:
                _walk(part)

    for feat in features:
        geom = feat.get("geometry") or {}
        _walk(geom.get("coordinates", []))
    if not lats:
        return None
    return {
        "west": min(lons) - pad,
        "east": max(lons) + pad,
        "south": min(lats) - pad,
        "north": max(lats) + pad,
    }


def _bounds_from_rings(
    rings: list[list[tuple[float, float]]],
    pad: float,
) -> dict[str, float] | None:
    """BBox from NCT outline rings (same geometry as the AQI map border)."""
    if not rings:
        return None
    lons = [pt[0] for ring in rings for pt in ring]
    lats = [pt[1] for ring in rings for pt in ring]
    if not lons:
        return None
    return {
        "west": min(lons) - pad,
        "east": max(lons) + pad,
        "south": min(lats) - pad,
        "north": max(lats) + pad,
    }


def _zoom_for_panel(
    bounds: dict[str, float],
    height_px: int,
    width_px: float = 420.0,
) -> float:
    """Mapbox zoom so full NCT fits the panel (same idea as Punjab MAP_ZOOM)."""
    lat_c = (bounds["north"] + bounds["south"]) / 2
    lat_rad = math.radians(lat_c)
    lat_span = max(bounds["north"] - bounds["south"], 1e-6)
    lon_span = max(bounds["east"] - bounds["west"], 1e-6)
    world = 256.0
    z_lon = math.log2(360 * width_px / (world * lon_span * math.cos(lat_rad)))
    z_lat = math.log2(180 * height_px / (world * lat_span))
    return float(min(z_lon, z_lat) - 0.08)


def _delhi_map_bounds(
    geojson: dict,
    boundary_rings: list[list[tuple[float, float]]] | None = None,
) -> dict[str, float] | None:
    bounds = _bounds_from_rings(boundary_rings or [], DELHI_MAP_PAD)
    if bounds is None:
        bounds = _geojson_bounds(geojson, pad=DELHI_MAP_PAD)
    return bounds


def _delhi_map_view(
    geojson: dict,
    map_height: int = AQI_MAP_HEIGHT,
    map_width: float = AQI_MAP_WIDTH,
    boundary_rings: list[list[tuple[float, float]]] | None = None,
) -> tuple[dict[str, float], float, dict[str, float] | None]:
    """Centre on NCT Delhi; prefer state outline bounds over district geojson."""
    bounds = _delhi_map_bounds(geojson, boundary_rings)
    if bounds:
        center = {
            "lat": (bounds["north"] + bounds["south"]) / 2,
            "lon": (bounds["east"] + bounds["west"]) / 2,
        }
        zoom = _zoom_for_panel(bounds, map_height, map_width)
        return center, zoom, bounds
    return DELHI_MAP_CENTER, DELHI_MAP_ZOOM, None


_CHART_TICK = dict(size=10, color="#111827")
_CHART_AXIS_LINE = dict(color="#111827")


def _bar_chart_axes(
    *, y_title: str, x_title: str = "", x_angle: int = 0,
) -> dict[str, dict]:
    """Dark axis labels — readable on white Streamlit background."""
    xaxis = dict(
        tickfont=_CHART_TICK,
        color="#111827",
        linecolor="#111827",
        automargin=True,
    )
    if x_title:
        xaxis["title"] = dict(text=x_title, font=dict(size=10, color="#111827"))
    if x_angle:
        xaxis["tickangle"] = x_angle
    return {
        "xaxis": xaxis,
        "yaxis": dict(
            title=dict(text=y_title, font=dict(size=10, color="#111827")),
            tickfont=_CHART_TICK,
            gridcolor="#d1d5db",
            zerolinecolor="#e5e7eb",
            color="#111827",
            linecolor="#111827",
        ),
    }


TRANSPORT_DONUT_HEIGHT = 218
TRANSPORT_DONUT_LEGEND = dict(
    orientation="h",
    y=-0.06,
    x=0.5,
    xanchor="center",
    yanchor="top",
    font=dict(size=7, color="#111827"),
    itemwidth=30,
    tracegroupgap=4,
)


def _transport_pie_figure(
    labels: list,
    values: list,
    colors: list[str],
    total: float,
) -> go.Figure:
    """Donut chart — thin legend swatches, matched size across transport pies."""
    n_leg = len(labels)
    bottom_margin = 58 if n_leg <= 4 else (72 if n_leg <= 6 else 88)
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.58,
            domain=dict(x=[0.02, 0.98], y=[0.06, 0.96]),
            textinfo="percent",
            textposition="inside",
            insidetextorientation="horizontal",
            textfont=dict(color="#111827", size=11),
            insidetextfont=dict(color="#111827", size=11),
            marker=dict(colors=colors, line=dict(color="#ffffff", width=1.5)),
            showlegend=True,
        )
    )
    fig.add_annotation(
        x=0.5,
        y=0.51,
        xref="paper",
        yref="paper",
        text=f"Total<br>{total:,.0f}",
        showarrow=False,
        font=dict(size=11, color="#1f2937"),
    )
    fig.update_layout(
        height=TRANSPORT_DONUT_HEIGHT,
        margin=dict(l=2, r=2, t=2, b=bottom_margin),
        paper_bgcolor="rgba(0,0,0,0)",
        legend={**TRANSPORT_DONUT_LEGEND, "y": -0.08 if n_leg <= 4 else -0.12},
        uniformtext=dict(minsize=9, mode="hide"),
    )
    return fig


def _ev_target_pie_figure(category: str, actual_pct: float, target: float = EV_TARGET) -> go.Figure:
    """EV donut: class + % on teal slice; small swatch legend below (not a large caption box)."""
    actual = max(float(actual_pct), 0.0)
    slice_lbl = f"{category}<br>{actual:.1f}%"
    fig = go.Figure(
        go.Pie(
            labels=["", ""],
            values=[target, actual],
            hole=0.62,
            domain=dict(x=[0.06, 0.94], y=[0.06, 0.94]),
            marker=dict(colors=["#e8edf4", "#0b7285"], line=dict(color="#ffffff", width=1)),
            text=["", slice_lbl],
            textinfo="text",
            textposition="inside",
            insidetextorientation="horizontal",
            textfont=dict(color="#ffffff", size=9),
            hoverinfo="skip",
            showlegend=False,
            sort=False,
        )
    )
    fig.update_layout(
        height=TRANSPORT_DONUT_HEIGHT,
        margin=dict(l=2, r=2, t=2, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _transport_colorbar_ticks(vmin: float, vmax: float, step: float = 2000.0) -> tuple[float, float, list[float], list[str]]:
    """Punjab-style legend: 0, 2k, 4k, … up to a rounded max."""
    vmax = float(max(vmax, vmin, 1.0))
    vmin = float(min(vmin, vmax))
    cmin = 0.0
    cmax = float(np.ceil(vmax / step) * step)
    if cmax < step:
        cmax = step
    tickvals = np.arange(cmin, cmax + step * 0.5, step).tolist()
    ticktext = [f"{v:,.0f}" for v in tickvals]
    return cmin, cmax, tickvals, ticktext


def _split_delhi_district_vehicles(total: float) -> pd.DataFrame:
    """Spread a city-level vehicle count across districts (population weights)."""
    if total <= 0 or DELHI_DIST_POP_TOTAL <= 0:
        return pd.DataFrame(columns=["district", "district_geo", "total_vehicles"])
    rows = [
        {
            "district": dist,
            "district_geo": dist,
            "total_vehicles": total * (pop / DELHI_DIST_POP_TOTAL),
        }
        for dist, pop in DELHI_DIST_POP.items()
    ]
    return pd.DataFrame(rows).sort_values("total_vehicles", ascending=False)


@st.cache_data(show_spinner="Loading Delhi dashboard datasets...")
def load_data() -> dict[str, pd.DataFrame]:
    daily = pd.read_csv(DATA / "daily_all.csv")
    aqi = pd.read_csv(DATA / "aqi_daily_all.csv")

    daily["city"] = daily["city"].astype(str).str.strip()
    daily["station"] = daily["station"].astype(str).str.strip()
    daily["pollutant"] = daily["pollutant"].astype(str).str.strip()
    daily["date"] = _to_dt(daily["date"])
    daily["station_val"] = pd.to_numeric(daily["station_val"], errors="coerce")
    daily["city_avg"] = pd.to_numeric(daily["city_avg"], errors="coerce")
    daily = daily.dropna(subset=["date"]).copy()
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.to_period("M").astype(str)

    aqi["city"] = aqi["city"].astype(str).str.strip()
    aqi["station"] = aqi["station"].astype(str).str.strip()
    aqi["date"] = _to_dt(aqi["date"])
    aqi["index_value"] = pd.to_numeric(aqi.get("index_value"), errors="coerce")
    aqi = aqi.dropna(subset=["date"]).copy()
    aqi["year"] = aqi["date"].dt.year
    aqi["month"] = aqi["date"].dt.to_period("M").astype(str)

    pol = pd.DataFrame(columns=["fy", "product", "consumption_kt"])
    pol_path = DATA / "delhi_pol_consumption.csv"
    if pol_path.exists():
        pol = pd.read_csv(pol_path)
        pol["fy"] = pol["fy"].astype(str).str.strip()
        pol["product"] = pol["product"].astype(str).str.strip()
        pol["consumption_kt"] = pd.to_numeric(pol["consumption_kt"], errors="coerce")

    imd_daily = pd.DataFrame(columns=["year", "month", "day", "tmax_c", "tmin_c", "rainfall_mm"])
    imd_summary = pd.DataFrame()
    if (DATA / "imd_met_daily.csv").exists():
        imd_daily = pd.read_csv(DATA / "imd_met_daily.csv")
        for c in ("tmax_c", "tmin_c", "rainfall_mm"):
            imd_daily[c] = pd.to_numeric(imd_daily.get(c), errors="coerce")
        imd_daily["day"] = pd.to_numeric(imd_daily.get("day"), errors="coerce")
    if (DATA / "imd_met_summary.csv").exists():
        imd_summary = pd.read_csv(DATA / "imd_met_summary.csv")

    vehicles = pd.DataFrame()
    vehicles_path = DATA / "delhi_all_data.csv"
    if vehicles_path.exists():
        vehicles = pd.read_csv(vehicles_path)
        vehicles.columns = [c.strip() for c in vehicles.columns]
        vehicles["total"] = pd.to_numeric(vehicles["total"], errors="coerce").fillna(0)
        vehicles["year"] = pd.to_numeric(vehicles["year"], errors="coerce")
        vehicles["fuel_category"] = vehicles["fuel_category"].astype(str).str.strip()
        vehicles["vec_class_category"] = vehicles["vec_class_category"].astype(str).str.strip()

    ncap_fund = pd.DataFrame(
        {
            "Funds": ["Fund Allocation", "Fund Released", "Fund Utilised"],
            "amount_cr": [103.30, 81.36, 27.00],
        }
    )
    fund_path = DATA / "delhi_ncap_fund.csv"
    if fund_path.exists():
        ncap_fund = pd.read_csv(fund_path)
        ncap_fund["Funds"] = ncap_fund["Funds"].astype(str).str.strip()
        ncap_fund["amount_cr"] = pd.to_numeric(ncap_fund["amount_cr"], errors="coerce")

    waste = pd.DataFrame(columns=["fy", "category", "value_tpd"])
    waste_path = DATA / "delhi_waste.csv"
    if waste_path.exists():
        waste = pd.read_csv(waste_path)
        waste["fy"] = waste["fy"].astype(str).str.strip()
        waste["category"] = waste["category"].astype(str).str.strip()
        waste["value_tpd"] = pd.to_numeric(waste["value_tpd"], errors="coerce")

    return {
        "daily": daily,
        "aqi": aqi,
        "vehicles": vehicles,
        "pol": pol,
        "imd_daily": imd_daily,
        "imd_summary": imd_summary,
        "ncap_fund": ncap_fund,
        "waste": waste,
    }


def _city_aqi_source_paths() -> list[Path]:
    """Prefer slim Delhi export in repo; fall back to full national master."""
    cwd = Path.cwd()
    return [
        DELHI_CITY_AQI_CSV,
        DATA / "master_aqi_daily.csv",
        MASTER_CSV,
        cwd / "AIR_QUALITY_DASHBOARD" / "data" / "master_aqi_daily.csv",
        cwd.parent / "AIR_QUALITY_DASHBOARD" / "data" / "master_aqi_daily.csv",
    ]


def _read_delhi_rows_from_master(path: Path) -> pd.DataFrame:
    """Read only Delhi rows from large master_aqi_daily.csv (chunked)."""
    usecols = ["date", "city", "index_value", "air_quality_category"]
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=250_000, low_memory=False):
        sub = chunk[chunk["city"].astype(str).str.strip().eq(DELHI_CITY)]
        if not sub.empty:
            parts.append(sub)
    if not parts:
        return pd.DataFrame(columns=usecols)
    return pd.concat(parts, ignore_index=True)


def _ensure_delhi_city_aqi_file() -> None:
    """One-time export: master → data/delhi_city_aqi.csv (for deploy without sibling folder)."""
    if DELHI_CITY_AQI_CSV.is_file():
        return
    for path in _city_aqi_source_paths():
        if path == DELHI_CITY_AQI_CSV or not path.is_file():
            continue
        if path.name != "master_aqi_daily.csv":
            continue
        city = _read_delhi_rows_from_master(path)
        if city.empty:
            continue
        out = city[["date", "index_value", "air_quality_category"]].copy()
        out.to_csv(DELHI_CITY_AQI_CSV, index=False)
        return


def _city_aqi_cache_key() -> str:
    mtimes = []
    for path in _city_aqi_source_paths():
        if path.is_file():
            mtimes.append(f"{path}:{path.stat().st_mtime_ns}")
    return "|".join(mtimes) if mtimes else "none"


@st.cache_data(show_spinner=False)
def _load_delhi_city_aqi_cached(_cache_key: str) -> pd.DataFrame:
    cols = ["date", "index_value", "air_quality_category"]
    _ensure_delhi_city_aqi_file()
    for path in _city_aqi_source_paths():
        if not path.is_file():
            continue
        if path.name == "master_aqi_daily.csv":
            m = _read_delhi_rows_from_master(path)
        else:
            usecols = cols if path == DELHI_CITY_AQI_CSV else ["date", "city", "index_value", "air_quality_category"]
            m = pd.read_csv(path, usecols=usecols, low_memory=False)
            if "city" in m.columns:
                m = m[m["city"].astype(str).str.strip().eq(DELHI_CITY)]
        m["date"] = _to_dt(m["date"]).dt.normalize()
        m["index_value"] = pd.to_numeric(m["index_value"], errors="coerce")
        m = m.dropna(subset=["date", "index_value"]).sort_values("date")
        if not m.empty:
            return m[cols].reset_index(drop=True)
    return pd.DataFrame(columns=cols)


def _load_delhi_city_aqi(aqi_stations: pd.DataFrame | None = None) -> pd.DataFrame:
    """CPCB city-wise daily AQI for Delhi (not station-level)."""
    cols = ["date", "index_value", "air_quality_category"]
    city = _load_delhi_city_aqi_cached(_city_aqi_cache_key())
    if not city.empty:
        return city
    if aqi_stations is not None and not aqi_stations.empty:
        stn = aqi_stations.copy()
        stn["date"] = _to_dt(stn["date"]).dt.normalize()
        stn["index_value"] = pd.to_numeric(stn["index_value"], errors="coerce")
        stn = stn.dropna(subset=["date", "index_value"])
        if not stn.empty:
            return (
                stn.groupby("date", as_index=False)
                .agg(
                    index_value=("index_value", "median"),
                    air_quality_category=("air_quality_category", "first"),
                )
                .sort_values("date")
            )
    return pd.DataFrame(columns=cols)


def main() -> None:
    st.set_page_config(
        page_title="Air Quality Meteorology And Source Activity Tracker — Delhi",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
        <style>
        /* ── White background + full-width on all screens ── */
        .stApp {background:#ffffff; overflow-x:clip;}
        [data-testid="stAppViewContainer"] {overflow-x:clip;}
        [data-testid="stSidebar"] {display:none;}
        [data-testid="collapsedControl"] {display:none;}
        /* Hide Streamlit default top bar (Fork / GitHub / menu) */
        [data-testid="stHeader"] {display:none !important; visibility:hidden !important; height:0 !important;}
        header[data-testid="stHeader"] {display:none !important;}
        [data-testid="stToolbar"] {display:none !important;}
        #stDecoration {display:none !important;}
        .stApp > header {display:none !important;}
        header.stAppHeader {display:none !important;}
        .block-container,
        [data-testid="stMainBlockContainer"],
        .stMainBlockContainer {
            max-width:100% !important; width:100% !important;
            padding-top:clamp(0.4rem, 1.5vw, 0.8rem) !important;
            padding-bottom:1rem !important;
            padding-left:clamp(0.35rem, 2vw, 1.5rem) !important;
            padding-right:clamp(0.35rem, 2vw, 1.5rem) !important;
        }
        [data-testid="stAppViewContainer"] .main {max-width:100% !important;}
        [data-testid="column"] {align-items:flex-start; min-width:0 !important;}
        [data-testid="stHorizontalBlock"] {gap:0.5rem; width:100% !important;}
        .js-plotly-plot, .js-plotly-plot .plotly, [data-testid="stPlotlyChart"] {
            width:100% !important; max-width:100% !important;
        }
        iframe {max-width:100% !important;}
        /* Transport / PPAC pie legends — thinner color swatches */
        .js-plotly-plot .legend .traces .legendtoggle path,
        .js-plotly-plot .legend rect {
            width: 14px !important;
        }

        /* ── Header ── */
        .aq-header {
            background:linear-gradient(135deg,#0d4f6e 0%,#0a7a8f 50%,#0d4f6e 100%);
            border-radius:10px; padding:16px 14px 12px; margin-bottom:12px;
            box-shadow:0 2px 12px rgba(0,0,0,.25);
            width:100%;
            display:grid;
            grid-template-columns:auto 1fr auto;
            align-items:center;
            gap:10px;
        }
        .aq-head-main {text-align:center; min-width:0;}
        .aq-title {
            color:#fff;
            font-size:clamp(0.85rem, 2.8vw, 2.35rem);
            font-weight:900;
            letter-spacing:clamp(0.02em, 0.35vw, 0.08em);
            margin:0;
            line-height:1.2;
            word-break:break-word;
        }
        .aq-sub {
            color:#b8e4f0;
            font-size:clamp(0.68rem, 1.2vw, 0.92rem);
            margin:2px 0 0;
            letter-spacing:.04em;
            line-height:1.35;
        }
        .aq-logo-wrap {display:flex;justify-content:flex-start;align-items:center;}
        .aq-logo-wrap img {height:46px;width:auto;border-radius:8px;background:rgba(255,255,255,.95);padding:2px 6px;}
        .aq-logo-spacer {height:1px;width:130px;}

        /* ── Panels ── */
        .panel {background:#f4f6fa;border:1px solid #d5dbe7;border-radius:10px;padding:10px 12px;height:auto;margin-bottom:10px;}
        .panel h4, .sector-title {
            margin:0;
            color:#111827 !important;
            font-size:1.18rem;
            font-weight:900 !important;
            letter-spacing:.02em;
            line-height:1.2;
            display:block !important;
            visibility:visible !important;
            opacity:1 !important;
            text-transform:uppercase;
        }
        .transport-target-head {
            color:#111827;
            font-size:1.05rem;
            font-weight:900;
            letter-spacing:.02em;
            line-height:1.15;
            margin:0;
            text-transform:uppercase;
            text-align:left;
            white-space:nowrap;
        }
        .panel-click {cursor:pointer;transition:box-shadow .15s,border-color .15s;}
        .panel-click:hover {box-shadow:0 4px 18px rgba(10,122,143,.25);border-color:#0a7a8f;}

        /* ── Typography helpers ── */
        .mini {color:#5a6b85;font-size:.84rem;margin:4px 0 7px;}
        .big  {font-size:2.2rem;font-weight:900;color:#1a2332;line-height:1.1;}

        /* ── Pollutant KV grid ── */
        .kv {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin-top:8px;}
        .kv div {background:#e9edf5;border-radius:7px;padding:6px;}
        .kv b   {display:block;color:#17253a;font-size:1.05rem;}
        .kv span{font-size:.82rem;color:#516580;}
        .kv-aqi div {background:transparent;border:1px solid #d7deea;border-radius:4px;}

        /* ── AQI legend pills ── */
        .aqi-legend {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:5px;margin:4px 0 6px;}
        .aqi-pill   {border-radius:999px;padding:5px 7px;font-size:.72rem;font-weight:700;color:#111827;text-align:center;}
        .maplibregl-ctrl-bottom-left,
        .maplibregl-ctrl-bottom-right,
        .mapboxgl-ctrl-bottom-left,
        .mapboxgl-ctrl-bottom-right {
            display: none !important;
        }

        /* ── Fund rows ── */
        .fund-compact {margin-top:0;background:transparent;padding:0;}
        .panel .fund-compact {background:transparent;border:none;padding:0;}
        .fund-line {display:flex;justify-content:space-between;align-items:baseline;gap:6px;padding:4px 0;border-bottom:1px solid #dde3ee;font-size:.84rem;}
        .fund-line:last-of-type {border-bottom:none;}
        .fund-lbl  {color:#516580;font-size:.78rem;flex:1;line-height:1.2;}
        .fund-val  {font-weight:700;color:#17253a;font-size:.9rem;white-space:nowrap;}

        /* ── Insights ── */
        .insights {background:#f4f6fa;border:1px solid #d5dbe7;border-radius:10px;padding:10px 12px;font-size:.9rem;line-height:1.45;}
        .insights h4 {font-size:1rem;color:#1a2332;}
        .ncap-fy {color:#5a6b85;font-size:.76rem;line-height:1.3;margin:4px 0;}
        .transport-target-offset {position:relative;top:-118px;}
        .poll-naaqs {font-size:.62rem;color:#64748b;line-height:1.25;display:block;margin-top:1px;}
        .ev-slice-mini {
            text-align:center;font-size:.76rem;font-weight:700;color:#17253a;
            margin:2px 0 3px;line-height:1.2;
        }
        .ev-teal-legend {
            display:inline-flex;align-items:center;gap:5px;
            font-size:.62rem;color:#516580;
            margin:0 auto 4px;padding:2px 6px;
            border:1px solid #d5dbe7;border-radius:4px;background:#f8fafc;
            max-width:100%;line-height:1.2;
        }
        .ev-teal-legend .swatch {
            width:9px;height:9px;border-radius:2px;background:#0b7285;flex-shrink:0;
        }
        .sector-head-row {display:flex;align-items:center;gap:8px;margin-bottom:6px; flex-wrap:wrap;}
        .sector-icon {font-size:1.35rem;line-height:1;flex-shrink:0;}

        /* ── 1366px — MacBook / medium laptop ── */
        @media (max-width:1366px) {
            .transport-target-offset {position:static !important;top:auto !important;}
            .panel h4, .sector-title {font-size:1.05rem;}
            .mini {font-size:.8rem;}
            .fund-lbl {font-size:.72rem;}
            .fund-val {font-size:.84rem;}
            .aqi-pill {font-size:.66rem;}
            .big {font-size:1.85rem;}
            .kv b {font-size:.95rem;}
            .aq-logo-spacer {width:90px;}
        }

        /* ── 1100px — small laptop ── */
        @media (max-width:1100px) {
            .big {font-size:1.45rem;}
            .aq-logo-wrap img {height:36px;}
            .aq-logo-spacer {width:70px;}
            .transport-target-head {font-size:.88rem; white-space:normal;}
        }

        /* ── 900px — tablet: stack main 3 columns ── */
        @media (max-width:900px) {
            [data-testid="stHorizontalBlock"] {flex-wrap:wrap !important;}
            [data-testid="column"] {
                min-width:100% !important;
                flex:0 0 100% !important;
                width:100% !important;
            }
            .transport-target-offset {position:static !important;top:auto !important;}
            .kv {grid-template-columns:repeat(2,minmax(0,1fr));}
            .aqi-legend {grid-template-columns:repeat(3,minmax(0,1fr));}
            .transport-target-head {white-space:normal;}
        }

        /* ── 600px — phone ── */
        @media (max-width:600px) {
            .aq-header {grid-template-columns:auto 1fr; padding:12px 10px;}
            .aq-logo-spacer {display:none;}
            .aq-logo-wrap img {height:28px;}
            .panel {padding:6px 8px;}
            .kv {grid-template-columns:repeat(2,minmax(0,1fr));}
            .aqi-legend {grid-template-columns:repeat(2,minmax(0,1fr));}
            .aqi-pill {font-size:.58rem; padding:4px 5px;}
            .fund-lbl {font-size:.68rem;}
            .fund-val {font-size:.78rem;}
            .poll-naaqs {font-size:.58rem;}
            .ev-slice-mini {font-size:.7rem;}
            .ev-teal-legend {font-size:.58rem;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Load data ──────────────────────────────────────────────────────────
    data = load_data()
    daily, aqi = data["daily"], data["aqi"]
    vehicles = data["vehicles"]
    pol = data["pol"]
    imd_daily = data["imd_daily"]
    imd_summary = data["imd_summary"]
    ncap_fund = data["ncap_fund"]
    waste = data["waste"]

    latest_year = int(daily["year"].dropna().max())
    delhi_boundary_rings = load_delhi_boundary()

    city_aqi = _load_delhi_city_aqi(aqi)
    if not city_aqi.empty:
        city_latest_date = city_aqi["date"].max()
        city_row = city_aqi[city_aqi["date"] == city_latest_date].iloc[-1]
        high_aqi = float(city_row["index_value"])
        high_cat = str(city_row["air_quality_category"]).strip() or "N/A"
    else:
        city_latest_date = pd.NaT
        high_aqi, high_cat = np.nan, "N/A"
    aqi_cat, aqi_color = _aqi_category(high_aqi)
    if high_cat == "N/A" and aqi_cat != "N/A":
        high_cat = aqi_cat
    latest_lbl = (
        pd.Timestamp(city_latest_date).strftime("%B %d, %Y")
        if pd.notna(city_latest_date) else "N/A"
    )

    pm25_daily = daily[
        (daily["pollutant"] == "PM2.5") & daily["station_val"].notna()
    ].copy()
    if pd.notna(city_latest_date):
        cap = pd.Timestamp(city_latest_date).normalize()
        pm25_daily = pm25_daily[pm25_daily["date"].dt.normalize() <= cap]
    pm25_map_date = pm25_daily["date"].max() if not pm25_daily.empty else pd.NaT
    pm25_lbl = (
        pd.Timestamp(pm25_map_date).strftime("%B %d, %Y")
        if pd.notna(pm25_map_date) else "N/A"
    )

    pm25_st = (
        pm25_daily[pm25_daily["date"] == pm25_map_date][["station", "station_val"]]
        .copy()
        if pd.notna(pm25_map_date)
        else pd.DataFrame(columns=["station", "station_val"])
    )
    pm25_st["station"] = pm25_st["station"].astype(str).str.strip()
    pm25_st["pm25_latest"] = pd.to_numeric(pm25_st["station_val"], errors="coerce")
    pm25_st = pm25_st.groupby("station", as_index=False)["pm25_latest"].mean()

    map_df = pm25_st.copy()
    map_df["pm25_category"] = map_df["pm25_latest"].apply(_pm25_category)

    def _map_row_coords(station: str) -> pd.Series:
        lat, lon, plat, plon = _station_map_coords(station, delhi_boundary_rings)
        return pd.Series({"lat": lat, "lon": lon, "lat_plot": plat, "lon_plot": plon})

    _coords = map_df["station"].apply(_map_row_coords)
    map_df = pd.concat([map_df, _coords], axis=1)
    map_df = map_df.dropna(subset=["lat", "lon"]).sort_values("pm25_latest", ascending=False)
    station_cycle = map_df["station"].dropna().tolist()
    active_station = station_cycle[0] if station_cycle else "N/A"

    if "aqi_swipe_idx" not in st.session_state:
        st.session_state.aqi_swipe_idx = 0
    if "aqi_selected_station" not in st.session_state:
        st.session_state.aqi_selected_station = None
    if "aqi_last_swipe_ts" not in st.session_state:
        st.session_state.aqi_last_swipe_ts = 0.0

    @st.fragment(run_every="2.2s")
    def _aqi_autoplay_tick() -> None:
        now_ts = pd.Timestamp.now().timestamp()
        if (
            st.session_state.aqi_selected_station is None
            and len(station_cycle) > 1
            and (now_ts - st.session_state.aqi_last_swipe_ts) >= 2.1
        ):
            st.session_state.aqi_swipe_idx = (st.session_state.aqi_swipe_idx + 1) % len(station_cycle)
            st.session_state.aqi_last_swipe_ts = now_ts
            st.rerun()

    _aqi_autoplay_tick()
    if st.session_state.aqi_selected_station in station_cycle:
        active_station = st.session_state.aqi_selected_station
    elif station_cycle:
        st.session_state.aqi_swipe_idx %= len(station_cycle)
        active_station = station_cycle[st.session_state.aqi_swipe_idx]

    pvals: dict[str, float] = {}
    snap = daily[
        (daily["date"] == pm25_map_date)
        & (daily["station"] == active_station)
    ] if pd.notna(pm25_map_date) else daily.iloc[0:0]
    for pol_name in ["PM2.5", "PM10", "NO2", "O3"]:
        vv = pd.to_numeric(snap[snap["pollutant"] == pol_name]["station_val"], errors="coerce")
        pvals[pol_name] = float(vv.mean()) if not vv.dropna().empty else np.nan

    # ── Transport: latest month in vehicle registry ───────────────────────
    transport_year = int(vehicles["year"].dropna().max()) if not vehicles.empty else latest_year
    month_rank = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    month_title = {
        "JAN": "January", "FEB": "February", "MAR": "March", "APR": "April",
        "MAY": "May", "JUN": "June", "JUL": "July", "AUG": "August",
        "SEP": "September", "OCT": "October", "NOV": "November", "DEC": "December",
    }
    v_year = vehicles[vehicles["year"] == transport_year].copy() if not vehicles.empty else vehicles.copy()
    if not v_year.empty:
        v_year["month"] = v_year["month"].astype(str).str.strip().str.upper()
        v_year["month_num"] = v_year["month"].map(month_rank)
        latest_month_num = int(v_year["month_num"].dropna().max())
        if pd.notna(city_latest_date) and int(city_latest_date.year) == transport_year:
            cap_m = int(city_latest_date.month)
            if latest_month_num > cap_m:
                latest_month_num = cap_m
    else:
        latest_month_num = None
    v_latest_month = (
        v_year[v_year["month_num"] == latest_month_num].copy()
        if latest_month_num is not None and not v_year.empty
        else pd.DataFrame(columns=v_year.columns if not v_year.empty else ["vec_class_category", "total"])
    )
    latest_month_key = next((m for m, n in month_rank.items() if n == latest_month_num), "N/A")
    latest_month_heading = f"{month_title.get(latest_month_key, latest_month_key)} {transport_year}"

    class_vehicle = (
        v_latest_month.groupby("vec_class_category", as_index=False)["total"].sum()
        .rename(columns={"total": "total_vehicles"})
        .query("total_vehicles > 0")
        .sort_values("total_vehicles", ascending=False)
    )
    transport_class_cycle = class_vehicle["vec_class_category"].astype(str).tolist()

    city_reg_total = float(v_latest_month["total"].sum()) if not v_latest_month.empty else 0.0
    district_vehicle = _split_delhi_district_vehicles(city_reg_total)

    total_reg = city_reg_total
    ev_rows = v_latest_month[v_latest_month["fuel_category"].str.contains("ev", case=False, na=False)]
    ev_total = float(ev_rows["total"].sum()) if not ev_rows.empty else 0.0
    ev_share = (ev_total / total_reg * 100.0) if total_reg > 0 else 0.0

    cat_rollup = (
        v_latest_month.groupby("vec_class_category", as_index=False)["total"].sum()
        .rename(columns={"vec_class_category": "category", "total": "total_all"})
        .query("total_all > 0")
        .sort_values("total_all", ascending=False)
    )
    if not cat_rollup.empty:
        cat_rollup = cat_rollup[
            ~cat_rollup["category"].astype(str).str.strip().str.lower().isin({"other", "e-rickshaw", "e rickshaw"})
        ].copy()
        ev_cat = (
            v_latest_month[v_latest_month["fuel_category"].str.contains("ev", case=False, na=False)]
            .groupby("vec_class_category", as_index=False)["total"].sum()
            .rename(columns={"vec_class_category": "category", "total": "ev_total"})
        )
        cat_rollup = cat_rollup.merge(ev_cat, left_on="category", right_on="category", how="left").fillna({"ev_total": 0.0})
        cat_rollup["actual_pct"] = np.where(
            cat_rollup["total_all"] > 0,
            (cat_rollup["ev_total"] / cat_rollup["total_all"]) * 100.0,
            0.0,
        )
    transport_category_cycle = cat_rollup["category"].astype(str).tolist() if not cat_rollup.empty else []

    if "transport_swipe_idx" not in st.session_state:
        st.session_state.transport_swipe_idx = 0
    if "transport_selected_class" not in st.session_state:
        st.session_state.transport_selected_class = None
    if "transport_last_swipe_ts" not in st.session_state:
        st.session_state.transport_last_swipe_ts = 0.0
    if "transport_cat_swipe_idx" not in st.session_state:
        st.session_state.transport_cat_swipe_idx = 0

    @st.fragment(run_every="2.8s")
    def _transport_autoplay_tick() -> None:
        now_ts = pd.Timestamp.now().timestamp()
        if (
            st.session_state.transport_selected_class is None
            and len(transport_class_cycle) > 1
            and (now_ts - st.session_state.transport_last_swipe_ts) >= 2.7
        ):
            st.session_state.transport_swipe_idx = (st.session_state.transport_swipe_idx + 1) % len(transport_class_cycle)
            if transport_category_cycle:
                st.session_state.transport_cat_swipe_idx = (
                    (st.session_state.transport_cat_swipe_idx + 1) % len(transport_category_cycle)
                )
            st.session_state.transport_last_swipe_ts = now_ts
            st.rerun()

    _transport_autoplay_tick()
    if st.session_state.transport_selected_class in transport_class_cycle:
        active_vehicle_class = st.session_state.transport_selected_class
    elif transport_class_cycle:
        st.session_state.transport_swipe_idx %= len(transport_class_cycle)
        active_vehicle_class = transport_class_cycle[st.session_state.transport_swipe_idx]
    else:
        active_vehicle_class = "N/A"

    if transport_category_cycle:
        st.session_state.transport_cat_swipe_idx %= len(transport_category_cycle)
        active_vehicle_category = transport_category_cycle[st.session_state.transport_cat_swipe_idx]
        cat_row = cat_rollup[cat_rollup["category"] == active_vehicle_category].head(1)
        active_cat_actual_pct = float(cat_row["actual_pct"].iloc[0]) if not cat_row.empty else 0.0
    else:
        active_vehicle_category = "N/A"
        active_cat_actual_pct = ev_share

    vehicle_type_df = pd.DataFrame(columns=["label", "value"])
    fuel_type_df = pd.DataFrame(columns=["label", "value"])
    if not v_latest_month.empty:
        vehicle_type_df = (
            v_latest_month.groupby("vec_class_category", as_index=False)["total"].sum()
            .rename(columns={"vec_class_category": "label", "total": "value"})
            .query("value > 0")
            .sort_values("value", ascending=False)
            .head(7)
        )
    if active_vehicle_class != "N/A" and not v_latest_month.empty:
        class_v = v_latest_month[v_latest_month["vec_class_category"] == active_vehicle_class].copy()
        if not class_v.empty:
            fuel_type_df = (
                class_v.groupby("fuel_category", as_index=False)["total"].sum()
                .rename(columns={"fuel_category": "label", "total": "value"})
                .query("value > 0")
                .sort_values("value", ascending=False)
                .head(7)
            )

    # ── NCAP / XV FC fund (Delhi, crore ₹) ────────────────────────────────
    fund_alloc = fund_released = fund_utilised = np.nan
    if not ncap_fund.empty:
        fmap = ncap_fund.set_index("Funds")["amount_cr"].to_dict()
        fund_alloc = float(fmap.get("Fund Allocation", np.nan))
        fund_released = float(fmap.get("Fund Released", np.nan))
        fund_utilised = float(fmap.get("Fund Utilised", np.nan))
    util_pct_alloc = (
        (fund_utilised / fund_alloc * 100.0)
        if pd.notna(fund_utilised) and pd.notna(fund_alloc) and fund_alloc > 0
        else np.nan
    )

    # ── Petrol / diesel (PPAC, Delhi) ─────────────────────────────────────
    pol_latest_fy = None
    petrol_kt = diesel_kt = np.nan
    pol_trend = pd.DataFrame(columns=["fy", "product", "consumption_kt"])
    if not pol.empty:
        pol_trend = pol.copy()
        pol_latest_fy = pol["fy"].max()
        latest_pol = pol[pol["fy"] == pol_latest_fy]
        petrol_kt = float(
            latest_pol.loc[latest_pol["product"].str.lower() == "petrol", "consumption_kt"].sum()
        )
        diesel_kt = float(
            latest_pol.loc[latest_pol["product"].str.lower() == "diesel", "consumption_kt"].sum()
        )

    # ── Solid waste (TPD by FY) ───────────────────────────────────────────
    waste_latest_fy = None
    w_gen = w_coll = w_treat = w_land = np.nan
    treat_pct = coll_pct = np.nan
    if not waste.empty:
        waste_latest_fy = waste["fy"].max()
        w_latest = waste[waste["fy"] == waste_latest_fy].set_index("category")["value_tpd"]
        w_gen = float(w_latest.get("Solid Waste Generation", np.nan))
        w_coll = float(w_latest.get("Solid Waste Collected", np.nan))
        w_treat = float(w_latest.get("Solid Waste Treated", np.nan))
        w_land = float(w_latest.get("Solid Waste Landfilled", np.nan))
        if pd.notna(w_treat) and pd.notna(w_coll) and w_coll > 0:
            treat_pct = w_treat / w_coll * 100.0
        if pd.notna(w_coll) and pd.notna(w_gen) and w_gen > 0:
            coll_pct = w_coll / w_gen * 100.0

    # ── IMD meteorology (recent from PDF) ─────────────────────────────────
    met_month = "May"
    met_year = latest_year
    met_as_of = "N/A"
    met_tmax = met_tmin = met_avg_max = met_avg_min = met_rain = np.nan
    if not imd_summary.empty:
        sm = imd_summary.iloc[0]
        met_month = str(sm.get("month", met_month))
        met_year = int(pd.to_numeric(sm.get("year"), errors="coerce") or met_year)
        met_as_of = str(sm.get("as_of", met_as_of))
        met_tmax = pd.to_numeric(sm.get("latest_tmax_c"), errors="coerce")
        met_tmin = pd.to_numeric(sm.get("latest_tmin_c"), errors="coerce")
        met_avg_max = pd.to_numeric(sm.get("avg_tmax_c"), errors="coerce")
        met_avg_min = pd.to_numeric(sm.get("avg_tmin_c"), errors="coerce")
        met_rain = pd.to_numeric(sm.get("total_rainfall_mm"), errors="coerce")

    imd_chart = imd_daily.dropna(subset=["tmax_c"]).copy() if not imd_daily.empty else imd_daily

    delhi_district_geojson = load_delhi_district_geojson()
    delhi_map_center, aqi_map_zoom, _delhi_map_bounds = _delhi_map_view(
        delhi_district_geojson,
        map_height=AQI_MAP_HEIGHT,
        map_width=AQI_MAP_WIDTH,
        boundary_rings=delhi_boundary_rings,
    )
    _, transport_map_zoom, _ = _delhi_map_view(
        delhi_district_geojson,
        map_height=TRANSPORT_MAP_HEIGHT,
        map_width=TRANSPORT_MAP_WIDTH,
        boundary_rings=delhi_boundary_rings,
    )

    # ══════════════════════ RENDER ═════════════════════════════════════════
    logo_html = ""
    spacer_html = ""
    if LOGO_PATH.is_file():
        logo_ext = LOGO_PATH.suffix.lower().lstrip(".") or "jpeg"
        if logo_ext == "jpg":
            logo_ext = "jpeg"
        logo_bytes = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
        logo_src = f"data:image/{logo_ext};base64,{logo_bytes}"
        logo_html = f'<div class="aq-logo-wrap"><img src="{logo_src}" alt="EnviroCatalysts"></div>'
        spacer_html = '<div class="aq-logo-spacer"></div>'
    st.markdown(
        '<div class="aq-header">'
        + logo_html +
        '<div class="aq-head-main">'
        '<div class="aq-title">Air Quality Meteorology And Source Activity Tracker</div>'
        '<div class="aq-sub">Delhi · Multi-Sector Environmental Intelligence</div>'
        '</div>'
        + spacer_html +
        '</div>',
        unsafe_allow_html=True,
    )

    main_l, main_m, side_col = st.columns([1.0, 1.25, 0.85], gap="small")

    with main_l:
        aqi_str = "—" if np.isnan(high_aqi) else f"{high_aqi:.0f}"
        st.markdown('<div id="aqi-box-click-start"></div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<div class='sector-title' style='margin:0 0 8px 0;'>AIR QUALITY ANALYTICS</div>", unsafe_allow_html=True)
            if not map_df.empty:
                delhi_rings = delhi_boundary_rings
                fig_map = px.scatter_map(
                    map_df,
                    lat="lat_plot",
                    lon="lon_plot",
                    size="pm25_latest",
                    color="pm25_category",
                    hover_name="station",
                    custom_data=["station", "pm25_latest"],
                    hover_data={
                        "pm25_latest": ":.1f",
                        "pm25_category": True,
                        "lat_plot": False,
                        "lon_plot": False,
                    },
                    color_discrete_map=PM25_COLOR_MAP,
                    category_orders={
                        "pm25_category": ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe", "N/A"]
                    },
                    size_max=17,
                )
                fig_map.update_layout(
                    map={
                        "style": "open-street-map",
                        "center": delhi_map_center,
                        "zoom": aqi_map_zoom,
                    },
                    dragmode="pan",
                    clickmode="event+select",
                    height=AQI_MAP_HEIGHT,
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    uirevision="delhi-aqi-map",
                )
                for ring in delhi_rings:
                    lons = [pt[0] for pt in ring]
                    lats = [pt[1] for pt in ring]
                    fig_map.add_trace(
                        go.Scattermap(
                            lon=lons,
                            lat=lats,
                            mode="lines",
                            line=dict(color="#1f2937", width=2.2),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )
                fig_map.update_traces(marker={"opacity": 0.9})
                try:
                    map_event = st.plotly_chart(
                        fig_map,
                        use_container_width=True,
                        config={
                            "displayModeBar": True,
                            "scrollZoom": True,
                            "modeBarButtonsToAdd": ["zoomInMap", "zoomOutMap", "resetViewMap"],
                        },
                        on_select="rerun",
                        key="aqi_station_map",
                    )
                    if isinstance(map_event, dict):
                        pts = map_event.get("selection", {}).get("points", [])
                        if pts:
                            cdata = pts[0].get("customdata")
                            if isinstance(cdata, (list, tuple)) and len(cdata) > 0:
                                picked_station = str(cdata[0]).strip()
                                if picked_station in station_cycle:
                                    st.session_state.aqi_selected_station = picked_station
                                    st.session_state.aqi_swipe_idx = station_cycle.index(picked_station)
                            else:
                                pidx = pts[0].get("point_index")
                                if isinstance(pidx, int) and 0 <= pidx < len(map_df):
                                    picked_station = str(map_df.iloc[pidx]["station"]).strip()
                                    if picked_station in station_cycle:
                                        st.session_state.aqi_selected_station = picked_station
                                        st.session_state.aqi_swipe_idx = station_cycle.index(picked_station)
                            st.rerun()
                except TypeError:
                    st.plotly_chart(
                        fig_map,
                        use_container_width=True,
                        config={
                            "displayModeBar": True,
                            "scrollZoom": True,
                            "modeBarButtonsToAdd": ["zoomInMap", "zoomOutMap", "resetViewMap"],
                        },
                    )

            st.markdown(
                f"""
                <a href="{AQI_DASH_URL}" target="_blank" style="text-decoration:none">
                  <div class="mini">City AQI (CPCB) · {latest_lbl}</div>
                  <div class="big" style="font-size:1.05rem;font-weight:600;line-height:1.2;">{DELHI_CITY}</div>
                </a>
                <div class="mini" style="margin:2px 0 6px;">
                  <a href="{AQI_DASH_URL}" target="_blank" style="font-weight:700;text-decoration:none;">More details</a>
                </div>
                """,
                unsafe_allow_html=True,
            )
            g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=0 if np.isnan(high_aqi) else float(high_aqi),
                number={"prefix": "AQI-", "font": {"size": 24, "color": "#1f2937"}},
                gauge={
                    "axis": {"range": [0, 500], "tickvals": [], "ticktext": []},
                    "bar": {"color": "rgba(0,0,0,0)"},
                    "steps": [
                        {"range": [0, 50], "color": "#00b050"},
                        {"range": [50, 100], "color": "#92d050"},
                        {"range": [100, 200], "color": "#ffff00"},
                        {"range": [200, 300], "color": "#ffa500"},
                        {"range": [300, 400], "color": "#ff0000"},
                        {"range": [400, 500], "color": "#7b0000"},
                    ],
                },
            ))
            g.update_layout(height=155, margin=dict(l=6, r=6, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)")
            g.add_annotation(
                x=0.5, y=0.18, xref="paper", yref="paper",
                text=f"<b>{high_cat}</b>",
                showarrow=False,
                font=dict(size=12, color=aqi_color),
            )
            st.plotly_chart(g, use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                f"""
                <div class="aqi-legend">
                  <div class="aqi-pill" style="background:#00b050;">Good (0-50)</div>
                  <div class="aqi-pill" style="background:#92d050;">Satisfactory (51-100)</div>
                  <div class="aqi-pill" style="background:#ffff00;">Moderate (101-200)</div>
                  <div class="aqi-pill" style="background:#ffa500;">Poor (201-300)</div>
                  <div class="aqi-pill" style="background:#ff0000;color:#fff;">Very Poor (301-400)</div>
                  <div class="aqi-pill" style="background:#7b0000;color:#fff;">Severe (401-500)</div>
                </div>
                <div class="mini" style="text-align:center;margin:8px 0 2px;line-height:1.25;">
                  <div style="font-weight:600;color:#17253a;font-size:.92rem;">{active_station}</div>
                  <div>Station 24h avg concentrations · {pm25_lbl}</div>
                </div>
                <div style="display:flex;justify-content:center;gap:6px;margin:6px 0 2px;">
                  {"".join(
                    f'<span style="width:7px;height:7px;border-radius:50%;display:inline-block;'
                    f'background:{"#0a7a8f" if i == (station_cycle.index(active_station) if active_station in station_cycle else 0) else "#c7d2e5"}"></span>'
                    for i in range(max(len(station_cycle), 1))
                  )}
                </div>
                <div class="kv kv-aqi">
                  <div><b>{'—' if np.isnan(pvals['PM2.5']) else f"{pvals['PM2.5']:.0f}"}</b><span>{_pollutant_kv_label('PM2.5')}</span></div>
                  <div><b>{'—' if np.isnan(pvals['PM10']) else f"{pvals['PM10']:.0f}"}</b><span>{_pollutant_kv_label('PM10')}</span></div>
                  <div><b>{'—' if np.isnan(pvals['NO2']) else f"{pvals['NO2']:.0f}"}</b><span>{_pollutant_kv_label('NO2')}</span></div>
                  <div><b>{'—' if np.isnan(pvals['O3']) else f"{pvals['O3']:.0f}"}</b><span>{_pollutant_kv_label('O3')}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # auto-swipe handled via timed rerun above (no full-page reload flicker)
        st.markdown('<div id="aqi-box-click-end"></div>', unsafe_allow_html=True)
        components.html(
            f"""
            <script>
            (function() {{
              const url = {AQI_DASH_URL!r};
              const doc = window.parent.document;
              const start = doc.getElementById("aqi-box-click-start");
              const end = doc.getElementById("aqi-box-click-end");
              if (!start || !end) return;

              function bindClick(el) {{
                if (!el || el.dataset?.aqiBoxBound === "1") return;
                el.dataset.aqiBoxBound = "1";
                el.style.cursor = "pointer";
                el.addEventListener("click", function(ev) {{
                  if (ev.target.closest("a")) return;
                  window.open(url, "_blank", "noopener,noreferrer");
                }});
              }}

              let node = start.nextElementSibling;
              while (node && node !== end) {{
                bindClick(node);
                node = node.nextElementSibling;
              }}
            }})();
            </script>
            """,
            height=0,
        )

        with st.container(border=True):
            st.markdown(
                f"""
                <div class="sector-head-row">
                  <span class="sector-icon">♻️</span>
                  <h4 class="sector-title" style="margin:0;">SOLID WASTE MANAGEMENT</h4>
                </div>
                <div class="ncap-fy">Delhi · FY {waste_latest_fy or "N/A"} · tonnes per day (TPD)</div>
                <div class="fund-compact">
                  <div class="fund-line"><span class="fund-lbl">Generation</span><span class="fund-val">{("—" if np.isnan(w_gen) else f"{w_gen:,.0f} TPD")}</span></div>
                  <div class="fund-line"><span class="fund-lbl">Collected</span><span class="fund-val">{("—" if np.isnan(w_coll) else f"{w_coll:,.0f} TPD")}</span></div>
                  <div class="fund-line"><span class="fund-lbl">Treated</span><span class="fund-val">{("—" if np.isnan(w_treat) else f"{w_treat:,.0f} TPD")}</span></div>
                  <div class="fund-line"><span class="fund-lbl">Landfilled</span><span class="fund-val">{("—" if np.isnan(w_land) else f"{w_land:,.0f} TPD")}</span></div>
                </div>
                <div style="margin-top:8px;">
                  <div style="height:8px;border-radius:999px;background:#dbe4f3;overflow:hidden;">
                    <div style="height:8px;width:{max(0, min(100, treat_pct if pd.notna(treat_pct) else 0)):.1f}%;background:linear-gradient(90deg,#2b8a3e,#74b816);"></div>
                  </div>
                  <div class="mini" style="margin:4px 0 0;text-align:right;">Treated vs collected: {("—" if np.isnan(treat_pct) else f"{treat_pct:.1f}%")}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if not waste.empty:
                w_chart = waste[waste["category"].isin(
                    ["Solid Waste Generation", "Solid Waste Collected", "Solid Waste Treated"]
                )].copy()
                w_chart["category"] = w_chart["category"].str.replace("Solid Waste ", "", regex=False)
                fig_waste = px.bar(
                    w_chart,
                    x="fy",
                    y="value_tpd",
                    color="category",
                    barmode="group",
                    color_discrete_map={
                        "Generation": "#64748b",
                        "Collected": "#0b7285",
                        "Treated": "#2b8a3e",
                    },
                    labels={"value_tpd": "TPD", "fy": "FY", "category": ""},
                )
                fig_waste.update_layout(
                    height=132,
                    margin=dict(l=48, r=8, t=30, b=52),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#111827"),
                    legend=dict(
                        orientation="h",
                        y=1.14,
                        yanchor="bottom",
                        x=0,
                        font=dict(size=8, color="#111827"),
                    ),
                    **_bar_chart_axes(y_title="TPD", x_title="FY", x_angle=-25),
                )
                st.plotly_chart(fig_waste, use_container_width=True, config={"displayModeBar": False})

    with main_m:
        st.markdown('<div id="transport-box-click-start"></div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(
                f"""<div style='display:flex;justify-content:space-between;align-items:baseline;margin:0 0 8px 0;'>
                  <div class='sector-title'>TRANSPORT SECTOR ANALYTICS</div>
                  <div style='text-align:right;line-height:1.2;'>
                    <div class='transport-target-head' style='margin:0;font-size:.82rem;white-space:nowrap;'>DELHI EV POLICY</div>
                    <div class='mini' style='margin:0;font-size:.68rem;'>{latest_month_heading}</div>
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )
            district_geojson = delhi_district_geojson
            if not district_vehicle.empty and district_geojson.get("features"):
                map_col, target_col = st.columns([2.15, 1.0], gap="small")
                with map_col:
                    data_min = float(district_vehicle["total_vehicles"].min())
                    data_max = float(district_vehicle["total_vehicles"].max())
                    cb_min, cb_max, cb_tickvals, cb_ticktext = _transport_colorbar_ticks(data_min, data_max)
                    fig_transport_map = px.choropleth_mapbox(
                        district_vehicle,
                        geojson=district_geojson,
                        locations="district_geo",
                        featureidkey="properties.Dist_Name",
                        color="total_vehicles",
                        range_color=(cb_min, cb_max),
                        color_continuous_scale=[
                            [0.00, "#fff8e1"],
                            [0.25, "#ffd166"],
                            [0.50, "#fca311"],
                            [0.75, "#e85d04"],
                            [1.00, "#9d0208"],
                        ],
                        mapbox_style="open-street-map",
                        center=delhi_map_center,
                        zoom=transport_map_zoom,
                        opacity=0.78,
                        hover_name="district",
                        hover_data={"total_vehicles": ":,.0f", "district_geo": False},
                        labels={"total_vehicles": "Registered vehicles (est.)"},
                    )
                    fig_transport_map.add_annotation(
                        x=0.01, y=0.97, xref="paper", yref="paper",
                        text=f"<b>{latest_month_heading}</b>",
                        showarrow=False,
                        font=dict(size=11, color="#000000", family="Arial Black"),
                        bgcolor="rgba(255,255,255,0.75)",
                        borderpad=3,
                        xanchor="left", yanchor="top",
                    )
                    if delhi_boundary_rings:
                        for ring in delhi_boundary_rings:
                            fig_transport_map.add_trace(
                                go.Scattermapbox(
                                    lon=[pt[0] for pt in ring],
                                    lat=[pt[1] for pt in ring],
                                    mode="lines",
                                    line=dict(color="#1f2937", width=2.2),
                                    hoverinfo="skip",
                                    showlegend=False,
                                )
                            )
                    fig_transport_map.update_layout(
                        dragmode="pan",
                        height=TRANSPORT_MAP_HEIGHT,
                        margin=dict(l=0, r=0, t=0, b=0),
                        mapbox=dict(
                            style="open-street-map",
                            center=delhi_map_center,
                            zoom=transport_map_zoom,
                        ),
                        mapbox_center=delhi_map_center,
                        mapbox_zoom=transport_map_zoom,
                        uirevision="delhi-transport-map",
                        coloraxis=dict(cmin=cb_min, cmax=cb_max),
                        coloraxis_colorbar=dict(
                            title=dict(text="Registered vehicles (count)", font=dict(color="#111827", size=8)),
                            orientation="v",
                            thickness=4,
                            x=0.98,
                            xanchor="right",
                            y=0.5,
                            yanchor="middle",
                            len=0.46,
                            ticks="outside",
                            tickmode="array",
                            tickvals=cb_tickvals,
                            ticktext=cb_ticktext,
                            nticks=len(cb_tickvals),
                            tickfont=dict(color="#111827", size=7),
                            bgcolor="rgba(255,255,255,0.55)",
                            outlinecolor="#cbd5e1",
                            outlinewidth=0.5,
                        ),
                    )
                    st.plotly_chart(
                        fig_transport_map,
                        use_container_width=True,
                        config={
                            "displayModeBar": True,
                            "scrollZoom": True,
                            "modeBarButtonsToAdd": ["zoomInMap", "zoomOutMap", "resetViewMap"],
                        },
                    )
                with target_col:
                    st.markdown("<div class='transport-target-offset'>", unsafe_allow_html=True)
                    fig_target = _ev_target_pie_figure(
                        active_vehicle_category, active_cat_actual_pct,
                    )
                    st.plotly_chart(fig_target, use_container_width=True, config={"displayModeBar": False})
                    st.markdown(
                        f"""
                        <div style="text-align:center;">
                          <div class="ev-slice-mini">{active_vehicle_category} · {active_cat_actual_pct:.1f}%</div>
                          <div class="ev-teal-legend">
                            <span class="swatch"></span>
                            <span>Teal = actual EV share</span>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"""
                        <div style="display:flex;justify-content:center;gap:6px;margin:2px 0 2px;">
                          {"".join(
                            f'<span style="width:6px;height:6px;border-radius:50%;display:inline-block;'
                            f'background:{"#0a7a8f" if i == (transport_category_cycle.index(active_vehicle_category) if active_vehicle_category in transport_category_cycle else 0) else "#c7d2e5"}"></span>'
                            for i in range(max(len(transport_category_cycle), 1))
                          )}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("District-wise transport map data is unavailable for the selected latest period.")

            st.markdown(
                f"""
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;align-items:center;margin:6px 0 2px;">
                  <div class="mini" style="margin:0;font-size:.82rem;font-weight:700;">{latest_month_heading}</div>
                  <div class="mini" style="margin:0;text-align:center;font-size:1.08rem;font-weight:600;color:#1a2332;line-height:1.05;">{active_vehicle_class}</div>
                  <div style="text-align:right;"><a href="{TRANSPORT_DASH_URL}" target="_blank" style="font-weight:700;text-decoration:none;font-size:.8rem;">More details</a></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if not vehicle_type_df.empty or not fuel_type_df.empty:
                c1, c2 = st.columns(2, gap="medium")
                vt_total = float(vehicle_type_df["value"].sum()) if not vehicle_type_df.empty else 0.0
                ft_total = float(fuel_type_df["value"].sum()) if not fuel_type_df.empty else 0.0
                pie_colors_vt = ["#0b7285", "#1d4ed8", "#7c3aed", "#f59e0b", "#2b8a3e", "#e03131", "#6b7280"]
                pie_colors_ft = ["#1d4ed8", "#0b7285", "#f59e0b", "#e8590c", "#7c3aed", "#c2255c", "#64748b"]
                with c1:
                    st.markdown(
                        "<div class='mini' style='text-align:center;margin:0 0 2px;font-weight:700;color:#111827;'>Vehicle Type</div>",
                        unsafe_allow_html=True,
                    )
                    if not vehicle_type_df.empty:
                        fig_vt = _transport_pie_figure(
                            vehicle_type_df["label"].tolist(),
                            vehicle_type_df["value"].tolist(),
                            pie_colors_vt[: len(vehicle_type_df)],
                            vt_total,
                        )
                        st.plotly_chart(fig_vt, use_container_width=True, config={"displayModeBar": False})
                with c2:
                    st.markdown(
                        "<div class='mini' style='text-align:center;margin:0 0 2px;font-weight:700;color:#111827;'>Fuel Type</div>",
                        unsafe_allow_html=True,
                    )
                    if not fuel_type_df.empty:
                        fig_ft = _transport_pie_figure(
                            fuel_type_df["label"].tolist(),
                            fuel_type_df["value"].tolist(),
                            pie_colors_ft[: len(fuel_type_df)],
                            ft_total,
                        )
                        st.plotly_chart(fig_ft, use_container_width=True, config={"displayModeBar": False})
                st.markdown(
                    f"""
                    <div style="display:flex;justify-content:center;gap:6px;margin:2px 0 4px;">
                      {"".join(
                        f'<span style="width:7px;height:7px;border-radius:50%;display:inline-block;'
                        f'background:{"#0a7a8f" if i == (transport_class_cycle.index(active_vehicle_class) if active_vehicle_class in transport_class_cycle else 0) else "#c7d2e5"}"></span>'
                        for i in range(max(len(transport_class_cycle), 1))
                      )}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="sector-head-row">
                  <span class="sector-icon">🌦️</span>
                  <h4 class="sector-title" style="margin:0;color:#111827 !important;">METEOROLOGY · DELHI</h4>
                </div>
                <div class="ncap-fy">{met_month} {met_year} · as of {met_as_of}</div>
                <div class="fund-compact">
                  <div class="fund-line"><span class="fund-lbl">Latest max</span><span class="fund-val">{("—" if np.isnan(met_tmax) else f"{met_tmax:.1f} °C")}</span></div>
                  <div class="fund-line"><span class="fund-lbl">Latest min</span><span class="fund-val">{("—" if np.isnan(met_tmin) else f"{met_tmin:.1f} °C")}</span></div>
                  <div class="fund-line"><span class="fund-lbl">Month avg max / min</span><span class="fund-val">{("—" if np.isnan(met_avg_max) else f"{met_avg_max:.1f}")} / {("—" if np.isnan(met_avg_min) else f"{met_avg_min:.1f}")} °C</span></div>
                  <div class="fund-line"><span class="fund-lbl">Month rainfall (so far)</span><span class="fund-val">{("—" if np.isnan(met_rain) else f"{met_rain:.1f} mm")}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if not imd_chart.empty:
                fig_met = go.Figure()
                fig_met.add_trace(
                    go.Scatter(
                        x=imd_chart["day"],
                        y=imd_chart["tmax_c"],
                        mode="lines+markers",
                        name="Max °C",
                        line=dict(color="#e03131", width=2),
                        marker=dict(size=4),
                    )
                )
                if imd_chart["tmin_c"].notna().any():
                    fig_met.add_trace(
                        go.Scatter(
                            x=imd_chart["day"],
                            y=imd_chart["tmin_c"],
                            mode="lines+markers",
                            name="Min °C",
                            line=dict(color="#1d4ed8", width=2),
                            marker=dict(size=4),
                        )
                    )
                fig_met.update_layout(
                    height=118,
                    margin=dict(l=0, r=0, t=4, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", y=1.22, x=0, font=dict(size=8, color="#111827")),
                    xaxis=dict(title="Day", dtick=3, tickfont=dict(color="#111827", size=8)),
                    yaxis=dict(title="°C", gridcolor="#d1d5db", tickfont=dict(color="#111827", size=8)),
                )
                st.plotly_chart(fig_met, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div id="transport-box-click-end"></div>', unsafe_allow_html=True)
        components.html(
            f"""
            <script>
            (function() {{
              const url = {TRANSPORT_DASH_URL!r};
              const doc = window.parent.document;
              const start = doc.getElementById("transport-box-click-start");
              const end = doc.getElementById("transport-box-click-end");
              if (!start || !end) return;
              function bindClick(el) {{
                if (!el || el.dataset?.transportBoxBound === "1") return;
                el.dataset.transportBoxBound = "1";
                el.style.cursor = "pointer";
                el.addEventListener("click", function(ev) {{
                  if (ev.target.closest("a")) return;
                  window.open(url, "_blank", "noopener,noreferrer");
                }});
              }}
              let node = start.nextElementSibling;
              while (node && node !== end) {{
                bindClick(node);
                node = node.nextElementSibling;
              }}
            }})();
            </script>
            """,
            height=0,
        )

    with side_col:
        with st.container(border=True):
            st.markdown(
                f"""
                <h4 class="sector-title" style="margin-bottom:6px;">NCAP · FINANCIAL PROGRESS</h4>
                <div class="ncap-fy">Delhi · XV FC / NCAP (₹ crore)</div>
                <div class="fund-compact">
                  <div class="fund-line"><span class="fund-lbl">Fund Allocation</span><span class="fund-val">₹{fund_alloc:,.2f} cr</span></div>
                  <div class="fund-line"><span class="fund-lbl">Fund Released</span><span class="fund-val">₹{fund_released:,.2f} cr</span></div>
                  <div class="fund-line"><span class="fund-lbl">Fund Utilised</span><span class="fund-val">₹{fund_utilised:,.2f} cr</span></div>
                </div>
                <div style="margin-top:8px;">
                  <div style="height:8px;border-radius:999px;background:#dbe4f3;overflow:hidden;">
                    <div style="height:8px;width:{max(0, min(100, util_pct_alloc if pd.notna(util_pct_alloc) else 0)):.1f}%;background:linear-gradient(90deg,#0ea5e9,#2563eb);"></div>
                  </div>
                  <div class="mini" style="margin:4px 0 0;text-align:right;color:#111827;font-weight:600;">Utilisation vs allocation: {("—" if np.isnan(util_pct_alloc) else f"{util_pct_alloc:.1f}%")}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if pd.notna(fund_alloc):
                fig_ncap = go.Figure()
                fig_ncap.add_bar(
                    x=["Allocation", "Released", "Utilised"],
                    y=[fund_alloc, fund_released, fund_utilised],
                    marker_color=["#f59e0b", "#2563eb", "#b91c1c"],
                    text=[
                        f"₹{fund_alloc:.1f}",
                        f"₹{fund_released:.1f}" if pd.notna(fund_released) else "",
                        f"₹{fund_utilised:.1f}" if pd.notna(fund_utilised) else "",
                    ],
                    textposition="outside",
                    textfont=dict(size=9, color="#111827"),
                )
                fig_ncap.update_layout(
                    height=140,
                    margin=dict(l=4, r=4, t=8, b=28),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#111827"),
                    showlegend=False,
                    **_bar_chart_axes(y_title="₹ crore"),
                )
                st.plotly_chart(fig_ncap, use_container_width=True, config={"displayModeBar": False})

        with st.container(border=True):
            st.markdown(
                f"""
                <h4 class="sector-title" style="margin-bottom:6px;">PPAC · STATE FUEL SALES</h4>
                <div class="ncap-fy">Delhi · annual PPAC data · &apos;000 metric tonnes · FY {pol_latest_fy or "N/A"}</div>
                <div class="fund-compact">
                  <div class="fund-line"><span class="fund-lbl">Petrol (MS)</span><span class="fund-val">{("—" if np.isnan(petrol_kt) else f"{petrol_kt:,.0f} kt")}</span></div>
                  <div class="fund-line"><span class="fund-lbl">Diesel (HSD)</span><span class="fund-val">{("—" if np.isnan(diesel_kt) else f"{diesel_kt:,.0f} kt")}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if pd.notna(petrol_kt) and pd.notna(diesel_kt) and (petrol_kt + diesel_kt) > 0:
                pol_pie_labels = [
                    f"Petrol {petrol_kt:,.0f} kt",
                    f"Diesel {diesel_kt:,.0f} kt",
                ]
                fig_pol_pie = go.Figure(
                    go.Pie(
                        labels=pol_pie_labels,
                        values=[petrol_kt, diesel_kt],
                        hole=0.52,
                        marker=dict(colors=["#1d4ed8", "#e03131"], line=dict(color="#fff", width=1.5)),
                        textinfo="percent",
                        textposition="inside",
                        textfont=dict(size=12, color="#ffffff", family="Arial Black"),
                        insidetextorientation="horizontal",
                        showlegend=True,
                        hovertemplate="%{label}<br>%{percent}<extra></extra>",
                    )
                )
                fig_pol_pie.update_layout(
                    height=TRANSPORT_DONUT_HEIGHT,
                    margin=dict(l=4, r=4, t=8, b=40),
                    paper_bgcolor="rgba(0,0,0,0)",
                    uniformtext=dict(minsize=11, mode="hide"),
                    legend={**TRANSPORT_DONUT_LEGEND, "font": dict(size=10, color="#111827")},
                )
                st.plotly_chart(fig_pol_pie, use_container_width=True, config={"displayModeBar": False})

            if not pol_trend.empty:
                trend = pol_trend.pivot_table(
                    index="fy", columns="product", values="consumption_kt", aggfunc="sum"
                ).reset_index()
                trend = trend.sort_values("fy").tail(6)
                trend["fy_short"] = trend["fy"].astype(str).apply(
                    lambda s: f"'{s[2:4]}-{s[7:9]}" if len(s) >= 9 and "-" in s else s
                )
                fig_pol_trend = go.Figure()
                if "Petrol" in trend.columns:
                    fig_pol_trend.add_bar(
                        x=trend["fy_short"],
                        y=trend["Petrol"],
                        name="Petrol",
                        marker_color="#1d4ed8",
                        hovertemplate="%{x}<br>Petrol: %{y:,.0f} kt<extra></extra>",
                    )
                if "Diesel" in trend.columns:
                    fig_pol_trend.add_bar(
                        x=trend["fy_short"],
                        y=trend["Diesel"],
                        name="Diesel",
                        marker_color="#e03131",
                        hovertemplate="%{x}<br>Diesel: %{y:,.0f} kt<extra></extra>",
                    )
                fig_pol_trend.update_layout(
                    barmode="group",
                    height=188,
                    margin=dict(l=44, r=8, t=32, b=48),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    bargap=0.22,
                    legend=dict(
                        orientation="h",
                        y=1.08,
                        yanchor="bottom",
                        x=0.5,
                        xanchor="center",
                        font=dict(size=11, color="#111827"),
                    ),
                    yaxis=dict(
                        title=dict(text="kt", font=dict(size=11, color="#111827")),
                        tickfont=dict(size=10, color="#111827"),
                        gridcolor="#d1d5db",
                        color="#111827",
                    ),
                    xaxis=dict(
                        title=None,
                        tickangle=-35,
                        tickfont=dict(size=10, color="#111827"),
                        color="#111827",
                        automargin=True,
                    ),
                )
                st.plotly_chart(fig_pol_trend, use_container_width=True, config={"displayModeBar": False})

if __name__ == "__main__":
    main()
