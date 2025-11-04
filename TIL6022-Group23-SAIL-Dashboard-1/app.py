# app.py

import re
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import HeatMap


# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).parent
SENSORS_XLSX = BASE_DIR / "Sensor Location Data.xlsx"
FLOW_CSV     = BASE_DIR / "SAIL2025_LVMA_data_3min_20August-25August2025_flow.csv"
CITY_CENTER  = [52.377956, 4.897070]  # Amsterdam fallback center
DEFAULT_WINDOW_MIN = 15

st.set_page_config(page_title="SAIL Sensors — Per-Sensor Counts & Heatmap", layout="wide")


# ---------------- HELPERS ----------------
def _norm(s: str) -> str:
    """Normalize IDs so codes from CSV match Excel codes."""
    s = str(s).strip().lower()
    s = re.sub(r'\.\d+$', '', s)           # drop trailing ".1"
    s = re.sub(r'[-_ ]+[a-z]$', '', s)     # drop trailing "-a" / "-b"
    s = re.sub(r'[^a-z0-9]', '', s)        # keep only a-z0-9
    return s


# ---------------- LOADERS ----------------
@st.cache_data(show_spinner=False)
def load_sensors(path: Path) -> pd.DataFrame:
    """
    Sensors Excel expected columns (case-insensitive):
      - 'Objectnummer'  (sensor code)
      - 'Locatienaam'   (human-friendly name)
      - 'Lat/Long'      (e.g., '52.37, 4.89') or separate 'lat'/'lon'
    """
    df = pd.read_excel(path)
    df.columns = [c.strip().lower() for c in df.columns]

    code_col = next((c for c in df.columns if "object" in c or "sensor" in c or c == "code"), None)
    name_col = next((c for c in df.columns if "locatie" in c or "name" in c or "label" in c), None)
    if code_col is None or name_col is None:
        raise ValueError("Could not find 'Objectnummer' (code) or 'Locatienaam' in sensors file.")

    if "lat/long" in df.columns:
        latlon = df["lat/long"].astype(str).str.split(",", n=1, expand=True)
        df["_lat"] = pd.to_numeric(latlon[0], errors="coerce")
        df["_lon"] = pd.to_numeric(latlon[1], errors="coerce")
    else:
        lat_col = next((c for c in df.columns if "lat" in c), None)
        lon_col = next((c for c in df.columns if "lon" in c), None)
        if not lat_col or not lon_col:
            raise ValueError("Lat/Long columns not found in sensors file.")
        df["_lat"] = pd.to_numeric(df[lat_col], errors="coerce")
        df["_lon"] = pd.to_numeric(df[lon_col], errors="coerce")

    out = df[[code_col, name_col, "_lat", "_lon"]].copy()
    out.columns = ["code", "location_name", "_lat", "_lon"]
    out = out.dropna(subset=["_lat", "_lon"])
    out["join_key"] = out["code"].apply(_norm)
    return out


@st.cache_data(show_spinner=False)
def load_flow_wide_to_long(path: Path) -> pd.DataFrame:
    """
    WIDE flow CSV (each sensor code is a column) -> LONG.
    Uses 'timestamp' (dd/mm/yyyy) + 'Time' (HH:MM:SS+TZ) to build a proper _t.
    """
    # Robust read (auto-detect delimiter)
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path)

    # --- Build _t from DATE + TIME, stripping timezone like +02:00 ---
    def find_col(cands):
        for c in df.columns:
            if c.strip().lower() in cands:
                return c
        return None

    date_col = find_col({"timestamp", "date", "datum"})
    time_col = find_col({"time", "tijd"})

    if date_col is None or time_col is None:
        # Fallbacks
        one_dt_col = find_col({"datetime", "dt", "_t"})
        if one_dt_col is not None:
            dt = pd.to_datetime(df[one_dt_col], errors="coerce", dayfirst=True)
        else:
            dt = pd.date_range("2025-08-20 00:00:00", periods=len(df), freq="3min")
        df["_t"] = dt
    else:
        d = df[date_col].astype(str).str.strip()
        # strip timezone (e.g. 00:12:00+02:00 -> 00:12:00)
        t = df[time_col].astype(str).str.strip().str.replace(r"\+.*", "", regex=True)
        dt = pd.to_datetime(d + " " + t, errors="coerce", dayfirst=True)
        if pd.api.types.is_datetime64tz_dtype(dt):
            dt = dt.dt.tz_localize(None)
        df["_t"] = dt

    # Keep only valid timestamps
    df = df[df["_t"].notna()].copy()

    # --- Melt WIDE -> LONG ---
    id_vars = ["_t"]
    drop_cols = {c for c in [date_col, time_col] if c is not None}
    value_vars = [c for c in df.columns if c not in id_vars and c not in drop_cols]

    long_df = df.melt(id_vars=id_vars, value_vars=value_vars,
                      var_name="code", value_name="value")

    # --- Numeric cleaning ---
    long_df["value"] = (
        long_df["value"].astype(str)
        .str.replace("\xa0", "", regex=False)  # NBSP
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)    # decimal comma -> dot
    )
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce").fillna(0)

    # --- Join key like sensors ---
    long_df["join_key"] = long_df["code"].apply(_norm)

    return long_df


def agg_window(long_df: pd.DataFrame, selected_dt: datetime, window_minutes: int) -> pd.DataFrame:
    """Sum values per join_key within ±window minutes around selected_dt."""
    start = selected_dt - timedelta(minutes=window_minutes)
    end   = selected_dt + timedelta(minutes=window_minutes)
    sub = long_df.loc[(long_df["_t"] >= start) & (long_df["_t"] <= end), ["join_key", "value"]]
    if sub.empty:
        return pd.DataFrame(columns=["join_key", "value_sum"])
    agg = sub.groupby("join_key", as_index=False)["value"].sum()
    agg.rename(columns={"value": "value_sum"}, inplace=True)
    return agg


def latest_nonzero_dt(long_df: pd.DataFrame) -> datetime:
    totals = long_df.groupby("_t", as_index=False)["value"].sum()
    nz = totals.loc[totals["value"] > 0]
    if not nz.empty:
        return nz["_t"].max().to_pydatetime()
    return long_df["_t"].min().to_pydatetime()


# ---------------- MAP ----------------
def make_base_map(sensors_df: pd.DataFrame) -> folium.Map:
    center = [sensors_df["_lat"].mean(), sensors_df["_lon"].mean()] if not sensors_df.empty else CITY_CENTER
    return folium.Map(location=center, zoom_start=13, tiles="cartodbpositron")


def _bubble_color(count: float) -> str:
    if count >= 200: return "#E74C3C"  # red
    if count >=  80: return "#F1C40F"  # amber
    if count >    0: return "#7DCEA0"  # green
    return "#95A5A6"                   # gray


def add_bubbles(m: folium.Map, df: pd.DataFrame, selected_dt: datetime, window_minutes: int) -> None:
    if df.empty:
        return
    low, high = int(df["count"].min()), int(df["count"].max())
    for _, r in df.iterrows():
        count = int(r["count"])
        color = _bubble_color(count)
        radius = 18 if high == low else int(18 + 36 * (count - low) / max(1, (high - low)))
        html = f"""
        <div style="
            width:{radius*2}px;height:{radius*2}px;border-radius:50%;
            background:{color};opacity:0.92;border:4px solid rgba(255,255,255,0.95);
            display:flex;align-items:center;justify-content:center;
            font-weight:700;color:#1B2631;box-shadow:0 2px 8px rgba(0,0,0,0.25);">
            {count}
        </div>
        """
        folium.Marker(
            location=[float(r["_lat"]), float(r["_lon"])],
            icon=folium.DivIcon(html=html),
            tooltip=(f"{r['location_name']}<br>"
                     f"<b>Count:</b> {count}<br>"
                     f"<b>Time:</b> {selected_dt:%Y-%m-%d %H:%M} (±{window_minutes}m)")
        ).add_to(m)


def add_heatmap(m: folium.Map, df: pd.DataFrame, radius_px: int) -> None:
    pts = [[float(r["_lat"]), float(r["_lon"]), float(r["count"])] for _, r in df.iterrows() if r["count"] > 0]
    if pts:
        HeatMap(pts, radius=radius_px, blur=radius_px*0.6, max_zoom=16).add_to(m)


# ---------------- UI ----------------
st.title("🌊 SAIL Sensors — Per-Sensor Counts & Heatmap")
tab_map, tab_detail = st.tabs(["🗺️ Map", "📈 Sensor Details"])

with st.sidebar:
    st.header("📁 Files")
    st.text(f"Sensors: {SENSORS_XLSX.name}")
    st.text(f"Flow: {FLOW_CSV.name}")
    viz_mode       = st.radio("Visualization", ["Bubbles", "Heatmap", "Both"], index=0)
    heat_radius_px = st.slider("Heatmap radius (px)", 10, 100, 48, 2)
    window_minutes = st.slider("± minutes around time (smoothing)", 0, 60, DEFAULT_WINDOW_MIN, 1)

# Load data
try:
    sensors   = load_sensors(SENSORS_XLSX)
    flow_long = load_flow_wide_to_long(FLOW_CSV)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

# ---- Date picker ----
dates = sorted(flow_long["_t"].dt.date.unique())
if not dates:
    st.error("No timestamps found in the flow file.")
    st.stop()

auto_dt   = latest_nonzero_dt(flow_long)
auto_date = auto_dt.date()
date_idx  = dates.index(auto_date) if auto_date in dates else 0

c1, c2 = st.columns([1, 1])
with c1:
    selected_date = st.selectbox(
        "📅 Pick a date",
        options=dates,
        index=date_idx,
        format_func=lambda d: d.strftime("%Y-%m-%d"),
    )

# ---- Time slider (time-only label; values are naive datetimes) ----
day_mask  = flow_long["_t"].dt.date == selected_date
day_times = pd.to_datetime(flow_long.loc[day_mask, "_t"], errors="coerce")

if pd.api.types.is_datetime64tz_dtype(day_times):
    day_times = day_times.dt.tz_localize(None)

day_times = day_times.dropna().sort_values().unique()

if len(day_times) == 0:
    st.warning("No times found for this date. Please choose another date.")
    st.stop()

t_min = pd.Timestamp(day_times[0]).to_pydatetime()
t_max = pd.Timestamp(day_times[-1]).to_pydatetime()

# If only one time, pad so min < max (Streamlit requirement)
if t_min == t_max:
    t_min = t_min - timedelta(minutes=3)
    t_max = t_max + timedelta(minutes=3)

# Default: latest non-zero on this date if available, else mid time
if auto_dt.date() == selected_date:
    default_dt = min(max(auto_dt, t_min), t_max)
else:
    mid_idx = len(day_times) // 2
    default_dt = min(max(pd.Timestamp(day_times[mid_idx]).to_pydatetime(), t_min), t_max)

with c2:
    selected_dt = st.slider(
        "⏰ Pick a time (single)",
        min_value=t_min,
        max_value=t_max,
        value=default_dt,
        step=timedelta(minutes=3),
        format="HH:mm:ss",
        help="Counts are summed within the ± window around the selected time."
    )

st.caption(f"Selected window: {selected_dt:%Y-%m-%d %H:%M} ± {window_minutes} minutes")

# ---- Aggregate + join ----
flow_agg   = agg_window(flow_long, selected_dt, window_minutes)
bubbles_df = sensors.merge(flow_agg, on="join_key", how="left")
bubbles_df["count"] = bubbles_df["value_sum"].fillna(0).astype(int)

# ---- Map ----
with tab_map:
    # ---- Map ----
    m = make_base_map(sensors)
    if viz_mode in ("Heatmap", "Both"):
        add_heatmap(m, bubbles_df, radius_px=heat_radius_px)
    if viz_mode in ("Bubbles", "Both"):
        add_bubbles(m, bubbles_df, selected_dt, window_minutes)

    st.components.v1.html(m.get_root().render(), height=650)

    # ---- KPIs ----
    total_people      = int(bubbles_df["count"].sum())
    sensors_with_data = int((bubbles_df["count"] > 0).sum())
    k1, k2, k3 = st.columns(3)
    k1.metric("📍 Sensors plotted", f"{len(sensors)}")
    k2.metric("📊 Sensors w/ data", f"{sensors_with_data}")
    k3.metric("👥 Total people (window)", f"{total_people}")

# ============================================================
# SECOND PAGE (Sensor Details) – added below existing map code
# ============================================================

st.markdown("---")  # divider
st.header("📈 Sensor Details")

# Tabs for the two pages (Map + Details)
tab_map, tab_detail = st.tabs(["🗺️ Map", "📈 Sensor Details"])

with tab_map:
    # keep your existing map and KPI content inside this block
    m = make_base_map(sensors)
    if viz_mode in ("Heatmap", "Both"):
        add_heatmap(m, bubbles_df, radius_px=heat_radius_px)
    if viz_mode in ("Bubbles", "Both"):
        add_bubbles(m, bubbles_df, selected_dt, window_minutes)
    st.components.v1.html(m.get_root().render(), height=650)

    total_people      = int(bubbles_df["count"].sum())
    sensors_with_data = int((bubbles_df["count"] > 0).sum())
    k1, k2, k3 = st.columns(3)
    k1.metric("📍 Sensors plotted", f"{len(sensors)}")
    k2.metric("📊 Sensors w/ data", f"{sensors_with_data}")
    k3.metric("👥 Total people (window)", f"{total_people}")

with tab_detail:
    import plotly.express as px

    st.subheader("Trend by Location")

    # Use location names from your 'sensors' DataFrame
    location = st.selectbox(
        "Choose location",
        options=sensors["location_name"].unique(),
        index=0,
        help="Select a location to see its sensor trend"
    )

    # Match the selected location’s sensor join_keys
    loc_keys = sensors.loc[sensors["location_name"] == location, "join_key"].tolist()
    detail_df = flow_long[flow_long["join_key"].isin(loc_keys)].copy()

    if detail_df.empty:
        st.warning("No data found for this location.")
    else:
        # Summarize counts by timestamp
        detail_agg = (
            detail_df.groupby("_t", as_index=False)["value"].sum()
            .sort_values("_t")
        )

        # Line chart
        fig = px.line(
            detail_agg,
            x="_t",
            y="value",
            title=f"{location} — People over Time",
            labels={"_t": "Time", "value": "Flow Count"},
        )
        fig.update_layout(height=450, margin=dict(l=10, r=10, b=10, t=50))
        st.plotly_chart(fig, use_container_width=True)

        # small summary
        st.caption(
            f"Showing sensor data for **{location}** "
            f"from {detail_agg['_t'].min():%Y-%m-%d %H:%M} "
            f"to {detail_agg['_t'].max():%Y-%m-%d %H:%M}"
        )
