# app.py  — your original map app + a safe extra page (Sensor Details)

import re
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import plotly.express as px
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

# ---------------- SIMPLE PAGE ROUTER (ADD) ----------------
# This does NOT change your logic; it just lets the user pick which page to view.
if "page" not in st.session_state:
    st.session_state.page = "map"
# ---------------------------------------------------------

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

with st.sidebar:
    st.header("📁 Files")
    st.text(f"Sensors: {SENSORS_XLSX.name}")
    st.text(f"Flow: {FLOW_CSV.name}")
    viz_mode       = st.radio("Visualization", ["Bubbles", "Heatmap", "Both"], index=0)
    heat_radius_px = st.slider("Heatmap radius (px)", 10, 100, 48, 2)
    window_minutes = st.slider("± minutes around time (smoothing)", 0, 60, DEFAULT_WINDOW_MIN, 1)

    st.markdown("---")
    page_choice = st.radio(
    "Page",
    ["🗺️ Map", "📈 Sensor Details", "▶️ Time-lapse"],
    index={"map":0, "details":1, "timelapse":2}.get(st.session_state.page, 0)
)

st.session_state.page = (
    "map" if page_choice.startswith("🗺️")
    else "details" if page_choice.startswith("📈")
    else "timelapse"
)


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

    use_whole_event = st.checkbox(
        "Show whole event (all dates)",
        value=False,
        help="Use all data from all days instead of a single date."
    )

# ---- Time slider (time-only label; values are naive datetimes) ----
if use_whole_event:
    day_mask = flow_long["_t"].notna()  
else:
    day_mask = flow_long["_t"].dt.date == selected_date

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

# ---- Time range slider (replaces the single time slider) ----
# Use your existing default_dt to build a sensible default range
range_start_default = max(t_min, default_dt - timedelta(minutes=window_minutes))
range_end_default   = min(t_max, default_dt + timedelta(minutes=window_minutes))

with c2:
    selected_start, selected_end = st.slider(
        "⏰ Pick a time range",
        min_value=t_min,
        max_value=t_max,
        value=(range_start_default, range_end_default),
        step=timedelta(minutes=3),
        format="HH:mm:ss",
        help="Choose start and end times. Counts are summed across this range."
    )

# Convert the chosen range to a midpoint + half-width so existing code keeps working
selected_dt = selected_start + (selected_end - selected_start) / 2
window_minutes = int(((selected_end - selected_start).total_seconds() / 60) / 2)

# ---- Caption + optional WHOLE-EVENT override (put this before Aggregate + join) ----
if use_whole_event:
    # Use the full dataset span
    selected_start = pd.Timestamp(flow_long["_t"].min()).to_pydatetime()
    selected_end   = pd.Timestamp(flow_long["_t"].max()).to_pydatetime()

    # Recompute midpoint + half-width so downstream code (agg_window, tooltips) still works
    selected_dt = selected_start + (selected_end - selected_start) / 2
    window_minutes = int(((selected_end - selected_start).total_seconds() / 60) / 2)

    st.caption(
        f"Whole event: {selected_start:%Y-%m-%d %H:%M} → "
        f"{selected_end:%H:%M} (midpoint {selected_dt:%H:%M}, ±{window_minutes} min)"
    )
else:
    st.caption(
        f"Selected range: {selected_start:%Y-%m-%d %H:%M} → "
        f"{selected_end:%H:%M} (midpoint {selected_dt:%H:%M}, ±{window_minutes} min)"
    )


# ---- Aggregate + join ----
flow_agg   = agg_window(flow_long, selected_dt, window_minutes)
bubbles_df = sensors.merge(flow_agg, on="join_key", how="left")
bubbles_df["count"] = bubbles_df["value_sum"].fillna(0).astype(int)

# ===================== SENSOR DETAILS PAGE ======================
# If user picked the details page, render it now and stop before map.
if st.session_state.page == "details":

    st.header("📈 Sensor Details")
    st.subheader("Trend by Location")

    if "location_name" not in sensors.columns:
        st.error("Column 'location_name' not found in sensors dataframe.")
        st.stop()

    locations = sensors["location_name"].dropna().sort_values().unique().tolist()
    if not locations:
        st.warning("No locations available in sensors metadata.")
        st.stop()

    location = st.selectbox(
        "Choose location",
        options=locations,
        index=0,
        help="This list comes from the sensors Excel file (Locatienaam)."
    )

    # Sensor(s) for that location → join_keys
    loc_keys = sensors.loc[sensors["location_name"] == location, "join_key"].tolist()

    # Pull time series from already-loaded flow_long
    detail_df = flow_long.loc[flow_long["join_key"].isin(loc_keys), ["_t", "value"]].copy()
    if detail_df.empty:
        st.warning("No data found for this location in the flow file.")
        st.stop()

    # Filter by current time range (if available)
    try:
        _start, _end = selected_start, selected_end
    except NameError:
        _start, _end = detail_df["_t"].min(), detail_df["_t"].max()

    detail_df = detail_df[(detail_df["_t"] >= _start) & (detail_df["_t"] <= _end)]

    # Aggregate by timestamp (sum if multiple sensors share a location)
    detail_agg = (
        detail_df.groupby("_t", as_index=False)["value"].sum()
                 .sort_values("_t")
    )

    # ---- KPI: Now vs 24h avg ----
    now_val = float(detail_agg["value"].iloc[-1]) if not detail_agg.empty else 0.0
    _24h_start = _end - pd.Timedelta(hours=24)
    df24 = flow_long.loc[
        (flow_long["join_key"].isin(loc_keys)) &
        (flow_long["_t"] >= _24h_start) & (flow_long["_t"] <= _end),
        ["_t","value"]
    ]
    avg24 = float(df24["value"].mean()) if not df24.empty else 0.0

    # ---- Plot (Plotly or fallback to Altair) ----
    try:
        import plotly.express as px
        fig = px.line(
            detail_agg,
            x="_t",
            y="value",
            labels={"_t": "Time", "value": "Flow Count"},
            title=f"{location} — People over Time"
        )
        fig.update_layout(height=450, margin=dict(l=10, r=10, b=10, t=50))
        st.plotly_chart(fig, use_container_width=True)
    except ModuleNotFoundError:
        import altair as alt
        chart = (
            alt.Chart(detail_agg)
               .mark_line()
               .encode(
                   x=alt.X("_t:T", title="Time"),
                   y=alt.Y("value:Q", title="Flow Count"),
                   tooltip=["_t:T", "value:Q"]
               )
               .properties(title=f"{location} — People over Time", height=450)
        )
        st.altair_chart(chart, use_container_width=True)

    # ---- KPI box below chart ----
    import streamlit.components.v1 as components
    delta_val = now_val - avg24
    delta_color = "#2ECC71" if delta_val >= 0 else "#E74C3C"
    kpi_html = f"""
    <div style="display:flex;flex-direction:column;align-items:flex-start;margin-top:1.2rem;">
      <div style="font-size:2.2rem;font-weight:600;color:white;">{now_val:,.2f}</div>
      <div style="color:{delta_color};background-color:{delta_color}20;
                  padding:.25rem .6rem;border-radius:8px;font-size:.9rem;
                  font-weight:500;margin-top:.3rem;">
        {'▲' if delta_val>=0 else '▼'} {delta_val:,.2f} vs 24 h avg
      </div>
    </div>
    """
    components.html(kpi_html, height=100)

    st.caption(
        f"Showing data for **{location}** "
        f"from {_start:%Y-%m-%d %H:%M} to {_end:%H:%M} "
        f"(points: {len(detail_agg):,})"
    )

    st.stop()  # make sure the map below does not run on this page
# ====================================================================

# ===================== TIME-LAPSE PAGE (Folium, same look as Map) ======================
if st.session_state.page == "timelapse":
    import streamlit as st
    import pandas as pd
    from folium.plugins import HeatMapWithTime, TimestampedGeoJson

    st.header("▶️ Time-lapse — People pattern over time (Folium)")

    # Controls for the timelapse page
    viz_choice = st.radio(
        "Animation style",
        ["Bubbles (like Map)", "Heatmap (like Map)"],
        horizontal=True
    )
    use_current_range = st.checkbox(
        "Use current time range only",
        value=False,
        help="If off, animates the full selected date."
    )

    # Build the animation window
    if use_current_range:
        ani_start, ani_end = selected_start, selected_end
    else:
        # animate the full selected day (clipped to data availability)
        mask_day = (flow_long["_t"].dt.date == selected_date)
        if mask_day.any():
            day_times = flow_long.loc[mask_day, "_t"].sort_values()
            ani_start, ani_end = day_times.min(), day_times.max()
        else:
            ani_start, ani_end = selected_start, selected_end

    # Prepare per-frame data: sum per sensor per timestamp, then join coords
    df_frames = (
        flow_long.loc[
            (flow_long["_t"] >= ani_start) & (flow_long["_t"] <= ani_end),
            ["_t", "join_key", "value"]
        ]
        .groupby(["_t", "join_key"], as_index=False)["value"].sum()
        .merge(sensors[["join_key", "location_name", "_lat", "_lon"]],
               on="join_key", how="inner")
        .sort_values("_t")
    )

    if df_frames.empty:
        st.warning("No data available for the selected period.")
        st.stop()

    # Create a base Folium map identical to your Map page
    m = make_base_map(sensors)

    # A tidy list of frame labels (e.g., '08:15')
    frame_times = df_frames["_t"].drop_duplicates().tolist()
    frame_labels = [t.strftime("%H:%M") for t in frame_times]

    if viz_choice.startswith("Heatmap"):
        # -------- HeatmapWithTime (same look/feel as your static heatmap) --------
        frames = []
        for t in frame_times:
            dt = df_frames.loc[df_frames["_t"] == t]
            # same triple format used by your static heatmap: [lat, lon, weight]
            frames.append([[float(r["_lat"]), float(r["_lon"]), float(r["value"])]
                           for _, r in dt.iterrows() if r["value"] > 0])

        HeatMapWithTime(
            frames,
            index=frame_labels,
            radius=heat_radius_px,
            auto_play=False,
            max_opacity=0.9,
            use_local_extrema=False
        ).add_to(m)

    else:
        # -------- TimestampedGeoJson (animated bubbles like your Map) --------
        # Keep bubble size scale consistent across frames
        vmin = float(df_frames["value"].min())
        vmax = float(df_frames["value"].max())
        # avoid zero-divide; make a gentle scale similar to your add_bubbles()
        def _radius_from_value(v: float) -> int:
            if vmax == vmin:
                return 18
            return int(18 + 36 * (v - vmin) / max(1.0, (vmax - vmin)))

        features = []
        for t in frame_times:
            dt = df_frames.loc[df_frames["_t"] == t]
            for _, r in dt.iterrows():
                val = float(r["value"])
                color = _bubble_color(val)  # reuse your color bucketing
                rad = _radius_from_value(val)
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(r["_lon"]), float(r["_lat"])]
                    },
                    "properties": {
                        "time": t.isoformat(),
                        "popup": f"{r['location_name']}<br><b>Count:</b> {int(val)}<br><b>Time:</b> {t:%Y-%m-%d %H:%M}",
                        "style": {
                            "color": color,
                            "fillColor": color,
                            "fillOpacity": 0.85,
                            "opacity": 0.95
                        },
                        "icon": "circle",
                        "iconstyle": {
                            "fillColor": color,
                            "fillOpacity": 0.85,
                            "stroke": True,
                            "radius": rad
                        }
                    }
                })

        TimestampedGeoJson(
            {
                "type": "FeatureCollection",
                "features": features
            },
            transition_time=200,   # ms between frames
            add_last_point=False,
            auto_play=False,
            loop=False,
            period="PT3M"          # your data are 3-minute steps; adjust if needed
        ).add_to(m)

    st.caption(
        f"Animating {ani_start:%Y-%m-%d %H:%M} → {ani_end:%H:%M} "
        f"({len(frame_times)} frames)"
    )

    # Render the Leaflet map in Streamlit
    st.components.v1.html(m.get_root().render(), height=650)

    # Optional: download the animation frame data
    with st.expander("Export frame data (CSV)"):
        st.download_button(
            "Download CSV",
            data=df_frames.to_csv(index=False).encode("utf-8"),
            file_name=f"timelapse_{selected_date:%Y%m%d}.csv",
            mime="text/csv"
        )

    st.stop()
# ====================================================================

# ====================================================================

# ---- Map (UNCHANGED) ----
m = make_base_map(sensors)
if viz_mode in ("Heatmap", "Both"):
    add_heatmap(m, bubbles_df, radius_px=heat_radius_px)
if viz_mode in ("Bubbles", "Both"):
    add_bubbles(m, bubbles_df, selected_dt, window_minutes)

st.components.v1.html(m.get_root().render(), height=650)

# ---- KPIs (UNCHANGED) ----
total_people      = int(bubbles_df["count"].sum())
sensors_with_data = int((bubbles_df["count"] > 0).sum())
k1, k2, k3 = st.columns(3)
k1.metric("📍 Sensors plotted", f"{len(sensors)}")
k2.metric("📊 Sensors w/ data", f"{sensors_with_data}")
k3.metric("👥 Total people (window)", f"{total_people}")
