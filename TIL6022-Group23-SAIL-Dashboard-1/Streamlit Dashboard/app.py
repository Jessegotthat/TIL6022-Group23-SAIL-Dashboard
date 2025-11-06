import re
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import folium
from folium.plugins import HeatMap

try:
    from streamlit_folium import st_folium
    _USE_ST_FOLIUM = True
except Exception:
    _USE_ST_FOLIUM = False


# ============================================================
# MAIN FUNCTION
# ============================================================
def run():
    # ---------------- CONFIG ----------------
    BASE_DIR = Path(__file__).parent
    SENSORS_XLSX = BASE_DIR / "Sensor Location Data.xlsx"
    FLOW_CSV = BASE_DIR / "SAIL2025_LVMA_data_3min_20August-25August2025_flow.csv"
    CITY_CENTER = [52.377956, 4.897070]  # Amsterdam fallback center
    DEFAULT_WINDOW_MIN = 15

    # ---------------- SIMPLE PAGE ROUTER ----------------
    if "page" not in st.session_state:
        st.session_state.page = "map"

    # ============================================================
    # HELPERS
    # ============================================================
    def _norm(s: str) -> str:
        s = str(s).strip().lower()
        s = re.sub(r'\.\d+$', '', s)
        s = re.sub(r'[-_ ]+[a-z]$', '', s)
        s = re.sub(r'[^a-z0-9]', '', s)
        return s

    # ---------------- LOADERS ----------------
    @st.cache_data(show_spinner=False)
    def load_sensors(path: Path) -> pd.DataFrame:
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
        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception:
            df = pd.read_csv(path)

        def find_col(cands):
            for c in df.columns:
                if c.strip().lower() in cands:
                    return c
            return None

        date_col = find_col({"timestamp", "date", "datum"})
        time_col = find_col({"time", "tijd"})

        if date_col is None or time_col is None:
            one_dt_col = find_col({"datetime", "dt", "_t"})
            if one_dt_col is not None:
                dt = pd.to_datetime(df[one_dt_col], errors="coerce", dayfirst=True)
            else:
                dt = pd.date_range("2025-08-20 00:00:00", periods=len(df), freq="3min")
            df["_t"] = dt
        else:
            d = df[date_col].astype(str).str.strip()
            t = df[time_col].astype(str).str.strip().str.replace(r"\+.*", "", regex=True)
            dt = pd.to_datetime(d + " " + t, errors="coerce", dayfirst=True)
            if pd.api.types.is_datetime64tz_dtype(dt):
                dt = dt.dt.tz_localize(None)
            df["_t"] = dt

        df = df[df["_t"].notna()].copy()
        id_vars = ["_t"]
        drop_cols = {c for c in [date_col, time_col] if c is not None}
        value_vars = [c for c in df.columns if c not in id_vars and c not in drop_cols]
        long_df = df.melt(id_vars=id_vars, value_vars=value_vars, var_name="code", value_name="value")

        long_df["value"] = (
            long_df["value"].astype(str)
            .str.replace("\xa0", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce").fillna(0)
        long_df["join_key"] = long_df["code"].apply(_norm)
        return long_df


    # ============================================================
    # LATEST NONZERO DATETIME
    # ============================================================
    def latest_nonzero_dt(long_df: pd.DataFrame) -> datetime:
        if long_df.empty or "_t" not in long_df.columns:
            return datetime.now()

        ts = long_df["_t"]
        if pd.api.types.is_datetime64tz_dtype(ts):
            ts = ts.dt.tz_localize(None)

        totals = long_df.groupby(ts, as_index=False)["value"].sum()
        nz = totals.loc[totals["value"] > 0]
        if not nz.empty:
            return pd.Timestamp(nz.iloc[-1][ts.name]).to_pydatetime()
        return pd.Timestamp(long_df["_t"].min()).to_pydatetime()

    # ============================================================
    # MAP CREATION + HELPERS
    # ============================================================
    def make_base_map(sensors_df: pd.DataFrame) -> folium.Map:
        center = [sensors_df["_lat"].mean(), sensors_df["_lon"].mean()] if not sensors_df.empty else CITY_CENTER
        return folium.Map(location=center, zoom_start=13, tiles="cartodbpositron")

    def _bubble_color(count: float) -> str:
        if count >= 200: return "#E74C3C"
        if count >= 80: return "#F1C40F"
        if count > 0: return "#7DCEA0"
        return "#95A5A6"

    def add_heatmap(m: folium.Map, df: pd.DataFrame, radius_px: int) -> None:
        pts = [[float(r["_lat"]), float(r["_lon"]), float(r["count"])] for _, r in df.iterrows() if r["count"] > 0]
        if pts:
            HeatMap(pts, radius=radius_px, blur=int(radius_px * 0.6), max_zoom=16).add_to(m)

    # ============================================================
    # MAIN UI (all your existing Streamlit code)
    # ============================================================
    st.title("🌊 SAIL Sensors — Per-Sensor Counts & Heatmap")

    with st.sidebar:
        st.header("📁 Files")
        st.text(f"Sensors: {SENSORS_XLSX.name}")
        st.text(f"Flow: {FLOW_CSV.name}")
        viz_mode = st.radio("Visualization", ["Bubbles", "Heatmap", "Both"], index=0)
        heat_radius_px = st.slider("Heatmap radius (px)", 10, 100, 48, 2)
        window_minutes = st.slider("± minutes around time (smoothing)", 0, 60, DEFAULT_WINDOW_MIN, 1)
        st.markdown("---")
        page_choice = st.radio("Page", ["🗺️ Map", "📈 Sensor Details", "▶️ Time-lapse"], index=0)

    st.session_state.page = (
        "map" if page_choice.startswith("🗺️")
        else "details" if page_choice.startswith("📈")
        else "timelapse"
    )

    # ---------------- LOAD DATA ----------------
    try:
        sensors = load_sensors(SENSORS_XLSX)
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
    st.session_state["use_whole_event"] = use_whole_event

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

# ---- Time range slider ----
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

# Convert range to midpoint + half-width so existing code keeps working
selected_dt = selected_start + (selected_end - selected_start) / 2
window_minutes = int(((selected_end - selected_start).total_seconds() / 60) / 2)

# ---- Caption + optional WHOLE-EVENT override ----
if use_whole_event:
    selected_start = pd.Timestamp(flow_long["_t"].min()).to_pydatetime()
    selected_end   = pd.Timestamp(flow_long["_t"].max()).to_pydatetime()

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

def agg_window(long_df: pd.DataFrame, selected_dt: datetime, window_minutes: int) -> pd.DataFrame:
    """
    Sum values per sensor (join_key) within ±window_minutes around selected_dt.
    Returns columns: join_key, value_sum
    """
    start = selected_dt - timedelta(minutes=window_minutes)
    end   = selected_dt + timedelta(minutes=window_minutes)

    # ensure timezone-naive comparison
    ts = long_df["_t"]
    if pd.api.types.is_datetime64tz_dtype(ts):
        long_df = long_df.copy()
        long_df["_t"] = ts.dt.tz_localize(None)
        ts = long_df["_t"]

    sub = long_df.loc[(ts >= start) & (ts <= end), ["join_key", "value"]]
    if sub.empty:
        return pd.DataFrame({"join_key": [], "value_sum": []})

    out = (
        sub.groupby("join_key", as_index=False)["value"]
           .sum()
           .rename(columns={"value": "value_sum"})
    )
    return out

# ---- Aggregate + join ----
flow_agg   = agg_window(flow_long, selected_dt, window_minutes)
bubbles_df = sensors.merge(flow_agg, on="join_key", how="left")
bubbles_df["count"] = bubbles_df["value_sum"].fillna(0).astype(int)

# ===================== SENSOR DETAILS PAGE ======================
if st.session_state.page == "details":

    st.header("📈 Sensor Details")
    st.subheader("Trend by Location")

    use_whole_event = st.session_state.get("use_whole_event", False)

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

    loc_keys = sensors.loc[sensors["location_name"] == location, "join_key"].tolist()
    detail_df = flow_long.loc[flow_long["join_key"].isin(loc_keys), ["_t", "value"]].copy()
    if detail_df.empty:
        st.warning("No data found for this location in the flow file.")
        st.stop()

    if use_whole_event:
        _start, _end = flow_long["_t"].min(), flow_long["_t"].max()
    else:
        _start, _end = selected_start, selected_end

    detail_df = detail_df[(detail_df["_t"] >= _start) & (detail_df["_t"] <= _end)]
    detail_agg = (
        detail_df.groupby("_t", as_index=False)["value"].sum()
                 .sort_values("_t")
    )

    now_val = float(detail_agg["value"].iloc[-1]) if not detail_agg.empty else 0.0
    _24h_start = _end - pd.Timedelta(hours=24)
    df24 = flow_long.loc[
        (flow_long["join_key"].isin(loc_keys)) &
        (flow_long["_t"] >= _24h_start) & (flow_long["_t"] <= _end),
        ["_t","value"]
    ]
    avg24 = float(df24["value"].mean()) if not df24.empty else 0.0

    try:
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

    st.stop()  # stop before map for this page

# ===================== TIME-LAPSE PAGE ======================
if st.session_state.page == "timelapse":
    from folium.plugins import HeatMapWithTime, TimestampedGeoJson

    st.header("▶️ Time-lapse — People pattern over time (Folium)")

    # Use the same visualization mode as sidebar (Map page)
    timelapse_mode = "Heatmap" if viz_mode in ("Heatmap", "Both") else "Bubbles"

    use_current_range = st.checkbox(
        "Use current time range only",
        value=False,
        help="If off, animates the full selected date."
    )

    # Build the animation window
    if use_current_range:
        ani_start, ani_end = selected_start, selected_end
    else:
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

    m = make_base_map(sensors)
    frame_times  = df_frames["_t"].drop_duplicates().tolist()
    frame_labels = [t.strftime("%H:%M") for t in frame_times]

    if timelapse_mode.startswith("Heatmap"):
        # Build frames with GLOBAL scaling so colors match Map page behavior
        frames = build_heatmap_frames_global(flow_long, sensors, frame_times)

        gradient = {
            0.00: '#0000ff',  # blue
            0.25: '#33a3ff',
            0.50: '#33ff57',  # green
            0.75: '#ffd43b',  # yellow
            1.00: '#ff3b30',  # red
        }

        HeatMapWithTime(
            frames,
            index=[t.strftime("%H:%M") for t in frame_times],
            radius=heat_radius_px,
            auto_play=False,
            max_opacity=0.9,
            use_local_extrema=False,
            gradient=gradient,
        ).add_to(m)

    else:
        # -------- Animated bubbles (TimestampedGeoJson) --------
        vmin = float(df_frames["value"].min())
        vmax = float(df_frames["value"].max())

        def _radius_from_value(v: float) -> int:
            if vmax == vmin:
                return 18
            return int(18 + 36 * (v - vmin) / max(1.0, (vmax - vmin)))

        features = []
        for t in frame_times:
            dt = df_frames.loc[df_frames["_t"] == t]
            for _, r in dt.iterrows():
                val = float(r["value"])
                color = _bubble_color(val)
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
            transition_time=200,
            add_last_point=False,
            auto_play=False,
            loop=False,
            period="PT3M"  # your data are 3-minute steps
        ).add_to(m)

    st.caption(
        f"Animating {ani_start:%Y-%m-%d %H:%M} → {ani_end:%H:%M} "
        f"({len(frame_times)} frames)"
    )
    st.components.v1.html(m.get_root().render(), height=650)
    st.stop()

# =================== MAP PAGE: MAP + TREND SIDE BY SIDE ===================
left_col, right_col = st.columns([3, 2], gap="large")

# ---------------- LEFT: MAP ----------------
with left_col:
    m = make_base_map(sensors)

    if viz_mode in ("Heatmap", "Both"):
        add_heatmap(m, bubbles_df, radius_px=heat_radius_px)

    if viz_mode in ("Bubbles", "Both"):
        add_bubbles(m, bubbles_df, selected_dt, window_minutes)

    if _USE_ST_FOLIUM:
        map_state = st_folium(m, height=650, width=None)

        clicked_name = None
        if isinstance(map_state, dict):
            pop = map_state.get("last_object_clicked_popup")
            if isinstance(pop, dict):
                clicked_name = pop.get("content")
            elif isinstance(pop, str):
                clicked_name = pop

        if clicked_name and st.session_state.get("clicked_location") != clicked_name:
            st.session_state["clicked_location"] = clicked_name
            st.rerun()
    else:
        st.components.v1.html(m.get_root().render(), height=650)

# ---------------- RIGHT: TREND ----------------
with right_col:
    st.subheader("Trend by Location")

    use_whole_event = st.session_state.get("use_whole_event", False)

    locations = sensors["location_name"].dropna().sort_values().unique().tolist()
    if not locations:
        st.warning("No locations available in sensors metadata.")
    else:
        clicked_loc = st.session_state.get("clicked_location")
        if clicked_loc in locations:
            location = clicked_loc
            st.caption(f"📍 Selected from map: **{location}**")
        else:
            if not bubbles_df.empty:
                location = bubbles_df.sort_values("count", ascending=False)["location_name"].iloc[0]
            else:
                location = locations[0]
            st.caption("Tip: click a bubble on the map to choose a location.")

        if use_whole_event:
            _start, _end = flow_long["_t"].min(), flow_long["_t"].max()
        else:
            _start, _end = selected_start, selected_end

        loc_keys = sensors.loc[sensors["location_name"] == location, "join_key"].tolist()

        detail_df = flow_long.loc[
            (flow_long["join_key"].isin(loc_keys)) &
            (flow_long["_t"] >= _start) & (flow_long["_t"] <= _end),
            ["_t", "value"]
        ].copy()

        if detail_df.empty:
            st.info("No data found for this location in the selected period.")
        else:
            detail_agg = (
                detail_df.groupby("_t", as_index=False)["value"]
                         .sum()
                         .sort_values("_t")
            )

            now_val = float(detail_agg["value"].iloc[-1])
            _24h_start = _end - pd.Timedelta(hours=24)
            df24 = flow_long.loc[
                (flow_long["join_key"].isin(loc_keys)) &
                (flow_long["_t"] >= _24h_start) & (flow_long["_t"] <= _end),
                ["_t", "value"]
            ]
            avg24 = float(df24["value"].mean()) if not df24.empty else 0.0

            try:
                fig = px.line(
                    detail_agg, x="_t", y="value",
                    labels={"_t": "Time", "value": "Flow Count"},
                    title=f"{location} — People over Time"
                )
                fig.update_layout(height=420, margin=dict(l=10, r=10, b=10, t=50))
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
                       .properties(title=f"{location} — People over Time", height=420)
                )
                st.altair_chart(chart, use_container_width=True)

            import streamlit.components.v1 as components
            delta_val = now_val - avg24
            delta_color = "#2ECC71" if delta_val >= 0 else "#E74C3C"
            kpi_html = f"""
            <div style="display:flex;flex-direction:column;align-items:flex-start;margin-top:1rem;">
              <div style="font-size:2.0rem;font-weight:600;color:white;">{now_val:,.2f}</div>
              <div style="color:{delta_color};background-color:{delta_color}20;
                          padding:.25rem .6rem;border-radius:8px;font-size:.9rem;margin-top:.3rem;">
                {'▲' if delta_val>=0 else '▼'} {delta_val:,.2f} vs 24 h avg
              </div>
            </div>
            """
            components.html(kpi_html, height=90)

            if use_whole_event:
                st.caption(
                    f"{location} • whole event "
                    f"{_start:%Y-%m-%d %H:%M} → {_end:%Y-%m-%d %H:%M} "
                    f"(points: {len(detail_agg):,})"
                )
            else:
                st.caption(
                    f"{location} • range {_start:%Y-%m-%d %H:%M} → {_end:%H:%M} "
                    f"(points: {len(detail_agg):,})"
                )

# ---- KPIs ----
total_people      = int(bubbles_df["count"].sum())
sensors_with_data = int((bubbles_df["count"] > 0).sum())
k1, k2, k3 = st.columns(3)
k1.metric("📍 Sensors plotted", f"{len(sensors)}")
k2.metric("📊 Sensors w/ data", f"{sensors_with_data}")
k3.metric("👥 Total people (window)", f"{total_people}")

if __name__ == "__main__":
    run()