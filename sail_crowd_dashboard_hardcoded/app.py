
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import altair as alt
import os, re
from datetime import timedelta
import os
# Use a free Carto/OSM basemap if no Mapbox token is available
MAP_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"

st.set_page_config(page_title="SAIL Crowd Flow (Hardcoded)", layout="wide")

st.title("SAIL 2025 • Crowd Flow Dashboard (Hardcoded to project files)")
st.caption("This build targets your actual files: wide flow CSV, Dutch Excel meta, Parquet TomTom/Vessel.")

# ===================================
# File locations (defaults)
# ===================================
DEFAULT_FLOW  = "inputs/SAIL2025_LVMA_data_3min_20August-25August2025_flow.csv"
DEFAULT_META  = "inputs/sensor-location.xlsx"
DEFAULT_TOMTOM = "data/20250820163000_stream.tomtom.analyze-sail.parquet"
DEFAULT_VESSEL = "data/20250820163000_stream.vessel-positions-anonymized-processed.analyze-sail.parquet"

# ===================================
# Helpers (schema-aware)
# ===================================
def parse_lat_lon(text):
    """Parse 'Lat/Long' field like '52.372634, 4.892071' -> (lat, lon)."""
    if pd.isna(text): return np.nan, np.nan
    s = str(text)
    m = re.findall(r"[-+]?\d+\.\d+", s)
    if len(m) >= 2:
        return float(m[0]), float(m[1])
    return np.nan, np.nan

def pick_width_col(cols):
    """Choose 'Effectieve  breedte' if present, else 'Breedte' (case/space-insensitive)."""
    norm = {c: re.sub(r"\s+", "", c.lower()) for c in cols}
    eff = [c for c in cols if "effectieve" in c.lower() and "breedte" in c.lower()]
    if eff: return eff[0]
    # fallback: exact 'Breedte' or closest
    for c in cols:
        if norm[c] == "breedte":
            return c
    # any column containing 'breedte'
    anyb = [c for c in cols if "breedte" in c.lower()]
    return anyb[0] if anyb else None

def load_meta_excel(path_or_buf):
    """Read Dutch meta Excel -> standardized ['sensor_base','name','lat','lon','width']."""
    meta_raw = pd.read_excel(path_or_buf)
    cols = meta_raw.columns.tolist()

    # Required base id
    id_col = None
    for c in cols:
        if "object" in c.lower():  # handles "Objectummer"
            id_col = c; break
    if id_col is None:
        st.error("Sensor meta: cannot find 'Objectummer' (base id) column.")
        st.stop()

    name_col = None
    for c in cols:
        if "locatie" in c.lower():
            name_col = c; break

    latlong_col = None
    for c in cols:
        if "lat/long" in c.lower() or ("lat" in c.lower() and "long" in c.lower()):
            latlong_col = c; break
    if latlong_col is None:
        st.error("Sensor meta: cannot find 'Lat/Long' column.")
        st.stop()

    width_col = pick_width_col(cols)
    if width_col is None:
        st.error("Sensor meta: cannot find 'Breedte' or 'Effectieve  breedte'.")
        st.stop()

    meta = pd.DataFrame({
        "sensor_base": meta_raw[id_col].astype(str).str.strip(),
        "name": meta_raw[name_col].astype(str).str.strip() if name_col else meta_raw[id_col].astype(str),
        "lat": meta_raw[latlong_col].apply(lambda x: parse_lat_lon(x)[0]),
        "lon": meta_raw[latlong_col].apply(lambda x: parse_lat_lon(x)[1]),
        "width": pd.to_numeric(meta_raw[width_col], errors="coerce")
    })
    return meta.dropna(subset=["lat","lon"])

SENSOR_COL_EXCLUDES = set(["hour","minute","day","month","weekday","is_weekend"])

def flow_wide_to_long(flow_wide: pd.DataFrame):
    """Input: wide CSV with 'timestamp' + many 'CMSA-XXX-YY_dir' columns.
       Output: long flow aggregated by base sensor id (sum over directions).
    """
    if "timestamp" not in flow_wide.columns:
        # try to find time-like column
        tcol = None
        for c in flow_wide.columns:
            if any(k in c.lower() for k in ["timestamp","time","datetime","date"]):
                tcol = c; break
        if tcol is None:
            st.error("Flow CSV: cannot find 'timestamp' column.")
            st.stop()
        flow_wide = flow_wide.rename(columns={tcol:"timestamp"})

    # Select sensor columns: CMSA-... plus suffix '_\d+'
    def is_sensor_col(c):
        if c in SENSOR_COL_EXCLUDES: return False
        if c == "timestamp": return False
        return bool(re.match(r"^CMSA-[A-Z0-9-]+_\d+$", c))
    sensor_cols = [c for c in flow_wide.columns if is_sensor_col(c)]
    if not sensor_cols:
        # fallback: any column that looks like CMSA-... with underscore
        sensor_cols = [c for c in flow_wide.columns if c not in SENSOR_COL_EXCLUDES and c!="timestamp" and "_" in c and c.startswith("CMSA-")]
    if not sensor_cols:
        st.error("Flow CSV: no sensor columns detected (expected like 'CMSA-..._0').")
        st.stop()

    df = flow_wide[["timestamp"] + sensor_cols].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    long = df.melt(id_vars="timestamp", var_name="sensor_dir", value_name="flow")
    long["sensor_base"] = long["sensor_dir"].str.split("_").str[0]
    # Sum flow across directions per sensor_base per timestamp
    agg = long.groupby(["timestamp","sensor_base"], as_index=False)["flow"].sum(min_count=1)
    return agg

def infer_native_freq(df):
    df = df.sort_values("timestamp")
    try:
        f = pd.infer_freq(df["timestamp"].iloc[:200])
        if f: return f
    except Exception:
        pass
    diffs = df["timestamp"].diff().dropna()
    return str(diffs.mode().iloc[0]) if not diffs.empty else "3min"

def build_snapshot(recent, meta):
    """Latest flow per sensor with occupancy labels."""
    idx = recent.groupby("sensor_base")["timestamp"].idxmax()
    current = recent.loc[idx, ["sensor_base","timestamp","flow"]].merge(meta, on="sensor_base", how="left")
    q1, q2 = current["flow"].quantile([0.33, 0.66])
    def occ(v): return "low" if v<q1 else ("medium" if v<q2 else "high")
    current["occupancy"] = current["flow"].apply(occ)
    return current

def to_naive_ts(s: pd.Series) -> pd.Series:
    """
    Parse to datetime and strip timezone, so result is naive datetime64[ns].
    We parse with utc=True to be robust, then drop tz info.
    """
    s = pd.to_datetime(s, errors="coerce", utc=True)
    # remove timezone to make it naive (no tz)
    return s.dt.tz_convert(None)

# ===================================
# Sidebar (uploads override defaults)
# ===================================
st.sidebar.header("Data sources (hardcoded defaults below)")
flow_up = st.sidebar.file_uploader("Flow time series (CSV, wide)", type=["csv"])
meta_up = st.sidebar.file_uploader("Sensor meta (Excel)", type=["xlsx","xls"])

st.sidebar.subheader("TomTom (Parquet)")
tomtom_up = st.sidebar.file_uploader("Upload TomTom Parquet", type=["parquet"])

st.sidebar.subheader("Vessel (Parquet)")
vessel_up = st.sidebar.file_uploader("Upload Vessel Parquet", type=["parquet"])

history_hours = st.sidebar.slider("History window (hours)", 1, 168, 24)
pred_steps = st.sidebar.slider("Prediction horizon (steps)", 4, 24, 8)

# ===================================
# Load core data (no sample)
# ===================================
# Sensor meta
meta_path = meta_up if meta_up is not None else DEFAULT_META
if (not hasattr(meta_path, "read")) and (not os.path.exists(meta_path)):
    st.error(f"Sensor meta not found: {meta_path}. Please upload the Excel file.")
    st.stop()
meta = load_meta_excel(meta_path)

# Flow wide CSV
flow_path = flow_up if flow_up is not None else DEFAULT_FLOW
if (not hasattr(flow_path, "read")) and (not os.path.exists(flow_path)):
    st.error(f"Flow CSV not found: {flow_path}. Please upload the wide flow CSV.")
    st.stop()
flow_wide = pd.read_csv(flow_path)
flow_long = flow_wide_to_long(flow_wide)  # -> ['timestamp','sensor_base','flow']

flow_long["timestamp"] = to_naive_ts(flow_long["timestamp"])

# Join with meta (keep only known sensors)
flow_long = flow_long.merge(meta[["sensor_base","name","lat","lon","width"]], on="sensor_base", how="inner")

# Native frequency and recent slice
# --- robust native frequency detection across sensors ---
gdiffs = (
    flow_long.sort_values(["sensor_base", "timestamp"])
             .groupby("sensor_base")["timestamp"]
             .diff()
)
# keep only positive diffs to avoid 0-day artifacts
gdiffs = gdiffs[gdiffs.notna() & (gdiffs > pd.Timedelta(0))]

# final step (Timedelta)
step = gdiffs.mode().iloc[0] if not gdiffs.empty else pd.Timedelta(minutes=3)

# convert step -> pandas offset alias for .dt.floor()
def step_to_alias(td: pd.Timedelta) -> str:
    secs = int(td.total_seconds())
    if secs % 3600 == 0:
        return f"{secs // 3600}H"
    elif secs % 60 == 0:
        return f"{secs // 60}min"
    else:
        return f"{secs}S"

freq_alias = step_to_alias(step)

latest_time = flow_long["timestamp"].max()
min_time = latest_time - pd.Timedelta(hours=history_hours)
recent = flow_long[flow_long["timestamp"] >= min_time]

# Current snapshot
current = build_snapshot(recent, meta)

current["timestamp_str"] = current["timestamp"].dt.strftime("%Y-%m-%d %H:%M")

# ===================================
# Optional datasets: TomTom/Vessel (Parquet)
# ===================================
def read_parquet_any(path_or_buf):
    # Try fastparquet first, then pyarrow
    try:
        return pd.read_parquet(path_or_buf, engine="fastparquet")
    except Exception:
        return pd.read_parquet(path_or_buf)  # pyarrow

tt_path = tomtom_up if tomtom_up is not None else (DEFAULT_TOMTOM if os.path.exists(DEFAULT_TOMTOM) else None)
v_path  = vessel_up if vessel_up is not None else (DEFAULT_VESSEL if os.path.exists(DEFAULT_VESSEL) else None)

tt = None
if tt_path is not None:
    try:
        tt_raw = read_parquet_any(tt_path)
        # Normalize time column
        tcol = None
        for c in tt_raw.columns:
            if any(k in c.lower() for k in ["timestamp","time","datetime","date"]):
                tcol = c; break
        if tcol is None:
            st.warning("TomTom parquet: no timestamp-like column found.")
        else:
            tt = tt_raw.rename(columns={tcol:"timestamp"}).copy()
            tt["timestamp"] = pd.to_datetime(tt["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
            # downselect common variables
            tt.columns = [str(c).lower() for c in tt.columns]
            keep = ["timestamp"]
            for c in ["avg_speed_kph","tti","speed","travel_time_index"]:
                if c in tt.columns: keep.append(c)
            tt = tt[keep].groupby("timestamp").mean(numeric_only=True).reset_index()
    except Exception as e:
        st.warning(f"Failed to read TomTom parquet: {e}")

vessel = None
if v_path is not None:
    try:
        v_raw = read_parquet_any(v_path)
        # Normalize columns
        tcol = None
        for c in v_raw.columns:
            if any(k in c.lower() for k in ["timestamp","time","datetime","date"]):
                tcol = c; break
        lat_col = None; lon_col = None
        for c in v_raw.columns:
            cl = c.lower()
            if lat_col is None and ("lat" in cl or "latitude" in cl): lat_col = c
            if lon_col is None and (cl=="lon" or "lng" in cl or "long" in cl or "longitude" in cl): lon_col = c
        if tcol and lat_col and lon_col:
            vessel = v_raw.rename(columns={tcol:"timestamp", lat_col:"lat", lon_col:"lon"}).copy()
            vessel["timestamp"] = pd.to_datetime(vessel["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
            vessel = vessel.dropna(subset=["lat","lon"])
        else:
            st.warning("Vessel parquet: missing timestamp/lat/lon columns.")
    except Exception as e:
        st.warning(f"Failed to read Vessel parquet: {e}")

# ===================================
# Tabs
# ===================================
tab_map, tab_detail, tab_corr = st.tabs(["🗺️ Map", "📈 Sensor Detail", "🔗 Correlation Lab"])

# with tab_map:
#     layers = []
#     # Flow heatmap (recent)
#     heat = pdk.Layer(
#         "HeatmapLayer",
#         data=recent.rename(columns={"lat":"LAT","lon":"LON"}),
#         get_position=["LON","LAT"],
#         get_weight="flow",
#         radiusPixels=60,
#         aggregation="MEAN"
#     )
#     layers.append(heat)

#     # Sensor points
#     color_map = {"low":[0, 180, 0], "medium":[255,165,0], "high":[200,30,30]}
#     points = pdk.Layer(
#         "ScatterplotLayer",
#         data=current.assign(
#             color=current["occupancy"].map(color_map),
#             radius=(current["flow"] - current["flow"].min() + 1.0) * 1.0
#         ),
#         get_position=["lon","lat"],
#         get_radius="radius",
#         get_fill_color="color",
#         pickable=True,
#     )
#     layers.append(points)

#     # Vessel heatmap (if available)
#     if vessel is not None:
#         v_recent = vessel[vessel["timestamp"] >= latest_time - pd.Timedelta(hours=6)]
#         v_layer = pdk.Layer(
#             "HeatmapLayer",
#             data=v_recent,
#             get_position=["lon","lat"],
#             radiusPixels=40
#         )
#         layers.append(v_layer)

#     center_lat = float(current["lat"].mean())
#     center_lon = float(current["lon"].mean())
#     tooltip = {
#         "html": "<b>{sensor_base}</b><br/>Flow now: {flow} (units)<br/>Updated: {timestamp_str}<br/>Occupancy: {occupancy}",
#         "style": {"color": "white"}
#     }
#     deck = pdk.Deck(
#         layers=layers,
#         initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=13, pitch=40),
#         map_style=MAP_STYLE,
#         tooltip=tooltip
#     )
#     st.pydeck_chart(deck, use_container_width=True)

#     st.markdown("#### Current sensor snapshot")
#     st.dataframe(current[["sensor_base","name","flow","occupancy","timestamp","lat","lon","width"]]
#                  .sort_values("flow", ascending=False), use_container_width=True)

with tab_map:
    # ---------- prep ----------
    # ensure tooltip uses a readable time string
    current = current.copy()
    if "timestamp_str" not in current.columns:
        current["timestamp_str"] = current["timestamp"].dt.strftime("%Y-%m-%d %H:%M")

    # compress dynamic range for heatmap
    rec_with_geo = recent.copy()
    rec_with_geo["log_flow"] = np.log1p(rec_with_geo["flow"])

    layers = []

    # ---------- sensor points (smaller + alpha + pixels) ----------
    color_map_rgba = {
        "low":    [0,   180,  0, 160],
        "medium": [255, 165,  0, 160],
        "high":   [200,  30, 30, 160],
    }
    points = pdk.Layer(
        "ScatterplotLayer",
        data=current.assign(
            rgba=current["occupancy"].map(color_map_rgba),
            # normalized radius, clamped to sane pixel size
            radius=np.clip((current["flow"] - current["flow"].min()) * 6 + 4, 4, 24),
        ),
        get_position=["lon","lat"],
        get_radius="radius",
        radius_units="pixels",
        radius_min_pixels=3,
        radius_max_pixels=24,
        get_fill_color="rgba",
        pickable=True,
        opacity=0.7,
    )

    # ---------- flow heatmap (put on top so gradient is visible) ----------
    heat = pdk.Layer(
        "HeatmapLayer",
        data=rec_with_geo.rename(columns={"lat":"LAT","lon":"LON"}),
        get_position=["LON","LAT"],
        get_weight="log_flow",    # use log-compressed weight
        radiusPixels=70,          # try 25–40 if you want sharper/blurrier
        intensity=1,
        threshold=0.02,
        colorRange=[
            [33,102,172,0],
            [67,147,195,80],
            [146,197,222,120],
            [209,229,240,160],
            [253,219,199,200],
            [239,138,98,255],
        ],
        opacity=0.85,
    )

    # order: points first, then heatmap so gradient shows above points
    layers = [points, heat]

    # ---------- vessel heatmap (optional; appended on top) ----------
    if vessel is not None:
        v_recent = vessel[vessel["timestamp"] >= latest_time - pd.Timedelta(hours=6)]
        v_layer = pdk.Layer(
            "HeatmapLayer",
            data=v_recent,
            get_position=["lon","lat"],
            radiusPixels=40,
            opacity=0.6,
        )
        layers.append(v_layer)

    center_lat = float(current["lat"].mean())
    center_lon = float(current["lon"].mean())
    tooltip = {
        "html": "<b>{sensor_base}</b><br/>Flow now: {flow} (units)<br/>Updated: {timestamp_str}<br/>Occupancy: {occupancy}",
        "style": {"color": "white"}
    }
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=13, pitch=40),
        map_style=MAP_STYLE,
        tooltip=tooltip
    )
    st.pydeck_chart(deck, use_container_width=True)

    st.markdown("#### Current sensor snapshot")
    st.dataframe(
        current[["sensor_base","name","flow","occupancy","timestamp","lat","lon","width"]]
        .sort_values("flow", ascending=False),
        use_container_width=True
    )

with tab_detail:
    st.markdown("Recent history and baseline prediction (replace later with your model).")
    sensor_ids = sorted(flow_long["sensor_base"].unique().tolist())
    sid = st.selectbox("Sensor", options=sensor_ids, index=0)

    df_s = flow_long[flow_long["sensor_base"]==sid].sort_values("timestamp")
    recent_hist = df_s[df_s["timestamp"] >= latest_time - pd.Timedelta(hours=72)][["timestamp","flow"]]

    # Baseline: last-3-day mean per time-of-day
    pat_cut = df_s["timestamp"].max() - pd.Timedelta(days=3)
    df_s2 = df_s.copy()
    df_s2["tod"] = df_s2["timestamp"].dt.floor(freq_alias).dt.time
    pattern = df_s2[df_s2["timestamp"]>=pat_cut].groupby("tod")["flow"].mean().reset_index()

    future_times = [latest_time + step*(i+1) for i in range(pred_steps)]
    fut = pd.DataFrame({"timestamp": future_times})
    fut["tod"] = fut["timestamp"].dt.time
    fut = fut.merge(pattern, on="tod", how="left").rename(columns={"flow":"flow_pred"})
    if fut["flow_pred"].isna().any():
        last = recent_hist["flow"].iloc[-1] if len(recent_hist)>0 else 0.0
        fut["flow_pred"] = fut["flow_pred"].fillna(last)

    base_chart = alt.Chart(recent_hist).mark_line().encode(
        x="timestamp:T", y=alt.Y("flow:Q", title="Flow (as in CSV units)")
    ).properties(height=300)
    pred_chart = alt.Chart(fut).mark_line(point=True).encode(
        x="timestamp:T", y="flow_pred:Q", tooltip=["timestamp:T","flow_pred:Q"]
    )
    st.altair_chart(base_chart + pred_chart, use_container_width=True)

    if len(recent_hist)>0:
        now_val = recent_hist["flow"].iloc[-1]
        mean24 = df_s[df_s["timestamp"] >= latest_time - pd.Timedelta(hours=24)]["flow"].mean()
        st.metric("Now", f"{now_val:.2f}", delta=f"{now_val-mean24:+.2f} vs 24h avg")

with tab_corr:
    st.markdown("Correlate a sensor's flow with TomTom or Vessel metrics (if available).")
    col1, col2, col3 = st.columns(3)
    with col1: sid2 = st.selectbox("Sensor (for correlation)", options=sensor_ids, index=0)
    with col2: win = st.slider("Window (hours)", 6, 72, 24)
    with col3: lag_steps = st.slider("Lag steps (X leads by)", 0, 8, 0)

    s_series = flow_long[flow_long["sensor_base"]==sid2][["timestamp","flow"]].set_index("timestamp").sort_index()
    s_series = s_series.last(f"{win}h").reset_index()

    def corr_kpi(x, y):
        if x.isna().any() or y.isna().any() or len(x)<5: return np.nan
        return float(np.corrcoef(x, y)[0,1])

    # TomTom correlation
    if tt is not None and len(tt)>0:
        tt_local = tt.copy().sort_values("timestamp")
        # Shift X (TomTom) forward by lag_steps
        tt_local["timestamp"] = tt_local["timestamp"] + step*lag_steps
        merged = s_series.merge(tt_local, on="timestamp", how="inner")
        tt_vars = [c for c in merged.columns if c not in ["timestamp","flow"]]
        if tt_vars:
            var = st.selectbox("TomTom variable", options=tt_vars, index=0)
            r = corr_kpi(merged["flow"], merged[var])
            st.metric(f"corr(flow, {var})", f"{r:.2f}" if not np.isnan(r) else "n/a")
            scat = alt.Chart(merged).mark_circle(size=60, opacity=0.5).encode(
                x=alt.X(var, title=f"TomTom {var}"),
                y=alt.Y("flow", title="Sensor flow"),
                tooltip=["timestamp:T","flow:Q", alt.Tooltip(var+":Q")]
            )
            reg = scat.transform_regression(var, "flow").mark_line()
            st.altair_chart((scat+reg).properties(height=300), use_container_width=True)
    else:
        st.info("TomTom parquet not loaded (upload in the sidebar).")

    # Vessel correlation: aggregate counts per native step
    if vessel is not None and len(vessel)>0:
        v = vessel.copy()
        v["cnt"] = 1
        v_agg = v.set_index("timestamp")["cnt"].resample(freq_alias).sum().reset_index().rename(columns={"cnt":"vessel_count"})
        # Shift X (Vessel) forward by lag_steps
        v_agg["timestamp"] = v_agg["timestamp"] + step*lag_steps
        merged2 = s_series.merge(v_agg, on="timestamp", how="inner")
        r2 = corr_kpi(merged2["flow"], merged2["vessel_count"])
        st.metric("corr(flow, vessel_count)", f"{r2:.2f}" if not np.isnan(r2) else "n/a")
        scat2 = alt.Chart(merged2).mark_circle(size=60, opacity=0.5).encode(
            x=alt.X("vessel_count:Q", title="Vessel count per step"),
            y=alt.Y("flow", title="Sensor flow"),
            tooltip=["timestamp:T","flow:Q","vessel_count:Q"]
        )
        reg2 = scat2.transform_regression("vessel_count", "flow").mark_line()
        st.altair_chart((scat2+reg2).properties(height=300), use_container_width=True)
    else:
        st.info("Vessel parquet not loaded (upload in the sidebar).")
