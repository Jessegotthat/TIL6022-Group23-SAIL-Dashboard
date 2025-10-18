import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster

# ---------- CONFIG ----------
DEFAULT_PATH = "Sensor Location Data.xlsx"   # change if your filename differs
CITY_CENTER = [52.377956, 4.897070]         # Amsterdam fallback

st.set_page_config(page_title="SAIL Sensor Dashboard", layout="wide")

@st.cache_data
def load_data(src):
    """Load from path (str) or uploaded file object."""
    if isinstance(src, str):
        if src.lower().endswith(".xlsx"):
            return pd.read_excel(src)
        elif src.lower().endswith(".csv"):
            return pd.read_csv(src)
        else:
            raise ValueError("Use .xlsx or .csv")
    else:
        name = (src.name or "").lower()
        if name.endswith(".xlsx"):
            return pd.read_excel(src)
        elif name.endswith(".csv"):
            return pd.read_csv(src)
        else:
            raise ValueError("Upload a .xlsx or .csv")

def coerce_num(s: pd.Series) -> pd.Series:
    # Handle strings and commas like "52,37"
    return pd.to_numeric(s.astype(str).str.replace(",", "."), errors="coerce")

def build_map(df, lat_col, lon_col, name_col):
    dff = df.copy()
    dff["_lat"] = coerce_num(dff[lat_col])
    dff["_lon"] = coerce_num(dff[lon_col])
    dff = dff.dropna(subset=["_lat", "_lon"])

    center = [dff["_lat"].mean(), dff["_lon"].mean()] if len(dff) else CITY_CENTER
    m = folium.Map(location=center, zoom_start=13, tiles="cartodbpositron")

    cluster = MarkerCluster().add_to(m)
    for _, r in dff.iterrows():
        folium.Marker(
            [float(r["_lat"]), float(r["_lon"])],
            popup=str(r.get(name_col, "")),
        ).add_to(cluster)
    return m, dff

def guess_columns(cols):
    # convert Index -> list to avoid "truth value of Index is ambiguous"
    cols = list(cols)
    lc = [str(c).lower() for c in cols]

    def pick(cands, default=None):
        for c in cands:
            if c in lc:
                return cols[lc.index(c)]
        # safe fallback
        if default is not None:
            return default
        return cols[0] if len(cols) > 0 else None

    lat = pick(["latitude", "lat", "y"])
    lon = pick(["longitude", "lon", "long", "x"])
    name = pick(["sensor", "name", "id", "label", "station"])
    return lat, lon, name

# ---------- SIDEBAR ----------
st.sidebar.title("Controls")
uploaded = st.sidebar.file_uploader("Upload .xlsx or .csv", type=["xlsx","csv"])
use_default = st.sidebar.checkbox("Use file from repo", value=(uploaded is None))

try:
    df = load_data(DEFAULT_PATH if use_default else uploaded)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

if df is None or df.empty:
    st.warning("No data loaded.")
    st.stop()

lat_guess, lon_guess, name_guess = guess_columns(df.columns)

col1, col2, col3 = st.sidebar.columns(3)
lat_col  = col1.selectbox("Latitude",  df.columns, index=list(df.columns).index(lat_guess) if lat_guess in df.columns else 0)
lon_col  = col2.selectbox("Longitude", df.columns, index=list(df.columns).index(lon_guess) if lon_guess in df.columns else 0)
name_col = col3.selectbox("Name",      df.columns, index=list(df.columns).index(name_guess) if name_guess in df.columns else 0)

search = st.sidebar.text_input("Filter name contains", "")
limit  = st.sidebar.number_input("Max points (0 = all)", min_value=0, max_value=200000, value=0, step=100)

with st.expander("Preview data", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)

# Apply filters
dff = df.copy()
if search:
    dff = dff[dff[name_col].astype(str).str.contains(search, case=False, na=False)]
if limit:
    dff = dff.head(int(limit))

# ---------- MAIN ----------
st.title("Sensor Location Map")
st.caption("Predictive Dashboard — SAIL 2025")

m, dff_valid = build_map(dff, lat_col, lon_col, name_col)
st.components.v1.html(m.get_root().render(), height=650)

# KPIs
k1, k2, k3 = st.columns(3)
k1.metric("Total rows", len(df))
k2.metric("After filters", len(dff))
k3.metric("Valid points plotted", len(dff_valid))

# Download filtered data
st.download_button(
    "Download filtered CSV",
    dff.to_csv(index=False).encode("utf-8"),
    file_name="sensors_filtered.csv",
    mime="text/csv",
)
