# streamlit_chart_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ---------- PATHS ----------
BASE_DIR = Path(__file__).parent

# Use relative paths based on script location
CROWD_XLSX = BASE_DIR / "Combined_Crowd_Data.xlsx"
VESSEL_PARQUET = BASE_DIR / "Vesselposition_data_20-24Aug2025.parquet"

# ---------- CONSTANTS ----------
# Overall vessel date window (inclusive start, exclusive end)
V_RANGE_START_TS = pd.Timestamp("2025-08-20 00:00:00")
V_RANGE_END_TS   = pd.Timestamp("2025-08-25 00:00:00")

# If your local event time in August is CEST (UTC+2), set +2 here
LOCAL_UTC_OFFSET = pd.Timedelta(hours=2)

st.set_page_config(page_title="SAIL 2025 Crowd & Vessel Dashboard", layout="wide")


# ---------- HELPERS ----------
def donut(value: int, total: int, title: str, main_color: str, remainder_color="#e6e6e6") -> go.Figure:
    v = int(value)
    T = int(total) if total and total > 0 else max(v, 1)
    remain = max(T - v, 0)

    fig = go.Figure(go.Pie(
        labels=[title, "Remainder"],
        values=[v, remain],
        hole=0.76,
        sort=False,
        direction="clockwise",
        textinfo="none",
        marker=dict(colors=[main_color, remainder_color])
    ))
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[
            dict(text=f"<b>{title}</b>", x=0.5, y=0.72, showarrow=False, font=dict(size=18, color="white")),
            dict(text=f"<b style='font-size:38px'>{v:,}</b>", x=0.5, y=0.50, showarrow=False),
        ],
    )
    return fig


def red_heat_colors(vals: np.ndarray) -> list:
    if len(vals) == 0:
        return []
    vmin = float(vals.min())
    vmax = float(max(vals.max(), 1))
    norm = (vals - vmin) / (vmax - vmin + 1e-9)
    return [f"rgba(255,{int(230-200*n)},{int(230-200*n)},0.95)" for n in norm]


@st.cache_data(show_spinner=False)
def load_crowd(path: str):
    # Expect 'timestamp' col + many area columns
    df = pd.read_excel(path)
    assert "timestamp" in df.columns, "Crowd file must have a 'timestamp' column."
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    area_cols = [c for c in df.columns if c != "timestamp"]
    df[area_cols] = df[area_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    df["total_crowd"] = df[area_cols].sum(axis=1)
    return df, area_cols


@st.cache_data(show_spinner=False)
def event_days_from_crowd(crowd_df: pd.DataFrame):
    # Show all days present in crowd data; vessel is filtered separately
    return sorted(crowd_df["timestamp"].dt.date.unique())


@st.cache_data(show_spinner=False)
def load_vessel(path: str,
                start_ts: pd.Timestamp,
                end_ts: pd.Timestamp,
                usecols=("id", "upload-timestamp")) -> pd.DataFrame:
    # Read only needed columns and keep UTC tz-aware internally
    df = pd.read_parquet(path, columns=list(usecols))
    ts = pd.to_datetime(df["upload-timestamp"], errors="coerce", utc=True)
    df = df.assign(upload_ts=ts).dropna(subset=["upload_ts"])
    # Filter to the global window in UTC; this prunes data massively
    mask = (df["upload_ts"] >= start_ts.tz_localize("UTC")) & (df["upload_ts"] < end_ts.tz_localize("UTC"))
    df = df.loc[mask, ["id", "upload_ts"]].sort_values("upload_ts").reset_index(drop=True)
    return df


def vessel_unique_up_to(vdf: pd.DataFrame, end_ts_utc: pd.Timestamp) -> int:
    # Exclusive end
    return vdf.loc[vdf["upload_ts"] < end_ts_utc, "id"].nunique()


def vessel_unique_today(vdf: pd.DataFrame, day0_utc: pd.Timestamp, day1_utc: pd.Timestamp) -> int:
    mask = (vdf["upload_ts"] >= day0_utc) & (vdf["upload_ts"] < day1_utc)
    return vdf.loc[mask, "id"].nunique()


def vessel_timeseries_bin(vdf: pd.DataFrame, day0_utc: pd.Timestamp, day1_utc: pd.Timestamp, freq="15min") -> pd.DataFrame:
    mask = (vdf["upload_ts"] >= day0_utc) & (vdf["upload_ts"] < day1_utc)
    sub = vdf.loc[mask].copy()
    if sub.empty:
        idx = pd.date_range(day0_utc, day1_utc, freq=freq, inclusive="left", tz="UTC")
        return pd.DataFrame({"bin": idx, "unique_ids": 0})
    sub["bin"] = sub["upload_ts"].dt.floor(freq)
    ts = sub.groupby("bin")["id"].nunique().reset_index(name="unique_ids")
    return ts.sort_values("bin").reset_index(drop=True)


# ---------- LOAD DATA ----------
crowd_df, area_cols = load_crowd(CROWD_XLSX)
event_days = event_days_from_crowd(crowd_df)
vdf = load_vessel(VESSEL_PARQUET, V_RANGE_START_TS, V_RANGE_END_TS)

# ---------- HEADER + CONTROLS (aligned via form) ----------
st.markdown("## SAIL Amsterdam 2025 — Crowd Management Dashboard")

with st.form("controls", clear_on_submit=False):
    c1, c2, c3, c4 = st.columns([1.1, 0.9, 0.9, 0.6])
    with c1:
        day = st.selectbox("Date", event_days, index=0, format_func=lambda d: d.strftime("%Y-%m-%d"))
    with c2:
        start_str = st.selectbox("Start", [f"{h:02d}:00" for h in range(24)], index=9)
    with c3:
        end_str   = st.selectbox("End",   [f"{h:02d}:00" for h in range(1, 25)], index=18)
    with c4:
        st.markdown("&nbsp;", unsafe_allow_html=True)  # aligns button vertically with inputs
        submitted = st.form_submit_button("Update", use_container_width=True)

# Build selected window
sh, sm = map(int, start_str.split(":"))
eh, em = map(int, end_str.split(":"))
day0_local = pd.Timestamp(day).replace(hour=0, minute=0, second=0, microsecond=0)
day1_local = day0_local + pd.Timedelta(days=1)
day_start_local = pd.Timestamp(day).replace(hour=sh, minute=sm)
day_end_local   = pd.Timestamp(day).replace(hour=eh, minute=em)
if day_end_local <= day_start_local:
    day_end_local = day_start_local + pd.Timedelta(hours=1)

# For vessel, convert the local window to UTC once
day0_utc = (day0_local - LOCAL_UTC_OFFSET).tz_localize("UTC")
day1_utc = (day1_local - LOCAL_UTC_OFFSET).tz_localize("UTC")
day_start_utc = (day_start_local - LOCAL_UTC_OFFSET).tz_localize("UTC")
day_end_utc   = (day_end_local   - LOCAL_UTC_OFFSET).tz_localize("UTC")

st.caption(f"**Window:** {day_start_local} → {day_end_local}")

# ---------- CROWD METRICS ----------
event_start = crowd_df["timestamp"].min()
event_end   = crowd_df["timestamp"].max()

grand_total_5days = crowd_df.loc[(crowd_df["timestamp"] >= event_start) &
                                 (crowd_df["timestamp"] <= event_end), "total_crowd"].sum()

current_visitors_upto = crowd_df.loc[(crowd_df["timestamp"] >= event_start) &
                                     (crowd_df["timestamp"] <  day_end_local), "total_crowd"].sum()

today_mask_full = (crowd_df["timestamp"] >= day0_local) & (crowd_df["timestamp"] < day1_local)
today_total_full_day = crowd_df.loc[today_mask_full, "total_crowd"].sum()

today_mask_upto = (crowd_df["timestamp"] >= day0_local) & (crowd_df["timestamp"] < day_end_local)
today_visitors_upto  = crowd_df.loc[today_mask_upto, "total_crowd"].sum()

# Crowd line (3 min bins, value-only hover)
vis_ts = (crowd_df.loc[(crowd_df["timestamp"] >= day0_local) & (crowd_df["timestamp"] < day1_local),
                       ["timestamp", "total_crowd"]]
                .set_index("timestamp")["total_crowd"]
                .resample("3min").sum().reset_index())

# Area Occupation Top-10 (snapshot at last point in window)
snap = crowd_df.loc[(crowd_df["timestamp"] >= day_start_local) & (crowd_df["timestamp"] < min(day_end_local, day1_local))]
if snap.empty:
    top_series = pd.Series(dtype=int)
else:
    last_row = snap.iloc[-1][area_cols]
    top_series = last_row.sort_values(ascending=False).head(10)

# ---------- VESSEL METRICS ----------
vessel_all5_total = vdf["id"].nunique()
vessel_current_num = vessel_unique_up_to(vdf, day_end_utc)
vessel_today_denom = vessel_unique_today(vdf, day0_utc, day1_utc)
vessel_today_num   = vessel_unique_today(vdf, day0_utc, day_end_utc)
vessel_ts = vessel_timeseries_bin(vdf, day0_utc, day1_utc, freq="15min")

# ---------- CHARTS ----------
d1, d2 = st.columns(2)
with d1:
    st.plotly_chart(donut(today_visitors_upto, today_total_full_day, "Today Number of Visitor", "#FFA500"),
                    use_container_width=True)
with d2:
    st.plotly_chart(donut(vessel_today_num, vessel_today_denom, "Today Number of Vessel", "#2E9AFE"),
                    use_container_width=True)

d3, d4 = st.columns(2)
with d3:
    st.plotly_chart(donut(current_visitors_upto, grand_total_5days, "Current Number of Visitor", "#FFA500"),
                    use_container_width=True)
with d4:
    st.plotly_chart(donut(vessel_current_num, vessel_all5_total, "Current Number of Vessel", "#2E9AFE"),
                    use_container_width=True)

c5, c6 = st.columns(2)
with c5:
    fig_vis_line = px.line(vis_ts, x="timestamp", y="total_crowd", title="Crowd (Today) — total per 3 minutes")
    fig_vis_line.update_traces(line=dict(shape="spline", width=2, color="#FFA500"),
                               hovertemplate="%{y:,}<extra></extra>")
    fig_vis_line.update_layout(margin=dict(l=0, r=0, t=30, b=0),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               hovermode="x unified", xaxis_title="", yaxis_title="People")
    st.plotly_chart(fig_vis_line, use_container_width=True)

with c6:
    # Convert UTC bins to naive timestamps for display if desired
    vts_disp = vessel_ts.copy()
    vts_disp["bin"] = (vts_disp["bin"] + LOCAL_UTC_OFFSET).dt.tz_localize(None)
    fig_ves_line = px.line(vts_disp, x="bin", y="unique_ids",
                           title="Vessel (Today) — unique IDs per 15 minutes")
    fig_ves_line.update_traces(line=dict(shape="spline", width=2, color="#2E9AFE"),
                               hovertemplate="%{y:,}<extra></extra>")
    fig_ves_line.update_layout(margin=dict(l=0, r=0, t=30, b=0),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               hovermode="x unified", xaxis_title="", yaxis_title="Vessels")
    st.plotly_chart(fig_ves_line, use_container_width=True)

# Area bars
fig_area = go.Figure(go.Bar(
    x=top_series.index.astype(str),
    y=top_series.values,
    marker_color=red_heat_colors(top_series.values.astype(float)),
    text=[f"{int(v):,}" for v in top_series.values],
    textposition="inside",
    insidetextanchor="middle",
    textfont=dict(size=13, color="white"),
))
fig_area.update_layout(
    title=dict(text="Area Occupation — Top 10 (last point in window)", y=0.98),
    xaxis_title="Area",
    yaxis=dict(title="People", tickformat=",d"),
    margin=dict(l=20, r=20, t=80, b=60),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    bargap=0.25,
)
st.plotly_chart(fig_area, use_container_width=True)

st.caption("Tip: adjust Date/Start/End above and press Update. Donuts compare current vs total (grey).")
