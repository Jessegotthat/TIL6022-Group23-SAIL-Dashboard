import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ---------- PATHS (edit if needed) ----------
CROWD_XLSX = r"C:\Users\muham\github-classroom\TIL6022-Group23-SAIL-Dashboard\Combined_Crowd_Data.xlsx"
VESSEL_PARQUET = r"C:\Users\muham\github-classroom\TIL6022-Group23-SAIL-Dashboard\Vesselposition_data_20-24Aug2025.parquet"


# ---------- CONSTANTS ----------
# Vessel date range (inclusive start, exclusive end)
V_RANGE_START_TS = pd.Timestamp("2025-08-20 00:00:00")
V_RANGE_END_TS   = pd.Timestamp("2025-08-25 00:00:00")
V_USECOLS = ["id", "upload-timestamp"]

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
        # Two annotations INSIDE the donut:
        # 1) Title (white), just above the value
        # 2) Value (big, centered)
        annotations=[
            dict(  # title
                text=f"<b>{title}</b>",
                x=0.5, y=0.72, showarrow=False,
                font=dict(size=18, color="white")  # bigger, white
            ),
            dict(  # value
                text=f"<b style='font-size:38px'>{v:,}</b>",
                x=0.5, y=0.50, showarrow=False
            ),
        ],
    )
    return fig


def red_heat_colors(vals: np.ndarray) -> list:
    """Light→dark red scale based on value."""
    if len(vals) == 0:
        return []
    vmin = float(vals.min())
    vmax = float(max(vals.max(), 1))
    norm = (vals - vmin) / (vmax - vmin + 1e-9)
    return [f"rgba(255,{int(230-200*n)},{int(230-200*n)},0.95)" for n in norm]

@st.cache_data(show_spinner=False)
def load_crowd(path: str):
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
    days = sorted(crowd_df["timestamp"].dt.date.unique())
    # keep only days within the vessel data window (20–24 Aug)
    days = [d for d in days if V_RANGE_START_TS.date() <= d < V_RANGE_END_TS.date()]
    return days

def load_vessel(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["upload-timestamp"] = pd.to_datetime(df["upload-timestamp"], errors="coerce", utc=True)
    df["upload-timestamp"] = df["upload-timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    return df.sort_values("upload-timestamp").dropna(subset=["upload-timestamp"])

@st.cache_data(show_spinner=False)
def vessel_all5_unique(vdf: pd.DataFrame) -> int:
    return vdf["id"].nunique()

def vessel_unique_up_to(vdf: pd.DataFrame, end_ts: pd.Timestamp) -> int:
    return vdf.loc[vdf["upload-timestamp"] <= end_ts, "id"].nunique()

def vessel_unique_today(vdf: pd.DataFrame, day: pd.Timestamp, end_ts: pd.Timestamp | None) -> int:
    day0, day1 = pd.Timestamp(day).normalize(), pd.Timestamp(day).normalize() + pd.Timedelta(days=1)
    if end_ts is not None:
        day1 = min(day1, pd.Timestamp(end_ts))
    mask = (vdf["upload-timestamp"] >= day0) & (vdf["upload-timestamp"] <= day1)
    return vdf.loc[mask, "id"].nunique()

def vessel_timeseries_15min(vdf: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    day0, day1 = pd.Timestamp(day).normalize(), pd.Timestamp(day).normalize() + pd.Timedelta(days=1)
    mask = (vdf["upload-timestamp"] >= day0) & (vdf["upload-timestamp"] < day1)
    sub = vdf.loc[mask].copy()
    if sub.empty:
        idx = pd.date_range(day0, day1, freq="15min", inclusive="left")
        return pd.DataFrame({"bin": idx, "unique_ids": 0})
    sub["bin"] = sub["upload-timestamp"].dt.floor("15min")
    ts = sub.groupby("bin")["id"].nunique().reset_index(name="unique_ids")
    return ts.sort_values("bin").reset_index(drop=True)

# ---------- LOAD CROWD ----------
crowd_df, area_cols = load_crowd(CROWD_XLSX)
event_days = event_days_from_crowd(crowd_df)

# ---------- HEADER + CONTROLS ----------
st.markdown("## SAIL Amsterdam 2025 — Crowd Management Dashboard")

c1, c2, c3, c4 = st.columns([1.1, 0.9, 0.9, 0.6])
with c1:
    day = st.selectbox("Date", event_days, index=0, format_func=lambda d: d.strftime("%Y-%m-%d"))
with c2:
    start_str = st.selectbox("Start", [f"{h:02d}:00" for h in range(24)], index=9)
with c3:
    end_str   = st.selectbox("End",   [f"{h:02d}:00" for h in range(1,25)], index=18)
with c4:
    # simple button to make it feel interactive; state updates immediately anyway
    st.button("Update", use_container_width=True)

sh, sm = map(int, start_str.split(":"))
eh, em = map(int, end_str.split(":"))
day_start = pd.Timestamp(day).replace(hour=sh, minute=sm)
day_end   = pd.Timestamp(day).replace(hour=eh, minute=em)
if day_end <= day_start:
    day_end = day_start + pd.Timedelta(hours=1)

st.caption(f"**Window:** {day_start} → {day_end}")

# ---------- CROWD METRICS ----------
event_start = crowd_df["timestamp"].min()
event_end   = crowd_df["timestamp"].max()

# Donut logic:
# - Current Visitor: numerator = sum from event start up to 'day_end'; denominator = grand sum across all 5 days
grand_total_5days = crowd_df.loc[(crowd_df["timestamp"] >= event_start) &
                                 (crowd_df["timestamp"] <= event_end), "total_crowd"].sum()
current_visitors_upto = crowd_df.loc[(crowd_df["timestamp"] >= event_start) &
                                     (crowd_df["timestamp"] <= day_end), "total_crowd"].sum()

# - Today Visitor: numerator = sum from day0 up to min(day_end, day1); denominator = full selected day
day0, day1 = pd.Timestamp(day), pd.Timestamp(day) + pd.Timedelta(days=1)
today_total_full_day = crowd_df.loc[(crowd_df["timestamp"] >= day0) &
                                    (crowd_df["timestamp"] < day1), "total_crowd"].sum()
today_visitors_upto  = crowd_df.loc[(crowd_df["timestamp"] >= day0) &
                                    (crowd_df["timestamp"] <= min(day_end, day1)), "total_crowd"].sum()

# Crowd line (3 min)
vis_ts = (crowd_df.loc[(crowd_df["timestamp"] >= day0) & (crowd_df["timestamp"] < day1),
                       ["timestamp", "total_crowd"]]
                .set_index("timestamp")["total_crowd"]
                .resample("3min").sum().reset_index())

# Area Occupation Top-10 (snapshot at last point in window; toggle below for average if you want)
snap = crowd_df.loc[(crowd_df["timestamp"] >= day_start) & (crowd_df["timestamp"] <= min(day_end, day1))]
if snap.empty:
    top_series = pd.Series(dtype=int)
else:
    # snapshot:
    last_row = snap.iloc[-1][area_cols]
    top_series = last_row.sort_values(ascending=False).head(10)
    # If you want cumulative per area during the window instead:
    # top_series = snap[area_cols].sum().sort_values(ascending=False).head(10)

# ---------- VESSEL METRICS (chunked) ----------
vdf = load_vessel(VESSEL_PARQUET)

vessel_all5_total = vessel_all5_unique(vdf)
vessel_current_num = vessel_unique_up_to(vdf, day_end)
vessel_today_denom = vessel_unique_today(vdf, day, end_ts=day1)
vessel_today_num   = vessel_unique_today(vdf, day, end_ts=day_end)
vessel_ts = vessel_timeseries_15min(vdf, day)

# ---------- CHARTS LAYOUT ----------
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
    fig_ves_line = px.line(vessel_ts, x="bin", y="unique_ids", title="Vessel (Today) — unique IDs per 15 minutes")
    fig_ves_line.update_traces(line=dict(shape="spline", width=2, color="#2E9AFE"), 
                               hovertemplate="%{y:,}<extra></extra>")
    fig_ves_line.update_layout(margin=dict(l=0, r=0, t=30, b=0),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               hovermode="x unified", xaxis_title="", yaxis_title="Vessels")
    st.plotly_chart(fig_ves_line, use_container_width=True)

# Area bars (heat + labels, with safe margins so labels don't overlap the title)
fig_area = go.Figure(go.Bar(
    x=top_series.index.astype(str),
    y=top_series.values,
    marker_color=red_heat_colors(top_series.values.astype(float)),
    text=[f"{int(v):,}" for v in top_series.values],
    textposition="inside",              # avoids overlap with title
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

# Footer hint
st.caption("Tip: change Date/Start/End at the top and click Update. Donuts show value vs total (grey).")
