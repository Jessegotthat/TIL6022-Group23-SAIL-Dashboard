import streamlit as st
import sys
from pathlib import Path

# Add the "Streamlit Dashboard" folder to Python's path
sys.path.append(str(Path(__file__).parent / "Streamlit Dashboard"))

# Import the dashboards from inside the folder
import app
import streamlit_chart_dashboard_v2 as charts

st.set_page_config(page_title="SAIL Dashboard", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["App Dashboard", "Chart Dashboard"])

if page == "App Dashboard":
    app.run()
elif page == "Chart Dashboard":
    charts.run()
