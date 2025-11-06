# Main Streamlit.py
import streamlit as st
import app
import streamlit_chart_dashboard_v2 as charts

st.set_page_config(page_title="SAIL Dashboard", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["App Dashboard", "Chart Dashboard"])

if page == "App Dashboard":
    app.run()
elif page == "Chart Dashboard":
    charts.run()
