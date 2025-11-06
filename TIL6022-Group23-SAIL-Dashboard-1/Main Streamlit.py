import runpy
from pathlib import Path
import streamlit as st

# Paths to your two dashboards (adjust names if needed)
ROOT = Path(__file__).parent
APP1_PATH = ROOT / "Streamlit Dashboard" / "app.py"
APP2_PATH = ROOT / "Streamlit Dashboard" / "streamlit_chart_dashboard_v2.py"

# Configure once here to avoid "set_page_config called multiple times"
st.set_page_config(page_title="SAIL Unified Dashboard", layout="wide")

st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", ["App Dashboard", "Chart Dashboard"])

def run_streamlit_script(script_path: Path, allow_set_page_config: bool = False):
    g = {
        "__name__": "__main__",
        "__file__": str(script_path),
        "st": st,
    }

    # If the child script calls st.set_page_config, make it a no-op
    if not allow_set_page_config:
        def _noop(*args, **kwargs):
            pass
        g["st"].set_page_config = _noop

    # Execute the script
    runpy.run_path(str(script_path), init_globals=g)

if choice == "App Dashboard":
    # First page: we already called set_page_config above, so keep it disabled here
    run_streamlit_script(APP1_PATH, allow_set_page_config=False)
else:
    # Second page: also disable set_page_config to prevent duplicate calls
    run_streamlit_script(APP2_PATH, allow_set_page_config=False)
