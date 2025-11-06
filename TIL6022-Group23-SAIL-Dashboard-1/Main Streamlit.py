# Main Streamlit.py (auto-discovers child dashboards)
import os, sys, runpy
from pathlib import Path
import streamlit as st
from contextlib import contextmanager

st.set_page_config(page_title="SAIL Unified Dashboard", layout="wide")

ROOT = Path(__file__).resolve().parent

# ---- locate child scripts robustly ----
def find_script(filename: str) -> Path | None:
    # Try common locations relative to ROOT
    candidates = [
        ROOT / filename,
        ROOT / "Streamlit Dashboard" / filename,
        ROOT.parent / "Streamlit Dashboard" / filename,
        ROOT / "TIL6022-Group23-SAIL-Dashboard" / "Streamlit Dashboard" / filename,
        ROOT / "TIL6022-Group23-SAIL-Dashboard-1" / "Streamlit Dashboard" / filename,
    ]
    for c in candidates:
        if c.exists():
            return c

    # As a fallback, scan a few levels deep
    for p in ROOT.rglob(filename):
        return p
    return None

APP1_PATH = find_script("app.py")
APP2_PATH = find_script("streamlit_chart_dashboard_v2.py")

with st.sidebar:
    st.title("Navigation")
    st.caption(f"Project root: `{ROOT}`")
    if not APP1_PATH:
        st.error("Couldn’t find `app.py`")
    if not APP2_PATH:
        st.error("Couldn’t find `streamlit_chart_dashboard_v2.py`")
    choice = st.radio("Go to", ["App Dashboard", "Chart Dashboard"])

@contextmanager
def temp_chdir(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)

@contextmanager
def disable_set_page_config():
    import streamlit as _st
    orig = _st.set_page_config
    def _noop(*a, **k): pass
    _st.set_page_config = _noop
    try:
        yield
    finally:
        _st.set_page_config = orig

def run_streamlit_script(script_path: Path):
    if not script_path or not script_path.exists():
        st.error(f"File not found: {script_path}")
        return
    g = {"__name__": "__main__", "__file__": str(script_path)}
    sys.path.insert(0, str(script_path.parent))
    try:
        with temp_chdir(script_path.parent), disable_set_page_config():
            try:
                runpy.run_path(str(script_path), init_globals=g)
            except Exception as e:
                st.error(f"Error running {script_path.name}")
                st.exception(e)
    finally:
        if sys.path and sys.path[0] == str(script_path.parent):
            sys.path.pop(0)

if choice == "App Dashboard":
    run_streamlit_script(APP1_PATH)
else:
    run_streamlit_script(APP2_PATH)
