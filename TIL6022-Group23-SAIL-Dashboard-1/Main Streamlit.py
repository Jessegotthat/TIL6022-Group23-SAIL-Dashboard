# Main Streamlit.py — unified host for two Streamlit apps
import os
import sys
import runpy
from pathlib import Path
from contextlib import contextmanager
import streamlit as st

# ---------- Page config ----------
st.set_page_config(page_title="SAIL Unified Dashboard", layout="wide")

ROOT = Path(__file__).resolve().parent

# ---------- helpers ----------
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
    original = _st.set_page_config
    def _noop(*args, **kwargs): pass
    _st.set_page_config = _noop
    try:
        yield
    finally:
        _st.set_page_config = original

def find_script(filename: str) -> Path | None:
    # common locations
    candidates = [
        ROOT / filename,
        ROOT / "Streamlit Dashboard" / filename,
        ROOT.parent / "Streamlit Dashboard" / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    # fallback scan a few levels
    for p in ROOT.rglob(filename):
        return p
    return None

APP1_PATH = find_script("app.py")
APP2_PATH = find_script("streamlit_chart_dashboard_v2.py")

# ---------- UI ----------
st.sidebar.title("Navigation")
st.sidebar.caption(f"Project root: `{ROOT}`")
if not APP1_PATH:
    st.sidebar.error("Couldn’t find `app.py`")
if not APP2_PATH:
    st.sidebar.error("Couldn’t find `streamlit_chart_dashboard_v2.py`")

choice = st.sidebar.radio("Go to", ["Flow Dashboard", "Chart Dashboard"])

# ---------- runner ----------
def run_streamlit_script(script_path: Path):
    """Load a child script and explicitly call its run() if present."""
    if not script_path or not script_path.exists():
        st.error(f"File not found: {script_path}")
        return

    # Child will see its own directory as CWD, and we’ll suppress duplicate page_config
    g = {"__file__": str(script_path)}
    sys.path.insert(0, str(script_path.parent))
    try:
        with temp_chdir(script_path.parent), disable_set_page_config():
            try:
                child_globals = runpy.run_path(str(script_path), init_globals=g)
                run_func = child_globals.get("run")
                if callable(run_func):
                    run_func()
                else:
                    st.error(f"`run()` not found in {script_path.name}")
            except Exception as e:
                st.error(f"Error running {script_path.name}")
                st.exception(e)
    finally:
        if sys.path and sys.path[0] == str(script_path.parent):
            sys.path.pop(0)

# ---------- route ----------
if choice == "Flow Dashboard":
    run_streamlit_script(APP1_PATH)
else:
    run_streamlit_script(APP2_PATH)
