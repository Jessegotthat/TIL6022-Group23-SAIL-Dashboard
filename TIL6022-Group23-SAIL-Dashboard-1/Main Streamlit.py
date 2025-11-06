# Main Streamlit.py
import os
import sys
import runpy
from pathlib import Path
import streamlit as st
from contextlib import contextmanager

ROOT = Path(__file__).parent
APP1_PATH = ROOT / "Streamlit Dashboard" / "app.py"
APP2_PATH = ROOT / "Streamlit Dashboard" / "streamlit_chart_dashboard_v2.py"

st.set_page_config(page_title="SAIL Unified Dashboard", layout="wide")

st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", ["App Dashboard", "Chart Dashboard"])

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

def run_streamlit_script(script_path: Path, block_set_page_config: bool = True):
    g = {"__name__": "__main__", "__file__": str(script_path)}
    sys.path.insert(0, str(script_path.parent))
    try:
        with temp_chdir(script_path.parent):
            ctx = disable_set_page_config() if block_set_page_config else nullcontext()
            with ctx:
                try:
                    runpy.run_path(str(script_path), init_globals=g)
                except Exception as e:
                    st.error(f"Error running {script_path.name}")
                    st.exception(e)
    finally:
        if sys.path and sys.path[0] == str(script_path.parent):
            sys.path.pop(0)

class nullcontext:
    def __enter__(self): return None
    def __exit__(self, *exc): return False

if choice == "App Dashboard":
    run_streamlit_script(APP1_PATH, block_set_page_config=True)
else:
    run_streamlit_script(APP2_PATH, block_set_page_config=True)
