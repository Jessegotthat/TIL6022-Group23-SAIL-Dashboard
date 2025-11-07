SAIL Unified Dashboard — README
=================================

This repository bundles two Streamlit apps into one launcher (main_streamlit.py). Follow the steps below to run it smoothly.

------------------------------------------------------------
1) Folder layout
------------------------------------------------------------
Put all scripts and data files in one folder (e.g., main_streamlit/).  
Required files in the same folder:

- main_streamlit.py  ← the unified launcher  
- app.py  
- streamlit_chart_dashboard_v2.py

Data files (same folder):
- Combined_Crowd_Data.xlsx  
- SAIL2025_LVMA_data_3min_20August-25August2025_flow.csv  
- Sensor Location Data.xlsx  
- Vesselposition_data_20-24Aug2025.parquet  (see download note below)

Folder structure example:

main_streamlit/
├─ main_streamlit.py
├─ app.py
├─ streamlit_chart_dashboard_v2.py
├─ Combined_Crowd_Data.xlsx
├─ SAIL2025_LVMA_data_3min_20August-25August2025_flow.csv
├─ Sensor Location Data.xlsx
└─ Vesselposition_data_20-24Aug2025.parquet

------------------------------------------------------------
2) Get the large Parquet file
------------------------------------------------------------
Because Vesselposition_data_20-24Aug2025.parquet is large, download it first:

Link: https://drive.google.com/file/d/1Pn0t4yXQZzL4mS0QVnFergsrc7OVeIJU/view?usp=drive_link  
Save it directly into the same folder as above (main_streamlit/).

------------------------------------------------------------
3) Environment & dependencies
------------------------------------------------------------
Python version: 3.10–3.11 recommended

Install packages:
# (optional but recommended) create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# install dependencies
pip install --upgrade pip
pip install streamlit pandas numpy plotly openpyxl pyarrow

If using Anaconda:
conda create -n saildash python=3.11 -y
conda activate saildash
pip install streamlit pandas numpy plotly openpyxl pyarrow

------------------------------------------------------------
4) Run the unified Streamlit app
------------------------------------------------------------
From the same folder:
streamlit run main_streamlit.py

Then open the local URL shown in the terminal (usually http://localhost:8501/).
Use the sidebar to switch between:
- Flow Dashboard (loads app.py)
- Chart Dashboard (loads streamlit_chart_dashboard_v2.py)

------------------------------------------------------------
5) Troubleshooting
------------------------------------------------------------
• “Couldn’t find app.py or streamlit_chart_dashboard_v2.py”
  → Ensure both are in the same folder as main_streamlit.py.

• “File not found (CSV/XLSX/Parquet)”
  → Confirm all data files are in the same folder and filenames match.

• “ModuleNotFoundError: No module named 'openpyxl' / 'pyarrow'”
  → Install missing packages: pip install openpyxl pyarrow

• “Port already in use”
  → Run on another port: streamlit run main_streamlit.py --server.port 8502

• “Multiple st.set_page_config error”
  → Already handled by main_streamlit.py. If still appears, restart the app.

------------------------------------------------------------
6) Notes for developers
------------------------------------------------------------
- The launcher executes each child script in its own working directory.
- Duplicate st.set_page_config() calls are safely suppressed.
- Optionally, you can define run() in each child script; the launcher will call it automatically.

That's it — you're ready to explore the SAIL Unified Dashboard!
