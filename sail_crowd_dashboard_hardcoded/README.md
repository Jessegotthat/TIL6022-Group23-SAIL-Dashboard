# SAIL Crowd Flow Dashboard (Hardcoded to your files)


*Yuxuan:I changed the path into a relative path for simplicity for midterm check,*
        *which means as long as you keep this structure, the code will load the data automatically:*
        
        sail_crowd_dashboard_hardcoded/
        ├─ app.py
        ├─ requirements.txt
        ├─ README.md
        ├─ inputs/
        │  ├─ SAIL2025_LVMA_data_3min_20August-25August2025_flow.csv
        │  └─ Sensor Location Data.xlsx
        └─ data/
        ├─ 20250820163000_stream.tomtom.analyze-sail.parquet
        └─ 20250820163000_stream.vessel-positions-anonymized-processed.analyze-sail.parquet

*You can move the whole folder anywhere—just don’t change the files inside.*

*To run the code, below are codes you need to paste and run in Anaconda Prompt LINE BY LINE:*
*(Replace the first line with your actual project path)*

    cd /d "D:\work\TUD\学期间\TIL6022 TIL Python Programming\Project\sail_crowd_dashboard_hardcoded"

    conda create -n saildash python=3.10 -y
    conda activate saildash

    pip install -r requirements.txt
    streamlit run app.py

*The second line is to create a new evironment. You can name it as you wish. after creating it(might take some time,please wait with patience), you do not need to do that next time.*
*The forth line is to install dependencies. This makes sure that packages we use are at same versions. Also one-time unless requirements change.*

*When it starts, your browser should open at http://localhost:8501. To stop the app, press Ctrl+C in the terminal(Anaconda Prompt).*

*To re-run after you edit code, click Rerun in the website (top-right).*


*Annotations below are useless for now. You do not need to check that yet.*




English-only UI. Defaults to your file paths; if missing, you can upload files in the sidebar.

<!-- ## Default file paths (override via sidebar uploads)
- Flow CSV (wide): `/mnt/data/SAIL2025_LVMA_data_3min_20August-25August2025_flow.csv`
- Sensor meta (Excel): `/mnt/data/Sensor Location Data.xlsx`
- TomTom Parquet: `data/20250820163000_stream.tomtom.analyze-sail.parquet` (relative; upload if not found)
- Vessel Parquet: `data/20250820163000_stream.vessel-positions-anonymized-processed.analyze-sail.parquet` (relative; upload if not found) -->

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- Flow CSV is expected to be **wide**: a `timestamp` column, many sensor-direction columns like `CMSA-GAKH-01_0`, and possibly helper columns (`hour`, `minute`, ...). The app auto-melts to long and **aggregates directions** to sensor base id (before `_`). 
- Sensor meta Excel columns expected (Dutch): `Objectummer` (sensor base id), `Locatienaam` (name), `Lat/Long` ('lat, lon'), `Breedte` and/or `Effectieve  breedte` (width). The app prefers effective width if present.
- TomTom/Vessel are **Parquet** as in your notebook (fastparquet/pyarrow supported). Upload if defaults are not present.
