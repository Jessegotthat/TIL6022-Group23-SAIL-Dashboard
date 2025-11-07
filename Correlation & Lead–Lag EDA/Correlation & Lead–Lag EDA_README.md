# SAIL 2025 — Correlation & Lead–Lag EDA

Exploratory data analysis (EDA) to understand how **road traffic** (TomTom) and **vessel activity** relate to **pedestrian flow** during SAIL 2025.  
The script parses raw inputs, builds aggregated **driver features** at a common cadence (default **3-minute**), computes **Pearson correlations** and **lead–lag profiles**, and exports summary tables and plots to `outputs/`.

> **Note (中文):** 这是该脚本的 README。脚本会读取行人流量、TomTom道路交通与船舶定位数据，统一到 3 分钟频率，生成若干驱动特征，并计算与人流的相关与超前-滞后关系，同时输出表格与图像。

---

## 1) What this script does

1. **Read & reshape pedestrian flow**
   - Input: wide CSV with `timestamp` and many per-sensor angle columns (e.g. `CMSA-XXXX_0`, `CMSA-XXXX_120`, …)
   - **Combine angles → per-sensor flow**; optionally average across sensors to get **city-wide mean**.
2. **Load TomTom traffic**
   - Prefer **parquet** if available; otherwise parse the **nested CSV** (`time`, `data` where `data` is a micro-CSV `id,traffic_level` lines).
   - Aggregate to **3-minute** features: `traffic_level_mean`, `traffic_level_var`, `traffic_level_chg` (first difference).
3. **Load Vessel positions**
   - Prefer **parquet** if available; otherwise read CSV and aggregate to **3-minute** features:
     `vessel_count`, `vessel_avg_speed`, optionally `vessel_avg_length`, `vessel_avg_beam`, plus **per-type counts** (top-K).
4. **Align on timestamp**
   - Force timestamps to **tz-naive** and **inner-join** on a common time index.
5. **Compute statistics**
   - **Pearson correlation** vs `human_flow`.
   - **Lead–lag correlation** scanning ±12 steps (±36 min at 3-min cadence).
6. **Plot & export**
   - Correlation matrix, **lead–lag curves**, and **small-multiples time series**.
   - Save CSVs and PNGs in `outputs/`.

---

## 2) Repository / file layout (expected)

```bash
project_root/
├─ inputs/
│ ├─ SAIL2025_LVMA_data_3min_20August-25August2025_flow.csv
│ └─ sensor-location.xlsx
├─ data/
│ ├─ TomTom_data_20-24Aug2025.csv
│ ├─ Vesselposition_data_20-24Aug2025.csv
│ ├─ _cache/
│ │ ├─ tomtom_quick.parquet # preferred if present
│ │ └─ vessel_quick.parquet # preferred if present
│ ├─ tt_minute.parquet # optional fallback
│ └─ vessel_minute.parquet # optional fallback
├─ outputs/ # auto-created
│ └─ (tables & figures will be saved here)
├─ (optional) data/NWB_roads/wegen_in_out.shp
└─ EDA.ipynb # this script
```

## 3) How to run

You can open and execute the notebook **EDA.ipynb** in any Jupyter-compatible environment:

### Option A — Jupyter Notebook / JupyterLab
```bash
jupyter notebook EDA.ipynb
```
or
```bash
jupyter lab EDA.ipynb
```