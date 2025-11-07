# XGBoost Crowd Flow Models
## Introduction
This folder contains two Jupyter notebooks that implement XGBoost-based
models for predicting pedestrian flow in the SAIL 2025 LVMA dataset.

-   XGboost_single.ipynb --- trains and evaluates an XGBoost model for a
    single sensor.\
    Used to verify model performance and validate feature engineering
    logic.

-   XGboost_multiple.ipynb --- runs a looped training process across all
    sensors.\
    Each sensor model is automatically optimized and evaluated using
    Optuna.

Both notebooks are part of the crowd management prediction pipeline for
the SAIL 2025 Dashboard project.

------------------------------------------------------------------------

## How to Run
### 1. Environment Setup
Make sure your Python environment includes the following libraries:
``` bash
pip install pandas numpy xgboost optuna scikit-learn matplotlib seaborn
```
Or using Conda:
``` bash
conda install pandas numpy xgboost optuna scikit-learn matplotlib seaborn
```

------------------------------------------------------------------------

### 2. Data Placement
Place the required CSV files in the same directory as the notebooks:

    training_dataset_tomtom_vessel_ready.csv
    SAIL2025_LVMA_data_3min_20August-25August2025_flow.csv

------------------------------------------------------------------------

### 3. Run the Notebooks
Open Jupyter Notebook or VS Code and run:

-   For single-sensor model:

    ``` bash
    XGboost_single.ipynb
    ```

    Trains one sensor model and visualizes its prediction performance.

-   For multiple-sensor model:

    ``` bash
    XGboost_multiple.ipynb
    ```

    Automatically loops through all sensors, performs Optuna
    optimization, and saves performance summaries.

------------------------------------------------------------------------

### 4. Outputs
-   Model performance summaries (R², RMSE, MAE)
-   Visualizations of predicted vs. actual flows
-   Optional model parameter files (if saving enabled)

All outputs are saved in the same directory as the notebook unless
otherwise specified.
