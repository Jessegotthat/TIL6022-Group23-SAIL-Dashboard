# TIL6022-Group23-SAIL-Dashboard
Group Assignment Repository

Brief description: SAIL Amsterdam 2025 is expected to attract millions of visitors from the Netherlands and abroad, making it one of the largest public events in Europe. With such high visitor numbers, effective crowd management is essential to ensure safety and prevent congestion-related risks. This project focuses on developing a crowd management dashboard prototype that can monitor and forecast short-term crowd flows in specific areas of the event. By combining real-time data, predictive models, and interactive map visualizations, the dashboard aims to support event managers in anticipating safety risks, responding to anomalies, and making data-driven decisions. Ultimately, the system will also serve as a reference tool for preventing undesirable incidents in future large-scale events.

Steps of Data Analysis
1. Data Preprocessing:
    o	Clean missing values, align time intervals (3-minute resolution).
    o	Aggregate and smooth data to reveal trends.
2. Exploratory Analysis:
    o	Identify temporal patterns (daily peaks, event-related surges).
    o	Detect variability across locations.
3.	Forecasting Approaches:
    o	Baseline models: Moving Average, Autoregressive (AR).
    o	Online correction: Trend correction, sliding window retraining.
    o	Uncertainty representation: Mean ± standard deviation to generate forecast intervals.
4.	Evaluation Metrics:
    o	RMSE/MAE for overall prediction accuracy.
    o	Peak timing error (difference between predicted vs. observed peak).
    o	Anomaly count (flagging unrealistic predictions, e.g., sudden extreme values).
