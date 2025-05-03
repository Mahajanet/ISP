import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# --- Load and combine CARRA precipitation NetCDF files ---
carra_files = sorted(glob("/Users/jahnavimahajan/Projects/ISP/raw_data/nn/precip_nn/precip_thver_nn/pr_thver_*.nc")) 
if not carra_files:
    raise FileNotFoundError("No NetCDF files found matching pattern: pr_thver_*.nc in precip_thver_nn/")
carra_datasets = [xr.open_dataset(fp) for fp in carra_files]
carra_combined = xr.concat(carra_datasets, dim='time')

# Convert CARRA to DataFrame and resample to daily totals
carra_time = pd.to_datetime(carra_combined['time'].values)
carra_df = pd.DataFrame({'pr': carra_combined['pr'].values}, index=carra_time)
carra_daily = carra_df['pr'].resample('D').sum()

# Create zero-filled in-situ data for same date range
in_situ_daily = pd.Series(0, index=carra_daily.index)

# --- Align both datasets ---
aligned = pd.DataFrame({
    'CARRA': carra_daily,
    'In_Situ': in_situ_daily
}).dropna()

# === Error Metrics ===
mae = mean_absolute_error(aligned['In_Situ'], aligned['CARRA'])
rmse = mean_squared_error(aligned['In_Situ'], aligned['CARRA'], squared=False)
correlation = aligned['In_Situ'].corr(aligned['CARRA'])  # will be NaN due to no variation
bias = (aligned['CARRA'] - aligned['In_Situ']).mean()

# Print results in terminal
print("\n📊 Statistical Summary (Þverfjall - Precipitation, No In-Situ Data):")
print(f"Mean Absolute Error (MAE): {mae:.2f} mm")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} mm")
print(f"Correlation Coefficient: {correlation if pd.notna(correlation) else 'N/A (constant zero baseline)'}")
print(f"Bias (CARRA - In Situ): {bias:.2f} mm")

# --- Plot 1: Daily Time Series ---
plt.figure(figsize=(15, 6))
plt.plot(aligned.index, aligned['CARRA'], label='CARRA (Nearest Neighbor)', alpha=0.7)
plt.plot(aligned.index, aligned['In_Situ'], label='In Situ (Assumed Zero)', alpha=0.7, linestyle='--')
plt.title("Daily Precipitation: CARRA vs Zero Baseline (Þverfjall)")
plt.ylabel("Precipitation (mm)")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 2: Metrics as Bar Chart ---
metrics = [mae, rmse, bias]
labels = ['MAE', 'RMSE', 'Bias']
colors = ['skyblue', 'salmon', 'violet']

plt.figure(figsize=(8, 5))
plt.bar(labels, metrics, color=colors)
plt.title("CARRA vs Zero Baseline - Statistical Comparison (Þverfjall Precip)")
plt.ylabel("Value")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# --- Plot 3: Scatter Plot ---
plt.figure(figsize=(6, 6))
plt.scatter(aligned['In_Situ'], aligned['CARRA'], alpha=0.5)
plt.title("Scatter: CARRA vs Zero Baseline Precipitation (Þverfjall)")
plt.xlabel("In Situ (0 mm)")
plt.ylabel("CARRA (mm)")
plt.grid(True)
plt.tight_layout()
plt.show()
