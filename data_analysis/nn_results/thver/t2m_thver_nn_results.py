import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# --- Load and combine CARRA T2M NetCDF files ---
carra_files = sorted(glob("/Users/jahnavimahajan/Projects/ISP/raw_data/nn/t2m_nn/t2m_thver_nn/t2m_thver_*.nc")) 
if not carra_files:
    raise FileNotFoundError("No NetCDF files found matching pattern: t2m_thver_*.nc in t2m_thver_nn/")
carra_datasets = [xr.open_dataset(fp) for fp in carra_files]
carra_combined = xr.concat(carra_datasets, dim='time')

# Convert CARRA to DataFrame and resample to daily mean (convert from K to °C)
carra_time = pd.to_datetime(carra_combined['time'].values)
carra_df = pd.DataFrame({'t2m': carra_combined['t2m'].values - 273.15}, index=carra_time)
carra_daily = carra_df['t2m'].resample('D').mean()

# --- Load Excel in situ data (IMO format) ---
excel_path = "/Users/jahnavimahajan/Projects/ISP/raw_data/in_situ.xlsx"  
df_in_situ = pd.read_excel(excel_path, sheet_name='Observations - 2636')

# Convert to datetime and set index
df_in_situ['TIMI'] = pd.to_datetime(df_in_situ['TIMI'])
df_in_situ.set_index('TIMI', inplace=True)

# Extract and resample the T (temperature in °C) column to daily mean
in_situ_daily = df_in_situ['T'].dropna().resample('D').mean()

# --- Align both datasets ---
aligned = pd.DataFrame({
    'CARRA': carra_daily,
    'In_Situ': in_situ_daily
}).dropna()

# === Error Metrics ===
mae = mean_absolute_error(aligned['In_Situ'], aligned['CARRA'])
rmse = mean_squared_error(aligned['In_Situ'], aligned['CARRA'], squared=False)
correlation = aligned['In_Situ'].corr(aligned['CARRA'])
bias = (aligned['CARRA'] - aligned['In_Situ']).mean()

# Print results in terminal
print("\n📊 Statistical Summary (Þverfjall):")
print(f"Mean Absolute Error (MAE): {mae:.2f} °C")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} °C")
print(f"Correlation Coefficient: {correlation:.2f}")
print(f"Bias (CARRA - In Situ): {bias:.2f} °C")

# --- Plot 1: Daily Time Series ---
plt.figure(figsize=(15, 6))
plt.plot(aligned.index, aligned['CARRA'], label='CARRA (Nearest Neighbor)', alpha=0.7)
plt.plot(aligned.index, aligned['In_Situ'], label='In Situ (Station 2615)', alpha=0.7)
plt.title("Daily 2m Temperature: CARRA vs In Situ (Þverfjall)")
plt.ylabel("Temperature (°C)")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 2: Metrics as Bar Chart ---
metrics = [mae, rmse, correlation, bias]
labels = ['MAE', 'RMSE', 'Correlation', 'Bias']
colors = ['skyblue', 'salmon', 'lightgreen', 'violet']

plt.figure(figsize=(8, 5))
plt.bar(labels, metrics, color=colors)
plt.title("CARRA vs In Situ - Temperature Statistical Comparison (Þverfjall)")
plt.ylabel("Value")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# --- Plot 3: Scatter Plot ---
plt.figure(figsize=(6, 6))
plt.scatter(aligned['In_Situ'], aligned['CARRA'], alpha=0.5)
min_temp = min(aligned.min())
max_temp = max(aligned.max())
plt.plot([min_temp, max_temp], [min_temp, max_temp], 'r--', label="Perfect Agreement")
plt.title("Scatter: CARRA vs In Situ Temperature (Þverfjall)")
plt.xlabel("In Situ (°C)")
plt.ylabel("CARRA (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
