import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# --- Load and combine CARRA NetCDF files ---
carra_files = sorted(glob("/Users/jahnavimahajan/Projects/ISP/Steph/precip_nn/precip_isa_nn/pr_isa_*.nc")) 
carra_datasets = [xr.open_dataset(fp) for fp in carra_files]
carra_combined = xr.concat(carra_datasets, dim='time')

# Convert CARRA to DataFrame and resample to daily totals
carra_time = pd.to_datetime(carra_combined['time'].values)
carra_df = pd.DataFrame({'pr': carra_combined['pr'].values}, index=carra_time)
carra_daily = carra_df['pr'].resample('D').sum()

# --- Load Excel in situ data (IMO format) ---
excel_path = "/Users/jahnavimahajan/Projects/ISP/Steph/Jahnavi_Weatherdata_IMO_20250428.xlsx"  
df_in_situ = pd.read_excel(excel_path, sheet_name='Observations - 2642')

# Convert to datetime and set index
df_in_situ['TIMI'] = pd.to_datetime(df_in_situ['TIMI'])
df_in_situ.set_index('TIMI', inplace=True)

# Extract and resample the R (precipitation) column to daily totals
in_situ_daily = df_in_situ['R'].dropna().resample('D').sum()

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
print("\n📊 Statistical Summary:")
print(f"Mean Absolute Error (MAE): {mae:.2f} mm")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} mm")
print(f"Correlation Coefficient: {correlation:.2f}")
print(f"Bias (CARRA - In Situ): {bias:.2f} mm")

# --- Plot 1: Daily Time Series ---
plt.figure(figsize=(15, 6))
plt.plot(aligned.index, aligned['CARRA'], label='CARRA (Nearest Neighbor)', alpha=0.7)
plt.plot(aligned.index, aligned['In_Situ'], label='In Situ (Station 2642)', alpha=0.7)
plt.title("Daily Precipitation: CARRA vs In Situ (Ísafjörður)")
plt.ylabel("Precipitation (mm)")
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
plt.title("CARRA vs In Situ - Statistical Comparison")
plt.ylabel("Value")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# --- Plot 3: Scatter Plot ---
plt.figure(figsize=(6, 6))
plt.scatter(aligned['In_Situ'], aligned['CARRA'], alpha=0.5)
max_val = max(aligned.max())
plt.plot([0, max_val], [0, max_val], 'r--', label="Perfect Agreement")
plt.title("Scatter: CARRA vs In Situ Precipitation")
plt.xlabel("In Situ (mm)")
plt.ylabel("CARRA (mm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
