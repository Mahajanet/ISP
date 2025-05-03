import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import mean_absolute_error
import numpy as np

# --- Load and combine CARRA wind direction files for Þverfjall ---
carra_files = sorted(glob("/Users/jahnavimahajan/Projects/ISP/raw_data/nn/wind_dir_nn/d10m_thver_nn/d10m_thver_*.nc")) 
if not carra_files:
    raise FileNotFoundError("No CARRA wind direction files found for Þverfjall")
datasets = [xr.open_dataset(fp) for fp in carra_files]
combined = xr.concat(datasets, dim='time')

# Convert to daily average wind direction
carra_time = pd.to_datetime(combined['time'].values)
carra_df = pd.DataFrame({'wind_dir': combined['d10m'].values}, index=carra_time)
carra_daily = carra_df['wind_dir'].resample('D').mean()

# --- Load in-situ Excel data for Þverfjall (Station 2636) ---
excel_path = "/Users/jahnavimahajan/Projects/ISP/raw_data/in_situ.xlsx"
df_in_situ = pd.read_excel(excel_path, sheet_name='Observations - 2636')
df_in_situ['TIMI'] = pd.to_datetime(df_in_situ['TIMI'])
df_in_situ.set_index('TIMI', inplace=True)
in_situ_daily = df_in_situ['D'].dropna().resample('D').mean()

# --- Align datasets ---
aligned = pd.DataFrame({'CARRA': carra_daily, 'In_Situ': in_situ_daily}).dropna()

# --- Compute angular error ---
angular_diff = np.abs(aligned['CARRA'] - aligned['In_Situ']) % 360
angular_diff = np.where(angular_diff > 180, 360 - angular_diff, angular_diff)
mae = angular_diff.mean()

print("\n📊 Wind Direction Comparison (Þverfjall):")
print(f"Mean Absolute Angular Error: {mae:.2f}°")

# --- Plotting ---
plt.figure(figsize=(15, 6))
plt.plot(aligned.index, aligned['CARRA'], label='CARRA', alpha=0.7)
plt.plot(aligned.index, aligned['In_Situ'], label='In Situ', alpha=0.7)
plt.title("Daily Wind Direction: CARRA vs In Situ (Þverfjall)")
plt.ylabel("Wind Direction (°)")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
