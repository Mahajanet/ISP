import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Load and combine CARRA wind speed files for Ísafjörður ---
carra_files = sorted(glob("/Users/jahnavimahajan/Projects/ISP/raw_data/nn/wind_speed_nn/f10m_isa_nn/f10m_isa_*.nc")) 
if not carra_files:
    raise FileNotFoundError("No CARRA wind speed files found for Ísafjörður")
datasets = [xr.open_dataset(fp) for fp in carra_files]
combined = xr.concat(datasets, dim='time')

# Convert to daily average wind speed
carra_time = pd.to_datetime(combined['time'].values)
carra_df = pd.DataFrame({'wind_speed': combined['f10m'].values}, index=carra_time)
carra_daily = carra_df['wind_speed'].resample('D').mean()

# --- Load in-situ Excel data for Ísafjörður (Station 2642) ---
excel_path = "/Users/jahnavimahajan/Projects/ISP/raw_data/in_situ.xlsx"
df_in_situ = pd.read_excel(excel_path, sheet_name='Observations - 2642')
df_in_situ['TIMI'] = pd.to_datetime(df_in_situ['TIMI'])
df_in_situ.set_index('TIMI', inplace=True)
in_situ_daily = df_in_situ['F'].dropna().resample('D').mean()

# --- Align datasets ---
aligned = pd.DataFrame({'CARRA': carra_daily, 'In_Situ': in_situ_daily}).dropna()

# --- Metrics ---
mae = mean_absolute_error(aligned['In_Situ'], aligned['CARRA'])
rmse = mean_squared_error(aligned['In_Situ'], aligned['CARRA'], squared=False)
correlation = aligned['In_Situ'].corr(aligned['CARRA'])
bias = (aligned['CARRA'] - aligned['In_Situ']).mean()

print("\n📊 Wind Speed Comparison (Ísafjörður):")
print(f"MAE: {mae:.2f} m/s")
print(f"RMSE: {rmse:.2f} m/s")
print(f"Correlation: {correlation:.2f}")
print(f"Bias: {bias:.2f} m/s")

# --- Plotting ---
plt.figure(figsize=(15, 6))
plt.plot(aligned.index, aligned['CARRA'], label='CARRA', alpha=0.7)
plt.plot(aligned.index, aligned['In_Situ'], label='In Situ', alpha=0.7)
plt.title("Fig 3.1.10 Daily Wind Speed: CARRA vs In Situ (Ísafjörður)")
plt.ylabel("Wind Speed (m/s)")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
