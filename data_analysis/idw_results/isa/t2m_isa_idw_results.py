import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# --- 1) Load & combine IDW‐interpolated CARRA T2M for Ísafjörður (isa) ---
idw_pattern = "raw_data/idw/isa/t2m/isa_t2m_t2m_day_*.nc"
idw_files = sorted(glob(idw_pattern))
if not idw_files:
    raise FileNotFoundError(f"No NetCDF files found matching pattern: {idw_pattern}")

# Load and concat datasets
idw_dsets = [xr.open_dataset(fp) for fp in idw_files]
idw_combined = xr.concat(idw_dsets, dim="time")

# Detect variable name for T2M
t2m_var = [v for v in idw_combined.data_vars if 't2m' in v.lower()][0]

# Convert time and extract temperature in °C, then resample to daily mean
times = pd.to_datetime(idw_combined["time"].values)
t2m_vals_c = idw_combined[t2m_var].values - 273.15
idw_daily = pd.Series(t2m_vals_c, index=times).resample("D").mean()

# --- 2) Load in‐situ temperature (Station 2642) from Excel ---
excel_path = "raw_data/in_situ.xlsx"
df_insitu = pd.read_excel(excel_path, sheet_name="Observations - 2642", parse_dates=["TIMI"])
df_insitu.set_index("TIMI", inplace=True)
in_situ_daily = df_insitu["T"].dropna().resample("D").mean()

# --- 3) Align both datasets ---
aligned = pd.DataFrame({
    "IDW": idw_daily,
    "In_Situ": in_situ_daily
}).dropna()

if aligned.empty:
    raise ValueError("Aligned dataset is empty. Check for overlapping dates or data issues.")

# --- 4) Compute error metrics ---
mae = mean_absolute_error(aligned["In_Situ"], aligned["IDW"])
rmse = mean_squared_error(aligned["In_Situ"], aligned["IDW"], squared=False)
correlation = aligned["In_Situ"].corr(aligned["IDW"])
bias = (aligned["IDW"] - aligned["In_Situ"]).mean()

print("\n📊 IDW‐Interpolated CARRA vs In Situ (Station 2642) – T2M")
print(f"Mean Absolute Error (MAE):       {mae:.2f} °C")
print(f"Root Mean Squared Error (RMSE):  {rmse:.2f} °C")
print(f"Correlation Coefficient:         {correlation:.2f}")
print(f"Bias (IDW − In Situ):            {bias:.2f} °C")

# --- 5) Plot 1: Daily time series comparison ---
plt.figure(figsize=(15, 6))
plt.plot(aligned.index, aligned["IDW"], label="CARRA (IDW)", alpha=0.7)
plt.plot(aligned.index, aligned["In_Situ"], label="In Situ (2642)", alpha=0.7)
plt.title("Daily 2m Temperature: IDW‐Interpolated CARRA vs In Situ (Ísafjörður)")
plt.ylabel("Temperature (°C)")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 6) Plot 2: Scatter with 1:1 line ---
plt.figure(figsize=(6, 6))
plt.scatter(aligned["In_Situ"], aligned["IDW"], alpha=0.5)
min_temp = min(aligned.min())
max_temp = max(aligned.max())
plt.plot([min_temp, max_temp], [min_temp, max_temp], "r--", label="1:1 line")
plt.title("Scatter: IDW‐Interpolated CARRA vs In Situ T2M (Ísafjörður)")
plt.xlabel("In Situ (°C)")
plt.ylabel("CARRA (IDW) (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
