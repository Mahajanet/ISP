#!/usr/bin/env python3

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- 1) Load & combine kriging CARRA t2m NetCDF files for Ísafjörður ---
carra_files = sorted(glob("raw_data/kriging/isa/t2m/t2m_isa_t2m_day_ISL*.nc"))
if not carra_files:
    raise FileNotFoundError(
        "No NetCDF files found matching pattern: isa_t2m_*.nc in raw_data/kriging/isa/t2m/"
    )

# open each file and concatenate along the time axis
carra_datasets = [xr.open_dataset(fp) for fp in carra_files]
carra_combined = xr.concat(carra_datasets, dim="time")

# flatten to 1-D, convert from K to °C, attach datetime index, and daily-mean
carra_time   = pd.to_datetime(carra_combined["time"].values)
t2m_K_flat   = carra_combined["t2m"].values.ravel()
t2m_C_flat   = t2m_K_flat - 273.15
carra_series = (
    pd.Series(t2m_C_flat, index=carra_time)
      .resample("D")
      .mean()
)

# --- 2) Load the in-situ sheet (Station 2642) and daily-mean its “T” column (already °C) ---
df_insitu       = (
    pd.read_excel(
        "raw_data/in_situ.xlsx",
        sheet_name="Observations - 2642",
        parse_dates=["TIMI"]
    )
    .set_index("TIMI")
)
in_situ_series = df_insitu["T"].dropna().resample("D").mean()

# --- 3) Align and drop any days missing in either series ---
aligned = pd.DataFrame({
    "Kriged CARRA t2m (°C)": carra_series,
    "In Situ T (°C)":        in_situ_series
}).dropna()

# --- 4) Compute error metrics ---
mae  = mean_absolute_error(aligned["In Situ T (°C)"], aligned["Kriged CARRA t2m (°C)"])
rmse = mean_squared_error(aligned["In Situ T (°C)"], aligned["Kriged CARRA t2m (°C)"], squared=False)
corr = aligned["In Situ T (°C)"].corr(aligned["Kriged CARRA t2m (°C)"])
bias = (aligned["Kriged CARRA t2m (°C)"] - aligned["In Situ T (°C)"]).mean()

print("\n📊 Kriged‐CARRA vs In-Situ (Station 2642) – 2 m Air Temp (°C)")
print(f"Mean Absolute Error (MAE):       {mae:.2f} °C")
print(f"Root Mean Squared Error (RMSE):  {rmse:.2f} °C")
print(f"Correlation Coefficient:         {corr:.2f}")
print(f"Bias (Kriged − In Situ):         {bias:.2f} °C")

# --- 5) Plot 1: Daily-mean time series (°C) ---
plt.figure(figsize=(15, 6))
plt.plot(aligned.index, aligned["Kriged CARRA t2m (°C)"], label="Kriged CARRA t2m", alpha=0.7)
plt.plot(aligned.index, aligned["In Situ T (°C)"],        label="In-Situ T",        alpha=0.7)
plt.title("Fig 3.5.4 Daily Mean 2 m Temperature: Kriged CARRA vs In Situ (Ísafjörður)")
plt.ylabel("Temperature (°C)")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 6) Plot 2: Scatter with 1:1 line (°C) ---
plt.figure(figsize=(6, 6))
plt.scatter(
    aligned["In Situ T (°C)"],
    aligned["Kriged CARRA t2m (°C)"],
    alpha=0.5
)
m = max(aligned.max())
plt.plot([0, m], [0, m], "r--", label="1:1 line")
plt.title("Fig 3.5.5 Scatter: Kriged CARRA vs In Situ 2 m Temperature")
plt.xlabel("In Situ (°C)")
plt.ylabel("Kriged CARRA (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
