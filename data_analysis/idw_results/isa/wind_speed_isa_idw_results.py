#!/usr/bin/env python3

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# --- 1) Load & combine IDW‐interpolated CARRA 10 m wind‐speed files for Ísafjörður ---
idw_files = sorted(glob("raw_data/idw/isa/si10/isa_si10_*.nc"))
if not idw_files:
    raise FileNotFoundError(
        "No NetCDF files found matching pattern: isa_si10_*.nc in raw_data/idw/isa/si10/"
    )

# open each file and concatenate along the time axis
datasets = [xr.open_dataset(fp) for fp in idw_files]
combined = xr.concat(datasets, dim="time")

# flatten the (time,1) array to 1-D, attach a datetime index, and daily-mean
times        = pd.to_datetime(combined["time"].values)
ws_flat      = combined["si10"].values.ravel()
idw_ws_daily = pd.Series(ws_flat, index=times).resample("D").mean()

# --- 2) Load the in-situ sheet (Station 2642) and daily-mean its “F” column ---
df_insitu       = (
    pd.read_excel(
        "raw_data/in_situ.xlsx",
        sheet_name="Observations - 2642",
        parse_dates=["TIMI"]
    )
    .set_index("TIMI")
)
insitu_ws_daily = df_insitu["F"].dropna().resample("D").mean()

# --- 3) Align and drop any days missing in either series ---
aligned = pd.DataFrame({
    "IDW_CARRA": idw_ws_daily,
    "In_Situ":   insitu_ws_daily
}).dropna()

# --- 4) Compute error metrics ---
mae  = mean_absolute_error(aligned["In_Situ"], aligned["IDW_CARRA"])
rmse = mean_squared_error(aligned["In_Situ"], aligned["IDW_CARRA"], squared=False)
corr = aligned["In_Situ"].corr(aligned["IDW_CARRA"])
bias = (aligned["IDW_CARRA"] - aligned["In_Situ"]).mean()

print("\n📊 IDW‐Interpolated CARRA vs In Situ (Station 2642) – 10 m Wind Speed")
print(f"Mean Absolute Error (MAE):       {mae:.2f} m/s")
print(f"Root Mean Squared Error (RMSE):  {rmse:.2f} m/s")
print(f"Correlation Coefficient:         {corr:.2f}")
print(f"Bias (IDW_CARRA − In Situ):      {bias:.2f} m/s")

# --- 5) Plot 1: Daily-mean time series ---
plt.figure(figsize=(15, 6))
plt.plot(aligned.index, aligned["IDW_CARRA"], label="IDW CARRA u10", alpha=0.7)
plt.plot(aligned.index, aligned["In_Situ"],   label="In Situ F",     alpha=0.7)
plt.title("Daily Mean 10 m Wind Speed: IDW‐Interpolated CARRA vs In Situ (Ísafjörður)")
plt.ylabel("Wind Speed (m/s)")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 6) Plot 2: Scatter with 1:1 line ---
plt.figure(figsize=(6, 6))
plt.scatter(aligned["In_Situ"], aligned["IDW_CARRA"], alpha=0.5)
m = max(aligned.max())
plt.plot([0, m], [0, m], "r--", label="1:1 line")
plt.title("Scatter: IDW CARRA vs In Situ 10 m Wind Speed")
plt.xlabel("In Situ (m/s)")
plt.ylabel("IDW CARRA (m/s)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
