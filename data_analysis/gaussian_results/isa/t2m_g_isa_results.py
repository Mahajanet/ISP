#!/usr/bin/env python3

import sys
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- 1) LOAD GAUSSIAN OUTPUT (thver, t2m) ---

GAUSS_PATTERN = (
    "/Users/jahnavimahajan/Projects/ISP/raw_data/gaussian"
    "/thver/t2m/thver_t2m_*.nc"
)

gauss_files = sorted(glob(GAUSS_PATTERN))
if not gauss_files:
    raise FileNotFoundError(f"No NetCDFs found: {GAUSS_PATTERN}")

# open & concat along time
gauss_ds = xr.concat([xr.open_dataset(f) for f in gauss_files], dim="time")

# pull out the t2m series (Kelvin → °C)
gauss_time   = pd.to_datetime(gauss_ds["time"].values)
gauss_values = gauss_ds["t2m"].values - 273.15
gauss_series = pd.Series(gauss_values, index=gauss_time)

# resample to daily mean and normalize to midnight
gauss_daily = gauss_series.resample("D").mean()
gauss_daily.index = gauss_daily.index.normalize()

# --- 2) LOAD IN-SITU EXCEL (Station 2642) ---

EXCEL_PATH = "/Users/jahnavimahajan/Projects/ISP/raw_data/in_situ.xlsx"
SHEET_NAME = "Observations - 2642"

df_insitu = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
df_insitu["TIMI"] = pd.to_datetime(df_insitu["TIMI"])
df_insitu.set_index("TIMI", inplace=True)

# in-situ T is already in °C
insitu_daily = df_insitu["T"].dropna().resample("D").mean()
insitu_daily.index = insitu_daily.index.normalize()

# --- 3) ALIGN & CHECK OVERLAP ---

aligned = pd.DataFrame({
    "Gaussian": gauss_daily,
    "In_Situ":  insitu_daily
}).dropna()

if aligned.empty:
    print("⚠️  No overlapping dates between Gaussian and In–Situ!")
    print(f"   Gaussian covers {gauss_daily.index.min()} → {gauss_daily.index.max()}")
    print(f"   In–Situ  covers {insitu_daily.index.min()} → {insitu_daily.index.max()}")
    sys.exit(1)

# --- 4) ERROR METRICS ---

mae         = mean_absolute_error(aligned["In_Situ"], aligned["Gaussian"])
rmse        = mean_squared_error(aligned["In_Situ"], aligned["Gaussian"], squared=False)
correlation = aligned["In_Situ"].corr(aligned["Gaussian"])
bias        = (aligned["Gaussian"] - aligned["In_Situ"]).mean()

print("\n📊 Statistical Summary:")
print(f"  Mean Absolute Error (MAE):     {mae:.2f} °C")
print(f"  Root Mean Squared Error (RMSE): {rmse:.2f} °C")
print(f"  Correlation Coefficient:        {correlation:.2f}")
print(f"  Bias (Gaussian − In Situ):       {bias:.2f} °C\n")

# --- 5) PLOTTING ---

# Time series
plt.figure(figsize=(14,5))
plt.plot(aligned.index, aligned["Gaussian"],
         label="Gaussian (3×3 interp)", alpha=0.7)
plt.plot(aligned.index, aligned["In_Situ"],
         label="In Situ (Station 2642)", alpha=0.7)
plt.title("Fig 3.3.4 Daily 2 m Temperature: Gaussian vs In Situ (Þverá, Station 2636)")
plt.ylabel("Temperature (°C)")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Scatter
plt.figure(figsize=(6,6))
plt.scatter(aligned["In_Situ"], aligned["Gaussian"], alpha=0.5)
mn, mx = aligned.min().min(), aligned.max().max()
plt.plot([mn, mx], [mn, mx], "r--", label="Perfect Agreement")
plt.title("Fig 3.3.5 Scatter: Gaussian vs In Situ Temperature (Þverá)")
plt.xlabel("In Situ (°C)")
plt.ylabel("Gaussian (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
