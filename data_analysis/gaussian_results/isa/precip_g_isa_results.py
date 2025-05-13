#!/usr/bin/env python3

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
GAUSS_PATTERN = "/Users/jahnavimahajan/Projects/ISP/raw_data/gaussian/isa/pr/isa_pr_*.nc"
EXCEL_PATH    = "/Users/jahnavimahajan/Projects/ISP/raw_data/in_situ.xlsx"
SHEET_NAME    = "Observations - 2642"  # your station’s sheet
LOGO          = "📊"

# --- Load & combine Gaussian‐smoothed NetCDF files for ISA precipitation ---
files = sorted(glob(GAUSS_PATTERN))
if not files:
    raise FileNotFoundError(f"No NetCDF found: {GAUSS_PATTERN}")
ds_list = [xr.open_dataset(fp) for fp in files]
gauss = xr.concat(ds_list, dim="time")

# Convert to pandas Series and resample daily sums
time = pd.to_datetime(gauss["time"].values)
gauss_pr = pd.Series(gauss["pr"].values, index=time).resample("D").sum()

# --- Load in situ Excel data ---
df_insitu = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
df_insitu["TIMI"] = pd.to_datetime(df_insitu["TIMI"])
df_insitu.set_index("TIMI", inplace=True)

# Assumes precip column is named 'R'
insitu_pr = df_insitu["R"].dropna().resample("D").sum()

# --- Align & drop any days missing in either series ---
df = pd.DataFrame({
    "Gaussian": gauss_pr,
    "In_Situ": insitu_pr
}).dropna()

# === Compute error metrics ===
mae         = mean_absolute_error(df["In_Situ"], df["Gaussian"])
rmse        = mean_squared_error(df["In_Situ"], df["Gaussian"], squared=False)
corr        = df["In_Situ"].corr(df["Gaussian"])
bias        = (df["Gaussian"] - df["In_Situ"]).mean()

print(f"\n{LOGO} Statistical Summary for ISA precipitation:")
print(f"  Mean Absolute Error (MAE):      {mae:.2f} mm")
print(f"  Root Mean Squared Error (RMSE): {rmse:.2f} mm")
print(f"  Correlation Coefficient:       {corr:.2f}")
print(f"  Bias (Gaussian – In Situ):     {bias:.2f} mm")

# --- Plot 1: Daily Time Series ---
plt.figure(figsize=(14,6))
plt.plot(df.index, df["Gaussian"], label="CARRA‐Gaussian (isa)", alpha=0.7)
plt.plot(df.index, df["In_Situ"], label="In Situ (2642)", alpha=0.7)
plt.title("Fig 3.3.1 Daily Precipitation: Gaussian‐smoothed CARRA vs In Situ (Ísafjörður)")
plt.ylabel("Precipitation (mm)")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 2: Scatter ---
plt.figure(figsize=(6,6))
plt.scatter(df["In_Situ"], df["Gaussian"], alpha=0.5)
m = max(df.max())
plt.plot([0, m], [0, m], "r--", label="1:1 line")
plt.title("Fig 3.3.2 Scatter: Gaussian‐smoothed CARRA vs In Situ Precipitation")
plt.xlabel("In Situ (mm)")
plt.ylabel("CARRA‐Gaussian (mm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
