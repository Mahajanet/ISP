#!/usr/bin/env python3

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import numpy as np

# --- Load and combine Gaussian wind‐direction outputs for Þverá ---
gauss_files = sorted(glob(
    "/Users/jahnavimahajan/Projects/ISP/raw_data/gaussian/thver/wdir10/thver_wdir10_*.nc"
))
if not gauss_files:
    raise FileNotFoundError(
        "No Gaussian wind‐direction files found for Þverá"
    )
gauss_dsets = [xr.open_dataset(fp) for fp in gauss_files]
gauss_combined = xr.concat(gauss_dsets, dim="time")

# Convert to daily mean wind direction
gauss_time = pd.to_datetime(gauss_combined["time"].values)
gauss_df   = pd.DataFrame(
    {"Gaussian": gauss_combined["wdir10"].values},
    index=gauss_time
)
gauss_daily = gauss_df["Gaussian"].resample("D").mean()

# --- Load in‐situ Excel data for Ísafjörður (Station 2642) ---
#    (no Þverá in‐situ precipitation, so we still use sheet 2642)
excel_path = "/Users/jahnavimahajan/Projects/ISP/raw_data/in_situ.xlsx"
df_insitu  = pd.read_excel(
    excel_path,
    sheet_name="Observations - 2642",
    parse_dates=["TIMI"]
)
df_insitu.set_index("TIMI", inplace=True)
in_situ_daily = (
    df_insitu["D"]
    .dropna()
    .resample("D")
    .mean()
)

# --- Align the two series on common dates ---
aligned = pd.DataFrame({
    "Gaussian": gauss_daily,
    "In_Situ":  in_situ_daily
}).dropna()

if aligned.empty:
    raise ValueError("⚠️ No overlapping dates between Gaussian and in‐situ wind direction!")

# --- Compute mean absolute angular error ---
angular_diff = np.abs(aligned["Gaussian"] - aligned["In_Situ"]) % 360
angular_diff = np.where(
    angular_diff > 180,
    360 - angular_diff,
    angular_diff
)
mae = angular_diff.mean()

print("\n📊 Wind Direction Comparison (Þverá):")
print(f"Mean Absolute Angular Error: {mae:.2f}°")

# --- Plot daily time series ---
plt.figure(figsize=(15, 6))
plt.plot(aligned.index, aligned["Gaussian"], label="Gaussian", alpha=0.7)
plt.plot(aligned.index, aligned["In_Situ"],  label="In Situ",  alpha=0.7)
plt.title("Fig 3.3.9 Daily Wind Direction: Gaussian vs In Situ (Þverá, Station 2636)")
plt.ylabel("Wind Direction (°)")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
