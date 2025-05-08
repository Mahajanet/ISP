#!/usr/bin/env python3

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# --- Load and combine Gaussian wind speed files for Ísafjörður ---
gauss_files = sorted(glob(
    "/Users/jahnavimahajan/Projects/ISP/raw_data/gaussian/isa/si10/isa_si10_*.nc"
))
if not gauss_files:
    raise FileNotFoundError("No Gaussian wind speed files found for Ísafjörður")
gauss_dsets   = [xr.open_dataset(fp) for fp in gauss_files]
gauss_combined = xr.concat(gauss_dsets, dim="time")

# Convert to daily average wind speed
gauss_time   = pd.to_datetime(gauss_combined["time"].values)
gauss_df     = pd.DataFrame(
    {"Gaussian": gauss_combined["si10"].values},
    index=gauss_time
)
gauss_daily  = gauss_df["Gaussian"].resample("D").mean()

# --- Load in-situ Excel data for Ísafjörður (Station 2642) ---
excel_path   = "/Users/jahnavimahajan/Projects/ISP/raw_data/in_situ.xlsx"
df_insitu    = pd.read_excel(
    excel_path,
    sheet_name="Observations - 2642",
    parse_dates=["TIMI"]
)
df_insitu.set_index("TIMI", inplace=True)
in_situ_daily = (
    df_insitu["F"]
    .dropna()
    .resample("D")
    .mean()
)

# --- Align datasets on common dates ---
aligned = pd.DataFrame({
    "Gaussian": gauss_daily,
    "In_Situ":  in_situ_daily
}).dropna()

if aligned.empty:
    raise ValueError("⚠️ No overlapping dates between Gaussian and in-situ wind speed!")

# --- Compute error metrics ---
mae         = mean_absolute_error(aligned["In_Situ"], aligned["Gaussian"])
rmse        = mean_squared_error(aligned["In_Situ"], aligned["Gaussian"], squared=False)
correlation = aligned["In_Situ"].corr(aligned["Gaussian"])
bias        = (aligned["Gaussian"] - aligned["In_Situ"]).mean()

print("\n📊 Wind Speed Comparison (Ísafjörður):")
print(f"Mean Absolute Error (MAE): {mae:.2f} m/s")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} m/s")
print(f"Correlation Coefficient: {correlation:.2f}")
print(f"Bias (Gaussian - In Situ): {bias:.2f} m/s")

# --- Plot daily time series ---
plt.figure(figsize=(15, 6))
plt.plot(aligned.index, aligned["Gaussian"], label="Gaussian", alpha=0.7)
plt.plot(aligned.index, aligned["In_Situ"],  label="In Situ",  alpha=0.7)
plt.title("Daily Wind Speed: Gaussian vs In Situ (Ísafjörður)")
plt.ylabel("Wind Speed (m/s)")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
