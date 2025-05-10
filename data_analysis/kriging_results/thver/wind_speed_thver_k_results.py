#!/usr/bin/env python3

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- 1) Load & combine kriging 10 m wind‐speed (si10) files for Þverá grid cell ---
krig_files = sorted(glob("raw_data/kriging/thver/si10/si10_thver_F10m*_daily.nc"))
if not krig_files:
    raise FileNotFoundError("No kriging si10 files found for Þverá in raw_data/kriging/thver/si10/")

krig_dsets    = [xr.open_dataset(fp) for fp in krig_files]
krig_combined = xr.concat(krig_dsets, dim="time")

# --- 2) Flatten to 1-D and compute daily means ---
krig_time      = pd.to_datetime(krig_combined["time"].values)
ws_flat        = krig_combined["si10"].values.ravel()
krig_ws_daily  = pd.Series(ws_flat, index=krig_time).resample("D").mean()

# --- 3) Load in-situ Þverá (Station 2636) and daily-mean its “F” column ---
df_insitu      = (
    pd.read_excel(
        "raw_data/in_situ.xlsx",
        sheet_name="Observations - 2636",
        parse_dates=["TIMI"]
    )
    .set_index("TIMI")
)
insitu_ws_daily = df_insitu["F"].dropna().resample("D").mean()

# --- 4) Align & drop non-overlapping dates ---
aligned = pd.DataFrame({
    "Kriging": krig_ws_daily,
    "In_Situ": insitu_ws_daily
}).dropna()

# --- 5) Compute error metrics ---
mae  = mean_absolute_error(aligned["In_Situ"], aligned["Kriging"])
rmse = mean_squared_error(aligned["In_Situ"], aligned["Kriging"], squared=False)
corr = aligned["In_Situ"].corr(aligned["Kriging"])
bias = (aligned["Kriging"] - aligned["In_Situ"]).mean()

print("\n📊 Wind Speed Comparison (Þverá, Station 2636) – Kriging")
print(f"Mean Absolute Error (MAE):       {mae:.2f} m/s")
print(f"Root Mean Squared Error (RMSE):  {rmse:.2f} m/s")
print(f"Correlation Coefficient:         {corr:.2f}")
print(f"Bias (Kriging − In Situ):        {bias:.2f} m/s")

# --- 6) Plot 1: Daily-mean time series ---
plt.figure(figsize=(15, 6))
plt.plot(aligned.index, aligned["Kriging"], label="Kriging", alpha=0.7)
plt.plot(aligned.index, aligned["In_Situ"], label="In Situ", alpha=0.7)
plt.title("Daily Mean 10 m Wind Speed: Kriging vs In Situ (Þverá, Station 2636)")
plt.ylabel("Wind Speed (m/s)")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 7) Plot 2: Scatter with 1:1 line ---
plt.figure(figsize=(6, 6))
plt.scatter(aligned["In_Situ"], aligned["Kriging"], alpha=0.5)
m = aligned.values.max()
plt.plot([0, m], [0, m], "r--", label="1:1 line")
plt.title("Scatter: Kriging vs In Situ Wind Speed")
plt.xlabel("In Situ (m/s)")
plt.ylabel("Kriging (m/s)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
