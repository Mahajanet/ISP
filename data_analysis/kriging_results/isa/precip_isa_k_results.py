#!/usr/bin/env python3

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- 1) Load & combine kriging‐interpolated CARRA precipitation for Ísafjörður ---
KRIG_PATTERN = "raw_data/kriging/isa/pr/pr_isa_pr_daily_*.nc"
krig_files = sorted(glob(KRIG_PATTERN))
if not krig_files:
    raise FileNotFoundError(
        f"No NetCDF files found matching pattern: {KRIG_PATTERN}"
    )

# open each file and concatenate along time
datasets     = [xr.open_dataset(fp) for fp in krig_files]
krig_combined = xr.concat(datasets, dim="time")

# flatten to 1-D, attach datetime index, then daily‐sum
times      = pd.to_datetime(krig_combined["time"].values)
pr_flat    = krig_combined["pr"].values.ravel()
krig_daily = pd.Series(pr_flat, index=times).resample("D").sum()

# --- 2) Load in situ Excel data (Station 2642) and daily‐sum its “R” column ---
df_insitu    = (
    pd.read_excel(
        "raw_data/in_situ.xlsx",
        sheet_name="Observations - 2642",
        parse_dates=["TIMI"]
    )
    .set_index("TIMI")
)
insitu_daily = df_insitu["R"].dropna().resample("D").sum()

# --- 3) Align & drop any days missing in either series ---
aligned = pd.DataFrame({
    "Kriging": krig_daily,
    "In_Situ": insitu_daily
}).dropna()

# --- 4) Compute error metrics ---
mae         = mean_absolute_error(aligned["In_Situ"], aligned["Kriging"])
rmse        = mean_squared_error(aligned["In_Situ"], aligned["Kriging"], squared=False)
correlation = aligned["In_Situ"].corr(aligned["Kriging"])
bias        = (aligned["Kriging"] - aligned["In_Situ"]).mean()

print("\n📊 Kriging‐Interpolated CARRA vs In Situ (Station 2642) — Precipitation")
print(f"  MAE:       {mae:.2f} mm")
print(f"  RMSE:      {rmse:.2f} mm")
print(f"  Correlation: {correlation:.2f}")
print(f"  Bias:      {bias:.2f} mm")

# --- 5) Plot daily time series ---
plt.figure(figsize=(15, 6))
plt.plot(aligned.index, aligned["Kriging"], label="CARRA (kriging)", alpha=0.7)
plt.plot(aligned.index, aligned["In_Situ"], label="In Situ",       alpha=0.7)
plt.title("Fig 3.5.1 Daily Precipitation: Kriging CARRA vs In Situ (Ísafjörður)")
plt.ylabel("Precipitation (mm)")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 6) Plot scatter with 1:1 line ---
plt.figure(figsize=(6, 6))
plt.scatter(aligned["In_Situ"], aligned["Kriging"], alpha=0.5)
m = max(aligned.max())
plt.plot([0, m], [0, m], "r--", label="1:1 line")
plt.title("Fig 3.5.2 Scatter: Kriging CARRA vs In Situ Precipitation")
plt.xlabel("In Situ (mm)")
plt.ylabel("CARRA (kriging) (mm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
