import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

# --- 1) Load & combine Gaussian‐smoothed CARRA precip for Þverfjöll ---
gauss_files = sorted(glob("raw_data/gaussian/thver/pr/thver_pr_*.nc"))
if not gauss_files:
    raise FileNotFoundError("No Gaussian precip files found in raw_data/gaussian/thver/pr/")

datasets = [
    xr.open_dataset(fp)
      .drop_vars(["height", "latitude", "longitude"], errors="ignore")
    for fp in gauss_files
]
ds = xr.concat(datasets, dim="time", coords="minimal")

# Flatten singleton spatial dims and build a daily‐sum series
times       = pd.to_datetime(ds["time"].values)
precip_vals = ds["pr"].values.ravel()
gauss_daily = pd.Series(precip_vals, index=times).resample("D").sum()

# --- 2) Create a zero‐line “In Situ” series for Þverfjöll (no data) ---
in_situ_zero = pd.Series(0.0, index=gauss_daily.index)

# --- 3) Plot both series ---
plt.figure(figsize=(15,5))
plt.plot(
    gauss_daily.index, gauss_daily,
    label="Gaussian‐smoothed CARRA", lw=1, color="tab:blue"
)
plt.plot(
    in_situ_zero.index, in_situ_zero,
    label="In Situ (no data)", linestyle="--", color="tab:gray"
)

plt.title("Daily Precipitation: Gaussian CARRA vs In Situ (Þverá, Station 2636)")
plt.xlabel("Date")
plt.ylabel("Precipitation (mm)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
