import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

# --- 1) Load & combine IDW‐interpolated CARRA precip for Þverá grid cell ---
idw_files = sorted(glob("raw_data/idw/thver/pr/thver_pr_*.nc"))
if not idw_files:
    raise FileNotFoundError("No IDW precip files found in raw_data/idw/thver/pr/")

datasets = [
    xr.open_dataset(fp)
      .drop_vars(["height", "latitude", "longitude"], errors="ignore")
    for fp in idw_files
]
ds = xr.concat(datasets, dim="time", coords="minimal")

# flatten singleton spatial dim and build a daily‐sum series
times       = pd.to_datetime(ds["time"].values)
precip_vals = ds["pr"].values.ravel()
idw_daily   = pd.Series(precip_vals, index=times).resample("D").sum()

# --- 2) Create a zero‐line “In Situ” series ---
#    Same index, all zeros, since no in-situ precipitation data for Þverá
in_situ_zero = pd.Series(0.0, index=idw_daily.index)

# --- 3) Plot both series ---
plt.figure(figsize=(15,5))
plt.plot(idw_daily.index,    idw_daily,    label="IDW CARRA (thver)", color="tab:orange", lw=1)
plt.plot(in_situ_zero.index, in_situ_zero, label="In Situ (no data)", linestyle="--", color="tab:gray")

plt.title("Daily Precipitation: IDW‐Interpolated CARRA vs In Situ (Þverá, Station 2636)")
plt.xlabel("Date")
plt.ylabel("Precipitation (mm)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
