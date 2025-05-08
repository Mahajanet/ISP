import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

# --- 1) Load & combine elevation-adjusted CARRA precip for Þverá grid cell ---
carra_files = sorted(glob("raw_data/elevation_adjusted/isa/pr/pr_isa_*.nc"))
if not carra_files:
    raise FileNotFoundError("No CARRA precip files found in elevation_adjusted/isa/pr/")

datasets = [
    xr.open_dataset(fp)
      .drop_vars(["height", "latitude", "longitude"], errors="ignore")
    for fp in carra_files
]
ds = xr.concat(datasets, dim="time", coords="minimal")

# flatten singleton spatial dim and build a daily‐sum series
times       = pd.to_datetime(ds["time"].values)
precip_vals = ds["pr"].values.ravel()
carra_daily = pd.Series(precip_vals, index=times).resample("D").sum()

# --- 2) Create a zero‐line “In Situ” series ---
#    Same index, all zeros, to indicate no in-situ precipitation data
in_situ_zero = pd.Series(0.0, index=carra_daily.index)

# --- 3) Plot both series ---
plt.figure(figsize=(15,5))
plt.plot(carra_daily.index,   carra_daily,   label="CARRA (elev-adj)", color="tab:blue", lw=1)
plt.plot(in_situ_zero.index,  in_situ_zero,  label="In Situ (no data)", linestyle="--", color="tab:gray")

plt.title("Daily Precipitation: Elev-Adjusted CARRA vs In Situ (Þverá, Station 2636)")
plt.xlabel("Date")
plt.ylabel("Precipitation (mm)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
