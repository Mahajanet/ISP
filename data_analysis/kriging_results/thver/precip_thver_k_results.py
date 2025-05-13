import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

# --- 1) Load & combine kriging‐interpolated CARRA precipitation for Þverá (Station 2636) ---
krig_files = sorted(glob("raw_data/kriging/thver/pr/pr_thver_pr_daily_*.nc"))
if not krig_files:
    raise FileNotFoundError(
        "No kriging precipitation files found in raw_data/kriging/thver/pr/"
    )

# Open each file, drop any extra coords so concat aligns on time only
datasets = [
    xr.open_dataset(fp)
      .drop_vars(["height", "latitude", "longitude"], errors="ignore")
    for fp in krig_files
]
ds = xr.concat(datasets, dim="time", coords="minimal")

# Flatten the singleton spatial dimension and build a daily‐sum series
times       = pd.to_datetime(ds["time"].values)
precip_vals = ds["pr"].values.ravel()
krig_daily  = pd.Series(precip_vals, index=times).resample("D").sum()

# --- 2) Create a zero‐line “In Situ” series ---
#     (Þverá has no in‐situ precipitation data, so we plot zeros)
in_situ_zero = pd.Series(0.0, index=krig_daily.index)

# --- 3) Plot both series ---
plt.figure(figsize=(15,5))
plt.plot(
    krig_daily.index,
    krig_daily,
    label="Kriging CARRA (thver)",
    color="tab:green",
    lw=1
)
plt.plot(
    in_situ_zero.index,
    in_situ_zero,
    label="In Situ (no data)",
    linestyle="--",
    color="tab:gray"
)

plt.title("Fig 3.5.3 Daily Precipitation: Kriging CARRA vs In Situ (Þverá, Station 2636)")
plt.xlabel("Date")
plt.ylabel("Precipitation (mm)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
