#!/usr/bin/env python3

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import mean_squared_error

# 1) Load & combine kriging CARRA wind‐direction files for Ísafjörður
KRIGING_PATTERN = "raw_data/kriging/isa/wdir10/wdir10_isa_*_daily.nc"
files = sorted(glob(KRIGING_PATTERN))
if not files:
    raise FileNotFoundError(f"No kriging wind‐direction files found matching: {KRIGING_PATTERN}")

# drop any unwanted coords so concat only aligns on time
datasets = []
for fp in files:
    ds = xr.open_dataset(fp).drop_vars(["height", "latitude", "longitude"], errors="ignore")
    datasets.append(ds)

combined = xr.concat(datasets, dim="time", coords="minimal")

# 2) Build pandas Series of instantaneous wind‐direction and compute daily mean
times         = pd.to_datetime(combined["time"].values)
wind_dir_flat = combined["wdir10"].values.ravel()
carra_series  = pd.Series(wind_dir_flat, index=times)
carra_daily   = carra_series.resample("D").mean()

# 3) Load in‐situ station data (Station 2642) and daily‐mean its “D” column
df_insitu = (
    pd.read_excel(
        "raw_data/in_situ.xlsx",
        sheet_name="Observations - 2642",
        parse_dates=["TIMI"]
    )
    .set_index("TIMI")
)
in_situ_daily = df_insitu["D"].dropna().resample("D").mean()

# 4) Align the two series on dates present in both
aligned = pd.DataFrame({
    "Kriged CARRA": carra_daily,
    "In_Situ":      in_situ_daily
}).dropna()

if aligned.empty:
    raise RuntimeError("⚠️ No overlapping dates between kriged CARRA and in‐situ wind direction!")

# 5) Compute angular error metrics (signed in [–180,180), then abs)
diff_signed  = (aligned["Kriged CARRA"] - aligned["In_Situ"] + 180) % 360 - 180
angular_diff = np.abs(diff_signed)

mae  = angular_diff.mean()
rmse = np.sqrt(mean_squared_error(np.zeros_like(diff_signed), diff_signed))
bias = diff_signed.mean()

print("\n📊 Wind Direction Comparison (Ísafjörður, Kriging):")
print(f"  Mean Absolute Angular Error (MAE): {mae:.2f}°")
print(f"  Root Mean Squared Error (RMSE):    {rmse:.2f}°")
print(f"  Mean Bias (signed):                {bias:.2f}°")

# 6) Plot daily time series
plt.figure(figsize=(15, 6))
plt.plot(aligned.index, aligned["Kriged CARRA"], label="Kriged CARRA", alpha=0.7)
plt.plot(aligned.index, aligned["In_Situ"],      label="In Situ",      alpha=0.7)
plt.title("Fig 3.5.8 Daily Mean Wind Direction: Kriged CARRA vs In Situ (Ísafjörður)")
plt.ylabel("Wind Direction (°)")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 7) Scatter plot with 1:1 line
plt.figure(figsize=(6, 6))
plt.scatter(aligned["In_Situ"], aligned["Kriged CARRA"], alpha=0.5)
plt.plot([0, 360], [0, 360], "r--", label="1:1 line")
plt.xlim(0, 360); plt.ylim(0, 360)
plt.title("Scatter: Kriged CARRA vs In Situ Daily Wind Direction")
plt.xlabel("In Situ (°)")
plt.ylabel("Kriged CARRA (°)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
