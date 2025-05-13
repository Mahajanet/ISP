import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# --- 1) Load & combine elevation‐adjusted CARRA 10 m wind‐speed (u10) files for Ísafjörður ---
carra_files = sorted(glob("raw_data/elevation_adjusted/isa/si10/si10_isa_*.nc"))
if not carra_files:
    raise FileNotFoundError(
        "No NetCDF files found matching pattern: u10_isa_*.nc in raw_data/elevation_adjusted/isa/u10/"
    )

# open each year and concatenate along the time axis
carra_datasets = [xr.open_dataset(fp) for fp in carra_files]
carra_combined = xr.concat(carra_datasets, dim="time")

# flatten the (time,1) array to 1-D, attach a datetime index, and daily-mean
carra_time     = pd.to_datetime(carra_combined["time"].values)
u10_flat       = carra_combined["10si"].values.ravel()
carra_ws_daily = pd.Series(u10_flat, index=carra_time).resample("D").mean()

# --- 2) Load the in-situ sheet (Station 2642) and daily-mean its “F” column (10 min wind speed) ---
df_insitu     = (
    pd.read_excel(
        "raw_data/in_situ.xlsx",
        sheet_name="Observations - 2642",
        parse_dates=["TIMI"]
    )
    .set_index("TIMI")
)
insitu_ws_daily = df_insitu["F"].dropna().resample("D").mean()

# --- 3) Align and drop any days missing in either series ---
aligned = pd.DataFrame({
    "CARRA_u10": carra_ws_daily,
    "In_Situ":   insitu_ws_daily
}).dropna()

# --- 4) Compute error metrics ---
mae  = mean_absolute_error(aligned["In_Situ"], aligned["CARRA_u10"])
rmse = mean_squared_error(aligned["In_Situ"], aligned["CARRA_u10"], squared=False)
corr = aligned["In_Situ"].corr(aligned["CARRA_u10"])
bias = (aligned["CARRA_u10"] - aligned["In_Situ"]).mean()

print("\n📊 Elevation-Adjusted CARRA vs In Situ (Station 2642) – Wind Speed")
print(f"Mean Absolute Error (MAE):       {mae:.2f} m/s")
print(f"Root Mean Squared Error (RMSE):  {rmse:.2f} m/s")
print(f"Correlation Coefficient:         {corr:.2f}")
print(f"Bias (CARRA_u10 − In Situ):      {bias:.2f} m/s")

# --- 5) Plot 1: Daily-mean time series ---
plt.figure(figsize=(15, 6))
plt.plot(aligned.index, aligned["CARRA_u10"], label="CARRA u10 (elev-adj)", alpha=0.7)
plt.plot(aligned.index, aligned["In_Situ"],    label="In-Situ F (Station 2642)", alpha=0.7)
plt.title("Fig 3.4.10 Daily Mean 10 m Wind Speed: Elev-Adjusted CARRA vs In Situ (Ísafjörður)")
plt.ylabel("Wind Speed (m/s)")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 6) Plot 2: Scatter with 1:1 line ---
plt.figure(figsize=(6, 6))
plt.scatter(aligned["In_Situ"], aligned["CARRA_u10"], alpha=0.5)
m = max(aligned.max())
plt.plot([0, m], [0, m], "r--", label="1:1 line")
plt.title("Scatter: Elev-Adjusted CARRA vs In Situ Wind Speed")
plt.xlabel("In Situ (m/s)")
plt.ylabel("CARRA (m/s)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
