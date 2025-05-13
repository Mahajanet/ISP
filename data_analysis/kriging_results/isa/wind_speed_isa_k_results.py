import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# --- 1) Load & combine kriging‐interpolated CARRA 10 m wind‐speed (si10) files for Ísafjörður ---
krig_files = sorted(glob("raw_data/kriging/isa/si10/si10_isa_F10m*_daily.nc"))
if not krig_files:
    raise FileNotFoundError(
        "No NetCDF files found matching pattern: isa_si10_*_daily.nc in raw_data/kriging/isa/si10/"
    )

# open each year and concatenate along the time axis
krig_dsets   = [xr.open_dataset(fp) for fp in krig_files]
krig_combined = xr.concat(krig_dsets, dim="time")

# flatten the (time,1) array to 1-D, attach a datetime index, and daily-mean
krig_time      = pd.to_datetime(krig_combined["time"].values)
ws_flat        = krig_combined["si10"].values.ravel()
krig_ws_daily  = pd.Series(ws_flat, index=krig_time).resample("D").mean()

# --- 2) Load the in-situ sheet (Station 2642) and daily-mean its “F” column (10 min wind speed) ---
df_insitu      = (
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
    "Kriging_u10": krig_ws_daily,
    "In_Situ":     insitu_ws_daily
}).dropna()

# --- 4) Compute error metrics ---
mae  = mean_absolute_error(aligned["In_Situ"], aligned["Kriging_u10"])
rmse = mean_squared_error(aligned["In_Situ"], aligned["Kriging_u10"], squared=False)
corr = aligned["In_Situ"].corr(aligned["Kriging_u10"])
bias = (aligned["Kriging_u10"] - aligned["In_Situ"]).mean()

print("\n📊 Kriging-Interpolated CARRA vs In Situ (Station 2642) – 10 m Wind Speed")
print(f"Mean Absolute Error (MAE):       {mae:.2f} m/s")
print(f"Root Mean Squared Error (RMSE):  {rmse:.2f} m/s")
print(f"Correlation Coefficient:         {corr:.2f}")
print(f"Bias (Kriging_u10 − In Situ):    {bias:.2f} m/s")

# --- 5) Plot 1: Daily-mean time series ---
plt.figure(figsize=(15, 6))
plt.plot(aligned.index, aligned["Kriging_u10"], label="Kriging u10", alpha=0.7)
plt.plot(aligned.index, aligned["In_Situ"],     label="In-Situ F (2642)", alpha=0.7)
plt.title("Fig 3.5.10 Daily Mean 10 m Wind Speed: Kriging vs In Situ (Ísafjörður)")
plt.ylabel("Wind Speed (m/s)")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 6) Plot 2: Scatter with 1:1 line ---
plt.figure(figsize=(6, 6))
plt.scatter(aligned["In_Situ"], aligned["Kriging_u10"], alpha=0.5)
m = aligned.max().max()
plt.plot([0, m], [0, m], "r--", label="1:1 line")
plt.title("Scatter: Kriging vs In Situ Wind Speed")
plt.xlabel("In Situ (m/s)")
plt.ylabel("Kriging (m/s)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
