import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- 1) Load & combine IDW‐interpolated CARRA 2 m temperature files for Þverá (Station 2636) --
idw_files = sorted(glob("raw_data/idw/thver/t2m/thver_t2m_*.nc"))
if not idw_files:
    raise FileNotFoundError(
        "No NetCDF files found matching pattern: thver_t2m_*.nc in raw_data/idw/thver/t2m/"
    )

# open each file and concatenate along the time axis
idw_dsets = [xr.open_dataset(fp) for fp in idw_files]
idw_combined = xr.concat(idw_dsets, dim="time")

# flatten to 1-D, convert from K → °C, attach datetime index, and daily-mean
times       = pd.to_datetime(idw_combined["time"].values)
t2m_K_flat  = idw_combined["t2m"].values.ravel()
t2m_C_flat  = t2m_K_flat - 273.15
idw_series  = pd.Series(t2m_C_flat, index=times).resample("D").mean()

# --- 2) Load the in-situ sheet (Station 2636 – Þverá) and daily-mean its “T” column (already °C) ---
df_insitu     = (
    pd.read_excel(
        "raw_data/in_situ.xlsx",
        sheet_name="Observations - 2636",
        parse_dates=["TIMI"]
    )
    .set_index("TIMI")
)
insitu_series = df_insitu["T"].dropna().resample("D").mean()

# --- 3) Align and drop any days missing in either series ---
aligned = pd.DataFrame({
    "IDW_t2m_°C":  idw_series,
    "In_Situ_°C":  insitu_series
}).dropna()

# --- 4) Compute error metrics ---
mae  = mean_absolute_error(aligned["In_Situ_°C"], aligned["IDW_t2m_°C"])
rmse = mean_squared_error(aligned["In_Situ_°C"], aligned["IDW_t2m_°C"], squared=False)
corr = aligned["In_Situ_°C"].corr(aligned["IDW_t2m_°C"])
bias = (aligned["IDW_t2m_°C"] - aligned["In_Situ_°C"]).mean()

print("\n📊 IDW-Interpolated CARRA vs In Situ (Station 2636 – Þverá) – 2 m Air Temp (°C)")
print(f"Mean Absolute Error (MAE):       {mae:.2f} °C")
print(f"Root Mean Squared Error (RMSE):  {rmse:.2f} °C")
print(f"Correlation Coefficient:         {corr:.2f}")
print(f"Bias (IDW − In Situ):            {bias:.2f} °C")

# --- 5) Plot 1: Daily-mean time series (°C) ---
plt.figure(figsize=(15, 6))
plt.plot(aligned.index, aligned["IDW_t2m_°C"],  label="IDW CARRA t2m", alpha=0.7)
plt.plot(aligned.index, aligned["In_Situ_°C"],  label="In Situ T (2636)", alpha=0.7)
plt.title("Daily Mean 2 m Temperature: IDW-Interpolated CARRA vs In Situ (Þverá)")
plt.ylabel("Temperature (°C)")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 6) Plot 2: Scatter with 1:1 line (°C) ---
plt.figure(figsize=(6, 6))
plt.scatter(aligned["In_Situ_°C"], aligned["IDW_t2m_°C"], alpha=0.5)
m = max(aligned["In_Situ_°C"].max(), aligned["IDW_t2m_°C"].max())
plt.plot([0, m], [0, m], "r--", label="1:1 line")
plt.title("Scatter: IDW CARRA vs In Situ 2 m Temperature")
plt.xlabel("In Situ (°C)")
plt.ylabel("IDW CARRA (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
