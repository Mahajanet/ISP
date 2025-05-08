import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# --- 1) Load & combine elevation-adjusted CARRA t2m NetCDF files ---
carra_files = sorted(glob("raw_data/elevation_adjusted/isa/t2m/t2m_isa_*.nc"))
if not carra_files:
    raise FileNotFoundError(
        "No NetCDF files found matching pattern: t2m_isa_*.nc in raw_data/elevation_adjusted/isa/t2m/"
    )

# open each year and concatenate along the time axis
carra_datasets = [xr.open_dataset(fp) for fp in carra_files]
carra_combined = xr.concat(carra_datasets, dim="time")

# flatten to 1-D, convert from K to °C, attach datetime index, and daily-mean
carra_time   = pd.to_datetime(carra_combined["time"].values)
t2m_K_flat   = carra_combined["t2m"].values.ravel()
t2m_C_flat   = t2m_K_flat - 273.15
carra_series = (
    pd.Series(t2m_C_flat, index=carra_time)
      .resample("D")
      .mean()
)

# --- 2) Load the in-situ sheet (Station 2642) and daily-mean its “T” column (already °C) ---
df_insitu       = (
    pd.read_excel(
        "raw_data/in_situ.xlsx",
        sheet_name="Observations - 2642",
        parse_dates=["TIMI"]
    )
    .set_index("TIMI")
)
in_situ_series = df_insitu["T"].dropna().resample("D").mean()

# --- 3) Align and drop any days missing in either series ---
aligned = pd.DataFrame({
    "CARRA_t2m_°C": carra_series,
    "In_Situ_°C":   in_situ_series
}).dropna()

# --- 4) Compute error metrics ---
mae  = mean_absolute_error(aligned["In_Situ_°C"], aligned["CARRA_t2m_°C"])
rmse = mean_squared_error(aligned["In_Situ_°C"], aligned["CARRA_t2m_°C"], squared=False)
corr = aligned["In_Situ_°C"].corr(aligned["CARRA_t2m_°C"])
bias = (aligned["CARRA_t2m_°C"] - aligned["In_Situ_°C"]).mean()

print("\n📊 Elevation-Adjusted CARRA vs In Situ (Station 2642) – 2 m Air Temp (°C)")
print(f"Mean Absolute Error (MAE):       {mae:.2f} °C")
print(f"Root Mean Squared Error (RMSE):  {rmse:.2f} °C")
print(f"Correlation Coefficient:         {corr:.2f}")
print(f"Bias (CARRA − In Situ):          {bias:.2f} °C")

# --- 5) Plot 1: Daily-mean time series (°C) ---
plt.figure(figsize=(15, 6))
plt.plot(aligned.index, aligned["CARRA_t2m_°C"], label="CARRA t2m (elev-adj)", alpha=0.7)
plt.plot(aligned.index, aligned["In_Situ_°C"],   label="In-Situ T (2642)", alpha=0.7)
plt.title("Daily Mean 2 m Temperature: Elev-Adjusted CARRA vs In Situ (Ísafjörður)")
plt.ylabel("Temperature (°C)")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 6) Plot 2: Scatter with 1:1 line (°C) ---
plt.figure(figsize=(6, 6))
plt.scatter(aligned["In_Situ_°C"], aligned["CARRA_t2m_°C"], alpha=0.5)
m = max(aligned.max())
plt.plot([0, m], [0, m], "r--", label="1:1 line")
plt.title("Scatter: Elev-Adj CARRA vs In Situ 2 m Temperature")
plt.xlabel("In Situ (°C)")
plt.ylabel("CARRA (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
