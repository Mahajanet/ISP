import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- 1) Load & combine IDW‐interpolated CARRA 10 m wind‐speed (si10) for Þverá ---
idw_files = sorted(glob("raw_data/idw/thver/si10/thver_si10_*.nc"))
if not idw_files:
    raise FileNotFoundError(
        "No NetCDF files found matching pattern: thver_si10_*.nc in raw_data/idw/thver/si10/"
    )

# open each, concat on time
datasets = [xr.open_dataset(fp) for fp in idw_files]
combined = xr.concat(datasets, dim="time")

# flatten to 1-D, attach datetime index, compute daily means
times           = pd.to_datetime(combined["time"].values)
ws_flat         = combined["si10"].values.ravel()
idw_ws_daily    = pd.Series(ws_flat, index=times).resample("D").mean()

# --- 2) Load the in-situ sheet for Þverá (Station 2636) and daily-mean its “F” column ---
df_insitu       = (
    pd.read_excel(
        "raw_data/in_situ.xlsx",
        sheet_name="Observations - 2636",
        parse_dates=["TIMI"]
    )
    .set_index("TIMI")
)
insitu_ws_daily = df_insitu["F"].dropna().resample("D").mean()

# --- 3) Align & drop any days missing ---
aligned = pd.DataFrame({
    "IDW_CARRA": idw_ws_daily,
    "In_Situ":   insitu_ws_daily
}).dropna()

# --- 4) Compute error metrics ---
mae  = mean_absolute_error(aligned["In_Situ"], aligned["IDW_CARRA"])
rmse = mean_squared_error(aligned["In_Situ"], aligned["IDW_CARRA"], squared=False)
corr = aligned["In_Situ"].corr(aligned["IDW_CARRA"])
bias = (aligned["IDW_CARRA"] - aligned["In_Situ"]).mean()

print("\n📊 IDW‐Interpolated CARRA vs In Situ (Station 2636 – Þverá) – Wind Speed")
print(f"Mean Absolute Error (MAE):       {mae:.2f} m/s")
print(f"Root Mean Squared Error (RMSE):  {rmse:.2f} m/s")
print(f"Correlation Coefficient:         {corr:.2f}")
print(f"Bias (IDW_CARRA − In Situ):      {bias:.2f} m/s")

# --- 5) Plot 1: Daily-mean time series ---
plt.figure(figsize=(15, 6))
plt.plot(aligned.index, aligned["IDW_CARRA"], label="IDW CARRA u10", alpha=0.7)
plt.plot(aligned.index, aligned["In_Situ"],    label="In-Situ F (2636)", alpha=0.7)
plt.title("Daily Mean 10 m Wind Speed: IDW CARRA vs In Situ (Þverá, Station 2636)")
plt.ylabel("Wind Speed (m/s)")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 6) Plot 2: Scatter with 1:1 line ---
plt.figure(figsize=(6, 6))
plt.scatter(aligned["In_Situ"], aligned["IDW_CARRA"], alpha=0.5)
m = max(aligned.max())
plt.plot([0, m], [0, m], "r--", label="1:1 line")
plt.title("Scatter: IDW CARRA vs In Situ Wind Speed")
plt.xlabel("In Situ (m/s)")
plt.ylabel("IDW CARRA (m/s)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
