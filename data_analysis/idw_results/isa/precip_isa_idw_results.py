import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# --- 1) Load & combine IDW‐interpolated CARRA precipitation for Ísafjörður (isa) ---
idw_pattern = "raw_data/idw/isa/pr/isa_pr_*.nc"
idw_files   = sorted(glob(idw_pattern))
if not idw_files:
    raise FileNotFoundError(f"No NetCDF files found matching pattern: {idw_pattern}")

# open each file and concatenate along the time axis
idw_dsets = [xr.open_dataset(fp) for fp in idw_files]
idw_combined = xr.concat(idw_dsets, dim="time")

# flatten to pandas Series and resample to daily totals
times        = pd.to_datetime(idw_combined["time"].values)
precip_vals  = idw_combined["pr"].values
idw_daily    = pd.Series(precip_vals, index=times).resample("D").sum()

# --- 2) Load in‐situ precipitation (Station 2642) from Excel ---
excel_path   = "raw_data/in_situ.xlsx"
sheet_name   = "Observations - 2642"
df_insitu    = pd.read_excel(excel_path, sheet_name=sheet_name, parse_dates=["TIMI"])
df_insitu.set_index("TIMI", inplace=True)
in_situ_daily = df_insitu["R"].dropna().resample("D").sum()

# --- 3) Align the two series on their common dates ---
aligned = pd.DataFrame({
    "IDW":      idw_daily,
    "In_Situ":  in_situ_daily
}).dropna()

# --- 4) Compute error metrics ---
mae         = mean_absolute_error(aligned["In_Situ"], aligned["IDW"])
rmse        = mean_squared_error(aligned["In_Situ"], aligned["IDW"], squared=False)
correlation = aligned["In_Situ"].corr(aligned["IDW"])
bias        = (aligned["IDW"] - aligned["In_Situ"]).mean()

print("\n📊 IDW‐Interpolated CARRA vs In Situ (Station 2642)")
print(f"Mean Absolute Error (MAE):       {mae:.2f} mm")
print(f"Root Mean Squared Error (RMSE):  {rmse:.2f} mm")
print(f"Correlation Coefficient:         {correlation:.2f}")
print(f"Bias (IDW − In Situ):            {bias:.2f} mm")

# --- 5) Plot 1: Daily time series comparison ---
plt.figure(figsize=(15, 6))
plt.plot(aligned.index, aligned["IDW"],      label="CARRA (IDW)", alpha=0.7)
plt.plot(aligned.index, aligned["In_Situ"],  label="In Situ (2642)", alpha=0.7)
plt.title("Daily Precipitation: IDW‐Interpolated CARRA vs In Situ (Ísafjörður)")
plt.ylabel("Precipitation (mm)")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 6) Plot 2: Scatter with 1:1 line ---
plt.figure(figsize=(6, 6))
plt.scatter(aligned["In_Situ"], aligned["IDW"], alpha=0.5)
m = max(aligned.max())
plt.plot([0, m], [0, m], "r--", label="1:1 line")
plt.title("Scatter: IDW‐Interpolated CARRA vs In Situ Precipitation")
plt.xlabel("In Situ (mm)")
plt.ylabel("CARRA (IDW) (mm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
