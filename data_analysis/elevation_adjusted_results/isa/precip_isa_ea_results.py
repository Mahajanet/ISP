import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# --- Load and combine elevation‐adjusted CARRA NetCDF files ---
carra_files = sorted(glob("raw_data/elevation_adjusted/isa/pr/pr_isa_*.nc"))
if not carra_files:
    raise FileNotFoundError(
        "No NetCDF files found matching pattern: pr_isa_*.nc in raw_data/elevation_adjusted/isa/"
    )

# open each year, concatenate along the time axis
carra_datasets = [xr.open_dataset(fp) for fp in carra_files]
carra_combined = xr.concat(carra_datasets, dim="time")

# Convert to pandas, resample to daily totals
carra_time = pd.to_datetime(carra_combined["time"].values)
carra_df = pd.DataFrame({"pr": carra_combined["pr"].values}, index=carra_time)
carra_daily = carra_df["pr"].resample("D").sum()

# --- Load Excel in situ data (Station 2642) ---
excel_path = "raw_data/in_situ.xlsx"
df_in_situ = pd.read_excel(
    excel_path,
    sheet_name="Observations - 2642",
    parse_dates=["TIMI"],
)
df_in_situ.set_index("TIMI", inplace=True)
# daily totals
in_situ_daily = df_in_situ["R"].dropna().resample("D").sum()

# --- Align the two series on common dates ---
aligned = pd.DataFrame({
    "CARRA_adj": carra_daily,
    "In_Situ":  in_situ_daily
}).dropna()

# === Compute error metrics ===
mae         = mean_absolute_error(aligned["In_Situ"], aligned["CARRA_adj"])
rmse        = mean_squared_error(aligned["In_Situ"], aligned["CARRA_adj"], squared=False)
correlation = aligned["In_Situ"].corr(aligned["CARRA_adj"])
bias        = (aligned["CARRA_adj"] - aligned["In_Situ"]).mean()

print("\n📊 Elevation‐Adjusted CARRA vs In Situ (Station 2642)")
print(f"Mean Absolute Error (MAE):       {mae:.2f} mm")
print(f"Root Mean Squared Error (RMSE):  {rmse:.2f} mm")
print(f"Correlation Coefficient:         {correlation:.2f}")
print(f"Bias (CARRA_adj − In Situ):      {bias:.2f} mm")

# --- Plot 1: Daily time series ---
plt.figure(figsize=(15, 6))
plt.plot(aligned.index, aligned["CARRA_adj"], label="CARRA (elev-adj)", alpha=0.7)
plt.plot(aligned.index, aligned["In_Situ"],  label="In Situ (2642)",   alpha=0.7)
plt.title("Fig 3.4.1 Daily Precipitation: Elev-Adjusted CARRA vs In Situ (Ísafjörður)")
plt.ylabel("Precipitation (mm)")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 2: Scatter plot with 1:1 line ---
plt.figure(figsize=(6, 6))
plt.scatter(aligned["In_Situ"], aligned["CARRA_adj"], alpha=0.5)
max_val = max(aligned.max())
plt.plot([0, max_val], [0, max_val], "r--", label="1:1 line")
plt.title("Fig 3.4.2 Scatter: Elev-Adjusted CARRA vs In Situ Precipitation")
plt.xlabel("In Situ (mm)")
plt.ylabel("CARRA (mm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
