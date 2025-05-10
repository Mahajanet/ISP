import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- 1) Load & combine kriging‐interpolated CARRA 2 m temperature for Þverá (Station 2636) ---
krig_files = sorted(glob("raw_data/kriging/thver/t2m/t2m_thver_t2m_day_ISL*.nc"))
if not krig_files:
    raise FileNotFoundError(
        "No kriging t2m files found matching pattern: thver_t2m_T2m*_daily.nc in raw_data/kriging/thver/t2m/"
    )

# open & concat along time
krig_ds = xr.concat([xr.open_dataset(fp) for fp in krig_files], dim="time")

# pull out the 2 m temp series (Kelvin → °C)
time        = pd.to_datetime(krig_ds["time"].values)
t2m_K       = krig_ds["t2m"].values
t2m_C       = t2m_K - 273.15
krig_series = pd.Series(t2m_C, index=time).resample("D").mean()
krig_series.index = krig_series.index.normalize()

# --- 2) Load in-situ Excel (Þverá, Station 2636) ---
df_insitu = pd.read_excel(
    "raw_data/in_situ.xlsx",
    sheet_name="Observations - 2636",
    parse_dates=["TIMI"]
).set_index("TIMI")
insitu_daily = df_insitu["T"].dropna().resample("D").mean()
insitu_daily.index = insitu_daily.index.normalize()

# --- 3) Align & drop non-overlapping days ---
aligned = pd.DataFrame({
    "Kriging":  krig_series,
    "In_Situ":  insitu_daily
}).dropna()

if aligned.empty:
    print("⚠️ No overlapping dates between kriging output and in-situ data!")
    print(f"  Kriging covers {krig_series.index.min()} → {krig_series.index.max()}")
    print(f"  In Situ covers {insitu_daily.index.min()} → {insitu_daily.index.max()}")
    exit(1)

# --- 4) Compute error metrics ---
mae         = mean_absolute_error(aligned["In_Situ"], aligned["Kriging"])
rmse        = mean_squared_error(aligned["In_Situ"], aligned["Kriging"], squared=False)
corr        = aligned["In_Situ"].corr(aligned["Kriging"])
bias        = (aligned["Kriging"] - aligned["In_Situ"]).mean()

print("\n📊 Kriging CARRA vs In Situ (Þverá, Station 2636) – 2 m Air Temp (°C)")
print(f"  MAE:     {mae:.2f} °C")
print(f"  RMSE:    {rmse:.2f} °C")
print(f"  Corr:    {corr:.2f}")
print(f"  Bias:    {bias:.2f} °C")

# --- 5) Plot: daily‐mean time series ---
plt.figure(figsize=(14,5))
plt.plot(aligned.index, aligned["Kriging"], label="Kriging CARRA", alpha=0.7)
plt.plot(aligned.index, aligned["In_Situ"], label="In Situ T",     alpha=0.7)
plt.title("Daily Mean 2 m Temperature: Kriging vs In Situ (Þverá)")
plt.ylabel("Temperature (°C)")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()

# --- 6) Plot: scatter with 1:1 line ---
plt.figure(figsize=(6,6))
plt.scatter(aligned["In_Situ"], aligned["Kriging"], alpha=0.5)
mn, mx = aligned.min().min(), aligned.max().max()
plt.plot([mn,mx],[mn,mx],"r--",label="1:1 line")
plt.title("Scatter: Kriging vs In Situ 2 m Temperature")
plt.xlabel("In Situ (°C)")
plt.ylabel("Kriging (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
