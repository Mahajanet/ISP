import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import numpy as np

# === File patterns and variable names ===
patterns = {
    'Elev-Adjusted':    "raw_data/elevation_adjusted/isa/si10/si10_isa_*.nc",
    'Gaussian':         "raw_data/gaussian/isa/si10/isa_si10_*.nc",
    'IDW':              "raw_data/idw/isa/si10/isa_si10_*.nc",
    'Kriging':          "raw_data/kriging/isa/si10/si10_isa_F10m*_daily.nc",
    'Nearest Neighbor': "raw_data/nn/wind_speed_nn/f10m_isa_nn/f10m_isa_*.nc",
}
var_names = {
    'Elev-Adjusted': '10si',
    'Gaussian': 'si10',
    'IDW': 'si10',
    'Kriging': 'si10',
    'Nearest Neighbor': 'f10m'
}

# === Load & resample each method to daily, then quarterly ===
def load_daily_series(pattern, varname):
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files for pattern: {pattern}")
    datasets = []
    for fp in files:
        ds = xr.open_dataset(fp)
        ds = ds.drop_vars(["height", "latitude", "longitude"], errors="ignore")
        datasets.append(ds)
    combined = xr.concat(datasets, dim="time", coords="minimal")
    times = pd.to_datetime(combined["time"].values)
    values = combined[varname].values.ravel()
    return pd.Series(values, index=times).resample("D").mean()

# === Build quarterly series for each method ===
quarterlies = {}

for name in patterns:
    daily = load_daily_series(patterns[name], var_names[name])
    quarterly = daily.resample("Q").mean()
    quarterlies[name] = quarterly

# === Load in-situ wind speed and resample to quarterly ===
df_insitu = pd.read_excel("raw_data/in_situ.xlsx", sheet_name="Observations - 2642", parse_dates=["TIMI"])
df_insitu.set_index("TIMI", inplace=True)
in_situ_q = df_insitu["F"].dropna().resample("Q").mean()
quarterlies["In Situ"] = in_situ_q

# === Combine into DataFrame & filter only 2020–2024 ===
df_quarterly = pd.DataFrame(quarterlies).sort_index()
df_quarterly = df_quarterly[df_quarterly.index.year <= 2024]

quarters = df_quarterly.index.to_period("Q")
quarter_labels = [f"Q{q.quarter} {q.year}" for q in quarters]

# === Plot grouped bar chart ===
methods = df_quarterly.columns.tolist()
n = len(methods)
x = np.arange(len(df_quarterly))
width = 0.8 / n

fig, ax = plt.subplots(figsize=(14, 6))
for i, m in enumerate(methods):
    ax.bar(x + i * width, df_quarterly[m], width, label=m)

group_centers = x + (n * width) / 2
ax.set_xticks(group_centers)
ax.set_xticklabels(quarter_labels, rotation=45)

ax.set_title("Quarterly Mean 10 m Wind Speed: CARRA Methods vs In Situ (Ísafjörður)", fontsize=16)
ax.set_ylabel("Wind Speed (m/s)", fontsize=14)
ax.legend(frameon=False, ncol=2, fontsize=12)
ax.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("quarterly_wind_speed_filtered.png", dpi=300)
plt.show()
