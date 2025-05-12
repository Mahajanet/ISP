import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import numpy as np

# === SET YOUR BASE PROJECT DIRECTORY ===
BASE_DIR = "/Users/jahnavimahajan/Projects/ISP"

# === 1) File patterns & variable names (absolute paths) ===
patterns = {
    'Elev-Adjusted':    f"{BASE_DIR}/raw_data/elevation_adjusted/isa/wdir10/wdir10_isa_*.nc",
    'Gaussian':         f"{BASE_DIR}/raw_data/gaussian/isa/wdir10/isa_wdir10_*.nc",
    'IDW':              f"{BASE_DIR}/raw_data/idw/isa/wdir10/isa_wdir10_*.nc",
    'Kriging':          f"{BASE_DIR}/raw_data/kriging/isa/wdir10/wdir10_isa_*_daily.nc",
    'Nearest Neighbor': f"{BASE_DIR}/raw_data/nn/wind_dir_nn/d10m_isa_nn/d10m_isa_*.nc",
}
var_names = {
    'Elev-Adjusted': 'wdir10',
    'Gaussian': 'wdir10',
    'IDW': 'wdir10',
    'Kriging': 'wdir10',
    'Nearest Neighbor': 'd10m'
}

# === 2) Load and resample to daily means ===
def load_daily(pattern, varname):
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files for pattern: {pattern}")
    ds = xr.concat(
        [xr.open_dataset(f).drop_vars(["height", "latitude", "longitude"], errors="ignore") for f in files],
        dim="time", coords="minimal"
    )
    times = pd.to_datetime(ds["time"].values)
    vals = ds[varname].values.ravel()
    return pd.Series(vals, index=times).resample("D").mean()

# === 3) Load all methods ===
carr = {name: load_daily(patterns[name], var_names[name]) for name in patterns}

# === 4) Load and resample in-situ data ===
df0 = pd.read_excel(f"{BASE_DIR}/raw_data/in_situ.xlsx", sheet_name="Observations - 2642", parse_dates=["TIMI"])
df0.set_index("TIMI", inplace=True)
carr["In Situ"] = df0["D"].dropna().resample("D").mean()

# === 5) Align all series, forward-fill short gaps ===
df_all = pd.DataFrame(carr)
df_all = df_all.ffill(limit=3)
df_daily = df_all.dropna(thresh=5)  # allow one method to be missing at most

# === 6) Resample to quarterly means ===
df_quarterly = df_daily.resample("Q").mean()
quarters = df_quarterly.index.to_period("Q")
quarter_labels = [f"Q{q.quarter} {q.year}" for q in quarters]

# === 7) Plot grouped bar chart ===
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

ax.set_title("Quarterly Mean Wind Direction: CARRA Methods vs In Situ (Ísafjörður)", fontsize=16)
ax.set_ylabel("Wind Direction (°)", fontsize=14)
ax.legend(frameon=False, ncol=2, fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(f"{BASE_DIR}/quarterly_wind_dir_all_years.png", dpi=300)
plt.show()
