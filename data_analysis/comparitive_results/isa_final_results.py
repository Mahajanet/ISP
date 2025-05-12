import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import numpy as np

BASE_DIR = "raw_data"

all_vars = {
    "Wind Speed (m/s)": {
        "patterns": {
            'Elev-Adjusted':    f"{BASE_DIR}/elevation_adjusted/isa/si10/si10_isa_*.nc",
            'Gaussian':         f"{BASE_DIR}/gaussian/isa/si10/isa_si10_*.nc",
            'IDW':              f"{BASE_DIR}/idw/isa/si10/isa_si10_*.nc",
            'Kriging':          f"{BASE_DIR}/kriging/isa/si10/si10_isa_F10m*_daily.nc",
            'Nearest Neighbor': f"{BASE_DIR}/nn/wind_speed_nn/f10m_isa_nn/f10m_isa_*.nc",
        },
        "var_names": {
            'Elev-Adjusted': '10si',
            'Gaussian': 'si10',
            'IDW': 'si10',
            'Kriging': 'si10',
            'Nearest Neighbor': 'f10m'
        },
        "in_situ_col": "F",
        "agg_func": "mean"
    },
    "Wind Direction (°)": {
        "patterns": {
            'Elev-Adjusted':    f"{BASE_DIR}/elevation_adjusted/isa/wdir10/wdir10_isa_*.nc",
            'Gaussian':         f"{BASE_DIR}/gaussian/isa/wdir10/isa_wdir10_*.nc",
            'IDW':              f"{BASE_DIR}/idw/isa/wdir10/isa_wdir10_*.nc",
            'Kriging':          f"{BASE_DIR}/kriging/isa/wdir10/wdir10_isa_*_daily.nc",
            'Nearest Neighbor': f"{BASE_DIR}/nn/wind_dir_nn/d10m_isa_nn/d10m_isa_*.nc",
        },
        "var_names": {
            'Elev-Adjusted': 'wdir10',
            'Gaussian': 'wdir10',
            'IDW': 'wdir10',
            'Kriging': 'wdir10',
            'Nearest Neighbor': 'd10m'
        },
        "in_situ_col": "D",
        "agg_func": "mean"
    },
    "2 m Temperature (°C)": {
        "patterns": {
            'Elev-Adjusted':    f"{BASE_DIR}/elevation_adjusted/isa/t2m/t2m_isa_*.nc",
            'Gaussian':         f"{BASE_DIR}/gaussian/isa/t2m/isa_t2m_*.nc",
            'IDW':              f"{BASE_DIR}/idw/isa/t2m/isa_t2m_t2m_day_ISL*.nc",
            'Kriging':          f"{BASE_DIR}/kriging/isa/t2m/t2m_isa_t2m_day_ISL*.nc",
            'Nearest Neighbor': f"{BASE_DIR}/nn/t2m_nn/t2m_isa_nn/t2m_isa_*.nc",
        },
        "var_names": {
            'Elev-Adjusted': 't2m',
            'Gaussian': 't2m',
            'IDW': 't2m',
            'Kriging': 't2m',
            'Nearest Neighbor': 't2m'
        },
        "in_situ_col": "T",
        "agg_func": "mean",
        "kelvin_to_celsius": True
    },
    "Precipitation (mm/qtr)": {
        "patterns": {
            'Elev-Adjusted':    f"{BASE_DIR}/elevation_adjusted/isa/pr/pr_isa_*.nc",
            'Gaussian':         f"{BASE_DIR}/gaussian/isa/pr/isa_pr_*.nc",
            'IDW':              f"{BASE_DIR}/idw/isa/pr/isa_pr_*.nc",
            'Kriging':          f"{BASE_DIR}/kriging/isa/pr/pr_isa_pr_daily_*.nc",
            'Nearest Neighbor': f"{BASE_DIR}/nn/precip_nn/precip_isa_nn/pr_isa_*.nc",
        },
        "var_names": {
            'Elev-Adjusted': 'pr',
            'Gaussian': 'pr',
            'IDW': 'pr',
            'Kriging': 'pr',
            'Nearest Neighbor': 'pr'
        },
        "in_situ_col": "R",
        "agg_func": "sum"
    },
}

def load_daily_series(pattern, varname, kelvin_to_c=False):
    files = sorted(glob(pattern))
    if not files:
        return pd.Series(dtype=float)
    datasets = []
    for fp in files:
        ds = xr.open_dataset(fp)
        ds = ds.drop_vars(["height", "latitude", "longitude"], errors="ignore")
        datasets.append(ds)
    combined = xr.concat(datasets, dim="time", coords="minimal")
    times = pd.to_datetime(combined["time"].values)
    values = combined[varname].values.ravel()
    if kelvin_to_c:
        values = values - 273.15
    return pd.Series(values, index=times).resample("D").mean()

# === Full quarter range for reindexing ===
all_quarters = pd.date_range("2020-01-01", "2024-12-31", freq="Q")

# === Plot generation ===
for title, cfg in all_vars.items():
    carr = {}
    for method in cfg["patterns"]:
        kelvin = cfg.get("kelvin_to_celsius", False)
        carr[method] = load_daily_series(cfg["patterns"][method], cfg["var_names"][method], kelvin_to_c=kelvin)

    df0 = pd.read_excel("raw_data/in_situ.xlsx", sheet_name="Observations - 2642", parse_dates=["TIMI"])
    df0.set_index("TIMI", inplace=True)
    carr["In Situ"] = df0[cfg["in_situ_col"]].dropna().resample("D").mean()

    df_all = pd.DataFrame(carr)
    df_all = df_all[df_all.index.year <= 2024]
    df_quarterly = df_all.resample("Q").agg(cfg["agg_func"]).reindex(all_quarters)

    quarters = df_quarterly.index.to_period("Q")
    quarter_labels = [f"Q{q.quarter} {q.year}" for q in quarters]
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
    ax.set_title(f"Quarterly {title}: CARRA Methods vs In Situ (Ísafjörður)", fontsize=16)
    ax.set_ylabel(title, fontsize=14)
    ax.legend(frameon=False, ncol=2, fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    filename = title.lower().split()[0].replace("(", "").replace(")", "").replace("/", "")
    plt.savefig(f"quarterly_{filename}_comparison_bars.png", dpi=300)
    plt.show()
