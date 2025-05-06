import os
import numpy as np
import pandas as pd
import xarray as xr
import warnings

warnings.filterwarnings("ignore")

# ------------------ CONFIGURATION ------------------

variables = {
    "t2m": {
        "input_dir": "V:/vedur/reikn/CARRA_ISL/T2M/t2m_3hr/one_year_per_gribfile",
        "file_prefix": "CF_T2M_ISL",
        "var_name": "t2m"
    },
    "wdir10": {
        "input_dir": "V:/vedur/reikn/CARRA_ISL/D10m/d10m_3hr/one_year_per_gribfile",
        "file_prefix": "CF_D10m",
        "var_name": "wdir10"
    },
    "si10": {
        "input_dir": "V:/vedur/reikn/CARRA_ISL/F10m/f10m_3hr/one_year_per_gribfile",
        "file_prefix": "CF_F10m",
        "var_name": "si10"
    },
    "pr": {
        "input_dir": "V:/vedur/reikn/CARRA_ISL/PR/pr_3hr/Monthly_files",
        "file_prefix": "CF_PR_",
        "var_name": "pr"
    }
}

output_root = "V:/ofanflod/verk/vakt/steph/python/jahnavi/output_westfjords"
years = list(range(2020, 2025))
months = pd.date_range(start="2020-01", end="2025-01", freq="MS").strftime("%Y-%m").tolist()

lat_min = 65.5
lat_max = 66.5
lon_min = -24.5
lon_max = -21.5

# ------------------ HELPERS ------------------

def open_dataset(file_path, suffix):
    if suffix == ".grib":
        ds = xr.open_dataset(file_path, engine="cfgrib", backend_kwargs={'indexpath': ''})
    else:
        ds = xr.open_dataset(file_path)

    if "longitude" in ds.coords:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
        ds = ds.sortby('longitude')  # Ensure longitudes are increasing
    return ds

def extract_subset(ds, varname):
    """Vectorized extraction within bounding box"""
    if "latitude" not in ds.dims or "longitude" not in ds.dims:
        raise ValueError("Missing spatial coordinates in dataset.")

    subset = ds[varname].sel(
        latitude=slice(lat_max, lat_min),  # descending latitude
        longitude=slice(lon_min, lon_max)
    ).dropna(dim="latitude", how="all").dropna(dim="longitude", how="all")

    return subset

# ------------------ MAIN ------------------

for var_key, var_info in variables.items():
    print(f"\n[Processing] Variable: {var_key}")
    suffix = ".nc" if var_key == "pr" else ".grib"
    dates = months if var_key == "pr" else years

    for date in dates:
        file_name = f"{var_info['file_prefix']}{date}{suffix}"
        file_path = os.path.join(var_info["input_dir"], file_name)

        if not os.path.isfile(file_path):
            print(f"  [Missing] {file_name}")
            continue

        try:
            print(f"  [Open] {file_name}")
            ds = open_dataset(file_path, suffix)
            varname = var_info["var_name"]

            print(f"  [Subset] Filtering data in bounding box...")
            subset = extract_subset(ds, varname)

            if subset.size == 0:
                print(f"  [Skip] No data in bounding box.")
                continue

            print(f"  [Reshape] Flattening spatial grid to points...")
            stacked = subset.stack(point=("latitude", "longitude"))

            out_dir = os.path.join(output_root, var_key)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{var_key}_westfjords_{date}.nc")

            print(f"  [Save] Saving to NetCDF...")
            stacked.to_netcdf(out_path)
            print(f"  [Done] Saved: {out_path}")

        except Exception as e:
            print(f"  [Error] {file_name}: {e}")
