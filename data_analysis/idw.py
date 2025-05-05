import os
import numpy as np
import pandas as pd
import xarray as xr
from haversine import haversine
import warnings

warnings.filterwarnings("ignore")

# ------------------ CONFIGURATION ------------------

variables = {
    "t2m": {
        "input_dir": "V:/vedur/reikn/CARRA_ISL/T2M/t2m_3hr/one_year_per_gribfile",
        "file_prefix": "CF_T2M_ISL",
        "var_name": "t2m",
        "elev_method": True
    },
    "wdir10": {
        "input_dir": "V:/vedur/reikn/CARRA_ISL/D10m/d10m_3hr/one_year_per_gribfile",
        "file_prefix": "CF_D10m",
        "var_name": "wdir10",
        "elev_method": False
    },
    "si10": {
        "input_dir": "V:/vedur/reikn/CARRA_ISL/F10m/f10m_3hr/one_year_per_gribfile",
        "file_prefix": "CF_F10m",
        "var_name": "si10",
        "elev_method": False
    },
    "pr": {
        "input_dir": "V:/vedur/reikn/CARRA_ISL/PR/pr_3hr/Monthly_files",
        "file_prefix": "CF_PR_",
        "var_name": "pr",
        "elev_method": False
    }
}

stations = {
    "isa": {"lat": 66.0596, "lon": -23.1699, "elevation": 2.2},
    "thver": {"lat": 66.0444, "lon": -23.3074, "elevation": 741.0}
}

output_root = "V:/ofanflod/verk/vakt/steph/python/jahnavi/output"
radius_km = 50
years = list(range(2020, 2025))
months = pd.date_range(start="2020-01", end="2025-01", freq="MS").strftime("%Y-%m").tolist()
created_dirs = set()

# ------------------ HELPERS ------------------

def get_file_path(var_info, date, suffix):
    return os.path.join(var_info["input_dir"], f"{var_info['file_prefix']}{date}{suffix}")

def make_output_dir(out_dir):
    if out_dir not in created_dirs:
        os.makedirs(out_dir, exist_ok=True)
        print(f"      Created folder: {out_dir}")
        created_dirs.add(out_dir)

def open_dataset(file_path, suffix):
    return xr.open_dataset(
        file_path,
        engine="cfgrib" if suffix == ".grib" else None,
        backend_kwargs={'indexpath': ''} if suffix == ".grib" else None
    )

def get_variable(ds, varname, lat, lon, timestep=None):
    try:
        sel = ds.sel(latitude=lat, longitude=lon, method="nearest")
        return sel[varname].isel(time=timestep).values if timestep is not None else sel[varname].values
    except:
        return None

def extract_nearby_points(ds, varname, lat, lon):
    coords, dists, values = [], [], []
    lats = ds.latitude.values
    lons = ds.longitude.values
    for pt_lat in lats:
        for pt_lon in lons:
            d = haversine((lat, lon), (float(pt_lat), float(pt_lon)))
            if d <= radius_km:
                v = get_variable(ds, varname, float(pt_lat), float(pt_lon))
                if v is not None:
                    coords.append((float(pt_lat), float(pt_lon)))
                    dists.append(d)
                    values.append(v)
    return coords, dists, values

def apply_idw(target, coords, values, dists):
    weights = 1 / (np.array(dists)[:, np.newaxis] ** 2)
    return np.sum(values * weights, axis=0) / np.sum(weights, axis=0)

# ------------------ MAIN ------------------

for var_key, var_info in variables.items():
    print(f"\n[IDW] Processing variable: {var_key}")
    suffix = ".nc" if var_key == "pr" else ".grib"
    dates = months if var_key == "pr" else years

    for station_key, station in stations.items():
        print(f"  [Station] {station_key}")
        lat, lon, elev = station["lat"], station["lon"], station["elevation"]

        for date in dates:
            file_path = get_file_path(var_info, date, suffix)
            print(f"    [File] Checking: {file_path}")
            if not os.path.isfile(file_path):
                print(f"    [Skip] File not found.")
                continue

            try:
                print(f"    [Open] Loading dataset...")
                ds = open_dataset(file_path, suffix)
                varname = var_info["var_name"]
                time_vals = ds.time.values

                print(f"    [Scan] Locating nearby grid points within {radius_km} km...")
                coords, dists, values = extract_nearby_points(ds, varname, lat, lon)
                print(f"    [Scan] Found {len(values)} valid points.")

                if values:
                    print(f"    [IDW] Applying inverse-distance weighting...")
                    out_dir = f"{output_root}/{station_key}/{var_key}/idw"
                    make_output_dir(out_dir)
                    result = apply_idw((lat, lon), coords, np.array(values), dists)

                    out_path = f"{out_dir}/{var_key}_{station_key}_{date}.nc"
                    print(f"    [Save] Writing result to {out_path}")
                    xr.Dataset({varname: ("time", result)}, coords={"time": time_vals}).to_netcdf(out_path)

                    print(f"    [Done] {station_key} | {var_key} | {date}")
                else:
                    print(f"    [Skip] No valid nearby values for IDW.")
            except Exception as e:
                print(f"    [Error] {e}")
