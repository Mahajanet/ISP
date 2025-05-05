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
    }
}

stations = {
    "isa": {"lat": 66.0596, "lon": -23.1699, "elevation": 2.2},
    "thver": {"lat": 66.0444, "lon": -23.3074, "elevation": 741.0}
}

output_root = "V:/ofanflod/verk/vakt/steph/python/jahnavi/output"
years = list(range(2020, 2025))
lapse_rate = -6.5 / 1000
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

def elevation_adjusted(values, station_elev, grid_elev):
    delta_z = station_elev - grid_elev
    return values + lapse_rate * delta_z

# ------------------ MAIN ------------------

for var_key, var_info in variables.items():
    print(f"\n[Elevation] Processing variable: {var_key}")
    suffix = ".grib"
    dates = years

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

                # ---- FIX LONGITUDE RANGE HERE ----
                if "longitude" in ds.coords:
                    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
                    print(f"    [Fix] Longitude range adjusted to -180..180")

                varname = var_info["var_name"]
                time_vals = ds.time.values

                print(f"    [Retrieve] Getting value at station location...")
                v = get_variable(ds, varname, lat, lon)

                print(f"    [Adjust] Applying elevation lapse rate...")
                corrected = elevation_adjusted(v, elev, 100)

                out_dir = f"{output_root}/{station_key}/{var_key}/elevation_adjusted"
                make_output_dir(out_dir)
                out_path = f"{out_dir}/{var_key}_{station_key}_{date}.nc"
                print(f"    [Save] Writing to {out_path}")
                xr.Dataset({varname: ("time", corrected)}, coords={"time": time_vals}).to_netcdf(out_path)

                print(f"    [Done] {station_key} | {var_key} | {date}")
            except Exception as e:
                print(f"    [Error] {e}")
