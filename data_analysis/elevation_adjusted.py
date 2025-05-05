import os
import numpy as np
import pandas as pd
import xarray as xr
import warnings

warnings.filterwarnings("ignore")

# ------------------ CONFIGURATION ------------------

variables = {
    "wdir10": {
        "input_dir": "V:/vedur/reikn/CARRA_ISL/D10m/d10m_3hr/one_year_per_gribfile",
        "file_prefix": "CF_D10m",
        "var_name": "wdir10",
        "elev_method": False,
        "is_monthly": False
    },
    "si10": {
        "input_dir": "V:/vedur/reikn/CARRA_ISL/F10m/f10m_3hr/one_year_per_gribfile",
        "file_prefix": "CF_F10m",
        "var_name": "si10",
        "elev_method": False,
        "is_monthly": False
    },
    "pr": {
        "input_dir": "V:/vedur/reikn/CARRA_ISL/PR/pr_3hr/Monthly_files",
        "file_prefix": "CF_PR_",
        "var_name": "pr",
        "elev_method": False,
        "is_monthly": True
    }
}

stations = {
    "isa": {"lat": 66.0596, "lon": -23.1699, "elevation": 2.2},
    "thver": {"lat": 66.0444, "lon": -23.3074, "elevation": 741.0}
}

output_root = "V:/ofanflod/verk/vakt/steph/python/jahnavi/output"
years = list(range(2020, 2025))
months = pd.date_range(start="2020-01", end="2025-01", freq="MS").strftime("%Y-%m").tolist()
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
    suffix = ".nc" if var_info["is_monthly"] else ".grib"
    dates = months if var_info["is_monthly"] else years

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

                if "longitude" in ds.coords:
                    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
                    ds = ds.sortby("longitude")
                    print(f"    [Fix] Longitude range adjusted and sorted (-180 to 180)")

                    # Longitude bounds check
                    lon_min = ds.longitude.min().item()
                    lon_max = ds.longitude.max().item()
                    if not (lon_min <= lon <= lon_max):
                        print(f"    [Skip] Station longitude {lon} is out of dataset bounds ({lon_min} to {lon_max})")
                        continue


                varname = var_info["var_name"]
                time_vals = ds.time.values

                print(f"    [Retrieve] Getting value at station location...")
                v = get_variable(ds, varname, lat, lon)

                if v is None:
                    print(f"    [Skip] Could not retrieve variable.")
                    continue

                if var_info["elev_method"]:
                    print(f"    [Adjust] Applying elevation lapse rate...")
                    corrected = elevation_adjusted(v, elev, 100)
                else:
                    corrected = v

                method_label = "elevation_adjusted" if var_info["elev_method"] else "raw"
                out_dir = f"{output_root}/{station_key}/{var_key}/{method_label}"
                make_output_dir(out_dir)
                out_path = f"{out_dir}/{var_key}_{station_key}_{date}.nc"

                print(f"    [Save] Writing to {out_path}")
                xr.Dataset({varname: ("time", corrected)}, coords={"time": time_vals}).to_netcdf(out_path)

                print(f"    [Done] {station_key} | {var_key} | {date}")
            except Exception as e:
                print(f"    [Error] {e}")
