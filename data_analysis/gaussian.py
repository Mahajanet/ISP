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
alpha = 3.0
Rp = 1000.0
gradT = -0.5 / 100
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
    ds = xr.open_dataset(
        file_path,
        engine="cfgrib" if suffix == ".grib" else None,
        backend_kwargs={'indexpath': ''} if suffix == ".grib" else None
    )

    # ✅ Fix longitudes: convert 0–360 to -180–180, then sort
    if "longitude" in ds.coords:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
        ds = ds.sortby("longitude")
        print(f"    [Fix] Longitude range adjusted and sorted (-180 to 180)")

    return ds

def get_variable(ds, varname, lat, lon, timestep=None):
    try:
        sel = ds.sel(latitude=lat, longitude=lon, method="nearest")
        return sel[varname].isel(time=timestep).values if timestep is not None else sel[varname].values
    except:
        return None

def apply_gaussian(target, coords, values, times, varname, station_elev):
    result = []
    weights = []
    skipped = 0

    for i, (lat, lon) in enumerate(coords):
        dist_m = haversine((target[0], target[1]), (lat, lon)) * 1000
        weight = max(np.exp(-alpha * (dist_m / Rp)**2) - np.exp(-alpha), 0.0)
        if weight <= 1e-8:
            skipped += 1
            continue
        v = values[i]
        if varname == "t2m":
            v = v + gradT * (station_elev - 100)
        result.append(v)
        weights.append(weight)

    if result:
        result = np.array(result)
        weights = np.array(weights)[:, np.newaxis]
        combined = np.sum(result * weights, axis=0) / np.sum(weights, axis=0)
        print(f"      Gaussian combined from {len(result)} points (skipped {skipped})")
        return combined
    print(f"      No valid points after weighting")
    return None

def extract_gaussian_points(ds, varname, lat, lon):
    coords = []
    values = []
    lats = ds.latitude.values
    lons = ds.longitude.values

    lat_range = (lat - radius_km / 111, lat + radius_km / 111)
    lon_range = (lon - radius_km / 80, lon + radius_km / 80)

    for pt_lat in lats:
        if pt_lat < lat_range[0] or pt_lat > lat_range[1]:
            continue
        for pt_lon in lons:
            if pt_lon < lon_range[0] or pt_lon > lon_range[1]:
                continue
            d = haversine((lat, lon), (float(pt_lat), float(pt_lon)))
            if d <= radius_km:
                v = get_variable(ds, varname, float(pt_lat), float(pt_lon))
                if v is not None:
                    coords.append((float(pt_lat), float(pt_lon)))
                    values.append(v)
    return coords, values

# ------------------ MAIN ------------------

for var_key, var_info in variables.items():
    print(f"\n[Gaussian] Processing variable: {var_key}")
    is_monthly = var_key == "pr"
    suffix = ".nc" if is_monthly else ".grib"
    dates = months if is_monthly else years

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

                print(f"    [Scan] Gathering candidate points...")
                coords, values = extract_gaussian_points(ds, varname, lat, lon)
                print(f"    [Scan] Found {len(coords)} candidate points")

                if values:
                    print(f"    [Gaussian] Applying weights and combining...")
                    out_dir = f"{output_root}/{station_key}/{var_key}/gaussian"
                    make_output_dir(out_dir)
                    result = apply_gaussian((lat, lon), coords, values, time_vals, varname, elev)

                    if result is not None:
                        out_path = f"{out_dir}/{var_key}_{station_key}_{date}.nc"
                        print(f"    [Save] Writing to {out_path}")
                        xr.Dataset({varname: ("time", result)}, coords={"time": time_vals}).to_netcdf(out_path)
                        print(f"    [Done] {station_key} | {var_key} | {date}")
                    else:
                        print(f"    [Skip] No result generated")
                else:
                    print(f"    [Skip] No valid points for Gaussian")
            except Exception as e:
                print(f"    [Error] {e}")
