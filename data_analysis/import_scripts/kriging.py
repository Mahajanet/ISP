import os
import numpy as np
import pandas as pd
import xarray as xr
from haversine import haversine
from pykrige.ok import OrdinaryKriging
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

stations = {
    "isa": {"lat": 66.0596, "lon": -23.1699},
    "thver": {"lat": 66.0444, "lon": -23.3074}
}

output_root = "V:/ofanflod/verk/vakt/steph/python/jahnavi/output"
radius_km = 50
max_kriging_points = 200
years = list(range(2020, 2025))
months = pd.date_range(start="2020-01", end="2025-01", freq="MS").strftime("%Y-%m").tolist()

# ------------------ HELPERS ------------------

def get_file_path(var_info, date, suffix):
    return os.path.join(var_info["input_dir"], f"{var_info['file_prefix']}{date}{suffix}")

def make_output_dir(path):
    os.makedirs(path, exist_ok=True)

def open_dataset(file_path, suffix):
    ds = xr.open_dataset(
        file_path,
        engine="cfgrib" if suffix == ".grib" else None,
        backend_kwargs={'indexpath': ''} if suffix == ".grib" else None
    )
    if "longitude" in ds.coords:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
        ds = ds.sortby("longitude")
    return ds

def get_variable(ds, varname, lat, lon, timestep=None):
    try:
        sel = ds.sel(latitude=lat, longitude=lon, method="nearest")
        return sel[varname].isel(time=timestep).values if timestep is not None else sel[varname].values
    except:
        return None

def extract_points(ds, varname, lat, lon, radius_km, max_points):
    coords, dists, values = [], [], []
    lats = ds.latitude.values
    lons = ds.longitude.values

    lat_range = (lat - radius_km / 111, lat + radius_km / 111)
    lon_range = (lon - radius_km / 80, lon + radius_km / 80)

    for la in lats:
        if not lat_range[0] <= la <= lat_range[1]:
            continue
        for lo in lons:
            if not lon_range[0] <= lo <= lon_range[1]:
                continue
            d = haversine((lat, lon), (float(la), float(lo)))
            if d <= radius_km:
                v = get_variable(ds, varname, float(la), float(lo))
                if v is not None:
                    coords.append((float(la), float(lo)))
                    dists.append(d)
                    values.append(v)

    if len(coords) > max_points:
        idx = np.argsort(dists)[:max_points]
        coords = [coords[i] for i in idx]
        values = [values[i] for i in idx]
        print(f"    [Trimmed] to {len(coords)} closest points")

    return coords, values

def krige_all_timesteps(target, coords, values, times):
    estimates = []
    lons, lats = zip(*coords)
    for t in range(len(times)):
        try:
            v = np.array([val[t] for val in values])
            ok = OrdinaryKriging(lons, lats, v, variogram_model="linear", verbose=False)
            z, _ = ok.execute("points", [target[1]], [target[0]])
            estimates.append(z[0])
        except Exception as e:
            print(f"      [Skip timestep {t}] {e}")
            estimates.append(np.nan)
    return np.array(estimates)

# ------------------ MAIN ------------------

for var_key, var_info in variables.items():
    print(f"\nProcessing variable: {var_key}")
    suffix = ".nc" if var_key == "pr" else ".grib"
    dates = months if var_key == "pr" else years

    for station_key, station in stations.items():
        print(f"  Station: {station_key}")
        lat, lon = station["lat"], station["lon"]

        for date in dates:
            file_path = get_file_path(var_info, date, suffix)
            print(f"    File: {file_path}")
            if not os.path.isfile(file_path):
                print("    [Skip] File not found.")
                continue

            try:
                ds = open_dataset(file_path, suffix)
                varname = var_info["var_name"]
                time_vals = ds.time.values

                # Bounds check
                if not (
                    ds.longitude.min().item() <= lon <= ds.longitude.max().item() and
                    ds.latitude.min().item() <= lat <= ds.latitude.max().item()
                ):
                    print("    [Skip] Station outside dataset bounds.")
                    continue

                coords, values = extract_points(ds, varname, lat, lon, radius_km, max_kriging_points)
                print(f"    Found {len(coords)} valid points")

                if len(values) > 3:
                    print(f"    Interpolating {len(time_vals)} timesteps...")
                    result = krige_all_timesteps((lat, lon), coords, values, time_vals)
                    out_dir = os.path.join(output_root, station_key, var_key, "kriging")
                    make_output_dir(out_dir)
                    out_path = os.path.join(out_dir, f"{var_key}_{station_key}_{date}.nc")
                    xr.Dataset({varname: ("time", result)}, coords={"time": time_vals}).to_netcdf(out_path)
                    print("    [Saved]")
                else:
                    print("    [Skip] Not enough points for kriging.")
            except Exception as e:
                print(f"    [Error] {e}")
