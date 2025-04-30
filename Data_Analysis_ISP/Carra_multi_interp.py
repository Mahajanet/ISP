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
    "isa": {"lat": 66.0596, "lon": -23.1699, "elevation": 20},
    "thver": {"lat": 66.0444, "lon": -23.3074, "elevation": 275}
}

radius_km = 50
years = list(range(2020, 2025))
months = pd.date_range(start="2020-01", end="2025-01", freq="MS").strftime("%Y-%m").tolist()
lapse_rate = -6.5 / 1000
gradT = -0.5 / 100
alpha = 3.0
Rp = 1000.0

# ------------------ HELPERS ------------------

def get_grid_points(ds):
    return [(float(lat), float(lon)) for lat in ds.latitude.values for lon in ds.longitude.values]

def get_variable(ds, varname, lat, lon, timestep=None):
    try:
        sel = ds.sel(latitude=lat, longitude=lon, method="nearest")
        return sel[varname].isel(time=timestep).values if timestep is not None else sel[varname].values
    except:
        return None

def elevation_adjusted(values, station_elev, grid_elev):
    delta_z = station_elev - grid_elev
    return values + lapse_rate * delta_z

def apply_idw(target, coords, values, dists):
    weights = 1 / (np.array(dists)[:, np.newaxis] ** 2)
    return np.sum(values * weights, axis=0) / np.sum(weights, axis=0)

def apply_kriging(target, coords, values, times):
    estimates = []
    for t in range(min(1, len(times))):  # Speed: only 1 timestep
        v = np.array([d[t] for d in values])
        lons, lats = zip(*coords)
        ok = OrdinaryKriging(lons, lats, v, variogram_model="linear", verbose=False)
        z, _ = ok.execute("points", [target[1]], [target[0]])
        estimates.append(z[0])
    return np.pad(estimates, (0, len(times)-len(estimates)), constant_values=np.nan)

def apply_gaussian(target, coords, values, times, varname, station_elev):
    result = []
    weights = []
    for i, (lat, lon) in enumerate(coords):
        dist = haversine((target[0], target[1]), (lat, lon)) * 1000
        weight = max(np.exp(-alpha * (dist / Rp)**2) - np.exp(-alpha), 0.0)
        if weight > 0:
            v = values[i]
            if varname == "t2m":
                grid_elev = 100
                v = v + gradT * (station_elev - grid_elev)
            result.append(v)
            weights.append(weight)
    if result:
        result = np.array(result)
        weights = np.array(weights)[:, np.newaxis]
        return np.sum(result * weights, axis=0) / np.sum(weights, axis=0)
    return None

# ------------------ MAIN ------------------

methods = ["elevation_adjusted", "idw", "kriging", "gaussian"]
output_root = "./output"
created_dirs = set()

for var_key, var_info in variables.items():
    is_monthly = var_key == "pr"
    suffix = ".nc" if is_monthly else ".grib"
    dates = months if is_monthly else years

    for station_key, station in stations.items():
        lat, lon, elev = station["lat"], station["lon"], station["elevation"]

        for date in dates:
            file_name = f"{var_info['file_prefix']}{date}{suffix}"
            file_path = os.path.join(var_info["input_dir"], file_name)

            if not os.path.isfile(file_path):
                continue

            try:
                ds = xr.open_dataset(file_path, engine="cfgrib" if suffix == ".grib" else None, backend_kwargs={'indexpath': ''} if suffix == ".grib" else None)
                varname = var_info["var_name"]
                time_vals = ds.time.values
                grid_points = get_grid_points(ds)

                coords, dists, values = [], [], []
                for pt_lat, pt_lon in grid_points:
                    d = haversine((lat, lon), (pt_lat, pt_lon))
                    if d <= radius_km:
                        v = get_variable(ds, varname, pt_lat, pt_lon)
                        if v is not None:
                            coords.append((pt_lat, pt_lon))
                            dists.append(d)
                            values.append(v)
                values = np.array(values)

                if var_info["elev_method"]:
                    out_dir = f"{output_root}/{station_key}/{var_key}/elevation_adjusted"
                    os.makedirs(out_dir, exist_ok=True)
                    created_dirs.add(out_dir)
                    v = get_variable(ds, varname, lat, lon)
                    corrected = elevation_adjusted(v, elev, 100)
                    xr.Dataset({varname: ("time", corrected)}, coords={"time": time_vals})\
                        .to_netcdf(f"{out_dir}/{var_key}_{station_key}_{date}.nc")

                if len(values) > 0:
                    out_dir = f"{output_root}/{station_key}/{var_key}/idw"
                    os.makedirs(out_dir, exist_ok=True)
                    created_dirs.add(out_dir)
                    result = apply_idw((lat, lon), coords, values, dists)
                    xr.Dataset({varname: ("time", result)}, coords={"time": time_vals})\
                        .to_netcdf(f"{out_dir}/{var_key}_{station_key}_{date}.nc")

                if len(values) > 3:
                    out_dir = f"{output_root}/{station_key}/{var_key}/kriging"
                    os.makedirs(out_dir, exist_ok=True)
                    created_dirs.add(out_dir)
                    result = apply_kriging((lat, lon), coords, values, time_vals)
                    xr.Dataset({varname: ("time", result)}, coords={"time": time_vals})\
                        .to_netcdf(f"{out_dir}/{var_key}_{station_key}_{date}.nc")

                if len(values) > 0:
                    out_dir = f"{output_root}/{station_key}/{var_key}/gaussian"
                    os.makedirs(out_dir, exist_ok=True)
                    created_dirs.add(out_dir)
                    result = apply_gaussian((lat, lon), coords, values, time_vals, varname, elev)
                    if result is not None:
                        xr.Dataset({varname: ("time", result)}, coords={"time": time_vals})\
                            .to_netcdf(f"{out_dir}/{var_key}_{station_key}_{date}.nc")

                print(f"✅ {station_key} | {var_key} | {date}")

            except Exception as e:
                print(f"❌ Error {var_key} {station_key} {date}: {e}")

# Report total output folders
num_dirs = len(created_dirs)
expected_dirs = len(variables) * len(stations) * len(methods)
print(f"\n✅ Created {num_dirs} output folders (expected max = {expected_dirs})")
for d in sorted(created_dirs): print(" -", d)
