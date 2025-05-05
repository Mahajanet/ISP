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
    }
}

stations = {
    "isa": {"lat": 66.0596, "lon": -23.1699, "elevation": 2.2},
    "thver": {"lat": 66.0444, "lon": -23.3074, "elevation": 741.0}
}

output_root = "V:/ofanflod/verk/vakt/steph/python/jahnavi/output"
radius_km = 50
max_kriging_points = 200
years = list(range(2020, 2025))
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

def apply_kriging(target, coords, values, times):
    estimates = []
    for t in range(min(1, len(times))):
        v = np.array([d[t] for d in values])
        lons, lats = zip(*coords)
        ok = OrdinaryKriging(lons, lats, v, variogram_model="linear", verbose=False)
        z, _ = ok.execute("points", [target[1]], [target[0]])
        estimates.append(z[0])
    return np.pad(estimates, (0, len(times) - len(estimates)), constant_values=np.nan)

def extract_relevant_points(ds, varname, lat, lon, radius_km, max_points):
    coords, dists, values = [], [], []
    lats = ds.latitude.values
    lons = ds.longitude.values

    # Approx bounding box (1 deg lat ≈ 111 km)
    lat_range = (lat - radius_km / 111, lat + radius_km / 111)
    lon_range = (lon - radius_km / 80, lon + radius_km / 80)  # use ~80 km/deg at Icelandic latitudes

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
                    dists.append(d)
                    values.append(v)

    # Sort by distance and keep closest N
    if len(coords) > max_points:
        idx = np.argsort(dists)[:max_points]
        coords = [coords[i] for i in idx]
        values = [values[i] for i in idx]
        print(f"    [Trim] Reduced to {len(coords)} closest points for kriging")

    return coords, values

# ------------------ MAIN ------------------

for var_key, var_info in variables.items():
    print(f"\n[Kriging] Processing variable: {var_key}")
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

                # Fix longitude
                if "longitude" in ds.coords:
                    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
                    print(f"    [Fix] Longitude range adjusted to -180..180")

                varname = var_info["var_name"]
                time_vals = ds.time.values

                print(f"    [Scan] Finding relevant grid points within {radius_km} km...")
                coords, values = extract_relevant_points(ds, varname, lat, lon, radius_km, max_kriging_points)
                print(f"    [Scan] Selected {len(coords)} points for kriging")

                if len(values) > 3:
                    print(f"    [Kriging] Interpolating for first timestep...")
                    out_dir = f"{output_root}/{station_key}/{var_key}/kriging"
                    make_output_dir(out_dir)
                    result = apply_kriging((lat, lon), coords, values, time_vals)

                    out_path = f"{out_dir}/{var_key}_{station_key}_{date}.nc"
                    print(f"    [Save] Writing to {out_path}")
                    xr.Dataset({varname: ("time", result)}, coords={"time": time_vals}).to_netcdf(out_path)

                    print(f"    [Done] {station_key} | {var_key} | {date}")
                else:
                    print(f"    [Skip] Not enough points for kriging.")
            except Exception as e:
                print(f"    [Error] {e}")
