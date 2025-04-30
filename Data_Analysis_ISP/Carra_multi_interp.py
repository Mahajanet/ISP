import os
import numpy as np
import pandas as pd
import xarray as xr
from haversine import haversine
from pykrige.ok import OrdinaryKriging
import warnings
warnings.filterwarnings("ignore")

# ------------------ CONFIGURATION ------------------

#PLEASE CHANGE INPUT DIRS ACCORDINGLY FOR EACH VARIABLE
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

radius_km = 50  # for IDW and Kriging
years = list(range(2020, 2025))
months = pd.date_range(start="2020-01", end="2025-01", freq="MS").strftime("%Y-%m").tolist()
lapse_rate = -6.5 / 1000  # °C per meter

# ------------------ HELPER FUNCTIONS ------------------

def get_latlon_grid(ds):
    return [(float(lat), float(lon)) for lat in ds.latitude.values for lon in ds.longitude.values]

def get_variable_at(ds, varname, lat, lon, time_index=None):
    try:
        point = ds.sel(latitude=lat, longitude=lon, method="nearest")
        return point[varname].isel(time=time_index).values if time_index is not None else point[varname].values
    except Exception:
        return None

def apply_idw(target, latlon_list, data_list, distances):
    weights = 1 / (np.array(distances)[:, np.newaxis] ** 2)
    weighted = np.sum(data_list * weights, axis=0) / np.sum(weights, axis=0)
    return weighted

def apply_kriging(target, latlon_list, data_list, times):
    estimates = []
    for t in range(min(1, len(times))):  # Use one time step for speed
        sample_vals = np.array([d[t] for d in data_list])
        lons, lats = zip(*latlon_list)
        ok = OrdinaryKriging(lons, lats, sample_vals, variogram_model="linear", verbose=False)
        z, _ = ok.execute("points", [target[1]], [target[0]])
        estimates.append(z[0])
    return np.pad(estimates, (0, len(times)-len(estimates)), constant_values=np.nan)

def apply_elevation_correction(data, station_elev, grid_elev=100):
    return data + lapse_rate * (station_elev - grid_elev)

# ------------------ MAIN PROCESSING ------------------

for var_key, var_info in variables.items():
    is_monthly = var_key == "pr"
    suffix = ".nc" if is_monthly else ".grib"
    dates = months if is_monthly else years

    for station_key, station in stations.items():
        lat, lon, elev = station["lat"], station["lon"], station["elevation"]

        for method in ["elevation_adjusted", "idw", "kriging"]:
            if method == "elevation_adjusted" and not var_info["elev_method"]:
                continue
            os.makedirs(f"./output/{station_key}/{var_key}/{method}", exist_ok=True)

        for date in dates:
            file_name = f"{var_info['file_prefix']}{date}{suffix}"
            file_path = os.path.join(var_info["input_dir"], file_name)

            if not os.path.isfile(file_path):
                print(f"Missing file: {file_path}")
                continue

            try:
                ds = xr.open_dataset(file_path, engine="cfgrib" if suffix == ".grib" else None, backend_kwargs={'indexpath': ''} if suffix == ".grib" else None)
                varname = var_info["var_name"]
                time_vals = ds.time.values
                grid_points = get_latlon_grid(ds)

                # Gather nearby grid values
                values, coords, dists = [], [], []
                for pt_lat, pt_lon in grid_points:
                    dist = haversine((lat, lon), (pt_lat, pt_lon))
                    if dist <= radius_km:
                        val = get_variable_at(ds, varname, pt_lat, pt_lon)
                        if val is not None:
                            values.append(val)
                            coords.append((pt_lat, pt_lon))
                            dists.append(dist)

                values = np.array(values)

                # IDW
                if len(values) > 0:
                    idw_data = apply_idw((lat, lon), coords, values, dists)
                    ds_idw = xr.Dataset({varname: ("time", idw_data)}, coords={"time": time_vals})
                    ds_idw.to_netcdf(f"./output/{station_key}/{var_key}/idw/{var_key}_{station_key}_{date}.nc")

                # Kriging
                if len(values) > 3:
                    kriged_data = apply_kriging((lat, lon), coords, values, time_vals)
                    ds_krig = xr.Dataset({varname: ("time", kriged_data)}, coords={"time": time_vals})
                    ds_krig.to_netcdf(f"./output/{station_key}/{var_key}/kriging/{var_key}_{station_key}_{date}.nc")

                # Elevation Adjustment (only t2m)
                if var_info["elev_method"]:
                    nearest_val = get_variable_at(ds, varname, lat, lon)
                    corrected = apply_elevation_correction(nearest_val, elev)
                    ds_elev = xr.Dataset({varname: ("time", corrected)}, coords={"time": time_vals})
                    ds_elev.to_netcdf(f"./output/{station_key}/{var_key}/elevation_adjusted/{var_key}_{station_key}_{date}.nc")

                print(f"✅ {station_key} | {var_key} | {date}")

            except Exception as e:
                print(f"❌ {var_key} | {station_key} | {date} | {e}")
