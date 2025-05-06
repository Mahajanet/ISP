#!/usr/bin/env python3
import os
import warnings

import numpy as np
import xarray as xr
from haversine import haversine, Unit

warnings.filterwarnings("ignore")

# === CONFIGURATION ===
INPUT_DIR    = "/Users/jahnavimahajan/Projects/ISP/carra_data"
OUTPUT_ROOT  = "/Users/jahnavimahajan/Projects/ISP/raw_data/idw"
RADIUS_KM    = 150.0  # search radius for nearby grid points

STATIONS = {
    "isa":   {"lat": 66.0596, "lon": -23.1699},
    "thver": {"lat": 66.0444, "lon": -23.3074},
}

# map all possible names in the files to our standard 4
VAR_MAP = {
    "t2m":   "t2m",
    "pr":    "pr",
    "10si":  "si10", "si10": "si10",
    "D10":   "wdir10", "wdir10": "wdir10",
}

def normalize_coords(ds):
    """Rename lat/lon coords to 'lat'/'lon', shift lon from 0–360 to –180–180."""
    if "latitude" in ds.coords:
        ds = ds.rename({"latitude": "lat"})
    if "longitude" in ds.coords:
        ds = ds.rename({"longitude": "lon"})
    if "lon" in ds.coords:
        ds["lon"] = (((ds["lon"] + 180) % 360) - 180)
        ds = ds.sortby("lon")
    return ds

def detect_variable(ds):
    """Return (file_var_name, std_name) or (None, None) if nothing matches."""
    for fv, std in VAR_MAP.items():
        if fv in ds.data_vars:
            return fv, std
    return None, None

def extract_nearby(ds, fv, station_lat, station_lon):
    """
    Return lists of (distance_km, time_series_array) for every grid point
    within RADIUS_KM.
    """
    da = ds[fv]
    # drop height dim if present
    if "height" in da.dims:
        da = da.squeeze("height", drop=True)

    lats = ds["lat"].values
    lons = ds["lon"].values
    times = da["time"].values
    values = da.values  # shape: (time, lat, lon)

    pts = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            d = haversine((station_lat, station_lon),
                          (float(lat), float(lon)),
                          unit=Unit.KILOMETERS)
            if d <= RADIUS_KM:
                pts.append((d, values[:, i, j]))

    return times, pts

def idw_average(times, pts):
    """
    Given times and list of (dist, series) pairs,
    compute IDW per timestamp and return a 1D array.
    """
    if not pts:
        return None

    ds, series = zip(*pts)
    ds = np.array(ds)
    w = 1.0 / (ds**2)
    w /= w.sum()

    # stack series into shape (n_pts, n_times)
    arr = np.stack(series, axis=0)
    # weighted sum across the first axis → shape (n_times,)
    return np.tensordot(w, arr, axes=(0, 0))

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for fname in sorted(os.listdir(INPUT_DIR)):
        if not fname.endswith(".nc"):
            continue

        path = os.path.join(INPUT_DIR, fname)
        print(f"Processing file: {fname}")

        try:
            ds = xr.open_dataset(path)
        except Exception as e:
            print(f"  ❌ could not open: {e}")
            continue

        ds = normalize_coords(ds)
        fv, std = detect_variable(ds)
        if fv is None:
            print("  ⚠️  no recognized variable in this file, skipping.")
            continue

        print(f"  → variable detected: '{fv}' → '{std}'")
        times, pts = extract_nearby(ds, fv, 0, 0)  # dummy; we'll redo per station

        # actually do each station
        for stn, info in STATIONS.items():
            print(f"    • station: {stn}")
            times, pts = extract_nearby(ds, fv,
                                        station_lat=info["lat"],
                                        station_lon=info["lon"])
            result = idw_average(times, pts)
            if result is None:
                print("      ⚠️  no grid points within radius, skipping.")
                continue

            out_dir = os.path.join(OUTPUT_ROOT, stn, std)
            os.makedirs(out_dir, exist_ok=True)

            out_path = os.path.join(out_dir, f"{stn}_{std}_{fname}")
            xr.Dataset(
                { std: ("time", result) },
                coords={"time": times}
            ).to_netcdf(out_path)

            print(f"      ✔️  wrote: {out_path}")

    print("All done.")

if __name__ == "__main__":
    main()
