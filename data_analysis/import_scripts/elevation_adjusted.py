#!/usr/bin/env python3
import os
import re
import logging

import numpy as np
import xarray as xr
from haversine import haversine


# ─── CONFIGURATION ─────────────────────────────────────────────────────────────

INPUT_DIR = "/Users/jahnavimahajan/Projects/ISP/carra_data"
OUTPUT_DIR = "/Users/jahnavimahajan/Projects/ISP/raw_data/elevation_adjusted"

# station definitions
STATIONS = {
    "isa":  {"lat": 66.0596, "lon": -23.1699, "elev": 2.2},
    "thver":{"lat": 66.0444, "lon": -23.3074, "elev": 741.0},
}

# which files match which variable, and fallback name if needed
VARIABLES = {
    "wdir10": {"pattern": r"D10m.*\.nc$", "fallback": "D10"},
    "si10":   {"pattern": r"F10m.*\.nc$", "fallback": "10si"},
    "pr":     {"pattern": r"pr_daily_.*\.nc$", "fallback": "pr"},
    "t2m":    {"pattern": r"t2m_day_ISL.*\.nc$", "fallback": "t2m"},
}


# ─── PICK BEST CELL ────────────────────────────────────────────────────────────

def pick_best_cell(
    ds,
    station_lat,
    station_lon,
    station_elev,
    max_radius_km: float = 50,
    max_elev_diff_m: float = 500,
    alpha: float = 0.7
):
    """
    Find the (lat_idx, lon_idx) minimizing:
       alpha * (horiz_dist / max_radius_km)
     + (1-alpha) * (|grid_elev - station_elev| / max_elev_diff_m)
    among cells within max_radius_km horizontally.
    """
    # 1D coordinate arrays
    lats = ds["latitude"].values  # shape (nlat,)
    lons = ds["longitude"].values # shape (nlon,)
    # make 2D grids
    lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")
    # flatten for haversine calls
    pts = np.column_stack((lat2d.ravel(), lon2d.ravel()))
    # horizontal distance (km)
    d_h = np.array([
        haversine((station_lat, station_lon), (lat, lon))
        for lat, lon in pts
    ]).reshape(lat2d.shape)
    # mask cells beyond search radius
    mask_far = d_h > max_radius_km

    # get grid elevation if available (else assume zero)
    if "height" in ds.coords:
        try:
            grid_elev = float(ds["height"].values)
        except Exception:
            grid_elev = station_elev
    else:
        grid_elev = station_elev

    # vertical difference field (m)
    d_v = np.abs(grid_elev - station_elev) * np.ones_like(d_h)

    # normalized components
    dh_norm = d_h / max_radius_km
    dv_norm = d_v / max_elev_diff_m

    # combined score
    score = alpha * dh_norm + (1 - alpha) * dv_norm
    score[mask_far] = np.inf

    # pick argmin
    idx_flat = np.argmin(score)
    return np.unravel_index(idx_flat, score.shape)


# ─── PROCESS SINGLE FILE ───────────────────────────────────────────────────────

def process_file(file_path: str, var_key: str, var_info: dict):
    filename = os.path.basename(file_path)
    # extract first 4-digit year for naming
    m = re.search(r"\d{4}", filename)
    year = m.group(0) if m else "unknown"

    logging.info(f"Opening {filename}")
    ds = xr.open_dataset(file_path)

    # rename coords if needed
    rename_map = {}
    if "lat" in ds.coords:
        rename_map["lat"] = "latitude"
    if "lon" in ds.coords:
        rename_map["lon"] = "longitude"
    if rename_map:
        ds = ds.rename(rename_map)
        logging.info(f"  renamed coords {rename_map}")

    # wrap longitudes to -180..180
    if (ds["longitude"] > 180).any():
        ds = ds.assign_coords(
            longitude=(((ds.longitude + 180) % 360) - 180)
        ).sortby("longitude")
        logging.info("  adjusted longitudes to -180..180")

    # choose which variable to read
    if var_key in ds.data_vars:
        var_to_use = var_key
    else:
        fb = var_info["fallback"]
        if fb in ds.data_vars:
            var_to_use = fb
            logging.info(f"  using fallback variable '{fb}'")
        else:
            logging.error(
                f"  neither '{var_key}' nor fallback '{fb}' found in {filename}, skipping"
            )
            return

    for station, meta in STATIONS.items():
        lat0, lon0, elev0 = meta["lat"], meta["lon"], meta["elev"]
        logging.info(f"  Station {station}: ({lat0:.4f}, {lon0:.4f}), elev={elev0}m")

        # pick best cell
        lat_idx, lon_idx = pick_best_cell(ds, lat0, lon0, elev0)
        chosen_lat = float(ds["latitude"].values[lat_idx])
        chosen_lon = float(ds["longitude"].values[lon_idx])
        logging.info(f"    selected grid cell lat={chosen_lat:.4f}, lon={chosen_lon:.4f}")

        # extract timeseries
        data = ds[var_to_use].isel(latitude=lat_idx, longitude=lon_idx)

        # write out
        out_dir = os.path.join(OUTPUT_DIR, station, var_key)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{var_key}_{station}_{year}.nc")
        data.to_netcdf(out_file)
        logging.info(f"    wrote {out_file}")


# ─── MAIN DRIVER ───────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.info("Starting elevation-aware extraction")

    for var_key, var_info in VARIABLES.items():
        pat = re.compile(var_info["pattern"])
        candidates = sorted(
            f for f in os.listdir(INPUT_DIR)
            if pat.match(f)
        )
        logging.info(f"Found {len(candidates)} files for '{var_key}'")
        for fname in candidates:
            path = os.path.join(INPUT_DIR, fname)
            try:
                process_file(path, var_key, var_info)
            except Exception:
                logging.exception(f"Error in processing {fname}")

    logging.info("All done.")


if __name__ == "__main__":
    main()
