import os
import numpy as np
import xarray as xr
from pathlib import Path

# === CONFIGURATION ===
data_folder = "/Users/jahnavimahajan/Projects/ISP/carra_data"
output_base = Path("/Users/jahnavimahajan/Projects/ISP/raw_data/gaussian")
stations = {
    "isa": {"lat": 66.0596, "lon": -23.1699, "elevation": 2.2},
    "thver": {"lat": 66.0444, "lon": -23.3074, "elevation": 741.0}
}

# Mapping real variable names in files to the standard names we want to output
variable_name_map = {
    "t2m": "t2m",
    "pr": "pr",
    "si10": "si10", "10si": "si10",
    "wdir10": "wdir10", "D10": "wdir10"
}

# Search order for variables in the files
search_priority = ["t2m", "pr", "si10", "wdir10"]

# === HELPER FUNCTIONS ===
def shift_and_fix_coords(ds):
    """Rename and adjust coordinates if necessary."""
    if "latitude" in ds.coords and "longitude" in ds.coords:
        ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    if "lon" in ds.coords:
        ds["lon"] = (((ds["lon"] + 180) % 360) - 180)  # Normalize longitude to [-180, 180]
        ds = ds.sortby("lon")  # Sort by longitude to handle wrapping correctly
    return ds

def extract_variable(ds):
    """Extract the appropriate variable from the dataset."""
    for real_var, std_var in variable_name_map.items():
        if real_var in ds.data_vars:
            return real_var, std_var
    return None, None

def apply_gaussian_weights(grid_lats, grid_lons, station_lat, station_lon, sigma=0.1):
    """Apply Gaussian weights to the grid based on distances to the station."""
    dlat = np.radians(grid_lats - station_lat)
    dlon = np.radians(grid_lons - station_lon)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(station_lat)) * np.cos(np.radians(grid_lats)) * np.sin(dlon / 2) ** 2
    distances = 2 * np.arcsin(np.sqrt(a))
    weights = np.exp(-(distances ** 2) / (2 * sigma ** 2))
    return weights / np.sum(weights)  # Normalize weights

def interpolate_with_gaussian(data_array, weights):
    """Interpolate the data using Gaussian weights."""
    flat = data_array.reshape(data_array.shape[0], -1)  # Flatten spatial dimensions
    flat_weights = weights.ravel()  # Flatten weights
    weighted = np.sum(flat * flat_weights, axis=1)  # Apply weights and sum
    return weighted

# === MAIN SCRIPT ===
for filename in os.listdir(data_folder):
    if not filename.endswith(".nc"):
        continue

    filepath = os.path.join(data_folder, filename)
    print(f"Opening file: {filename}")

    try:
        ds = xr.open_dataset(filepath)
        ds = shift_and_fix_coords(ds)
    except Exception as e:
        print(f"Failed to open {filename}: {e}")
        continue

    real_var, std_var = extract_variable(ds)
    if not real_var:
        print(f"Skipping {filename}: no valid variable found.")
        continue

    data = ds[real_var]
    if "height" in data.dims:
        data = data.squeeze(dim="height", drop=True)  # Drop height dimension if present

    if not set(data.dims).issuperset({"time", "lat", "lon"}):
        print(f"Skipping {filename}: unsupported shape {data.shape}")
        continue

    print(f"Using variable: {std_var}")
    lats = ds["lat"].values
    lons = ds["lon"].values
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    for station_name, station_info in stations.items():
        print(f"Interpolating for station: {station_name}")
        weights = apply_gaussian_weights(lat_grid, lon_grid, station_info["lat"], station_info["lon"])
        result = interpolate_with_gaussian(data.values, weights)

        # Ensure the directory exists
        out_dir = output_base / station_name / std_var
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{station_name}_{std_var}_{filename}"
        xr.Dataset({std_var: (["time"], result)}, coords={"time": ds["time"]}).to_netcdf(out_path)
        print(f"Saved to: {out_path}")
