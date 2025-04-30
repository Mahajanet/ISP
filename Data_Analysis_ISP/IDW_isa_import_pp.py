import os
import xarray as xr
import numpy as np
import pandas as pd
from haversine import haversine

# Directory paths
input_dir = "V:/vedur/reikn/CARRA_ISL/PR/pr_3hr/Monthly_files"
output_dir = "V:/ofanflod/verk/vakt/steph/python/jahnavi/precip_idw/precip_isa_idw"

lat_target = 66.0596
lon_target = -23.1699
radius_km = 50  # Radius around station to include for interpolation

months = pd.date_range(start="2020-01", end="2025-01", freq="MS").strftime("%Y-%m").tolist()

for year_month in months:
    file_path = os.path.join(input_dir, f"CF_PR_{year_month}.nc")
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        continue

    try:
        ds = xr.open_dataset(file_path)

        # Extract lat/lon grid
        lats = ds.latitude.values
        lons = ds.longitude.values
        latlon_grid = np.array([(lat, lon) for lat in lats for lon in lons])
        distances = np.array([haversine((lat_target, lon_target), (lat, lon)) for lat, lon in latlon_grid])

        # Mask points within radius
        mask = distances <= radius_km
        if not np.any(mask):
            print(f"No grid points found within radius for {year_month}")
            continue

        # Reshape to grid
        distances = distances[mask]
        latlon_grid = latlon_grid[mask]
        weights = 1 / (distances ** 2)

        # Get values for those grid points
        pr_list = []
        for lat, lon in latlon_grid:
            pr_val = ds.sel(latitude=lat, longitude=lon, method="nearest").pr
            pr_list.append(pr_val)

        pr_stack = xr.concat(pr_list, dim="points")
        pr_weighted = (pr_stack * weights[:, None]).sum(dim="points") / np.sum(weights)

        # Save output
        ds_3hr = xr.Dataset({"pr": ("time", pr_weighted)}, coords={"time": ds.time.values})
        output_path = os.path.join(output_dir, f"pr_isa_{year_month}.nc")
        ds_3hr.to_netcdf(output_path)
        print(f"Saved: {output_path}")

    except Exception as e:
        print(f"Error for {year_month}: {e}")
