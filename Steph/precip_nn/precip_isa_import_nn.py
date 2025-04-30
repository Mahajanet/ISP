import os
import xarray as xr
import numpy as np
import pandas as pd

# Directory paths
input_dir = "V:/vedur/reikn/CARRA_ISL/PR/pr_3hr/Monthly_files"
output_dir = "V:/ofanflod/verk/vakt/steph/python/jahnavi/precip_nn/precip_isa_nn"

# GPS coordinates for Ísafjörður
lat_target = 66.0596
lon_target = -23.1699

# Create list of year-month strings from 2020-01 to 2025-01
months = pd.date_range(start="2020-01", end="2025-01", freq="MS").strftime("%Y-%m").tolist()

# Loop through each year_month
for year_month in months:
    file_name = f"CF_PR_{year_month}.nc"
    file_path = os.path.join(input_dir, file_name)

    if not os.path.isfile(file_path):
        print(f"File not found for {year_month}: {file_path}")
        continue

    try:
        # Load NetCDF file
        ds = xr.open_dataset(file_path)

        # Select nearest grid point
        selected_point = ds.sel(latitude=lat_target, longitude=lon_target, method="nearest")

        # Extract time and precipitation
        time = selected_point.time.values
        pr = selected_point.pr.values  # You had a typo (T2M before!)

        # Create new dataset
        ds_3hr = xr.Dataset(
            {"pr": ("time", pr)},
            coords={"time": time},
        )

        # Save to NetCDF
        output_path = os.path.join(output_dir, f"pr_isa_{year_month}.nc")
        ds_3hr.to_netcdf(output_path)

        print(f"Saved: {output_path}")

    except Exception as e:
        print(f"Failed to process {file_name}: {e}")
