import os
import xarray as xr
import numpy as np
import pandas as pd

# --- Paths ---
input_dir = "V:/vedur/reikn/CARRA_ISL/T2M/t2m_3hr/Monthly_files"  # Adjust for temperature directory
output_dir = "V:/ofanflod/verk/vakt/steph/python/jahnavi/temp_elev_adj/temp_isa_elev"

# --- Ísafjörður Station Info ---
lat_target = 66.0596
lon_target = -23.1699
station_elevation = 20  # in meters (actual station elevation)
lapse_rate = -6.5 / 1000  # °C per meter

# --- Month Loop Setup ---
months = pd.date_range(start="2020-01", end="2025-01", freq="MS").strftime("%Y-%m").tolist()

# --- Loop through each month ---
for year_month in months:
    file_name = f"CF_T2M_{year_month}.nc"
    file_path = os.path.join(input_dir, file_name)

    if not os.path.isfile(file_path):
        print(f"File not found for {year_month}: {file_path}")
        continue

    try:
        # Load the NetCDF file
        ds = xr.open_dataset(file_path)

        # Select nearest grid cell
        nearest_point = ds.sel(latitude=lat_target, longitude=lon_target, method="nearest")

        # Extract time and temperature
        time = nearest_point.time.values
        t2m_raw = nearest_point.T2M.values

        # --- Elevation Adjustment ---
        # Try to get grid elevation (if it's stored in the dataset as a variable or attribute)
        if 'z' in nearest_point:
            grid_elevation = nearest_point.z.values
        else:
            # Manually set approximate grid elevation (you can refine this if needed)
            grid_elevation = 100  # meters — replace if known from metadata

        # Apply elevation correction
        elevation_diff = station_elevation - grid_elevation
        t2m_corrected = t2m_raw + lapse_rate * elevation_diff

        # Save to new dataset
        ds_corrected = xr.Dataset(
            {"T2M_corrected": ("time", t2m_corrected)},
            coords={"time": time}
        )

        output_path = os.path.join(output_dir, f"T2M_isa_{year_month}.nc")
        ds_corrected.to_netcdf(output_path)
        print(f"Saved: {output_path}")

    except Exception as e:
        print(f"Failed to process {file_name}: {e}")
