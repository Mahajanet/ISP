import os
import xarray as xr
import numpy as np

# Directory where the GRIB files are stored
input_dir = "V:/vedur/reikn/CARRA_ISL/T2M/t2m_3hr/one_year_per_gribfile"
output_dir = "V:/ofanflod/verk/vakt/steph/python/jahnavi/t2m_isa"

# GPS coordinates for Ísafjörður
lat_target = 66.0596
lon_target = -23.1699

# Loop through years from 1990 to 2024
for year in range(1990, 2025):
    file_name = f"CF_T2M_ISL{year}.grib"
    file_path = os.path.join(input_dir, file_name)

    # Check if the file exists
    if not os.path.isfile(file_path):
        print(f"File not found for year {year}: {file_path}")
        continue

    try:
        # Load GRIB file
        ds = xr.open_dataset(file_path, engine="cfgrib", backend_kwargs={'indexpath': ''})

        # Select nearest grid point
        selected_point = ds.sel(latitude=lat_target, longitude=lon_target, method="nearest")

        # Extract time and t2m (temperature at 2m)
        time = selected_point.time.values
        T2M = selected_point.t2m.values

        # Create new dataset
        ds_3hr = xr.Dataset(
            {"t2m": ("time", T2M)},
            coords={"time": time},
        )

        # Save to NetCDF
        output_path = os.path.join(output_dir, f"t2m_isa_{year}.nc")
        ds_3hr.to_netcdf(output_path)

        print(f"Saved: {output_path}")
       
    except Exception as e:
        print(f"Failed to process {file_name}: {e}")
