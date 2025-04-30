import os
import xarray as xr
import numpy as np

# Directory where the GRIB files are stored
input_dir = "V:/vedur/reikn/CARRA_ISL/F10m/f10m_3hr/one_year_per_gribfile"
output_dir = "V:/ofanflod/verk/vakt/steph/python/jahnavi/wind_speed_nn/f10m_isa_nn"

# GPS coordinates for Ísafjörður
lat_target = 66.0596
lon_target = -23.1699

# Loop through years from 2020 to 2025
for year in range(2020, 2025):
    file_name = f"CF_F10m{year}.grib"
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

        # Extract time and f10m (wind force at 10 m)
        time = selected_point.time.values
        f10m = selected_point.si10.values

        # Create new dataset
        ds_3hr = xr.Dataset(
            {"f10m": ("time", f10m)},
            coords={"time": time},
        )

        # Save to NetCDF
        output_path = os.path.join(output_dir, f"f10m_isa_{year}.nc")
        ds_3hr.to_netcdf(output_path)

        print(f"Saved: {output_path}")
        
    except Exception as e:
        print(f"Failed to process {file_name}: {e}")