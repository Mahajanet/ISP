import os
import xarray as xr

# -------- CONFIGURATION --------
input_path = "/Users/jahnavimahajan/Downloads/CF_D10m2024.grib"
output_dir = "/Users/jahnavimahajan/Desktop/netcdf_output"
output_filename = "D10m2024_daily.nc"
output_path = os.path.join(output_dir, output_filename)

# -------- ENSURE OUTPUT DIRECTORY EXISTS --------
os.makedirs(output_dir, exist_ok=True)

# -------- LOAD GRIB FILE --------
print(f"Opening GRIB file: {input_path}")
ds = xr.open_dataset(input_path, engine="cfgrib")

# -------- FIX LONGITUDE IF NEEDED --------
if (ds.longitude > 180).any():
    print("Fixing longitude range from 0–360 to -180–180...")
    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
    ds = ds.sortby("longitude")

# -------- RESAMPLE TO DAILY IF TIME EXISTS --------
if "time" in ds.coords:
    print("Resampling to daily means...")
    ds = ds.resample(time="1D").mean()

# -------- SAVE TO NETCDF --------
try:
    print(f"Saving NetCDF to: {output_path}")
    ds.to_netcdf(output_path)
    print("Done! File saved successfully.")
except Exception as e:
    print(f"Failed to save NetCDF: {e}")
