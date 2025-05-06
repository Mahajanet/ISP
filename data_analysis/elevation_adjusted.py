import os
import re
import numpy as np
import xarray as xr

input_dir = "/Users/jahnavimahajan/Projects/ISP/carra_data"
output_dir = "/Users/jahnavimahajan/Projects/ISP/raw_data/elevation_adjusted"

station_meta = {
    "isa": {"lat": 66.0596, "lon": -23.1699},
    "thver": {"lat": 66.0444, "lon": -23.3074}
}

variables = {
    "wdir10": {"pattern": r"D10m.*\.nc", "fallback": "D10"},
    "si10": {"pattern": r"F10m.*\.nc", "fallback": "10si"},
    "pr": {"pattern": r"pr_daily_.*\.nc", "fallback": "pr"},
    "t2m": {"pattern": r"t2m_day_ISL.*\.nc", "fallback": "t2m"},
}

def get_nearest(ds, target_lat, target_lon):
    if 'latitude' not in ds.coords or 'longitude' not in ds.coords:
        raise ValueError("Missing 'latitude' or 'longitude' in dataset coords.")

    lats = ds['latitude']
    lons = ds['longitude']

    if lats.ndim != 1 or lons.ndim != 1:
        raise ValueError("Expected 1D latitude and longitude coordinates.")

    lat_idx = np.abs(lats - target_lat).argmin().item()
    lon_idx = np.abs(lons - target_lon).argmin().item()
    return lats[lat_idx].item(), lons[lon_idx].item(), lat_idx, lon_idx

def process_file(file_path, var, var_info):
    filename = os.path.basename(file_path)
    year_match = re.findall(r"\d{4}", filename)
    year = year_match[0] if year_match else "unknown"

    try:
        ds = xr.open_dataset(file_path)

        # Rename coordinates if necessary
        rename_coords = {}
        if 'lat' in ds.coords: rename_coords['lat'] = 'latitude'
        if 'lon' in ds.coords: rename_coords['lon'] = 'longitude'
        if rename_coords:
            print(f"      Renaming coordinates: {rename_coords}")
            ds = ds.rename(rename_coords)

        # Fix longitudes
        if (ds.longitude > 180).any():
            print("      Adjusting longitudes from 0–360 to -180–180")
            ds['longitude'] = (((ds['longitude'] + 180) % 360) - 180).sortby(ds['longitude'])

        # Confirm variable to use
        if var not in ds.data_vars:
            if var_info["fallback"] in ds.data_vars:
                print(f"      Using fallback variable '{var_info['fallback']}' instead of expected '{var}'")
                var_to_use = var_info["fallback"]
            else:
                raise ValueError(f"Variable '{var}' and fallback '{var_info['fallback']}' not found. Available: {list(ds.data_vars)}")
        else:
            var_to_use = var

        for station, meta in station_meta.items():
            print(f"    Station: {station} @ ({meta['lat']}, {meta['lon']})")
            try:
                lat_val, lon_val, lat_idx, lon_idx = get_nearest(ds, meta["lat"], meta["lon"])
                print(f"      Nearest point coords: lat={lat_val}, lon={lon_val}")

                data = ds[var_to_use].isel(latitude=lat_idx, longitude=lon_idx)

                out_dir = os.path.join(output_dir, station, var)
                os.makedirs(out_dir, exist_ok=True)
                out_file = os.path.join(out_dir, f"{var}_{station}_{year}.nc")

                data.to_netcdf(out_file)
                print(f"      Saved: {out_file}")
            except Exception as e:
                print(f"      ERROR processing {filename}: \"{e}\"")
    except Exception as e:
        print(f"    ERROR opening {filename}: \"{e}\"")

def main():
    print(f"Scanning input directory: {input_dir}\n")

    all_errors = []

    for var, info in variables.items():
        print(f"Processing variable: {var}")
        pattern = re.compile(info["pattern"])
        files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if pattern.match(f)]
        print(f"    Found {len(files)} files matching '{info['pattern']}'\n")

        for file_path in sorted(files):
            print(f"  File: {os.path.basename(file_path)}")
            try:
                process_file(file_path, var, info)
            except Exception as e:
                all_errors.append((file_path, str(e)))
                print(f"    General error: {e}")
            print()

    if all_errors:
        print("Processing completed with some warnings or errors.")
    else:
        print("✅ Processing complete.")

if __name__ == "__main__":
    main()
