import os
import re
import numpy as np
import xarray as xr

input_dir = "/Users/jahnavimahajan/Projects/ISP/carra_data"
output_dir = "/Users/jahnavimahajan/Projects/ISP/raw_data/elevation_adjusted"

station_meta = {
    "isa": {"lat": 66.0596, "lon": -23.1699, "elev": 2.2},
    "thver": {"lat": 66.0444, "lon": -23.3074, "elev": 741.0}
}

variables = {
    "wdir10": {"pattern": r"D10m.*\.nc", "fallback": "D10"},
    "si10": {"pattern": r"F10m.*\.nc", "fallback": "10si"},
    "pr": {"pattern": r"pr_daily_.*\.nc", "fallback": "pr"},
    "t2m": {"pattern": r"t2m_day_ISL.*\.nc", "fallback": "t2m"},
}

def get_nearest(ds, target_lat, target_lon):
    lats = ds['latitude']
    lons = ds['longitude']
    lat_idx = np.abs(lats - target_lat).argmin().item()
    lon_idx = np.abs(lons - target_lon).argmin().item()
    return lats[lat_idx].item(), lons[lon_idx].item(), lat_idx, lon_idx

def apply_elevation_correction(data, ds, lat_idx, lon_idx, station_elev, var):
    if 'height' not in ds.coords:
        return data  # no elevation data, skip correction

    # Assume grid elevation is constant across time
    try:
        grid_elev = float(ds['height'].values)
    except Exception:
        return data

    delta_h = station_elev - grid_elev  # in meters

    if var == 't2m':
        lapse_rate = -0.0065  # °C per meter
        correction = delta_h * lapse_rate
        return data + correction

    elif var == 'pr':
        scale = 1 + (delta_h / 1000) * 0.05  # ~5% per 1000m
        return data * scale

    elif var == 'si10':
        scale = 1 + (delta_h / 1000) * 0.05  # ~5% per 1000m
        return data * scale

    return data

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

                if var in ['t2m', 'pr', 'si10']:
                    print("      Applying elevation adjustment...")
                    data = apply_elevation_correction(data, ds, lat_idx, lon_idx, meta["elev"], var)

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
