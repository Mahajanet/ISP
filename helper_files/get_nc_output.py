import xarray as xr

files = [
    "/Users/jahnavimahajan/Projects/ISP/carra_data/D10m2020_daily.nc",
    "/Users/jahnavimahajan/Projects/ISP/carra_data/F10m2022_daily.nc",
    "/Users/jahnavimahajan/Projects/ISP/carra_data/pr_daily_2023.nc",
    "/Users/jahnavimahajan/Projects/ISP/carra_data/t2m_day_ISL2024.nc"
]

for file in files:
    print(f"=== {file} ===")
    ds = xr.open_dataset(file)
    print(ds)
    print("\n")
