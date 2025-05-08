#!/usr/bin/env python3

import os
import math
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

# === CONFIGURATION ===
DATA_FOLDER      = Path("/Users/jahnavimahajan/Projects/ISP/carra_data")
OUTPUT_ROOT      = Path("/Users/jahnavimahajan/Projects/ISP/raw_data/gaussian")
STATIONS         = {
    "isa":   {"lat": 66.0596, "lon": -23.1699, "elev": 2.2},
    "thver": {"lat": 66.0444, "lon": -23.3074, "elev": 741.0},
}
VARIABLE_NAME_MAP = {
    "t2m":    "t2m", "T2m": "t2m",
    "pr":     "pr",
    "si10":   "si10", "10si": "si10",
    "wdir10": "wdir10", "D10":  "wdir10",
}
EARTH_R        = 6_371_000.0  # Earth radius [m]
Rp             = 50_000.0     # Gaussian radius [m]
LAPSE          = -0.0065      # default lapse‐rate (°C per m)
RESAMPLE_RULE  = "D"          # daily resampling rule
LOG            = "[CARA-GAUSS]"

#-------------------------------------------------------------------------------

def shift_and_fix_coords(ds):
    """Rename latitude/longitude to lat,lon and normalize lon to [-180,180]."""
    if "latitude" in ds.coords and "longitude" in ds.coords:
        ds = ds.rename({"latitude":"lat","longitude":"lon"})
    if "lon" in ds.coords:
        ds["lon"] = ((ds["lon"] + 180) % 360) - 180
        ds = ds.sortby("lon")
    return ds

def extract_variable(ds, fname):
    """Pick real_var,std_var from VARIABLE_NAME_MAP or fall back to first data_var."""
    for real, std in VARIABLE_NAME_MAP.items():
        if real in ds.data_vars:
            return real, std
    avail = list(ds.data_vars)
    print(f"{LOG}  • WARNING: no key for `{fname}`; available = {avail}")
    fb = avail[0]
    print(f"{LOG}  • falling back to `{fb}`")
    return fb, fb

def find_row_col(lat0, lon0, da, orog_arr):
    """Nearest grid‐cell indices for (lat0,lon0) in da + its elevation."""
    lats = da["lat"].values
    lons = da["lon"].values
    i    = int(np.argmin(np.abs(lats - lat0)))
    j    = int(np.argmin(np.abs(lons - lon0)))
    elev = float(orog_arr[i,j]) if orog_arr is not None else np.nan
    return i, j, elev

def find_row_col_neighbor(lat0, lon0, da, orog_arr, di, dj):
    """
    Offset by di,dj from the nearest cell to lat0,lon0.
    Returns i,j,cell_elev,cell_lat,cell_lon,dist_m.
    """
    i0, j0, _ = find_row_col(lat0, lon0, da, orog_arr)
    i  = max(0, min(i0+di, da["lat"].size-1))
    j  = max(0, min(j0+dj, da["lon"].size-1))
    latn = float(da["lat"].values[i])
    lonn = float(da["lon"].values[j])
    dlat = math.radians(latn - lat0)
    dlon = math.radians(lonn - lon0)
    a    = (math.sin(dlat/2)**2 +
            math.cos(math.radians(lat0)) *
            math.cos(math.radians(latn)) *
            math.sin(dlon/2)**2)
    dist = 2 * EARTH_R * math.asin(math.sqrt(a))
    elev = float(orog_arr[i,j]) if orog_arr is not None else np.nan
    return i, j, elev, latn, lonn, dist

#-------------------------------------------------------------------------------

def cara_interp(lat0, lon0, elev0, ds, real_var, std_var):
    """
    3×3 Gaussian‐weighted interpolation +
    dynamic lapse‐rate for t2m only if orography is present.
    Returns a pandas.Series, daily‐resampled.
    """
    # 1) Prepare variable array
    da = ds[real_var]
    for d in ("height","level","heightAboveGround"):
        if d in da.dims:
            da = da.squeeze({d:1}, drop=True)

    # 2) Extract orography if available
    orog_arr = None
    if "orog" in ds:
        og = ds["orog"]
        for d in ("height","level"):
            if d in og.dims:
                og = og.squeeze({d:1}, drop=True)
        orog_arr = og.values

    arr   = da.values.astype(float)               # shape (time, lat, lon)
    times = pd.to_datetime(da["time"].values)     # length T

    # 3) Gather 3×3 neighbours
    neigh_vals = []
    dz         = []
    dists      = []
    for di in (-1,0,1):
      for dj in (-1,0,1):
        i, j, cell_elev, _, _, dist = find_row_col_neighbor(
            lat0, lon0, da, orog_arr, di, dj
        )
        vals = da.isel(lat=i, lon=j).values.astype(float)  # (time,)
        neigh_vals.append(vals)
        dz.append(cell_elev - elev0)
        dists.append(dist)

    neigh_vals = np.vstack(neigh_vals)  # shape (9, T)
    dz         = np.array(dz)           # shape (9,)
    dists      = np.array(dists)        # shape (9,)

    # 4–5) **Only for temperature** and **only if** we have orog ⇒ dynamic lapse‐rate
    if std_var == "t2m" and orog_arr is not None:
        T = neigh_vals.shape[1]
        slopes = np.full(T, LAPSE, dtype=float)
        for t in range(T):
            y  = neigh_vals[:, t]
            ok = ~np.isnan(y) & ~np.isnan(dz)
            if ok.sum() >= 2:
                try:
                    slopes[t], _ = np.polyfit(dz[ok], y[ok], 1)
                except np.linalg.LinAlgError:
                    slopes[t] = LAPSE
        corrected = neigh_vals - slopes[None, :] * dz[:, None]
    else:
        # either not t2m or no orog ⇒ no vertical adjustment
        corrected = neigh_vals

    # 6) Horizontal Gaussian weights
    wts = np.exp(-0.5 * (dists / Rp)**2)
    wts /= wts.sum()

    # 7) Weighted sum → raw timeseries
    ts = corrected.T.dot(wts)  # shape (T,)

    # 8) Wrap in pandas.Series + daily resample
    s = pd.Series(ts, index=times)
    if std_var == "pr":
        s = s.resample(RESAMPLE_RULE).sum()
    else:
        s = s.resample(RESAMPLE_RULE).mean()

    return s

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"{LOG} start")

    for fname in sorted(os.listdir(DATA_FOLDER)):
        if not fname.endswith(".nc"):
            continue
        path = DATA_FOLDER / fname
        print(f"\n{LOG} opening `{fname}`")
        try:
            ds = xr.open_dataset(path)
        except Exception as e:
            print(f"{LOG}  ✗ open_dataset failed: {e}")
            continue

        ds = shift_and_fix_coords(ds)
        real_var, std_var = extract_variable(ds, fname)

        # dims check + squeeze vertical dims
        da = ds[real_var]
        for d in ("height","level"):
            if d in da.dims:
                da = da.squeeze({d:1}, drop=True)
        if not set(da.dims).issuperset({"time","lat","lon"}):
            print(f"{LOG}  • skipping `{fname}`: dims {da.dims}")
            ds.close()
            continue

        # process each station
        for station_name, info in STATIONS.items():
            print(f"{LOG}  → Station `{station_name}`")
            series = cara_interp(
                info["lat"], info["lon"], info["elev"],
                ds, real_var, std_var
            )

            out_ds  = xr.Dataset(
                {std_var: (["time"], series.values)},
                coords={"time": series.index.values}
            )
            out_dir = OUTPUT_ROOT / station_name / std_var
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{station_name}_{std_var}_{fname}"
            try:
                out_ds.to_netcdf(out_path)
                print(f"{LOG}    ✔ saved → {out_path}")
            except Exception as e:
                print(f"{LOG}    ✗ save failed: {e}")

        ds.close()

    print(f"\n{LOG} all done.")

