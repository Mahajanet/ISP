# ISP Project

This repository contains all of the code, data and results for our interpolation and validation of CARRA meteorological fields against in‑situ observations at two stations (Ísafjörður “isa” and Þverá “thver”). All work is complete—below is a guided tour of the folder structure and what lives where.

---

## 📂 Root

- **`.gitignore`** – patterns for files we don’t track in Git  
- **`README.md`** – this file

---

## 📂 carra_data

Raw GRIB/NetCDF CARRA model output, organized by variable & year.  
This is the _source_ CARRA data before any elevation‑adjustment or interpolation.

---

## 📂 data_analysis

All of our analyses, comparisons and result‐generation scripts, organized by method and station.

### 📂 elevation_adjusted_…/isa & …/thver  
Scripts that apply a simple cell‑by‑cell elevation correction (lapse rate) to CARRA fields.  
Each station folder contains four scripts:
- `precip_<station>_ea_results.py`  
- `t2m_<station>_ea_results.py`  
- `wind_dir_<station>_ea_results.py`  
- `wind_speed_<station>_ea_results.py`  

These load the elevation‑adjusted files, compare to in‑situ, compute error metrics and produce time‑series & scatter plots.

### 📂 gaussian_results  
Results and comparison scripts for the 3×3 Gaussian interpolation method.  
Contains one subfolder per station (`isa`, `thver`), each with four `gauss_<var>_<station>_results.py` scripts.

### 📂 idw_results  
Same structure as Gaussian, but for the Inverse‑Distance‑Weighting (IDW) method.

### 📂 kriging_results  
Same structure again, but for Ordinary Kriging.

### 📂 nn_results  
Same structure for the Nearest‐Neighbor fallback method (used as a baseline).

---

## 📂 import_scripts

“Build” scripts that read raw CARRA input and write out daily NetCDFs under `raw_data`:

1. **`carra_multi_interpolation.py`** – driver for running all five interpolation schemes  
2. **`elevation_adjusted.py`** – applies lapse‑rate elevation correction  
3. **`gaussian.py`** – 3×3 Gaussian spatial interpolation  
4. **`idw.py`** – inverse‑distance weighting interpolation  
5. **`kriging.py`** – ordinary kriging interpolation  

They all share the same input/output logic: read from **`carra_data`**, write daily files into **`raw_data/<method>/<station>/<var>/…`**.

---

## 📂 helper_files

Utility modules used by the import scripts:

- **`get_nc_output.py`** – common NetCDF I/O helpers  
- **`gribtonetcdf.py`** – converts raw GRIB files to NetCDF  

---

## 📂 raw_data

All intermediate and final NetCDF files, organized by:
raw_data/
├── elevation_adjusted/… # output from elevation_adjusted.py
├── gaussian/… # output from gaussian.py
├── idw/… # output from idw.py
├── kriging/… # output from kriging.py
└── nn/… # output from nearest‑neighbor
└── in_situ.xlsx # station observations (Excel)

### How to reproduce

1. **Convert** raw GRIB → NetCDF with `helper_files/gribtonetcdf.py`.  
2. **Run** each import script in `import_scripts/` to build daily NetCDFs under `raw_data/`.  
3. **Run** the per‑method comparison scripts in `data_analysis/..._results/` to compute metrics and generate plots.

All code is ready to go—just adjust the paths at top of each script, and install necessary packages. 

