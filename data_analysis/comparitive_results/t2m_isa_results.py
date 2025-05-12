import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import numpy as np

# 1) Load & resample each CARRA method into a daily‐mean Series (°C)
patterns = {
    'Elev-Adjusted':    "raw_data/elevation_adjusted/isa/t2m/t2m_isa_*.nc",
    'Gaussian':         "raw_data/gaussian/isa/t2m/isa_t2m_*.nc",
    'IDW':              'raw_data/idw/isa/t2m/isa_t2m_t2m_day_ISL*.nc',
    'Kriging':          "raw_data/kriging/isa/t2m/t2m_isa_t2m_day_ISL*.nc",
    'Nearest Neighbor': 'raw_data/nn/t2m_nn/t2m_isa_nn/t2m_isa_*.nc',
}

def load_daily(pattern):
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files for pattern: {pattern}")
    ds = xr.concat([xr.open_dataset(f) for f in files], dim='time')
    times = pd.to_datetime(ds['time'].values)
    vals_K = ds['t2m'].values.ravel()
    vals_C = vals_K - 273.15
    return pd.Series(vals_C, index=times).resample('D').mean()

carr = {name: load_daily(pat) for name, pat in patterns.items()}

# 2) Load & resample in-situ station data
df0 = (
    pd.read_excel(
        'raw_data/in_situ.xlsx',
        sheet_name='Observations - 2642',
        parse_dates=['TIMI']
    )
    .set_index('TIMI')
)
carr['In Situ'] = df0['T'].dropna().resample('D').mean()

# 3) Align on days present in all series
df_daily = pd.DataFrame(carr).dropna()

# 4) Aggregate to quarterly means
df_quarterly = df_daily.resample('Q').mean()

# --- Generate x-axis labels like 'Q1 2020', 'Q2 2020', etc.
quarters = df_quarterly.index.to_period('Q')
quarter_labels = [f"Q{q.quarter} {q.year}" for q in quarters]

# 5) Plot grouped bar chart with proper labels
methods = df_quarterly.columns.tolist()
n = len(methods)
x = np.arange(len(df_quarterly))        # one group per quarter
width = 0.8 / n                          # width of each bar

fig, ax = plt.subplots(figsize=(14, 6))

for i, m in enumerate(methods):
    ax.bar(
        x + i * width,
        df_quarterly[m],
        width,
        label=m
    )

# Center x-ticks and apply custom labels
group_centers = x + (n * width) / 2
ax.set_xticks(group_centers)
ax.set_xticklabels(quarter_labels, rotation=45)

# Styling
ax.set_title('Quarterly Mean 2 m Temperature: CARRA Methods vs In Situ (Ísafjörður)', fontsize=16)
ax.set_ylabel('Temperature (°C)', fontsize=14)
ax.legend(frameon=False, ncol=2, fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('quarterly_t2m_comparison_bars.png', dpi=300)
plt.show()
