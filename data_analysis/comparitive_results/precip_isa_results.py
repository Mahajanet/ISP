import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import numpy as np

# 1) Load & resample each CARRA method into a daily‐sum Series
patterns = {
    'Elev-Adjusted':    'raw_data/elevation_adjusted/isa/pr/pr_isa_*.nc',
    'Gaussian':         'raw_data/gaussian/isa/pr/isa_pr_*.nc',
    'IDW':              'raw_data/idw/isa/pr/isa_pr_*.nc',
    'Kriging':          'raw_data/kriging/isa/pr/pr_isa_pr_daily_*.nc',
    'Nearest Neighbor': 'raw_data/nn/precip_nn/precip_isa_nn/pr_isa_*.nc',
}

def load_daily(pattern):
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files for pattern: {pattern}")
    ds = xr.concat([xr.open_dataset(f) for f in files], dim='time')
    times = pd.to_datetime(ds['time'].values)
    vals  = ds['pr'].values.ravel()
    return pd.Series(vals, index=times).resample('D').sum()

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
carr['In Situ'] = df0['R'].dropna().resample('D').sum()

# 3) Align on days present in all series
df_daily = pd.DataFrame(carr).dropna()

# 4) Aggregate to quarterly sums
df_quarterly = df_daily.resample('Q').sum()

# --- Generate x-axis labels like 'Q1 2020', 'Q2 2020', etc.
quarters = df_quarterly.index.to_period('Q')
quarter_labels = [f"Q{q.quarter} {q.year}" for q in quarters]

# 5) Plot grouped bar chart with correct x-axis labels
methods = df_quarterly.columns.tolist()
n = len(methods)
x = np.arange(len(df_quarterly))        # one group per quarter
width = 0.8 / n                         # width of each bar

fig, ax = plt.subplots(figsize=(14, 6))

for i, m in enumerate(methods):
    ax.bar(
        x + i * width,
        df_quarterly[m],
        width,
        label=m
    )

# Compute centers of each 6‐bar group
group_centers = x + (n * width) / 2

# Show tick labels for each quarter
ax.set_xticks(group_centers)
ax.set_xticklabels(quarter_labels, rotation=45)

# Styling
ax.set_title('Quarterly Total Precipitation: CARRA Methods vs In Situ (Ísafjörður)', fontsize=16)
ax.set_ylabel('Precipitation (mm quarter⁻¹)', fontsize=14)
ax.legend(frameon=False, ncol=2, fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('quarterly_precip_comparison_bars.png', dpi=300)
plt.show()
