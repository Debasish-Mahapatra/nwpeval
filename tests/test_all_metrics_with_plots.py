"""
Comprehensive NWPeval metrics test with TIME SERIES visualization.
Uses Radar (observation) vs 2-Moment (model).
Computes metrics along lat/lon (keeping time axis) to get time series.
Saves all plots to /Users/dev/PLOTS/nwpeval-test/ at 450 dpi.
"""
import sys
sys.path.insert(0, '/Users/dev/PROJECTS/nwpeval-main')

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from nwpeval import (
    # Continuous metrics
    mae, rmse, acc, r2, nrmse, pcc, mbd,
    tse, evs, nmse, fv, sdr, vif,
    nae, rmb, mape, lmbe, smse, gmb, sbs, aev,
    cosine_similarity,
    # Spatial
    fss,
    # Categorical
    ets, pod, far, csi, hss, pss, gss, fb, hkd, orss,
    seds, eds, sedi, f1, mcc, ba, npv, jaccard, gain, lift,
    # Probabilistic
    bss, rpss,
)

# Create output directory
output_dir = '/Users/dev/PLOTS/nwpeval-test'
os.makedirs(output_dir, exist_ok=True)

# Load data
data_dir = '/Volumes/crucial-ssd/ALARO/manaus-production-runs/rainfall-regridded-to-imerge/masked-production-final/common-valid-time-production'
obs = xr.open_dataset(f'{data_dir}/Radar_common_valid.nc')['rainfall_rate']
model = xr.open_dataset(f'{data_dir}/2-Moment_common_valid.nc')['total_rain']

print(f"Obs shape: {obs.shape}, Model shape: {model.shape}")
print(f"Dims: {obs.dims}")

# Spatial dimensions for aggregation (keep time)
spatial_dims = ['lat', 'lon']
thresholds = [0.1, 1.0, 5.0]

# Get time coordinate
time_coord = obs.coords['time'].values if 'time' in obs.coords else np.arange(obs.shape[0])
print(f"Time points: {len(time_coord)}")

# ============================================================
# 1. CONTINUOUS METRICS TIME SERIES
# ============================================================
print("\n" + "="*60)
print("COMPUTING CONTINUOUS METRICS TIME SERIES")
print("="*60)

continuous_ts = {}

# MAE
print("  Computing MAE...")
continuous_ts['MAE'] = mae(obs, model, dim=spatial_dims)

# RMSE
print("  Computing RMSE...")
continuous_ts['RMSE'] = rmse(obs, model, dim=spatial_dims)

# PCC
print("  Computing PCC...")
continuous_ts['PCC'] = pcc(obs, model, dim=spatial_dims)

# MBD
print("  Computing MBD (Bias)...")
continuous_ts['MBD'] = mbd(obs, model, dim=spatial_dims)

# R2
print("  Computing R²...")
continuous_ts['R2'] = r2(obs, model, dim=spatial_dims)

# Plot continuous metrics time series
fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
axes = axes.flatten()

for i, (name, ts) in enumerate(continuous_ts.items()):
    if i >= len(axes):
        break
    ax = axes[i]
    ax.plot(time_coord, ts.values, 'b-', linewidth=0.5, alpha=0.7)
    # Add rolling mean
    window = min(100, len(ts)//10)
    if window > 1:
        rolling_mean = pd.Series(ts.values).rolling(window=window, center=True).mean()
        ax.plot(time_coord, rolling_mean, 'r-', linewidth=2, label=f'{window}-point mean')
    ax.set_ylabel(name, fontsize=12)
    ax.set_title(f'{name} Time Series', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

# Remove unused subplot
if len(continuous_ts) < len(axes):
    axes[-1].axis('off')

plt.xlabel('Time', fontsize=12)
plt.suptitle('Continuous Metrics Time Series: Radar vs 2-Moment', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/09_continuous_timeseries.png', dpi=450, bbox_inches='tight')
plt.close()
print("  Saved: 09_continuous_timeseries.png")

# ============================================================
# 2. CATEGORICAL METRICS TIME SERIES (for each threshold)
# ============================================================
print("\n" + "="*60)
print("COMPUTING CATEGORICAL METRICS TIME SERIES")
print("="*60)

categorical_funcs = {
    'POD': pod,
    'FAR': far,
    'CSI': csi,
    'ETS': ets,
    'HSS': hss,
    'FB': fb,
}

for thresh in thresholds:
    print(f"\n  Threshold = {thresh}")
    categorical_ts = {}
    
    for name, func in categorical_funcs.items():
        print(f"    Computing {name}...")
        try:
            result = func(obs, model, threshold=thresh, dim=spatial_dims)
            categorical_ts[name] = result
        except Exception as e:
            print(f"    ERROR: {e}")
            categorical_ts[name] = None
    
    # Plot
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    
    for i, (name, ts) in enumerate(categorical_ts.items()):
        if ts is None:
            continue
        ax = axes[i]
        ax.plot(time_coord, ts.values, 'b-', linewidth=0.5, alpha=0.7)
        # Rolling mean
        window = min(100, len(ts)//10)
        if window > 1:
            rolling_mean = pd.Series(ts.values).rolling(window=window, center=True).mean()
            ax.plot(time_coord, rolling_mean, 'r-', linewidth=2, label=f'{window}-point mean')
        ax.set_ylabel(name, fontsize=12)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # Set reasonable y-limits for ratios
        if name in ['POD', 'FAR', 'CSI', 'ETS', 'HSS']:
            ax.set_ylim(-0.1, 1.1)
    
    plt.xlabel('Time', fontsize=12)
    plt.suptitle(f'Categorical Metrics Time Series (Threshold={thresh})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/10_categorical_timeseries_thresh{thresh}.png', dpi=450, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 10_categorical_timeseries_thresh{thresh}.png")

# ============================================================
# 3. FSS TIME SERIES (for different neighborhoods)
# ============================================================
print("\n" + "="*60)
print("COMPUTING FSS TIME SERIES")
print("="*60)

neighborhood_sizes = [3, 9, 21]
thresh = 1.0  # Use single threshold for FSS time series

fig, axes = plt.subplots(len(neighborhood_sizes), 1, figsize=(14, 10), sharex=True)

for i, n_size in enumerate(neighborhood_sizes):
    print(f"  Computing FSS with neighborhood={n_size}...")
    try:
        fss_ts = fss(obs, model, threshold=thresh, neighborhood_size=n_size, 
                     spatial_dims=['lat', 'lon'], reduction_dim=['lat', 'lon'])
        
        ax = axes[i]
        ax.plot(time_coord, fss_ts.values, 'b-', linewidth=0.5, alpha=0.7)
        
        # Rolling mean
        window = min(100, len(fss_ts)//10)
        if window > 1:
            rolling_mean = pd.Series(fss_ts.values).rolling(window=window, center=True).mean()
            ax.plot(time_coord, rolling_mean, 'r-', linewidth=2, label=f'{window}-point mean')
        
        ax.axhline(y=0.5, color='green', linestyle='--', linewidth=1, label='Skillful threshold')
        ax.set_ylabel(f'FSS (n={n_size})', fontsize=12)
        ax.set_title(f'Neighborhood = {n_size} grid points', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
    except Exception as e:
        print(f"    ERROR: {e}")
        axes[i].text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=axes[i].transAxes)

plt.xlabel('Time', fontsize=12)
plt.suptitle(f'FSS Time Series (Threshold={thresh})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/11_fss_timeseries.png', dpi=450, bbox_inches='tight')
plt.close()
print("  Saved: 11_fss_timeseries.png")

# ============================================================
# 4. COMBINED SKILL SCORES TIME SERIES
# ============================================================
print("\n" + "="*60)
print("COMPUTING COMBINED SKILL SCORES")
print("="*60)

skill_metrics = ['ETS', 'HSS', 'CSI', 'POD']
thresh = 1.0

fig, ax = plt.subplots(figsize=(14, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for name, color in zip(skill_metrics, colors):
    print(f"  Computing {name}...")
    func = {'ETS': ets, 'HSS': hss, 'CSI': csi, 'POD': pod}[name]
    ts = func(obs, model, threshold=thresh, dim=spatial_dims)
    
    # Plot rolling mean only for clarity
    window = min(100, len(ts)//10)
    if window > 1:
        rolling_mean = pd.Series(ts.values).rolling(window=window, center=True).mean()
        ax.plot(time_coord, rolling_mean, '-', linewidth=2, color=color, label=name)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Skill Score', fontsize=12)
ax.set_title(f'Skill Scores Time Series (Threshold={thresh}, Rolling Mean)', fontsize=14, fontweight='bold')
ax.set_ylim(-0.1, 1.1)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/12_skill_scores_combined.png', dpi=450, bbox_inches='tight')
plt.close()
print("  Saved: 12_skill_scores_combined.png")

# ============================================================
# 5. BIAS AND ERROR TIME SERIES
# ============================================================
print("\n" + "="*60)
print("COMPUTING BIAS AND ERROR TIME SERIES")
print("="*60)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# MAE and RMSE
ax = axes[0]
print("  Plotting MAE and RMSE...")
window = min(100, len(continuous_ts['MAE'])//10)

mae_rolling = pd.Series(continuous_ts['MAE'].values).rolling(window=window, center=True).mean()
rmse_rolling = pd.Series(continuous_ts['RMSE'].values).rolling(window=window, center=True).mean()

ax.plot(time_coord, mae_rolling, 'b-', linewidth=2, label='MAE')
ax.plot(time_coord, rmse_rolling, 'r-', linewidth=2, label='RMSE')
ax.set_ylabel('Error (mm/h)', fontsize=12)
ax.set_title('Error Metrics Time Series', fontsize=12, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Bias (MBD)
ax = axes[1]
print("  Plotting Bias...")
mbd_rolling = pd.Series(continuous_ts['MBD'].values).rolling(window=window, center=True).mean()
ax.plot(time_coord, mbd_rolling, 'g-', linewidth=2, label='Mean Bias')
ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.fill_between(time_coord, mbd_rolling, 0, alpha=0.3, 
                color='green' if np.nanmean(mbd_rolling) >= 0 else 'red')
ax.set_ylabel('Bias (mm/h)', fontsize=12)
ax.set_xlabel('Time', fontsize=12)
ax.set_title('Mean Bias Time Series', fontsize=12, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.suptitle('Error and Bias: Radar vs 2-Moment', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/13_error_bias_timeseries.png', dpi=450, bbox_inches='tight')
plt.close()
print("  Saved: 13_error_bias_timeseries.png")

# ============================================================
# 6. DIURNAL CYCLE (if time is datetime)
# ============================================================
print("\n" + "="*60)
print("COMPUTING DIURNAL CYCLE")
print("="*60)

try:
    # Try to extract hour from time coordinate
    time_pd = pd.to_datetime(time_coord)
    hours = time_pd.hour
    
    # Group metrics by hour
    metrics_by_hour = {
        'MAE': [],
        'POD': [],
        'CSI': [],
    }
    
    # Compute POD and CSI time series
    print("  Computing POD time series...")
    pod_ts = pod(obs, model, threshold=1.0, dim=spatial_dims)
    print("  Computing CSI time series...")
    csi_ts = csi(obs, model, threshold=1.0, dim=spatial_dims)
    
    for hour in range(24):
        mask = hours == hour
        metrics_by_hour['MAE'].append(np.nanmean(continuous_ts['MAE'].values[mask]))
        metrics_by_hour['POD'].append(np.nanmean(pod_ts.values[mask]))
        metrics_by_hour['CSI'].append(np.nanmean(csi_ts.values[mask]))
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    for ax, (name, values) in zip(axes, metrics_by_hour.items()):
        ax.bar(range(24), values, color='steelblue', edgecolor='black')
        ax.set_xlabel('Hour (UTC)', fontsize=12)
        ax.set_ylabel(name, fontsize=12)
        ax.set_title(f'{name} Diurnal Cycle', fontsize=12, fontweight='bold')
        ax.set_xticks(range(0, 24, 3))
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Diurnal Cycle of Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/14_diurnal_cycle.png', dpi=450, bbox_inches='tight')
    plt.close()
    print("  Saved: 14_diurnal_cycle.png")
    
except Exception as e:
    print(f"  Could not compute diurnal cycle: {e}")

# ============================================================
print("\n" + "="*60)
print(f"ALL PLOTS SAVED TO: {output_dir}")
print("="*60)
print("\nFiles created:")
for f in sorted(os.listdir(output_dir)):
    if f.endswith('.png'):
        print(f"  - {f}")
