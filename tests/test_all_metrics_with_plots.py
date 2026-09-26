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

# The thick lines pool all points of consecutive blocks of time steps and score
# them together. A rolling mean of per-step scores would weight a step with one
# event like a step with a thousand, and a dry step has no POD or FSS at all.
window = max(1, min(100, len(time_coord) // 10))


def pooled_blocks(func, **kwargs):
    """Score each block of `window` time steps over all its points.

    Returns the time at the centre of each block and the scores. Pass the
    reduction over 'step' and space in kwargs (dim=..., or reduction_dim= for FSS).
    """
    o = obs.coarsen(time=window, boundary='trim').construct(time=('block', 'step'))
    m = model.coarsen(time=window, boundary='trim').construct(time=('block', 'step'))
    n_blocks = o.sizes['block']
    centre = time_coord[:n_blocks * window].reshape(n_blocks, window)[:, window // 2]
    return centre, func(o, m, **kwargs).values


pooled_label = f'pooled over {window} steps'

# ============================================================
# 1. CONTINUOUS METRICS TIME SERIES
# ============================================================
print("\n" + "="*60)
print("COMPUTING CONTINUOUS METRICS TIME SERIES")
print("="*60)

continuous_funcs = {'MAE': mae, 'RMSE': rmse, 'PCC': pcc, 'MBD': mbd, 'R2': r2}
continuous_ts = {}
for name, func in continuous_funcs.items():
    print(f"  Computing {name}...")
    continuous_ts[name] = func(obs, model, dim=spatial_dims)

# Plot continuous metrics time series
fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
axes = axes.flatten()

for i, (name, ts) in enumerate(continuous_ts.items()):
    if i >= len(axes):
        break
    ax = axes[i]
    ax.plot(time_coord, ts.values, 'b-', linewidth=0.5, alpha=0.7)
    if window > 1:
        centre, pooled = pooled_blocks(continuous_funcs[name], dim=['step'] + spatial_dims)
        ax.plot(centre, pooled, 'r-', linewidth=2, label=pooled_label)
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
        if window > 1:
            centre, pooled = pooled_blocks(categorical_funcs[name], threshold=thresh,
                                           dim=['step'] + spatial_dims)
            ax.plot(centre, pooled, 'r-', linewidth=2, label=pooled_label)
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

        if window > 1:
            centre, pooled = pooled_blocks(fss, threshold=thresh, neighborhood_size=n_size,
                                           spatial_dims=['lat', 'lon'],
                                           reduction_dim=['step', 'lat', 'lon'])
            ax.plot(centre, pooled, 'r-', linewidth=2, label=pooled_label)

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

    # Plot the pooled blocks only for clarity
    if window > 1:
        centre, pooled = pooled_blocks(func, threshold=thresh, dim=['step'] + spatial_dims)
        ax.plot(centre, pooled, '-', linewidth=2, color=color, label=name)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Skill Score', fontsize=12)
ax.set_title(f'Skill Scores Time Series (Threshold={thresh}, {pooled_label})', fontsize=14, fontweight='bold')
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
reduce_blocks = ['step'] + spatial_dims

centre, mae_pooled = pooled_blocks(mae, dim=reduce_blocks)
_, rmse_pooled = pooled_blocks(rmse, dim=reduce_blocks)

ax.plot(centre, mae_pooled, 'b-', linewidth=2, label='MAE')
ax.plot(centre, rmse_pooled, 'r-', linewidth=2, label='RMSE')
ax.set_ylabel('Error (mm/h)', fontsize=12)
ax.set_title(f'Error Metrics Time Series ({pooled_label})', fontsize=12, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Bias (MBD)
ax = axes[1]
print("  Plotting Bias...")
_, mbd_pooled = pooled_blocks(mbd, dim=reduce_blocks)
ax.plot(centre, mbd_pooled, 'g-', linewidth=2, label='Mean Bias')
ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.fill_between(centre, mbd_pooled, 0, alpha=0.3,
                color='green' if np.nanmean(mbd_pooled) >= 0 else 'red')
ax.set_ylabel('Bias (mm/h)', fontsize=12)
ax.set_xlabel('Time', fontsize=12)
ax.set_title(f'Mean Bias Time Series ({pooled_label})', fontsize=12, fontweight='bold')
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
    # For each hour of day, pool all points of all days at that hour and score
    # them together (averaging per-step scores is not the same thing).
    # xr.align with join='exact' raises if the grids differ: building the
    # Dataset directly would pad a mismatch with NaN instead.
    obs_a, model_a = xr.align(obs, model, join='exact')
    by_hour = xr.Dataset({'obs': obs_a, 'model': model_a}).groupby('time.hour')

    print("  Computing MAE, POD and CSI for each hour...")
    metrics_by_hour = {
        'MAE': by_hour.map(lambda g: mae(g.obs, g.model)),
        'POD': by_hour.map(lambda g: pod(g.obs, g.model, threshold=1.0)),
        'CSI': by_hour.map(lambda g: csi(g.obs, g.model, threshold=1.0)),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, (name, values) in zip(axes, metrics_by_hour.items()):
        ax.bar(values['hour'], values, color='steelblue', edgecolor='black')
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
