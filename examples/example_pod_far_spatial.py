import xarray as xr
import matplotlib.pyplot as plt
from nwpeval import pod, far

# File paths
model_file = "/Users/dev/PROJECTS/nwp_metrics_package/examples/india_model_output_005deg_irregular_storm.nc"
obs_file = "/Users/dev/PROJECTS/nwp_metrics_package/examples/india_obs_output_005deg_irregular_storm.nc"

# Read the model and observation data from NetCDF files
model_data = xr.open_dataset(model_file)
obs_data = xr.open_dataset(obs_file)

# Specify the variable names for model and observation data
model_var = 'lightning_density'  # Replace with the actual variable name from the model file
obs_var = 'lightning_density'  # Replace with the actual variable name from the observation file

obs = obs_data[obs_var]
model = model_data[model_var]
threshold = 0.0005

# POD and FAR are ratios of contingency-table counts. To aggregate them, pool
# the counts with `dim` rather than averaging per-time-step scores: an average
# weights a time step with one event the same as one with a thousand, and time
# steps without events have no POD at all.

# Maps: counts pooled over time at every grid point
pod_map = pod(obs, model, threshold=threshold, dim='time')
far_map = far(obs, model, threshold=threshold, dim='time')

# Create spatial plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

pod_map.plot(ax=ax1)
ax1.set_title('POD')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')

far_map.plot(ax=ax2)
ax2.set_title('FAR')
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')

plt.tight_layout()
plt.savefig('spatial_plots.png')

# Diurnal cycle: for each hour of day, counts pooled over space and all days
pairs = xr.Dataset({'obs': obs, 'model': model})
pod_diurnal = pairs.groupby('time.hour').map(lambda g: pod(g.obs, g.model, threshold=threshold))
far_diurnal = pairs.groupby('time.hour').map(lambda g: far(g.obs, g.model, threshold=threshold))

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(pod_diurnal.hour, pod_diurnal, label='POD')
ax.plot(far_diurnal.hour, far_diurnal, label='FAR')

ax.set_title('Diurnal Cycle of POD and FAR')
ax.set_xlabel('Hour')
ax.set_ylabel('Value')
ax.legend()

plt.tight_layout()
plt.savefig('diurnal_cycle_plot.png')
