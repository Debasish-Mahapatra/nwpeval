import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from nwpeval import NWP_Stats

# File paths
model_file = "/Users/dev/PROJECTS/nwp_metrics_package/examples/india_model_output_005deg_irregular_storm.nc"
obs_file = "/Users/dev/PROJECTS/nwp_metrics_package/examples/india_obs_output_005deg_irregular_storm.nc"

# Read the model and observation data from NetCDF files
model_data = xr.open_dataset(model_file)
obs_data = xr.open_dataset(obs_file)

# Specify the variable names for model and observation data
model_var = 'lightning_density'  # Replace with the actual variable name from the model file
obs_var = 'lightning_density'  # Replace with the actual variable name from the observation file

# Create an instance of NWPMetrics
metrics = NWP_Stats(obs_data[obs_var], model_data[model_var])

# Calculate POD and FAR spatially at every time step
pod_values = metrics.compute_pod(threshold=0.0005, dim=None)
far_values = metrics.compute_far(threshold=0.0005, dim=None)

# Compute the spatial average of POD and FAR at each time step
#pod_spatial_avg = pod_values.mean(dim=['lat', 'lon'])
#far_spatial_avg = far_values.mean(dim=['lat', 'lon'])

pod_temporal_avg = pod_values.mean(dim='time')
far_temporal_avg = far_values.mean(dim='time')

# Create spatial plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot the temporally averaged POD values
pod_temporal_avg.plot(ax=ax1)
ax1.set_title('Temporally Averaged POD')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')

# Plot the temporally averaged FAR values
far_temporal_avg.plot(ax=ax2)
ax2.set_title('Temporally Averaged FAR')
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')

plt.tight_layout()
plt.savefig('spatial_plots.png')

# Create a diurnal cycle plot
fig, ax = plt.subplots(figsize=(8, 6))

# Compute the spatial average of POD and FAR at each time step
pod_spatial_avg = pod_values.mean(dim=['lon', 'lat'])
far_spatial_avg = far_values.mean(dim=['lon', 'lat'])

# Plot the time series of spatially averaged POD and FAR
ax.plot(pod_spatial_avg.time, pod_spatial_avg, label='POD')
ax.plot(far_spatial_avg.time, far_spatial_avg, label='FAR')

ax.set_title('Diurnal Cycle of POD and FAR')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()

plt.tight_layout()
plt.savefig('diurnal_cycle_plot.png')