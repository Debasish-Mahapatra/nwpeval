import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import nwpeval as nw

# Load observation and model data
obs_data = xr.open_dataset("india_obs_output_1km_realistic_storm.nc")
model_data = xr.open_dataset("india_model_output_1km_realistic_storm.nc")

# Extract lightning density variables
obs_lightning = obs_data["lightning_density"] 
model_lightning = model_data["lightning_density"] 

# Create an instance of the NWP_Stats class
metrics_obj = nw.NWP_Stats(obs_lightning, model_lightning)

# Define the thresholds for metric calculations
thresholds = {
    'SEDS': 0.0002,
    'SEDI': 0.0002,
    'RPSS': 0.0002
}

# Calculate time-averaged metrics
metrics_time_avg = {}
for metric, threshold in thresholds.items():
    metrics_time_avg[metric] = metrics_obj.compute_metrics([metric], thresholds={metric: threshold}, dim="time")[metric]

# Calculate area-average diurnal cycle metrics
metrics_diurnal = {}
for metric, threshold in thresholds.items():
    metrics_diurnal[metric] = metrics_obj.compute_metrics([metric], thresholds={metric: threshold}, dim=["lat", "lon"])[metric].groupby("time.hour").mean()

# Plot time-averaged metrics
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
for i, metric in enumerate(thresholds.keys()):
    metrics_time_avg[metric].plot(ax=axs[i], cmap="coolwarm", vmin=-1, vmax=1)
    axs[i].set_title(f"Time-Averaged {metric}")
    axs[i].set_xlabel("Longitude")
    axs[i].set_ylabel("Latitude")
fig.tight_layout()
plt.savefig("metrics_time_avg.png")

# Plot area-average diurnal cycle metrics
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
for i, metric in enumerate(thresholds.keys()):
    metrics_diurnal[metric].plot(ax=axs[i])
    axs[i].set_title(f"Area-Average Diurnal Cycle {metric}")
    axs[i].set_xlabel("Hour")
    axs[i].set_ylabel(metric)
    axs[i].grid(True)
fig.tight_layout()
plt.savefig("metrics_diurnal.png")

plt.show()




import numpy as np
import xarray as xr
import nwpeval as nw

# Generate random data with longitude and latitude dimensions
lon = np.linspace(0, 360, 10)
lat = np.linspace(-90, 90, 5)
time = np.arange(100)

obs_data = xr.DataArray(np.random.rand(100, 5, 10), dims=('time', 'lat', 'lon'), coords={'time': time, 'lat': lat, 'lon': lon})
model_data = xr.DataArray(np.random.rand(100, 5, 10), dims=('time', 'lat', 'lon'), coords={'time': time, 'lat': lat, 'lon': lon})

# Create an instance of the NWP_Stats class
metrics_obj = nw.NWP_Stats(obs_data, model_data)

# Define the thresholds for the metrics
thresholds = {
    'SEDS': 0.6,
    'SEDI': 0.7,
    'RPSS': 0.8
}

# Compute the metrics
metrics = ['SEDS', 'SEDI', 'RPSS']
metric_values = metrics_obj.compute_metrics(metrics, thresholds=thresholds)

print(metric_values)




import numpy as np
import xarray as xr
import nwpeval as nw

# Generate random data without longitude and latitude dimensions
time = np.arange(100)

obs_data = xr.DataArray(np.random.rand(100), dims=('time'), coords={'time': time})
model_data = xr.DataArray(np.random.rand(100), dims=('time'), coords={'time': time})

# Create an instance of the NWP_Stats class
metrics_obj = nw.NWP_Stats(obs_data, model_data)

# Define the thresholds for the metrics
thresholds = {
    'SEDS': 0.6,
    'SEDI': 0.7,
    'RPSS': 0.8
}

# Compute the metrics
metrics = ['SEDS', 'SEDI', 'RPSS']
metric_values = metrics_obj.compute_metrics(metrics, thresholds=thresholds)

print(metric_values)



import numpy as np
import xarray as xr
import nwpeval as nw

# Generate random data as lists
obs_data = np.random.rand(100).tolist()
model_data = np.random.rand(100).tolist()

# Convert the lists to xarray.DataArray objects
obs_data_array = xr.DataArray(obs_data, dims=['time'])
model_data_array = xr.DataArray(model_data, dims=['time'])

# Create an instance of the NWP_Stats class
metrics_obj = nw.NWP_Stats(obs_data_array, model_data_array)

# Define the thresholds for the metrics
thresholds = {
    'SEDS': 0.6,
    'SEDI': 0.7,
    'RPSS': 0.8
}

# Compute the metrics
metrics = ['SEDS', 'SEDI', 'RPSS']
metric_values = metrics_obj.compute_metrics(metrics, thresholds=thresholds)

print(metric_values)