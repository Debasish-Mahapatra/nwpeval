import xarray as xr
import numpy as np
import pandas as pd
import scipy.ndimage

# Define India's geographical boundaries and resolution
lat_start, lat_end = 8.0, 37.0 # Latitude from 8°N to 37°N
lon_start, lon_end = 68.0, 97.0 # Longitude from 68°E to 97°E
lat = np.arange(lat_start, lat_end + 0.05, 0.05)
lon = np.arange(lon_start, lon_end + 0.05, 0.05)

# Time array for one day at hourly intervals
time = pd.date_range("2023-01-01", periods=24, freq='H')

# Initialize empty data arrays
model_data = np.zeros((len(time), len(lat), len(lon)))
obs_data = np.zeros((len(time), len(lat), len(lon)))

# Define initial position (in grid indices) of the storm
storm_center = [len(lat) // 2, len(lon) // 2] # Central India

# Define movement per hour for the storm (in grid indices)
storm_movement = [1, 2] # Move south-east

# Generate lightning density data
for t in range(len(time)):
    # Move storm
    storm_center = [storm_center[0] + storm_movement[0], storm_center[1] + storm_movement[1]]
    
    # Ensure storm center stays within the grid boundaries
    storm_center[0] = min(max(storm_center[0], 0), len(lat) - 1)
    storm_center[1] = min(max(storm_center[1], 0), len(lon) - 1)

    # Simulate storm for model data
    storm_model = np.zeros_like(model_data[0])
    storm_model[storm_center[0], storm_center[1]] = 1 # Peak density at center
    storm_model = scipy.ndimage.gaussian_filter(storm_model, sigma=[10, 20]) # Spread out the storm irregularly
    storm_model = scipy.ndimage.gaussian_filter(storm_model, sigma=[5, 15]) # Make it more irregular
    storm_model *= np.random.rand(*storm_model.shape) * 300 # Randomize density between 0 and 300
    model_data[t] = storm_model

    # Simulate storm for observation data (slightly different from model)
    storm_obs = np.zeros_like(obs_data[0])
    storm_obs[storm_center[0] + 2, storm_center[1] - 2] = 1 # Peak density at a slightly different center
    storm_obs = scipy.ndimage.gaussian_filter(storm_obs, sigma=[12, 18]) # Spread out the storm differently and irregularly
    storm_obs = scipy.ndimage.gaussian_filter(storm_obs, sigma=[7, 13]) # Make it more irregular differently
    storm_obs *= np.random.rand(*storm_obs.shape) * 300 # Randomize density between 0 and 300 differently
    obs_data[t] = storm_obs

# Create xarray Datasets for the model and observation
ds_model = xr.Dataset(
    {
        "lightning_density": (["time", "lat", "lon"], model_data)
    },
    coords={
        "time": time,
        "lat": lat,
        "lon": lon
    }
)

ds_obs = xr.Dataset(
    {
        "lightning_density": (["time", "lat", "lon"], obs_data)
    },
    coords={
        "time": time,
        "lat": lat,
        "lon": lon
    }
)

# Set attributes and encodings
for ds in [ds_model, ds_obs]:
    ds.lightning_density.attrs = {
        'units': 'flashes km-2 hour-1',
        'long_name': 'lightning flash density',
        'standard_name': 'lightning_flash_density'
    }
    ds.lon.attrs = {'standard_name': 'longitude', 'units': 'degrees_east'}
    ds.lat.attrs = {'standard_name': 'latitude', 'units': 'degrees_north'}
    ds['time'].encoding['units'] = 'hours since 2023-01-01 00:00:00'
    ds['time'].encoding['calendar'] = 'gregorian'

# Save the datasets to NetCDF files
ds_model.to_netcdf('india_model_output_005deg_irregular_storm.nc')
ds_obs.to_netcdf('india_obs_output_005deg_irregular_storm.nc')

print("NetCDF files created for India with an irregular storm:")
print("Model: india_model_output_005deg_irregular_storm.nc")
print("Observation: india_obs_output_005deg_irregular_storm.nc")







