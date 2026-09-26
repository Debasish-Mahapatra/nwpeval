## NWPeval

NWPeval is a Python package designed to facilitate the evaluation and analysis of numerical weather prediction (NWP) models. It provides a comprehensive set of metrics and tools to assess the performance of NWP models by comparing their output with observed weather data.

## Features

- **65 metric functions** (62 different scores: GSS, HKD and Jaccard are other
  names for ETS, PSS and CSI) including:
  - Continuous: MAE, RMSE, R², NRMSE, PCC, and more
  - Categorical: POD, FAR, CSI, ETS, HSS, and more
  - Probabilistic: BSS, SBS, and a simplified yes/no RPSS
  - Distributional: KL Divergence, Hellinger, Wasserstein, and more
  
- Flexible computation along specified dimensions
- Support for threshold-based metrics
- Integration with xarray and NumPy
- Compatible with 2D, 3D, and 4D datasets

## Installation

```shell
pip install nwpeval
```

The examples also need matplotlib and scipy: `pip install "nwpeval[examples]"`.

## Usage

### New API (Recommended)

The new modular API provides standalone functions for each metric:

```python
import xarray as xr
from nwpeval import rmse, mae, acc, pod, fss

# Load observed and modeled data as xarray DataArrays
obs_data = xr.DataArray(...)
model_data = xr.DataArray(...)

# Compute metrics directly
rmse_value = rmse(obs_data, model_data)
mae_value = mae(obs_data, model_data)
acc_value = acc(obs_data, model_data)

print(f"RMSE: {rmse_value}")
print(f"MAE: {mae_value}")
print(f"ACC: {acc_value}")
```

#### Example: Threshold-based metrics
```python
from nwpeval import pod, far, ets, csi

# Compute categorical metrics with threshold
threshold = 0.5
pod_value = pod(obs_data, model_data, threshold=threshold)
far_value = far(obs_data, model_data, threshold=threshold)
ets_value = ets(obs_data, model_data, threshold=threshold)

print(f"POD: {pod_value}")
print(f"FAR: {far_value}")
print(f"ETS: {ets_value}")
```

#### Example: Metrics along specific dimensions
```python
from nwpeval import rmse, mae

# Compute metrics along latitude/longitude (get time series)
rmse_timeseries = rmse(obs_data, model_data, dim=['lat', 'lon'])

# Compute metrics along time (get spatial map)
mae_spatial = mae(obs_data, model_data, dim='time')
```

#### Example: Fractions Skill Score (FSS)
```python
from nwpeval import fss

# Compute FSS with threshold and neighborhood size
fss_value = fss(
    obs_data, 
    model_data, 
    threshold=0.5, 
    neighborhood_size=5,
    spatial_dims=['lat', 'lon']
)
```

If `spatial_dims` is left out, FSS looks for lat/lon, latitude/longitude, y/x,
rlat/rlon or south_north/west_east, and raises an error rather than guess.

A full worked example (threshold x neighbourhood table, useful-skill level,
percentile thresholds, diurnal cycle, heatmap) is in
[examples/example_fss.py](https://github.com/Debasish-Mahapatra/nwpeval/blob/main/examples/example_fss.py).

![FSS by rain threshold and neighbourhood width, from examples/example_fss.py](https://raw.githubusercontent.com/Debasish-Mahapatra/nwpeval/main/examples/fss_heatmap.png)

#### Example: Distribution comparison metrics
```python
from nwpeval import mkldiv, jsdiv, hellinger, wasserstein

kl = mkldiv(obs_data, model_data)
js = jsdiv(obs_data, model_data)
hell = hellinger(obs_data, model_data)
wass = wasserstein(obs_data, model_data)
```

These answer two different questions:

- `wasserstein` compares the spread of values, like two histograms. Where the
  values are does not matter.
- The other ten (`mkldiv`, `jsdiv`, `hellinger`, `tv`, `chisquare`,
  `intersection`, `bhattacharyya`, `chernoff`, `renyi`, `tsallis`) compare
  where the mass is. Each field is scaled to sum to 1 over `dim` and compared
  point by point.

So the same storm moved to another place scores 0 (perfect) on `wasserstein`
but 1 (worst) on `hellinger`. To compare value distributions with the other
ten, pass histograms (counts per bin) instead of the fields.

#### Example: Metrics with parameters
```python
from nwpeval import lehmer_mean, chernoff, renyi, tsallis

# Metrics that require additional parameters
lehmer = lehmer_mean(obs_data, model_data, p=3)
chern = chernoff(obs_data, model_data, alpha=0.7)
ren = renyi(obs_data, model_data, alpha=0.8)
tsal = tsallis(obs_data, model_data, alpha=0.9)
```

---

### Legacy API (Deprecated)

> **Warning**: The `NWP_Stats` class is deprecated and will be removed in a future version. Please migrate to the new standalone functions.

```python
import xarray as xr
from nwpeval import NWP_Stats  # Shows DeprecationWarning

# Load observed and modeled data as xarray DataArrays
obs_data = xr.DataArray(...)
model_data = xr.DataArray(...)

# Create an instance of NWP_Stats
nwp_stats = NWP_Stats(obs_data, model_data)

# Compute basic metrics
metrics = ['MAE', 'RMSE', 'ACC']
results = nwp_stats.compute_metrics(metrics)

# Print the results
for metric, value in results.items():
    print(f"{metric}: {value}")
```

#### Legacy: Computing metrics with thresholds
```python
from nwpeval import NWP_Stats

nwp_stats = NWP_Stats(obs_data, model_data)

thresholds = {
    'FSS': 0.6,
    'FSS_neighborhood': 5,
    'ETS': 0.7,
    'POD': 0.5
}

metrics = ['FSS', 'ETS', 'POD']
results = nwp_stats.compute_metrics(metrics, thresholds=thresholds)
```

#### Legacy: Computing metrics along dimensions
```python
from nwpeval import NWP_Stats

nwp_stats = NWP_Stats(obs_data, model_data)

metrics = ['MAE', 'RMSE', 'ACC']
dimensions = ['lat', 'lon']
results = nwp_stats.compute_metrics(metrics, dim=dimensions)
```

---

## Missing Data and Aggregation

- `obs` and `model` must share identical coordinates; a mismatch raises a
  `ValueError` rather than silently dropping points.
- NaN is missing in every metric: a point missing in either input is dropped
  from both, and never counted as a correct "no event".
- A score that is undefined (e.g. POD with no observed event, or R² when the
  observations never change) is NaN, not 0.
- Aggregate ratio scores (POD, FAR, CSI, ETS, FSS, ...) by pooling counts with
  `dim` (`reduction_dim` for FSS), not by averaging per-time-step scores. See the
  [documentation](https://github.com/Debasish-Mahapatra/nwpeval/blob/main/docs/NWPeval_Documentation.md#missing-data-alignment-and-aggregation).

---

## Available Metrics

| Category | Metrics |
|----------|---------|
| **Continuous** | MAE, RMSE, ACC, R², NRMSE, PCC, MBD, TSE, EVS, NMSE, FV, SDR, VIF, MAD, IQR, NAE, RMB, MAPE, WMAE, ASS, RSS, QSS, LMBE, SMSE, GMB, AEV, Cosine Similarity |
| **Categorical** | ETS, POD, FAR, CSI, HSS, PSS, GSS, FB, HKD, ORSS, SEDS, EDS, SEDI, F1, MCC, BA, NPV, Jaccard, Gain, Lift |
| **Spatial** | FSS |
| **Probabilistic** | BSS and SBS (the model input is a probability from 0 to 1), RPSS (simplified: turns the model into yes/no) |
| **Distributional** | MKLDIV, JSDIV, Hellinger, Wasserstein, TV, Chi-Square, Intersection, Bhattacharyya, Chernoff, Rényi, Tsallis |
| **Mean** | Harmonic Mean, Geometric Mean, Lehmer Mean |

Some names mean something else in other fields. VIF here is var(model)/var(obs) - 1,
not the regression VIF; Gain is plain accuracy; NMSE divides by mean(obs)²; GMB is
model over obs. See the
[documentation](https://github.com/Debasish-Mahapatra/nwpeval/blob/main/docs/NWPeval_Documentation.md#notes-on-names)
for details.

---

## Help

```python
# Get help on a specific function
from nwpeval import rmse
help(rmse)

# List all available metrics
import nwpeval
print(dir(nwpeval))
```

For more detailed usage instructions, see the [Documentation](https://github.com/Debasish-Mahapatra/nwpeval/blob/main/docs/NWPeval_Documentation.md) and [examples](https://github.com/Debasish-Mahapatra/nwpeval/tree/main/examples).


## NEXT UPDATE 

Instability (Thunderstorm) indices:
- Total Totals Index (TT)
- K Index (KI)
- Lifted Index (LI)
- CAPE, SWEAT, and more!


## Contributing

Contributions are welcome! Please open an issue or submit a pull request on the GitHub repository.

## License

NWPeval is licensed under the [MIT License](https://github.com/Debasish-Mahapatra/nwpeval/blob/main/LICENSE).

## Acknowledgments

Thanks to the developers of NumPy, xarray, and SciPy.

## Contact

- **Debasish Mahapatra**
- Email: debasish.atmos@gmail.com | Debasish.mahapatra@ugent.be 

I hope you find NWPeval useful in evaluating your numerical weather prediction models!
