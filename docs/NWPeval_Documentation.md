
## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [New API (Recommended)](#new-api-recommended)
   - [Standalone Metric Functions](#standalone-metric-functions)
   - [Available Metrics](#available-metrics)
5. [Legacy API (Deprecated)](#legacy-api-deprecated)
   - [NWP_Stats Class](#nwp_stats-class)
6. [Examples](#examples)
7. [API Reference](#api-reference)
8. [Contributing](#contributing)
9. [License](#license)
10. [Contact](#contact)

---

## Introduction

NWPeval is a Python package designed for evaluating Numerical Weather Prediction (NWP) models. It provides **65 evaluation metrics** to assess model performance against observed data.

The package integrates seamlessly with `xarray` for efficient multi-dimensional data handling.

---

## Installation

```shell
pip install nwpeval
```

---

## Quick Start

```python
import xarray as xr
from nwpeval import rmse, mae, pod

# Load your data
obs = xr.DataArray(...)
model = xr.DataArray(...)

# Compute metrics
print(f"RMSE: {rmse(obs, model)}")
print(f"MAE: {mae(obs, model)}")
print(f"POD: {pod(obs, model, threshold=0.5)}")
```

---

## New API (Recommended)

### Standalone Metric Functions

Each metric is available as a standalone function:

```python
from nwpeval import rmse, mae, acc, pod, fss, bss
```

#### Basic Usage

```python
from nwpeval import rmse, mae, acc

result = rmse(obs_data, model_data)
result = mae(obs_data, model_data, dim='time')  # Along specific dimension
result = acc(obs_data, model_data, dim=['lat', 'lon'])  # Multiple dimensions
```

#### Threshold-based Metrics

```python
from nwpeval import pod, far, ets, csi, hss

pod_value = pod(obs_data, model_data, threshold=0.5)
far_value = far(obs_data, model_data, threshold=0.5)
ets_value = ets(obs_data, model_data, threshold=0.5)
```

#### Fractions Skill Score (FSS)

```python
from nwpeval import fss

result = fss(
    obs_data, 
    model_data, 
    threshold=0.5, 
    neighborhood_size=5,
    spatial_dims=['lat', 'lon'],
    reduction_dim='time'
)
```

#### Metrics with Additional Parameters

```python
from nwpeval import wmae, lehmer_mean, chernoff, renyi, tsallis

# Weighted MAE
result = wmae(obs_data, model_data, weights=weight_array)

# Parameterized metrics
result = lehmer_mean(obs_data, model_data, p=3)
result = chernoff(obs_data, model_data, alpha=0.5)
result = renyi(obs_data, model_data, alpha=0.8)
result = tsallis(obs_data, model_data, alpha=0.9)
```

#### Reference-based Metrics

```python
from nwpeval import ass, rss, qss

result = ass(obs_data, model_data, reference_error=ref_error)
result = qss(obs_data, model_data, reference_forecast=climatology)
```

---

### Available Metrics

| Category | Function | Description |
|----------|----------|-------------|
| **Continuous** | `mae` | Mean Absolute Error |
| | `rmse` | Root Mean Square Error |
| | `acc` | Anomaly Correlation Coefficient |
| | `r2` | Coefficient of Determination |
| | `nrmse` | Normalized RMSE |
| | `pcc` | Pearson Correlation Coefficient |
| | `mbd` | Mean Bias Deviation |
| | `tse` | Total Squared Error |
| | `evs` | Explained Variance Score |
| | `nmse` | Normalized Mean Squared Error |
| | `fv` | Fractional Variance |
| | `sdr` | Standard Deviation Ratio |
| | `vif` | Variance Inflation Factor |
| | `mad` | Median Absolute Deviation |
| | `iqr` | Interquartile Range |
| | `nae` | Normalized Absolute Error |
| | `rmb` | Relative Mean Bias |
| | `mape` | Mean Absolute Percentage Error |
| | `wmae` | Weighted Mean Absolute Error |
| | `ass` | Absolute Skill Score |
| | `rss` | Relative Skill Score |
| | `qss` | Quadratic Skill Score |
| | `lmbe` | Logarithmic Mean Bias Error |
| | `smse` | Scaled Mean Squared Error |
| | `gmb` | Geometric Mean Bias |
| | `sbs` | Symmetric Brier Score |
| | `aev` | Adjusted Explained Variance |
| | `cosine_similarity` | Cosine Similarity |
| **Spatial** | `fss` | Fractions Skill Score |
| **Categorical** | `ets` | Equitable Threat Score |
| | `pod` | Probability of Detection |
| | `far` | False Alarm Ratio |
| | `csi` | Critical Success Index |
| | `hss` | Heidke Skill Score |
| | `pss` | Peirce Skill Score |
| | `gss` | Gilbert Skill Score |
| | `fb` | Frequency Bias |
| | `hkd` | Hanssen-Kuipers Discriminant |
| | `orss` | Odds Ratio Skill Score |
| | `seds` | Symmetric Extreme Dependency Score |
| | `eds` | Extreme Dependency Score |
| | `sedi` | Symmetric Extremal Dependence Index |
| | `f1` | F1 Score |
| | `mcc` | Matthews Correlation Coefficient |
| | `ba` | Balanced Accuracy |
| | `npv` | Negative Predictive Value |
| | `jaccard` | Jaccard Similarity Coefficient |
| | `gain` | Gain |
| | `lift` | Lift |
| **Probabilistic** | `bss` | Brier Skill Score |
| | `rpss` | Ranked Probability Skill Score |
| **Distributional** | `mkldiv` | Mean Kullback-Leibler Divergence |
| | `jsdiv` | Jensen-Shannon Divergence |
| | `hellinger` | Hellinger Distance |
| | `wasserstein` | Wasserstein Distance |
| | `tv` | Total Variation Distance |
| | `chisquare` | Chi-Square Distance |
| | `intersection` | Intersection |
| | `bhattacharyya` | Bhattacharyya Distance |
| | `chernoff` | Chernoff Distance |
| | `renyi` | Rényi Divergence |
| | `tsallis` | Tsallis Divergence |
| **Mean** | `harmonic_mean` | Element-wise Harmonic Mean |
| | `geometric_mean` | Element-wise Geometric Mean |
| | `lehmer_mean` | Lehmer Mean |

---

## Legacy API (Deprecated)

> ⚠️ **Warning**: The `NWP_Stats` class is deprecated and will be removed in a future version. Please migrate to standalone functions.

### NWP_Stats Class

```python
from nwpeval import NWP_Stats  # Shows DeprecationWarning

nwp_stats = NWP_Stats(obs_data, model_data)

# Compute single metric
mae_value = nwp_stats.compute_mae()
rmse_value = nwp_stats.compute_rmse()

# Compute multiple metrics
metrics = ['MAE', 'RMSE', 'ACC']
results = nwp_stats.compute_metrics(metrics)

# With thresholds
thresholds = {'FSS': 0.5, 'FSS_neighborhood': 5, 'POD': 0.5}
results = nwp_stats.compute_metrics(['FSS', 'POD'], thresholds=thresholds)

# Along dimensions
results = nwp_stats.compute_metrics(['MAE', 'RMSE'], dim=['lat', 'lon'])
```

### Migration Guide

| Legacy (Deprecated) | New (Recommended) |
|---------------------|-------------------|
| `NWP_Stats(obs, model).compute_mae()` | `mae(obs, model)` |
| `NWP_Stats(obs, model).compute_rmse(dim='time')` | `rmse(obs, model, dim='time')` |
| `NWP_Stats(obs, model).compute_pod(threshold=0.5)` | `pod(obs, model, threshold=0.5)` |

---

## Examples

### Example 1: Basic Continuous Metrics

```python
import xarray as xr
from nwpeval import rmse, mae, r2, pcc

obs = xr.DataArray(...)
model = xr.DataArray(...)

print(f"RMSE: {rmse(obs, model):.4f}")
print(f"MAE: {mae(obs, model):.4f}")
print(f"R²: {r2(obs, model):.4f}")
print(f"PCC: {pcc(obs, model):.4f}")
```

### Example 2: Categorical Metrics with Threshold

```python
from nwpeval import pod, far, csi, ets, hss

threshold = 1.0  # e.g., 1mm precipitation

metrics = {
    'POD': pod(obs, model, threshold=threshold),
    'FAR': far(obs, model, threshold=threshold),
    'CSI': csi(obs, model, threshold=threshold),
    'ETS': ets(obs, model, threshold=threshold),
    'HSS': hss(obs, model, threshold=threshold),
}

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")
```

### Example 3: Spatial Analysis (Time-averaged Maps)

```python
from nwpeval import rmse, mae

# Get spatial distribution of errors (average over time)
rmse_map = rmse(obs, model, dim='time')
mae_map = mae(obs, model, dim='time')

# Plot the maps
rmse_map.plot()
```

### Example 4: Temporal Analysis (Area-averaged Time Series)

```python
from nwpeval import rmse, pcc

# Get time series of metrics (average over space)
rmse_ts = rmse(obs, model, dim=['lat', 'lon'])
pcc_ts = pcc(obs, model, dim=['lat', 'lon'])
```

### Example 5: Distribution Comparison

```python
from nwpeval import jsdiv, hellinger, wasserstein

js = jsdiv(obs, model)
hell = hellinger(obs, model)
wass = wasserstein(obs, model)

print(f"Jensen-Shannon: {js:.4f}")
print(f"Hellinger: {hell:.4f}")
print(f"Wasserstein: {wass:.4f}")
```

---

## API Reference

### Function Signature Patterns

**Continuous metrics** (no threshold):
```python
metric(obs_data, model_data, dim=None) -> xarray.DataArray
```

**Threshold-based metrics**:
```python
metric(obs_data, model_data, threshold, dim=None) -> xarray.DataArray
```

**FSS** (spatial metric):
```python
fss(obs_data, model_data, threshold, neighborhood_size, 
    spatial_dims=None, reduction_dim=None) -> xarray.DataArray
```

**Parameterized metrics**:
```python
lehmer_mean(obs_data, model_data, p, dim=None) -> xarray.DataArray
chernoff(obs_data, model_data, alpha, dim=None) -> xarray.DataArray
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `obs_data` | `xarray.DataArray` | Observed data |
| `model_data` | `xarray.DataArray` | Model/forecast data |
| `dim` | `str`, `list`, or `None` | Dimension(s) to compute over |
| `threshold` | `float` | Threshold for binary classification |
| `neighborhood_size` | `int` | Window size for FSS |

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT License

## Contact

- **Debasish Mahapatra**
- Email: debasish.atmos@gmail.com | Debasish.mahapatra@ugent.be