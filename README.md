## NWPeval

NWPeval is a Python package designed to facilitate the evaluation and analysis of numerical weather prediction (NWP) models. It provides a comprehensive set of metrics and tools to assess the performance of NWP models by comparing their output with observed weather data.

## Features

- Supports a wide range of evaluation metrics, including:
  - Mean Absolute Error (MAE)
  - Root Mean Square Error (RMSE)
  - Anomaly Correlation Coefficient (ACC)
  - Fractions Skill Score (FSS)
  - Equitable Threat Score (ETS)
  - Probability of Detection (POD)
  - False Alarm Ratio (FAR)
  - Critical Success Index (CSI)
  - Brier Skill Score (BSS)
  - Heidke Skill Score (HSS)
  - Peirce Skill Score (PSS)
  - Gilbert Skill Score (GS)
  - Symmetric Extreme Dependency Score (SEDS)
  - Frequency Bias (FB)
  - Gilbert Skill Score (GSS)
  - Hanssen-Kuipers Discriminant (H-KD)
  - Odds Ratio Skill Score (ORSS)
  - Extreme Dependency Score (EDS)
  - Symmetric Extremal Dependence Index (SEDI)
  - Ranked Probability Skill Score (RPSS)
  - Total Squared Error (TSE)
  - Explained Variance Score (EVS)
  - Normalized Mean Squared Error (NMSE)
  - Fractional Variance (FV)
  - Pearson Correlation Coefficient (PCC)
  - Standard Deviation Ratio (SDR)
  - Variance Inflation Factor (VIF)
  - Median Absolute Deviation (MAD)
  - Interquartile Range (IQR)
  - Coefficient of Determination (R^2)
  - Normalized Absolute Error (NAE)
  - Relative Mean Bias (RMB)
  - Mean Absolute Percentage Error (MAPE)
  - Weighted Mean Absolute Error (WMAE)
  - Absolute Skill Score (ASS)
  - Relative Skill Score (RSS)
  - Quadratic Skill Score (QSS)
  - Normalized Root Mean Squared Error (NRMSE)
  - Logarithmic Mean Bias Error (LMBE)
  - Scaled Mean Squared Error (SMSE)
  - Mean Bias Deviation (MBD)
  - Geometric Mean Bias (GMB)
  - Symmetric Brier Score (SBS)
  - Adjusted Explained Variance (AEV)
  - Cosine Similarity
  - F1 Score
  - Matthews Correlation Coefficient (MCC)
  - Balanced Accuracy (BA)
  - Negative Predictive Value (NPV)
  - Jaccard Similarity Coefficient
  - Gain
  - Lift
  - Mean Kullback-Leibler Divergence (MKLDIV)
  - Jensen-Shannon Divergence (JSDIV)
  - Hellinger Distance
  - Wasserstein Distance
  - Total Variation Distance
  - Chi-Square Distance
  - Intersection
  - Bhattacharyya Distance
  - Harmonic Mean
  - Geometric Mean
  - Lehmer Mean
  - Chernoff Distance
  - Rényi Divergence
  - Tsallis Divergence

- Flexible computation of metrics along specified dimensions or over the entire dataset.
- Support for threshold-based metrics with customizable threshold values.
- Integration with xarray and NumPy for efficient computation and data handling.
- Compatibility with both time series and spatial data, supporting 2D, 3D, and 4D datasets.
- Detailed Examples to guide users in utilizing the package effectively.

## Installation

You can install NWPeval using pip:

```shell
pip install nwpeval
```

## Usage

Here are a few examples of how to use the package 

Example 1: Computing basic metrics
```python
import xarray as xr
from nwpeval import NWP_Stats

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

Example 2: Computing metrics with thresholds
```python
import xarray as xr
from nwpeval import NWP_Stats

# Load observed and modeled data as xarray DataArrays
obs_data = xr.DataArray(...)
model_data = xr.DataArray(...)

# Create an instance of NWP_Stats
nwp_stats = NWP_Stats(obs_data, model_data)

# Define thresholds for specific metrics
thresholds = {
    'FSS': 0.6,
    'FSS_neighborhood': 5,
    'ETS': 0.7,
    'POD': 0.5
}

# Compute metrics with thresholds
metrics = ['FSS', 'ETS', 'POD']
results = nwp_stats.compute_metrics(metrics, thresholds=thresholds)

# Print the results
for metric, value in results.items():
    print(f"{metric}: {value}")
```

Example 3: Computing metrics along specific dimensions
```python
import xarray as xr
from nwpeval import NWP_Stats

# Load observed and modeled data as xarray DataArrays
obs_data = xr.DataArray(...)
model_data = xr.DataArray(...)

# Create an instance of NWP_Stats
nwp_stats = NWP_Stats(obs_data, model_data)

# Compute metrics along specific dimensions
metrics = ['MAE', 'RMSE', 'ACC']
dimensions = ['lat', 'lon']
results = nwp_stats.compute_metrics(metrics, dim=dimensions)

# Print the results
for metric, value in results.items():
    print(f"{metric}:")
    print(value)
```

Example 4: Computing probabilistic metrics with custom thresholds
```python
import xarray as xr
from nwpeval import NWP_Stats

# Load observed and modeled data as xarray DataArrays
obs_data = xr.DataArray(...)
model_data = xr.DataArray(...)

# Create an instance of NWP_Stats
nwp_stats = NWP_Stats(obs_data, model_data)

# Define custom thresholds for probabilistic metrics
thresholds = {
    'SEDS': 0.6,
    'SEDI': 0.7,
    'RPSS': 0.8
}

# Compute probabilistic metrics with custom thresholds
metrics = ['SEDS', 'SEDI', 'RPSS']
results = nwp_stats.compute_metrics(metrics, thresholds=thresholds)

# Print the results
for metric, value in results.items():
    print(f"{metric}: {value}")
```

Example 5: Computing weighted metrics
```python
import xarray as xr
from nwpeval import NWP_Stats

# Load observed and modeled data as xarray DataArrays
obs_data = xr.DataArray(...)
model_data = xr.DataArray(...)

# Create an instance of NWP_Stats
nwp_stats = NWP_Stats(obs_data, model_data)

# Define weights for weighted metrics
weights = xr.DataArray(...)

# Compute weighted metrics
metrics = ['WMAE']
thresholds = {'WMAE_weights': weights}
results = nwp_stats.compute_metrics(metrics, thresholds=thresholds)

# Print the results
for metric, value in results.items():
    print(f"{metric}: {value}")
```

Example 6: Computing distribution comparison metrics
```python
import xarray as xr
from nwpeval import NWP_Stats

# Load observed and modeled data as xarray DataArrays
obs_data = xr.DataArray(...)
model_data = xr.DataArray(...)

# Create an instance of NWP_Stats
nwp_stats = NWP_Stats(obs_data, model_data)

# Compute distribution comparison metrics
metrics = ['MKLDIV', 'JSDIV', 'Hellinger', 'Wasserstein']
results = nwp_stats.compute_metrics(metrics)

# Print the results
for metric, value in results.items():
    print(f"{metric}: {value}")
```

Example 7: Computing reference-based metrics
```python
import xarray as xr
from nwpeval import NWP_Stats

# Load observed, modeled, and reference data as xarray DataArrays
obs_data = xr.DataArray(...)
model_data = xr.DataArray(...)
reference_data = xr.DataArray(...)

# Create an instance of NWP_Stats
nwp_stats = NWP_Stats(obs_data, model_data)

# Define thresholds for reference-based metrics
thresholds = {
    'ASS_reference_error': reference_data,
    'RSS_reference_skill': 0.6,
    'QSS_reference_forecast': reference_data
}

# Compute reference-based metrics
metrics = ['ASS', 'RSS', 'QSS']
results = nwp_stats.compute_metrics(metrics, thresholds=thresholds)

# Print the results
for metric, value in results.items():
    print(f"{metric}: {value}")
```

Example 8: Computing metrics with custom parameters
```python
import xarray as xr
from nwpeval import NWP_Stats

# Load observed and modeled data as xarray DataArrays
obs_data = xr.DataArray(...)
model_data = xr.DataArray(...)

# Create an instance of NWP_Stats
nwp_stats = NWP_Stats(obs_data, model_data)

# Define custom parameters for metrics
thresholds = {
    'LehmerMean_p': 3,
    'Chernoff_alpha': 0.7,
    'Renyi_alpha': 0.8,
    'Tsallis_alpha': 0.9
}

# Compute metrics with custom parameters
metrics = ['LehmerMean', 'Chernoff', 'Renyi', 'Tsallis']
results = nwp_stats.compute_metrics(metrics, thresholds=thresholds)

# Print the results
for metric, value in results.items():
    print(f"{metric}: {value}")
```

These examples demonstrate the versatility of the package in handling different requirements and scenarios. They showcase the computation of probabilistic metrics with custom thresholds, weighted metrics, distribution comparison metrics, reference-based metrics, and metrics with custom parameters.

By providing a wide range of metrics and the flexibility to customize their computation, the package enables users to perform comprehensive evaluations of NWP models based on their specific needs and requirements.



For more detailed usage instructions and [Documentation](docs/NWPeval%20Documentation.md) and [examples](examples), please refer to the Examples directory.


## NEXT UPDATE 

Instability (Thunderstorm) indices, 

  - Total Totals Index (TT)
  - Delta-T Index (DTI)
  - K Index (KI)
  - Lifted Index (LI)
  - Showalter Index (SI)
  - Deep Convective Index (DCI)
  - Severe Weather Threat Index (SWEAT)
  - Convective Available Potential Energy (CAPE)
    and More !!!



## Contributing

Contributions to NWPeval are welcome! If you encounter any issues, have suggestions for improvements, or would like to contribute new features, please open an issue or submit a pull request on the GitHub repository.

## License

NWPeval is licensed under the [MIT License](LICENSE).

## Acknowledgments

I would like to express my gratitude to the developers and contributors of the libraries and tools used in building NWPeval, including NumPy, xarray, and SciPy.

## Contact

For any questions, feedback, or inquiries, please contact the maintainer:

- Name: Debasish Mahapatra
- Email: debasish.atmos@gmail.com | Debasish.mahapatra@ugent.be 

I hope you find NWPeval useful in evaluating and analyzing your numerical weather prediction models!
