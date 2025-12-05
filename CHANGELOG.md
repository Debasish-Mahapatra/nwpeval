# Changelog

## Version 1.6.0 (2024-12-05)

### New Features
- **Modular Metrics API**: All 65 metrics are now available as standalone functions
  - New import style: `from nwpeval import rmse, mae, fss, pod`
  - Each metric in its own file under `nwpeval/metrics/`
- **NWP_Stats class deprecated**: Shows deprecation warning, will be removed in v2.0

### Bug Fixes
- **FSS**: Fixed mse_ref formula (was using mean squared instead of mean of squares)
- **MCC**: Fixed integer overflow and added range clipping to [-1, 1]
- **EDS**: Corrected formula to use proper log ratios
- **BSS**: Fixed climatology Brier Score calculation
- **RPSS**: Fixed dim handling for binary case
- **AEV**: Corrected Adjusted Explained Variance formula

### Documentation
- Updated README with new API examples
- Rewrote documentation to prioritize new standalone functions
- Added migration guide from NWP_Stats to standalone functions

### Tests
- Added `tests/test_all_metrics.py` - validates all 65 metrics
- Added `tests/test_all_metrics_with_plots.py` - comprehensive test with plots

---

## Version 1.5.1beta5 

### MAJOR REVISION OF CODE 

Did a major fix to ```comute_rpss``` to work for both scalar and non-scalar values.

``` python

    def compute_rpss(self, threshold, dim=None):
        """
        Compute the Ranked Probability Skill Score (RPSS) for a given threshold.
    
        Args:
            threshold (float): The threshold value for binary classification.
            dim (str, list, or None): The dimension(s) along which to compute the RPSS.
                                  If None, compute the RPSS over the entire data.
    
        Returns:
            xarray.DataArray: The computed RPSS values.
        """
        # Convert data to binary based on the threshold
        obs_binary = (self.obs_data >= threshold).astype(int)
        model_binary = (self.model_data >= threshold).astype(int)
    
        # Calculate the RPS for the model data
        rps_model = ((model_binary.cumsum(dim) - obs_binary.cumsum(dim)) ** 2).mean(dim=dim)
    
        # Calculate the RPS for the climatology (base rate)
        base_rate = obs_binary.mean(dim=dim)
        rps_climo = ((xr.full_like(model_binary, 0).cumsum(dim) - obs_binary.cumsum(dim)) ** 2).mean(dim=dim)
        rps_climo = rps_climo + base_rate * (1 - base_rate)
    
        # Calculate the RPSS
        rpss = 1 - rps_model / rps_climo
    
        return rpss

```

The updated `compute_rpss` method will work correctly for both scalar and non-scalar `base_rate` values.

In the context of xarray and dimensions/coordinates in a dataset, a scalar value refers to a single value that does not depend on any dimensions. It is a 0-dimensional value. On the other hand, a non-scalar value is an array or a DataArray that depends on one or more dimensions and has corresponding coordinates.

Let's consider an example to illustrate the difference:

Suppose we have a dataset with dimensions "time", "lat", and "lon". The dataset contains a variable "temperature" with corresponding coordinates for each dimension.

- Scalar value: If we calculate the mean temperature over all dimensions using `temperature.mean()`, the resulting value will be a scalar. It will be a single value that does not depend on any dimensions.

- Non-scalar value: If we calculate the mean temperature over a specific dimension, such as `temperature.mean(dim="time")`, the resulting value will be a non-scalar DataArray. It will have dimensions "lat" and "lon" and corresponding coordinates, but it will not depend on the "time" dimension anymore.

In the updated `compute_rpss` method, the line `base_rate = obs_binary.mean(dim=dim)` calculates the mean of `obs_binary` over the specified dimensions `dim`. If `dim` is None, it will calculate the mean over all dimensions, resulting in a scalar value. If `dim` is a specific dimension or a list of dimensions, it will calculate the mean over those dimensions, resulting in a non-scalar DataArray.

The subsequent lines of code in the `compute_rpss` method handle both cases correctly:

```python
rps_climo = ((xr.full_like(model_binary, 0).cumsum(dim) - obs_binary.cumsum(dim)) ** 2).mean(dim=dim)
rps_climo = rps_climo + base_rate * (1 - base_rate)
```

If `base_rate` is a scalar value, it will be broadcasted to match the shape of `rps_climo`, and the calculation will be performed element-wise. If `base_rate` is a non-scalar DataArray, it will be aligned with `rps_climo` based on the common dimensions, and the calculation will be performed element-wise.

Now, whether this will work with data of different coordinates??? The updated `compute_rpss` method should work correctly as long as the dimensions and coordinates of `obs_binary` and `model_binary` are compatible. The method relies on xarray's broadcasting and alignment rules to handle data with different coordinates.

However, it's important to note that if the coordinates of `obs_binary` and `model_binary` are completely different or incompatible, you may encounter issues with dimension alignment or broadcasting. In such cases, you would need to ensure that the coordinates are properly aligned or resampled before applying the `compute_rpss` method.

In summary, the updated `compute_rpss` method should work correctly for both scalar and non-scalar `base_rate` values, and it should handle data with different coordinates as long as the dimensions and coordinates are compatible between `obs_binary` and `model_binary`.

### Bug Fixes

- Fixed minor bugs and improved code stability.

### Other Changes

- The package has been moved from the 3-Alpha stage to the 4-Beta stage in development, indicating that it has undergone further testing and refinement.

Please note that this is a beta release (version 1.5.1beta5), and while it includes significant enhancements and bug fixes, it may still have some known limitations or issues. We encourage users to provide feedback and report any bugs they encounter.

We appreciate your interest in the NWPeval package and thank you for your support!