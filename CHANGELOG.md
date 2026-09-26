# Changelog

## Version 1.6.4 (2026-09-26)

### Bug fixes
- Renyi and Tsallis replaced zeros in the model by the smallest float. For
  alpha > 1, where the model had no mass at a point with observed mass, they
  gave a large finite number (Renyi 706, Tsallis 4.7e153 at alpha = 1.5)
  instead of +inf. They now use the exact limits. Renyi, Chernoff and
  Bhattacharyya are +inf (was NaN) when the two fields share no mass. Renyi
  and Tsallis now reject alpha < 0.
- R2, EVS, FV, SDR, VIF, SMSE and AEV returned huge numbers (R2 = -3.5e31)
  instead of NaN when the observations were constant at a value such as 0.1:
  rounding makes the computed variance tiny but not zero. PCC and ACC
  (without a climatology) returned about 1e-16 instead of NaN. Constant
  observations are now found from the values themselves.
- FSS without `spatial_dims` used the last two dimensions when it found
  neither lat/lon nor x/y, so data ordered (latitude, longitude, time) was
  smoothed over time with no warning. It now also knows latitude/longitude,
  rlat/rlon and south_north/west_east, and raises a `ValueError` instead of
  guessing.
- Legacy `NWP_Stats.compute_metrics` crashed without a `thresholds` dict and
  silently skipped unknown metric names. Missing thresholds now use the
  defaults, unknown names raise a `ValueError`, and `'HKD'` works as well as
  `'H-KD'`.

### Packaging
- scipy and matplotlib are no longer installed with nwpeval; the package
  never imported them. The examples need them: `pip install "nwpeval[examples]"`.

### Documentation and examples
- README: fixed the broken link to the documentation. Links and the FSS
  heatmap now use full GitHub addresses, so they also work on PyPI.
- New notes on the two kinds of distributional metrics (Wasserstein compares
  the spread of values, the other ten compare where the mass is) and on names
  used differently elsewhere (VIF, Gain, NMSE, GMB). BSS and SBS are listed as
  probabilistic scores that need a forecast probability; RPSS is marked as a
  yes/no version. "65 metrics" now says that 3 are aliases.
- The diurnal-cycle recipe now aligns obs and model with `join='exact'`
  first: building `xr.Dataset({'obs': obs, 'model': model})` directly padded
  mismatched grids with NaN instead of raising.
- Corrected the 1.6.2 notes on the MCC overflow and on which packages the
  code imports, and the AEV and EDS docstrings of the legacy class.
- `lightning_data_nc_gen.py` runs on pandas 3, and the lightning examples
  read the files it writes.
- The two local data scripts under `tests/` now give BSS and SBS a forecast
  probability, and pool counts for their diurnal cycle and smoothed time
  series instead of averaging per-time-step scores.

### Tests
- Exact limits of the alpha-divergences, constant observations, FSS
  dimension detection and the legacy dispatcher.

## Version 1.6.3 (2026-09-22)

### Bug fixes
- MKLDIV returned 0 (perfect agreement) when the model field had no mass,
  e.g. no rain anywhere in a time step. It now returns NaN, like the other
  distribution metrics. Where the model misses observed mass it is still +inf.

### Tests
- `tests/test_metric_coverage.py`: every per-dimension result (`dim='time'`,
  spatial, all) of every metric against numpy references, with gaps in obs,
  model or both; parameter sweeps; the legacy `NWP_Stats.compute_metrics` for
  all 65 names; an extra ensemble dimension; dask-backed inputs.

## Version 1.6.2 (2026-05-05, updated 2026-09-22)

### Update 2026-09-22

All formulas were already correct on complete data. This update fixes how
the metrics handle missing data, undefined cases, large samples and grids.

#### Bug fixes
- Missing data (NaN) is now dropped from both inputs in every metric.
  - Categorical scores (POD, FAR, CSI, FB, ETS/GSS, HSS, PSS/HKD, ORSS, SEDS,
    EDS, SEDI, F1, MCC, BA, NPV, Jaccard, Gain, Lift) turned NaN into "no
    event" before building the contingency table, so the confusion-matrix NaN
    guard never ran. On the Finley table plus 1,050 missing observations,
    FAR went 0.72 -> 0.81 and CSI 0.23 -> 0.16.
  - BSS and RPSS counted missing observations as non-events.
  - MBD, FV, SDR, VIF, GMB, ACC, AEV, cosine similarity, WMAE and all
    distributional metrics reduced obs and model over different samples when
    their NaN patterns differed (NRMSE, R2, EVS, NMSE, NAE, RMB, SMSE and QSS
    when the model had gaps).
- FSS treated missing points as dry and scored only points whose whole
  window fitted inside the grid (none at all once the window was wider than
  the domain, which returned NaN). Fractions are now taken over valid
  neighbours only and every valid point is scored.
- Undefined scores return NaN instead of 0: POD, FAR, CSI, FB, ETS, HSS,
  MCC, BSS and RPSS. PSS returned -POFD when no event was observed; SEDS
  returned 1 when there were no events at all. F1 is now 2TP/(2TP+FP+FN), which is 0 (not NaN) when
  there are no hits but some false alarms or misses.
- MCC overflowed int64 from about 120,000 points (more when events are
  rare) and then returned NaN or a false 1.0 (a perfect score). The bug was
  in 1.6.1; MCC values from that version on large grids should be redone.
  Contingency counts are now floats.
- SEDI, EDS and SEDS clipped probabilities to 1e-10, giving about -0.82
  instead of the limit -1 when there are no hits. They now return the exact
  limits, and NaN where the limit is path-dependent.
- Wasserstein compared the raw arrays element-wise, so a model array with a
  different dimension order gave a wrong distance.
- ASS with a per-element reference error and `dim=None` returned an array
  instead of a scalar.

#### Behaviour changes
- Inputs whose coordinates differ (even by floating-point noise) now raise a
  `ValueError`. Previously xarray's inner join silently dropped the unmatched
  points, or returned NaN when nothing matched.
- TSE and the distributional metrics return NaN instead of 0 when there is
  no valid point.

#### Documentation and examples
- New section on missing data, alignment and aggregation: pool
  contingency counts with `dim` rather than averaging per-time scores.
- The POD/FAR and diurnal-cycle examples now pool counts.
- ACC docstring: without a climatology, the obs mean is used for both
  anomalies, so a mean model bias lowers the score.

#### Tests
- `tests/test_metric_correctness.py`: every metric against an independent
  numpy reference on clean data, obs gaps and model gaps; Finley published
  values; undefined cases; MCC at 3 million points; misaligned coordinates for
  every public metric; FSS against a loop-based brute force.
- `tests/conftest.py` skips the two data-dependent scripts under `tests/`.

### Release 2026-05-05

#### Bug fixes
- Mathematical / formula corrections
  - EDS: corrected sign error in numerator
  - SEDS: replaced non-canonical formula with `[log(p) + log(p_F)] / log(s) - 1`
  - Lift: now `precision / base_rate` (was its reciprocal)
  - MAD and IQR: now operate on residuals (model - obs); previously ignored obs
  - AEV: implemented degrees-of-freedom adjustment (was a copy of EVS)
  - Wasserstein: now W1 over sorted samples (was treating array index as bins)
  - Symmetric Brier Score: now operates on probabilistic forecast vs binary obs
- Zero-division and zero-variance guards added to: F1, BA, NPV, Jaccard,
  Gain, QSS, WMAE, R2, NRMSE, NMSE, RMB, NAE, MAPE, EVS, FV, SDR, VIF,
  SMSE, ORSS, cosine similarity, ACC.
- Distributional metric input validation (mkldiv, jsdiv, hellinger, tv,
  chisquare, intersection, bhattacharyya, chernoff, renyi, tsallis, gmb)
  now reject negative inputs and guard zero totals; renyi and tsallis raise
  on alpha == 1.
- ASS and RSS now respect the `dim` argument.
- ACC implementation now matches its docstring (uncentred anomaly form).
- Confusion matrix now masks NaN cells via `notnull()`.
- Harmonic mean and geometric mean now handle zeros and negatives.
- Fix dispatcher passing `dim` as `climatology` to `compute_acc`.

#### Refactor
- Three pairs of duplicate metrics consolidated into aliases:
  GSS -> ETS, HKD -> PSS, Jaccard -> CSI.
- All `NWP_Stats.compute_*` methods now delegate to the standalone metric
  functions, eliminating duplicated bugs in the legacy class.
- Legacy `NWP_Stats.confusion_matrix` delegates to the canonical helper.

#### Data loading
- `nwpeval.utils.load_data` now supports HDF5 files (`.h5`, `.hdf5`,
  `.hdf`) via the `h5netcdf` engine.
- GRIB import check now correctly verifies `cfgrib` (the runtime engine)
  instead of the unused `pygrib`.
- Both optional engines are imported lazily.

#### Packaging
- `pygrib` removed from hard dependencies; `cfgrib` and `h5netcdf` moved
  to `extras_require` (`pip install nwpeval[grib]`, `[hdf5]`, `[all]`).
- `requirements.txt` cleaned up: removed unused `scikit-learn` and
  `pygrib`; added `pandas` (used by `nwpeval.utils`), and `scipy` and
  `matplotlib`, which only the examples use.
- Python 3.10-3.12 added to classifiers; 3.6-3.7 removed.

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