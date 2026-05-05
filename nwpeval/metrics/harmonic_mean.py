"""Harmonic Mean (element-wise between obs and model)."""
import numpy as np
import xarray as xr


def harmonic_mean(obs_data, model_data, dim=None):
    """
    Compute the element-wise Harmonic Mean between obs and model data.

    Returns 2 * obs * model / (obs + model) for each corresponding element.
    Elements where obs == 0 or model == 0 yield 0; elements where the sum
    obs + model is exactly 0 yield NaN.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim: Unused, kept for API consistency.

    Returns:
        xarray.DataArray: Element-wise harmonic mean of obs and model.
    """
    total = obs_data + model_data
    return xr.where(total == 0, np.nan, 2 * obs_data * model_data / total)
