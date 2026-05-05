"""Geometric Mean (element-wise between obs and model)."""
import numpy as np
import xarray as xr


def geometric_mean(obs_data, model_data, dim=None):
    """
    Compute the element-wise Geometric Mean between obs and model data.

    Returns sqrt(obs * model) for each corresponding element. Elements
    where either obs or model is negative yield NaN, since the geometric
    mean is only defined for non-negative inputs.

    Args:
        obs_data (xarray.DataArray): The observed data (must be >= 0).
        model_data (xarray.DataArray): The modeled data (must be >= 0).
        dim: Unused, kept for API consistency.

    Returns:
        xarray.DataArray: Element-wise geometric mean of obs and model.
    """
    valid = (obs_data >= 0) & (model_data >= 0)
    return xr.where(valid, np.sqrt(obs_data * model_data), np.nan)
