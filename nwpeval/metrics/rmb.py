"""Relative Mean Bias (RMB)."""
import numpy as np
import xarray as xr


def rmb(obs_data, model_data, dim=None):
    """
    Compute the Relative Mean Bias (RMB).

    RMB = sum(model - obs) / sum(obs).

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed RMB values. Returns NaN where the
        observation sum is zero.
    """
    bias = (model_data - obs_data).sum(dim=dim)
    obs_sum = obs_data.sum(dim=dim)
    return xr.where(obs_sum == 0, np.nan, bias / obs_sum)
