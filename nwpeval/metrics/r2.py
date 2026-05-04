"""Coefficient of Determination (R^2)."""
import numpy as np
import xarray as xr


def r2(obs_data, model_data, dim=None):
    """
    Compute the Coefficient of Determination (R^2).

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed R^2 values. Returns NaN where the
        observation variance is zero.
    """
    ssr = ((model_data - obs_data) ** 2).sum(dim=dim)
    sst = ((obs_data - obs_data.mean(dim=dim)) ** 2).sum(dim=dim)
    return xr.where(sst == 0, np.nan, 1 - ssr / sst)
