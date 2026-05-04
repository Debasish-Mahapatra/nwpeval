"""Fractional Variance (FV)."""
import numpy as np
import xarray as xr


def fv(obs_data, model_data, dim=None):
    """
    Compute the Fractional Variance (FV).

    FV = var(model) / var(obs).

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed FV values. Returns NaN where the
        observation variance is zero.
    """
    obs_var = obs_data.var(dim=dim)
    model_var = model_data.var(dim=dim)
    return xr.where(obs_var == 0, np.nan, model_var / obs_var)
