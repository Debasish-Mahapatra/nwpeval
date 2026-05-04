"""Explained Variance Score (EVS)."""
import numpy as np
import xarray as xr


def evs(obs_data, model_data, dim=None):
    """
    Compute the Explained Variance Score (EVS).

    EVS = 1 - var(obs - model) / var(obs).

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed EVS values. Returns NaN where the
        observation variance is zero.
    """
    obs_var = obs_data.var(dim=dim)
    err_var = (obs_data - model_data).var(dim=dim)
    return xr.where(obs_var == 0, np.nan, 1 - err_var / obs_var)
