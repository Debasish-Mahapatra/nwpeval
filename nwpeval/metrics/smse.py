"""Scaled Mean Squared Error (SMSE)."""
import numpy as np
import xarray as xr


def smse(obs_data, model_data, dim=None):
    """
    Compute the Scaled Mean Squared Error (SMSE).

    SMSE = MSE / var(obs).

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed SMSE values. Returns NaN where the
        observation variance is zero.
    """
    mse = ((model_data - obs_data) ** 2).mean(dim=dim)
    obs_var = obs_data.var(dim=dim)
    return xr.where(obs_var == 0, np.nan, mse / obs_var)
