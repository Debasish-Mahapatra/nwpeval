"""Scaled Mean Squared Error (SMSE)."""
import numpy as np
import xarray as xr
from ._base import constant, paired


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
    obs_data, model_data = paired(obs_data, model_data)
    mse = ((model_data - obs_data) ** 2).mean(dim=dim)
    obs_var = obs_data.var(dim=dim)
    return xr.where(constant(obs_data, dim), np.nan, mse / obs_var)
