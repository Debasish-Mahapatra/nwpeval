"""Normalized Mean Squared Error (NMSE)."""
import numpy as np
import xarray as xr
from ._base import paired


def nmse(obs_data, model_data, dim=None):
    """
    Compute the Normalized Mean Squared Error (NMSE).

    NMSE = MSE / (mean(obs))^2.

    Air-quality studies (Chang and Hanna, 2004) often define NMSE as
    MSE / (mean(obs) * mean(model)) instead.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed NMSE values. Returns NaN where the
        observation mean is zero.
    """
    obs_data, model_data = paired(obs_data, model_data)
    mse = ((model_data - obs_data) ** 2).mean(dim=dim)
    obs_mean = obs_data.mean(dim=dim)
    return xr.where(obs_mean == 0, np.nan, mse / (obs_mean ** 2))
