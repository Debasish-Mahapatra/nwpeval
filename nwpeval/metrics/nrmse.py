"""Normalized Root Mean Square Error (NRMSE)."""
import numpy as np
import xarray as xr


def nrmse(obs_data, model_data, dim=None):
    """
    Compute the Normalized Root Mean Squared Error (NRMSE).

    NRMSE = RMSE / mean(obs).

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed NRMSE values. Returns NaN where the
        observation mean is zero.
    """
    rmse_val = np.sqrt(((model_data - obs_data) ** 2).mean(dim=dim))
    obs_mean = obs_data.mean(dim=dim)
    return xr.where(obs_mean == 0, np.nan, rmse_val / obs_mean)
