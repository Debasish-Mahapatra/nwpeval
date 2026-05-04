"""Normalized Absolute Error (NAE)."""
import numpy as np
import xarray as xr


def nae(obs_data, model_data, dim=None):
    """
    Compute the Normalized Absolute Error (NAE).

    NAE = sum(|model - obs|) / sum(|obs|).

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed NAE values. Returns NaN where the
        sum of |obs| is zero.
    """
    abs_error = np.abs(model_data - obs_data).sum(dim=dim)
    abs_obs = np.abs(obs_data).sum(dim=dim)
    return xr.where(abs_obs == 0, np.nan, abs_error / abs_obs)
