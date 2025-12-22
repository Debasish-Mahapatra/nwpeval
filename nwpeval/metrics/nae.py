"""Normalized Absolute Error (NAE)."""
import numpy as np


def nae(obs_data, model_data, dim=None):
    """
    Compute the Normalized Absolute Error (NAE).
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed NAE values.
    """
    abs_error = np.abs(model_data - obs_data).sum(dim=dim)
    abs_obs = np.abs(obs_data).sum(dim=dim)
    return abs_error / abs_obs
