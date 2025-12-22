"""Normalized Root Mean Square Error (NRMSE)."""
import numpy as np


def nrmse(obs_data, model_data, dim=None):
    """
    Compute the Normalized Root Mean Squared Error (NRMSE).
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed NRMSE values.
    """
    rmse = np.sqrt(((model_data - obs_data) ** 2).mean(dim=dim))
    obs_mean = obs_data.mean(dim=dim)
    return rmse / obs_mean
