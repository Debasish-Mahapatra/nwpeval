"""Root Mean Square Error (RMSE)."""
import numpy as np


def rmse(obs_data, model_data, dim=None):
    """
    Calculate the Root Mean Square Error (RMSE).
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed RMSE values.
    """
    return np.sqrt(((obs_data - model_data) ** 2).mean(dim=dim))
