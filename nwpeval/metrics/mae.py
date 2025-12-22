"""Mean Absolute Error (MAE)."""
import numpy as np


def mae(obs_data, model_data, dim=None):
    """
    Calculate the Mean Absolute Error (MAE).
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed MAE values.
    """
    return np.abs(obs_data - model_data).mean(dim=dim)
