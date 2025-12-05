"""Geometric Mean (element-wise between obs and model)."""
import numpy as np


def geometric_mean(obs_data, model_data, dim=None):
    """
    Compute the element-wise Geometric Mean between obs and model data.
    
    Returns sqrt(obs * model) for each corresponding element.
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim: Unused, kept for API consistency.
    
    Returns:
        xarray.DataArray: Element-wise geometric mean of obs and model.
    """
    return np.sqrt(obs_data * model_data)
