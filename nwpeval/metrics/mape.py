"""Mean Absolute Percentage Error (MAPE)."""
import numpy as np


def mape(obs_data, model_data, dim=None):
    """
    Compute the Mean Absolute Percentage Error (MAPE).
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed MAPE values.
    """
    abs_percent_error = np.abs((model_data - obs_data) / obs_data)
    return 100 * abs_percent_error.mean(dim=dim)
