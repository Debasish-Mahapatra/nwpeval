"""Anomaly Correlation Coefficient (ACC)."""
import xarray as xr


def acc(obs_data, model_data, dim=None):
    """
    Calculate the Anomaly Correlation Coefficient (ACC).
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed ACC values.
    """
    return xr.corr(obs_data, model_data, dim=dim)
