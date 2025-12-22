"""Pearson Correlation Coefficient (PCC)."""
import xarray as xr


def pcc(obs_data, model_data, dim=None):
    """
    Compute the Pearson Correlation Coefficient (PCC).
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed PCC values.
    """
    return xr.corr(model_data, obs_data, dim=dim)
