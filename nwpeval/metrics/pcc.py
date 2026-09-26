"""Pearson Correlation Coefficient (PCC)."""
import xarray as xr
from ._base import constant, paired


def pcc(obs_data, model_data, dim=None):
    """
    Compute the Pearson Correlation Coefficient (PCC).
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed PCC values. NaN where either input
        does not vary.
    """
    obs_data, model_data = paired(obs_data, model_data)
    corr = xr.corr(model_data, obs_data, dim=dim)
    return corr.where(~(constant(obs_data, dim) | constant(model_data, dim)))
