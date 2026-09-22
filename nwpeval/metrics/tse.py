"""Total Squared Error (TSE)."""
from ._base import paired


def tse(obs_data, model_data, dim=None):
    """
    Compute the Total Squared Error (TSE).
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed TSE values. NaN where there is no
        valid obs/model pair.
    """
    obs_data, model_data = paired(obs_data, model_data)
    return ((model_data - obs_data) ** 2).sum(dim=dim, min_count=1)
