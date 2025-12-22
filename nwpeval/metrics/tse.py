"""Total Squared Error (TSE)."""


def tse(obs_data, model_data, dim=None):
    """
    Compute the Total Squared Error (TSE).
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed TSE values.
    """
    return ((model_data - obs_data) ** 2).sum(dim=dim)
