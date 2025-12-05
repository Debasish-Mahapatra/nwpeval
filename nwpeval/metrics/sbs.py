"""Symmetric Brier Score (SBS)."""


def sbs(obs_data, model_data, dim=None):
    """
    Compute the Symmetric Brier Score (SBS).
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed SBS values.
    """
    return ((model_data - obs_data) ** 2).mean(dim=dim)
