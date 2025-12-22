"""Coefficient of Determination (R²)."""


def r2(obs_data, model_data, dim=None):
    """
    Compute the Coefficient of Determination (R²).
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed R² values.
    """
    ssr = ((model_data - obs_data) ** 2).sum(dim=dim)
    sst = ((obs_data - obs_data.mean(dim=dim)) ** 2).sum(dim=dim)
    return 1 - ssr / sst
