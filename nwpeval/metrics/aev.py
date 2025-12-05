"""Adjusted Explained Variance (AEV)."""


def aev(obs_data, model_data, dim=None):
    """
    Compute the Adjusted Explained Variance (AEV).
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed AEV values.
    """
    obs_var = obs_data.var(dim=dim)
    err_var = (obs_data - model_data).var(dim=dim)
    return 1 - err_var / obs_var
