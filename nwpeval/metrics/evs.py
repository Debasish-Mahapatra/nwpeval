"""Explained Variance Score (EVS)."""


def evs(obs_data, model_data, dim=None):
    """
    Compute the Explained Variance Score (EVS).
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed EVS values.
    """
    obs_var = obs_data.var(dim=dim)
    err_var = (obs_data - model_data).var(dim=dim)
    return 1 - err_var / obs_var
