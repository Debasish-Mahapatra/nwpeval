"""Fractional Variance (FV)."""


def fv(obs_data, model_data, dim=None):
    """
    Compute the Fractional Variance (FV).
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed FV values.
    """
    obs_var = obs_data.var(dim=dim)
    model_var = model_data.var(dim=dim)
    return model_var / obs_var
