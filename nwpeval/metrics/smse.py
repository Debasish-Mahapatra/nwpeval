"""Scaled Mean Squared Error (SMSE)."""


def smse(obs_data, model_data, dim=None):
    """
    Compute the Scaled Mean Squared Error (SMSE).
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed SMSE values.
    """
    mse = ((model_data - obs_data) ** 2).mean(dim=dim)
    obs_var = obs_data.var(dim=dim)
    return mse / obs_var
