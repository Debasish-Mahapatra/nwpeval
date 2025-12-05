"""Normalized Mean Squared Error (NMSE)."""


def nmse(obs_data, model_data, dim=None):
    """
    Compute the Normalized Mean Squared Error (NMSE).
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed NMSE values.
    """
    mse = ((model_data - obs_data) ** 2).mean(dim=dim)
    obs_mean = obs_data.mean(dim=dim)
    return mse / (obs_mean ** 2)
