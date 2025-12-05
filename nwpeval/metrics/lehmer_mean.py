"""Lehmer Mean."""


def lehmer_mean(obs_data, model_data, p, dim=None):
    """
    Compute the Lehmer Mean.
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        p (float): The power parameter for the Lehmer Mean.
        dim: Unused, kept for API consistency.
    
    Returns:
        xarray.DataArray: The computed Lehmer Mean values.
    """
    obs_pow = obs_data ** p
    model_pow = model_data ** p
    return (obs_pow + model_pow) / (obs_data ** (p - 1) + model_data ** (p - 1))
