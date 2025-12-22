"""Harmonic Mean (element-wise between obs and model)."""


def harmonic_mean(obs_data, model_data, dim=None):
    """
    Compute the element-wise Harmonic Mean between obs and model data.
    
    Returns 2 / (1/obs + 1/model) for each corresponding element.
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim: Unused, kept for API consistency.
    
    Returns:
        xarray.DataArray: Element-wise harmonic mean of obs and model.
    """
    obs_inv = 1 / obs_data
    model_inv = 1 / model_data
    return 2 / (obs_inv + model_inv)
