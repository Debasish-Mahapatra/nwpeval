"""Standard Deviation Ratio (SDR)."""


def sdr(obs_data, model_data, dim=None):
    """
    Compute the Standard Deviation Ratio (SDR).
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed SDR values.
    """
    obs_std = obs_data.std(dim=dim)
    model_std = model_data.std(dim=dim)
    return model_std / obs_std
