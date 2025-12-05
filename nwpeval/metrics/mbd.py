"""Mean Bias Deviation (MBD)."""


def mbd(obs_data, model_data, dim=None):
    """
    Compute the Mean Bias Deviation (MBD).
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed MBD values.
    """
    return model_data.mean(dim=dim) - obs_data.mean(dim=dim)
