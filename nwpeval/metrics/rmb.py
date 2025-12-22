"""Relative Mean Bias (RMB)."""


def rmb(obs_data, model_data, dim=None):
    """
    Compute the Relative Mean Bias (RMB).
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed RMB values.
    """
    bias = (model_data - obs_data).sum(dim=dim)
    obs_sum = obs_data.sum(dim=dim)
    return bias / obs_sum
