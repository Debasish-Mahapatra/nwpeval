"""Tsallis Divergence."""


def tsallis(obs_data, model_data, alpha, dim=None):
    """
    Compute the Tsallis Divergence.
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        alpha (float): The parameter for the Tsallis Divergence (alpha != 1).
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed Tsallis Divergence values.
    """
    obs_prob = obs_data / obs_data.sum(dim=dim)
    model_prob = model_data / model_data.sum(dim=dim)
    return (1 / (alpha - 1)) * ((obs_prob ** alpha / model_prob ** (alpha - 1)).sum(dim=dim) - 1)
