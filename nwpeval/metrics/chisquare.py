"""Chi-Square Distance."""


def chisquare(obs_data, model_data, dim=None):
    """
    Compute the Chi-Square Distance.
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed Chi-Square Distance values.
    """
    obs_prob = obs_data / obs_data.sum(dim=dim)
    model_prob = model_data / model_data.sum(dim=dim)
    return ((obs_prob - model_prob) ** 2 / model_prob).sum(dim=dim)
