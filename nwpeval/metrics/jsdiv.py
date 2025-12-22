"""Jensen-Shannon Divergence (JSDIV)."""
import numpy as np


def jsdiv(obs_data, model_data, dim=None):
    """
    Compute the Jensen-Shannon Divergence (JSDIV).
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed JSDIV values.
    """
    obs_prob = obs_data / obs_data.sum(dim=dim)
    model_prob = model_data / model_data.sum(dim=dim)
    m = 0.5 * (obs_prob + model_prob)
    return 0.5 * ((obs_prob * np.log(obs_prob / m)).sum(dim=dim) + 
                  (model_prob * np.log(model_prob / m)).sum(dim=dim))
