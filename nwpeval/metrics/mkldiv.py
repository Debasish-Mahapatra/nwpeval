"""Mean Kullback-Leibler Divergence (MKLDIV)."""
import numpy as np


def mkldiv(obs_data, model_data, dim=None):
    """
    Compute the Mean Kullback-Leibler Divergence (MKLDIV).
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed MKLDIV values.
    """
    obs_prob = obs_data / obs_data.sum(dim=dim)
    model_prob = model_data / model_data.sum(dim=dim)
    return (obs_prob * np.log(obs_prob / model_prob)).sum(dim=dim)
