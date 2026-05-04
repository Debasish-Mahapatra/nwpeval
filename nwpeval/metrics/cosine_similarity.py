"""Cosine Similarity."""
import numpy as np
import xarray as xr


def cosine_similarity(obs_data, model_data, dim=None):
    """
    Compute the Cosine Similarity.

    cos(theta) = sum(model * obs) / (||model|| * ||obs||).

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed Cosine Similarity values. Returns NaN
        where either norm is zero.
    """
    dot_product = (model_data * obs_data).sum(dim=dim)
    model_norm = np.sqrt((model_data ** 2).sum(dim=dim))
    obs_norm = np.sqrt((obs_data ** 2).sum(dim=dim))
    denom = model_norm * obs_norm
    return xr.where(denom == 0, np.nan, dot_product / denom)
