"""Geometric Mean Bias (GMB)."""
import numpy as np
from ._base import paired, ratio


def gmb(obs_data, model_data, dim=None):
    """
    Compute the Geometric Mean Bias (GMB).

    GMB = exp(mean(log(model))) / exp(mean(log(obs)))
        = geometric_mean(model) / geometric_mean(obs).

    GMB > 1 means the model is too high. The MG of Chang and Hanna (2004)
    is the other way round (obs over model), i.e. 1 / GMB.

    Inputs must be strictly positive. A pair where either value is
    non-positive (or missing) is dropped from both means.

    Args:
        obs_data (xarray.DataArray): The observed data (must be > 0).
        model_data (xarray.DataArray): The modeled data (must be > 0).
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed GMB values.
    """
    obs_data, model_data = paired(obs_data, model_data)
    positive = (obs_data > 0) & (model_data > 0)
    model_geom = np.exp(np.log(model_data.where(positive)).mean(dim=dim))
    obs_geom = np.exp(np.log(obs_data.where(positive)).mean(dim=dim))
    return ratio(model_geom, obs_geom)
