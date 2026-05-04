"""Geometric Mean Bias (GMB)."""
import numpy as np
import xarray as xr


def gmb(obs_data, model_data, dim=None):
    """
    Compute the Geometric Mean Bias (GMB).

    GMB = exp(mean(log(model))) / exp(mean(log(obs)))
        = geometric_mean(model) / geometric_mean(obs).

    Inputs must be strictly positive. Non-positive values are masked to NaN
    so they do not silently produce -inf or warnings under log.

    Args:
        obs_data (xarray.DataArray): The observed data (must be > 0).
        model_data (xarray.DataArray): The modeled data (must be > 0).
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed GMB values.
    """
    obs_safe = xr.where(obs_data > 0, obs_data, np.nan)
    model_safe = xr.where(model_data > 0, model_data, np.nan)
    model_geom = np.exp(np.log(model_safe).mean(dim=dim, skipna=True))
    obs_geom = np.exp(np.log(obs_safe).mean(dim=dim, skipna=True))
    return xr.where(obs_geom == 0, np.nan, model_geom / obs_geom)
