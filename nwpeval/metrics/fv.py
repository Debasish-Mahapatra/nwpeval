"""Fractional Variance (FV)."""
import numpy as np
import xarray as xr
from ._base import constant, paired


def fv(obs_data, model_data, dim=None):
    """
    Compute the Fractional Variance (FV).

    FV = var(model) / var(obs).

    FV, VIF (= FV - 1) and SDR (= sqrt(FV)) carry the same information.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed FV values. Returns NaN where the
        observation variance is zero.
    """
    obs_data, model_data = paired(obs_data, model_data)
    obs_var = obs_data.var(dim=dim)
    model_var = model_data.var(dim=dim)
    return xr.where(constant(obs_data, dim), np.nan, model_var / obs_var)
