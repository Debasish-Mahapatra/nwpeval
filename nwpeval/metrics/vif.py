"""Variance Inflation Factor (VIF)."""
import numpy as np
import xarray as xr
from ._base import constant, paired


def vif(obs_data, model_data, dim=None):
    """
    Compute the Variance Inflation Factor (VIF).

    VIF = var(model) / var(obs) - 1.

    This is not the variance inflation factor of regression (1 / (1 - R^2)
    of a predictor). It is :func:`fv` minus 1.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed VIF values. Returns NaN where the
        observation variance is zero.
    """
    obs_data, model_data = paired(obs_data, model_data)
    obs_var = obs_data.var(dim=dim)
    model_var = model_data.var(dim=dim)
    return xr.where(constant(obs_data, dim), np.nan, model_var / obs_var - 1)
