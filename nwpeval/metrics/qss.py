"""Quadratic Skill Score (QSS)."""
import numpy as np
import xarray as xr


def qss(obs_data, model_data, reference_forecast, dim=None):
    """
    Compute the Quadratic Skill Score (QSS).

    QSS = 1 - MSE_model / MSE_reference.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        reference_forecast (xarray.DataArray or float): The reference forecast.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed QSS values. Returns NaN where the
        reference MSE is zero (the reference is itself a perfect forecast).
    """
    mse_model = ((model_data - obs_data) ** 2).mean(dim=dim)
    mse_ref = ((reference_forecast - obs_data) ** 2).mean(dim=dim)
    return xr.where(mse_ref == 0, np.nan, 1 - mse_model / mse_ref)
