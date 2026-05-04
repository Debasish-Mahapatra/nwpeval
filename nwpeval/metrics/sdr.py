"""Standard Deviation Ratio (SDR)."""
import numpy as np
import xarray as xr


def sdr(obs_data, model_data, dim=None):
    """
    Compute the Standard Deviation Ratio (SDR).

    SDR = std(model) / std(obs).

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed SDR values. Returns NaN where the
        observation standard deviation is zero.
    """
    obs_std = obs_data.std(dim=dim)
    model_std = model_data.std(dim=dim)
    return xr.where(obs_std == 0, np.nan, model_std / obs_std)
