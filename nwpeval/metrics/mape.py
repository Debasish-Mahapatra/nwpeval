"""Mean Absolute Percentage Error (MAPE)."""
import numpy as np
import xarray as xr


def mape(obs_data, model_data, dim=None):
    """
    Compute the Mean Absolute Percentage Error (MAPE).

    MAPE = 100 * mean(|model - obs| / |obs|), with elements where obs == 0
    excluded from the mean.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed MAPE values.
    """
    valid = obs_data != 0
    obs_safe = xr.where(valid, obs_data, np.nan)
    abs_percent_error = np.abs((model_data - obs_data) / obs_safe)
    return 100 * abs_percent_error.mean(dim=dim, skipna=True)
