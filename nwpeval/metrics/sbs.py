"""Symmetric Brier Score (SBS)."""
import numpy as np
import xarray as xr


def sbs(obs_data, model_data, dim=None):
    """
    Compute the Symmetric Brier Score (SBS).

    SBS = mean[(p - o)^2 + ((1 - p) - (1 - o))^2] = 2 * mean[(p - o)^2]
    where `model_data` is a probabilistic forecast in [0, 1] and
    `obs_data` is a binary observation (0 or 1).

    Args:
        obs_data (xarray.DataArray): The binary observed data (0 or 1).
        model_data (xarray.DataArray): The probabilistic forecast (in [0, 1]).
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed SBS values.
    """
    p = xr.where((model_data >= 0) & (model_data <= 1), model_data, np.nan)
    o = xr.where((obs_data == 0) | (obs_data == 1), obs_data, np.nan)
    return 2 * ((p - o) ** 2).mean(dim=dim)
