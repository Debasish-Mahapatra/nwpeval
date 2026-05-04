"""Logarithmic Mean Bias Error (LMBE)."""
import numpy as np


def lmbe(obs_data, model_data, dim=None):
    """
    Compute the Logarithmic Mean Bias Error (LMBE).

    LMBE = mean(log(model + 1) - log(obs + 1)).

    Inputs are expected to be non-negative; values strictly less than -1
    yield NaN due to the domain of the logarithm.

    Args:
        obs_data (xarray.DataArray): The observed data (assumed >= 0).
        model_data (xarray.DataArray): The modeled data (assumed >= 0).
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed LMBE values.
    """
    return (np.log1p(model_data) - np.log1p(obs_data)).mean(dim=dim)
