"""Mean Kullback-Leibler Divergence (MKLDIV)."""
import numpy as np
import xarray as xr


def mkldiv(obs_data, model_data, dim=None):
    """
    Compute the Kullback-Leibler divergence D_KL(P || Q).

    P (from `obs_data`) and Q (from `model_data`) are formed by normalising
    each input to sum to 1 over `dim`. Both inputs must be non-negative.
    Where p > 0 and q == 0 the divergence is +inf; where p == 0 the term
    contributes 0 (by convention 0 * log(0) = 0).

    Args:
        obs_data (xarray.DataArray): The observed data (must be >= 0).
        model_data (xarray.DataArray): The modeled data (must be >= 0).
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The KL divergence.
    """
    obs_safe = xr.where(obs_data >= 0, obs_data, np.nan)
    model_safe = xr.where(model_data >= 0, model_data, np.nan)
    obs_total = obs_safe.sum(dim=dim)
    model_total = model_safe.sum(dim=dim)
    obs_prob = xr.where(obs_total == 0, np.nan, obs_safe / obs_total)
    model_prob = xr.where(model_total == 0, np.nan, model_safe / model_total)

    ratio = xr.where(model_prob == 0, np.inf, obs_prob / model_prob)
    log_ratio = xr.where(obs_prob == 0, 0.0, np.log(xr.where(ratio > 0, ratio, 1.0)))
    term = xr.where(obs_prob == 0, 0.0, obs_prob * log_ratio)
    term = xr.where((obs_prob > 0) & (model_prob == 0), np.inf, term)
    return term.sum(dim=dim)
