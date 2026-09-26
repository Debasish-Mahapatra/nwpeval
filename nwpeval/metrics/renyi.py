"""Renyi Divergence."""
import numpy as np
import xarray as xr
from ._base import alpha_sum, distributions, safe_log


def renyi(obs_data, model_data, alpha, dim=None):
    """
    Compute the Renyi Divergence of order alpha (alpha >= 0, alpha != 1).

    D_alpha(P || Q) = 1/(alpha - 1) * log(sum(p^alpha * q^(1-alpha)))
    where p and q are probability distributions formed by normalising
    `obs_data` and `model_data` over `dim`.

    The divergence is +inf where the two fields share no mass, and for
    alpha > 1 also wherever the model has no mass at a point where obs
    has some (like :func:`mkldiv`, its alpha -> 1 limit).

    Points missing (NaN) or negative in either input are dropped from both
    distributions, so P and Q always cover the same bins.

    P and Q describe where the mass is (e.g. where the rain falls), so this
    compares the two fields point by point, not the spread of their values.
    For that, use :func:`wasserstein` or pass histograms (counts per bin).

    Args:
        obs_data (xarray.DataArray): The observed data (must be >= 0).
        model_data (xarray.DataArray): The modeled data (must be >= 0).
        alpha (float): The order parameter, alpha >= 0 and alpha != 1.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed Renyi Divergence values.
    """
    if alpha == 1:
        raise ValueError("Renyi divergence is undefined at alpha == 1.")
    if alpha < 0:
        raise ValueError(f"Renyi divergence needs alpha >= 0, got {alpha}.")
    obs_prob, model_prob = distributions(obs_data, model_data, dim)
    inner = alpha_sum(obs_prob, model_prob, alpha, dim)
    # inner == 0 means no shared mass (alpha < 1): the divergence is +inf
    return xr.where(inner > 0, safe_log(inner) / (alpha - 1), np.inf).where(inner.notnull())
