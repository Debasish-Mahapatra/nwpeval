"""Mean Kullback-Leibler Divergence (MKLDIV)."""
import numpy as np
import xarray as xr
from ._base import distributions, safe_log


def mkldiv(obs_data, model_data, dim=None):
    """
    Compute the Kullback-Leibler divergence D_KL(P || Q).

    P (from `obs_data`) and Q (from `model_data`) are formed by normalising
    each input to sum to 1 over `dim`. Both inputs must be non-negative.
    Where p > 0 and q == 0 the divergence is +inf; where p == 0 the term
    contributes 0 (by convention 0 * log(0) = 0).

    Points missing (NaN) or negative in either input are dropped from both
    distributions, so P and Q always cover the same bins.

    P and Q describe where the mass is (e.g. where the rain falls), so this
    compares the two fields point by point, not the spread of their values.
    For that, use :func:`wasserstein` or pass histograms (counts per bin).

    Args:
        obs_data (xarray.DataArray): The observed data (must be >= 0).
        model_data (xarray.DataArray): The modeled data (must be >= 0).
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The KL divergence.
    """
    obs_prob, model_prob = distributions(obs_data, model_data, dim)

    term = xr.where(obs_prob > 0, obs_prob * safe_log(obs_prob / model_prob.where(model_prob > 0)), 0.0)
    term = xr.where((obs_prob > 0) & (model_prob == 0), np.inf, term)
    # Undefined (NaN) wherever either distribution is, e.g. a field with no rain at all
    term = term.where(obs_prob.notnull() & model_prob.notnull())
    return term.sum(dim=dim, min_count=1)
