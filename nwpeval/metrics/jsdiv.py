"""Jensen-Shannon Divergence (JSDIV)."""
import numpy as np
import xarray as xr
from ._base import distributions


def jsdiv(obs_data, model_data, dim=None):
    """
    Compute the Jensen-Shannon divergence between two empirical distributions.

    JSD(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)  where M = 0.5*(P + Q).
    Inputs must be non-negative. By convention 0 * log(0) = 0, so empty
    bins in P or Q contribute zero.

    Points missing (NaN) or negative in either input are dropped from both
    distributions, so P and Q always cover the same bins.

    Args:
        obs_data (xarray.DataArray): The observed data (must be >= 0).
        model_data (xarray.DataArray): The modeled data (must be >= 0).
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The JS divergence.
    """
    obs_prob, model_prob = distributions(obs_data, model_data, dim)

    m = 0.5 * (obs_prob + model_prob)

    def _kl(p, q):
        ratio = xr.where(q > 0, p / xr.where(q == 0, 1.0, q), np.inf)
        log_ratio = xr.where(p == 0, 0.0, np.log(xr.where(ratio > 0, ratio, 1.0)))
        return xr.where(p == 0, 0.0, p * log_ratio).sum(dim=dim, min_count=1)

    return 0.5 * (_kl(obs_prob, m) + _kl(model_prob, m))
