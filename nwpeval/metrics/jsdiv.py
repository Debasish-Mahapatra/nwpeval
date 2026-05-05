"""Jensen-Shannon Divergence (JSDIV)."""
import numpy as np
import xarray as xr


def jsdiv(obs_data, model_data, dim=None):
    """
    Compute the Jensen-Shannon divergence between two empirical distributions.

    JSD(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)  where M = 0.5*(P + Q).
    Inputs must be non-negative. By convention 0 * log(0) = 0, so empty
    bins in P or Q contribute zero.

    Args:
        obs_data (xarray.DataArray): The observed data (must be >= 0).
        model_data (xarray.DataArray): The modeled data (must be >= 0).
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The JS divergence.
    """
    obs_safe = xr.where(obs_data >= 0, obs_data, np.nan)
    model_safe = xr.where(model_data >= 0, model_data, np.nan)
    obs_total = obs_safe.sum(dim=dim)
    model_total = model_safe.sum(dim=dim)
    obs_prob = xr.where(obs_total == 0, np.nan, obs_safe / obs_total)
    model_prob = xr.where(model_total == 0, np.nan, model_safe / model_total)

    m = 0.5 * (obs_prob + model_prob)

    def _kl(p, q):
        ratio = xr.where(q > 0, p / xr.where(q == 0, 1.0, q), np.inf)
        log_ratio = xr.where(p == 0, 0.0, np.log(xr.where(ratio > 0, ratio, 1.0)))
        return xr.where(p == 0, 0.0, p * log_ratio).sum(dim=dim)

    return 0.5 * (_kl(obs_prob, m) + _kl(model_prob, m))
