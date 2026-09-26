"""Tsallis Divergence."""
from ._base import alpha_sum, distributions


def tsallis(obs_data, model_data, alpha, dim=None):
    """
    Compute the Tsallis Divergence of order alpha (alpha >= 0, alpha != 1).

    D_alpha(P || Q) = 1/(alpha - 1) * (sum(p^alpha * q^(1-alpha)) - 1)

    For alpha > 1 the divergence is +inf wherever the model has no mass at
    a point where obs has some.

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
        xarray.DataArray: The computed Tsallis Divergence values.
    """
    if alpha == 1:
        raise ValueError("Tsallis divergence is undefined at alpha == 1.")
    if alpha < 0:
        raise ValueError(f"Tsallis divergence needs alpha >= 0, got {alpha}.")
    obs_prob, model_prob = distributions(obs_data, model_data, dim)
    inner = alpha_sum(obs_prob, model_prob, alpha, dim)
    return (inner - 1) / (alpha - 1)
