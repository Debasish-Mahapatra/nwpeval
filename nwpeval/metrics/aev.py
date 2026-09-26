"""Adjusted Explained Variance (AEV)."""
from ._base import constant, paired, ratio, sample_size


def aev(obs_data, model_data, dim=None, n_predictors=1):
    """
    Compute the Adjusted Explained Variance (AEV).

    AEV = 1 - (1 - EVS) * (n - 1) / (n - p - 1)
    where n is the number of valid obs/model pairs along `dim`, p is the
    number of predictors, and EVS is the explained variance score.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.
        n_predictors (int): Number of predictors used by the model. Defaults to 1.

    Returns:
        xarray.DataArray: The computed AEV values.
    """
    obs_data, model_data = paired(obs_data, model_data)
    obs_var = obs_data.var(dim=dim).where(~constant(obs_data, dim))
    evs = 1 - ratio((obs_data - model_data).var(dim=dim), obs_var)

    n = sample_size(obs_data, dim=dim)
    denom = n - n_predictors - 1
    factor = (n - 1) / denom.where(denom > 0)
    return 1 - (1 - evs) * factor
