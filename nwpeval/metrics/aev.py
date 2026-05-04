"""Adjusted Explained Variance (AEV)."""
import numpy as np
import xarray as xr


def aev(obs_data, model_data, dim=None, n_predictors=1):
    """
    Compute the Adjusted Explained Variance (AEV).

    AEV = 1 - (1 - EVS) * (n - 1) / (n - p - 1)
    where n is the sample size along `dim`, p is the number of predictors,
    and EVS is the explained variance score.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.
        n_predictors (int): Number of predictors used by the model. Defaults to 1.

    Returns:
        xarray.DataArray: The computed AEV values.
    """
    obs_var = obs_data.var(dim=dim)
    err_var = (obs_data - model_data).var(dim=dim)
    evs = xr.where(obs_var == 0, np.nan, 1 - err_var / obs_var)

    if dim is None:
        n = obs_data.size
    elif isinstance(dim, str):
        n = obs_data.sizes[dim]
    else:
        n = int(np.prod([obs_data.sizes[d] for d in dim]))

    denom = n - n_predictors - 1
    if denom <= 0:
        return xr.full_like(evs, np.nan, dtype=float)
    factor = (n - 1) / denom
    return 1 - (1 - evs) * factor
