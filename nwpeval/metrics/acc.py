"""Anomaly Correlation Coefficient (ACC)."""
import numpy as np
import xarray as xr


def acc(obs_data, model_data, climatology=None, dim=None):
    """
    Calculate the Anomaly Correlation Coefficient (ACC).

    ACC measures the correlation between forecast and observation anomalies,
    where anomalies are deviations from a climatological reference. This is
    the uncentred form:
        ACC = sum(f' * o') / sqrt(sum(f'^2) * sum(o'^2))
    where f' = model - climatology and o' = obs - climatology.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled/forecast data.
        climatology (xarray.DataArray, optional): The climatological reference.
            If None, the mean of obs_data over the specified dimensions is used.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed ACC values.
    """
    if climatology is None:
        climatology = obs_data.mean(dim=dim)

    obs_anom = obs_data - climatology
    model_anom = model_data - climatology

    numerator = (obs_anom * model_anom).sum(dim=dim)
    denominator = np.sqrt((obs_anom ** 2).sum(dim=dim) * (model_anom ** 2).sum(dim=dim))

    return xr.where(denominator == 0, np.nan, numerator / denominator)
