"""Anomaly Correlation Coefficient (ACC)."""
import xarray as xr


def acc(obs_data, model_data, climatology=None, dim=None):
    """
    Calculate the Anomaly Correlation Coefficient (ACC).
    
    The ACC measures the correlation between forecast and observation anomalies,
    where anomalies are deviations from a climatological reference.
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled/forecast data.
        climatology (xarray.DataArray, optional): The climatological reference.
            If None, the mean of obs_data over the specified dimensions is used.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed ACC values.
    
    Note:
        ACC = sum(f' * o') / sqrt(sum(f'^2) * sum(o'^2))
        where f' = model - climatology and o' = obs - climatology
    """
    if climatology is None:
        climatology = obs_data.mean(dim=dim)
    
    obs_anomaly = obs_data - climatology
    model_anomaly = model_data - climatology
    
    return xr.corr(obs_anomaly, model_anomaly, dim=dim)
