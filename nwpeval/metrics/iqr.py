"""Interquartile Range (IQR) of residuals."""


def iqr(obs_data, model_data, dim=None):
    """
    Compute the Interquartile Range (IQR) of forecast residuals.

    Residuals are defined as model_data - obs_data. IQR is Q3 - Q1
    of the residuals.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed IQR values.
    """
    residuals = model_data - obs_data
    q1 = residuals.quantile(0.25, dim=dim)
    q3 = residuals.quantile(0.75, dim=dim)
    return q3 - q1
