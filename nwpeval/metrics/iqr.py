"""Interquartile Range (IQR)."""


def iqr(obs_data, model_data, dim=None):
    """
    Compute the Interquartile Range (IQR).
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed IQR values.
    """
    q1 = model_data.quantile(0.25, dim=dim)
    q3 = model_data.quantile(0.75, dim=dim)
    return q3 - q1
