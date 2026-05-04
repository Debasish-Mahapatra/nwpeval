"""Lehmer Mean (element-wise between obs and model)."""


def lehmer_mean(obs_data, model_data, p, dim=None):
    """
    Compute the element-wise Lehmer Mean of order ``p`` between obs and model.

    L_p(obs, model) = (obs**p + model**p) / (obs**(p-1) + model**(p-1)).

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        p (float): The power parameter for the Lehmer Mean.
        dim: Unused, kept for API consistency.

    Returns:
        xarray.DataArray: Element-wise Lehmer mean of obs and model.
    """
    obs_pow = obs_data ** p
    model_pow = model_data ** p
    return (obs_pow + model_pow) / (obs_data ** (p - 1) + model_data ** (p - 1))
