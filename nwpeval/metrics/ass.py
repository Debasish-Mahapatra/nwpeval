"""Absolute Skill Score (ASS)."""
import numpy as np
import xarray as xr


def ass(obs_data, model_data, reference_error, dim=None):
    """
    Compute the Absolute Skill Score (ASS).

    ASS = 1 - mean(|model - obs|) / mean(reference_error)

    `reference_error` may either be a scalar reference MAE or an array of
    per-element absolute errors from a reference forecast; in the latter case
    its mean over `dim` is used.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        reference_error (xarray.DataArray or float): The reference absolute
            error (per-element) or its aggregate.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed ASS values.
    """
    abs_error = np.abs(model_data - obs_data).mean(dim=dim)

    if isinstance(reference_error, xr.DataArray) and reference_error.ndim > 0 and dim is not None:
        ref = reference_error.mean(dim=dim) if any(d in reference_error.dims for d in ([dim] if isinstance(dim, str) else dim)) else reference_error
    else:
        ref = reference_error

    return xr.where(ref == 0, np.nan, 1 - abs_error / ref)
