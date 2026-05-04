"""Base utilities shared across all metrics."""
import numpy as np
import xarray as xr


def confusion_matrix(obs_binary, model_binary, dim=None):
    """
    Compute the confusion matrix for binary classification.

    NaN values in either input are excluded from all four counts so they
    cannot silently inflate the true-negative cell.

    Args:
        obs_binary (xarray.DataArray): The binarized observed data (0/1).
        model_binary (xarray.DataArray): The binarized modeled data (0/1).
        dim (str, list, or None): The dimension(s) along which to compute.
            If None, compute over the entire data.

    Returns:
        tuple: (tn, fp, fn, tp) - confusion matrix values.
    """
    valid = obs_binary.notnull() & model_binary.notnull()

    tn = ((obs_binary == 0) & (model_binary == 0) & valid).sum(dim=dim)
    fp = ((obs_binary == 0) & (model_binary == 1) & valid).sum(dim=dim)
    fn = ((obs_binary == 1) & (model_binary == 0) & valid).sum(dim=dim)
    tp = ((obs_binary == 1) & (model_binary == 1) & valid).sum(dim=dim)

    return tn, fp, fn, tp
