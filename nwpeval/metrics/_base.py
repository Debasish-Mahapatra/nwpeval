"""Base utilities shared across all metrics."""
import numpy as np


def confusion_matrix(obs_binary, model_binary, dim=None):
    """
    Compute the confusion matrix for binary classification.

    Args:
        obs_binary (xarray.DataArray): The binarized observed data.
        model_binary (xarray.DataArray): The binarized modeled data.
        dim (str, list, or None): The dimension(s) along which to compute.
                                  If None, compute over the entire data.

    Returns:
        tuple: (tn, fp, fn, tp) - confusion matrix values.
    """
    tn = (obs_binary == 0) & (model_binary == 0)
    fp = (obs_binary == 0) & (model_binary == 1)
    fn = (obs_binary == 1) & (model_binary == 0)
    tp = (obs_binary == 1) & (model_binary == 1)

    if dim is not None:
        tn = tn.sum(dim=dim)
        fp = fp.sum(dim=dim)
        fn = fn.sum(dim=dim)
        tp = tp.sum(dim=dim)

    return tn, fp, fn, tp
