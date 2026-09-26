"""Base utilities shared across all metrics.

Every metric follows the same three rules:

1. Inputs must share coordinates. Shared dimensions are aligned with
   ``join="exact"``, so labels that differ (even by floating-point noise)
   raise instead of silently shrinking the sample.
2. NaN means missing. A point missing in either input is dropped from both,
   so obs and model statistics are always computed over the same samples.
3. A score whose denominator is zero is undefined and returned as NaN,
   never as 0, so it cannot bias an average.
"""
import numpy as np
import xarray as xr


def check_aligned(*arrays):
    """
    Raise ``ValueError`` if the DataArrays do not share identical coordinates.

    Non-DataArray arguments (scalars, None) are ignored.
    """
    arrays = [a for a in arrays if isinstance(a, xr.DataArray)]
    if len(arrays) < 2:
        return
    try:
        xr.align(*arrays, join="exact", copy=False)
    except ValueError as exc:
        raise ValueError(
            "obs_data and model_data (and any weights or reference) must have "
            "identical coordinates on their shared dimensions. Put them on the "
            "same grid first, e.g. `model = model.interp_like(obs)`, or "
            "`obs, model = xr.align(obs, model, join='inner')` to keep only the "
            f"common points. xarray reported: {exc}"
        ) from exc


def paired(obs_data, model_data, *others):
    """
    Check alignment and mask every input wherever any input is missing.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        *others: Extra inputs used element-wise alongside obs and model
            (weights, reference forecasts). Scalars pass through unchanged.

    Returns:
        tuple: The inputs in the same order, NaN wherever any DataArray
        input is NaN.
    """
    check_aligned(obs_data, model_data, *others)
    valid = obs_data.notnull() & model_data.notnull()
    for other in others:
        if isinstance(other, xr.DataArray):
            valid = valid & other.notnull()
    return tuple(
        a.where(valid) if isinstance(a, xr.DataArray) else a
        for a in (obs_data, model_data) + others
    )


def binarize(data, threshold):
    """Return 1.0 where ``data >= threshold``, 0.0 below it and NaN where missing."""
    return (data >= threshold).where(data.notnull())


def ratio(numerator, denominator):
    """``numerator / denominator``, NaN where the denominator is zero."""
    return numerator / denominator.where(denominator != 0)


def confusion_matrix(obs_binary, model_binary, dim=None):
    """
    Compute the confusion matrix for binary classification.

    NaN values in either input are excluded from all four counts so they
    cannot silently inflate the true-negative cell. Use :func:`binarize` to
    build the inputs so that missing data stays NaN.

    Args:
        obs_binary (xarray.DataArray): The binarized observed data (0/1/NaN).
        model_binary (xarray.DataArray): The binarized modeled data (0/1/NaN).
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


def contingency(obs_data, model_data, threshold, dim=None):
    """
    Contingency-table counts for ``value >= threshold`` events.

    Missing points in either input are excluded. Counts are returned as
    floats so that products such as ``tp * tn`` cannot overflow int64 on
    large samples.

    Returns:
        tuple: (tn, fp, fn, tp) as float DataArrays.
    """
    obs_data, model_data = paired(obs_data, model_data)
    counts = confusion_matrix(
        binarize(obs_data, threshold), binarize(model_data, threshold), dim
    )
    return tuple(c.astype(float) for c in counts)


def distributions(obs_data, model_data, dim=None):
    """
    Normalise obs and model to probability distributions over ``dim``.

    Both inputs are masked to the points where both are valid and
    non-negative, so P and Q always cover the same bins.

    Returns:
        tuple: (p, q) DataArrays that each sum to 1 over ``dim``.
    """
    obs_data, model_data = paired(obs_data, model_data)
    valid = (obs_data >= 0) & (model_data >= 0)
    obs_data = obs_data.where(valid)
    model_data = model_data.where(valid)
    return (
        ratio(obs_data, obs_data.sum(dim=dim)),
        ratio(model_data, model_data.sum(dim=dim)),
    )


def sample_size(data, dim=None):
    """Number of non-missing points along ``dim`` (all dims if None)."""
    return data.notnull().sum(dim=dim)


def constant(data, dim=None):
    """
    True where ``data`` has fewer than two different valid values along ``dim``.

    Checked on the values, not on a computed variance: rounding makes the
    variance of constant data a tiny positive number (0.1, 0.1, 0.1 gives
    1.9e-34), and a score divided by it would blow up instead of being NaN.
    """
    return ~(data.max(dim=dim) > data.min(dim=dim))


def alpha_sum(p, q, alpha, dim=None):
    """
    ``sum(p**alpha * q**(1 - alpha))`` over ``dim`` for ``alpha >= 0``.

    Each bin takes its exact limit: a bin with p == 0 adds 0, and a bin
    with p > 0 and q == 0 adds 0 for alpha < 1 and +inf for alpha > 1.
    NaN where P or Q is undefined.
    """
    term = xr.where(p > 0, p ** alpha * q.where(q > 0) ** (1 - alpha), 0.0)
    term = xr.where((p > 0) & (q == 0), np.inf if alpha > 1 else 0.0, term)
    return term.where(p.notnull() & q.notnull()).sum(dim=dim, min_count=1)


def safe_log(x):
    """Natural log that returns NaN (without warnings) outside (0, inf)."""
    return np.log(x.where(x > 0))
