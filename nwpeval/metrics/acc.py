"""Anomaly Correlation Coefficient (ACC)."""
from ._base import check_aligned, constant, paired, ratio


def acc(obs_data, model_data, climatology=None, dim=None):
    """
    Calculate the Anomaly Correlation Coefficient (ACC).

    ACC measures the correlation between forecast and observation anomalies,
    where anomalies are deviations from a climatological reference. This is
    the uncentred form:
        ACC = sum(f' * o') / sqrt(sum(f'^2) * sum(o'^2))
    where f' = model - climatology and o' = obs - climatology.

    Supply a real climatology whenever one exists. Without it, the mean of
    the observations over ``dim`` is used for both anomalies, so a mean bias
    in the model lowers the score. For a bias-insensitive correlation use
    :func:`pcc`.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled/forecast data.
        climatology (xarray.DataArray, optional): The climatological reference.
            If None, the mean of obs_data over the specified dimensions
            (valid obs/model pairs only) is used.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed ACC values.
    """
    obs_data, model_data = paired(obs_data, model_data)
    if climatology is None:
        # Where obs never change, ACC is undefined: mask the mean so the
        # result is NaN instead of a score of rounding noise
        climatology = obs_data.mean(dim=dim).where(~constant(obs_data, dim))
    else:
        check_aligned(obs_data, climatology)

    obs_anom = obs_data - climatology
    model_anom = model_data - climatology

    numerator = (obs_anom * model_anom).sum(dim=dim)
    denominator = ((obs_anom ** 2).sum(dim=dim) * (model_anom ** 2).sum(dim=dim)) ** 0.5
    return ratio(numerator, denominator)
