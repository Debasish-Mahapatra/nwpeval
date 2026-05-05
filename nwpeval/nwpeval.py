import warnings
import numpy as np
import xarray as xr

from .metrics import (
    acc as _acc,
    seds as _seds,
    eds as _eds,
    sedi as _sedi,
    orss as _orss,
    nmse as _nmse,
    fv as _fv,
    sdr as _sdr,
    vif as _vif,
    mad as _mad,
    iqr as _iqr,
    r2 as _r2,
    nae as _nae,
    rmb as _rmb,
    mape as _mape,
    ass as _ass,
    rss as _rss,
    nrmse as _nrmse,
    lmbe as _lmbe,
    smse as _smse,
    gmb as _gmb,
    sbs as _sbs,
    aev as _aev,
    cosine_similarity as _cosine_similarity,
    f1 as _f1,
    ba as _ba,
    npv as _npv,
    lift as _lift,
    wasserstein as _wasserstein,
    bhattacharyya as _bhattacharyya,
    lehmer_mean as _lehmer_mean,
    chernoff as _chernoff,
    renyi as _renyi,
    tsallis as _tsallis,
    evs as _evs,
    mae as _mae,
    rmse as _rmse,
    pcc as _pcc,
    mbd as _mbd,
    tse as _tse,
    fss as _fss,
    ets as _ets,
    pod as _pod,
    far as _far,
    csi as _csi,
    bss as _bss,
    rpss as _rpss,
    hss as _hss,
    pss as _pss,
    fb as _fb,
    mcc as _mcc,
    gain as _gain,
    qss as _qss,
    wmae as _wmae,
    harmonic_mean as _harmonic_mean,
    geometric_mean as _geometric_mean,
    mkldiv as _mkldiv,
    jsdiv as _jsdiv,
    hellinger as _hellinger,
    tv as _tv,
    chisquare as _chisquare,
    intersection as _intersection,
)


def help(cls):
    print("Available methods in the NWP_Stats class:")
    print("-------------------------------------------")
    for method_name in dir(cls):
        if method_name.startswith("compute_"):
            method = getattr(cls, method_name)
            print(f"{method_name}:")
            print(method.__doc__)
            print("---")


class NWP_Stats:
    """
    DEPRECATED: Use standalone metric functions instead.
    
    Example:
        # Old way (deprecated)
        stats = NWP_Stats(obs, model)
        result = stats.compute_rmse()
        
        # New way (recommended)
        from nwpeval import rmse
        result = rmse(obs, model)
    """
    
    def __init__(self, obs_data, model_data):
        """
        Initialize the NWPMetrics object with observed and modeled data.
        
        .. deprecated::
            NWP_Stats is deprecated. Use standalone metric functions instead.
            Example: `from nwpeval import rmse; rmse(obs, model)`
        
        Args:
            obs_data (xarray.DataArray): The observed data.
            model_data (xarray.DataArray): The modeled data.
        """
        warnings.warn(
            "NWP_Stats is deprecated and will be removed in a future version. "
            "Use standalone metric functions instead: "
            "`from nwpeval import rmse, mae, fss; rmse(obs, model)`",
            DeprecationWarning,
            stacklevel=2
        )
        self.obs_data = obs_data
        self.model_data = model_data

    def compute_metrics(self, metrics, dim=None, thresholds=None):
        """
        Compute the specified metrics.
        
        Args:
            metrics (list): A list of metric names to compute.
            dim (str, list, or None): The dimension(s) along which to compute the metrics.
                                      If None, compute the metrics over the entire data.
            thresholds (dict): A dictionary containing threshold values for specific metrics.
        
        Returns:
            dict: A dictionary containing the computed metric values.
        """
        metric_values = {}
        for metric in metrics:
            if metric == 'MAE':
                metric_values[metric] = self.compute_mae(dim)
            elif metric == 'RMSE':
                metric_values[metric] = self.compute_rmse(dim)
            elif metric == 'ACC':
                metric_values[metric] = self.compute_acc(dim=dim)
            elif metric == 'FSS':
                threshold = thresholds.get('FSS', 0.5)
                neighborhood_size = thresholds.get('FSS_neighborhood', 3)
                spatial_dims = thresholds.get('FSS_spatial_dims', None)
                metric_values[metric] = self.compute_fss(threshold, neighborhood_size, spatial_dims, dim)
            elif metric == 'ETS':
                threshold = thresholds.get('ETS', 0.5)
                metric_values[metric] = self.compute_ets(threshold, dim)
            elif metric == 'POD':
                threshold = thresholds.get('POD', 0.5)
                metric_values[metric] = self.compute_pod(threshold, dim)
            elif metric == 'FAR':
                threshold = thresholds.get('FAR', 0.5)
                metric_values[metric] = self.compute_far(threshold, dim)
            elif metric == 'CSI':
                threshold = thresholds.get('CSI', 0.5)
                metric_values[metric] = self.compute_csi(threshold, dim)
            elif metric == 'BSS':
                threshold = thresholds.get('BSS', 0.5)
                metric_values[metric] = self.compute_bss(threshold, dim)
            elif metric == 'HSS':
                threshold = thresholds.get('HSS', 0.5)
                metric_values[metric] = self.compute_hss(threshold, dim)
            elif metric == 'PSS':
                threshold = thresholds.get('PSS', 0.5)
                metric_values[metric] = self.compute_pss(threshold, dim)
            elif metric == 'SEDS':
                threshold = thresholds.get('SEDS', 0.5)
                metric_values[metric] = self.compute_seds(threshold, dim)
            elif metric == 'FB':
                threshold = thresholds.get('FB', 0.5)
                metric_values[metric] = self.compute_fb(threshold, dim)
            elif metric == 'GSS':
                threshold = thresholds.get('GSS', 0.5)
                metric_values[metric] = self.compute_gss(threshold, dim)
            elif metric == 'H-KD':
                threshold = thresholds.get('H-KD', 0.5)
                metric_values[metric] = self.compute_hkd(threshold, dim)
            elif metric == 'ORSS':
                threshold = thresholds.get('ORSS', 0.5)
                metric_values[metric] = self.compute_orss(threshold, dim)
            elif metric == 'EDS':
                threshold = thresholds.get('EDS', 0.5)
                metric_values[metric] = self.compute_eds(threshold, dim)
            elif metric == 'SEDI':
                threshold = thresholds.get('SEDI', 0.5)
                metric_values[metric] = self.compute_sedi(threshold, dim)
            elif metric == 'RPSS':
                threshold = thresholds.get('RPSS', 0.5)
                metric_values[metric] = self.compute_rpss(threshold, dim)
            elif metric == 'TSE':
                metric_values[metric] = self.compute_tse(dim)
            elif metric == 'EVS':
                metric_values[metric] = self.compute_evs(dim)
            elif metric == 'NMSE':
                metric_values[metric] = self.compute_nmse(dim)
            elif metric == 'FV':
                metric_values[metric] = self.compute_fv(dim)
            elif metric == 'PCC':
                metric_values[metric] = self.compute_pcc(dim)
            elif metric == 'SDR':
                metric_values[metric] = self.compute_sdr(dim)
            elif metric == 'VIF':
                metric_values[metric] = self.compute_vif(dim)
            elif metric == 'MAD':
                metric_values[metric] = self.compute_mad(dim)
            elif metric == 'IQR':
                metric_values[metric] = self.compute_iqr(dim)
            elif metric == 'R2':
                metric_values[metric] = self.compute_r2(dim)
            elif metric == 'NAE':
                metric_values[metric] = self.compute_nae(dim)
            elif metric == 'RMB':
                metric_values[metric] = self.compute_rmb(dim)
            elif metric == 'MAPE':
                metric_values[metric] = self.compute_mape(dim)
            elif metric == 'WMAE':
                weights = thresholds.get('WMAE_weights')
                metric_values[metric] = self.compute_wmae(weights, dim)
            elif metric == 'ASS':
                reference_error = thresholds.get('ASS_reference_error')
                metric_values[metric] = self.compute_ass(reference_error, dim)
            elif metric == 'RSS':
                reference_skill = thresholds.get('RSS_reference_skill')
                metric_values[metric] = self.compute_rss(reference_skill, dim)
            elif metric == 'QSS':
                reference_forecast = thresholds.get('QSS_reference_forecast')
                metric_values[metric] = self.compute_qss(reference_forecast, dim)
            elif metric == 'NRMSE':
                metric_values[metric] = self.compute_nrmse(dim)
            elif metric == 'LMBE':
                metric_values[metric] = self.compute_lmbe(dim)
            elif metric == 'SMSE':
                metric_values[metric] = self.compute_smse(dim)
            elif metric == 'MBD':
                metric_values[metric] = self.compute_mbd(dim)
            elif metric == 'GMB':
                metric_values[metric] = self.compute_gmb(dim)
            elif metric == 'SBS':
                metric_values[metric] = self.compute_sbs(dim)
            elif metric == 'AEV':
                metric_values[metric] = self.compute_aev(dim)
            elif metric == 'CosineSimilarity':
                metric_values[metric] = self.compute_cosine_similarity(dim)
            elif metric == 'F1':
                threshold = thresholds.get('F1', 0.5)
                metric_values[metric] = self.compute_f1(threshold, dim)
            elif metric == 'MCC':
                threshold = thresholds.get('MCC', 0.5)
                metric_values[metric] = self.compute_mcc(threshold, dim)
            elif metric == 'BA':
                threshold = thresholds.get('BA', 0.5)
                metric_values[metric] = self.compute_ba(threshold, dim)
            elif metric == 'NPV':
                threshold = thresholds.get('NPV', 0.5)
                metric_values[metric] = self.compute_npv(threshold, dim)
            elif metric == 'Jaccard':
                threshold = thresholds.get('Jaccard', 0.5)
                metric_values[metric] = self.compute_jaccard(threshold, dim)
            elif metric == 'Gain':
                threshold = thresholds.get('Gain', 0.5)
                metric_values[metric] = self.compute_gain(threshold, dim)
            elif metric == 'Lift':
                threshold = thresholds.get('Lift', 0.5)
                metric_values[metric] = self.compute_lift(threshold, dim)
            elif metric == 'MKLDIV':
                metric_values[metric] = self.compute_mkldiv(dim)
            elif metric == 'JSDIV':
                metric_values[metric] = self.compute_jsdiv(dim)
            elif metric == 'Hellinger':
                metric_values[metric] = self.compute_hellinger(dim)
            elif metric == 'Wasserstein':
                metric_values[metric] = self.compute_wasserstein(dim)
            elif metric == 'TV':
                metric_values[metric] = self.compute_tv(dim)
            elif metric == 'ChiSquare':
                metric_values[metric] = self.compute_chisquare(dim)
            elif metric == 'Intersection':
                metric_values[metric] = self.compute_intersection(dim)
            elif metric == 'Bhattacharyya':
                metric_values[metric] = self.compute_bhattacharyya(dim)
            elif metric == 'HarmonicMean':
                metric_values[metric] = self.compute_harmonic_mean(dim)
            elif metric == 'GeometricMean':
                metric_values[metric] = self.compute_geometric_mean(dim)
            elif metric == 'LehmerMean':
                p = thresholds.get('LehmerMean_p', 2)
                metric_values[metric] = self.compute_lehmer_mean(p, dim)
            elif metric == 'Chernoff':
                alpha = thresholds.get('Chernoff_alpha', 0.5)
                metric_values[metric] = self.compute_chernoff(alpha, dim)
            elif metric == 'Renyi':
                alpha = thresholds.get('Renyi_alpha', 0.5)
                metric_values[metric] = self.compute_renyi(alpha, dim)
            elif metric == 'Tsallis':
                alpha = thresholds.get('Tsallis_alpha', 0.5)
                metric_values[metric] = self.compute_tsallis(alpha, dim)
        return metric_values
    
    def confusion_matrix(self, obs_binary, model_binary, dim=None):
        """
        Compute the confusion matrix for binary classification.

        Delegates to :func:`nwpeval.metrics.confusion_matrix`, which masks
        NaN entries and always sums (including when ``dim`` is None).

        Args:
            obs_binary (xarray.DataArray): The binarized observed data.
            model_binary (xarray.DataArray): The binarized modeled data.
            dim (str, list, or None): Dimension(s) to compute over. If None,
                sum over the entire array.

        Returns:
            tuple: (tn, fp, fn, tp).
        """
        from .metrics._base import confusion_matrix as _confusion_matrix
        return _confusion_matrix(obs_binary, model_binary, dim=dim)
    
    def compute_mae(self, dim=None):
        """Calculate the Mean Absolute Error (MAE)."""
        return _mae(self.obs_data, self.model_data, dim=dim)

    def compute_rmse(self, dim=None):
        """Calculate the Root Mean Square Error (RMSE)."""
        return _rmse(self.obs_data, self.model_data, dim=dim)

    def compute_acc(self, climatology=None, dim=None):
        """Calculate the Anomaly Correlation Coefficient (ACC).

        Args:
            climatology (xarray.DataArray, optional): The climatological reference.
                If None, the mean of obs_data over the specified dimensions is used.
            dim (str, list, or None): Dimension(s) to compute over.
        """
        return _acc(self.obs_data, self.model_data, climatology=climatology, dim=dim)

    def compute_fss(self, threshold, neighborhood_size, spatial_dims=None, reduction_dim=None):
        """
        Compute the Fractions Skill Score (FSS) for a given threshold and neighborhood size.
        """
        return _fss(
            self.obs_data,
            self.model_data,
            threshold,
            neighborhood_size,
            spatial_dims=spatial_dims,
            reduction_dim=reduction_dim,
        )

    def compute_ets(self, threshold, dim=None):
        """Compute the Equitable Threat Score (ETS) for a given threshold."""
        return _ets(self.obs_data, self.model_data, threshold, dim=dim)

    def compute_pod(self, threshold, dim=None):
        """Compute the Probability of Detection (POD) for a given threshold."""
        return _pod(self.obs_data, self.model_data, threshold, dim=dim)

    def compute_far(self, threshold, dim=None):
        """Compute the False Alarm Ratio (FAR) for a given threshold."""
        return _far(self.obs_data, self.model_data, threshold, dim=dim)

    def compute_csi(self, threshold, dim=None):
        """Compute the Critical Success Index (CSI) for a given threshold."""
        return _csi(self.obs_data, self.model_data, threshold, dim=dim)

    def compute_bss(self, threshold, dim=None):
        """Compute the Brier Skill Score (BSS) for a given threshold."""
        return _bss(self.obs_data, self.model_data, threshold, dim=dim)

    def compute_hss(self, threshold, dim=None):
        """Compute the Heidke Skill Score (HSS) for a given threshold."""
        return _hss(self.obs_data, self.model_data, threshold, dim=dim)

    def compute_pss(self, threshold, dim=None):
        """Compute the Peirce Skill Score (PSS) for a given threshold."""
        return _pss(self.obs_data, self.model_data, threshold, dim=dim)

    def compute_seds(self, threshold, dim=None):
        """
        Compute the Symmetric Extreme Dependency Score (SEDS) for a given threshold.
        
        Args:
            threshold (float): The threshold value for binary classification.
            dim (str, list, or None): The dimension(s) along which to compute the SEDS.
                                      If None, compute the SEDS over the entire data.
        
        Returns:
            xarray.DataArray: The computed SEDS values.
        """
        return _seds(self.obs_data, self.model_data, threshold, dim=dim)

    def compute_fb(self, threshold, dim=None):
        """Compute the Frequency Bias (FB) for a given threshold."""
        return _fb(self.obs_data, self.model_data, threshold, dim=dim)

    def compute_gss(self, threshold, dim=None):
        """
        Compute the Gilbert Skill Score (GSS) for a given threshold.

        Alias of :meth:`compute_ets` (GSS and ETS are the same metric).
        """
        return self.compute_ets(threshold, dim)

    def compute_hkd(self, threshold, dim=None):
        """
        Compute the Hanssen-Kuipers Discriminant (HKD) for a given threshold.

        Alias of :meth:`compute_pss` (HKD and PSS are the same metric).
        """
        return self.compute_pss(threshold, dim)

    def compute_orss(self, threshold, dim=None):
        """
        Compute the Odds Ratio Skill Score (ORSS) for a given threshold.
        
        Args:
            threshold (float): The threshold value for binary classification.
            dim (str, list, or None): The dimension(s) along which to compute the ORSS.
                                      If None, compute the ORSS over the entire data.
        
        Returns:
            xarray.DataArray: The computed ORSS values.
        """
        return _orss(self.obs_data, self.model_data, threshold, dim=dim)

    def compute_eds(self, threshold, dim=None):
        """
        Compute the Extreme Dependency Score (EDS) for a given threshold.
        
        EDS is designed for rare events and measures the association between
        forecasts and observations using the formula:
        EDS = 2 * log(p) / log(q) - 1
        where p = (tp + fn) / n (base rate) and q = tp / (tp + fp + fn + tn) (hit rate)
        
        Args:
            threshold (float): The threshold value for binary classification.
            dim (str, list, or None): The dimension(s) along which to compute the EDS.
                                      If None, compute the EDS over the entire data.
        
        Returns:
            xarray.DataArray: The computed EDS values.
        """
        return _eds(self.obs_data, self.model_data, threshold, dim=dim)

    def compute_sedi(self, threshold, dim=None):
        """
        Compute the Symmetric Extremal Dependence Index (SEDI) for a given threshold.
        
        Args:
            threshold (float): The threshold value for binary classification.
            dim (str, list, or None): The dimension(s) along which to compute the SEDI.
                                      If None, compute the SEDI over the entire data.
        
        Returns:
            xarray.DataArray: The computed SEDI values.
        """
        return _sedi(self.obs_data, self.model_data, threshold, dim=dim)

    def compute_rpss(self, threshold, dim=None):
        """Compute the Ranked Probability Skill Score (RPSS) for a given threshold."""
        return _rpss(self.obs_data, self.model_data, threshold, dim=dim)

    def compute_tse(self, dim=None):
        """Compute the Total Squared Error (TSE)."""
        return _tse(self.obs_data, self.model_data, dim=dim)

    def compute_evs(self, dim=None):
        """
        Compute the Explained Variance Score (EVS).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the EVS.
                                      If None, compute the EVS over the entire data.
        
        Returns:
            xarray.DataArray: The computed EVS values.
        """
        return _evs(self.obs_data, self.model_data, dim=dim)

    def compute_nmse(self, dim=None):
        """
        Compute the Normalized Mean Squared Error (NMSE).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the NMSE.
                                      If None, compute the NMSE over the entire data.
        
        Returns:
            xarray.DataArray: The computed NMSE values.
        """
        return _nmse(self.obs_data, self.model_data, dim=dim)

    def compute_fv(self, dim=None):
        """
        Compute the Fractional Variance (FV).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the FV.
                                      If None, compute the FV over the entire data.
        
        Returns:
            xarray.DataArray: The computed FV values.
        """
        return _fv(self.obs_data, self.model_data, dim=dim)

    def compute_pcc(self, dim=None):
        """Compute the Pearson Correlation Coefficient (PCC)."""
        return _pcc(self.obs_data, self.model_data, dim=dim)

    def compute_sdr(self, dim=None):
        """
        Compute the Standard Deviation Ratio (SDR).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the SDR.
                                      If None, compute the SDR over the entire data.
        
        Returns:
            xarray.DataArray: The computed SDR values.
        """
        return _sdr(self.obs_data, self.model_data, dim=dim)

    def compute_vif(self, dim=None):
        """
        Compute the Variance Inflation Factor (VIF).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the VIF.
                                      If None, compute the VIF over the entire data.
        
        Returns:
            xarray.DataArray: The computed VIF values.
        """
        return _vif(self.obs_data, self.model_data, dim=dim)

    def compute_mad(self, dim=None):
        """
        Compute the Median Absolute Deviation (MAD).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the MAD.
                                      If None, compute the MAD over the entire data.
        
        Returns:
            xarray.DataArray: The computed MAD values.
        """
        return _mad(self.obs_data, self.model_data, dim=dim)

    def compute_iqr(self, dim=None):
        """
        Compute the Interquartile Range (IQR).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the IQR.
                                      If None, compute the IQR over the entire data.
        
        Returns:
            xarray.DataArray: The computed IQR values.
        """
        return _iqr(self.obs_data, self.model_data, dim=dim)

    def compute_r2(self, dim=None):
        """
        Compute the Coefficient of Determination (R^2).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the R^2.
                                      If None, compute the R^2 over the entire data.
        
        Returns:
            xarray.DataArray: The computed R^2 values.
        """
        return _r2(self.obs_data, self.model_data, dim=dim)

    def compute_nae(self, dim=None):
        """
        Compute the Normalized Absolute Error (NAE).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the NAE.
                                      If None, compute the NAE over the entire data.
        
        Returns:
            xarray.DataArray: The computed NAE values.
        """
        return _nae(self.obs_data, self.model_data, dim=dim)

    def compute_rmb(self, dim=None):
        """
        Compute the Relative Mean Bias (RMB).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the RMB.
                                      If None, compute the RMB over the entire data.
        
        Returns:
            xarray.DataArray: The computed RMB values.
        """
        return _rmb(self.obs_data, self.model_data, dim=dim)

    def compute_mape(self, dim=None):
        """
        Compute the Mean Absolute Percentage Error (MAPE).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the MAPE.
                                      If None, compute the MAPE over the entire data.
        
        Returns:
            xarray.DataArray: The computed MAPE values.
        """
        return _mape(self.obs_data, self.model_data, dim=dim)

    def compute_wmae(self, weights, dim=None):
        """Compute the Weighted Mean Absolute Error (WMAE)."""
        return _wmae(self.obs_data, self.model_data, weights, dim=dim)

    def compute_ass(self, reference_error, dim=None):
        """
        Compute the Absolute Skill Score (ASS).
        
        Args:
            reference_error (xarray.DataArray): The reference error values.
            dim (str, list, or None): The dimension(s) along which to compute the ASS.
                                      If None, compute the ASS over the entire data.
        
        Returns:
            xarray.DataArray: The computed ASS values.
        """
        return _ass(self.obs_data, self.model_data, reference_error, dim=dim)

    def compute_rss(self, reference_skill, dim=None):
        """
        Compute the Relative Skill Score (RSS).
        
        Args:
            reference_skill (xarray.DataArray): The reference skill values.
            dim (str, list, or None): The dimension(s) along which to compute the RSS.
                                      If None, compute the RSS over the entire data.
        
        Returns:
            xarray.DataArray: The computed RSS values.
        """
        return _rss(self.obs_data, self.model_data, reference_skill, dim=dim)

    def compute_qss(self, reference_forecast, dim=None):
        """Compute the Quadratic Skill Score (QSS)."""
        return _qss(self.obs_data, self.model_data, reference_forecast, dim=dim)

    def compute_nrmse(self, dim=None):
        """
        Compute the Normalized Root Mean Squared Error (NRMSE).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the NRMSE.
                                      If None, compute the NRMSE over the entire data.
        
        Returns:
            xarray.DataArray: The computed NRMSE values.
        """
        return _nrmse(self.obs_data, self.model_data, dim=dim)

    def compute_lmbe(self, dim=None):
        """
        Compute the Logarithmic Mean Bias Error (LMBE).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the LMBE.
                                      If None, compute the LMBE over the entire data.
        
        Returns:
            xarray.DataArray: The computed LMBE values.
        """
        return _lmbe(self.obs_data, self.model_data, dim=dim)

    def compute_smse(self, dim=None):
        """
        Compute the Scaled Mean Squared Error (SMSE).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the SMSE.
                                      If None, compute the SMSE over the entire data.
        
        Returns:
            xarray.DataArray: The computed SMSE values.
        """
        return _smse(self.obs_data, self.model_data, dim=dim)

    def compute_mbd(self, dim=None):
        """Compute the Mean Bias Deviation (MBD)."""
        return _mbd(self.obs_data, self.model_data, dim=dim)

    def compute_gmb(self, dim=None):
        """
        Compute the Geometric Mean Bias (GMB).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the GMB.
                                      If None, compute the GMB over the entire data.
        
        Returns:
            xarray.DataArray: The computed GMB values.
        """
        return _gmb(self.obs_data, self.model_data, dim=dim)

    def compute_sbs(self, dim=None):
        """
        Compute the Symmetric Brier Score (SBS).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the SBS.
                                      If None, compute the SBS over the entire data.
        
        Returns:
            xarray.DataArray: The computed SBS values.
        """
        return _sbs(self.obs_data, self.model_data, dim=dim)

    def compute_aev(self, dim=None):
        """
        Compute the Adjusted Explained Variance (AEV).
        
        This is similar to EVS but adjusts for the degrees of freedom.
        AEV = 1 - (error variance / observation variance)
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the AEV.
                                      If None, compute the AEV over the entire data.
        
        Returns:
            xarray.DataArray: The computed AEV values.
        """
        return _aev(self.obs_data, self.model_data, dim=dim)

    def compute_cosine_similarity(self, dim=None):
        """
        Compute the Cosine Similarity.
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the Cosine Similarity.
                                      If None, compute the Cosine Similarity over the entire data.
        
        Returns:
            xarray.DataArray: The computed Cosine Similarity values.
        """
        return _cosine_similarity(self.obs_data, self.model_data, dim=dim)

    def compute_f1(self, threshold, dim=None):
        """
        Compute the F1 Score for a given threshold.
        
        Args:
            threshold (float): The threshold value for binary classification.
            dim (str, list, or None): The dimension(s) along which to compute the F1 Score.
                                      If None, compute the F1 Score over the entire data.
        
        Returns:
            xarray.DataArray: The computed F1 Score values.
        """
        return _f1(self.obs_data, self.model_data, threshold, dim=dim)

    def compute_mcc(self, threshold, dim=None):
        """Compute the Matthews Correlation Coefficient (MCC) for a given threshold."""
        return _mcc(self.obs_data, self.model_data, threshold, dim=dim)

    def compute_ba(self, threshold, dim=None):
        """
        Compute the Balanced Accuracy (BA) for a given threshold.
        
        Args:
            threshold (float): The threshold value for binary classification.
            dim (str, list, or None): The dimension(s) along which to compute the BA.
                                      If None, compute the BA over the entire data.
        
        Returns:
            xarray.DataArray: The computed BA values.
        """
        return _ba(self.obs_data, self.model_data, threshold, dim=dim)

    def compute_npv(self, threshold, dim=None):
        """
        Compute the Negative Predictive Value (NPV) for a given threshold.
        
        Args:
            threshold (float): The threshold value for binary classification.
            dim (str, list, or None): The dimension(s) along which to compute the NPV.
                                      If None, compute the NPV over the entire data.
        
        Returns:
            xarray.DataArray: The computed NPV values.
        """
        return _npv(self.obs_data, self.model_data, threshold, dim=dim)

    def compute_jaccard(self, threshold, dim=None):
        """
        Compute the Jaccard Similarity Coefficient for a given threshold.

        Alias of :meth:`compute_csi` (Jaccard and CSI are the same metric).
        """
        return self.compute_csi(threshold, dim)

    def compute_gain(self, threshold, dim=None):
        """Compute the Gain for a given threshold."""
        return _gain(self.obs_data, self.model_data, threshold, dim=dim)

    def compute_lift(self, threshold, dim=None):
        """
        Compute the Lift for a given threshold.
        
        Args:
            threshold (float): The threshold value for binary classification.
            dim (str, list, or None): The dimension(s) along which to compute the Lift.
                                      If None, compute the Lift over the entire data.
        
        Returns:
            xarray.DataArray: The computed Lift values.
        """
        return _lift(self.obs_data, self.model_data, threshold, dim=dim)

    def compute_mkldiv(self, dim=None):
        """Compute the Mean Kullback-Leibler Divergence (MKLDIV)."""
        return _mkldiv(self.obs_data, self.model_data, dim=dim)

    def compute_jsdiv(self, dim=None):
        """Compute the Jensen-Shannon Divergence (JSDIV)."""
        return _jsdiv(self.obs_data, self.model_data, dim=dim)

    def compute_hellinger(self, dim=None):
        """Compute the Hellinger Distance."""
        return _hellinger(self.obs_data, self.model_data, dim=dim)

    def compute_wasserstein(self, dim=None):
        """
        Compute the Wasserstein Distance.
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the Wasserstein Distance.
                                      If None, compute the Wasserstein Distance over the entire data.
        
        Returns:
            xarray.DataArray: The computed Wasserstein Distance values.
        """
        return _wasserstein(self.obs_data, self.model_data, dim=dim)

    def compute_tv(self, dim=None):
        """Compute the Total Variation Distance."""
        return _tv(self.obs_data, self.model_data, dim=dim)

    def compute_chisquare(self, dim=None):
        """Compute the Chi-Square Distance."""
        return _chisquare(self.obs_data, self.model_data, dim=dim)

    def compute_intersection(self, dim=None):
        """Compute the histogram-intersection similarity."""
        return _intersection(self.obs_data, self.model_data, dim=dim)

    def compute_bhattacharyya(self, dim=None):
        """
        Compute the Bhattacharyya Distance.
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the Bhattacharyya Distance.
                                      If None, compute the Bhattacharyya Distance over the entire data.
        
        Returns:
            xarray.DataArray: The computed Bhattacharyya Distance values.
        """
        return _bhattacharyya(self.obs_data, self.model_data, dim=dim)

    def compute_harmonic_mean(self, dim=None):
        """Compute the element-wise Harmonic Mean between obs and model data."""
        return _harmonic_mean(self.obs_data, self.model_data, dim=dim)

    def compute_geometric_mean(self, dim=None):
        """Compute the element-wise Geometric Mean between obs and model data."""
        return _geometric_mean(self.obs_data, self.model_data, dim=dim)

    def compute_lehmer_mean(self, p, dim=None):
        """
        Compute the element-wise Lehmer Mean of order ``p`` between obs and model.

        Note: This is an element-wise combination, not an aggregation over dim.

        Args:
            p (float): The power parameter for the Lehmer Mean.
            dim: Unused, kept for API consistency.

        Returns:
            xarray.DataArray: Element-wise Lehmer mean of obs and model.
        """
        return _lehmer_mean(self.obs_data, self.model_data, p, dim=dim)

    def compute_chernoff(self, alpha, dim=None):
        """
        Compute the Chernoff Distance.
        
        Args:
            alpha (float): The parameter for the Chernoff Distance (0 < alpha < 1).
            dim (str, list, or None): The dimension(s) along which to compute the Chernoff Distance.
                                      If None, compute the Chernoff Distance over the entire data.
        
        Returns:
            xarray.DataArray: The computed Chernoff Distance values.
        """
        return _chernoff(self.obs_data, self.model_data, alpha, dim=dim)

    def compute_renyi(self, alpha, dim=None):
        """
        Compute the Rényi Divergence.
        
        Args:
            alpha (float): The parameter for the Rényi Divergence (alpha != 1).
            dim (str, list, or None): The dimension(s) along which to compute the Rényi Divergence.
                                      If None, compute the Rényi Divergence over the entire data.
        
        Returns:
            xarray.DataArray: The computed Rényi Divergence values.
        """
        return _renyi(self.obs_data, self.model_data, alpha, dim=dim)

    def compute_tsallis(self, alpha, dim=None):
        """
        Compute the Tsallis Divergence.
        
        Args:
            alpha (float): The parameter for the Tsallis Divergence (alpha != 1).
            dim (str, list, or None): The dimension(s) along which to compute the Tsallis Divergence.
                                      If None, compute the Tsallis Divergence over the entire data.
        
        Returns:
            xarray.DataArray: The computed Tsallis Divergence values.
        """
        return _tsallis(self.obs_data, self.model_data, alpha, dim=dim)
    
    
    #Further addition of metrics to the code. 
    

