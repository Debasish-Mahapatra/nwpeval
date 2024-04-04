import numpy as np
import xarray as xr
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


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
    def __init__(self, obs_data, model_data):
        """
        Initialize the NWPMetrics object with observed and modeled data.
        
        Args:
            obs_data (xarray.DataArray): The observed data.
            model_data (xarray.DataArray): The modeled data.
        """
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
                metric_values[metric] = self.compute_acc(dim)
            elif metric == 'FSS':
                threshold = thresholds.get('FSS', 0.5)
                neighborhood_size = thresholds.get('FSS_neighborhood', 3)
                metric_values[metric] = self.compute_fss(threshold, neighborhood_size, dim)
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
            elif metric == 'GS':
                threshold = thresholds.get('GS', 0.5)
                metric_values[metric] = self.compute_gs(threshold, dim)
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
    
        Args:
            obs_binary (xarray.DataArray): The binarized observed data.
            model_binary (xarray.DataArray): The binarized modeled data.
            dim (str, list, or None): The dimension(s) along which to compute the confusion matrix.
                                  If None, compute the confusion matrix over the entire data.
    
        Returns:
            tuple: A tuple containing the confusion matrix values (tn, fp, fn, tp).
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
    
    def compute_mae(self, dim=None):
        """Calculate the Mean Absolute Error (MAE)."""
        return np.abs(self.obs_data - self.model_data).mean(dim=dim)

    def compute_rmse(self, dim=None):
        """Calculate the Root Mean Square Error (RMSE)."""
        return np.sqrt(((self.obs_data - self.model_data) ** 2).mean(dim=dim))

    def compute_acc(self, dim=None):
        """Calculate the Anomaly Correlation Coefficient (ACC)."""
        return xr.corr(self.obs_data, self.model_data, dim=dim)

    def compute_fss(self, threshold, neighborhood_size, dim=None):
        """
        Compute the Fractions Skill Score (FSS) for a given threshold and neighborhood size.
        
        Args:
            threshold (float): The threshold value for binary classification.
            neighborhood_size (int): The size of the neighborhood window.
            dim (str, list, or None): The dimension(s) along which to compute the FSS.
                                      If None, compute the FSS over the entire data.
        
        Returns:
            xarray.DataArray: The computed FSS values.
        """
        # Convert data to binary based on the threshold
        obs_binary = (self.obs_data >= threshold).astype(int)
        model_binary = (self.model_data >= threshold).astype(int)
        
        # Compute the fractions within each neighborhood
        obs_fractions = obs_binary.rolling({dim: neighborhood_size}, center=True).mean()
        model_fractions = model_binary.rolling({dim: neighborhood_size}, center=True).mean()
        
        # Calculate the mean squared error (MSE) of the fractions
        mse = ((obs_fractions - model_fractions) ** 2).mean(dim=dim)
        
        # Calculate the reference MSE
        obs_fraction_mean = obs_fractions.mean(dim=dim)
        model_fraction_mean = model_fractions.mean(dim=dim)
        mse_ref = obs_fraction_mean * (1 - obs_fraction_mean) + model_fraction_mean * (1 - model_fraction_mean)
        
        # Calculate the FSS
        fss = 1 - mse / mse_ref
        
        return fss

    def compute_ets(self, threshold, dim=None):
        """
        Compute the Equitable Threat Score (ETS) for a given threshold.
        
        Args:
            threshold (float): The threshold value for binary classification.
            dim (str, list, or None): The dimension(s) along which to compute the ETS.
                                      If None, compute the ETS over the entire data.
        
        Returns:
            xarray.DataArray: The computed ETS values.
        """
        # Convert data to binary based on the threshold
        obs_binary = (self.obs_data >= threshold).astype(int)
        model_binary = (self.model_data >= threshold).astype(int)
        
        # Calculate the confusion matrix
        tn, fp, fn, tp = self.confusion_matrix(obs_binary, model_binary, dim)
        
        # Calculate the ETS
        hits_random = (tp + fp) * (tp + fn) / (tp + fp + fn + tn)
        ets = (tp - hits_random) / (tp + fp + fn - hits_random)
        
        return ets

    def compute_pod(self, threshold, dim=None):
        """
        Compute the Probability of Detection (POD) for a given threshold.
        
        Args:
            threshold (float): The threshold value for binary classification.
            dim (str, list, or None): The dimension(s) along which to compute the POD.
                                      If None, compute the POD over the entire data.
        
        Returns:
            xarray.DataArray: The computed POD values.
        """
        # Convert data to binary based on the threshold
        obs_binary = (self.obs_data >= threshold).astype(int)
        model_binary = (self.model_data >= threshold).astype(int)
        
        # Calculate the confusion matrix
        tn, fp, fn, tp = self.confusion_matrix(obs_binary, model_binary, dim)
        
        # Calculate the POD
        pod = tp / (tp + fn)
        
        return pod

    def compute_far(self, threshold, dim=None):
        """
        Compute the False Alarm Ratio (FAR) for a given threshold.
        
        Args:
            threshold (float): The threshold value for binary classification.
            dim (str, list, or None): The dimension(s) along which to compute the FAR.
                                      If None, compute the FAR over the entire data.
        
        Returns:
            xarray.DataArray: The computed FAR values.
        """
        # Convert data to binary based on the threshold
        obs_binary = (self.obs_data >= threshold).astype(int)
        model_binary = (self.model_data >= threshold).astype(int)
        
        # Calculate the confusion matrix
        tn, fp, fn, tp = self.confusion_matrix(obs_binary, model_binary, dim)
        
        # Calculate the FAR
        far = fp / (tp + fp)
        
        return far

    def compute_csi(self, threshold, dim=None):
        """
        Compute the Critical Success Index (CSI) for a given threshold.
        
        Args:
            threshold (float): The threshold value for binary classification.
            dim (str, list, or None): The dimension(s) along which to compute the CSI.
                                      If None, compute the CSI over the entire data.
        
        Returns:
            xarray.DataArray: The computed CSI values.
        """
        # Convert data to binary based on the threshold
        obs_binary = (self.obs_data >= threshold).astype(int)
        model_binary = (self.model_data >= threshold).astype(int)
        
        # Calculate the confusion matrix
        tn, fp, fn, tp = self.confusion_matrix(obs_binary, model_binary, dim)
        
        # Calculate the CSI
        csi = tp / (tp + fp + fn)
        
        return csi

    def compute_bss(self, threshold, dim=None):
        """
        Compute the Brier Skill Score (BSS) for a given threshold.
        
        Args:
            threshold (float): The threshold value for binary classification.
            dim (str, list, or None): The dimension(s) along which to compute the BSS.
                                      If None, compute the BSS over the entire data.
        
        Returns:
            xarray.DataArray: The computed BSS values.
        """
        # Convert data to binary based on the threshold
        obs_binary = (self.obs_data >= threshold).astype(int)
        
        # Calculate the Brier score for the model data
        bs_model = ((self.model_data - obs_binary) ** 2).mean(dim=dim)
        
        # Calculate the Brier score for the climatology (base rate)
        base_rate = obs_binary.mean(dim=dim)
        bs_climo = base_rate * (1 - base_rate)
        
        # Calculate the BSS
        bss = 1 - bs_model / bs_climo
        
        return bss

    def compute_hss(self, threshold, dim=None):
        """
        Compute the Heidke Skill Score (HSS) for a given threshold.
        
        Args:
            threshold (float): The threshold value for binary classification.
            dim (str, list, or None): The dimension(s) along which to compute the HSS.
                                      If None, compute the HSS over the entire data.
        
        Returns:
            xarray.DataArray: The computed HSS values.
        """
        # Convert data to binary based on the threshold
        obs_binary = (self.obs_data >= threshold).astype(int)
        model_binary = (self.model_data >= threshold).astype(int)
        
        # Calculate the confusion matrix
        tn, fp, fn, tp = self.confusion_matrix(obs_binary, model_binary, dim)
        
        # Calculate the HSS
        hss = 2 * (tp * tn - fp * fn) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
        
        return hss

    def compute_pss(self, threshold, dim=None):
        """
        Compute the Peirce Skill Score (PSS) for a given threshold.
        
        Args:
            threshold (float): The threshold value for binary classification.
            dim (str, list, or None): The dimension(s) along which to compute the PSS.
                                      If None, compute the PSS over the entire data.
        
        Returns:
            xarray.DataArray: The computed PSS values.
        """
        # Convert data to binary based on the threshold
        obs_binary = (self.obs_data >= threshold).astype(int)
        model_binary = (self.model_data >= threshold).astype(int)
        
        # Calculate the confusion matrix
        tn, fp, fn, tp = self.confusion_matrix(obs_binary, model_binary, dim)
        
        # Calculate the PSS
        pss = (tp / (tp + fn)) - (fp / (fp + tn))
        
        return pss

    def compute_gs(self, threshold, dim=None):
        """
        Compute the Gilbert Skill Score (GS) for a given threshold.
        
        Args:
            threshold (float): The threshold value for binary classification.
            dim (str, list, or None): The dimension(s) along which to compute the GS.
                                      If None, compute the GS over the entire data.
        
        Returns:
            xarray.DataArray: The computed GS values.
        """
        # Convert data to binary based on the threshold
        obs_binary = (self.obs_data >= threshold).astype(int)
        model_binary = (self.model_data >= threshold).astype(int)
        
        # Calculate the confusion matrix
        tn, fp, fn, tp = self.confusion_matrix(obs_binary, model_binary, dim)
        
        # Calculate the GS
        gs = (tp - ((tp + fp) * (tp + fn) / (tp + fp + fn + tn))) / (tp + fp + fn - ((tp + fp) * (tp + fn) / (tp + fp + fn + tn)))
        
        return gs

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
        # Convert data to binary based on the threshold
        obs_binary = (self.obs_data >= threshold).astype(int)
        model_binary = (self.model_data >= threshold).astype(int)
        
        # Calculate the confusion matrix
        tn, fp, fn, tp = self.confusion_matrix(obs_binary, model_binary, dim)
        
        # Calculate the POD and POFD
        pod = tp / (tp + fn)
        pofd = fp / (fp + tn)
        
        # Calculate the SEDS
        seds = (np.log(pod) - np.log(pofd) + np.log(1 - pofd) - np.log(1 - pod)) / (np.log(pod) + np.log(1 - pofd))
        
        return seds

    def compute_fb(self, threshold, dim=None):
        """
        Compute the Frequency Bias (FB) for a given threshold.
        
        Args:
            threshold (float): The threshold value for binary classification.
            dim (str, list, or None): The dimension(s) along which to compute the FB.
                                      If None, compute the FB over the entire data.
        
        Returns:
            xarray.DataArray: The computed FB values.
        """
        # Convert data to binary based on the threshold
        obs_binary = (self.obs_data >= threshold).astype(int)
        model_binary = (self.model_data >= threshold).astype(int)
        
        # Calculate the confusion matrix
        tn, fp, fn, tp = self.confusion_matrix(obs_binary, model_binary, dim)
        
        # Calculate the FB
        fb = (tp + fp) / (tp + fn)
        
        return fb

    def compute_gss(self, threshold, dim=None):
        """
        Compute the Gilbert Skill Score (GSS) for a given threshold.
        
        Args:
            threshold (float): The threshold value for binary classification.
            dim (str, list, or None): The dimension(s) along which to compute the GSS.
                                      If None, compute the GSS over the entire data.
        
        Returns:
            xarray.DataArray: The computed GSS values.
        """
        # Convert data to binary based on the threshold
        obs_binary = (self.obs_data >= threshold).astype(int)
        model_binary = (self.model_data >= threshold).astype(int)
        
        # Calculate the confusion matrix
        tn, fp, fn, tp = self.confusion_matrix(obs_binary, model_binary, dim)
        
        # Calculate the GSS
        gss = (tp - ((tp + fp) * (tp + fn) / (tp + fp + fn + tn))) / (tp + fp + fn - ((tp + fp) * (tp + fn) / (tp + fp + fn + tn)))
        
        return gss

    def compute_hkd(self, threshold, dim=None):
        """
        Compute the Hanssen-Kuipers Discriminant (H-KD) for a given threshold.
        
        Args:
            threshold (float): The threshold value for binary classification.
            dim (str, list, or None): The dimension(s) along which to compute the H-KD.
                                      If None, compute the H-KD over the entire data.
        
        Returns:
            xarray.DataArray: The computed H-KD values.
        """
        # Convert data to binary based on the threshold
        obs_binary = (self.obs_data >= threshold).astype(int)
        model_binary = (self.model_data >= threshold).astype(int)
        
        # Calculate the confusion matrix
        tn, fp, fn, tp = self.confusion_matrix(obs_binary, model_binary, dim)
        
        # Calculate the H-KD
        hkd = (tp / (tp + fn)) - (fp / (fp + tn))
        
        return hkd

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
        # Convert data to binary based on the threshold
        obs_binary = (self.obs_data >= threshold).astype(int)
        model_binary = (self.model_data >= threshold).astype(int)
        
        # Calculate the confusion matrix
        tn, fp, fn, tp = self.confusion_matrix(obs_binary, model_binary, dim)
        
        # Calculate the ORSS
        odds_ratio = (tp * tn) / (fp * fn)
        orss = (odds_ratio - 1) / (odds_ratio + 1)
        
        return orss

    def compute_eds(self, threshold, dim=None):
        """
        Compute the Extreme Dependency Score (EDS) for a given threshold.
        
        Args:
            threshold (float): The threshold value for binary classification.
            dim (str, list, or None): The dimension(s) along which to compute the EDS.
                                      If None, compute the EDS over the entire data.
        
        Returns:
            xarray.DataArray: The computed EDS values.
        """
        # Convert data to binary based on the threshold
        obs_binary = (self.obs_data >= threshold).astype(int)
        model_binary = (self.model_data >= threshold).astype(int)
        
        # Calculate the confusion matrix
        tn, fp, fn, tp = self.confusion_matrix(obs_binary, model_binary, dim)
        
        # Calculate the EDS
        eds = (tp / (tp + fn)) * (tp / (tp + fp))
        
        return eds

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
        # Convert data to binary based on the threshold
        obs_binary = (self.obs_data >= threshold).astype(int)
        model_binary = (self.model_data >= threshold).astype(int)
        
        # Calculate the confusion matrix
        tn, fp, fn, tp = self.confusion_matrix(obs_binary, model_binary, dim)
        
        # Calculate the POD and POFD
        pod = tp / (tp + fn)
        pofd = fp / (fp + tn)
        
        # Calculate the SEDI
        sedi = (np.log(pod) - np.log(pofd) + np.log(1 - pofd) - np.log(1 - pod)) / (np.log(pod) + np.log(1 - pofd) + np.log(1 - pod) + np.log(pofd))
        
        return sedi

    def compute_rpss(self, threshold, dim=None):
        """
        Compute the Ranked Probability Skill Score (RPSS) for a given threshold.
    
        Args:
            threshold (float): The threshold value for binary classification.
            dim (str, list, or None): The dimension(s) along which to compute the RPSS.
                                  If None, compute the RPSS over the entire data.
    
        Returns:
            xarray.DataArray: The computed RPSS values.
        """
        # Convert data to binary based on the threshold
        obs_binary = (self.obs_data >= threshold).astype(int)
        model_binary = (self.model_data >= threshold).astype(int)
    
        # Calculate the RPS for the model data
        rps_model = ((model_binary.cumsum(dim) - obs_binary.cumsum(dim)) ** 2).mean(dim=dim)
    
        # Calculate the RPS for the climatology (base rate)
        base_rate = obs_binary.mean(dim=dim)
        rps_climo = ((xr.full_like(model_binary, 0).cumsum(dim) - obs_binary.cumsum(dim)) ** 2).mean(dim=dim)
        rps_climo = rps_climo + base_rate * (1 - base_rate)
    
        # Calculate the RPSS
        rpss = 1 - rps_model / rps_climo
    
        return rpss

    def compute_tse(self, dim=None):
        """
        Compute the Total Squared Error (TSE).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the TSE.
                                      If None, compute the TSE over the entire data.
        
        Returns:
            xarray.DataArray: The computed TSE values.
        """
        return ((self.model_data - self.obs_data) ** 2).sum(dim=dim)

    def compute_evs(self, dim=None):
        """
        Compute the Explained Variance Score (EVS).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the EVS.
                                      If None, compute the EVS over the entire data.
        
        Returns:
            xarray.DataArray: The computed EVS values.
        """
        obs_var = self.obs_data.var(dim=dim)
        err_var = (self.obs_data - self.model_data).var(dim=dim)
        return 1 - err_var / obs_var

    def compute_nmse(self, dim=None):
        """
        Compute the Normalized Mean Squared Error (NMSE).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the NMSE.
                                      If None, compute the NMSE over the entire data.
        
        Returns:
            xarray.DataArray: The computed NMSE values.
        """
        mse = ((self.model_data - self.obs_data) ** 2).mean(dim=dim)
        obs_mean = self.obs_data.mean(dim=dim)
        return mse / (obs_mean ** 2)

    def compute_fv(self, dim=None):
        """
        Compute the Fractional Variance (FV).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the FV.
                                      If None, compute the FV over the entire data.
        
        Returns:
            xarray.DataArray: The computed FV values.
        """
        obs_var = self.obs_data.var(dim=dim)
        model_var = self.model_data.var(dim=dim)
        return model_var / obs_var

    def compute_pcc(self, dim=None):
        """
        Compute the Pearson Correlation Coefficient (PCC).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the PCC.
                                      If None, compute the PCC over the entire data.
        
        Returns:
            xarray.DataArray: The computed PCC values.
        """
        return xr.corr(self.model_data, self.obs_data, dim=dim)

    def compute_sdr(self, dim=None):
        """
        Compute the Standard Deviation Ratio (SDR).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the SDR.
                                      If None, compute the SDR over the entire data.
        
        Returns:
            xarray.DataArray: The computed SDR values.
        """
        obs_std = self.obs_data.std(dim=dim)
        model_std = self.model_data.std(dim=dim)
        return model_std / obs_std

    def compute_vif(self, dim=None):
        """
        Compute the Variance Inflation Factor (VIF).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the VIF.
                                      If None, compute the VIF over the entire data.
        
        Returns:
            xarray.DataArray: The computed VIF values.
        """
        obs_var = self.obs_data.var(dim=dim)
        model_var = self.model_data.var(dim=dim)
        return model_var / obs_var - 1

    def compute_mad(self, dim=None):
        """
        Compute the Median Absolute Deviation (MAD).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the MAD.
                                      If None, compute the MAD over the entire data.
        
        Returns:
            xarray.DataArray: The computed MAD values.
        """
        return (np.abs(self.model_data - self.model_data.median(dim=dim))).median(dim=dim)

    def compute_iqr(self, dim=None):
        """
        Compute the Interquartile Range (IQR).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the IQR.
                                      If None, compute the IQR over the entire data.
        
        Returns:
            xarray.DataArray: The computed IQR values.
        """
        q1 = self.model_data.quantile(0.25, dim=dim)
        q3 = self.model_data.quantile(0.75, dim=dim)
        return q3 - q1

    def compute_r2(self, dim=None):
        """
        Compute the Coefficient of Determination (R^2).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the R^2.
                                      If None, compute the R^2 over the entire data.
        
        Returns:
            xarray.DataArray: The computed R^2 values.
        """
        ssr = ((self.model_data - self.obs_data) ** 2).sum(dim=dim)
        sst = ((self.obs_data - self.obs_data.mean(dim=dim)) ** 2).sum(dim=dim)
        return 1 - ssr / sst

    def compute_nae(self, dim=None):
        """
        Compute the Normalized Absolute Error (NAE).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the NAE.
                                      If None, compute the NAE over the entire data.
        
        Returns:
            xarray.DataArray: The computed NAE values.
        """
        abs_error = np.abs(self.model_data - self.obs_data).sum(dim=dim)
        abs_obs = np.abs(self.obs_data).sum(dim=dim)
        return abs_error / abs_obs

    def compute_rmb(self, dim=None):
        """
        Compute the Relative Mean Bias (RMB).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the RMB.
                                      If None, compute the RMB over the entire data.
        
        Returns:
            xarray.DataArray: The computed RMB values.
        """
        bias = (self.model_data - self.obs_data).sum(dim=dim)
        obs_sum = self.obs_data.sum(dim=dim)
        return bias / obs_sum

    def compute_mape(self, dim=None):
        """
        Compute the Mean Absolute Percentage Error (MAPE).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the MAPE.
                                      If None, compute the MAPE over the entire data.
        
        Returns:
            xarray.DataArray: The computed MAPE values.
        """
        abs_percent_error = np.abs((self.model_data - self.obs_data) / self.obs_data)
        return 100 * abs_percent_error.mean(dim=dim)

    def compute_wmae(self, weights, dim=None):
        """
        Compute the Weighted Mean Absolute Error (WMAE).
        
        Args:
            weights (xarray.DataArray): The weights for each data point.
            dim (str, list, or None): The dimension(s) along which to compute the WMAE.
                                      If None, compute the WMAE over the entire data.
        
        Returns:
            xarray.DataArray: The computed WMAE values.
        """
        weighted_abs_error = weights * np.abs(self.model_data - self.obs_data)
        return weighted_abs_error.sum(dim=dim) / weights.sum(dim=dim)

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
        abs_error = np.abs(self.model_data - self.obs_data)
        return 1 - abs_error / reference_error

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
        model_skill = 1 - np.abs(self.model_data - self.obs_data) / self.obs_data
        return (model_skill - reference_skill) / (1 - reference_skill)

    def compute_qss(self, reference_forecast, dim=None):
        """
        Compute the Quadratic Skill Score (QSS).
        
        Args:
            reference_forecast (xarray.DataArray): The reference forecast values.
            dim (str, list, or None): The dimension(s) along which to compute the QSS.
                                      If None, compute the QSS over the entire data.
        
        Returns:
            xarray.DataArray: The computed QSS values.
        """
        mse_model = ((self.model_data - self.obs_data) ** 2).mean(dim=dim)
        mse_ref = ((reference_forecast - self.obs_data) ** 2).mean(dim=dim)
        return 1 - mse_model / mse_ref

    def compute_nrmse(self, dim=None):
        """
        Compute the Normalized Root Mean Squared Error (NRMSE).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the NRMSE.
                                      If None, compute the NRMSE over the entire data.
        
        Returns:
            xarray.DataArray: The computed NRMSE values.
        """
        rmse = np.sqrt(((self.model_data - self.obs_data) ** 2).mean(dim=dim))
        obs_mean = self.obs_data.mean(dim=dim)
        return rmse / obs_mean

    def compute_lmbe(self, dim=None):
        """
        Compute the Logarithmic Mean Bias Error (LMBE).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the LMBE.
                                      If None, compute the LMBE over the entire data.
        
        Returns:
            xarray.DataArray: The computed LMBE values.
        """
        return (np.log(self.model_data + 1) - np.log(self.obs_data + 1)).mean(dim=dim)

    def compute_smse(self, dim=None):
        """
        Compute the Scaled Mean Squared Error (SMSE).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the SMSE.
                                      If None, compute the SMSE over the entire data.
        
        Returns:
            xarray.DataArray: The computed SMSE values.
        """
        mse = ((self.model_data - self.obs_data) ** 2).mean(dim=dim)
        obs_var = self.obs_data.var(dim=dim)
        return mse / obs_var

    def compute_mbd(self, dim=None):
        """
        Compute the Mean Bias Deviation (MBD).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the MBD.
                                      If None, compute the MBD over the entire data.
        
        Returns:
            xarray.DataArray: The computed MBD values.
        """
        return self.model_data.mean(dim=dim) - self.obs_data.mean(dim=dim)

    def compute_gmb(self, dim=None):
        """
        Compute the Geometric Mean Bias (GMB).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the GMB.
                                      If None, compute the GMB over the entire data.
        
        Returns:
            xarray.DataArray: The computed GMB values.
        """
        model_mean = np.exp(np.log(self.model_data).mean(dim=dim))
        obs_mean = np.exp(np.log(self.obs_data).mean(dim=dim))
        return model_mean / obs_mean

    def compute_sbs(self, dim=None):
        """
        Compute the Symmetric Brier Score (SBS).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the SBS.
                                      If None, compute the SBS over the entire data.
        
        Returns:
            xarray.DataArray: The computed SBS values.
        """
        return ((self.model_data - self.obs_data) ** 2).mean(dim=dim)

    def compute_aev(self, dim=None):
        """
        Compute the Adjusted Explained Variance (AEV).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the AEV.
                                      If None, compute the AEV over the entire data.
        
        Returns:
            xarray.DataArray: The computed AEV values.
        """
        obs_var = self.obs_data.var(dim=dim)
        err_var = (self.obs_data - self.model_data).var(dim=dim)
        return 1 - (err_var - obs_var) / obs_var

    def compute_cosine_similarity(self, dim=None):
        """
        Compute the Cosine Similarity.
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the Cosine Similarity.
                                      If None, compute the Cosine Similarity over the entire data.
        
        Returns:
            xarray.DataArray: The computed Cosine Similarity values.
        """
        dot_product = (self.model_data * self.obs_data).sum(dim=dim)
        model_norm = np.sqrt((self.model_data ** 2).sum(dim=dim))
        obs_norm = np.sqrt((self.obs_data ** 2).sum(dim=dim))
        return dot_product / (model_norm * obs_norm)

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
        obs_binary = (self.obs_data >= threshold).astype(int)
        model_binary = (self.model_data >= threshold).astype(int)
        tp = ((obs_binary == 1) & (model_binary == 1)).sum(dim=dim)
        fp = ((obs_binary == 0) & (model_binary == 1)).sum(dim=dim)
        fn = ((obs_binary == 1) & (model_binary == 0)).sum(dim=dim)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return 2 * (precision * recall) / (precision + recall)

    def compute_mcc(self, threshold, dim=None):
        """
        Compute the Matthews Correlation Coefficient (MCC) for a given threshold.
        
        Args:
            threshold (float): The threshold value for binary classification.
            dim (str, list, or None): The dimension(s) along which to compute the MCC.
                                      If None, compute the MCC over the entire data.
        
        Returns:
            xarray.DataArray: The computed MCC values.
        """
        obs_binary = (self.obs_data >= threshold).astype(int)
        model_binary = (self.model_data >= threshold).astype(int)
        tp = ((obs_binary == 1) & (model_binary == 1)).sum(dim=dim)
        tn = ((obs_binary == 0) & (model_binary == 0)).sum(dim=dim)
        fp = ((obs_binary == 0) & (model_binary == 1)).sum(dim=dim)
        fn = ((obs_binary == 1) & (model_binary == 0)).sum(dim=dim)
        mcc_numerator = (tp * tn) - (fp * fn)
        mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return mcc_numerator / mcc_denominator

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
        obs_binary = (self.obs_data >= threshold).astype(int)
        model_binary = (self.model_data >= threshold).astype(int)
        tp = ((obs_binary == 1) & (model_binary == 1)).sum(dim=dim)
        tn = ((obs_binary == 0) & (model_binary == 0)).sum(dim=dim)
        fn = ((obs_binary == 1) & (model_binary == 0)).sum(dim=dim)
        fp = ((obs_binary == 0) & (model_binary == 1)).sum(dim=dim)
        return 0.5 * ((tp / (tp + fn)) + (tn / (tn + fp)))

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
        obs_binary = (self.obs_data >= threshold).astype(int)
        model_binary = (self.model_data >= threshold).astype(int)
        tn = ((obs_binary == 0) & (model_binary == 0)).sum(dim=dim)
        fn = ((obs_binary == 1) & (model_binary == 0)).sum(dim=dim)
        return tn / (tn + fn)

    def compute_jaccard(self, threshold, dim=None):
        """
        Compute the Jaccard Similarity Coefficient for a given threshold.
        
        Args:
            threshold (float): The threshold value for binary classification.
            dim (str, list, or None): The dimension(s) along which to compute the Jaccard Similarity Coefficient.
                                      If None, compute the Jaccard Similarity Coefficient over the entire data.
        
        Returns:
            xarray.DataArray: The computed Jaccard Similarity Coefficient values.
        """
        obs_binary = (self.obs_data >= threshold).astype(int)
        model_binary = (self.model_data >= threshold).astype(int)
        intersection = (obs_binary & model_binary).sum(dim=dim)
        union = (obs_binary | model_binary).sum(dim=dim)
        return intersection / union

    def compute_gain(self, threshold, dim=None):
        """
        Compute the Gain for a given threshold.
        
        Args:
            threshold (float): The threshold value for binary classification.
            dim (str, list, or None): The dimension(s) along which to compute the Gain.
                                      If None, compute the Gain over the entire data.
        
        Returns:
            xarray.DataArray: The computed Gain values.
        """
        obs_binary = (self.obs_data >= threshold).astype(int)
        model_binary = (self.model_data >= threshold).astype(int)
        tp = ((obs_binary == 1) & (model_binary == 1)).sum(dim=dim)
        fp = ((obs_binary == 0) & (model_binary == 1)).sum(dim=dim)
        tn = ((obs_binary == 0) & (model_binary == 0)).sum(dim=dim)
        fn = ((obs_binary == 1) & (model_binary == 0)).sum(dim=dim)
        return (tp + tn) / (tp + fp + tn + fn)

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
        obs_binary = (self.obs_data >= threshold).astype(int)
        model_binary = (self.model_data >= threshold).astype(int)
        tp = ((obs_binary == 1) & (model_binary == 1)).sum(dim=dim)
        fp = ((obs_binary == 0) & (model_binary == 1)).sum(dim=dim)
        fn = ((obs_binary == 1) & (model_binary == 0)).sum(dim=dim)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return precision / recall

    def compute_mkldiv(self, dim=None):
        """
        Compute the Mean Kullback-Leibler Divergence (MKLDIV).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the MKLDIV.
                                      If None, compute the MKLDIV over the entire data.
        
        Returns:
            xarray.DataArray: The computed MKLDIV values.
        """
        obs_prob = self.obs_data / self.obs_data.sum(dim=dim)
        model_prob = self.model_data / self.model_data.sum(dim=dim)
        return (obs_prob * np.log(obs_prob / model_prob)).sum(dim=dim)

    def compute_jsdiv(self, dim=None):
        """
        Compute the Jensen-Shannon Divergence (JSDIV).
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the JSDIV.
                                      If None, compute the JSDIV over the entire data.
        
        Returns:
            xarray.DataArray: The computed JSDIV values.
        """
        obs_prob = self.obs_data / self.obs_data.sum(dim=dim)
        model_prob = self.model_data / self.model_data.sum(dim=dim)
        m = 0.5 * (obs_prob + model_prob)
        return 0.5 * ((obs_prob * np.log(obs_prob / m)).sum(dim=dim) + (model_prob * np.log(model_prob / m)).sum(dim=dim))

    def compute_hellinger(self, dim=None):
        """
        Compute the Hellinger Distance.
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the Hellinger Distance.
                                      If None, compute the Hellinger Distance over the entire data.
        
        Returns:
            xarray.DataArray: The computed Hellinger Distance values.
        """
        obs_prob = self.obs_data / self.obs_data.sum(dim=dim)
        model_prob = self.model_data / self.model_data.sum(dim=dim)
        return np.sqrt(0.5 * ((np.sqrt(obs_prob) - np.sqrt(model_prob)) ** 2).sum(dim=dim))

    def compute_wasserstein(self, dim=None):
        """
        Compute the Wasserstein Distance.
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the Wasserstein Distance.
                                      If None, compute the Wasserstein Distance over the entire data.
        
        Returns:
            xarray.DataArray: The computed Wasserstein Distance values.
        """
        obs_cdf = self.obs_data.cumsum(dim=dim) / self.obs_data.sum(dim=dim)
        model_cdf = self.model_data.cumsum(dim=dim) / self.model_data.sum(dim=dim)
        return np.abs(obs_cdf - model_cdf).sum(dim=dim)

    def compute_tv(self, dim=None):
        """
        Compute the Total Variation Distance.
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the Total Variation Distance.
                                      If None, compute the Total Variation Distance over the entire data.
        
        Returns:
            xarray.DataArray: The computed Total Variation Distance values.
        """
        obs_prob = self.obs_data / self.obs_data.sum(dim=dim)
        model_prob = self.model_data / self.model_data.sum(dim=dim)
        return 0.5 * np.abs(obs_prob - model_prob).sum(dim=dim)

    def compute_chisquare(self, dim=None):
        """
        Compute the Chi-Square Distance.
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the Chi-Square Distance.
                                      If None, compute the Chi-Square Distance over the entire data.
        
        Returns:
            xarray.DataArray: The computed Chi-Square Distance values.
        """
        obs_prob = self.obs_data / self.obs_data.sum(dim=dim)
        model_prob = self.model_data / self.model_data.sum(dim=dim)
        return ((obs_prob - model_prob) ** 2 / model_prob).sum(dim=dim)

    def compute_intersection(self, dim=None):
        """
        Compute the Intersection.
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the Intersection.
                                      If None, compute the Intersection over the entire data.
        
        Returns:
            xarray.DataArray: The computed Intersection values.
        """
        obs_prob = self.obs_data / self.obs_data.sum(dim=dim)
        model_prob = self.model_data / self.model_data.sum(dim=dim)
        return np.minimum(obs_prob, model_prob).sum(dim=dim)

    def compute_bhattacharyya(self, dim=None):
        """
        Compute the Bhattacharyya Distance.
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the Bhattacharyya Distance.
                                      If None, compute the Bhattacharyya Distance over the entire data.
        
        Returns:
            xarray.DataArray: The computed Bhattacharyya Distance values.
        """
        obs_prob = self.obs_data / self.obs_data.sum(dim=dim)
        model_prob = self.model_data / self.model_data.sum(dim=dim)
        return -np.log((np.sqrt(obs_prob * model_prob)).sum(dim=dim))

    def compute_harmonic_mean(self, dim=None):
        """
        Compute the Harmonic Mean.
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the Harmonic Mean.
                                      If None, compute the Harmonic Mean over the entire data.
        
        Returns:
            xarray.DataArray: The computed Harmonic Mean values.
        """
        obs_inv = 1 / self.obs_data
        model_inv = 1 / self.model_data
        return 2 / (obs_inv + model_inv)

    def compute_geometric_mean(self, dim=None):
        """
        Compute the Geometric Mean.
        
        Args:
            dim (str, list, or None): The dimension(s) along which to compute the Geometric Mean.
                                      If None, compute the Geometric Mean over the entire data.
        
        Returns:
            xarray.DataArray: The computed Geometric Mean values.
        """
        return np.sqrt(self.obs_data * self.model_data)

    def compute_lehmer_mean(self, p, dim=None):
        """
        Compute the Lehmer Mean.
        
        Args:
            p (float): The power parameter for the Lehmer Mean.
            dim (str, list, or None): The dimension(s) along which to compute the Lehmer Mean.
                                      If None, compute the Lehmer Mean over the entire data.
        
        Returns:
            xarray.DataArray: The computed Lehmer Mean values.
        """
        obs_pow = self.obs_data ** p
        model_pow = self.model_data ** p
        return (obs_pow + model_pow) / (self.obs_data ** (p - 1) + self.model_data ** (p - 1))

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
        obs_prob = self.obs_data / self.obs_data.sum(dim=dim)
        model_prob = self.model_data / self.model_data.sum(dim=dim)
        
        return -np.log((obs_prob ** alpha * model_prob ** (1 - alpha)).sum(dim=dim))

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
        obs_prob = self.obs_data / self.obs_data.sum(dim=dim)
        model_prob = self.model_data / self.model_data.sum(dim=dim)
        return (1 / (alpha - 1)) * np.log((obs_prob ** alpha / model_prob ** (alpha - 1)).sum(dim=dim))

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
        obs_prob = self.obs_data / self.obs_data.sum(dim=dim)
        model_prob = self.model_data / self.model_data.sum(dim=dim)
        return (1 / (alpha - 1)) * ((obs_prob ** alpha / model_prob ** (alpha - 1)).sum(dim=dim) - 1)
    
    
    #Further addition of metrics to the code. 
    

