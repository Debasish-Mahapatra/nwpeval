
## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [NWP_Stats Class](#nwp_stats-class)
   - [Initialization](#initialization)
   - [Methods](#methods)
     - [compute_metrics](#compute_metrics)
     - [confusion_matrix](#confusion_matrix)
     - [compute_mae](#compute_mae)
     - [compute_rmse](#compute_rmse)
     - [compute_acc](#compute_acc)
     - [compute_fss](#compute_fss)
     - [compute_ets](#compute_ets)
     - [compute_pod](#compute_pod)
     - [compute_far](#compute_far)
     - [compute_csi](#compute_csi)
     - [compute_bss](#compute_bss)
     - [compute_hss](#compute_hss)
     - [compute_pss](#compute_pss)
     - [compute_gs](#compute_gs)
     - [compute_seds](#compute_seds)
     - [compute_fb](#compute_fb)
     - [compute_gss](#compute_gss)
     - [compute_hkd](#compute_hkd)
     - [compute_orss](#compute_orss)
     - [compute_eds](#compute_eds)
     - [compute_sedi](#compute_sedi)
     - [compute_rpss](#compute_rpss)
     - [compute_tse](#compute_tse)
     - [compute_evs](#compute_evs)
     - [compute_nmse](#compute_nmse)
     - [compute_fv](#compute_fv)
     - [compute_pcc](#compute_pcc)
     - [compute_sdr](#compute_sdr)
     - [compute_vif](#compute_vif)
     - [compute_mad](#compute_mad)
     - [compute_iqr](#compute_iqr)
     - [compute_r2](#compute_r2)
     - [compute_nae](#compute_nae)
     - [compute_rmb](#compute_rmb)
     - [compute_mape](#compute_mape)
     - [compute_wmae](#compute_wmae)
     - [compute_ass](#compute_ass)
     - [compute_rss](#compute_rss)
     - [compute_qss](#compute_qss)
     - [compute_nrmse](#compute_nrmse)
     - [compute_lmbe](#compute_lmbe)
     - [compute_smse](#compute_smse)
     - [compute_mbd](#compute_mbd)
     - [compute_gmb](#compute_gmb)
     - [compute_sbs](#compute_sbs)
     - [compute_aev](#compute_aev)
     - [compute_cosine_similarity](#compute_cosine_similarity)
     - [compute_f1](#compute_f1)
     - [compute_mcc](#compute_mcc)
     - [compute_ba](#compute_ba)
     - [compute_npv](#compute_npv)
     - [compute_jaccard](#compute_jaccard)
     - [compute_gain](#compute_gain)
     - [compute_lift](#compute_lift)
     - [compute_mkldiv](#compute_mkldiv)
     - [compute_jsdiv](#compute_jsdiv)
     - [compute_hellinger](#compute_hellinger)
     - [compute_wasserstein](#compute_wasserstein)
     - [compute_tv](#compute_tv)
     - [compute_chisquare](#compute_chisquare)
     - [compute_intersection](#compute_intersection)
     - [compute_bhattacharyya](#compute_bhattacharyya)
     - [compute_harmonic_mean](#compute_harmonic_mean)
     - [compute_geometric_mean](#compute_geometric_mean)
     - [compute_lehmer_mean](#compute_lehmer_mean)
     - [compute_chernoff](#compute_chernoff)
     - [compute_renyi](#compute_renyi)
     - [compute_tsallis](#compute_tsallis)
5. [Examples](#examples)
   - [Basic Usage](#basic-usage)
   - [Computing Metrics with Thresholds](#computing-metrics-with-thresholds)
   - [Computing Metrics Along Specific Dimensions](#computing-metrics-along-specific-dimensions)
   - [Computing Probabilistic Metrics with Custom Thresholds](#computing-probabilistic-metrics-with-custom-thresholds)
   - [Computing Weighted Metrics](#computing-weighted-metrics)
   - [Computing Distribution Comparison Metrics](#computing-distribution-comparison-metrics)
   - [Computing Reference-Based Metrics](#computing-reference-based-metrics)
   - [Computing Metrics with Custom Parameters](#computing-metrics-with-custom-parameters)
6. [Contributing](#contributing)
7. [License](#license)
8. [Citation](#citation)
9. [Contact](#contact)

## Introduction

NWPeval is a Python package designed for evaluating Numerical Weather Prediction (NWP) models. It provides a comprehensive set of metrics and tools to assess the performance of NWP models against observed data. The package aims to simplify the process of computing various statistical measures and skill scores commonly used in the NWP community.

The package is built around the `NWP_Stats` class, which encapsulates the functionality for computing metrics. It takes observed and modeled data as input and provides methods to compute a wide range of evaluation metrics. The package is designed to be user-friendly and flexible, allowing users to easily compute metrics over the entire dataset or along specific dimensions.

NWPeval seamlessly integrates with the popular `xarray` library, enabling efficient handling of multi-dimensional data arrays. It supports setting thresholds for binary classification metrics and provides customizable options for computing metrics with optional parameters and thresholds.

## Installation

To install the NWPeval package, you can use pip:

```shell
pip install nwpeval
```

Make sure you have Python installed on your system before running the above command.

## Getting Started

To start using NWPeval, you need to import the `NWP_Stats` class from the package:

```python
from nwpeval import NWP_Stats
```

Next, you need to prepare your observed and modeled data as `xarray.DataArray` objects. Ensure that the observed and modeled data have the same dimensions and shape.

```python
import xarray as xr

# Load observed data
obs_data = xr.DataArray(...)

# Load modeled data
model_data = xr.DataArray(...)
```

Once you have your data ready, you can create an instance of the `NWP_Stats` class by passing the observed and modeled data as arguments:

```python
nwp_stats = NWP_Stats(obs_data, model_data)
```

Now you are ready to compute various evaluation metrics using the methods provided by the `NWP_Stats` class.

## NWP_Stats Class

The `NWP_Stats` class is the core component of the NWPeval package. It provides methods to compute a wide range of evaluation metrics for NWP models.

### Initialization

To create an instance of the `NWP_Stats` class, you need to provide the observed and modeled data as `xarray.DataArray` objects:

```python
nwp_stats = NWP_Stats(obs_data, model_data)
```

### Methods

The `NWP_Stats` class provides the following methods to compute evaluation metrics:

#### compute_metrics

```python
compute_metrics(self, metrics, dim=None, thresholds=None)
```

This method computes the specified metrics using the observed and modeled data.

- `metrics` (list): A list of metric names to compute.
- `dim` (str, list, or None): The dimension(s) along which to compute the metrics. If None, compute the metrics over the entire data.
- `thresholds` (dict): A dictionary containing threshold values for specific metrics.

Returns a dictionary containing the computed metric values.

#### confusion_matrix

```python
confusion_matrix(self, obs_binary, model_binary, dim=None)
```

This method computes the confusion matrix for binary classification.

- `obs_binary` (xarray.DataArray): The binarized observed data.
- `model_binary` (xarray.DataArray): The binarized modeled data.
- `dim` (str, list, or None): The dimension(s) along which to compute the confusion matrix. If None, compute the confusion matrix over the entire data.

Returns a tuple containing the confusion matrix values (tn, fp, fn, tp).

#### compute_mae

```python
compute_mae(self, dim=None)
```

This method computes the Mean Absolute Error (MAE).

- `dim` (str, list, or None): The dimension(s) along which to compute the MAE. If None, compute the MAE over the entire data.

Returns the computed MAE value.

#### compute_rmse

```python
compute_rmse(self, dim=None)
```

This method computes the Root Mean Square Error (RMSE).

- `dim` (str, list, or None): The dimension(s) along which to compute the RMSE. If None, compute the RMSE over the entire data.

Returns the computed RMSE value.

#### compute_acc

```python
compute_acc(self, dim=None)
```

This method computes the Anomaly Correlation Coefficient (ACC).

- `dim` (str, list, or None): The dimension(s) along which to compute the ACC. If None, compute the ACC over the entire data.

Returns the computed ACC value.

#### compute_fss

```python
compute_fss(self, threshold, neighborhood_size, dim=None)
```

This method computes the Fractions Skill Score (FSS) for a given threshold and neighborhood size.

- `threshold` (float): The threshold value for binary classification.
- `neighborhood_size` (int): The size of the neighborhood window.
- `dim` (str, list, or None): The dimension(s) along which to compute the FSS. If None, compute the FSS over the entire data.

Returns the computed FSS value.

#### compute_ets

```python
compute_ets(self, threshold, dim=None)
```

This method computes the Equitable Threat Score (ETS) for a given threshold.

- `threshold` (float): The threshold value for binary classification.
- `dim` (str, list, or None): The dimension(s) along which to compute the ETS. If None, compute the ETS over the entire data.

Returns the computed ETS value.

#### compute_pod

```python
compute_pod(self, threshold, dim=None)
```

This method computes the Probability of Detection (POD) for a given threshold.

- `threshold` (float): The threshold value for binary classification.
- `dim` (str, list, or None): The dimension(s) along which to compute the POD. If None, compute the POD over the entire data.

Returns the computed POD value.

#### compute_far

```python
compute_far(self, threshold, dim=None)
```

This method computes the False Alarm Ratio (FAR) for a given threshold.

- `threshold` (float): The threshold value for binary classification.
- `dim` (str, list, or None): The dimension(s) along which to compute the FAR. If None, compute the FAR over the entire data.

Returns the computed FAR value.

#### compute_csi

```python
compute_csi(self, threshold, dim=None)
```

This method computes the Critical Success Index (CSI) for a given threshold.

- `threshold` (float): The threshold value for binary classification.
- `dim` (str, list, or None): The dimension(s) along which to compute the CSI. If None, compute the CSI over the entire data.

Returns the computed CSI value.

#### compute_bss

```python
compute_bss(self, threshold, dim=None)
```

This method computes the Brier Skill Score (BSS) for a given threshold.

- `threshold` (float): The threshold value for binary classification.
- `dim` (str, list, or None): The dimension(s) along which to compute the BSS. If None, compute the BSS over the entire data.

Returns the computed BSS value.

#### compute_hss

```python
compute_hss(self, threshold, dim=None)
```

This method computes the Heidke Skill Score (HSS) for a given threshold.

- `threshold` (float): The threshold value for binary classification.
- `dim` (str, list, or None): The dimension(s) along which to compute the HSS. If None, compute the HSS over the entire data.

Returns the computed HSS value.

#### compute_pss

```python
compute_pss(self, threshold, dim=None)
```

This method computes the Peirce Skill Score (PSS) for a given threshold.

- `threshold` (float): The threshold value for binary classification.
- `dim` (str, list, or None): The dimension(s) along which to compute the PSS. If None, compute the PSS over the entire data.

Returns the computed PSS value.

#### compute_gs

```python
compute_gs(self, threshold, dim=None)
```

This method computes the Gilbert Skill Score (GS) for a given threshold.

- `threshold` (float): The threshold value for binary classification.
- `dim` (str, list, or None): The dimension(s) along which to compute the GS. If None, compute the GS over the entire data.

Returns the computed GS value.

#### compute_seds

```python
compute_seds(self, threshold, dim=None)
```

This method computes the Symmetric Extreme Dependency Score (SEDS) for a given threshold.

- `threshold` (float): The threshold value for binary classification.
- `dim` (str, list, or None): The dimension(s) along which to compute the SEDS. If None, compute the SEDS over the entire data.

Returns the computed SEDS value.

#### compute_fb

```python
compute_fb(self, threshold, dim=None)
```

This method computes the Frequency Bias (FB) for a given threshold.

- `threshold` (float): The threshold value for binary classification.
- `dim` (str, list, or None): The dimension(s) along which to compute the FB. If None, compute the FB over the entire data.

Returns the computed FB value.

#### compute_gss

```python
compute_gss(self, threshold, dim=None)
```

This method computes the Gilbert Skill Score (GSS) for a given threshold.

- `threshold` (float): The threshold value for binary classification.
- `dim` (str, list, or None): The dimension(s) along which to compute the GSS. If None, compute the GSS over the entire data.

Returns the computed GSS value.

#### compute_hkd

```python
compute_hkd(self, threshold, dim=None)
```

This method computes the Hanssen-Kuipers Discriminant (H-KD) for a given threshold.

- `threshold` (float): The threshold value for binary classification.
- `dim` (str, list, or None): The dimension(s) along which to compute the H-KD. If None, compute the H-KD over the entire data.

Returns the computed H-KD value.

#### compute_orss

```python
compute_orss(self, threshold, dim=None)
```

This method computes the Odds Ratio Skill Score (ORSS) for a given threshold.

- `threshold` (float): The threshold value for binary classification.
- `dim` (str, list, or None): The dimension(s) along which to compute the ORSS. If None, compute the ORSS over the entire data.

Returns the computed ORSS value.

#### compute_eds

```python
compute_eds(self, threshold, dim=None)
```

This method computes the Extreme Dependency Score (EDS) for a given threshold.

- `threshold` (float): The threshold value for binary classification.
- `dim` (str, list, or None): The dimension(s) along which to compute the EDS. If None, compute the EDS over the entire data.

Returns the computed EDS value.

#### compute_sedi

```python
compute_sedi(self, threshold, dim=None)
```

This method computes the Symmetric Extremal Dependence Index (SEDI) for a given threshold.

- `threshold` (float): The threshold value for binary classification.
- `dim` (str, list, or None): The dimension(s) along which to compute the SEDI. If None, compute the SEDI over the entire data.

Returns the computed SEDI value. 

#### compute_rpss

```python
compute_rpss(self, threshold, dim=None)
```

This method computes the Ranked Probability Skill Score (RPSS) for a given threshold.

- `threshold` (float): The threshold value for binary classification.
- `dim` (str, list, or None): The dimension(s) along which to compute the RPSS. If None, compute the RPSS over the entire data.

Returns the computed RPSS value.

#### compute_tse

```python
compute_tse(self, dim=None)
```

This method computes the Total Squared Error (TSE).

- `dim` (str, list, or None): The dimension(s) along which to compute the TSE. If None, compute the TSE over the entire data.

Returns the computed TSE value.

#### compute_evs

```python
compute_evs(self, dim=None)
```

This method computes the Explained Variance Score (EVS).

- `dim` (str, list, or None): The dimension(s) along which to compute the EVS. If None, compute the EVS over the entire data.

Returns the computed EVS value.

#### compute_nmse

```python
compute_nmse(self, dim=None)
```

This method computes the Normalized Mean Squared Error (NMSE).

- `dim` (str, list, or None): The dimension(s) along which to compute the NMSE. If None, compute the NMSE over the entire data.

Returns the computed NMSE value.

#### compute_fv

```python
compute_fv(self, dim=None)
```

This method computes the Fractional Variance (FV).

- `dim` (str, list, or None): The dimension(s) along which to compute the FV. If None, compute the FV over the entire data.

Returns the computed FV value.

#### compute_pcc

```python
compute_pcc(self, dim=None)
```

This method computes the Pearson Correlation Coefficient (PCC).

- `dim` (str, list, or None): The dimension(s) along which to compute the PCC. If None, compute the PCC over the entire data.

Returns the computed PCC value.

#### compute_sdr

```python
compute_sdr(self, dim=None)
```

This method computes the Standard Deviation Ratio (SDR).

- `dim` (str, list, or None): The dimension(s) along which to compute the SDR. If None, compute the SDR over the entire data.

Returns the computed SDR value.

#### compute_vif

```python
compute_vif(self, dim=None)
```

This method computes the Variance Inflation Factor (VIF).

- `dim` (str, list, or None): The dimension(s) along which to compute the VIF. If None, compute the VIF over the entire data.

Returns the computed VIF value.

#### compute_mad

```python
compute_mad(self, dim=None)
```

This method computes the Median Absolute Deviation (MAD).

- `dim` (str, list, or None): The dimension(s) along which to compute the MAD. If None, compute the MAD over the entire data.

Returns the computed MAD value.

#### compute_iqr

```python
compute_iqr(self, dim=None)
```

This method computes the Interquartile Range (IQR).

- `dim` (str, list, or None): The dimension(s) along which to compute the IQR. If None, compute the IQR over the entire data.

Returns the computed IQR value.

#### compute_r2

```python
compute_r2(self, dim=None)
```

This method computes the Coefficient of Determination (R^2).

- `dim` (str, list, or None): The dimension(s) along which to compute the R^2. If None, compute the R^2 over the entire data.

Returns the computed R^2 value.

#### compute_nae

```python
compute_nae(self, dim=None)
```

This method computes the Normalized Absolute Error (NAE).

- `dim` (str, list, or None): The dimension(s) along which to compute the NAE. If None, compute the NAE over the entire data.

Returns the computed NAE value.

#### compute_rmb

```python
compute_rmb(self, dim=None)
```

This method computes the Relative Mean Bias (RMB).

- `dim` (str, list, or None): The dimension(s) along which to compute the RMB. If None, compute the RMB over the entire data.

Returns the computed RMB value.

#### compute_mape

```python
compute_mape(self, dim=None)
```

This method computes the Mean Absolute Percentage Error (MAPE).

- `dim` (str, list, or None): The dimension(s) along which to compute the MAPE. If None, compute the MAPE over the entire data.

Returns the computed MAPE value.

#### compute_wmae

```python
compute_wmae(self, weights, dim=None)
```

This method computes the Weighted Mean Absolute Error (WMAE).

- `weights` (xarray.DataArray): The weights for each data point.
- `dim` (str, list, or None): The dimension(s) along which to compute the WMAE. If None, compute the WMAE over the entire data.

Returns the computed WMAE value.

#### compute_ass

```python
compute_ass(self, reference_error, dim=None)
```

This method computes the Absolute Skill Score (ASS).

- `reference_error` (xarray.DataArray): The reference error values.
- `dim` (str, list, or None): The dimension(s) along which to compute the ASS. If None, compute the ASS over the entire data.

Returns the computed ASS value.

#### compute_rss

```python
compute_rss(self, reference_skill, dim=None)
```

This method computes the Relative Skill Score (RSS).

- `reference_skill` (xarray.DataArray): The reference skill values.
- `dim` (str, list, or None): The dimension(s) along which to compute the RSS. If None, compute the RSS over the entire data.

Returns the computed RSS value.

#### compute_qss

```python
compute_qss(self, reference_forecast, dim=None)
```

This method computes the Quadratic Skill Score (QSS).

- `reference_forecast` (xarray.DataArray): The reference forecast values.
- `dim` (str, list, or None): The dimension(s) along which to compute the QSS. If None, compute the QSS over the entire data.

Returns the computed QSS value.

#### compute_nrmse

```python
compute_nrmse(self, dim=None)
```

This method computes the Normalized Root Mean Squared Error (NRMSE).

- `dim` (str, list, or None): The dimension(s) along which to compute the NRMSE. If None, compute the NRMSE over the entire data.

Returns the computed NRMSE value.

#### compute_lmbe

```python
compute_lmbe(self, dim=None)
```

This method computes the Logarithmic Mean Bias Error (LMBE).

- `dim` (str, list, or None): The dimension(s) along which to compute the LMBE. If None, compute the LMBE over the entire data.

Returns the computed LMBE value.

#### compute_smse

```python
compute_smse(self, dim=None)
```

This method computes the Scaled Mean Squared Error (SMSE).

- `dim` (str, list, or None): The dimension(s) along which to compute the SMSE. If None, compute the SMSE over the entire data.

Returns the computed SMSE value.

#### compute_mbd

```python
compute_mbd(self, dim=None)
```

This method computes the Mean Bias Deviation (MBD).

- `dim` (str, list, or None): The dimension(s) along which to compute the MBD. If None, compute the MBD over the entire data.

Returns the computed MBD value.

#### compute_gmb

```python
compute_gmb(self, dim=None)
```

This method computes the Geometric Mean Bias (GMB).

- `dim` (str, list, or None): The dimension(s) along which to compute the GMB. If None, compute the GMB over the entire data.

Returns the computed GMB value.

#### compute_sbs

```python
compute_sbs(self, dim=None)
```

This method computes the Symmetric Brier Score (SBS).

- `dim` (str, list, or None): The dimension(s) along which to compute the SBS. If None, compute the SBS over the entire data.

Returns the computed SBS value.

#### compute_aev

```python
compute_aev(self, dim=None)
```

This method computes the Adjusted Explained Variance (AEV).

- `dim` (str, list, or None): The dimension(s) along which to compute the AEV. If None, compute the AEV over the entire data.

Returns the computed AEV value.

#### compute_cosine_similarity

```python
compute_cosine_similarity(self, dim=None)
```

This method computes the Cosine Similarity.

- `dim` (str, list, or None): The dimension(s) along which to compute the Cosine Similarity. If None, compute the Cosine Similarity over the entire data.

Returns the computed Cosine Similarity value.

#### compute_f1

```python
compute_f1(self, threshold, dim=None)
```

This method computes the F1 Score for a given threshold.

- `threshold` (float): The threshold value for binary classification.
- `dim` (str, list, or None): The dimension(s) along which to compute the F1 Score. If None, compute the F1 Score over the entire data.

Returns the computed F1 Score value.

#### compute_mcc

```python
compute_mcc(self, threshold, dim=None)
```

This method computes the Matthews Correlation Coefficient (MCC) for a given threshold.

- `threshold` (float): The threshold value for binary classification.
- `dim` (str, list, or None): The dimension(s) along which to compute the MCC. If None, compute the MCC over the entire data.

Returns the computed MCC value.

#### compute_ba

```python
compute_ba(self, threshold, dim=None)
```

This method computes the Balanced Accuracy (BA) for a given threshold.

- `threshold` (float): The threshold value for binary classification.
- `dim` (str, list, or None): The dimension(s) along which to compute the BA. If None, compute the BA over the entire data.

Returns the computed BA value.

#### compute_npv

```python
compute_npv(self, threshold, dim=None)
```

This method computes the Negative Predictive Value (NPV) for a given threshold.

- `threshold` (float): The threshold value for binary classification.
- `dim` (str, list, or None): The dimension(s) along which to compute the NPV. If None, compute the NPV over the entire data.

Returns the computed NPV value.

#### compute_jaccard

```python
compute_jaccard(self, threshold, dim=None)
```

This method computes the Jaccard Similarity Coefficient for a given threshold.

- `threshold` (float): The threshold value for binary classification.
- `dim` (str, list, or None): The dimension(s) along which to compute the Jaccard Similarity Coefficient. If None, compute the Jaccard Similarity Coefficient over the entire data.

Returns the computed Jaccard Similarity Coefficient value.

#### compute_gain

```python
compute_gain(self, threshold, dim=None)
```

This method computes the Gain for a given threshold.

- `threshold` (float): The threshold value for binary classification.
- `dim` (str, list, or None): The dimension(s) along which to compute the Gain. If None, compute the Gain over the entire data.

Returns the computed Gain value.

#### compute_lift

```python
compute_lift(self, threshold, dim=None)
```

This method computes the Lift for a given threshold.

- `threshold` (float): The threshold value for binary classification.
- `dim` (str, list, or None): The dimension(s) along which to compute the Lift. If None, compute the Lift over the entire data.

Returns the computed Lift value.

#### compute_mkldiv

```python
compute_mkldiv(self, dim=None)
```

This method computes the Mean Kullback-Leibler Divergence (MKLDIV).

- `dim` (str, list, or None): The dimension(s) along which to compute the MKLDIV. If None, compute the MKLDIV over the entire data.

Returns the computed MKLDIV value.

#### compute_jsdiv

```python
compute_jsdiv(self, dim=None)
```

This method computes the Jensen-Shannon Divergence (JSDIV).

- `dim` (str, list, or None): The dimension(s) along which to compute the JSDIV. If None, compute the JSDIV over the entire data.

Returns the computed JSDIV value.

#### compute_hellinger

```python
compute_hellinger(self, dim=None)
```

This method computes the Hellinger Distance.

- `dim` (str, list, or None): The dimension(s) along which to compute the Hellinger Distance. If None, compute the Hellinger Distance over the entire data.

Returns the computed Hellinger Distance value.

#### compute_wasserstein

```python
compute_wasserstein(self, dim=None)
```

This method computes the Wasserstein Distance.

- `dim` (str, list, or None): The dimension(s) along which to compute the Wasserstein Distance. If None, compute the Wasserstein Distance over the entire data.

Returns the computed Wasserstein Distance value.

#### compute_tv

```python
compute_tv(self, dim=None)
```

This method computes the Total Variation Distance.

- `dim` (str, list, or None): The dimension(s) along which to compute the Total Variation Distance. If None, compute the Total Variation Distance over the entire data.

Returns the computed Total Variation Distance value.

#### compute_chisquare

```python
compute_chisquare(self, dim=None)
```

This method computes the Chi-Square Distance.

- `dim` (str, list, or None): The dimension(s) along which to compute the Chi-Square Distance. If None, compute the Chi-Square Distance over the entire data.

Returns the computed Chi-Square Distance value.

#### compute_intersection

```python
compute_intersection(self, dim=None)
```

This method computes the Intersection.

- `dim` (str, list, or None): The dimension(s) along which to compute the Intersection. If None, compute the Intersection over the entire data.

Returns the computed Intersection value.

#### compute_bhattacharyya

```python
compute_bhattacharyya(self, dim=None)
```

This method computes the Bhattacharyya Distance.

- `dim` (str, list, or None): The dimension(s) along which to compute the Bhattacharyya Distance. If None, compute the Bhattacharyya Distance over the entire data.

Returns the computed Bhattacharyya Distance value.

#### compute_harmonic_mean

```python
compute_harmonic_mean(self, dim=None)
```

This method computes the Harmonic Mean.

- `dim` (str, list, or None): The dimension(s) along which to compute the Harmonic Mean. If None, compute the Harmonic Mean over the entire data.

Returns the computed Harmonic Mean value.

#### compute_geometric_mean

```python
compute_geometric_mean(self, dim=None)
```

This method computes the Geometric Mean.

- `dim` (str, list, or None): The dimension(s) along which to compute the Geometric Mean. If None, compute the Geometric Mean over the entire data.

Returns the computed Geometric Mean value.

#### compute_lehmer_mean

```python
compute_lehmer_mean(self, p, dim=None)
```

This method computes the Lehmer Mean.

- `p` (float): The power parameter for the Lehmer Mean.
- `dim` (str, list, or None): The dimension(s) along which to compute the Lehmer Mean. If None, compute the Lehmer Mean over the entire data.

Returns the computed Lehmer Mean value.

#### compute_chernoff

```python
compute_chernoff(self, alpha, dim=None)
```

This method computes the Chernoff Distance.

- `alpha` (float): The parameter for the Chernoff Distance (0 < alpha < 1).
- `dim` (str, list, or None): The dimension(s) along which to compute the Chernoff Distance. If None, compute the Chernoff Distance over the entire data.

Returns the computed Chernoff Distance value.

#### compute_renyi

```python
compute_renyi(self, alpha, dim=None)
```

This method computes the Rényi Divergence.

- `alpha` (float): The parameter for the Rényi Divergence (alpha != 1).
- `dim` (str, list, or None): The dimension(s) along which to compute the Rényi Divergence. If None, compute the Rényi Divergence over the entire data.

Returns the computed Rényi Divergence value.

#### compute_tsallis

```python
compute_tsallis(self, alpha, dim=None)
```

This method computes the Tsallis Divergence.

- `alpha` (float): The parameter for the Tsallis Divergence (alpha != 1).
- `dim` (str, list, or None): The dimension(s) along which to compute the Tsallis Divergence. If None, compute the Tsallis Divergence over the entire data.

Returns the computed Tsallis Divergence value.

## Examples

Here are some examples demonstrating how to use the NWPeval package:

### Basic Usage

```python
import xarray as xr
from nwpeval import NWP_Stats

# Load observed and modeled data as xarray DataArrays
obs_data = xr.DataArray(...)
model_data = xr.DataArray(...)

# Create an instance of NWP_Stats
nwp_stats = NWP_Stats(obs_data, model_data)

# Compute basic metrics
metrics = ['MAE', 'RMSE', 'ACC']
results = nwp_stats.compute_metrics(metrics)

# Print the results
for metric, value in results.items():
    print(f"{metric}: {value}")
```

### Computing Metrics with Thresholds

```python
import xarray as xr
from nwpeval import NWP_Stats

# Load observed and modeled data as xarray DataArrays
obs_data = xr.DataArray(...)
model_data = xr.DataArray(...)

# Create an instance of NWP_Stats
nwp_stats = NWP_Stats(obs_data, model_data)

# Define thresholds for specific metrics
thresholds = {
    'FSS': 0.6,
    'FSS_neighborhood': 5,
    'ETS': 0.7,
    'POD': 0.5
}

# Compute metrics with thresholds
metrics = ['FSS', 'ETS', 'POD']
results = nwp_stats.compute_metrics(metrics, thresholds=thresholds)

# Print the results
for metric, value in results.items():
    print(f"{metric}: {value}")
```

### Computing Metrics Along Specific Dimensions

```python
import xarray as xr
from nwpeval import NWP_Stats

# Load observed and modeled data as xarray DataArrays
obs_data = xr.DataArray(...)
model_data = xr.DataArray(...)

# Create an instance of NWP_Stats
nwp_stats = NWP_Stats(obs_data, model_data)

# Compute metrics along specific dimensions
metrics = ['MAE', 'RMSE', 'ACC']
dimensions = ['lat', 'lon']
results = nwp_stats.compute_metrics(metrics, dim=dimensions)

# Print the results
for metric, value in results.items():
    print(f"{metric}:")
    print(value)
```

### Computing Probabilistic Metrics with Custom Thresholds

```python
import xarray as xr
from nwpeval import NWP_Stats

# Load observed and modeled data as xarray DataArrays
obs_data = xr.DataArray(...)
model_data = xr.DataArray(...)

# Create an instance of NWP_Stats
nwp_stats = NWP_Stats(obs_data, model_data)

# Define custom thresholds for probabilistic metrics
thresholds = {
    'SEDS': 0.6,
    'SEDI': 0.7,
    'RPSS': 0.8
}

# Compute probabilistic metrics with custom thresholds
metrics = ['SEDS', 'SEDI', 'RPSS']
results = nwp_stats.compute_metrics(metrics, thresholds=thresholds)

# Print the results
for metric, value in results.items():
    print(f"{metric}: {value}")
```

### Computing Weighted Metrics

```python
import xarray as xr
from nwpeval import NWP_Stats

# Load observed and modeled data as xarray DataArrays
obs_data = xr.DataArray(...)
model_data = xr.DataArray(...)

# Create an instance of NWP_Stats
nwp_stats = NWP_Stats(obs_data, model_data)

# Define weights for weighted metrics
weights = xr.DataArray(...)

# Compute weighted metrics
metrics = ['WMAE']
thresholds = {'WMAE_weights': weights}
results = nwp_stats.compute_metrics(metrics, thresholds=thresholds)

# Print the results
for metric, value in results.items():
    print(f"{metric}: {value}")
```

### Computing Distribution Comparison Metrics

```python
import xarray as xr
from nwpeval import NWP_Stats

# Load observed and modeled data as xarray DataArrays
obs_data = xr.DataArray(...)
model_data = xr.DataArray(...)

# Create an instance of NWP_Stats
nwp_stats = NWP_Stats(obs_data, model_data)

# Compute distribution comparison metrics
metrics = ['MKLDIV', 'JSDIV', 'Hellinger', 'Wasserstein']
results = nwp_stats.compute_metrics(metrics)

# Print the results
for metric, value in results.items():
    print(f"{metric}: {value}")
```

### Computing Reference-Based Metrics

```python
import xarray as xr
from nwpeval import NWP_Stats

# Load observed, modeled, and reference data as xarray DataArrays
obs_data = xr.DataArray(...)
model_data = xr.DataArray(...)
reference_data = xr.DataArray(...)

# Create an instance of NWP_Stats
nwp_stats = NWP_Stats(obs_data, model_data)

# Define thresholds for reference-based metrics
thresholds = {
    'ASS_reference_error': reference_data,
    'RSS_reference_skill': 0.6,
    'QSS_reference_forecast': reference_data
}

# Compute reference-based metrics
metrics = ['ASS', 'RSS', 'QSS']
results = nwp_stats.compute_metrics(metrics, thresholds=thresholds)

# Print the results
for metric, value in results.items():
    print(f"{metric}: {value}")
```

### Computing Metrics with Custom Parameters

```python
import xarray as xr
from nwpeval import NWP_Stats

# Load observed and modeled data as xarray DataArrays
obs_data = xr.DataArray(...)
model_data = xr.DataArray(...)

# Create an instance of NWP_Stats
nwp_stats = NWP_Stats(obs_data, model_data)

# Define custom parameters for metrics
thresholds = {
    'LehmerMean_p': 3,
    'Chernoff_alpha': 0.7,
    'Renyi_alpha': 0.8,
    'Tsallis_alpha': 0.9
}

# Compute metrics with custom parameters
metrics = ['LehmerMean', 'Chernoff', 'Renyi', 'Tsallis']
results = nwp_stats.compute_metrics(metrics, thresholds=thresholds)

# Print the results
for metric, value in results.items():
    print(f"{metric}: {value}")
```

## Contributing

Contributions to NWPeval are welcome! If you find any issues or have suggestions for improvements, please open an issue on the [GitHub repository](https://github.com/Debasish-Mahapatra/nwpeval). If you would like to contribute code, you can fork the repository and submit a pull request.

## License

NWPeval is released under the MIT License. See the [LICENSE](https://github.com/Debasish-Mahapatra/nwpeval/blob/main/LICENSE) file for more details.

## Citation

If you use NWPeval in any scientific publications or research, please cite the package as follows:

```
Mahapatra, D. (2024). NWPeval: A Python Package for Evaluating Numerical Weather Prediction Models. GitHub repository. https://github.com/Debasish-Mahapatra/nwpeval
```

## Contact

For any questions, suggestions, or feedback, please contact Debasish Mahapatra at [debasish.mahapatra@ugent.be].