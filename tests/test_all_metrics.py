"""
Test all nwpeval metrics with real data.
Uses Radar (observation) vs 2-Moment (model).
"""
import sys
sys.path.insert(0, '/Users/dev/PROJECTS/nwpeval-main')

import xarray as xr
import numpy as np
from nwpeval import (
    # Continuous metrics
    mae, rmse, acc, r2, nrmse, pcc, mbd,
    tse, evs, nmse, fv, sdr, vif, mad, iqr,
    nae, rmb, mape, lmbe, smse, gmb, sbs, aev,
    cosine_similarity,
    # Spatial
    fss,
    # Categorical
    ets, pod, far, csi, hss, pss, gss, fb, hkd, orss,
    seds, eds, sedi, f1, mcc, ba, npv, jaccard, gain, lift,
    # Probabilistic
    bss, rpss,
    # Distributional
    mkldiv, jsdiv, hellinger, wasserstein, tv, chisquare,
    intersection, bhattacharyya,
    # Means
    harmonic_mean, geometric_mean,
)

# Load data
data_dir = '/Volumes/crucial-ssd/ALARO/manaus-production-runs/rainfall-regridded-to-imerge/masked-production-final/common-valid-time-production'
obs = xr.open_dataset(f'{data_dir}/Radar_common_valid.nc')['rainfall_rate']
model = xr.open_dataset(f'{data_dir}/2-Moment_common_valid.nc')['total_rain']

print(f"Obs shape: {obs.shape}, Model shape: {model.shape}")
print(f"Obs range: [{float(obs.min()):.3f}, {float(obs.max()):.3f}]")
print(f"Model range: [{float(model.min()):.3f}, {float(model.max()):.3f}]")

dims = list(obs.dims)
threshold = 1.0

print("\n" + "="*60)
print("CONTINUOUS METRICS")
print("="*60)

continuous_metrics = {
    'MAE': lambda: mae(obs, model, dim=dims),
    'RMSE': lambda: rmse(obs, model, dim=dims),
    'ACC': lambda: acc(obs, model, dim=dims),
    'R2': lambda: r2(obs, model, dim=dims),
    'NRMSE': lambda: nrmse(obs, model, dim=dims),
    'PCC': lambda: pcc(obs, model, dim=dims),
    'MBD': lambda: mbd(obs, model, dim=dims),
    'TSE': lambda: tse(obs, model, dim=dims),
    'EVS': lambda: evs(obs, model, dim=dims),
    'NMSE': lambda: nmse(obs, model, dim=dims),
    'FV': lambda: fv(obs, model, dim=dims),
    'SDR': lambda: sdr(obs, model, dim=dims),
    'VIF': lambda: vif(obs, model, dim=dims),
    'MAD': lambda: mad(obs, model, dim=dims),
    'IQR': lambda: iqr(obs, model, dim=dims),
    'NAE': lambda: nae(obs, model, dim=dims),
    'RMB': lambda: rmb(obs, model, dim=dims),
    'MAPE': lambda: mape(obs, model, dim=dims),
    'LMBE': lambda: lmbe(obs, model, dim=dims),
    'SMSE': lambda: smse(obs, model, dim=dims),
    'GMB': lambda: gmb(obs, model, dim=dims),
    'SBS': lambda: sbs(obs, model, dim=dims),
    'AEV': lambda: aev(obs, model, dim=dims),
    'Cosine Similarity': lambda: cosine_similarity(obs, model, dim=dims),
}

for name, func in continuous_metrics.items():
    try:
        result = func()
        val = float(result.values) if hasattr(result, 'values') else float(result)
        print(f"  {name}: {val:.6f} ✓")
    except Exception as e:
        print(f"  {name}: ERROR - {e}")

print("\n" + "="*60)
print(f"CATEGORICAL METRICS (threshold={threshold})")
print("="*60)

categorical_metrics = {
    'POD': lambda: pod(obs, model, threshold=threshold, dim=dims),
    'FAR': lambda: far(obs, model, threshold=threshold, dim=dims),
    'CSI': lambda: csi(obs, model, threshold=threshold, dim=dims),
    'ETS': lambda: ets(obs, model, threshold=threshold, dim=dims),
    'HSS': lambda: hss(obs, model, threshold=threshold, dim=dims),
    'PSS': lambda: pss(obs, model, threshold=threshold, dim=dims),
    'GSS': lambda: gss(obs, model, threshold=threshold, dim=dims),
    'FB': lambda: fb(obs, model, threshold=threshold, dim=dims),
    'HKD': lambda: hkd(obs, model, threshold=threshold, dim=dims),
    'ORSS': lambda: orss(obs, model, threshold=threshold, dim=dims),
    'SEDS': lambda: seds(obs, model, threshold=threshold, dim=dims),
    'EDS': lambda: eds(obs, model, threshold=threshold, dim=dims),
    'SEDI': lambda: sedi(obs, model, threshold=threshold, dim=dims),
    'F1': lambda: f1(obs, model, threshold=threshold, dim=dims),
    'MCC': lambda: mcc(obs, model, threshold=threshold, dim=dims),
    'BA': lambda: ba(obs, model, threshold=threshold, dim=dims),
    'NPV': lambda: npv(obs, model, threshold=threshold, dim=dims),
    'Jaccard': lambda: jaccard(obs, model, threshold=threshold, dim=dims),
    'Gain': lambda: gain(obs, model, threshold=threshold, dim=dims),
    'Lift': lambda: lift(obs, model, threshold=threshold, dim=dims),
}

for name, func in categorical_metrics.items():
    try:
        result = func()
        val = float(result.values) if hasattr(result, 'values') else float(result)
        print(f"  {name}: {val:.6f} ✓")
    except Exception as e:
        print(f"  {name}: ERROR - {e}")

print("\n" + "="*60)
print("SPATIAL METRICS (FSS)")
print("="*60)

try:
    result = fss(obs, model, threshold=threshold, neighborhood_size=5, spatial_dims=['lat', 'lon'])
    print(f"  FSS (n=5): {float(result.mean()):.6f} ✓")
except Exception as e:
    print(f"  FSS: ERROR - {e}")

print("\n" + "="*60)
print(f"PROBABILISTIC METRICS (threshold={threshold})")
print("="*60)

probabilistic_metrics = {
    'BSS': lambda: bss(obs, model, threshold=threshold, dim=dims),
    'RPSS': lambda: rpss(obs, model, threshold=threshold, dim=dims),
}

for name, func in probabilistic_metrics.items():
    try:
        result = func()
        val = float(result.values) if hasattr(result, 'values') else float(result)
        print(f"  {name}: {val:.6f} ✓")
    except Exception as e:
        print(f"  {name}: ERROR - {e}")

print("\n" + "="*60)
print("DISTRIBUTIONAL METRICS")
print("="*60)

# Use only positive values for distributional metrics
obs_pos = obs.where(obs > 0, 0.001)
model_pos = model.where(model > 0, 0.001)

distributional_metrics = {
    'MKLDIV': lambda: mkldiv(obs_pos, model_pos, dim=dims),
    'JSDIV': lambda: jsdiv(obs_pos, model_pos, dim=dims),
    'Hellinger': lambda: hellinger(obs_pos, model_pos, dim=dims),
    'Wasserstein': lambda: wasserstein(obs_pos, model_pos, dim=dims),
    'TV': lambda: tv(obs_pos, model_pos, dim=dims),
    'ChiSquare': lambda: chisquare(obs_pos, model_pos, dim=dims),
    'Intersection': lambda: intersection(obs_pos, model_pos, dim=dims),
    'Bhattacharyya': lambda: bhattacharyya(obs_pos, model_pos, dim=dims),
}

for name, func in distributional_metrics.items():
    try:
        result = func()
        val = float(result.values) if hasattr(result, 'values') else float(result)
        print(f"  {name}: {val:.6f} ✓")
    except Exception as e:
        print(f"  {name}: ERROR - {e}")

print("\n" + "="*60)
print("MEAN METRICS")
print("="*60)

mean_metrics = {
    'Harmonic Mean': lambda: harmonic_mean(obs_pos, model_pos),
    'Geometric Mean': lambda: geometric_mean(obs_pos, model_pos),
}

for name, func in mean_metrics.items():
    try:
        result = func()
        val = float(result.mean().values) if hasattr(result, 'mean') else float(result)
        print(f"  {name} (mean): {val:.6f} ✓")
    except Exception as e:
        print(f"  {name}: ERROR - {e}")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
