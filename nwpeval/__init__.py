"""
NWPeval - Numerical Weather Prediction Evaluation Metrics

New API (recommended):
    from nwpeval import rmse, mae, fss, pod
    result = rmse(obs, model)

Legacy API (deprecated):
    from nwpeval import NWP_Stats
    stats = NWP_Stats(obs, model)  # Shows deprecation warning
    stats.compute_rmse()
"""

# Export all standalone metric functions (65 total)
from .metrics import (
    # Utilities
    confusion_matrix,
    # Continuous (29)
    mae, rmse, acc, r2, nrmse, pcc, mbd,
    tse, evs, nmse, fv, sdr, vif, mad, iqr,
    nae, rmb, mape, wmae, ass, rss, qss,
    lmbe, smse, gmb, sbs, aev, cosine_similarity,
    # Spatial (1)
    fss,
    # Categorical (20)
    ets, pod, far, csi, hss, pss, gss, fb, hkd, orss,
    seds, eds, sedi, f1, mcc, ba, npv, jaccard, gain, lift,
    # Probabilistic (2)
    bss, rpss,
    # Distributional (11)
    mkldiv, jsdiv, hellinger, wasserstein, tv, chisquare,
    intersection, bhattacharyya, chernoff, renyi, tsallis,
    # Means (3)
    harmonic_mean, geometric_mean, lehmer_mean,
)

# Legacy class (deprecated)
from .nwpeval import NWP_Stats

__all__ = [
    # New API - 65 metrics
    'confusion_matrix',
    'mae', 'rmse', 'acc', 'r2', 'nrmse', 'pcc', 'mbd',
    'tse', 'evs', 'nmse', 'fv', 'sdr', 'vif', 'mad', 'iqr',
    'nae', 'rmb', 'mape', 'wmae', 'ass', 'rss', 'qss',
    'lmbe', 'smse', 'gmb', 'sbs', 'aev', 'cosine_similarity',
    'fss',
    'ets', 'pod', 'far', 'csi', 'hss', 'pss', 'gss', 'fb', 'hkd', 'orss',
    'seds', 'eds', 'sedi', 'f1', 'mcc', 'ba', 'npv', 'jaccard', 'gain', 'lift',
    'bss', 'rpss',
    'mkldiv', 'jsdiv', 'hellinger', 'wasserstein', 'tv', 'chisquare',
    'intersection', 'bhattacharyya', 'chernoff', 'renyi', 'tsallis',
    'harmonic_mean', 'geometric_mean', 'lehmer_mean',
    # Legacy
    'NWP_Stats',
]