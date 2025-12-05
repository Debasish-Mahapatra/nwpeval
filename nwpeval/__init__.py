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

# Export all standalone metric functions
from .metrics import (
    # Utilities
    confusion_matrix,
    # Continuous
    mae, rmse, acc, r2, nrmse, pcc, mbd,
    # Spatial
    fss,
    # Categorical
    ets, pod, far, csi, hss, pss, gss, fb, hkd, orss, seds, eds, sedi,
    # Probabilistic
    bss, rpss,
)

# Legacy class (deprecated)
from .nwpeval import NWP_Stats

__all__ = [
    # New API
    'confusion_matrix',
    'mae', 'rmse', 'acc', 'r2', 'nrmse', 'pcc', 'mbd',
    'fss',
    'ets', 'pod', 'far', 'csi', 'hss', 'pss', 'gss', 'fb', 'hkd', 'orss', 'seds', 'eds', 'sedi',
    'bss', 'rpss',
    # Legacy
    'NWP_Stats',
]