"""
NWPeval Metrics Module

All metrics are available as standalone functions:
    from nwpeval import rmse, mae, fss, pod, etc.
"""

# Base utilities
from ._base import confusion_matrix

# Continuous metrics
from .mae import mae
from .rmse import rmse
from .acc import acc
from .r2 import r2
from .nrmse import nrmse
from .pcc import pcc
from .mbd import mbd

# Spatial metrics
from .fss import fss

# Categorical/binary metrics
from .ets import ets
from .pod import pod
from .far import far
from .csi import csi
from .hss import hss
from .pss import pss
from .gss import gss
from .fb import fb
from .hkd import hkd
from .orss import orss
from .seds import seds
from .eds import eds
from .sedi import sedi

# Probabilistic metrics
from .bss import bss
from .rpss import rpss

__all__ = [
    # Utilities
    'confusion_matrix',
    # Continuous
    'mae', 'rmse', 'acc', 'r2', 'nrmse', 'pcc', 'mbd',
    # Spatial
    'fss',
    # Categorical
    'ets', 'pod', 'far', 'csi', 'hss', 'pss', 'gss', 'fb', 'hkd', 'orss', 'seds', 'eds', 'sedi',
    # Probabilistic
    'bss', 'rpss',
]
