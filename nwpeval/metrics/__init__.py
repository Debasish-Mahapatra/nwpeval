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
from .tse import tse
from .evs import evs
from .nmse import nmse
from .fv import fv
from .sdr import sdr
from .vif import vif
from .mad import mad
from .iqr import iqr
from .nae import nae
from .rmb import rmb
from .mape import mape
from .wmae import wmae
from .ass import ass
from .rss import rss
from .qss import qss
from .lmbe import lmbe
from .smse import smse
from .gmb import gmb
from .sbs import sbs
from .aev import aev
from .cosine_similarity import cosine_similarity

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
from .f1 import f1
from .mcc import mcc
from .ba import ba
from .npv import npv
from .jaccard import jaccard
from .gain import gain
from .lift import lift

# Probabilistic metrics
from .bss import bss
from .rpss import rpss

# Distributional metrics
from .mkldiv import mkldiv
from .jsdiv import jsdiv
from .hellinger import hellinger
from .wasserstein import wasserstein
from .tv import tv
from .chisquare import chisquare
from .intersection import intersection
from .bhattacharyya import bhattacharyya
from .chernoff import chernoff
from .renyi import renyi
from .tsallis import tsallis

# Mean metrics
from .harmonic_mean import harmonic_mean
from .geometric_mean import geometric_mean
from .lehmer_mean import lehmer_mean

__all__ = [
    # Utilities
    'confusion_matrix',
    # Continuous
    'mae', 'rmse', 'acc', 'r2', 'nrmse', 'pcc', 'mbd',
    'tse', 'evs', 'nmse', 'fv', 'sdr', 'vif', 'mad', 'iqr',
    'nae', 'rmb', 'mape', 'wmae', 'ass', 'rss', 'qss',
    'lmbe', 'smse', 'gmb', 'sbs', 'aev', 'cosine_similarity',
    # Spatial
    'fss',
    # Categorical
    'ets', 'pod', 'far', 'csi', 'hss', 'pss', 'gss', 'fb', 'hkd', 'orss',
    'seds', 'eds', 'sedi', 'f1', 'mcc', 'ba', 'npv', 'jaccard', 'gain', 'lift',
    # Probabilistic
    'bss', 'rpss',
    # Distributional
    'mkldiv', 'jsdiv', 'hellinger', 'wasserstein', 'tv', 'chisquare',
    'intersection', 'bhattacharyya', 'chernoff', 'renyi', 'tsallis',
    # Means
    'harmonic_mean', 'geometric_mean', 'lehmer_mean',
]
