import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import itertools
from matplotlib.lines import Line2D
from collections.abc import Hashable, Iterable
import xskillscore as xs


markers_list = [mk for mk in list(Line2D.markers.keys()) if mk not in ['.', ',', '']]


def _get_rad_deg_pts():
    rad_pts = np.arccos([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,1.0])
    deg_pts = np.rad2deg(rad_pts)
    return rad_pts, deg_pts

def _set_axes(ax, rmax=1, ref_sd=1):
    rad_pts, deg_pts = _get_rad_deg_pts()
    ax.plot(np.arccos([0.8, 0.8]), [0, rmax], ls='--', c='darkred', lw=0.8)
    ax.plot(np.arccos([0.6, 0.6]), [0, rmax], ls='--', c='darkred', lw=0.8)
    ax.plot(np.linspace(0, np.pi/2), np.zeros_like(np.linspace(0, np.pi/2)) + ref_sd, '--k')

    ax.set_thetagrids(deg_pts, np.round(np.cos(rad_pts), 2))
    ax.grid(True, alpha=0.4)

    trans, _, _ = ax.get_xaxis_text1_transform(-10)
    ax.text(np.deg2rad(45), -0.1, "Correlation", transform=trans, rotation=45-90, ha="center", va="center")
    ax.set_ylabel('Standardised Standard Deviations', labelpad=30)
    ax.tick_params(labelleft=True, labelright=True, labeltop=False, labelbottom=True)
    ax.set_thetalim(0, np.pi/2)
    return rad_pts, deg_pts

def set_rmax(ax, rticks=None, rmax=1, nlev=5, ref_sd=1):
    """
    Configure radial limits, draw RMS contours and tick marks.
    Returns a dict with 'contours' and a few helper arrays if needed.
    """
    
    rad_pts, deg_pts = _set_axes(ax, rmax=rmax, ref_sd=ref_sd)

    deg_last = np.rad2deg(np.arccos([0.90, 0.95, 0.99, 1.0]))
    p9095 = list(np.cos(np.deg2rad(np.linspace(deg_last[0], deg_last[1], 5))))
    p9599 = list(np.cos(np.deg2rad(np.linspace(deg_last[1], deg_last[2], 5))))
    p99100 = list(np.cos(np.deg2rad(np.linspace(deg_last[2], deg_last[3], 5))))

    for t in np.arccos([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.90] + p9095 + p9599 + p99100):
        ax.plot([t, t], [rmax, rmax * 0.99], '-k', lw=1)
    for t in rad_pts:
        ax.plot([t, t], [rmax, rmax * 0.97], '-k', lw=1)

    rs, ts = np.meshgrid(np.linspace(0, rmax), np.linspace(0, np.pi/2))
    rms = np.sqrt(1 + rs**2 - 2 * rs * np.cos(ts))
    contours = ax.contour(ts, rs, rms, levels=nlev, colors='grey', linewidths=0.6, alpha=0.6)

    fmt = {}
    for idx, l in enumerate(contours.levels):
        fmt[l] = (str(round(l, 2)) + ' RMS') if idx == 3 else round(l, 2)
    ax.clabel(contours, contours.levels, inline=True, fmt=fmt, colors='k')

    # Compute 'nice' radial tick locations and rounded labels
    # def _nice_ticks(maxval, nticks=4):
    #     if maxval <= 0:
    #         return np.array([0.0])
    #     raw_step = float(maxval) / float(nticks)
    #     mag = 10 ** np.floor(np.log10(raw_step))
    #     nice_frac = np.array([1.0, 2.0, 2.5, 5.0, 10.0])
    #     frac = raw_step / mag
    #     idx = np.searchsorted(nice_frac, frac, side='left')
    #     if idx >= len(nice_frac):
    #         idx = len(nice_frac) - 1
    #     step = nice_frac[idx] * mag
    #     ticks = np.arange(0.0, maxval + 0.5 * step, step)
    #     if ticks[-1] < maxval:
    #         ticks = np.append(ticks, ticks[-1] + step)
    #     # ensure 0 is present
    #     if ticks[0] != 0:
    #         ticks = np.insert(ticks, 0, 0.0)
    #     return np.unique(np.round(ticks, 12))
    
    
    # def _nice_ticks(maxval):
        # if (maxval%0.20) == 0:
        #     return np.arange(0, maxval+0.1, 0.20)
        # elif (maxval%0.5) == 0:
        #     return np.arange(0, maxval+0.1, 0.50)
        # else:
        #     return np.arange(0, maxval+0.1, 0.10)
    if rticks is None:
        ticks = np.arange(0, rmax, 0.2)
    else:
        ticks=rticks
    # If 1.0 lies within the axis range, ensure it's present so we can label it 'REF'
    if (1.0 < rmax) and not np.any(np.isclose(ticks, 1.0)):
        ticks = np.sort(np.append(ticks, 1.0))

    # Choose a formatting precision based on the smallest tick spacing
    if ticks.size > 1:
        spacing = np.min(np.diff(ticks))
    else:
        spacing = ticks[0] if ticks[0] != 0 else 1.0
    if spacing <= 0:
        decimals = 2
    else:
        decimals = max(0, int(-np.floor(np.log10(spacing))))

    labels = []
    for t in ticks:
        if np.isclose(t, 1.0):
            labels.append('REF')
        else:
            # format using the computed decimals, but strip trailing zeros
            fmt_str = ('{:.' + str(decimals) + 'f}').format(t)
            # remove unnecessary .0
            if '.' in fmt_str:
                fmt_str = fmt_str.rstrip('0').rstrip('.')
            labels.append(fmt_str)

    ax.set_rgrids(ticks, labels)
    # ax.set_rmax(rmax)
    
        
    return 
    # return dict(contours=contours, rad_pts=rad_pts, deg_pts=deg_pts, deg_last=deg_last,
    #             p9095=p9095, p9599=p9599, p99100=p99100)

def add_taylor_point(ax, corr_data, std_data, rmax=1, legend=False, label=None, **kwargs):
    """
    Plot a single Taylor point on `ax` (polar).
    """
    _set_axes(ax, rmax=rmax)
    theta_ = np.arccos(corr_data)
    ax.plot(theta_, std_data, ls='none', ms=10, mfc='none', alpha=1,
            markeredgewidth=1.4, label=label, **kwargs)
    if legend:
        ax.legend()
    return ax

def taylor_helper(ax, model_names=None, model_data=None, ref_data=None,
                  corr_data=None, std_data=None, rmax=1, c_arr=None, marker_arr=None, **kwargs):
    """
    Main helper that accepts either:
      - `corr_data` and `std_data` (arrays of points), or
      - `model_data` (list-like) and `ref_data` (DataArray/array) to compute corr/std from data.
    It plots the points onto `ax`.
    """
    # If explicit corr/std provided
    if model_data is None and ref_data is None:
        # expect arrays or lists
        if corr_data is None or std_data is None:
            raise ValueError("Either provide model_data+ref_data or corr_data+std_data.")
        # assume corresponding arrays of same length
        for i, (c, s) in enumerate(zip(corr_data, std_data)):
            add_taylor_point(ax, c, s, rmax=rmax, label=(None if not model_names else model_names[i]),
                             marker=(None if marker_arr is None else marker_arr[i]),
                             color=(None if c_arr is None else c_arr[i]), **kwargs)
        return ax

    # Otherwise compute correlation and normalized std from model_data vs ref_data
    if isinstance(model_data, list) or isinstance(model_data, (np.ndarray, Iterable)):
        if c_arr is None:
            clrs = ['#8800b0', '#169057', '#006cff', 'r', 'violet']
        else:
            clrs = c_arr

        for idx, m_data in enumerate(model_data):
            corr = xs.pearson_r(m_data, ref_data)
            std = m_data.std() / ref_data.std()
            mark = markers_list[idx] if marker_arr is None else marker_arr[idx]
            add_taylor_point(ax, corr, std, rmax=rmax, marker=mark,
                             color=(clrs[idx] if any(clrs) else None),
                             label=(None if model_names is None else model_names[idx]), **kwargs)
        return ax

    # Fallback: single model_data (not list)
    corr = xs.pearson_r(model_data, ref_data)
    std = model_data.std() / ref_data.std()
    add_taylor_point(ax, corr, std, rmax=rmax, label=(model_names[0] if model_names else None), **kwargs)
    return ax

def draw_taylor(ax, model_names=None, model_data=None, ref_data=None,
                corr_data=None, std_data=None, rticks=None, rmax=1, set_rmax_nlev=5,
                c_arr=None, marker_arr=None, ref_sd=1, legend=False, **kwargs):
    """
    High level function: prepares axes, plots points and configures rmax/contours.
    Returns the axis and `extras` dict from `set_rmax`.
    """
    if rticks is not None:
        rmax = max(rticks)
    _set_axes(ax, rmax=rmax, ref_sd=ref_sd)
    taylor_helper(ax, model_names=model_names, model_data=model_data, ref_data=ref_data,
                  corr_data=corr_data, std_data=std_data, rmax=rmax, c_arr=c_arr, marker_arr=marker_arr,
                  legend=legend, **kwargs)
    
    set_rmax(ax, rticks=rticks, rmax=rmax, nlev=set_rmax_nlev, ref_sd=ref_sd)
    
    return ax





