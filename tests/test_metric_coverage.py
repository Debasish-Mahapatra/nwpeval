"""Extended coverage for nwpeval: per-dimension results, parameters, legacy class,
extra dimensions, dask, element-wise means and FSS reduction_dim.

Every reduction is checked slice by slice against an independent numpy reference
on the jointly valid pairs of that slice.
"""
import warnings

import numpy as np
import pytest
import xarray as xr

import nwpeval as nw

warnings.filterwarnings("ignore")

NT, NY, NX = 30, 8, 9
DIMS = ("time", "y", "x")
COORDS = {"time": np.arange(NT), "y": np.arange(NY) * 0.1, "x": np.arange(NX) * 0.1}
T = 2.5
rng = np.random.default_rng(7)
OBS = rng.gamma(2.0, 1.0, (NT, NY, NX)) + 0.05
MODEL = 0.8 * OBS + rng.gamma(1.5, 0.6, (NT, NY, NX))
W = rng.uniform(0.2, 2.0, (NT, NY, NX))
REF = OBS.mean() + rng.normal(0, 1, (NT, NY, NX))
PROB = rng.uniform(0, 1, (NT, NY, NX))
OBS_EVENT = (rng.random((NT, NY, NX)) < PROB).astype(float)
OBS_GAPS = np.zeros((NT, NY, NX), bool)
OBS_GAPS[:, :, :2] = True                      # whole cells missing (empty slices)
OBS_GAPS |= rng.random((NT, NY, NX)) < 0.06
MODEL_GAPS = rng.random((NT, NY, NX)) < 0.08


def da(a):
    return xr.DataArray(a, dims=DIMS, coords=COORDS)


def gappy(case, o, m):
    o, m = o.copy(), m.copy()
    if case in ("obs gaps", "both gaps"):
        o[OBS_GAPS] = np.nan
    if case in ("model gaps", "both gaps"):
        m[MODEL_GAPS] = np.nan
    return o, m


CASES = ["clean", "obs gaps", "model gaps", "both gaps"]
REDUCTIONS = {"time": ("y", "x"), ("y", "x"): ("time",), ("time", "y", "x"): ()}


def slice_indices(dim):
    if dim == "time":
        return [(slice(None), i, j) for i in range(NY) for j in range(NX)]
    if dim == ("y", "x"):
        return [(t, slice(None), slice(None)) for t in range(NT)]
    return [(slice(None), slice(None), slice(None))]


def result_at(result, dim, k):
    keep = REDUCTIONS[dim]
    vals = result.transpose(*keep).values if keep else result.values
    if dim == "time":
        return float(vals[k // NX, k % NX])
    if dim == ("y", "x"):
        return float(vals[k])
    return float(vals)


# ------------------------------------------------------------------ references (1-D)
def ref_continuous(name, o, m, w, r):
    e = m - o
    mse = np.mean(e**2)
    evs = 1 - np.var(o - m) / np.var(o)
    c = o.mean()
    return {
        "mae": lambda: np.mean(np.abs(e)),
        "rmse": lambda: np.sqrt(mse),
        "acc": lambda: np.sum((o - c) * (m - c)) / np.sqrt(np.sum((o - c) ** 2) * np.sum((m - c) ** 2)),
        "r2": lambda: 1 - np.sum(e**2) / np.sum((o - o.mean()) ** 2),
        "nrmse": lambda: np.sqrt(mse) / o.mean(),
        "pcc": lambda: np.corrcoef(o, m)[0, 1],
        "mbd": lambda: m.mean() - o.mean(),
        "tse": lambda: np.sum(e**2),
        "evs": lambda: evs,
        "nmse": lambda: mse / o.mean() ** 2,
        "fv": lambda: np.var(m) / np.var(o),
        "sdr": lambda: np.std(m) / np.std(o),
        "vif": lambda: np.var(m) / np.var(o) - 1,
        "mad": lambda: np.median(np.abs(e - np.median(e))),
        "iqr": lambda: np.percentile(e, 75) - np.percentile(e, 25),
        "nae": lambda: np.sum(np.abs(e)) / np.sum(np.abs(o)),
        "rmb": lambda: np.sum(e) / np.sum(o),
        "mape": lambda: 100 * np.mean(np.abs(e) / np.abs(o)),
        "wmae": lambda: np.sum(w * np.abs(e)) / np.sum(w),
        "ass": lambda: 1 - np.mean(np.abs(e)) / 0.9,
        "rss": lambda: ((1 - np.mean(np.abs(e) / np.abs(o))) - 0.2) / 0.8,
        "qss": lambda: 1 - mse / np.mean((r - o) ** 2),
        "lmbe": lambda: np.mean(np.log1p(m) - np.log1p(o)),
        "smse": lambda: mse / np.var(o),
        "gmb": lambda: np.exp(np.mean(np.log(m))) / np.exp(np.mean(np.log(o))),
        "aev": lambda: 1 - (1 - evs) * (o.size - 1) / (o.size - 2) if o.size > 2 else np.nan,
        "cosine_similarity": lambda: np.sum(m * o) / np.sqrt(np.sum(m**2) * np.sum(o**2)),
    }[name]()


def ref_categorical(name, o, m):
    a = np.float64(np.sum((o >= T) & (m >= T))); b = np.float64(np.sum((o < T) & (m >= T)))
    c = np.float64(np.sum((o >= T) & (m < T))); d = np.float64(np.sum((o < T) & (m < T)))
    n = a + b + c + d
    h, f, p, q = a / (a + c), b / (b + d), (a + c) / n, (a + b) / n
    hr = (a + b) * (a + c) / n
    ets = (a - hr) / (a + b + c - hr)
    return {
        "pod": h, "far": b / (a + b), "csi": a / (a + b + c), "jaccard": a / (a + b + c),
        "fb": (a + b) / (a + c), "ets": ets, "gss": ets,
        "hss": 2 * (a * d - b * c) / ((a + c) * (c + d) + (a + b) * (b + d)),
        "pss": h - f, "hkd": h - f, "orss": (a * d - b * c) / (a * d + b * c),
        "seds": (np.log(p) + np.log(q)) / np.log(a / n) - 1,
        "eds": 2 * np.log(p) / np.log(p * h) - 1,
        "sedi": (np.log(f) - np.log(h) - np.log(1 - f) + np.log(1 - h))
        / (np.log(f) + np.log(h) + np.log(1 - f) + np.log(1 - h)),
        "f1": 2 * a / (2 * a + b + c),
        "mcc": (a * d - b * c) / np.sqrt((a + b) * (a + c) * (d + b) * (d + c)),
        "ba": 0.5 * (h + d / (d + b)), "npv": d / (d + c), "gain": (a + d) / n,
        "lift": (a / (a + b)) / p,
    }[name]


def ref_distributional(name, o, m, alpha=0.3):
    p, q = o / o.sum(), m / m.sum()
    mid = 0.5 * (p + q)
    inner = np.sum(p**alpha * q ** (1 - alpha))
    return {
        "mkldiv": lambda: np.sum(p * np.log(p / q)),
        "jsdiv": lambda: 0.5 * np.sum(p * np.log(p / mid)) + 0.5 * np.sum(q * np.log(q / mid)),
        "hellinger": lambda: np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)),
        "tv": lambda: 0.5 * np.sum(np.abs(p - q)),
        "chisquare": lambda: np.sum((p - q) ** 2 / q),
        "intersection": lambda: np.sum(np.minimum(p, q)),
        "bhattacharyya": lambda: -np.log(np.sum(np.sqrt(p * q))),
        "chernoff": lambda: -np.log(inner),
        "renyi": lambda: np.log(inner) / (alpha - 1),
        "tsallis": lambda: (inner - 1) / (alpha - 1),
        "wasserstein": lambda: np.mean(np.abs(np.sort(o) - np.sort(m))),
    }[name]()


def ref_probabilistic(name, o, m):
    ob = (o >= T).astype(float)
    climo = np.mean((ob.mean() - ob) ** 2)
    if name == "bss":
        return 1 - np.mean((m - ob) ** 2) / climo
    return 1 - np.mean(((m >= T).astype(float) - ob) ** 2) / climo


CONTINUOUS = ["mae", "rmse", "acc", "r2", "nrmse", "pcc", "mbd", "tse", "evs", "nmse", "fv",
              "sdr", "vif", "mad", "iqr", "nae", "rmb", "mape", "wmae", "ass", "rss", "qss",
              "lmbe", "smse", "gmb", "aev", "cosine_similarity"]
CATEGORICAL = ["pod", "far", "csi", "fb", "ets", "gss", "hss", "pss", "hkd", "orss", "seds",
               "eds", "sedi", "f1", "mcc", "ba", "npv", "jaccard", "gain", "lift"]
DISTRIBUTIONAL = ["mkldiv", "jsdiv", "hellinger", "tv", "chisquare", "intersection",
                  "bhattacharyya", "chernoff", "renyi", "tsallis", "wasserstein"]
EXTREMAL = {"sedi", "eds", "seds"}


def reference(name, o, m, w, r):
    """Reference on the jointly valid pairs of one slice; NaN when undefined."""
    ok = np.isfinite(o) & np.isfinite(m) & np.isfinite(w) & np.isfinite(r)
    o, m, w, r = o[ok], m[ok], w[ok], r[ok]
    if o.size == 0:
        return np.nan
    with np.errstate(all="ignore"):
        if name in CONTINUOUS or name == "sbs":
            v = 2 * np.mean((m - o) ** 2) if name == "sbs" else ref_continuous(name, o, m, w, r)
        elif name in CATEGORICAL:
            v = ref_categorical(name, o, m)
        elif name in ("bss", "rpss"):
            v = ref_probabilistic(name, o, m)
        else:
            v = ref_distributional(name, o, m)
    v = float(v)
    if name not in DISTRIBUTIONAL and np.isinf(v):
        v = np.nan  # a ratio with a zero denominator is undefined
    return v


def call(name, o, m, dim):
    f = getattr(nw, name)
    if name in CATEGORICAL or name in ("bss", "rpss"):
        return f(o, m, T, dim=dim)
    if name in ("chernoff", "renyi", "tsallis"):
        return f(o, m, 0.3, dim=dim)
    extra = {"wmae": (da(W),), "ass": (0.9,), "rss": (0.2,), "qss": (da(REF),)}
    return f(o, m, *extra.get(name, ()), dim=dim)


def inputs_for(name, case):
    if name == "sbs":
        return gappy(case, OBS_EVENT, PROB)
    if name == "bss":
        return gappy(case, OBS, PROB)
    return gappy(case, OBS, MODEL)


def agrees(name, got, want):
    if np.isfinite(want):
        return np.isfinite(got) and abs(got - want) <= 1e-9 * max(1.0, abs(want))
    if np.isnan(got):
        return True
    if name in EXTREMAL and got in (-1.0, 1.0):
        return True  # documented limit where the plain formula has log(0)
    return np.isinf(want) and got == want


ALL_REFERENCED = CONTINUOUS + ["sbs"] + CATEGORICAL + ["bss", "rpss"] + DISTRIBUTIONAL


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("dim", list(REDUCTIONS), ids=["time", "space", "all"])
@pytest.mark.parametrize("name", ALL_REFERENCED)
def test_every_slice_matches_reference(name, dim, case):
    o, m = inputs_for(name, case)
    d = list(dim) if isinstance(dim, tuple) else dim
    result = call(name, da(o), da(m), d)
    assert set(result.dims) == set(REDUCTIONS[dim])
    bad = []
    for k, idx in enumerate(slice_indices(dim)):
        want = reference(name, o[idx].ravel(), m[idx].ravel(), W[idx].ravel(), REF[idx].ravel())
        got = result_at(result, dim, k)
        if not agrees(name, got, want):
            bad.append((idx, got, want))
    assert not bad, f"{len(bad)} slices differ, first: {bad[0]}"


# ------------------------------------------------------------------ parameters
@pytest.mark.parametrize("alpha", [0.1, 0.3, 0.5, 0.9, 1.5, 2.0])
@pytest.mark.parametrize("name", ["chernoff", "renyi", "tsallis"])
def test_alpha_values(name, alpha):
    if name == "chernoff" and alpha >= 1:
        pytest.skip("Chernoff is defined for 0 < alpha < 1")
    o, m = gappy("both gaps", OBS, MODEL)
    ok = np.isfinite(o) & np.isfinite(m)
    want = ref_distributional(name, o[ok], m[ok], alpha)
    got = float(getattr(nw, name)(da(o), da(m), alpha))
    assert got == pytest.approx(want, rel=1e-9)


@pytest.mark.parametrize("name", ["renyi", "tsallis"])
def test_alpha_one_raises(name):
    with pytest.raises(ValueError):
        getattr(nw, name)(da(OBS), da(MODEL), 1)


@pytest.mark.parametrize("p", [0, 1, 2, 3, 0.5])
def test_lehmer_mean(p):
    o, m = gappy("both gaps", OBS, MODEL)
    got = nw.lehmer_mean(da(o), da(m), p).values
    want = (o**p + m**p) / (o ** (p - 1) + m ** (p - 1))
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_harmonic_and_geometric_means():
    o, m = gappy("both gaps", OBS, MODEL)
    np.testing.assert_allclose(nw.harmonic_mean(da(o), da(m)).values, 2 * o * m / (o + m), rtol=1e-12)
    np.testing.assert_allclose(nw.geometric_mean(da(o), da(m)).values, np.sqrt(o * m), rtol=1e-12)
    zero = da(np.zeros((NT, NY, NX)))
    assert np.isnan(nw.harmonic_mean(zero, zero).values).all()
    neg = nw.geometric_mean(da(-OBS), da(MODEL))
    assert np.isnan(neg.values).all()


@pytest.mark.parametrize("k", [1, 2, 5])
@pytest.mark.parametrize("dim", ["time", ["y", "x"]])
def test_aev_predictors_per_slice(k, dim):
    o, m = gappy("both gaps", OBS, MODEL)
    result = nw.aev(da(o), da(m), dim=dim, n_predictors=k)
    key = "time" if dim == "time" else ("y", "x")
    for s, idx in enumerate(slice_indices(key)):
        oo, mm = o[idx].ravel(), m[idx].ravel()
        ok = np.isfinite(oo) & np.isfinite(mm)
        oo, mm = oo[ok], mm[ok]
        n = oo.size
        want = np.nan if n - k - 1 <= 0 or n == 0 else 1 - (np.var(oo - mm) / np.var(oo)) * (n - 1) / (n - k - 1)
        assert agrees("aev", result_at(result, key, s), want)


def test_ass_per_element_reference_per_slice():
    o, m = gappy("both gaps", OBS, MODEL)
    ref_err = np.abs(REF - OBS)
    result = nw.ass(da(o), da(m), da(ref_err), dim="time")
    for s, idx in enumerate(slice_indices("time")):
        oo, mm, rr = o[idx], m[idx], ref_err[idx]
        ok = np.isfinite(oo) & np.isfinite(mm)
        want = 1 - np.mean(np.abs(mm[ok] - oo[ok])) / np.mean(rr[ok]) if ok.any() else np.nan
        assert agrees("ass", result_at(result, "time", s), want)


def test_acc_with_climatology_per_cell():
    o, m = gappy("both gaps", OBS, MODEL)
    clim = xr.DataArray(OBS.mean(axis=0) * 0.9, dims=("y", "x"), coords={"y": COORDS["y"], "x": COORDS["x"]})
    result = nw.acc(da(o), da(m), climatology=clim, dim="time")
    for s, idx in enumerate(slice_indices("time")):
        oo, mm = o[idx], m[idx]
        ok = np.isfinite(oo) & np.isfinite(mm)
        c = clim.values[idx[1], idx[2]]
        a, b = oo[ok] - c, mm[ok] - c
        want = np.sum(a * b) / np.sqrt(np.sum(a**2) * np.sum(b**2)) if ok.any() else np.nan
        assert agrees("acc", result_at(result, "time", s), want)


# ------------------------------------------------------------------ FSS reduction_dim
def fss_parts(o, m, size):
    """Per-point (O-M)^2 and O^2+M^2, fractions over valid neighbours (loop reference)."""
    half = size // 2
    num = np.full(o.shape, np.nan); den = np.full(o.shape, np.nan)
    for t in range(o.shape[0]):
        ok = np.isfinite(o[t]) & np.isfinite(m[t])
        ob = np.where(ok, o[t] >= T, 0.0); mb = np.where(ok, m[t] >= T, 0.0)
        for i in range(o.shape[1]):
            for j in range(o.shape[2]):
                if not ok[i, j]:
                    continue
                win = (slice(max(i - half, 0), i + half + 1), slice(max(j - half, 0), j + half + 1))
                n = ok[win].sum()
                fo, fm = ob[win].sum() / n, mb[win].sum() / n
                num[t, i, j] = (fo - fm) ** 2
                den[t, i, j] = fo**2 + fm**2
    return num, den


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("size", [1, 3, 7])
@pytest.mark.parametrize("red", ["time", ("y", "x")], ids=["time", "space"])
def test_fss_reduction_dim(red, size, case):
    o, m = gappy(case, OBS, MODEL)
    num, den = fss_parts(o, m, size)
    result = nw.fss(da(o), da(m), T, size, spatial_dims=["y", "x"],
                    reduction_dim=list(red) if isinstance(red, tuple) else red)
    for s, idx in enumerate(slice_indices(red)):
        nn, dd = num[idx], den[idx]
        ok = np.isfinite(nn)
        want = 1 - nn[ok].mean() / dd[ok].mean() if ok.any() and dd[ok].mean() > 0 else np.nan
        assert agrees("fss", result_at(result, red, s), want)


# ------------------------------------------------------------------ all 65 through one table
def all_calls():
    def c(name):
        f = getattr(nw, name)
        if name in CATEGORICAL or name in ("bss", "rpss"):
            return lambda o, m, dim=None: f(o, m, 1.0, dim=dim)
        if name in ("chernoff", "renyi", "tsallis"):
            return lambda o, m, dim=None: f(o, m, 0.4, dim=dim)
        if name == "lehmer_mean":
            return lambda o, m, dim=None: f(o, m, 2, dim=dim)
        if name == "wmae":
            return lambda o, m, dim=None: f(o, m, xr.ones_like(o) * 2, dim=dim)
        if name == "ass":
            return lambda o, m, dim=None: f(o, m, 0.9, dim=dim)
        if name == "rss":
            return lambda o, m, dim=None: f(o, m, 0.2, dim=dim)
        if name == "qss":
            return lambda o, m, dim=None: f(o, m, o * 0 + 1.3, dim=dim)
        if name == "fss":
            return lambda o, m, dim=None: f(o, m, 1.0, 3, spatial_dims=["y", "x"], reduction_dim=dim)
        return lambda o, m, dim=None: f(o, m, dim=dim)
    return {name: c(name) for name in set(nw.__all__) - {"confusion_matrix", "NWP_Stats"}}


CALLS = all_calls()
ELEMENTWISE = {"harmonic_mean", "geometric_mean", "lehmer_mean"}


@pytest.mark.parametrize("name", sorted(CALLS))
def test_extra_member_dimension(name):
    """A model with an ensemble dimension scores each member as if passed alone."""
    o, m = gappy("both gaps", OBS, MODEL)
    obs = da(o) / 2
    ens = xr.concat([da(m) / 2, da(m * 1.3) / 2], dim="member")
    dims = None if name in ELEMENTWISE else list(DIMS)
    together = CALLS[name](obs, ens, dims)
    for k in range(2):
        alone = CALLS[name](obs, ens.isel(member=k), dims)
        xr.testing.assert_allclose(together.isel(member=k, drop=True).transpose(*alone.dims),
                                   alone.reset_coords(drop=True) if hasattr(alone, "reset_coords") else alone)


@pytest.mark.parametrize("name", sorted(CALLS))
def test_dask_matches_numpy(name):
    if name == "mad":
        pytest.xfail("dask cannot take a median over all dimensions at once (dim=None)")
    o, m = gappy("both gaps", OBS, MODEL)
    eager = CALLS[name](da(o) / 2, da(m) / 2)
    lazy = CALLS[name](da(o).chunk({"time": 7}) / 2, da(m).chunk({"time": 7}) / 2)
    xr.testing.assert_allclose(eager, lazy.compute())


# ------------------------------------------------------------------ legacy class
LEGACY = {
    "MAE": ("mae", ()), "RMSE": ("rmse", ()), "ACC": ("acc", ()), "TSE": ("tse", ()),
    "EVS": ("evs", ()), "NMSE": ("nmse", ()), "FV": ("fv", ()), "PCC": ("pcc", ()),
    "SDR": ("sdr", ()), "VIF": ("vif", ()), "MAD": ("mad", ()), "IQR": ("iqr", ()),
    "R2": ("r2", ()), "NAE": ("nae", ()), "RMB": ("rmb", ()), "MAPE": ("mape", ()),
    "NRMSE": ("nrmse", ()), "LMBE": ("lmbe", ()), "SMSE": ("smse", ()), "MBD": ("mbd", ()),
    "GMB": ("gmb", ()), "SBS": ("sbs", ()), "AEV": ("aev", ()),
    "CosineSimilarity": ("cosine_similarity", ()),
    "MKLDIV": ("mkldiv", ()), "JSDIV": ("jsdiv", ()), "Hellinger": ("hellinger", ()),
    "Wasserstein": ("wasserstein", ()), "TV": ("tv", ()), "ChiSquare": ("chisquare", ()),
    "Intersection": ("intersection", ()), "Bhattacharyya": ("bhattacharyya", ()),
    "HarmonicMean": ("harmonic_mean", ()), "GeometricMean": ("geometric_mean", ()),
}
LEGACY_THRESHOLD = {"ETS": "ets", "POD": "pod", "FAR": "far", "CSI": "csi", "BSS": "bss",
                    "HSS": "hss", "PSS": "pss", "SEDS": "seds", "FB": "fb", "GSS": "gss",
                    "H-KD": "hkd", "ORSS": "orss", "EDS": "eds", "SEDI": "sedi", "RPSS": "rpss",
                    "F1": "f1", "MCC": "mcc", "BA": "ba", "NPV": "npv", "Jaccard": "jaccard",
                    "Gain": "gain", "Lift": "lift"}


def legacy_expected(key, o, m, dim):
    th = {k: 1.0 for k in LEGACY_THRESHOLD}
    if key in LEGACY:
        return getattr(nw, LEGACY[key][0])(o, m, dim=dim) if key != "ACC" else nw.acc(o, m, dim=dim)
    if key in LEGACY_THRESHOLD:
        return getattr(nw, LEGACY_THRESHOLD[key])(o, m, th[key], dim=dim)
    return {
        "FSS": lambda: nw.fss(o, m, 1.0, 3, reduction_dim=dim),
        "WMAE": lambda: nw.wmae(o, m, xr.ones_like(o), dim=dim),
        "ASS": lambda: nw.ass(o, m, 0.9, dim=dim),
        "RSS": lambda: nw.rss(o, m, 0.2, dim=dim),
        "QSS": lambda: nw.qss(o, m, o * 0 + 1.3, dim=dim),
        "LehmerMean": lambda: nw.lehmer_mean(o, m, 3, dim=dim),
        "Chernoff": lambda: nw.chernoff(o, m, 0.4, dim=dim),
        "Renyi": lambda: nw.renyi(o, m, 0.4, dim=dim),
        "Tsallis": lambda: nw.tsallis(o, m, 0.4, dim=dim),
    }[key]()


LEGACY_KEYS = sorted(list(LEGACY) + list(LEGACY_THRESHOLD) +
                     ["FSS", "WMAE", "ASS", "RSS", "QSS", "LehmerMean", "Chernoff", "Renyi", "Tsallis"])


def test_legacy_dispatcher_covers_65_names():
    assert len(LEGACY_KEYS) == 65


def test_legacy_name_list_matches_the_dispatcher():
    from nwpeval.nwpeval import METRIC_NAMES
    assert sorted(METRIC_NAMES) == sorted(LEGACY_KEYS + ["HKD"])


@pytest.mark.parametrize("dim", [None, "time"])
@pytest.mark.parametrize("key", LEGACY_KEYS)
def test_legacy_compute_metrics_matches_functions(key, dim):
    o, m = gappy("both gaps", OBS, MODEL)
    o, m = da(o) / 2, da(m) / 2
    thresholds = {k: 1.0 for k in LEGACY_THRESHOLD}
    thresholds.update({"FSS": 1.0, "FSS_neighborhood": 3, "WMAE_weights": xr.ones_like(o),
                       "ASS_reference_error": 0.9, "RSS_reference_skill": 0.2,
                       "QSS_reference_forecast": o * 0 + 1.3, "LehmerMean_p": 3,
                       "Chernoff_alpha": 0.4, "Renyi_alpha": 0.4, "Tsallis_alpha": 0.4})
    with pytest.warns(DeprecationWarning):
        stats = nw.NWP_Stats(o, m)
    if key in ("HarmonicMean", "GeometricMean", "LehmerMean") and dim is not None:
        pytest.skip("element-wise means ignore dim")
    got = stats.compute_metrics([key], dim=dim, thresholds=thresholds)[key]
    xr.testing.assert_allclose(got, legacy_expected(key, o, m, dim))
