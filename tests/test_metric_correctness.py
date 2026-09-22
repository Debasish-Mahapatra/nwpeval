"""Correctness tests for every nwpeval metric.

Each metric is checked against an independent numpy reference on clean data,
with observations missing and with model values missing. The reference always
uses the jointly valid obs/model pairs. Further tests cover published values,
undefined scores, large samples, coordinate alignment and the FSS edge rules.
"""
import warnings

import numpy as np
import pytest
import xarray as xr

import nwpeval as nw

warnings.filterwarnings("ignore", category=RuntimeWarning)

NT, NY, NX = 30, 10, 12
DIMS = ("time", "y", "x")
COORDS = {"time": np.arange(NT), "y": np.arange(NY) * 0.1, "x": np.arange(NX) * 0.1}
THRESHOLD = 2.5


def da(values):
    return xr.DataArray(values, dims=DIMS, coords=COORDS)


def blank(values, mask):
    out = values.copy()
    out[mask] = np.nan
    return out


_rng = np.random.default_rng(42)
OBS = _rng.gamma(2.0, 1.0, (NT, NY, NX)) + 0.05
MODEL = 0.8 * OBS + _rng.gamma(1.5, 0.6, (NT, NY, NX))
WEIGHTS = _rng.uniform(0.2, 2.0, (NT, NY, NX))
REFERENCE = OBS.mean() + _rng.normal(0, 1, (NT, NY, NX))
PROB = _rng.uniform(0, 1, (NT, NY, NX))
OBS_EVENT = (_rng.random((NT, NY, NX)) < PROB).astype(float)

# Missing obs: a block outside the "footprint" plus random gaps. Missing model: random gaps.
OBS_MISSING = np.zeros((NT, NY, NX), bool)
OBS_MISSING[:, :, :4] = True
OBS_MISSING |= _rng.random((NT, NY, NX)) < 0.05
MODEL_MISSING = _rng.random((NT, NY, NX)) < 0.08

CASES = ["clean", "obs missing", "model missing"]


def case_inputs(case, obs=OBS, model=MODEL):
    if case == "obs missing":
        return blank(obs, OBS_MISSING), model
    if case == "model missing":
        return obs, blank(model, MODEL_MISSING)
    return obs, model


def valid_pairs(obs, model, *extra):
    ok = np.isfinite(obs) & np.isfinite(model)
    for e in extra:
        ok &= np.isfinite(e)
    return (obs[ok], model[ok]) + tuple(e[ok] for e in extra)


def assert_close(got, want):
    got = float(got)
    if np.isnan(want):
        assert np.isnan(got)
    else:
        assert got == pytest.approx(want, rel=1e-9, abs=1e-12)


# --------------------------------------------------------------------------- continuous
def continuous_reference(name, o, m):
    o, m, w, r = valid_pairs(o, m, WEIGHTS, REFERENCE)
    e = m - o
    mse = np.mean(e**2)
    evs = 1 - np.var(o - m) / np.var(o)
    c = o.mean()
    return {
        "mae": np.mean(np.abs(e)),
        "rmse": np.sqrt(mse),
        "acc": np.sum((o - c) * (m - c)) / np.sqrt(np.sum((o - c) ** 2) * np.sum((m - c) ** 2)),
        "r2": 1 - np.sum(e**2) / np.sum((o - o.mean()) ** 2),
        "nrmse": np.sqrt(mse) / o.mean(),
        "pcc": np.corrcoef(o, m)[0, 1],
        "mbd": m.mean() - o.mean(),
        "tse": np.sum(e**2),
        "evs": evs,
        "nmse": mse / o.mean() ** 2,
        "fv": np.var(m) / np.var(o),
        "sdr": np.std(m) / np.std(o),
        "vif": np.var(m) / np.var(o) - 1,
        "mad": np.median(np.abs(e - np.median(e))),
        "iqr": np.percentile(e, 75) - np.percentile(e, 25),
        "nae": np.sum(np.abs(e)) / np.sum(np.abs(o)),
        "rmb": np.sum(e) / np.sum(o),
        "mape": 100 * np.mean(np.abs(e) / np.abs(o)),
        "wmae": np.sum(w * np.abs(e)) / np.sum(w),
        "ass": 1 - np.mean(np.abs(e)) / 0.9,
        "rss": ((1 - np.mean(np.abs(e) / np.abs(o))) - 0.2) / 0.8,
        "qss": 1 - mse / np.mean((r - o) ** 2),
        "lmbe": np.mean(np.log1p(m) - np.log1p(o)),
        "smse": mse / np.var(o),
        "gmb": np.exp(np.mean(np.log(m))) / np.exp(np.mean(np.log(o))),
        "aev": 1 - (1 - evs) * (o.size - 1) / (o.size - 2),
        "cosine_similarity": np.sum(m * o) / np.sqrt(np.sum(m**2) * np.sum(o**2)),
    }[name]


def call_continuous(name, o, m):
    f = getattr(nw, name)
    extra = {"wmae": (da(WEIGHTS),), "ass": (0.9,), "rss": (0.2,), "qss": (da(REFERENCE),)}
    return f(o, m, *extra.get(name, ()))


CONTINUOUS = ["mae", "rmse", "acc", "r2", "nrmse", "pcc", "mbd", "tse", "evs", "nmse", "fv",
              "sdr", "vif", "mad", "iqr", "nae", "rmb", "mape", "wmae", "ass", "rss", "qss",
              "lmbe", "smse", "gmb", "aev", "cosine_similarity"]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("name", CONTINUOUS)
def test_continuous_matches_reference(name, case):
    o, m = case_inputs(case)
    assert_close(call_continuous(name, da(o), da(m)), continuous_reference(name, o, m))


@pytest.mark.parametrize("case", CASES)
def test_sbs_matches_reference(case):
    o, p = case_inputs(case, OBS_EVENT, PROB)
    oo, pp = valid_pairs(o, p)
    assert_close(nw.sbs(da(o), da(p)), 2 * np.mean((pp - oo) ** 2))


# --------------------------------------------------------------------------- categorical
def table(o, m, t=THRESHOLD):
    o, m = valid_pairs(o, m)
    return tuple(float(np.sum(x)) for x in (
        (o >= t) & (m >= t), (o < t) & (m >= t), (o >= t) & (m < t), (o < t) & (m < t)))


def categorical_reference(name, a, b, c, d):
    n = a + b + c + d
    h, f, p, q = a / (a + c), b / (b + d), (a + c) / n, (a + b) / n
    hits_random = (a + b) * (a + c) / n
    ets = (a - hits_random) / (a + b + c - hits_random)
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


CATEGORICAL = ["pod", "far", "csi", "fb", "ets", "gss", "hss", "pss", "hkd", "orss", "seds",
               "eds", "sedi", "f1", "mcc", "ba", "npv", "jaccard", "gain", "lift"]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("name", CATEGORICAL)
def test_categorical_matches_reference(name, case):
    o, m = case_inputs(case)
    got = getattr(nw, name)(da(o), da(m), THRESHOLD)
    assert_close(got, categorical_reference(name, *table(o, m)))


def finley_arrays(extra_missing=0):
    """Finley (1884) tornado table: 28 hits, 72 false alarms, 23 misses, 2680 correct negatives."""
    obs = np.r_[np.ones(28), np.zeros(72), np.ones(23), np.zeros(2680)]
    model = np.r_[np.ones(28), np.ones(72), np.zeros(23), np.zeros(2680)]
    if extra_missing:
        obs = np.r_[obs, np.full(extra_missing, np.nan)]
        model = np.r_[model, np.random.default_rng(0).integers(0, 2, extra_missing)]
    return xr.DataArray(obs, dims="t"), xr.DataArray(model.astype(float), dims="t")


# Published values for the Finley table (Wilks, Statistical Methods in the Atmospheric Sciences).
FINLEY = {"pod": 0.549, "far": 0.720, "csi": 0.228, "fb": 1.961, "pss": 0.523, "hkd": 0.523,
          "hss": 0.355, "ets": 0.216, "gss": 0.216, "orss": 0.957}


@pytest.mark.parametrize("name", sorted(FINLEY))
def test_finley_published_values(name):
    obs, model = finley_arrays()
    assert float(getattr(nw, name)(obs, model, 0.5)) == pytest.approx(FINLEY[name], abs=5e-4)


@pytest.mark.parametrize("name", CATEGORICAL)
def test_missing_observations_do_not_change_categorical_scores(name):
    obs, model = finley_arrays()
    obs_gappy, model_gappy = finley_arrays(extra_missing=1050)
    f = getattr(nw, name)
    assert_close(f(obs_gappy, model_gappy, 0.5), float(f(obs, model, 0.5)))


def one_d(values):
    return xr.DataArray(np.asarray(values, float), dims="t")


@pytest.mark.parametrize("name", ["pod", "far", "csi", "fb", "ets", "hss", "pss", "orss",
                                  "seds", "eds", "sedi", "f1", "mcc", "ba", "lift"])
def test_no_events_anywhere_is_undefined(name):
    zeros = one_d([0, 0, 0, 0])
    assert np.isnan(float(getattr(nw, name)(zeros, zeros, 1)))


@pytest.mark.parametrize("name", ["pod", "fb", "pss", "sedi", "eds", "ba", "lift", "mcc"])
def test_scores_needing_an_observed_event_are_undefined_without_one(name):
    obs, model = one_d([0, 0, 0, 0]), one_d([1, 1, 0, 0])
    assert np.isnan(float(getattr(nw, name)(obs, model, 1)))


@pytest.mark.parametrize("name, value", [("far", 1.0), ("csi", 0.0), ("f1", 0.0), ("ets", 0.0),
                                         ("hss", 0.0), ("gain", 0.5), ("npv", 1.0)])
def test_defined_scores_without_observed_events(name, value):
    obs, model = one_d([0, 0, 0, 0]), one_d([1, 1, 0, 0])
    assert float(getattr(nw, name)(obs, model, 1)) == pytest.approx(value)


def test_mcc_does_not_overflow_on_large_samples():
    rng = np.random.default_rng(3)
    obs = rng.random(3_000_000) < 0.5
    model = np.where(rng.random(obs.size) < 0.8, obs, ~obs)
    a, b = float(np.sum(obs & model)), float(np.sum(~obs & model))
    c, d = float(np.sum(obs & ~model)), float(np.sum(~obs & ~model))
    want = (a * d - b * c) / np.sqrt((a + b) * (a + c) * (d + b) * (d + c))
    got = nw.mcc(xr.DataArray(obs.astype(float), dims="t"),
                 xr.DataArray(model.astype(float), dims="t"), 0.5)
    assert float(got) == pytest.approx(want, rel=1e-12)


@pytest.mark.parametrize("name", ["sedi", "eds", "seds"])
def test_extremal_scores_take_their_limit_without_hits(name):
    obs = one_d(np.r_[np.ones(10), np.zeros(90)])
    model = one_d(np.r_[np.zeros(10), np.ones(9), np.zeros(81)])
    assert float(getattr(nw, name)(obs, model, 1)) == -1.0


@pytest.mark.parametrize("name", ["sedi", "eds", "seds"])
def test_extremal_scores_are_one_for_a_perfect_forecast(name):
    obs = one_d(np.r_[np.ones(10), np.zeros(90)])
    assert float(getattr(nw, name)(obs, obs, 1)) == pytest.approx(1.0)


# --------------------------------------------------------------------------- probabilistic
@pytest.mark.parametrize("case", CASES)
def test_bss_matches_reference(case):
    o, p = case_inputs(case, OBS, PROB)
    oo, pp = valid_pairs(o, p)
    ob = (oo >= THRESHOLD).astype(float)
    want = 1 - np.mean((pp - ob) ** 2) / np.mean((ob.mean() - ob) ** 2)
    assert_close(nw.bss(da(o), da(p), THRESHOLD), want)


@pytest.mark.parametrize("case", CASES)
def test_rpss_matches_reference(case):
    o, m = case_inputs(case)
    oo, mm = valid_pairs(o, m)
    ob, mb = (oo >= THRESHOLD).astype(float), (mm >= THRESHOLD).astype(float)
    want = 1 - np.mean((mb - ob) ** 2) / np.mean((ob.mean() - ob) ** 2)
    assert_close(nw.rpss(da(o), da(m), THRESHOLD), want)


@pytest.mark.parametrize("name", ["bss", "rpss"])
def test_skill_scores_undefined_for_constant_observations(name):
    obs, model = one_d(np.zeros(6)), one_d(np.linspace(0, 1, 6))
    assert np.isnan(float(getattr(nw, name)(obs, model, 1)))


# --------------------------------------------------------------------------- distributional
def distributional_reference(name, o, m):
    o, m = valid_pairs(o, m)
    p, q = o / o.sum(), m / m.sum()
    mid = 0.5 * (p + q)
    return {
        "mkldiv": np.sum(p * np.log(p / q)),
        "jsdiv": 0.5 * np.sum(p * np.log(p / mid)) + 0.5 * np.sum(q * np.log(q / mid)),
        "hellinger": np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)),
        "tv": 0.5 * np.sum(np.abs(p - q)),
        "chisquare": np.sum((p - q) ** 2 / q),
        "intersection": np.sum(np.minimum(p, q)),
        "bhattacharyya": -np.log(np.sum(np.sqrt(p * q))),
        "chernoff": -np.log(np.sum(p**0.3 * q**0.7)),
        "renyi": np.log(np.sum(p**0.3 * q**0.7)) / (0.3 - 1),
        "tsallis": (np.sum(p**0.3 * q**0.7) - 1) / (0.3 - 1),
        "wasserstein": np.mean(np.abs(np.sort(o) - np.sort(m))),
    }[name]


DISTRIBUTIONAL = ["mkldiv", "jsdiv", "hellinger", "tv", "chisquare", "intersection",
                  "bhattacharyya", "chernoff", "renyi", "tsallis", "wasserstein"]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("name", DISTRIBUTIONAL)
def test_distributional_matches_reference(name, case):
    o, m = case_inputs(case)
    f = getattr(nw, name)
    got = f(da(o), da(m), 0.3) if name in ("chernoff", "renyi", "tsallis") else f(da(o), da(m))
    assert_close(got, distributional_reference(name, o, m))


@pytest.mark.parametrize("name", DISTRIBUTIONAL[:-1])
@pytest.mark.parametrize("dry", ["obs", "model"])
def test_distributional_undefined_when_one_field_has_no_mass(name, dry):
    wet, zero = one_d([1.0, 2.0, 3.0]), one_d([0.0, 0.0, 0.0])
    obs, model = (zero, wet) if dry == "obs" else (wet, zero)
    f = getattr(nw, name)
    got = f(obs, model, 0.3) if name in ("chernoff", "renyi", "tsallis") else f(obs, model)
    assert np.isnan(float(got))


def test_mkldiv_is_infinite_where_the_model_misses_observed_mass():
    assert float(nw.mkldiv(one_d([1.0, 1.0]), one_d([1.0, 0.0]))) == np.inf
    assert float(nw.mkldiv(one_d([1.0, 0.0]), one_d([1.0, 1.0]))) == pytest.approx(np.log(2))


def test_wasserstein_with_dim_matches_reference_per_slice():
    o, m = case_inputs("obs missing")
    got = nw.wasserstein(da(o), da(m), dim=["y", "x"])
    for t in range(NT):
        oo, mm = valid_pairs(o[t], m[t])
        assert_close(got.isel(time=t), np.mean(np.abs(np.sort(oo) - np.sort(mm))))


def test_wasserstein_ignores_dimension_order():
    o, m = da(OBS), da(MODEL)
    straight = nw.wasserstein(o, m, dim="y")
    swapped = nw.wasserstein(o, m.transpose("x", "y", "time"), dim="y")
    xr.testing.assert_allclose(straight, swapped)


# --------------------------------------------------------------------------- alignment
def all_metric_calls():
    """Every public metric, called with obs and model only."""
    calls = {name: (lambda f: lambda o, m: f(o, m))(getattr(nw, name))
             for name in ["mae", "rmse", "acc", "r2", "nrmse", "pcc", "mbd", "tse", "evs",
                          "nmse", "fv", "sdr", "vif", "mad", "iqr", "nae", "rmb", "mape",
                          "lmbe", "smse", "gmb", "sbs", "aev", "cosine_similarity",
                          "mkldiv", "jsdiv", "hellinger", "wasserstein", "tv", "chisquare",
                          "intersection", "bhattacharyya", "harmonic_mean", "geometric_mean"]}
    calls.update({name: (lambda f: lambda o, m: f(o, m, 0.5))(getattr(nw, name))
                  for name in CATEGORICAL + ["bss", "rpss", "chernoff", "renyi", "tsallis",
                                             "rss", "ass", "lehmer_mean"]})
    calls["wmae"] = lambda o, m: nw.wmae(o, m, xr.ones_like(o))
    calls["qss"] = lambda o, m: nw.qss(o, m, o * 0 + 1)
    calls["fss"] = lambda o, m: nw.fss(o, m, 0.5, 3)
    return calls


ALL_METRICS = all_metric_calls()


def test_every_public_metric_is_covered():
    public = set(nw.__all__) - {"confusion_matrix", "NWP_Stats"}
    assert public == set(ALL_METRICS)


@pytest.mark.parametrize("name", sorted(ALL_METRICS))
def test_misaligned_coordinates_raise(name):
    lat = np.arange(8) * 0.1
    obs = xr.DataArray(np.random.default_rng(5).random((8, 8)) + 0.1, dims=("lat", "lon"),
                       coords={"lat": lat, "lon": lat})
    model = obs.assign_coords(lat=lat + 1e-7)
    with pytest.raises(ValueError, match="identical coordinates"):
        ALL_METRICS[name](obs, model)


@pytest.mark.parametrize("name", sorted(ALL_METRICS))
def test_every_metric_runs_on_aligned_gappy_data(name):
    o, m = case_inputs("obs missing")
    result = ALL_METRICS[name](da(o) / 5, da(m) / 5)
    assert isinstance(result, xr.DataArray)


def test_ass_reduces_a_per_element_reference_over_all_dims():
    obs = one_d(np.random.default_rng(6).random(50))
    got = nw.ass(obs, obs + 0.1, one_d(np.full(50, 0.2)))
    assert got.ndim == 0
    assert float(got) == pytest.approx(0.5)


def test_legacy_class_uses_the_fixed_metrics():
    o, m = case_inputs("obs missing")
    with pytest.warns(DeprecationWarning):
        stats = nw.NWP_Stats(da(o), da(m))
    assert_close(stats.compute_pod(THRESHOLD), float(nw.pod(da(o), da(m), THRESHOLD)))


# --------------------------------------------------------------------------- FSS
def fss_brute_force(obs, model, threshold, size):
    """Loop-based FSS: fractions over valid neighbours, every valid point scored."""
    half = size // 2
    num = den = 0.0
    for t in range(obs.shape[0]):
        ok = np.isfinite(obs[t]) & np.isfinite(model[t])
        ob = np.where(ok, obs[t] >= threshold, 0.0)
        mb = np.where(ok, model[t] >= threshold, 0.0)
        for i in range(obs.shape[1]):
            for j in range(obs.shape[2]):
                if not ok[i, j]:
                    continue
                win = (slice(max(i - half, 0), i + half + 1), slice(max(j - half, 0), j + half + 1))
                n = ok[win].sum()
                fo, fm = ob[win].sum() / n, mb[win].sum() / n
                num += (fo - fm) ** 2
                den += fo**2 + fm**2
    return 1 - num / den


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("size", [1, 3, 5, 11, 41])
def test_fss_matches_brute_force(size, case):
    o, m = case_inputs(case)
    o, m = o[:6], m[:6]
    got = nw.fss(xr.DataArray(o, dims=DIMS), xr.DataArray(m, dims=DIMS), THRESHOLD, size,
                 spatial_dims=["y", "x"])
    assert_close(got, fss_brute_force(o, m, THRESHOLD, size))


def test_fss_unchanged_by_a_missing_border():
    o, m = OBS[:5], MODEL[:5]
    pad = ((0, 0), (7, 7), (7, 7))
    wide_o = np.pad(o, pad, constant_values=np.nan)
    wide_m = np.pad(m, pad, constant_values=5.0)  # rain in the model outside the footprint
    for size in (1, 5, 9, 15):
        narrow = nw.fss(xr.DataArray(o, dims=DIMS), xr.DataArray(m, dims=DIMS), THRESHOLD, size,
                        spatial_dims=["y", "x"])
        wide = nw.fss(xr.DataArray(wide_o, dims=DIMS), xr.DataArray(wide_m, dims=DIMS),
                      THRESHOLD, size, spatial_dims=["y", "x"])
        assert_close(wide, float(narrow))


def test_fss_scores_every_point_at_large_neighbourhoods():
    o, m = OBS[:1], MODEL[:1]
    got = nw.fss(xr.DataArray(o, dims=DIMS), xr.DataArray(m, dims=DIMS), THRESHOLD, 41,
                 spatial_dims=["y", "x"])
    fo, fm = np.mean(o >= THRESHOLD), np.mean(m >= THRESHOLD)
    assert_close(got, 1 - (fo - fm) ** 2 / (fo**2 + fm**2))


def test_fss_undefined_without_events():
    zeros = xr.DataArray(np.zeros((2, 5, 5)), dims=DIMS)
    assert np.isnan(float(nw.fss(zeros, zeros, 1, 3)))


def test_fss_rejects_non_positive_neighbourhood():
    with pytest.raises(ValueError):
        nw.fss(da(OBS), da(MODEL), THRESHOLD, 0)
