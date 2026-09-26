"""Fractions Skill Score (FSS) with nwpeval: a worked example.

Runs as is on synthetic hourly rain. To use your own data, replace the
`make_synthetic_rain()` call with something like:

    obs = xr.open_dataset("radar.nc")["rainfall"]        # (time, lat, lon), mm/h
    model = xr.open_dataset("model.nc")["rainfall"]
    model = model.interp_like(obs)                        # put the model on the obs grid

Requirements: pip install "nwpeval>=1.6.3" matplotlib pandas

FSS (Roberts and Lean, 2008) turns each field into 0/1 events (value >= threshold),
takes the event fraction in a square neighbourhood around every point, and compares
the two fraction fields:

    FSS = 1 - sum((O - M)^2) / sum(O^2 + M^2)

0 = no skill, 1 = perfect. The score rises with neighbourhood size; the forecast is
"useful" once FSS >= 0.5 + f0/2, where f0 is the observed event frequency.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
import xarray as xr

import nwpeval as nw

GRID_KM = 11.1  # grid spacing (0.1 deg), only used to label neighbourhoods in km


def make_synthetic_rain(n_hours=72, ny=60, nx=60, seed=1):
    """Hourly obs and model rain (mm/h) and a circular radar footprint.

    Model storms are displaced about 6 points east (with scatter) and are 30 %
    too intense, so the model gets the rain but in the wrong place: it has little
    skill at the grid scale and gains skill as the neighbourhood grows.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:ny, 0:nx]
    obs = np.zeros((n_hours, ny, nx))
    model = np.zeros_like(obs)
    for t in range(n_hours):
        for _ in range(rng.integers(2, 6)):  # a few storms per hour
            cy, cx = rng.uniform(5, ny - 5), rng.uniform(5, nx - 5)
            radius, peak = rng.uniform(1.5, 3.0), rng.gamma(2.0, 4.0)
            obs[t] += peak * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * radius**2))
            my, mx = cy + rng.normal(0, 3), cx + 6 + rng.normal(0, 3)
            model[t] += 1.3 * peak * np.exp(-((y - my) ** 2 + (x - mx) ** 2) / (2 * radius**2))

    coords = {
        "time": pd.date_range("2014-02-01", periods=n_hours, freq="h"),
        "lat": -3.0 + 0.1 * np.arange(ny),
        "lon": -60.0 + 0.1 * np.arange(nx),
    }
    dims = ("time", "lat", "lon")
    footprint = xr.DataArray(
        (y - ny / 2) ** 2 + (x - nx / 2) ** 2 <= (0.45 * ny) ** 2,
        dims=("lat", "lon"), coords={"lat": coords["lat"], "lon": coords["lon"]},
    )
    # Outside the radar footprint the observations are missing (NaN)
    obs = xr.DataArray(obs, dims=dims, coords=coords).where(footprint)
    model = xr.DataArray(model, dims=dims, coords=coords)
    return obs, model


obs, model = make_synthetic_rain()

# Rules nwpeval applies to every metric:
#   * obs and model must have identical coordinates (otherwise ValueError)
#   * NaN means missing: a point missing in either input is left out, not
#     treated as dry. So FSS is computed inside the radar footprint only.
# fss() applies the second rule itself. We also mask the model here so that the
# base rates and percentiles below use the same points.
model = model.where(obs.notnull())

# ---------------------------------------------------------------- 1. one score
# threshold is in data units (mm/h); neighborhood_size is a width in grid points
# (use odd numbers so the window is centred). By default the sums run over all
# points and all times, which gives the aggregate FSS for the whole period.
score = nw.fss(obs, model, threshold=1.0, neighborhood_size=5)
print(f"FSS at 1 mm/h over a 5-point ({5 * GRID_KM:.0f} km) neighbourhood: {float(score):.3f}\n")

# ------------------------------------------- 2. thresholds x neighbourhood table
thresholds = [0.1, 0.5, 1, 2, 5, 10]  # mm/h
sizes = [1, 3, 5, 9, 15, 25]  # grid points

table = xr.DataArray(
    [[float(nw.fss(obs, model, t, n)) for n in sizes] for t in thresholds],
    dims=("threshold", "size"), coords={"threshold": thresholds, "size": sizes},
)
# Useful skill: 0.5 + f0/2, with f0 the observed event frequency in the footprint
base_rate = np.array([float((obs >= t).where(obs.notnull()).mean()) for t in thresholds])
useful = 0.5 + base_rate / 2

print("FSS (rows: threshold in mm/h, columns: neighbourhood in grid points)")
print(table.to_pandas().round(3))
for t, u, row in zip(thresholds, useful, table.values):
    ok = [n for n, v in zip(sizes, row) if v >= u]
    where = f"from {ok[0]} points ({ok[0] * GRID_KM:.0f} km)" if ok else "at no size tested"
    print(f"  {t:>4g} mm/h: useful skill (>= {u:.3f}) {where}")

# ------------------------------------------------ 3. percentile thresholds
# A fixed threshold mixes placement and amount errors: a model that rains too
# much is penalised at every size. Thresholding each field at its OWN percentile
# removes the amount bias and scores placement only. Binarise first (keeping NaN
# as missing), then call fss() with threshold 0.5 on the 0/1 fields.
def events_above_percentile(field, q):
    return (field >= field.quantile(q / 100)).where(field.notnull())


print("\nPercentile thresholds (placement only), 9-point neighbourhood:")
for q in (90, 95, 99):
    fss_q = nw.fss(events_above_percentile(obs, q), events_above_percentile(model, q), 0.5, 9)
    print(f"  p{q}: FSS = {float(fss_q):.3f}   (useful >= {0.5 + (1 - q / 100) / 2:.3f})")

# ------------------------------------------------------ 4. FSS by hour of day
# FSS is a ratio of sums. To split it by hour of day, pool all time steps of each
# hour and score them together. Do NOT average FSS values of single time steps:
# that weights a dry hour the same as a stormy one and is not the aggregate score.
# xr.align(join="exact") raises if the grids differ; building the Dataset
# directly would silently pad a mismatch with NaN.
obs, model = xr.align(obs, model, join="exact")
pairs = xr.Dataset({"obs": obs, "model": model})
diurnal = pairs.groupby("time.hour").map(lambda g: nw.fss(g.obs, g.model, 1.0, 9))
print("\nFSS at 1 mm/h, 9 points, by hour of day (first 6 hours):")
print(diurnal.to_pandas().head(6).round(3).to_string())

per_step = nw.fss(obs, model, 1.0, 9, reduction_dim=["lat", "lon"])  # one FSS per time step
print(f"\naggregate FSS {float(nw.fss(obs, model, 1.0, 9)):.3f} "
      f"vs mean of per-step FSS {float(per_step.mean()):.3f} (the second is not the aggregate)")

# ---------------------------------------------------------------- 5. heatmap
fig, ax = plt.subplots(figsize=(7.5, 4.8))
image = ax.imshow(table.values, origin="lower", cmap="Blues", vmin=0, vmax=1, aspect="auto")
for i, u in enumerate(useful):
    for j, value in enumerate(table.values[i]):
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9,
                color="white" if value > 0.6 else "black")
        if value >= u:  # outline cells with useful skill
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=2, ec="#d95f02"))
ax.set_xticks(range(len(sizes)), [f"{n}\n{n * GRID_KM:.0f} km" for n in sizes])
ax.set_yticks(range(len(thresholds)), [f"{t:g}" for t in thresholds])
ax.set_xlabel("Neighbourhood width (grid points, km)")
ax.set_ylabel("Threshold (mm/h)")
ax.set_title("Fractions skill score of hourly rain, model vs radar")
ax.legend(handles=[Patch(fill=False, ec="#d95f02", lw=2, label="useful skill, FSS >= 0.5 + f0/2")],
          loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
cbar = fig.colorbar(image, ax=ax, label="FSS")
cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
fig.savefig("fss_heatmap.png", dpi=450, bbox_inches="tight")
print("\nsaved fss_heatmap.png")
