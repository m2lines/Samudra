# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Ocean-physics properties of the published OM4 stores, read live from OSN.

These need the network, so they skip unless pytest gets ``--om4-live``. The file
name doesn't end in ``_properties.py``, so property campaigns and the mutation
gate never collect it. Hypothesis picks the time steps, biased towards year and
leap-year boundaries, and each check reads only the steps it needs.

* ``SAMUDRA_OM4_LIVE_EXAMPLES`` sets examples per property (default 3).
* ``SAMUDRA_OM4_RELEASE`` picks the release (default ``v2026-09``).
* ``--om4-live-heavy`` adds checks that read a year of 1/4 degree data or re-run
  the pipeline. Run those next to the data, on the DTN.

Run from ``data/``: ``python -m pytest tests/test_om4_live.py --om4-live``.
"""

import functools
import os
import subprocess
import sys

import numpy as np
import pytest
import xarray as xr
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from ocean_preprocessing.__main__ import recorded_arguments

from samudra.invariants import (
    InvariantViolated,
    bounds,
    budgets,
    conservation,
    coupling,
    geometry,
    masks,
    releases,
    require_no_violations,
    timeaxis,
)
from samudra.invariants.core import Violation, weighted_mean
from tests.ocean_strategies import OM4_INTERFACES

pytestmark = [pytest.mark.om4_live, pytest.mark.invariant(status="agreed")]

OSN = "https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra"
RELEASE = os.environ.get("SAMUDRA_OM4_RELEASE", "v2026-09")
RESOLUTIONS = ("onedeg", "halfdeg", "quarterdeg")
LEVELS = 19
SEAM_X = 60.5  # 1 degree column at OM4's longitude seam (xh starts at -300)
live = settings(
    max_examples=int(os.environ.get("SAMUDRA_OM4_LIVE_EXAMPLES", "3")),
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@functools.cache
def store(resolution: str) -> xr.Dataset:
    return xr.open_zarr(f"{OSN}/{RELEASE}/om4_{resolution}/OM4.zarr", chunks={})


@functools.cache
def time_axis() -> np.ndarray:
    return store("onedeg").time.values


def boundary_steps() -> list[int]:
    """Indices of the last step of each year and the first of the next, where averaging bugs cluster."""
    stamps = time_axis()
    years = np.array([t.year for t in stamps])
    change = np.flatnonzero(np.diff(years)) + 1
    return sorted({int(i) for c in change for i in (c - 1, c)})


# Half the draws land on a year boundary. Deferred, so collection reads nothing.
steps = st.one_of(
    st.deferred(lambda: st.integers(0, len(time_axis()) - 1)),
    st.deferred(lambda: st.sampled_from(boundary_steps())),
)


def stacked(ds: xr.Dataset, base: str, t: int | None) -> xr.DataArray:
    """``{base}_0 .. {base}_18`` as one (lev, y, x) array, at step ``t`` if given."""
    levels = [ds[f"{base}_{k}"] for k in range(LEVELS)]
    if t is not None:
        levels = [level.isel(time=t) for level in levels]
    return xr.concat(levels, dim="lev").load()


@functools.cache
def volume(resolution: str) -> xr.DataArray:
    ds = store(resolution)
    return conservation.volume_weights(ds.areacello, ds.ocean_fraction, ds.dz).load()


@functools.cache
def surface_area(resolution: str) -> xr.DataArray:
    ds = store(resolution)
    return (ds.areacello * ds.ocean_fraction.isel(lev=0)).load()


@pytest.mark.parametrize("resolution", RESOLUTIONS)
@live
@given(t=steps)
def test_store_masks_agree_with_fields_and_fractions(resolution, t):
    # Tolerates the sub-1e-6 fraction slivers that the next test pins down, so
    # that this one still checks every field's NaN pattern at each step.
    require_no_violations(
        masks.flattened_store(store(resolution), time_index=t, fraction_atol=1e-6)
    )


@pytest.mark.xfail(
    strict=True,
    raises=InvariantViolated,
    reason="Arctic cells keep ocean_fraction slivers of 1e-8 to 3e-7 where "
    "mask_k is land: horizontal_regrid thresholds the mask but not the fraction",
)
@pytest.mark.parametrize("resolution", RESOLUTIONS)
def test_store_ocean_fraction_is_zero_on_land(resolution):
    ds = store(resolution)
    names = [f"mask_{k}" for k in range(LEVELS)] + ["ocean_fraction"]
    loaded = ds[names].load()
    require_no_violations(
        violation
        for k in range(LEVELS)
        for violation in masks.mask_matches_fraction(
            loaded[f"mask_{k}"], loaded.ocean_fraction.isel(lev=k), f"mask_{k}"
        )
    )


@pytest.mark.xfail(
    strict=True,
    raises=InvariantViolated,
    reason="#972: areacello uses the polar radius, 6356 km",
)
@pytest.mark.parametrize("resolution", RESOLUTIONS)
def test_store_cells_tile_the_sphere(resolution):
    require_no_violations(
        geometry.sphere_area(store(resolution).areacello.load(), rtol=1e-4)
    )


@pytest.mark.parametrize("resolution", RESOLUTIONS)
def test_store_layers_tile_the_om4_column(resolution):
    """The store's dz and lev fit OM4's 19 nominal layers, 0 to 6500 m. The store has no ilev."""
    ds = store(resolution)
    require_no_violations(
        geometry.layer_thickness(ds.dz.values, OM4_INTERFACES, ds.lev.values)
    )


@pytest.mark.parametrize("resolution", RESOLUTIONS)
def test_store_time_axis_is_om4s(resolution):
    require_no_violations(timeaxis.om4_five_day_axis(store(resolution).time.values))


@pytest.mark.parametrize("resolution", RESOLUTIONS)
@live
@given(t=steps)
def test_store_water_is_liquid_and_salty(resolution, t):
    ds = store(resolution)
    thetao, so = stacked(ds, "thetao", t), stacked(ds, "so", t)
    depth = xr.DataArray(ds.lev.values, dims="lev")
    require_no_violations(
        bounds.above_freezing(thetao, so, depth)
        + bounds.within(so, "so", 0.0, 42.0)
        + bounds.within(thetao, "thetao", -2.5, 40.0)
    )


@pytest.mark.parametrize("resolution", RESOLUTIONS)
@live
@given(t=steps)
def test_store_sea_level_has_no_global_mean(resolution, t):
    ds = store(resolution)
    require_no_violations(
        budgets.area_mean_sea_level(
            ds.zos.isel(time=t).load(), surface_area(resolution), atol=0.01
        )
    )


@pytest.mark.invariant(status="proposed")
@pytest.mark.parametrize("resolution", RESOLUTIONS)
@live
@given(t=steps)
def test_store_coasts_are_not_diluted_by_land(resolution, t):
    """Proposed (#974): no coastal cell is diluted towards zero-filled land.

    River plumes are fresh: the Orinoco, Amazon and Rio de la Plata mouths reach
    0.27 to 0.43 of their nearest wet neighbour's salinity in v2026-09. A cell
    that is 80% zero-filled land would reach 0.2, hence ``max_fraction``.
    """
    ds = store(resolution)
    require_no_violations(
        bounds.coastal_dilution(
            ds.so_0.isel(time=t).load(), ds.mask_0.load(), "so_0", max_fraction=0.2
        )
    )


def integrals(resolution: str, t: int) -> dict[str, float]:
    """Linear integrals that conservative regridding preserves."""
    ds = store(resolution)
    v = volume(resolution)
    a = surface_area(resolution)
    thetao, so = stacked(ds, "thetao", t), stacked(ds, "so", t)
    sst = ds.thetao_0.isel(time=t).load()
    return {
        "ocean_area": float(a.sum()),
        "ocean_volume": float(v.sum()),
        "heat_content": float(conservation.heat_content(thetao, v, ["lev", "y", "x"])),
        "salt_content": float(conservation.salt_content(so, v, ["lev", "y", "x"])),
        "mean_sst": float(weighted_mean(sst, a, ["y", "x"])),
        "surface_heat_flux": float((ds.hfds.isel(time=t) * a).sum()),
        "freshwater_flux": float((ds.wfo.isel(time=t) * a).sum()),
        "mean_zos": float(weighted_mean(ds.zos.isel(time=t), a, ["y", "x"])),
    }


@live
@given(t=steps)
def test_resolutions_agree_on_linear_integrals(t):
    """The three stores share one source, so conservative regridding leaves these equal."""
    totals = {r: integrals(r, t) for r in RESOLUTIONS}
    found = []
    for name in totals["onedeg"]:
        values = {r: totals[r][name] for r in RESOLUTIONS}
        atol = 1e-4 if name == "mean_zos" else 0.0
        reference = values["quarterdeg"]
        for r in ("onedeg", "halfdeg"):
            if not np.isclose(values[r], reference, rtol=1e-3, atol=atol):
                found.append(
                    Violation(
                        f"conservation.{name}_across_resolutions",
                        f"{r} {values[r]:.7g} vs quarterdeg {reference:.7g}",
                        {"time": t},
                    )
                )
    require_no_violations(found)


@live
@given(t=steps)
def test_coarsening_never_adds_variance_or_extremes(t):
    found = []
    pairs = (("quarterdeg", "halfdeg"), ("halfdeg", "onedeg"))
    for fine_r, coarse_r in pairs:
        fine, coarse = store(fine_r), store(coarse_r)
        for name in ("thetao_0", "so_0", "zos"):
            f, c = fine[name].isel(time=t).load(), coarse[name].isel(time=t).load()
            found += conservation.no_new_extremes(
                f, c, f"{name}_{fine_r}_to_{coarse_r}", atol=1e-4
            )
            found += conservation.variance_not_increased(
                f,
                surface_area(fine_r),
                c,
                surface_area(coarse_r),
                f"{name}_{fine_r}_to_{coarse_r}",
                rtol=1e-4,
            )
        ke_fine = 0.5 * (fine.uo_0.isel(time=t) ** 2 + fine.vo_0.isel(time=t) ** 2)
        ke_coarse = 0.5 * (
            coarse.uo_0.isel(time=t) ** 2 + coarse.vo_0.isel(time=t) ** 2
        )
        mean_fine = float(weighted_mean(ke_fine, surface_area(fine_r), ["y", "x"]))
        mean_coarse = float(
            weighted_mean(ke_coarse, surface_area(coarse_r), ["y", "x"])
        )
        if mean_coarse > mean_fine * (1 + 1e-4):
            found.append(
                Violation(
                    "conservation.kinetic_energy_not_increased",
                    f"surface KE {fine_r} {mean_fine:.5g} < {coarse_r} {mean_coarse:.5g}",
                )
            )
    require_no_violations(found)


@live
@given(t=steps)
def test_resolutions_share_large_scale_zonal_spectra(t):
    """Circumpolar Southern Ocean rows are all ocean, so their zonal spectra can be compared directly.

    The 1 and 1/4 degree rows don't nest: selecting rows by centre compared
    -59.83..-56.84 with -59.96..-56.96. The band now sits on 1 degree edges and
    the 1/4 degree rows count by their overlap with it.
    """
    fine, coarse = store("quarterdeg"), store("onedeg")
    # The 1 degree rows whose centres lie in 60S to 57S, edge to edge.
    edges = coarse.lat_b.values[:, 0]
    rows = np.flatnonzero((coarse.y.values >= -60.0) & (coarse.y.values <= -57.0))
    band = (float(edges[rows[0]]), float(edges[rows[-1] + 1]))
    require_no_violations(
        conservation.zonal_spectra_agree(
            fine.thetao_0.isel(time=t).load(),
            fine.lat_b.values,
            coarse.thetao_0.isel(time=t).load(),
            coarse.lat_b.values,
            band,
            "sst",
            rtol=0.05,
        )
    )


@functools.cache
def yearly_budget(
    resolution: str, start: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Heat content, area-integrated hfds, salt content and window centres in seconds,
    for 75 steps (a year plus each end).

    The top layer is ``dz_0 + zos`` thick: freshwater that raises sea level also
    dilutes it, and the product is what the model conserves. With a fixed top
    layer, a 1 cm change in sea level reads as a 3e-6 change in total salt.
    """
    ds = store(resolution).isel(time=slice(start, start + 75))
    v = volume(resolution)
    a = surface_area(resolution)
    names = [f"{base}_{k}" for base in ("thetao", "so") for k in range(LEVELS)]
    heat, flux, salt = [], [], []
    for t in range(ds.sizes["time"]):
        step = ds[[*names, "zos", "hfds"]].isel(time=t).load()
        thetao, so = stacked(step, "thetao", None), stacked(step, "so", None)
        v_t = v.copy()
        v_t[{"lev": 0}] = v.isel(lev=0) + (step.zos * a).fillna(0.0)
        heat.append(float(conservation.heat_content(thetao, v_t, ["lev", "y", "x"])))
        salt.append(float(conservation.salt_content(so, v_t, ["lev", "y", "x"])))
        flux.append(float((step.hfds * a).sum()))
    stamps = timeaxis.as_cftime(ds.time.values)
    seconds = np.array([timeaxis.seconds_between(stamps[0], s) for s in stamps])
    return np.array(heat), np.array(flux), np.array(salt), seconds


# Whole years with a step to spare at each end. Deferred, so collection reads nothing.
years = st.deferred(
    lambda: st.integers(0, len(time_axis()) // timeaxis.OM4_STEPS_PER_YEAR - 2)
).map(lambda year: timeaxis.OM4_STEPS_PER_YEAR * year)


@live
@given(start=years)
def test_heat_enters_only_at_the_surface(start):
    heat, flux, _, seconds = yearly_budget("onedeg", start)
    require_no_violations(
        budgets.heat_budget(
            heat, flux, seconds, min_correlation=0.95, slope_range=(0.9, 1.1)
        )
    )


@live
@given(start=years)
def test_forcing_drives_the_state_in_the_same_window(start):
    """If hfds were a step early or late, the tendency would correlate best at lag +-1."""
    heat, flux, _, seconds = yearly_budget("onedeg", start)
    tendency = budgets.central_tendency(heat, seconds)
    require_no_violations(coupling.response_peaks_at_lag_zero(flux, tendency, "hfds"))


@pytest.mark.invariant(status="proposed")
@live
@given(start=years)
def test_salt_is_conserved(start):
    """Proposed: v2026-09 1 degree salt varies by 3.8e-6 over a year, with zos in the
    top layer. Sea ice stores salt (SIS2 ice is about 5 g/kg), so the seasonal ice
    cycle may account for ~1e-6 of it; the bound needs agreeing once that's checked.
    """
    _, _, salt, _ = yearly_budget("onedeg", start)
    require_no_violations(budgets.salt_drift(salt, max_relative=1e-6))


@pytest.mark.invariant(status="proposed")
@live
@given(t=steps)
def test_surface_currents_follow_the_wind_like_ekman_drift(t):
    """Proposed: geostrophic currents may mask Ekman drift in a single 5-day mean."""
    ds = store("onedeg").isel(time=t)
    a = surface_area("onedeg")
    args = (ds.tauuo.load(), ds.tauvo.load(), ds.uo_0.load(), ds.vo_0.load())
    require_no_violations(
        coupling.stress_and_current_aligned(*args, a)
        + coupling.ekman_deflection(*args, ds.lat.load(), a)
    )


@pytest.mark.om4_live_heavy
@live
@given(start=years)
def test_heat_budget_residual_is_the_same_at_every_resolution(start):
    residuals = {}
    for r in RESOLUTIONS:
        heat, flux, _, seconds = yearly_budget(r, start)
        tendency = budgets.central_tendency(heat, seconds)
        ocean_area = float(surface_area(r).sum())
        residuals[r] = float(np.nanmean(tendency - flux) / ocean_area)
    require_no_violations(
        budgets.residuals_agree(residuals, "heat_budget_w_m2", atol=0.05)
    )


@pytest.mark.om4_live_heavy
def test_current_pipeline_reproduces_the_release(tmp_path):
    """Re-run today's pipeline on the release's raw inputs for 10 steps and compare with the store.

    Uses the release's recorded command line (``m2lines/cli_args``), with
    ``--small_run`` and a local output path.
    """
    published = store("onedeg")
    args = recorded_arguments(
        published.attrs["m2lines/cli_args"],
        frozenset({"output_path", "cluster", "n_workers", "threads_per_worker"}),
    )
    output = tmp_path / "rerun.zarr"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ocean_preprocessing",
            *args,
            f"--output_path={output}",
            "--small_run",
        ],
        check=True,
        env={
            **os.environ,
            "FSSPEC_S3_ENDPOINT_URL": "https://nyu1.osn.mghpcc.org",
            "FSSPEC_S3_ANON": "true",
        },
    )
    rerun = xr.open_zarr(output)
    reference = published.isel(time=slice(0, rerun.sizes["time"]))
    # #928 fixed the 60E seam (`om4_staggered_grid`) after v2026-09: u fields
    # differ there and must agree everywhere else.
    seam_fixed = frozenset(
        name for name in rerun.data_vars if name == "tauuo" or name.startswith("uo_")
    )
    seam = int(np.abs(rerun.x.values - SEAM_X).argmin())
    if abs(float(rerun.x[seam]) - SEAM_X) > 0.5:
        raise ValueError(
            f"no column of the rerun lies within half a degree of {SEAM_X}"
        )
    off_seam = xr.DataArray(np.arange(rerun.sizes["x"]) != seam, dims="x")
    require_no_violations(
        releases.stores_agree(
            reference,
            rerun,
            variables=sorted(
                set(rerun.data_vars) | {"areacello", "ocean_fraction", "dz"}
            ),
            expected_changes=seam_fixed,
        )
        + releases.stores_agree(
            reference[sorted(seam_fixed)].where(off_seam),
            rerun[sorted(seam_fixed)].where(off_seam),
        )
    )
