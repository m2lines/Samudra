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

pytestmark = [pytest.mark.om4_live, pytest.mark.invariant(status="agreed")]

OSN = "https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra"
RELEASE = os.environ.get("SAMUDRA_OM4_RELEASE", "v2026-09")
RESOLUTIONS = ("onedeg", "halfdeg", "quarterdeg")
LEVELS = 19
STEPS = 4745
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
    st.integers(0, STEPS - 1), st.deferred(lambda: st.sampled_from(boundary_steps()))
)


def stacked(ds: xr.Dataset, base: str, t: int) -> xr.DataArray:
    return xr.concat(
        [ds[f"{base}_{k}"].isel(time=t) for k in range(LEVELS)], dim="lev"
    ).load()


def volume(ds: xr.Dataset) -> xr.DataArray:
    return conservation.volume_weights(ds.areacello, ds.ocean_fraction, ds.dz).load()


def surface_area(ds: xr.Dataset) -> xr.DataArray:
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


@pytest.mark.xfail(strict=True, reason="#972: areacello uses the polar radius, 6356 km")
@pytest.mark.parametrize("resolution", RESOLUTIONS)
def test_store_cells_tile_the_sphere(resolution):
    require_no_violations(
        geometry.sphere_area(store(resolution).areacello.load(), rtol=1e-4)
    )


@pytest.mark.parametrize("resolution", RESOLUTIONS)
def test_store_layers_tile_the_om4_column(resolution):
    ds = store(resolution)
    dz = ds.dz.values
    ilev = np.concatenate([[0.0], np.cumsum(dz)])
    found = geometry.layer_thickness(dz, ilev, ds.lev.values)
    if ilev[-1] != 6500:
        found.append(
            Violation("geometry.column_depth", f"layers sum to {ilev[-1]} m, not 6500")
        )
    require_no_violations(found)


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
            ds.zos.isel(time=t).load(), surface_area(ds), atol=0.01
        )
    )


@pytest.mark.invariant(status="proposed")
@pytest.mark.parametrize("resolution", RESOLUTIONS)
@live
@given(t=steps)
def test_store_coasts_are_not_diluted_by_land(resolution, t):
    """Proposed: river mouths are fresh, so the threshold needs agreeing (#974)."""
    ds = store(resolution)
    require_no_violations(
        bounds.coastal_dilution(ds.so_0.isel(time=t).load(), ds.mask_0.load(), "so_0")
    )


def integrals(ds: xr.Dataset, t: int) -> dict[str, float]:
    """Linear integrals that conservative regridding preserves."""
    v = volume(ds)
    a = surface_area(ds)
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
    totals = {r: integrals(store(r), t) for r in RESOLUTIONS}
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
                surface_area(fine),
                c,
                surface_area(coarse),
                f"{name}_{fine_r}_to_{coarse_r}",
                rtol=1e-4,
            )
        ke_fine = 0.5 * (fine.uo_0.isel(time=t) ** 2 + fine.vo_0.isel(time=t) ** 2)
        ke_coarse = 0.5 * (
            coarse.uo_0.isel(time=t) ** 2 + coarse.vo_0.isel(time=t) ** 2
        )
        mean_fine = float(weighted_mean(ke_fine, surface_area(fine), ["y", "x"]))
        mean_coarse = float(weighted_mean(ke_coarse, surface_area(coarse), ["y", "x"]))
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
    """Circumpolar Southern Ocean rows are all ocean, so their zonal spectra can be compared directly."""
    band = (-60.0, -57.0)
    fine, coarse = store("quarterdeg"), store("onedeg")
    require_no_violations(
        conservation.zonal_spectra_agree(
            fine.thetao_0.isel(time=t).load(),
            fine.y,
            coarse.thetao_0.isel(time=t).load(),
            coarse.y,
            band,
            "sst",
            rtol=0.05,
        )
    )


@functools.cache
def yearly_budget(
    resolution: str, start: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Heat content, area-integrated hfds and salt content for 75 steps (a year plus each end)."""
    ds = store(resolution).isel(time=slice(start, start + 75))
    v = volume(store(resolution))
    a = surface_area(store(resolution))
    heat, flux, salt = [], [], []
    for t in range(ds.sizes["time"]):
        thetao, so = stacked(ds, "thetao", t), stacked(ds, "so", t)
        heat.append(float(conservation.heat_content(thetao, v, ["lev", "y", "x"])))
        salt.append(float(conservation.salt_content(so, v, ["lev", "y", "x"])))
        flux.append(float((ds.hfds.isel(time=t) * a).sum()))
    return np.array(heat), np.array(flux), np.array(salt)


years = st.integers(0, STEPS // 73 - 2).map(lambda year: 73 * year)


@live
@given(start=years)
def test_heat_enters_only_at_the_surface(start):
    heat, flux, _ = yearly_budget("onedeg", start)
    require_no_violations(
        budgets.heat_budget(heat, flux, min_correlation=0.95, slope_range=(0.9, 1.1))
    )


@live
@given(start=years)
def test_forcing_drives_the_state_in_the_same_window(start):
    """If hfds were a step early or late, the tendency would correlate best at lag +-1."""
    heat, flux, _ = yearly_budget("onedeg", start)
    tendency = budgets.central_tendency(heat, 5 * 86400.0)
    require_no_violations(coupling.response_peaks_at_lag_zero(flux, tendency, "hfds"))


@live
@given(start=years)
def test_salt_is_conserved(start):
    _, _, salt = yearly_budget("onedeg", start)
    require_no_violations(budgets.salt_drift(salt, max_relative=1e-6))


@pytest.mark.invariant(status="proposed")
@live
@given(t=steps)
def test_surface_currents_follow_the_wind_like_ekman_drift(t):
    """Proposed: geostrophic currents may mask Ekman drift in a single 5-day mean."""
    ds = store("onedeg").isel(time=t)
    a = surface_area(store("onedeg"))
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
        heat, flux, _ = yearly_budget(r, start)
        tendency = budgets.central_tendency(heat, 5 * 86400.0)
        ocean_area = float(surface_area(store(r)).sum())
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
    recorded = published.attrs["m2lines/cli_args"].split()
    args = [
        a
        for a in recorded[1:]
        if not a.startswith(
            ("--output_path", "--cluster", "--n_workers", "--threads_per_worker")
        )
    ]
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
    require_no_violations(
        releases.stores_agree(
            reference,
            rerun,
            variables=sorted(
                set(rerun.data_vars) | {"areacello", "ocean_fraction", "dz"}
            ),
        )
    )
