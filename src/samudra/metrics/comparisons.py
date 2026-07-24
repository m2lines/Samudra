# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Model and observation fields, aligned and ready to reduce.

Scoring a rollout and drawing it are the same pipeline up to the final
reduction: find the observed variable, derive the matching model field, restrict
both to the span and timestamps they share, and put the model on the
observation grid without breaking the quantities that are not linear. Only then
do the two jobs diverge -- one takes an area-weighted RMSE, the other draws a
map or a spectrum.

That shared prefix lives here so `samudra.metrics.report` and
`samudra.viz.observations` cannot drift: a difference between a logged number
and a drawn one would have to come from the reduction, which is the part the
reader can see.

Each builder returns a `Comparison`, which holds the model field on its own grid
as well as on the observation grid. Which one to reduce is not a detail: linear
differences may be interpolated first, but variance and eddy energy are
quadratic, and interpolating before reducing damps them by an amount set by the
resolution ratio rather than by the model.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
from dataclasses import dataclass
from typing import Self, cast

import numpy as np
import pandas as pd
import xarray as xr

from samudra.constants import DataLayout
from samudra.metrics import kernels, observations

logger = logging.getLogger(__name__)

SPATIAL_DIMS = ("lat", "lon")


def ever_finite(field: xr.DataArray) -> xr.DataArray:
    """A cell mask: true where the field carries data at any time."""
    finite = cast(xr.DataArray, np.isfinite(field))
    return finite.any("time") if "time" in field.dims else finite


def _without_time_mean(component: Comparison) -> Comparison:
    """The model side of a comparison, as an anomaly about its own time mean."""
    return dataclasses.replace(
        component, native=component.native - component.native.mean("time")
    )


@dataclass(frozen=True)
class Pairing:
    """How much of the observation ocean survived pairing with a model.

    Linear interpolation makes an observation cell NaN whenever its model-grid
    neighbours include land, so the comparison set is eroded by a band one model
    cell wide -- all of it coastal, which is where model error is largest. The
    eroded fraction therefore scales with model resolution: a coarse run is
    scored on a smaller and easier subset of the ocean than a fine one.
    Recording it makes that visible instead of leaving cross-resolution
    comparison merely "not comparable" in prose.
    """

    n_paired_cells: int
    n_observed_cells: int

    @property
    def paired_ocean_fraction(self) -> float:
        """The share of the product's ocean the model reaches."""
        if not self.n_observed_cells:
            return float("nan")
        return self.n_paired_cells / self.n_observed_cells

    def as_row(self) -> dict[str, float]:
        """The metric-frame columns this contributes."""
        return {
            "n_paired_cells": float(self.n_paired_cells),
            "paired_ocean_fraction": self.paired_ocean_fraction,
        }


@dataclass
class Comparison:
    """One model field and one observation field, aligned and comparable.

    Both cover the same span and the same timestamps. `native` is on the
    model's own grid and `on_obs_grid` is the interpolation of it, computed once
    and reused.
    """

    label: str
    native: xr.DataArray
    obs: xr.DataArray
    area: xr.DataArray

    @functools.cached_property
    def on_obs_grid(self) -> xr.DataArray:
        """The model field interpolated onto the observation grid."""
        return kernels.model_field_on_obs_grid(self.native, self.obs)

    @functools.cached_property
    def error_squared(self) -> xr.DataArray:
        """Squared model-minus-observation error, on the observation grid."""
        return (self.on_obs_grid - self.obs) ** 2

    @property
    def time(self) -> xr.DataArray:
        return self.obs["time"]

    def slice(self, time_slice: slice) -> Self:
        """Restrict the comparison to a time slice."""
        return dataclasses.replace(
            self,
            native=self.native.sel(time=time_slice),
            obs=self.obs.sel(time=time_slice),
        )

    @property
    def pairing(self) -> Pairing:
        """How much of the observation ocean this model reaches."""
        obs_mask = ever_finite(self.obs)
        return Pairing(
            n_paired_cells=int((obs_mask & ever_finite(self.on_obs_grid)).sum()),
            n_observed_cells=int(obs_mask.sum()),
        )


@dataclass
class LayerComparison(Comparison):
    """An ocean-heat-content layer, with the caveat that qualifies it.

    `ohc_per_area_layer_maps` integrates whatever levels a cell has. Where two
    products agree a column is shallow that is exact; where their bathymetry
    *disagrees* the deeper one integrates water the other lacks, and the
    difference reads as a heat deficit of roughly 2e9 J m^-2 per 50 m -- the
    size of the whole 0-700 m score.

    `bathymetry_disagreement_m` measures that directly, as the area-weighted
    metres of water the two sides differ by. Counting each side's incomplete
    columns separately would only bound it: two products can be shallow in the
    same places, or in different ones, and the fractions cannot tell those
    apart. `model_partial_columns` is kept beside it to say which side is the
    shallow one.
    """

    bathymetry_disagreement_m: float = float("nan")
    model_partial_columns: float = float("nan")


@dataclass
class VelocityComparison:
    """The two geostrophic velocity components, aligned together.

    Kept as a pair because the vector RMSE combines them before reducing over
    time, and because the derived energies need both on the native grid.
    """

    eastward: Comparison
    northward: Comparison
    kind: str = "absolute"

    @property
    def area(self) -> xr.DataArray:
        return self.eastward.area

    def _rebased(self) -> Self:
        """Put the model on the same footing as the observed velocity.

        DUACS reports either an absolute geostrophic velocity or an anomaly
        about its own reference period, and the model's velocity derived from
        `zos` is always absolute. Against the anomaly product the model's time
        mean has to come off, or the two are not the same quantity.

        It cannot be left to a detrend further downstream: kinetic energy is
        quadratic, so a retained mean flow contributes a constant *and* a cross
        term with the eddies, and no linear detrend of the energy series
        removes either.
        """
        if self.kind != "anomaly":
            return self
        return dataclasses.replace(
            self,
            eastward=_without_time_mean(self.eastward),
            northward=_without_time_mean(self.northward),
        )

    @property
    def time(self) -> xr.DataArray:
        return self.eastward.obs["time"]

    def slice(self, time_slice: slice) -> Self:
        """Restrict both components to a time slice.

        Slicing has to happen here rather than on a derived quantity: the
        energies reduce over time, so a mean taken before the slice is a mean of
        the wrong record and cannot be corrected afterwards.
        """
        # Rebased after slicing, not before: for the anomaly kind the mean has
        # to be the one over the span actually being compared.
        return dataclasses.replace(
            self,
            eastward=self.eastward.slice(time_slice),
            northward=self.northward.slice(time_slice),
        )._rebased()

    @functools.cached_property
    def vector_error_squared(self) -> xr.DataArray:
        """Squared vector error: a cell is penalised for direction, not just speed."""
        return self.eastward.error_squared + self.northward.error_squared

    def eddy_kinetic_energy(self) -> Comparison:
        """Instantaneous EKE, reduced on the native grid before regridding."""
        return Comparison(
            label=f"{self.eastward.label} EKE",
            native=kernels.instantaneous_surface_eke(
                self.eastward.native, self.northward.native
            ),
            obs=kernels.instantaneous_surface_eke(
                self.eastward.obs, self.northward.obs
            ),
            area=self.area,
        )

    def mean_eddy_kinetic_energy(self) -> Comparison:
        """Time-mean EKE map, for the figures rather than for scoring."""
        return Comparison(
            label=f"{self.eastward.label} mean EKE",
            native=kernels.mean_surface_eke(
                self.eastward.native, self.northward.native
            ),
            obs=kernels.mean_surface_eke(self.eastward.obs, self.northward.obs),
            area=self.area,
        )

    def kinetic_energy(self) -> Comparison:
        """Surface KE, likewise squared on the native grid before regridding."""
        return Comparison(
            label=f"{self.eastward.label} KE",
            native=0.5 * (self.eastward.native**2 + self.northward.native**2),
            obs=0.5 * (self.eastward.obs**2 + self.northward.obs**2),
            area=self.area,
        )


def align(
    model_field: xr.DataArray,
    obs_field: xr.DataArray,
    window: tuple[pd.Timestamp, pd.Timestamp] | None,
    context: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Restrict both fields to their common span and require timestamps to match.

    The requested window is intersected with the span the two products actually
    share, then exact agreement is required *within* that overlap. Products
    legitimately end at different dates -- DUACS's last centered 5-day mean
    predates OM4's final timestamp, for instance -- and demanding identical
    spans would make the caller responsible for knowing every product's exact
    coverage. Trimming the edges is safe; a disagreement anywhere inside the
    overlap is still an error, which is the property that actually matters.

    A `None` window means the full shared span, which is what the
    residual-variance and spectral diagnostics use: they characterise the whole
    rollout rather than the primary scoring years.
    """
    shared = (
        max(
            pd.Timestamp(model_field["time"].values.min()),
            pd.Timestamp(obs_field["time"].values.min()),
        ),
        min(
            pd.Timestamp(model_field["time"].values.max()),
            pd.Timestamp(obs_field["time"].values.max()),
        ),
    )
    if window is not None:
        requested = window
        window = (max(shared[0], window[0]), min(shared[1], window[1]))
        if window != requested:
            logger.info(
                "  %s: window trimmed to shared coverage %s..%s (requested %s..%s)",
                context,
                f"{window[0]:%Y-%m-%d}",
                f"{window[1]:%Y-%m-%d}",
                f"{requested[0]:%Y-%m-%d}",
                f"{requested[1]:%Y-%m-%d}",
            )
    else:
        window = shared

    if window[0] > window[1]:
        raise ValueError(
            f"{context}: model and observation coverage do not overlap "
            f"inside the requested window"
        )

    selection = slice(window[0], window[1])
    model_field = model_field.sel(time=selection)
    obs_field = obs_field.sel(time=selection)
    # Two empty axes compare equal, so the exact-match check below would pass
    # and the emptiness would only surface much later.
    if model_field.sizes.get("time", 0) == 0 or obs_field.sizes.get("time", 0) == 0:
        raise ValueError(
            f"{context}: no samples inside the requested window "
            f"{window[0]:%Y-%m-%d} to {window[1]:%Y-%m-%d}"
        )
    kernels.require_exact_time_match(model_field, obs_field, context)
    return model_field, obs_field


def whole_years(
    model_field: xr.DataArray, obs_field: xr.DataArray, context: str
) -> tuple[xr.DataArray, xr.DataArray]:
    """Restrict a monthly pair to the calendar years both cover in full.

    Equal-year blocks are what make the score and its bootstrap comparable, so
    a year missing a month has to go rather than be silently down-weighted.
    Dropping it here -- where the shortfall is visible and attributable --
    beats failing later in the kernel, which can only report that some year is
    ragged without knowing why.
    """
    months = pd.DatetimeIndex(model_field["time"].values)
    counts = pd.Series(1, index=months).groupby(months.year).sum()
    whole = sorted(int(str(year)) for year, n in counts.items() if n == 12)
    if not whole:
        raise ValueError(
            f"{context}: no calendar year is covered by twelve whole months "
            f"(months available: {months.min():%Y-%m} to {months.max():%Y-%m})"
        )
    dropped = sorted({int(str(year)) for year in counts.index} - set(whole))
    if dropped:
        logger.info(
            "  %s: scoring %d-%d; dropped partly covered year(s) %s",
            context,
            whole[0],
            whole[-1],
            dropped,
        )
    # By year membership, not a slice: a slice would only trim the ends and
    # leave a ragged interior year in place.
    keep = model_field["time"].dt.year.isin(whole)
    return model_field.sel(time=keep), obs_field.sel(time=keep)


def surface_velocity(
    rollout: xr.Dataset,
    duacs: xr.Dataset,
    window: tuple[pd.Timestamp, pd.Timestamp] | None,
    context: str,
    velocity_kind: str = "absolute",
) -> VelocityComparison:
    """Surface geostrophic velocity, derived from the rollout's sea level.

    The model has velocities of its own, but DUACS reports geostrophic velocity
    derived from altimetric sea level, so the comparable model field is derived
    the same way.
    """
    obs_u, obs_v = observations.duacs_velocity(duacs, velocity_kind)
    sim_u, sim_v = kernels.geostrophic_velocity_from_zos(
        rollout["zos"], lat_dim="lat", lon_dim="lon"
    )
    sim_u, obs_u = align(sim_u, obs_u, window, f"{context} eastward velocity")
    sim_v, obs_v = align(sim_v, obs_v, window, f"{context} northward velocity")
    area = duacs["area"]
    return VelocityComparison(
        eastward=Comparison(f"{context} eastward velocity", sim_u, obs_u, area),
        northward=Comparison(f"{context} northward velocity", sim_v, obs_v, area),
        kind=velocity_kind,
    )._rebased()


def sea_surface_temperature(
    rollout: xr.Dataset,
    oisst: xr.Dataset,
    window: tuple[pd.Timestamp, pd.Timestamp] | None,
    context: str,
) -> Comparison:
    """The rollout's top model level against OISST."""
    obs = oisst[observations.find_var_name(oisst, observations.SST_ALIASES)]
    sim, obs = align(rollout["thetao"].isel(lev=0), obs, window, f"{context} SST")
    return Comparison(f"{context} SST", sim, obs, oisst["area"])


def ohc_layer(
    rollout: xr.Dataset,
    argo: xr.Dataset,
    layer: kernels.DepthLayer,
    window: tuple[pd.Timestamp, pd.Timestamp] | None,
    context: str,
    model_dz: xr.DataArray,
    complete_years_only: bool = True,
) -> LayerComparison:
    """Per-area ocean heat content over a depth layer, against ARGO-IAP.

    ARGO-IAP is monthly, so both sides reduce to months and partially covered
    months are dropped rather than compared against full ones. `complete_years_
    only` additionally trims to whole calendar years, which the equal-year block
    reductions require; the diagnostics that are not scored that way pass False.
    """
    obs_temp = argo[
        observations.find_var_name(argo, observations.ARGO_TEMPERATURE_ALIASES)
    ]
    obs_depth = observations.find_coord_name(obs_temp, observations.DEPTH_ALIASES)
    assert obs_depth is not None  # find_coord_name raises when required

    obs = kernels.monthly_mean_of_complete_months(
        kernels.ohc_per_area_layer_maps(obs_temp, depth_name=obs_depth)[layer.label]
    )
    sim = kernels.monthly_mean_of_complete_months(
        kernels.ohc_per_area_layer_maps(
            rollout["thetao"], native_dz=model_dz, depth_name="lev"
        )[layer.label]
    )
    label = f"{context} OHC {layer.label}"
    sim, obs = align(sim, obs, window, label)
    if complete_years_only:
        # A rollout ending 24 December contributes no December once partial
        # months are dropped, so scoring its final year would weigh 11 months
        # against 12.
        sim, obs = whole_years(sim, obs, label)
    # Bathymetry does not change with time, so one step answers these and
    # spares a pass over the whole record.
    model_column = rollout["thetao"].isel(time=0)
    observed_column = obs_temp.isel(time=0)
    model_metres = kernels.integrated_layer_thickness(
        model_column, layer, native_dz=model_dz, depth_name="lev"
    )
    observed_metres = kernels.integrated_layer_thickness(
        observed_column, layer, depth_name=obs_depth
    )
    return LayerComparison(
        label,
        sim,
        obs,
        argo["area"],
        bathymetry_disagreement_m=kernels.area_weighted_mean(
            abs(
                kernels.model_field_on_obs_grid(model_metres, observed_metres)
                - observed_metres
            ),
            argo["area"],
        ),
        model_partial_columns=kernels.partial_column_fraction(
            model_column, layer, native_dz=model_dz, depth_name="lev"
        ),
    )


def model_depth_thickness(rollout: xr.Dataset, data_layout: DataLayout) -> xr.DataArray:
    """Re-exported so callers need only this module to build a comparison."""
    return observations.model_depth_thickness(rollout, data_layout)
