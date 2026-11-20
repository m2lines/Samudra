# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Shared spectral results consumed by scalar reporting and figures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from samudra.metrics import comparisons, kernels, spectra


@dataclass
class SpectralDiagnostic:
    kind: Literal["pooled", "interannual"]
    curves: dict[str, dict[str, Any]]
    title: str
    xlabel: str
    ylabel: str
    name: str
    time: xr.DataArray
    scores: dict[str, dict[str, float]] | None = None


def _on_common_cells(
    reference: xr.DataArray, runs: dict[str, xr.DataArray]
) -> tuple[xr.DataArray, dict[str, xr.DataArray]]:
    """Restrict the reference and every run to the cells they all carry.

    A run covers only the observation cells that survived pairing: its own land,
    plus the band regridding erodes around it, which is coastal and so is where
    the energy and the error both are. Reducing the reference over every cell
    the product has and each run over its own subset makes the gap between the
    curves a property of the masks, which a mean-state plot then presents as
    bias.
    """
    mask = None
    for field in runs.values():
        ever = comparisons.ever_finite(field)
        mask = ever if mask is None else (mask & ever)
    if mask is None:
        return reference, runs
    return reference.where(mask), {
        name: field.where(mask) for name, field in runs.items()
    }


def _over_one_window(items):
    """Trim every comparison to the span they all cover; return the reference too.

    A panel draws several runs beside one observation curve on one colour scale
    and one legend, so the reference has to cover the runs' period and no more,
    and the runs have to cover each other's. Built with a `None` window, each
    comparison spans only what that run and the product share; this intersects
    those.
    """
    starts = [pd.Timestamp(c.time.values.min()) for c in items.values()]
    ends = [pd.Timestamp(c.time.values.max()) for c in items.values()]
    window = slice(max(starts), min(ends))
    trimmed = {name: c.slice(window) for name, c in items.items()}
    first = next(iter(trimmed.values()))
    reference = first.obs if hasattr(first, "obs") else first.eastward.obs
    return reference, trimmed


def spectral_diagnostics(
    rollouts: dict[str, xr.Dataset],
    products: dict[str, xr.Dataset],
    velocity_kind: str = "absolute",
) -> list[SpectralDiagnostic]:
    """Spatial and temporal spectra: at which scales the model is wrong.

    A model can carry the right total variance and still distribute it over the
    wrong wavelengths. These are the only figures that show that directly, and
    the log10 scores beside each curve say how far off it is in dex.

    Every observation curve is reduced over the span the runs share. That is not
    only labelling: an eddy anomaly is taken about the mean of whatever record
    it is handed, so a reference built from the product's full archive is not
    the same quantity as a run built from eight years of it.
    """
    written: list[SpectralDiagnostic] = []
    duacs, oisst = products["duacs"], products["oisst"]

    # Trimmed once, up front: the energies below reduce over time, so a mean
    # taken before the span is settled is a mean of the wrong record.
    _, velocity = _over_one_window(
        {
            model: comparisons.surface_velocity(
                rollout, duacs, None, model, velocity_kind
            )
            for model, rollout in rollouts.items()
        }
    )
    obs_sst, temperature = _over_one_window(
        {
            model: comparisons.sea_surface_temperature(rollout, oisst, None, model)
            for model, rollout in rollouts.items()
        }
    )

    # --- surface EKE, spatial ------------------------------------------------
    mean_eke = {m: v.mean_eddy_kinetic_energy() for m, v in velocity.items()}
    obs_eke = next(iter(mean_eke.values())).obs
    eke_curves = {"DUACS": _region_curves(obs_eke, spectra.SPATIAL_REGIONS)}
    for model, item in mean_eke.items():
        # The model's own grid, like every other spectrum here. Interpolating
        # onto the finer observation grid is a low-pass filter, so it damps the
        # high wavenumbers this figure exists to compare -- and the wavenumbers
        # come back in rad/km either way, so the curves stay comparable.
        eke_curves[model] = _region_curves(item.native, spectra.SPATIAL_REGIONS)
    written.append(
        SpectralDiagnostic(
            kind="pooled",
            curves=eke_curves,
            title="Surface EKE spatial spectra vs DUACS",
            xlabel="wavenumber (rad km-1)",
            ylabel="k x PSD",
            name="eke_spatial_spectra_vs_duacs",
            time=next(iter(velocity.values())).eastward.obs["time"],
            scores=_curve_scores(eke_curves, "DUACS"),
        )
    )

    # --- SST anomaly, spatial ------------------------------------------------
    sst = temperature
    obs_ssta = kernels.without_seasonal_cycle(obs_sst)
    ssta_curves = {"OISST": _region_curves(obs_ssta, spectra.SPATIAL_REGIONS)}
    for model, item in sst.items():
        ssta_curves[model] = _region_curves(
            kernels.without_seasonal_cycle(item.native), spectra.SPATIAL_REGIONS
        )
    written.append(
        SpectralDiagnostic(
            kind="pooled",
            curves=ssta_curves,
            title="SST anomaly spatial spectra vs OISST",
            xlabel="wavenumber (rad km-1)",
            ylabel="k x PSD",
            name="ssta_spatial_spectra_vs_oisst",
            time=obs_sst["time"],
            scores=_curve_scores(ssta_curves, "OISST"),
        )
    )

    # --- surface KE, temporal ------------------------------------------------
    kinetic = {m: v.kinetic_energy() for m, v in velocity.items()}
    obs_ke = next(iter(kinetic.values())).obs
    obs_ke, ke_runs = _on_common_cells(
        obs_ke, {m: item.on_obs_grid for m, item in kinetic.items()}
    )
    ke_curves = {"DUACS": _temporal_curves(obs_ke, spectra.TEMPORAL_REGIONS)}
    ke_curves.update(
        {m: _temporal_curves(f, spectra.TEMPORAL_REGIONS) for m, f in ke_runs.items()}
    )
    written.append(
        SpectralDiagnostic(
            kind="pooled",
            curves=ke_curves,
            title="Surface geostrophic KE temporal spectra vs DUACS",
            xlabel="frequency (cycles yr-1)",
            ylabel="PSD",
            name="ke_temporal_spectra_vs_duacs",
            time=next(iter(velocity.values())).eastward.obs["time"],
            scores=_curve_scores(ke_curves, "DUACS"),
        )
    )

    # --- interannual bands ---------------------------------------------------
    first = next(iter(velocity.values()))
    eke_bands = {
        "DUACS": _yearly_bands(obs_eke_by_year(first.eastward.obs, first.northward.obs))
    }
    for model, item in velocity.items():
        eke_bands[model] = _yearly_bands(
            obs_eke_by_year(item.eastward.native, item.northward.native)
        )
    written.append(
        SpectralDiagnostic(
            kind="interannual",
            curves=eke_bands,
            title="Interannual surface EKE spatial spectra",
            xlabel="wavenumber (rad km-1)",
            ylabel="k x PSD",
            name="eke_spatial_spectra_interannual",
            time=first.eastward.obs["time"],
        )
    )

    ssta_bands = {"OISST": _yearly_bands(_anomaly_by_year(obs_ssta))}
    for model, item in sst.items():
        ssta_bands[model] = _yearly_bands(
            _anomaly_by_year(kernels.without_seasonal_cycle(item.native))
        )
    written.append(
        SpectralDiagnostic(
            kind="interannual",
            curves=ssta_bands,
            title="Interannual SST anomaly spatial spectra",
            xlabel="wavenumber (rad km-1)",
            ylabel="k x PSD",
            name="ssta_spatial_spectra_interannual",
            time=obs_sst["time"],
        )
    )

    ke_bands = {"DUACS": _yearly_temporal_bands(obs_ke)}
    for model, field in ke_runs.items():
        ke_bands[model] = _yearly_temporal_bands(field)
    written.append(
        SpectralDiagnostic(
            kind="interannual",
            curves=ke_bands,
            title="Interannual surface geostrophic KE temporal spectra",
            xlabel="frequency (cycles yr-1)",
            ylabel="PSD",
            name="ke_temporal_spectra_interannual",
            time=first.eastward.obs["time"],
        )
    )

    return written


def _anomaly_by_year(anomaly: xr.DataArray) -> dict[int, xr.DataArray]:
    """Split an anomaly field into complete calendar years, keeping the time axis.

    The anomaly must already be taken about the whole record's climatology, so
    that every year shares one baseline and the band shows real year-to-year
    spread rather than each year's deviation from itself.
    """
    return {
        year: anomaly.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
        for year in kernels.complete_calendar_years(anomaly["time"])
    }


def _yearly_temporal_bands(field: xr.DataArray) -> dict[str, tuple]:
    """Welch PSD of each complete calendar year, aggregated into a per-region band."""
    years = kernels.complete_calendar_years(field["time"])
    bands = {}
    for name, lon, lat in spectra.TEMPORAL_REGIONS:
        series = spectra.region_mean_series(field, lon, lat)
        stamps = pd.DatetimeIndex(series.index)
        curves = []
        for year in years:
            frequencies, power = spectra.welch_psd(series[stamps.year == year])
            if np.asarray(frequencies).size:
                curves.append((np.asarray(frequencies), np.asarray(power)))
        bands[name] = spectra.interannual_band(curves)
    return bands


def obs_eke_by_year(u: xr.DataArray, v: xr.DataArray) -> dict[int, xr.DataArray]:
    """Annual-mean EKE maps for each complete calendar year.

    A year the record only partly covers is dropped rather than averaged: the
    default rollout starts on 20 October, so its first year holds fifteen
    samples of one season and would enter the band as if it were a year.

    The eddy anomaly is taken about the mean of the whole record, not of each
    year: a per-year mean would absorb the year-to-year change in the mean flow,
    which is the variability these bands exist to show.
    """
    eke = kernels.instantaneous_surface_eke(u, v)
    return {
        year: eke.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).mean(
            "time", skipna=True
        )
        for year in kernels.complete_calendar_years(u["time"])
    }


def _region_curves(
    field: xr.DataArray, regions: tuple[tuple[str, slice, slice], ...]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        name: spectra.region_spectrum(field, lon, lat, name=name)
        for name, lon, lat in regions
    }


def _temporal_curves(
    field: xr.DataArray, regions: tuple[tuple[str, slice, slice], ...]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        name: spectra.welch_psd(spectra.region_mean_series(field, lon, lat))
        for name, lon, lat in regions
    }


def _yearly_bands(
    per_year: dict[int, xr.DataArray],
) -> dict[str, tuple]:
    bands = {}
    for name, lon, lat in spectra.SPATIAL_REGIONS:
        curves = [
            spectra.region_spectrum(field, lon, lat, name=f"{name} {year}")
            for year, field in per_year.items()
        ]
        bands[name] = spectra.interannual_band(
            [(x, y) for x, y in curves if getattr(x, "size", 0)]
        )
    return bands


def _curve_scores(
    curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]], reference: str
) -> dict[str, dict[str, float]]:
    """log10 RMSE of each run's spectrum against the reference, per region."""
    baseline = curves.get(reference, {})
    empty = (np.array([]), np.array([]))

    scores: dict[str, dict[str, float]] = {}
    for name, per_region in curves.items():
        if name == reference:
            continue
        scores[name] = {}
        for region, (wavenumber, power) in per_region.items():
            reference_wavenumber, reference_power = baseline.get(region, empty)
            scores[name][region] = spectra.log10_rmse_between_curves(
                reference_wavenumber, reference_power, wavenumber, power
            )
    return scores


def spectral_rows(results: list[SpectralDiagnostic]) -> list[dict]:
    """Export pooled log-spectrum errors, retaining unavailable comparisons as NaN."""
    rows = []
    for result in results:
        if result.kind != "pooled":
            continue
        stamps = pd.DatetimeIndex(result.time.values)
        for model, regions in (result.scores or {}).items():
            for region, value in regions.items():
                rows.append(
                    dict(
                        metric=result.name + "_log10_rmse",
                        model=model,
                        region=region,
                        value=value,
                        units="dex",
                        period_kind="full_overlap",
                        period_start=f"{stamps.min():%Y-%m-%d}",
                        period_end=f"{stamps.max():%Y-%m-%d}",
                        n_time_samples=len(stamps),
                    )
                )
    return rows
