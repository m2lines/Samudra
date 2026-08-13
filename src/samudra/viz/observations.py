# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Figures comparing a rollout against observations.

Every number drawn here comes from `samudra.metrics`, the same kernels
`samudra.eval` reduces with, so a value printed on a figure is the value logged
to W&B rather than a second implementation that happens to agree.

Figures are written to disk rather than logged as images: a full suite runs to
tens of megabytes per eval, which is worth a bucket and not worth W&B's
storage.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

import cartopy.crs as ccrs  # type: ignore
import cartopy.feature as cfeature  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes  # type: ignore

from samudra.constants import build_om4_spec
from samudra.metrics import comparisons, kernels, spectra
from samudra.viz.norms import percentile_norm, symmetric_percentile_norm

logger = logging.getLogger(__name__)

# viz assumes OM4 throughout; the observation figures inherit that.
_OM4_SPEC = build_om4_spec(prognostic_vars_key="thermo_dynamic_all")

PROJECTION = ccrs.Robinson(central_longitude=-150)
PLATE = ccrs.PlateCarree()

# Colour per run, with the reference first. Distinguishable in greyscale and to
# the most common colour-vision deficiencies.
SERIES_COLOURS = ("#000000", "#0072b2", "#d55e00", "#009e73", "#cc79a7")


# What an RMSE-map builder hands back: squared error over time, and the cell
# areas to weight it by.
_SquaredError = tuple[xr.DataArray, xr.DataArray]


def save(figure: plt.Figure, directory: str, name: str) -> str:
    """Write a figure as PDF and close it. Returns the path."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}.pdf")
    figure.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(figure)
    logger.info("  wrote %s", path)
    return path


def _annotate(value: float, units: str, row: pd.Series | None = None) -> str:
    """Format a metric for a panel title, with its interval when there is one."""
    if not np.isfinite(value):
        return "unavailable"
    text = f"{value:.4g} {units}".strip()
    if row is None:
        return text
    low, high = row.get("ci_low"), row.get("ci_high")
    if low is not None and high is not None and np.isfinite(low) and np.isfinite(high):
        text += f" [{low:.4g}, {high:.4g}]"
    return text


def _map_axes(axis: GeoAxes) -> None:
    axis.add_feature(cfeature.LAND, facecolor="0.85", zorder=2)
    axis.coastlines(linewidth=0.3, color="0.4", zorder=3)
    axis.set_global()


def map_panels(
    fields: dict[str, xr.DataArray],
    title: str,
    units: str,
    subtitles: dict[str, str] | None = None,
    cmap: str = "viridis",
    diverging: bool = False,
) -> plt.Figure:
    """One map per run, on a shared colour scale so panels are comparable."""
    names = list(fields)
    figure, axes = plt.subplots(
        1,
        len(names),
        figsize=(6.5 * len(names), 4.2),
        subplot_kw={"projection": PROJECTION},
        squeeze=False,
    )
    # Shared percentile limits, so one outlier cell cannot flatten every panel.
    # `viz.core` already computes these, including the degenerate cases a
    # constant or all-NaN field produces.
    pooled = list(fields.values())
    norm = symmetric_percentile_norm(pooled) if diverging else percentile_norm(pooled)

    mesh = None
    for axis, name in zip(axes.flat, names, strict=True):
        field = fields[name]
        mesh = axis.pcolormesh(
            field["lon"],
            field["lat"],
            field.values,
            transform=PLATE,
            cmap=cmap,
            norm=norm,
            shading="auto",
        )
        _map_axes(axis)
        label = name if subtitles is None else f"{name}\n{subtitles.get(name, '')}"
        axis.set_title(label, fontsize=10)

    if mesh is not None:
        bar = figure.colorbar(
            mesh, ax=list(axes.flat), orientation="horizontal", fraction=0.05, pad=0.05
        )
        bar.set_label(units)
    figure.suptitle(title, fontsize=12)
    return figure


def series_panel(
    series: dict[str, pd.Series],
    title: str,
    ylabel: str,
    annotations: dict[str, str] | None = None,
) -> plt.Figure:
    """One line per run against a shared time axis."""
    figure, axis = plt.subplots(figsize=(11, 4))
    for (name, values), colour in zip(series.items(), SERIES_COLOURS, strict=False):
        label = name if annotations is None else f"{name} — {annotations.get(name, '')}"
        axis.plot(values.index, values.to_numpy(), label=label, color=colour, lw=1.2)
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=12)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    return figure


def spectra_panel(
    curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    title: str,
    xlabel: str,
    ylabel: str,
    scores: dict[str, dict[str, float]] | None = None,
) -> plt.Figure:
    """One panel per region, each with every run's spectrum on log-log axes."""
    regions = list(next(iter(curves.values()), {}))
    if not regions:
        regions = ["(no data)"]
    figure, axes = plt.subplots(
        1, len(regions), figsize=(5.2 * len(regions), 4.2), squeeze=False
    )

    for axis, region in zip(axes.flat, regions, strict=True):
        for (name, per_region), colour in zip(
            curves.items(), SERIES_COLOURS, strict=False
        ):
            x, y = per_region.get(region, (np.array([]), np.array([])))
            if x.size == 0:
                continue
            label = name
            if scores is not None:
                score = scores.get(name, {}).get(region)
                if score is not None and np.isfinite(score):
                    label = f"{name} ({score:.3f} dex)"
            axis.loglog(x, y, label=label, color=colour, lw=1.4)
        axis.set_title(region, fontsize=10)
        axis.set_xlabel(xlabel)
        axis.grid(alpha=0.3, which="both")
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8)
        else:
            axis.text(
                0.5,
                0.5,
                "no spectrum:\nbox too small or\nnot fully observed",
                ha="center",
                va="center",
                transform=axis.transAxes,
                fontsize=9,
                color="0.5",
            )
    axes.flat[0].set_ylabel(ylabel)
    figure.suptitle(title, fontsize=12)
    return figure


def interannual_spectra_panel(
    bands: dict[str, dict[str, tuple]],
    title: str,
    xlabel: str,
    ylabel: str,
) -> plt.Figure:
    """Per-region geometric-mean spectra with their year-to-year spread.

    The shaded band is +/- one log standard deviation across years, so a model
    whose spectrum sits inside the observed band is within natural variability
    rather than merely close on average.
    """
    regions = list(next(iter(bands.values()), {})) or ["(no data)"]
    figure, axes = plt.subplots(
        1, len(regions), figsize=(5.2 * len(regions), 4.2), squeeze=False
    )
    for axis, region in zip(axes.flat, regions, strict=True):
        for (name, per_region), colour in zip(
            bands.items(), SERIES_COLOURS, strict=False
        ):
            band = per_region.get(region)
            if band is None:
                continue
            x, mean, lower, upper, years = band
            if getattr(x, "size", 0) == 0:
                continue
            axis.fill_between(x, lower, upper, color=colour, alpha=0.15, linewidth=0)
            axis.loglog(x, mean, color=colour, lw=1.6, label=f"{name} ({years} yr)")
        axis.set_title(region, fontsize=10)
        axis.set_xlabel(xlabel)
        axis.grid(alpha=0.3, which="both")
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8)
        else:
            axis.text(
                0.5,
                0.5,
                "no spectrum:\nbox too small or\nnot fully observed",
                ha="center",
                va="center",
                transform=axis.transAxes,
                fontsize=9,
                color="0.5",
            )
    axes.flat[0].set_ylabel(ylabel)
    figure.suptitle(title, fontsize=12)
    return figure


def annual_rmse_panel(frame: pd.DataFrame, title: str) -> plt.Figure:
    """Per-year totals behind each headline score, with its interval.

    A single number hides whether a model is steadily wrong or wrong in one
    year, which is the difference between a bias and an event.
    """
    annual = frame[frame["period_kind"] == "annual"]
    primary = frame[frame["period_kind"] == "primary_complete_years"]
    metrics = list(dict.fromkeys(annual["metric"] + annual["depth"].fillna("")))
    if not metrics:
        metrics = ["(none)"]

    figure, axes = plt.subplots(
        1, len(metrics), figsize=(4.6 * len(metrics), 3.8), squeeze=False
    )
    for axis, key in zip(axes.flat, metrics, strict=True):
        rows = annual[(annual["metric"] + annual["depth"].fillna("")) == key]
        for (name, group), colour in zip(
            rows.groupby("model"), SERIES_COLOURS, strict=False
        ):
            group = group.sort_values("year")
            axis.plot(
                group["year"],
                group["value"],
                marker="o",
                color=colour,
                label=name,
                lw=1.2,
            )
            head = primary[
                (primary["model"] == name)
                & ((primary["metric"] + primary["depth"].fillna("")) == key)
            ]
            if not head.empty:
                row = head.iloc[0]
                axis.axhline(row["value"], color=colour, ls="--", lw=0.9, alpha=0.7)
                if np.isfinite(row.get("ci_low", np.nan)):
                    axis.axhspan(
                        row["ci_low"], row["ci_high"], color=colour, alpha=0.10, lw=0
                    )
        axis.set_title(key, fontsize=9)
        axis.set_xlabel("year")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    axes.flat[0].set_ylabel("total RMSE")
    figure.suptitle(title, fontsize=12)
    return figure


def rmse_map_figures(
    rollouts: dict[str, xr.Dataset],
    products: dict[str, xr.Dataset],
    frame: pd.DataFrame,
    window: tuple[pd.Timestamp, pd.Timestamp],
    directory: str,
    velocity_kind: str = "absolute",
) -> list[str]:
    """Per-cell RMSE maps for every metric that produces one.

    Recomputes the maps rather than reading them back: `eval` emits scalars and
    rows, not fields, and using the same kernels keeps the annotated totals
    identical to the logged ones.
    """
    written: list[str] = []
    duacs, oisst, argo = products["duacs"], products["oisst"], products["argo"]

    def scalar(metric: str, model: str, depth: str | None = None) -> pd.Series | None:
        rows = frame[
            (frame["metric"] == metric)
            & (frame["model"] == model)
            & (frame["period_kind"] == "primary_complete_years")
            & (frame["depth"].isna() if depth is None else frame["depth"] == depth)
        ]
        return None if rows.empty else rows.iloc[0]

    def emit(
        name: str,
        metric: str,
        title: str,
        units: str,
        build: Callable[[str, xr.Dataset], tuple[xr.DataArray, xr.DataArray]],
        depth: str | None = None,
        cmap: str = "magma",
    ) -> None:
        maps, notes = {}, {}
        for model, rollout in rollouts.items():
            row = scalar(metric, model, depth)
            try:
                error_squared, area = build(model, rollout)
                # Reduce through the same kernel `eval` scores with, so the map
                # and the total annotated on it are one calculation. A plain
                # time mean here would weight years by their sample count while
                # the annotated total weights them equally.
                maps[model] = kernels.rmse_map_with_uncertainty(
                    error_squared,
                    area,
                    ("lat", "lon"),
                    f"{model} {metric}",
                    bootstrap_samples=0,
                )[0]
            except (ValueError, KeyError) as error:
                # A metric that was scored must get the map that explains it:
                # dropping the figure while its scalar goes to W&B leaves a
                # number nobody can check. A metric the frame never scored has
                # nothing to draw, and saying so is enough.
                if row is not None:
                    raise ValueError(
                        f"{model}: {metric} has a scalar in the metric frame "
                        f"but its RMSE map could not be built"
                    ) from error
                logger.info("  %s: %s was not scored, so no map", model, metric)
                continue
            notes[model] = (
                _annotate(float(row["value"]), units, row) if row is not None else ""
            )
        if maps:
            written.append(
                save(map_panels(maps, title, units, notes, cmap=cmap), directory, name)
            )

    # Derived once per model: the velocity pair costs a geostrophic derivation
    # and an alignment over the whole window, and both maps below want it.
    velocities: dict[str, comparisons.VelocityComparison] = {}

    def velocity_of(model: str, rollout: xr.Dataset) -> comparisons.VelocityComparison:
        if model not in velocities:
            velocities[model] = comparisons.surface_velocity(
                rollout, duacs, window, model, velocity_kind
            )
        return velocities[model]

    def velocity_map(model: str, rollout: xr.Dataset) -> _SquaredError:
        velocity = velocity_of(model, rollout)
        return velocity.vector_error_squared, velocity.area

    def eke_map(model: str, rollout: xr.Dataset) -> _SquaredError:
        eke = velocity_of(model, rollout).eddy_kinetic_energy()
        return eke.error_squared, eke.area

    def sst_map(model: str, rollout: xr.Dataset) -> _SquaredError:
        sst = comparisons.sea_surface_temperature(rollout, oisst, window, model)
        return sst.error_squared, sst.area

    emit(
        "surface_geostrophic_velocity_rmse_vs_duacs",
        "surface_geostrophic_velocity_vector_total_rmse",
        "Surface geostrophic velocity vector RMSE vs DUACS",
        "m s-1",
        velocity_map,
    )
    emit(
        "surface_eke_rmse_vs_duacs",
        "instantaneous_surface_eke_total_rmse",
        "Instantaneous surface EKE RMSE vs DUACS",
        "m2 s-2",
        eke_map,
    )
    emit(
        "sst_rmse_vs_oisst",
        "surface_sst_total_rmse",
        "SST RMSE vs OISST",
        "degC",
        sst_map,
    )

    for layer in kernels.OHC_LAYERS:

        def ohc_map(model: str, rollout: xr.Dataset, layer=layer) -> _SquaredError:
            ohc = comparisons.ohc_layer(
                rollout,
                argo,
                layer,
                window,
                model,
                comparisons.model_depth_thickness(rollout, _OM4_SPEC),
            )
            return ohc.error_squared, ohc.area

        slug = layer.label.split("(")[-1].strip(") ").replace("-", "_")
        emit(
            f"ohc_per_area_rmse_vs_argo_iap_{slug}",
            "ohc_per_area_total_rmse",
            f"OHC per-area RMSE vs ARGO-IAP, {layer.label}",
            "J m-2",
            ohc_map,
            depth=layer.label,
        )

    return written


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


def variance_map_figures(
    rollouts: dict[str, xr.Dataset],
    products: dict[str, xr.Dataset],
    frame: pd.DataFrame,
    directory: str,
) -> list[str]:
    """Residual-variance maps: how much unforced variability, and where.

    Variance is computed on each model's native grid and only the resulting map
    regridded. Interpolating the field first would smooth it and damp the
    variance by an amount set by the resolution ratio.

    Every panel covers the period all runs and the observations share, so the
    reference is not a longer record drawn beside shorter ones on a common
    colour scale -- and it matches the span behind the annotated numbers.
    """
    written: list[str] = []
    oisst, argo = products["oisst"], products["argo"]

    def note(metric: str, model: str, units: str, drawn: tuple[str, str]) -> str:
        rows = frame[
            (frame["metric"] == metric)
            & (frame["model"] == model)
            & (frame["period_kind"] == "full_overlap")
        ]
        if rows.empty:
            return ""
        row = rows.iloc[0]
        text = _annotate(float(row["value"]), units)
        # `report` scores each run over its own overlap with the product, while
        # the panels share one window so they can share a colour scale. With
        # runs of equal coverage those agree; when they do not, say so rather
        # than print a number for a period the map does not cover.
        scored = (str(row["period_start"]), str(row["period_end"]))
        if scored != drawn:
            text += f" (scored {scored[0]}..{scored[1]})"
        return text

    def draw(
        items: dict[str, comparisons.Comparison],
        reference_name: str,
        slug: str,
        title: str,
        units: str,
        rmse_metric: str,
        corr_metric: str,
    ) -> None:
        reference, items = _over_one_window(items)
        stamps = pd.DatetimeIndex(reference["time"].values)
        drawn = (f"{stamps.min():%Y-%m-%d}", f"{stamps.max():%Y-%m-%d}")
        obs_variance = kernels.residual_variance_map(reference)
        maps = {reference_name: obs_variance}
        notes = {reference_name: ""}
        for model, item in items.items():
            maps[model] = kernels.model_field_on_obs_grid(
                kernels.residual_variance_map(item.native), obs_variance
            )
            notes[model] = (
                f"map RMSE {note(rmse_metric, model, units, drawn)}; "
                f"corr {note(corr_metric, model, '', drawn)}"
            )
        written.append(
            save(
                map_panels(
                    maps,
                    f"{title}, {drawn[0]} to {drawn[1]}",
                    units,
                    notes,
                    cmap="cividis",
                ),
                directory,
                slug,
            )
        )

    draw(
        {
            model: comparisons.sea_surface_temperature(rollout, oisst, None, model)
            for model, rollout in rollouts.items()
        },
        "OISST",
        "sst_residual_variance_vs_oisst",
        "SST residual-anomaly variance vs OISST",
        "degC2",
        "surface_sst_residual_variance_map_rmse",
        "surface_sst_residual_variance_pattern_corr",
    )

    upper = kernels.OHC_LAYERS[0]
    draw(
        {
            model: comparisons.ohc_layer(
                rollout,
                argo,
                upper,
                None,
                model,
                comparisons.model_depth_thickness(rollout, _OM4_SPEC),
                complete_years_only=False,
            )
            for model, rollout in rollouts.items()
        },
        "ARGO-IAP",
        "ohc_upper700_residual_variance_vs_argo_iap",
        "Upper-700 m OHC residual-anomaly variance vs ARGO-IAP",
        "(J m-2)2",
        "ohc_upper700_per_area_residual_variance_map_rmse",
        "ohc_upper700_per_area_residual_variance_pattern_corr",
    )
    return written


def timeseries_figures(
    rollouts: dict[str, xr.Dataset],
    products: dict[str, xr.Dataset],
    directory: str,
    velocity_kind: str = "absolute",
) -> list[str]:
    """Global-mean series: the trend, the seasonal cycle, and what is left.

    The residual is the quantity of interest. Trend and seasonality are the easy
    parts to reproduce; what remains is the variability a model either captures
    or smooths away.
    """
    written: list[str] = []
    duacs, oisst, argo = products["duacs"], products["oisst"], products["argo"]

    def draw(
        items,
        reference_name: str,
        slug: str,
        title: str,
        units: str,
        derive=None,
    ) -> None:
        reference, items = _over_one_window(items)
        if derive is not None:
            # Anything reducing over time has to be derived after the window is
            # settled: an eddy anomaly is taken about the mean of whatever
            # record it is handed, and trimming afterwards cannot undo that.
            items = {name: derive(item) for name, item in items.items()}
            reference = next(iter(items.values())).obs
        area = next(iter(items.values())).area
        reference, runs = _on_common_cells(
            reference, {name: item.on_obs_grid for name, item in items.items()}
        )
        series = {reference_name: _global_mean(reference, area)}
        series.update({name: _global_mean(f, area) for name, f in runs.items()})
        if len(series) < 2:
            return

        trends = {
            name: (
                f"trend {kernels.series_linear_trend_per_year(values):+.3g} {units}/yr; "
                f"residual var {kernels.series_residual_variance(values):.3g}"
            )
            for name, values in series.items()
        }
        written.append(
            save(series_panel(series, title, units, trends), directory, slug)
        )

        residuals = {
            name: kernels.series_without_linear_trend(
                kernels.series_without_seasonal_cycle(values)
            )
            for name, values in series.items()
        }
        written.append(
            save(
                series_panel(
                    residuals, f"{title} — detrended and deseasonalized", units
                ),
                directory,
                f"{slug}_residual",
            )
        )

    draw(
        {
            model: comparisons.sea_surface_temperature(rollout, oisst, None, model)
            for model, rollout in rollouts.items()
        },
        "OISST",
        "global_sst",
        "Global mean SST",
        "degC",
    )
    draw(
        {
            model: comparisons.surface_velocity(
                rollout, duacs, None, model, velocity_kind
            )
            for model, rollout in rollouts.items()
        },
        "DUACS",
        "global_surface_eke",
        "Global mean surface EKE",
        "m2 s-2",
        derive=lambda velocity: velocity.eddy_kinetic_energy(),
    )
    upper = kernels.OHC_LAYERS[0]
    draw(
        {
            model: comparisons.ohc_layer(
                rollout,
                argo,
                upper,
                None,
                model,
                comparisons.model_depth_thickness(rollout, _OM4_SPEC),
                complete_years_only=False,
            )
            for model, rollout in rollouts.items()
        },
        "ARGO-IAP",
        "global_ohc_upper700",
        "Global mean upper-700 m OHC per area",
        "J m-2",
    )
    return written


def _global_mean(field: xr.DataArray, area: xr.DataArray) -> pd.Series:
    """Area-weighted global mean as a pandas series."""
    values = kernels.area_weighted_timeseries(field, area, ("lat", "lon"))
    return pd.Series(
        np.asarray(values.values, dtype=float),
        index=pd.DatetimeIndex(values["time"].values),
    )


def spectra_figures(
    rollouts: dict[str, xr.Dataset],
    products: dict[str, xr.Dataset],
    directory: str,
    velocity_kind: str = "absolute",
) -> list[str]:
    """Spatial and temporal spectra: at which scales the model is wrong.

    A model can carry the right total variance and still distribute it over the
    wrong wavelengths. These are the only figures that show that directly, and
    the log10 scores beside each curve say how far off it is in dex.

    Every observation curve is reduced over the span the runs share. That is not
    only labelling: an eddy anomaly is taken about the mean of whatever record
    it is handed, so a reference built from the product's full archive is not
    the same quantity as a run built from eight years of it.
    """
    written: list[str] = []
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
        save(
            spectra_panel(
                eke_curves,
                "Surface EKE spatial spectra vs DUACS",
                "wavenumber (rad km-1)",
                "k x PSD",
                _curve_scores(eke_curves, "DUACS"),
            ),
            directory,
            "eke_spatial_spectra_vs_duacs",
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
        save(
            spectra_panel(
                ssta_curves,
                "SST anomaly spatial spectra vs OISST",
                "wavenumber (rad km-1)",
                "k x PSD",
                _curve_scores(ssta_curves, "OISST"),
            ),
            directory,
            "ssta_spatial_spectra_vs_oisst",
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
        save(
            spectra_panel(
                ke_curves,
                "Surface geostrophic KE temporal spectra vs DUACS",
                "frequency (cycles yr-1)",
                "PSD",
                _curve_scores(ke_curves, "DUACS"),
            ),
            directory,
            "ke_temporal_spectra_vs_duacs",
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
        save(
            interannual_spectra_panel(
                eke_bands,
                "Interannual surface EKE spatial spectra",
                "wavenumber (rad km-1)",
                "k x PSD",
            ),
            directory,
            "eke_spatial_spectra_interannual",
        )
    )

    ssta_bands = {"OISST": _yearly_bands(_anomaly_by_year(obs_ssta))}
    for model, item in sst.items():
        ssta_bands[model] = _yearly_bands(
            _anomaly_by_year(kernels.without_seasonal_cycle(item.native))
        )
    written.append(
        save(
            interannual_spectra_panel(
                ssta_bands,
                "Interannual SST anomaly spatial spectra",
                "wavenumber (rad km-1)",
                "k x PSD",
            ),
            directory,
            "ssta_spatial_spectra_interannual",
        )
    )

    ke_bands = {"DUACS": _yearly_temporal_bands(obs_ke)}
    for model, field in ke_runs.items():
        ke_bands[model] = _yearly_temporal_bands(field)
    written.append(
        save(
            interannual_spectra_panel(
                ke_bands,
                "Interannual surface geostrophic KE temporal spectra",
                "frequency (cycles yr-1)",
                "PSD",
            ),
            directory,
            "ke_temporal_spectra_interannual",
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
