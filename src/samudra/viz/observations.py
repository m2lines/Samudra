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
from collections.abc import Callable, Iterable

import cartopy.crs as ccrs  # type: ignore
import cartopy.feature as cfeature  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes  # type: ignore

from samudra.constants import build_om4_spec
from samudra.metrics import kernels, observations, spectra

logger = logging.getLogger(__name__)

# viz assumes OM4 throughout; the observation figures inherit that.
_OM4_SPEC = build_om4_spec(prognostic_vars_key="thermo_dynamic_all")

PROJECTION = ccrs.Robinson(central_longitude=-150)
PLATE = ccrs.PlateCarree()

# Colour per run, with the reference first. Distinguishable in greyscale and to
# the most common colour-vision deficiencies.
SERIES_COLOURS = ("#000000", "#0072b2", "#d55e00", "#009e73", "#cc79a7")

# The residual-variance and spectral diagnostics characterise the whole
# rollout rather than the scoring window, so they pass a span that never
# trims and let the shared coverage decide.
_FULL_SPAN = (pd.Timestamp.min, pd.Timestamp.max)

# Steps a calendar year needs before it contributes to an interannual band.
# Twelve is roughly two months at the 5-day cadence: enough for an annual mean
# not to be one season.
MIN_STEPS_PER_YEAR = 12

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


def _robust_limits(fields: Iterable[xr.DataArray], diverging: bool = False):
    """Percentile colour limits, so one outlier cell cannot flatten a map."""
    pooled = np.concatenate(
        [np.asarray(f.values).ravel() for f in fields] or [np.array([np.nan])]
    )
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        return 0.0, 1.0
    if diverging:
        limit = float(np.percentile(np.abs(pooled), 98))
        return -limit, limit
    return float(np.percentile(pooled, 2)), float(np.percentile(pooled, 98))


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
    vmin, vmax = _robust_limits(fields.values(), diverging=diverging)

    mesh = None
    for axis, name in zip(axes.flat, names, strict=True):
        field = fields[name]
        mesh = axis.pcolormesh(
            field["lon"],
            field["lat"],
            field.values,
            transform=PLATE,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
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
                logger.warning("  %s: no %s map (%s)", model, metric, error)
                continue
            row = scalar(metric, model, depth)
            notes[model] = (
                _annotate(float(row["value"]), units, row) if row is not None else ""
            )
        if maps:
            written.append(
                save(map_panels(maps, title, units, notes, cmap=cmap), directory, name)
            )

    obs_u, obs_v = observations.duacs_velocity(duacs)

    def velocity_map(_model: str, rollout: xr.Dataset) -> _SquaredError:
        sim_u, sim_v = kernels.geostrophic_velocity_from_zos(
            rollout["zos"], lat_dim="lat", lon_dim="lon"
        )
        u_model, u_obs = _paired(sim_u, obs_u, window)
        v_model, v_obs = _paired(sim_v, obs_v, window)
        squared = (u_model - u_obs) ** 2 + (v_model - v_obs) ** 2
        return squared, duacs["area"]

    def eke_map(_model: str, rollout: xr.Dataset) -> _SquaredError:
        sim_u, sim_v = kernels.geostrophic_velocity_from_zos(
            rollout["zos"], lat_dim="lat", lon_dim="lon"
        )
        u_native, u_obs = _paired(sim_u, obs_u, window, regrid=False)
        v_native, v_obs = _paired(sim_v, obs_v, window, regrid=False)
        model_eke = kernels.model_field_on_obs_grid(
            kernels.instantaneous_surface_eke(u_native, v_native), u_obs
        )
        obs_eke = kernels.instantaneous_surface_eke(u_obs, v_obs)
        return (model_eke - obs_eke) ** 2, duacs["area"]

    def sst_map(_model: str, rollout: xr.Dataset) -> _SquaredError:
        obs_sst = oisst[observations.find_var_name(oisst, observations.SST_ALIASES)]
        model, obs = _paired(rollout["thetao"].isel(lev=0), obs_sst, window)
        return (model - obs) ** 2, oisst["area"]

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

    obs_temp = argo[
        observations.find_var_name(argo, observations.ARGO_TEMPERATURE_ALIASES)
    ]
    obs_depth = observations.find_coord_name(obs_temp, observations.DEPTH_ALIASES)
    assert obs_depth is not None
    obs_layers = kernels.ohc_per_area_layer_maps(obs_temp, depth_name=obs_depth)

    for layer in kernels.OHC_LAYERS:

        def ohc_map(_model: str, rollout: xr.Dataset, layer=layer) -> _SquaredError:
            sim_layers = kernels.ohc_per_area_layer_maps(
                rollout["thetao"],
                native_dz=observations.model_depth_thickness(rollout, _OM4_SPEC),
                depth_name="lev",
            )
            model = kernels.monthly_mean_of_complete_months(sim_layers[layer.label])
            obs = kernels.monthly_mean_of_complete_months(obs_layers[layer.label])
            model, obs = _paired(model, obs, window)
            return (model - obs) ** 2, argo["area"]

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


def _paired(
    model_field: xr.DataArray,
    obs_field: xr.DataArray,
    window: tuple[pd.Timestamp, pd.Timestamp],
    regrid: bool = True,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Restrict both to their shared span and put the model on the obs grid."""
    shared = (
        max(
            pd.Timestamp(model_field["time"].values.min()),
            pd.Timestamp(obs_field["time"].values.min()),
            window[0],
        ),
        min(
            pd.Timestamp(model_field["time"].values.max()),
            pd.Timestamp(obs_field["time"].values.max()),
            window[1],
        ),
    )
    model_field = model_field.sel(time=slice(*shared))
    obs_field = obs_field.sel(time=slice(*shared))
    kernels.require_exact_time_match(model_field, obs_field, "figure")
    if regrid:
        model_field = kernels.model_field_on_obs_grid(model_field, obs_field)
    return model_field, obs_field


def _common_span(
    model_fields: Iterable[xr.DataArray], obs_field: xr.DataArray
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The period every run and the observations all cover."""
    fields = [*model_fields, obs_field]
    return (
        max(pd.Timestamp(f["time"].values.min()) for f in fields),
        min(pd.Timestamp(f["time"].values.max()) for f in fields),
    )


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

    def note(metric: str, model: str, units: str) -> str:
        rows = frame[
            (frame["metric"] == metric)
            & (frame["model"] == model)
            & (frame["period_kind"] == "full_overlap")
        ]
        return "" if rows.empty else _annotate(float(rows.iloc[0]["value"]), units)

    obs_sst = oisst[observations.find_var_name(oisst, observations.SST_ALIASES)]
    sst_fields = {m: r["thetao"].isel(lev=0) for m, r in rollouts.items()}
    sst_span = _common_span(sst_fields.values(), obs_sst)
    obs_sst_variance = kernels.residual_variance_map(obs_sst.sel(time=slice(*sst_span)))
    sst_maps = {"OISST": obs_sst_variance}
    sst_notes = {"OISST": ""}
    for model, sst_field in sst_fields.items():
        native, _ = _paired(sst_field, obs_sst, sst_span, regrid=False)
        # Regrid against the observation *variance map*, not the field: the
        # reference must already be reduced over time or the time axis
        # broadcasts back into the result.
        sst_maps[model] = kernels.model_field_on_obs_grid(
            kernels.residual_variance_map(native), obs_sst_variance
        )
        rmse = note("surface_sst_residual_variance_map_rmse", model, "degC2")
        corr = note("surface_sst_residual_variance_pattern_corr", model, "")
        sst_notes[model] = f"map RMSE {rmse}; corr {corr}"
    written.append(
        save(
            map_panels(
                sst_maps,
                "SST residual-anomaly variance vs OISST",
                "degC2",
                sst_notes,
                cmap="cividis",
            ),
            directory,
            "sst_residual_variance_vs_oisst",
        )
    )

    obs_temp = argo[
        observations.find_var_name(argo, observations.ARGO_TEMPERATURE_ALIASES)
    ]
    obs_depth = observations.find_coord_name(obs_temp, observations.DEPTH_ALIASES)
    assert obs_depth is not None
    upper = kernels.OHC_LAYERS[0]
    obs_ohc = kernels.monthly_mean_of_complete_months(
        kernels.ohc_per_area_layer_maps(obs_temp, depth_name=obs_depth)[upper.label]
    )

    ohc_fields = {
        model: kernels.monthly_mean_of_complete_months(
            kernels.ohc_per_area_layer_maps(
                rollout["thetao"],
                native_dz=observations.model_depth_thickness(rollout, _OM4_SPEC),
                depth_name="lev",
            )[upper.label]
        )
        for model, rollout in rollouts.items()
    }
    ohc_span = _common_span(ohc_fields.values(), obs_ohc)
    obs_ohc_variance = kernels.residual_variance_map(obs_ohc.sel(time=slice(*ohc_span)))
    ohc_maps = {"ARGO-IAP": obs_ohc_variance}
    ohc_notes = {"ARGO-IAP": ""}
    for model, sim in ohc_fields.items():
        native, _ = _paired(sim, obs_ohc, ohc_span, regrid=False)
        ohc_maps[model] = kernels.model_field_on_obs_grid(
            kernels.residual_variance_map(native), obs_ohc_variance
        )
        rmse = note(
            "ohc_upper700_per_area_residual_variance_map_rmse", model, "(J m-2)2"
        )
        corr = note("ohc_upper700_per_area_residual_variance_pattern_corr", model, "")
        ohc_notes[model] = f"map RMSE {rmse}; corr {corr}"
    written.append(
        save(
            map_panels(
                ohc_maps,
                "Upper-700 m OHC residual-anomaly variance vs ARGO-IAP",
                "(J m-2)2",
                ohc_notes,
                cmap="cividis",
            ),
            directory,
            "ohc_upper700_residual_variance_vs_argo_iap",
        )
    )
    return written


def timeseries_figures(
    rollouts: dict[str, xr.Dataset],
    products: dict[str, xr.Dataset],
    directory: str,
) -> list[str]:
    """Global-mean series: the trend, the seasonal cycle, and what is left.

    The residual is the quantity of interest. Trend and seasonality are the easy
    parts to reproduce; what remains is the variability a model either captures
    or smooths away.
    """
    written: list[str] = []
    duacs, oisst, argo = products["duacs"], products["oisst"], products["argo"]

    def draw(series: dict[str, pd.Series], slug: str, title: str, units: str) -> None:
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
                    residuals,
                    f"{title} — detrended and deseasonalized",
                    units,
                ),
                directory,
                f"{slug}_residual",
            )
        )

    obs_sst = oisst[observations.find_var_name(oisst, observations.SST_ALIASES)]
    sst = {"OISST": _global_mean(obs_sst, oisst["area"])}
    for model, rollout in rollouts.items():
        native, obs = _paired(
            rollout["thetao"].isel(lev=0), obs_sst, _FULL_SPAN, regrid=True
        )
        sst[model] = _global_mean(native, oisst["area"])
    draw(sst, "global_sst", "Global mean SST", "degC")

    obs_u, obs_v = observations.duacs_velocity(duacs)
    eke = {
        "DUACS": _global_mean(
            kernels.instantaneous_surface_eke(obs_u, obs_v), duacs["area"]
        )
    }
    for model, rollout in rollouts.items():
        sim_u, sim_v = kernels.geostrophic_velocity_from_zos(
            rollout["zos"], lat_dim="lat", lon_dim="lon"
        )
        u_native, u_obs = _paired(sim_u, obs_u, _FULL_SPAN, regrid=False)
        v_native, _ = _paired(sim_v, obs_v, _FULL_SPAN, regrid=False)
        model_eke = kernels.model_field_on_obs_grid(
            kernels.instantaneous_surface_eke(u_native, v_native), u_obs
        )
        eke[model] = _global_mean(model_eke, duacs["area"])
    draw(eke, "global_surface_eke", "Global mean surface EKE", "m2 s-2")

    obs_temp = argo[
        observations.find_var_name(argo, observations.ARGO_TEMPERATURE_ALIASES)
    ]
    obs_depth = observations.find_coord_name(obs_temp, observations.DEPTH_ALIASES)
    assert obs_depth is not None
    upper = kernels.OHC_LAYERS[0]
    obs_ohc = kernels.monthly_mean_of_complete_months(
        kernels.ohc_per_area_layer_maps(obs_temp, depth_name=obs_depth)[upper.label]
    )
    ohc = {"ARGO-IAP": _global_mean(obs_ohc, argo["area"])}
    for model, rollout in rollouts.items():
        sim = kernels.monthly_mean_of_complete_months(
            kernels.ohc_per_area_layer_maps(
                rollout["thetao"],
                native_dz=observations.model_depth_thickness(rollout, _OM4_SPEC),
                depth_name="lev",
            )[upper.label]
        )
        model_ohc, _ = _paired(sim, obs_ohc, _FULL_SPAN, regrid=True)
        ohc[model] = _global_mean(model_ohc, argo["area"])
    draw(ohc, "global_ohc_upper700", "Global mean upper-700 m OHC per area", "J m-2")

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
) -> list[str]:
    """Spatial and temporal spectra: at which scales the model is wrong.

    A model can carry the right total variance and still distribute it over the
    wrong wavelengths. These are the only figures that show that directly, and
    the log10 scores beside each curve say how far off it is in dex.
    """
    written: list[str] = []
    duacs, oisst = products["duacs"], products["oisst"]
    obs_u, obs_v = observations.duacs_velocity(duacs)

    # --- surface EKE, spatial ------------------------------------------------
    obs_eke = kernels.mean_surface_eke(obs_u, obs_v)
    eke_curves = {"DUACS": _region_curves(obs_eke, spectra.SPATIAL_REGIONS)}
    for model, rollout in rollouts.items():
        sim_u, sim_v = kernels.geostrophic_velocity_from_zos(
            rollout["zos"], lat_dim="lat", lon_dim="lon"
        )
        u_native, u_obs = _paired(sim_u, obs_u, _FULL_SPAN, regrid=False)
        v_native, _ = _paired(sim_v, obs_v, _FULL_SPAN, regrid=False)
        model_eke = kernels.model_field_on_obs_grid(
            kernels.mean_surface_eke(u_native, v_native), obs_eke
        )
        eke_curves[model] = _region_curves(model_eke, spectra.SPATIAL_REGIONS)
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
    obs_sst = oisst[observations.find_var_name(oisst, observations.SST_ALIASES)]
    obs_ssta = kernels.calendar_day_anomaly(obs_sst)
    ssta_curves = {"OISST": _region_curves(obs_ssta, spectra.SPATIAL_REGIONS)}
    for model, rollout in rollouts.items():
        native, obs = _paired(
            rollout["thetao"].isel(lev=0), obs_sst, _FULL_SPAN, regrid=True
        )
        ssta_curves[model] = _region_curves(
            kernels.calendar_day_anomaly(native), spectra.SPATIAL_REGIONS
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
    obs_ke = 0.5 * (obs_u**2 + obs_v**2)
    ke_curves = {"DUACS": _temporal_curves(obs_ke, spectra.TEMPORAL_REGIONS)}
    for model, rollout in rollouts.items():
        sim_u, sim_v = kernels.geostrophic_velocity_from_zos(
            rollout["zos"], lat_dim="lat", lon_dim="lon"
        )
        u_model, u_obs = _paired(sim_u, obs_u, _FULL_SPAN)
        v_model, _ = _paired(sim_v, obs_v, _FULL_SPAN)
        ke_curves[model] = _temporal_curves(
            0.5 * (u_model**2 + v_model**2), spectra.TEMPORAL_REGIONS
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
    eke_bands = {"DUACS": _yearly_bands(obs_eke_by_year(obs_u, obs_v))}
    for model, rollout in rollouts.items():
        sim_u, sim_v = kernels.geostrophic_velocity_from_zos(
            rollout["zos"], lat_dim="lat", lon_dim="lon"
        )
        u_native, _ = _paired(sim_u, obs_u, _FULL_SPAN, regrid=False)
        v_native, _ = _paired(sim_v, obs_v, _FULL_SPAN, regrid=False)
        eke_bands[model] = _yearly_bands(obs_eke_by_year(u_native, v_native))
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
    for model, rollout in rollouts.items():
        native, _ = _paired(
            rollout["thetao"].isel(lev=0), obs_sst, _FULL_SPAN, regrid=True
        )
        ssta_bands[model] = _yearly_bands(
            _anomaly_by_year(kernels.calendar_day_anomaly(native))
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
    for model, rollout in rollouts.items():
        sim_u, sim_v = kernels.geostrophic_velocity_from_zos(
            rollout["zos"], lat_dim="lat", lon_dim="lon"
        )
        u_model, _ = _paired(sim_u, obs_u, _FULL_SPAN)
        v_model, _ = _paired(sim_v, obs_v, _FULL_SPAN)
        ke_bands[model] = _yearly_temporal_bands(0.5 * (u_model**2 + v_model**2))
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
    """Split an anomaly field into calendar years, keeping the time axis.

    The anomaly must already be taken about the whole record's climatology, so
    that every year shares one baseline and the band shows real year-to-year
    spread rather than each year's deviation from itself.
    """
    years = sorted({int(y) for y in pd.DatetimeIndex(anomaly["time"].values).year})
    by_year = {}
    for year in years:
        annual = anomaly.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
        if annual.sizes.get("time", 0) >= MIN_STEPS_PER_YEAR:
            by_year[year] = annual
    return by_year


def _yearly_temporal_bands(field: xr.DataArray) -> dict[str, tuple]:
    """Welch PSD of each calendar year, aggregated into a per-region band."""
    bands = {}
    for name, lon, lat in spectra.TEMPORAL_REGIONS:
        series = spectra.region_mean_series(field, lon, lat)
        stamps = pd.DatetimeIndex(series.index)
        curves = []
        for year in sorted({int(y) for y in stamps.year}):
            annual = series[stamps.year == year]
            if annual.size < MIN_STEPS_PER_YEAR:
                continue
            frequencies, power = spectra.welch_psd(annual)
            if np.asarray(frequencies).size:
                curves.append((np.asarray(frequencies), np.asarray(power)))
        bands[name] = spectra.interannual_band(curves)
    return bands


def obs_eke_by_year(u: xr.DataArray, v: xr.DataArray) -> dict[int, xr.DataArray]:
    """Annual-mean EKE maps, all sharing one multi-year velocity mean.

    The eddy anomaly is taken about the mean of the whole record, not of each
    year: a per-year mean would absorb the year-to-year change in the mean flow,
    which is the variability these bands exist to show.
    """
    eke = kernels.instantaneous_surface_eke(u, v)
    years = sorted({int(y) for y in pd.DatetimeIndex(u["time"].values).year})
    per_year = {}
    for year in years:
        annual = eke.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
        if annual.sizes.get("time", 0) < MIN_STEPS_PER_YEAR:
            continue
        per_year[year] = annual.mean("time", skipna=True)
    return per_year


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
