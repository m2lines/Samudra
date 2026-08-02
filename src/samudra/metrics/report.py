# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Driver that turns a finished rollout into observation metrics.

The output is one tidy long-format `pandas.DataFrame`: one row per
(metric, model, depth layer, period). That frame is the contract. `to_wandb`
flattens the primary rows into scalars for run-to-run comparison, and the whole
frame goes to CSV so the per-year detail behind each uncertainty interval stays
available.

Every metric follows the same shape, which is why they share so much code:
put the model on the observation grid, form the squared error, reduce it to a
per-cell map plus per-year totals, and area-weight the map to one scalar.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from samudra.metrics import kernels, observations
from samudra.utils.wandb import MetricsDict

logger = logging.getLogger(__name__)

# Column order for the emitted frame. Kept explicit so the CSV is stable across
# runs even when a metric produces no rows.
COLUMNS = [
    "metric",
    "model",
    "depth",
    "value",
    "units",
    "period_kind",
    "period_start",
    "period_end",
    "year",
    "n_time_samples",
    "annual_std",
    "ci_low",
    "ci_high",
    "map_weighted_rmse",
    "n_years",
    "n_bootstrap",
    "uncertainty_method",
    "grid_shape",
]

SPATIAL_DIMS = ("lat", "lon")


def _window_slice(window: tuple[pd.Timestamp, pd.Timestamp]) -> slice:
    return slice(window[0], window[1])


def _grid_shape(field: xr.DataArray) -> str:
    return f"{field.sizes.get('lat', 0)}x{field.sizes.get('lon', 0)}"


def _rows_for_metric(
    *,
    metric: str,
    model: str,
    units: str,
    error_squared: xr.DataArray,
    area: xr.DataArray,
    bootstrap_samples: int,
    depth: str | None = None,
) -> tuple[list[dict[str, Any]], xr.DataArray]:
    """Reduce a squared-error field to primary and per-year metric rows.

    Returns the rows and the per-cell RMSE map, so callers that also want the
    map (for figures, or for a downstream variance comparison) do not recompute.
    """
    context = f"{model} {metric}" + (f" [{depth}]" if depth else "")
    rmse_map, annual, summary = kernels.rmse_map_with_uncertainty(
        error_squared,
        area,
        SPATIAL_DIMS,
        context,
        bootstrap_samples=bootstrap_samples,
    )
    # Report the equal-year block estimate, which is the quantity the bootstrap
    # interval is built from, so the headline number and its interval are
    # consistent by construction.
    #
    # The alternative -- area-weighting the year-mean RMSE map -- is the same
    # pair of linear averages in the opposite order, and agrees exactly when the
    # finite-cell mask is the same every year. It diverges when an observation
    # product's coverage varies between years, and then the interval no longer
    # brackets the value printed beside it. Both readings fit "the square root
    # of the equal-year mean annual MSE"; this one is self-consistent.
    total = float(summary["block_aggregate_rmse"])
    map_weighted_total = kernels.area_weighted_map_rmse(rmse_map, area)
    time_index = pd.DatetimeIndex(error_squared["time"].values)

    primary = {
        "metric": metric,
        "model": model,
        "depth": depth,
        "value": total,
        "units": units,
        "period_kind": "primary_complete_years",
        "period_start": f"{time_index.min():%Y-%m-%d}",
        "period_end": f"{time_index.max():%Y-%m-%d}",
        "year": np.nan,
        "n_time_samples": int(time_index.size),
        "grid_shape": _grid_shape(rmse_map),
        # Kept alongside so the two aggregations stay comparable: a gap between
        # this and `value` means the observation coverage varied by year.
        "map_weighted_rmse": map_weighted_total,
        **summary,
    }
    primary.pop("primary_aggregation_method", None)
    primary.pop("block_aggregate_rmse", None)  # now reported as `value`

    rows = [primary]
    for year, value in annual.items():
        year_index = time_index[time_index.year == year]
        rows.append(
            {
                "metric": metric,
                "model": model,
                "depth": depth,
                "value": value,
                "units": units,
                "period_kind": "annual",
                "period_start": f"{year_index.min():%Y-%m-%d}",
                "period_end": f"{year_index.max():%Y-%m-%d}",
                "year": int(year),
                "n_time_samples": int(year_index.size),
                "grid_shape": _grid_shape(rmse_map),
            }
        )
    logger.info("  %s = %.6g %s", context, total, units)
    return rows, rmse_map


def _variance_map_rows(
    *,
    metric_prefix: str,
    model: str,
    units: str,
    model_field: xr.DataArray,
    obs_field: xr.DataArray,
    area: xr.DataArray,
) -> list[dict[str, Any]]:
    """Residual-variance map comparison: RMSE of the maps, plus pattern correlation.

    The two numbers answer different questions. The RMSE says whether the model
    has the right *amount* of residual variability; the pattern correlation says
    whether it puts that variability in the right *places*. A model can score
    well on one and badly on the other.

    `model_field` must be on the model's **native** grid. Variance is computed
    there and only the resulting variance map is regridded, because horizontal
    interpolation is linear while variance is quadratic: interpolating first
    smooths the field and systematically damps the variance. The effect scales
    with how much the regrid coarsens -- negligible for SST (quarter degree to
    quarter degree) but a ~6% underestimate for OHC (quarter to half degree).
    """
    model_var = kernels.residual_variance_map(model_field)
    obs_var = kernels.residual_variance_map(obs_field)
    model_var = kernels.model_field_on_obs_grid(model_var, obs_var)
    model_var, obs_var = xr.align(model_var, obs_var, join="inner")

    computed = xr.Dataset({"model": model_var, "obs": obs_var}).compute()
    model_var, obs_var = computed["model"], computed["obs"]

    rmse = kernels.area_weighted_map_rmse(model_var - obs_var, area)
    corr = kernels.area_weighted_pattern_corr(model_var, obs_var, area)
    time_index = pd.DatetimeIndex(model_field["time"].values)
    shared = {
        "model": model,
        "depth": None,
        "period_kind": "full_overlap",
        "period_start": f"{time_index.min():%Y-%m-%d}",
        "period_end": f"{time_index.max():%Y-%m-%d}",
        "year": np.nan,
        "n_time_samples": int(time_index.size),
        "grid_shape": _grid_shape(model_var),
    }
    logger.info(
        "  %s residual variance: map rmse = %.6g, pattern corr = %.4f",
        f"{model} {metric_prefix}",
        rmse,
        corr,
    )
    return [
        {
            "metric": f"{metric_prefix}_residual_variance_map_rmse",
            "value": rmse,
            "units": units,
            **shared,
        },
        {
            "metric": f"{metric_prefix}_residual_variance_pattern_corr",
            "value": corr,
            "units": "1",
            **shared,
        },
    ]


def _aligned(
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
    residual-variance diagnostics use: they characterise variability over the
    whole rollout rather than over the primary scoring years.
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

    model_field = model_field.sel(time=_window_slice(window))
    obs_field = obs_field.sel(time=_window_slice(window))
    if model_field.sizes.get("time", 0) == 0 or obs_field.sizes.get("time", 0) == 0:
        raise ValueError(
            f"{context}: no samples inside the requested window "
            f"{window[0]:%Y-%m-%d} to {window[1]:%Y-%m-%d}"
        )
    kernels.require_exact_time_match(model_field, obs_field, context)
    return model_field, obs_field


def _velocity_metrics(
    model: str,
    rollout: xr.Dataset,
    duacs: xr.Dataset,
    window: tuple[pd.Timestamp, pd.Timestamp],
    bootstrap_samples: int,
    velocity_kind: str,
) -> list[dict[str, Any]]:
    """Surface geostrophic velocity vector RMSE and instantaneous EKE RMSE vs DUACS."""
    obs_u, obs_v = observations.duacs_velocity(duacs, velocity_kind)
    area = duacs["area"]

    sim_u, sim_v = kernels.geostrophic_velocity_from_zos(
        rollout["zos"], lat_dim="lat", lon_dim="lon"
    )
    sim_u_native, obs_u = _aligned(sim_u, obs_u, window, f"{model} eastward velocity")
    sim_v_native, obs_v = _aligned(sim_v, obs_v, window, f"{model} northward velocity")

    # Velocity RMSE is a difference, which is linear, so regridding first is
    # exact -- and unavoidable, since the two fields must share a grid before
    # they can be subtracted.
    sim_u = kernels.model_field_on_obs_grid(sim_u_native, obs_u)
    sim_v = kernels.model_field_on_obs_grid(sim_v_native, obs_v)

    rows, _ = _rows_for_metric(
        metric="surface_geostrophic_velocity_vector_total_rmse",
        model=model,
        units="m s-1",
        # Vector RMSE combines both components before the time reduction, so a
        # cell is penalised for getting the direction wrong, not just the speed.
        error_squared=(sim_u - obs_u) ** 2 + (sim_v - obs_v) ** 2,
        area=area,
        bootstrap_samples=bootstrap_samples,
    )

    # EKE is quadratic, so it follows the same reduce-before-regrid rule as the
    # residual-variance maps: computed on the model's native grid, with only the
    # resulting EKE field interpolated. Regridding the velocities first would
    # smooth them and damp the energy, by an amount that depends on the
    # resolution ratio rather than on the model.
    #
    # This is a deliberate departure from the reference implementation, which
    # computes EKE from already-regridded velocities. The difference is small
    # today because DUACS (0.125 degrees) is finer than any model we run, so the
    # interpolation upsamples; it would grow into a real bias for a model finer
    # than DUACS.
    sim_eke_native = kernels.instantaneous_surface_eke(sim_u_native, sim_v_native)
    obs_eke = kernels.instantaneous_surface_eke(obs_u, obs_v)
    sim_eke = kernels.model_field_on_obs_grid(sim_eke_native, obs_eke)
    eke_rows, _ = _rows_for_metric(
        metric="instantaneous_surface_eke_total_rmse",
        model=model,
        units="m2 s-2",
        error_squared=(sim_eke - obs_eke) ** 2,
        area=area,
        bootstrap_samples=bootstrap_samples,
    )
    return rows + eke_rows


def _sst_metrics(
    model: str,
    rollout: xr.Dataset,
    oisst: xr.Dataset,
    window: tuple[pd.Timestamp, pd.Timestamp],
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    """SST RMSE and residual-variance map diagnostics vs OISST."""
    obs_all = oisst[observations.find_var_name(oisst, observations.SST_ALIASES)]
    area = oisst["area"]
    sim_all = rollout["thetao"].isel(lev=0)

    sim, obs = _aligned(sim_all, obs_all, window, f"{model} SST")
    rows, _ = _rows_for_metric(
        metric="surface_sst_total_rmse",
        model=model,
        units="degC",
        error_squared=(kernels.model_field_on_obs_grid(sim, obs) - obs) ** 2,
        area=area,
        bootstrap_samples=bootstrap_samples,
    )

    # Residual variance characterises variability across the whole rollout, not
    # only the complete-calendar-year scoring window.
    sim_full, obs_full = _aligned(sim_all, obs_all, None, f"{model} SST variance")
    rows += _variance_map_rows(
        metric_prefix="surface_sst",
        model=model,
        units="degC2",
        model_field=sim_full,  # native grid; variance is regridded, not the field
        obs_field=obs_full,
        area=area,
    )
    return rows


def _ohc_metrics(
    model: str,
    rollout: xr.Dataset,
    argo: xr.Dataset,
    window: tuple[pd.Timestamp, pd.Timestamp],
    bootstrap_samples: int,
    model_dz: xr.DataArray,
) -> list[dict[str, Any]]:
    """Per-area OHC RMSE by layer, plus upper-700 m residual-variance diagnostics."""
    obs_temp = argo[
        observations.find_var_name(argo, observations.ARGO_TEMPERATURE_ALIASES)
    ]
    obs_depth = observations.find_coord_name(obs_temp, observations.DEPTH_ALIASES)
    assert obs_depth is not None  # find_coord_name raises when required
    area = argo["area"]

    obs_layers = kernels.ohc_per_area_layer_maps(obs_temp, depth_name=obs_depth)
    sim_layers = kernels.ohc_per_area_layer_maps(
        rollout["thetao"], native_dz=model_dz, depth_name="lev"
    )

    rows: list[dict[str, Any]] = []
    upper_label = kernels.OHC_LAYERS[0].label
    for layer in kernels.OHC_LAYERS:
        # Monthly averaging is an explicit, documented reduction (ARGO-IAP is
        # a monthly product), unlike temporal interpolation which stays
        # forbidden. Partially covered months are dropped rather than compared
        # against a full month on the other side.
        sim_all = kernels.monthly_mean_of_complete_months(sim_layers[layer.label])
        obs_all = kernels.monthly_mean_of_complete_months(obs_layers[layer.label])

        sim, obs = _aligned(sim_all, obs_all, window, f"{model} OHC {layer.label}")
        layer_rows, _ = _rows_for_metric(
            metric="ohc_per_area_total_rmse",
            model=model,
            units="J m-2",
            error_squared=(kernels.model_field_on_obs_grid(sim, obs) - obs) ** 2,
            area=area,
            bootstrap_samples=bootstrap_samples,
            depth=layer.label,
        )
        rows += layer_rows

        if layer.label == upper_label:
            sim_full, obs_full = _aligned(
                sim_all, obs_all, None, f"{model} OHC {layer.label} variance"
            )
            rows += _variance_map_rows(
                metric_prefix="ohc_upper700_per_area",
                model=model,
                units="(J m-2)2",
                model_field=sim_full,  # native grid; variance is regridded, not the field
                obs_field=obs_full,
                area=area,
            )
    return rows


def compute_observation_metrics(
    rollouts: dict[str, xr.Dataset],
    *,
    duacs: xr.Dataset,
    oisst: xr.Dataset,
    argo: xr.Dataset,
    model_dz: dict[str, xr.DataArray],
    window: tuple[pd.Timestamp, pd.Timestamp],
    bootstrap_samples: int = kernels.DEFAULT_BOOTSTRAP_SAMPLES,
    velocity_kind: str = "absolute",
) -> pd.DataFrame:
    """Score one or more rollouts against the observation products.

    Args:
        rollouts: Label to rollout dataset, already on `(lat, lon)` dims via
            `observations.model_on_latlon_grid`. Typically the model under
            evaluation plus an OM4 baseline.
        duacs: Standardized DUACS product, for velocity and EKE.
        oisst: Standardized OISST product, for SST.
        argo: Standardized ARGO-IAP product, for ocean heat content.
        model_dz: Label to native layer thickness, for exact OHC integration.
        window: Inclusive start/end of the primary evaluation period.
        bootstrap_samples: Calendar-year block-bootstrap draws; 0 disables.
        velocity_kind: Which DUACS geostrophic velocity to compare against.

    Returns:
        Tidy frame with one row per (metric, model, depth, period).
    """
    frames: list[dict[str, Any]] = []
    for label, rollout in rollouts.items():
        logger.info("Scoring %r against observations", label)
        frames += _velocity_metrics(
            label, rollout, duacs, window, bootstrap_samples, velocity_kind
        )
        frames += _sst_metrics(label, rollout, oisst, window, bootstrap_samples)
        frames += _ohc_metrics(
            label, rollout, argo, window, bootstrap_samples, model_dz[label]
        )

    frame = pd.DataFrame(frames)
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[COLUMNS]


# Short, stable W&B key per (metric, depth). Keys have to survive refactors,
# because a renamed key silently breaks every historical run comparison.
_WANDB_KEYS = {
    ("surface_geostrophic_velocity_vector_total_rmse", None): "velocity/total_rmse",
    ("instantaneous_surface_eke_total_rmse", None): "eke/total_rmse",
    ("surface_sst_total_rmse", None): "sst/total_rmse",
    (
        "ohc_per_area_total_rmse",
        kernels.OHC_LAYERS[0].label,
    ): "ohc_0_700/per_area_total_rmse",
    (
        "ohc_per_area_total_rmse",
        kernels.OHC_LAYERS[1].label,
    ): "ohc_700_2000/per_area_total_rmse",
    (
        "surface_sst_residual_variance_map_rmse",
        None,
    ): "sst/residual_variance_map_rmse",
    (
        "surface_sst_residual_variance_pattern_corr",
        None,
    ): "sst/residual_variance_pattern_corr",
    (
        "ohc_upper700_per_area_residual_variance_map_rmse",
        None,
    ): "ohc_upper700/residual_variance_map_rmse",
    (
        "ohc_upper700_per_area_residual_variance_pattern_corr",
        None,
    ): "ohc_upper700/residual_variance_pattern_corr",
}

# Uncertainty columns worth promoting to their own scalars, so a run comparison
# can show the interval rather than only the point estimate.
_UNCERTAINTY_SUFFIXES = ("annual_std", "ci_low", "ci_high")


def to_wandb(frame: pd.DataFrame, primary_model: str) -> MetricsDict:
    """Flatten the primary rows into W&B scalars.

    The model under evaluation gets bare `obs/...` keys; every other model in
    the frame is namespaced under `obs/<model>/...` so the baseline never
    shadows the thing being evaluated.
    """
    metrics: MetricsDict = {}
    scored = frame[
        frame["period_kind"].isin(["primary_complete_years", "full_overlap"])
    ]

    for _, row in scored.iterrows():
        depth = row["depth"] if isinstance(row["depth"], str) else None
        suffix = _WANDB_KEYS.get((row["metric"], depth))
        if suffix is None:
            continue
        model = str(row["model"])
        prefix = "obs" if model == primary_model else f"obs/{_slug(model)}"
        metrics[f"{prefix}/{suffix}"] = float(row["value"])
        for column in _UNCERTAINTY_SUFFIXES:
            value = row.get(column)
            if value is not None and np.isfinite(value):
                metrics[f"{prefix}/{suffix}_{column}"] = float(value)

    # The evaluation window itself stays in the table and the CSV rather than
    # becoming a scalar: W&B scalars are floats, and a date is not one.
    primary_rows = scored[scored["model"] == primary_model]
    if not primary_rows.empty:
        n_years = primary_rows["n_years"].dropna()
        if not n_years.empty:
            metrics["obs/n_years"] = float(n_years.max())
    return metrics


def _slug(model: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in model.lower()).strip("_")
