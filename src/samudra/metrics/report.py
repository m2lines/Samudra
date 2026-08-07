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

from samudra.metrics import comparisons, kernels
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
    "n_paired_cells",
    "paired_ocean_fraction",
]

SPATIAL_DIMS = ("lat", "lon")


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
    pairing: dict[str, float] | None = None,
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
    # The equal-year block estimate, which is what the bootstrap interval is
    # built from, so a value can never fall outside its own interval. The
    # alternative below is the same two averages in the opposite order; they
    # diverge when a product's coverage varies by year.
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
        **(pairing or {}),
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
                **(pairing or {}),
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
        **comparisons.Comparison(metric_prefix, model_var, obs_var, area).pairing,
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


def _velocity_metrics(
    model: str,
    rollout: xr.Dataset,
    duacs: xr.Dataset,
    window: tuple[pd.Timestamp, pd.Timestamp],
    bootstrap_samples: int,
    velocity_kind: str,
) -> list[dict[str, Any]]:
    """Surface geostrophic velocity vector RMSE and instantaneous EKE RMSE vs DUACS."""
    velocity = comparisons.surface_velocity(
        rollout, duacs, window, model, velocity_kind
    )
    rows, _ = _rows_for_metric(
        metric="surface_geostrophic_velocity_vector_total_rmse",
        model=model,
        units="m s-1",
        error_squared=velocity.vector_error_squared,
        area=velocity.area,
        bootstrap_samples=bootstrap_samples,
        pairing=velocity.eastward.pairing,
    )

    eke = velocity.eddy_kinetic_energy()
    eke_rows, _ = _rows_for_metric(
        metric="instantaneous_surface_eke_total_rmse",
        model=model,
        units="m2 s-2",
        error_squared=eke.error_squared,
        area=eke.area,
        bootstrap_samples=bootstrap_samples,
        pairing=eke.pairing,
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
    scored = comparisons.sea_surface_temperature(rollout, oisst, window, model)
    rows, _ = _rows_for_metric(
        metric="surface_sst_total_rmse",
        model=model,
        units="degC",
        error_squared=scored.error_squared,
        area=scored.area,
        bootstrap_samples=bootstrap_samples,
        pairing=scored.pairing,
    )

    # Residual variance uses the whole rollout, not the scoring window.
    full = comparisons.sea_surface_temperature(
        rollout, oisst, None, f"{model} SST variance"
    )
    rows += _variance_map_rows(
        metric_prefix="surface_sst",
        model=model,
        units="degC2",
        model_field=full.native,  # native grid; variance is regridded, not the field
        obs_field=full.obs,
        area=full.area,
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
    rows: list[dict[str, Any]] = []
    upper_label = kernels.OHC_LAYERS[0].label
    for layer in kernels.OHC_LAYERS:
        scored = comparisons.ohc_layer(rollout, argo, layer, window, model, model_dz)
        layer_rows, _ = _rows_for_metric(
            metric="ohc_per_area_total_rmse",
            model=model,
            units="J m-2",
            error_squared=scored.error_squared,
            area=scored.area,
            bootstrap_samples=bootstrap_samples,
            depth=layer.label,
            pairing=scored.pairing,
        )
        rows += layer_rows

        if layer.label == upper_label:
            full = comparisons.ohc_layer(
                rollout,
                argo,
                layer,
                None,
                f"{model} variance",
                model_dz,
                complete_years_only=False,
            )
            rows += _variance_map_rows(
                metric_prefix="ohc_upper700_per_area",
                model=model,
                units="(J m-2)2",
                model_field=full.native,  # native grid; variance is regridded
                obs_field=full.obs,
                area=full.area,
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
        # Per metric: they score different spans, so one figure would describe
        # whichever scored the most years.
        n_years = row.get("n_years")
        if n_years is not None and np.isfinite(n_years):
            metrics[f"{prefix}/{suffix}_n_years"] = float(n_years)

    # The window stays in the table and CSV: W&B scalars are floats, and a date
    # is not one. There is no single `obs/n_years` -- metrics score different
    # spans, so it is reported per metric above.
    return metrics


def _slug(model: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in model.lower()).strip("_")
