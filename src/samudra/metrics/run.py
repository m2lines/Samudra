# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Glue between an eval job and the metric kernels.

`samudra.eval` calls `run_observation_metrics` once its rollout is on disk;
`python -m samudra.metrics` calls the same function against an existing
`predictions.zarr`, so a metrics-only rerun never repeats a multi-hour rollout.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr

from samudra.constants import DatasetSpec
from samudra.metrics import observations, report
from samudra.utils.data import stack_levels
from samudra.utils.location import ResolvedLocation
from samudra.utils.wandb import MetricsDict

if TYPE_CHECKING:
    from samudra.config import ObsMetricsConfig

logger = logging.getLogger(__name__)

METRICS_CSV_NAME = "observation_metrics.csv"


def open_predictions(output_dir: Path) -> xr.Dataset:
    """Open the rollout an eval job wrote, failing clearly when it is absent."""
    path = Path(output_dir) / "predictions.zarr"
    if not path.exists():
        raise FileNotFoundError(
            f"No rollout found at {path}. Observation metrics score a rollout that "
            "has been written to disk, so the eval job must run with save_zarr=true."
        )
    return xr.open_zarr(path, chunks={})


def analysis_ready(data: xr.Dataset, dataset_spec: DatasetSpec) -> xr.Dataset:
    """Put a dataset in depth-stacked form, matching the rollout writer's layout."""
    if any(
        name in data.data_vars
        for name in (
            f"thetao_{i}" for i in range(dataset_spec.num_prognostic_depth_levels)
        )
    ):
        return stack_levels(data, dataset_spec)
    return data


def run_observation_metrics(
    obs_cfg: ObsMetricsConfig,
    *,
    predictions: xr.Dataset,
    dataset_spec: DatasetSpec,
    data_root: ResolvedLocation,
    model_label: str,
    baselines: dict[str, xr.Dataset] | None = None,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, MetricsDict]:
    """Score a rollout against observations and return the frame plus W&B scalars.

    Args:
        obs_cfg: Product locations, scoring window, and bootstrap settings.
        predictions: The rollout, in the writer's depth-stacked layout.
        dataset_spec: Spec of the dataset the rollout came from.
        data_root: Root the observation locations resolve against.
        model_label: Name for the model under evaluation, used in W&B keys.
        baselines: Extra rollouts to score alongside it, keyed by label.
        output_dir: When given, the full frame is written here as CSV.

    Returns:
        The tidy metrics frame, and the flattened scalars for W&B.
    """
    start = time.perf_counter()
    products = observations.open_products(obs_cfg, data_root)

    rollouts = {
        model_label: observations.model_on_latlon_grid(predictions, dataset_spec)
    }
    model_dz = {
        model_label: observations.model_depth_thickness(
            rollouts[model_label], dataset_spec
        )
    }
    for label, data in (baselines or {}).items():
        prepared = observations.model_on_latlon_grid(
            analysis_ready(data, dataset_spec), dataset_spec
        )
        rollouts[label] = prepared
        model_dz[label] = observations.model_depth_thickness(prepared, dataset_spec)

    logger.info(
        "Computing observation metrics over %s to %s for: %s",
        obs_cfg.rmse_start,
        obs_cfg.rmse_end,
        ", ".join(rollouts),
    )
    frame = report.compute_observation_metrics(
        rollouts,
        duacs=products["duacs"],
        oisst=products["oisst"],
        argo=products["argo"],
        model_dz=model_dz,
        window=obs_cfg.window,
        bootstrap_samples=obs_cfg.bootstrap_samples,
        velocity_kind=obs_cfg.velocity_kind,
    )

    if output_dir is not None:
        csv_path = Path(output_dir) / METRICS_CSV_NAME
        frame.to_csv(csv_path, index=False)
        logger.info("Wrote %d metric rows to %s", len(frame), csv_path)

    elapsed = time.perf_counter() - start
    logger.info("Observation metrics took %.1f s", elapsed)

    scalars = report.to_wandb(frame, model_label)
    scalars["obs/seconds"] = elapsed
    return frame, scalars
