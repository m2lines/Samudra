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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr

from samudra.constants import DataLayout
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


def analysis_ready(data: xr.Dataset, data_layout: DataLayout) -> xr.Dataset:
    """Put a dataset in depth-stacked form, matching the rollout writer's layout."""
    if any(
        name in data.data_vars
        for name in (
            f"thetao_{i}" for i in range(data_layout.num_prognostic_depth_levels)
        )
    ):
        return stack_levels(data, data_layout)
    return data


@dataclass
class ObservationRun:
    """Everything one observation pass produces.

    `eval` wants the scalars, `viz` wants the products and the rollouts to draw
    from; both want the frame. Returning all of it means the pass happens once
    per job rather than once per consumer.
    """

    products: dict[str, xr.Dataset]
    rollouts: dict[str, xr.Dataset]
    frame: pd.DataFrame
    scalars: MetricsDict


def score_rollouts(
    obs_cfg: ObsMetricsConfig,
    *,
    rollouts: dict[str, xr.Dataset],
    data_layout: DataLayout,
    data_root: ResolvedLocation,
    primary_label: str,
    output_dir: Path | None = None,
) -> ObservationRun:
    """Open the observation products and score every rollout against them.

    Args:
        obs_cfg: Product locations, scoring window, and bootstrap settings.
        rollouts: Datasets to score, keyed by the label they are reported under.
            Depth-stacked or not; either layout is accepted.
        data_layout: Layout of the dataset the rollouts came from.
        data_root: Root the observation locations resolve against.
        primary_label: Which rollout the W&B scalars describe; the others are
            baselines, reported under their own keys.
        output_dir: When given, the full frame is written here as CSV.

    Returns:
        The products, the rollouts on the observation grid, the tidy frame, and
        the flattened scalars for W&B.
    """
    start = time.perf_counter()
    products = observations.open_products(obs_cfg, data_root)

    prepared: dict[str, xr.Dataset] = {}
    model_dz: dict[str, xr.DataArray] = {}
    for label, data in rollouts.items():
        on_grid = observations.model_on_latlon_grid(
            analysis_ready(data, data_layout), data_layout
        )
        prepared[label] = on_grid
        model_dz[label] = observations.model_depth_thickness(on_grid, data_layout)

    logger.info(
        "Computing observation metrics over %s to %s for: %s",
        obs_cfg.rmse_start,
        obs_cfg.rmse_end,
        ", ".join(prepared),
    )
    frame = report.compute_observation_metrics(
        prepared,
        duacs=products["duacs"],
        oisst=products["oisst"],
        argo=products["argo"],
        model_dz=model_dz,
        window=obs_cfg.window,
        bootstrap_samples=obs_cfg.bootstrap_samples,
        velocity_kind=obs_cfg.velocity_kind,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / METRICS_CSV_NAME
        frame.to_csv(csv_path, index=False)
        logger.info("Wrote %d metric rows to %s", len(frame), csv_path)

    elapsed = time.perf_counter() - start
    logger.info("Observation metrics took %.1f s", elapsed)

    scalars = report.to_wandb(frame, primary_label)
    scalars["obs/seconds"] = elapsed
    return ObservationRun(products, prepared, frame, scalars)


def run_observation_metrics(
    obs_cfg: ObsMetricsConfig,
    *,
    predictions: xr.Dataset,
    data_layout: DataLayout,
    data_root: ResolvedLocation,
    model_label: str,
    baselines: dict[str, xr.Dataset] | None = None,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, MetricsDict]:
    """Score a rollout against observations and return the frame plus W&B scalars.

    The eval-facing shape of `score_rollouts`, which `viz` uses directly because
    it also needs the products and the prepared rollouts to draw from.
    """
    scored = score_rollouts(
        obs_cfg,
        rollouts={model_label: predictions, **(baselines or {})},
        data_layout=data_layout,
        data_root=data_root,
        primary_label=model_label,
        output_dir=output_dir,
    )
    return scored.frame, scored.scalars
