# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
import time
from functools import cached_property
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, Field, WithJsonSchema

from samudra.config import TimeConfig
from samudra.config_base import TopLevelConfig
from samudra.utils.location import LocalLocation, Location, ResolvedLocation
from samudra.utils.logging import handle_logging
from samudra.viz.core import (
    PreparedVizGroundtruth,
    Viz,
    VizRun,
    prepare_viz_groundtruth,
)


@functools.cache
def _all_steps() -> set[str]:
    return set(_ordered_steps())


@functools.cache
def _ordered_steps() -> list[str]:
    return list(
        name.removeprefix("step_") for name in Viz.__dict__ if name.startswith("step_")
    )


def _check_step(v: str) -> str:
    if v not in _all_steps():
        raise ValueError(
            f"Invalid step: '{v}', expected one of: {', '.join(_ordered_steps())}"
        )
    return v


VizStep = Annotated[
    str,
    BeforeValidator(_check_step),
    WithJsonSchema({"type": "string", "enum": _ordered_steps()}),
]


class VizRunConfig(BaseModel):
    name: str
    location: Location
    variables: list[str] = Field(
        default_factory=lambda: ["thetao", "so", "uo", "vo", "tos", "zos"]
    )

    def build(self, data_root: ResolvedLocation) -> VizRun:
        return VizRun(
            name=self.name,
            data=data_root.resolve(self.location).open(chunks={}),
            variables=self.variables,
        )


class VizConfig(TopLevelConfig):
    base_output_dir: Path
    name: str
    dataset_name: str
    runs: list[VizRunConfig]
    data_root: Location | None = None
    groundtruth_location: Location
    basins_location: Location
    # TODO(jder): we could extract this from the run data?
    groundtruth_time_range: TimeConfig = Field(
        description="Dates from the rollout (not same as eval *input* dates; these are the dates the output is produced for during eval)"
    )
    steps: list[VizStep] | None = Field(
        default=None,
        description=f"Which steps to run; leave empty to run all steps. Possible values are: {', '.join(_ordered_steps())}",
    )
    not_steps: list[VizStep] = Field(
        default_factory=lambda: [],
        description="Steps to *not* run, takes precedence over `steps` (see that key for possible steps).",
    )
    debug: bool = Field(default=False, description="")

    @cached_property
    def output_path(self) -> Path:
        return Path(self.base_output_dir) / self.name

    def _data_root(self, default_root: ResolvedLocation) -> ResolvedLocation:
        if self.data_root is None:
            return default_root
        return default_root.resolve(self.data_root)

    def prepare_groundtruth(
        self,
        default_root: ResolvedLocation,
    ) -> PreparedVizGroundtruth:
        data_root = self._data_root(default_root)
        groundtruth_rollout = data_root.resolve(self.groundtruth_location).open(
            chunks={}
        )
        return prepare_viz_groundtruth(
            self.dataset_name,
            data_root.resolve(self.basins_location).open(),
            groundtruth_rollout,
            self.groundtruth_time_range.time_slice,
        )

    def build(
        self,
        default_root: ResolvedLocation,
        prepared_groundtruth: PreparedVizGroundtruth,
    ) -> Viz:
        data_root = self._data_root(default_root)
        return Viz(
            # TODO(jder): change to Path
            str(self.output_path),
            self.dataset_name,
            [run.build(data_root) for run in self.runs],
            prepared_groundtruth=prepared_groundtruth,
        )


logger = logging.getLogger(__name__)


def _run_with_prepared_groundtruth(
    cfg: VizConfig,
    prepared_groundtruth: PreparedVizGroundtruth,
):
    cfg.output_path.mkdir(parents=True, exist_ok=True)
    handle_logging(cfg.debug, cfg.output_path)
    cfg.save_yaml(cfg.output_path / "config.yaml")

    logger.info(f"Writing results to {cfg.output_path}")

    default_root = LocalLocation(path=Path.cwd())
    viz = cfg.build(
        default_root,
        prepared_groundtruth=prepared_groundtruth,
    )

    steps = [s for s in cfg.steps or _ordered_steps() if s not in cfg.not_steps]
    logger.info(f"Running steps: {', '.join(steps)}")

    # TODO(jder): could use a ProcessPoolExecutor here, but steps currently
    # are not exactly independent (some write to pred_dict which others read,
    # there's some appending to a metrics file.)
    for step in steps:
        _run_step(viz, step)


def main(cfg: VizConfig):
    default_root = LocalLocation(path=Path.cwd())
    prepared_groundtruth = cfg.prepare_groundtruth(default_root)
    _run_with_prepared_groundtruth(cfg, prepared_groundtruth)


def run_with_prepared_groundtruth(
    cfg: VizConfig,
    prepared_groundtruth: PreparedVizGroundtruth,
):
    _run_with_prepared_groundtruth(cfg, prepared_groundtruth)


def _run_step(viz: Viz, step: VizStep):
    logger.info(f"Running step {step}")
    start = time.perf_counter()
    try:
        getattr(viz, f"step_{step}")()
        end = time.perf_counter()
        logger.info(f"Step {step} took {end - start:.2f} seconds")
    except Exception as e:
        logger.error(f"Step {step} failed: {e}")
        raise
