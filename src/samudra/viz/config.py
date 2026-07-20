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
    VizTemplate,
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


def default_viz_variables() -> list[str]:
    return ["thetao", "so", "uo", "vo", "tos", "zos"]


class VizRunConfig(BaseModel):
    name: str
    location: Location
    variables: list[str] = Field(default_factory=default_viz_variables)

    def build(self, data_root: ResolvedLocation) -> VizRun:
        return VizRun(
            name=self.name,
            data=data_root.resolve(self.location).open(chunks={}),
            variables=self.variables,
        )


class VizTemplateConfig(TopLevelConfig):
    base_output_dir: Path
    dataset_name: str
    # Return a fresh default variable list for each config instance.
    variables: list[str] = Field(default_factory=default_viz_variables)
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

    @property
    def selected_steps(self) -> list[VizStep]:
        return [s for s in self.steps or _ordered_steps() if s not in self.not_steps]

    def build_template(self, default_root: ResolvedLocation) -> VizTemplate:
        data_root = self._data_root(default_root)
        return VizTemplate(
            dataset_name=self.dataset_name,
            data_root=data_root,
            variables=self.variables,
            prepared_groundtruth=self.prepare_groundtruth(default_root),
        )


class VizConfig(VizTemplateConfig):
    name: str
    runs: list[VizRunConfig]

    @cached_property
    def output_path(self) -> Path:
        return Path(self.base_output_dir) / self.name

    def build(self, default_root: ResolvedLocation) -> Viz:
        template = self.build_template(default_root)
        return template.instantiate(
            self.output_path,
            [run.build(template.data_root) for run in self.runs],
        )


logger = logging.getLogger(__name__)


def run_steps(viz: Viz, steps: list[VizStep]) -> None:
    logger.info(f"Running steps: {', '.join(steps)}")

    # TODO(jder): could use a ProcessPoolExecutor here, but steps currently
    # are not exactly independent (some write to pred_dict which others read,
    # there's some appending to a metrics file.)
    for step in steps:
        _run_step(viz, step)


def main(cfg: VizConfig):
    cfg.output_path.mkdir(parents=True, exist_ok=True)
    handle_logging(cfg.debug, cfg.output_path)
    cfg.save_yaml(cfg.output_path / "config.yaml")

    logger.info(f"Writing results to {cfg.output_path}")

    viz = cfg.build(LocalLocation(path=Path.cwd()))
    run_steps(viz, cfg.selected_steps)


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
