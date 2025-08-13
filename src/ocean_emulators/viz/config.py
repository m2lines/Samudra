import functools
import logging
import time
from functools import cached_property
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, Field, WithJsonSchema

from ocean_emulators.config import TimeConfig
from ocean_emulators.config_base import TopLevelConfig
from ocean_emulators.utils.location import LocalLocation, Location, ResolvedLocation
from ocean_emulators.utils.logging import handle_logging
from ocean_emulators.viz.core import Viz, VizRun


@functools.cache
def _all_steps() -> set[str]:
    return set(
        name.removeprefix("step_") for name in Viz.__dict__ if name.startswith("step_")
    )


def _check_step(v: str) -> str:
    if v not in _all_steps():
        raise ValueError(
            f"Invalid step: '{v}', expected one of: {', '.join(_all_steps())}"
        )
    return v


VizStep = Annotated[
    str,
    BeforeValidator(_check_step),
    WithJsonSchema({"type": "string", "enum": list(_all_steps())}),
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
    steps: list[VizStep] | None = None  # None means all
    not_steps: list[VizStep] = Field(default_factory=lambda: [])  # None means none
    debug: bool = False

    @cached_property
    def output_path(self) -> Path:
        return Path(self.base_output_dir) / self.name

    def build(self, default_root: ResolvedLocation) -> Viz:
        if self.data_root is None:
            data_root = default_root
        else:
            data_root = default_root.resolve(self.data_root)

        groundtruth_rollout = data_root.resolve(self.groundtruth_location).open(
            chunks={}
        )

        return Viz(
            # TODO(jder): change to Path
            str(self.output_path),
            self.dataset_name,
            [run.build(data_root) for run in self.runs],
            data_root.resolve(self.basins_location).open(),
            groundtruth_rollout,
            self.groundtruth_time_range.time_slice,
        )


logger = logging.getLogger(__name__)


def main(cfg: VizConfig):
    handle_logging(cfg.debug, cfg.output_path)

    logger.info(f"Writing results to {cfg.output_path}")
    cfg.output_path.mkdir(parents=True, exist_ok=True)
    cfg.save_yaml(cfg.output_path / "config.yaml")

    viz = cfg.build(LocalLocation(path=Path.cwd()))

    steps = [s for s in cfg.steps or _all_steps() if s not in cfg.not_steps]
    logger.info(f"Running steps: {', '.join(steps)}")

    # TODO(jder): could use a ProcessPoolExecutor here, but steps currently
    # are not exactly independent (some write to pred_dict which others read,
    # there's some appending to a metrics file.)
    for step in steps:
        _run_step(viz, step)


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
