from functools import cached_property
from pathlib import Path

from pydantic import BaseModel, Field

from ocean_emulators.config import TimeConfig
from ocean_emulators.config_base import TopLevelConfig
from ocean_emulators.utils.location import Location, ResolvedLocation
from ocean_emulators.viz.core import Viz, VizRun


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
