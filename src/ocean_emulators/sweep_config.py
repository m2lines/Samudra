from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pydantic
from pydantic import Field

from ocean_emulators.config_base import BaseConfig, TopLevelConfig


class OptunaStudyConfig(BaseConfig):
    name: str = "optuna"
    direction: Literal["minimize", "maximize"] = "minimize"
    n_trials: int = 20
    timeout_seconds: int | None = None
    sampler: Literal["tpe", "random"] = "tpe"
    seed: int | None = None
    storage: str | None = None
    load_if_exists: bool = True
    n_jobs: int = 1


class FloatParamConfig(BaseConfig):
    kind: Literal["float"] = "float"
    name: str
    path: str
    low: float
    high: float
    log: bool = False
    step: float | None = None

    @pydantic.model_validator(mode="after")
    def _validate_step_log(self) -> "FloatParamConfig":
        if self.step is not None and self.log:
            raise ValueError("float params cannot use both step and log.")
        if self.step is not None and self.step <= 0:
            raise ValueError("float param step must be > 0.")
        return self


class IntParamConfig(BaseConfig):
    kind: Literal["int"] = "int"
    name: str
    path: str
    low: int
    high: int
    step: int = 1
    log: bool = False

    @pydantic.model_validator(mode="after")
    def _validate_step_log(self) -> "IntParamConfig":
        if self.step <= 0:
            raise ValueError("int param step must be > 0.")
        if self.log and self.step != 1:
            raise ValueError("int params with log=true must use step=1.")
        return self


class CategoricalParamConfig(BaseConfig):
    kind: Literal["categorical"] = "categorical"
    name: str
    path: str
    choices: list[Any]


SweepParamConfig = FloatParamConfig | IntParamConfig | CategoricalParamConfig


class SweepConfig(TopLevelConfig):
    base_config: Path = Field(
        description="Path to a TrainConfig YAML used as the sweep baseline."
    )
    base_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Nested overrides applied to the base config before trials.",
    )
    study: OptunaStudyConfig = OptunaStudyConfig()
    parameters: list[SweepParamConfig]
    output_dir: str = "sweeps"
    objective: Literal["val", "inference"] = "val"
    trial_name_template: str = "{study_name}/trial_{trial_number:04d}"
