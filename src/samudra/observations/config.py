# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Observation training config, composed from Samudra's existing config types."""

from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import Field, model_validator

from samudra.config import ExperimentConfig, TrainBackendConfig
from samudra.config_base import BaseConfig, TopLevelConfig
from samudra.models.surface_initialized import SurfaceInitializedConfig
from samudra.utils.location import LocalLocation, Location, UnresolvedLocation
from samudra.utils.schedule import LinearCooldownSchedulerConfig, SchedulerConfig

MIXES = {
    "obs": ("obs",),
    "om4-obs": ("obs", "om4"),
    "om4-cm4-obs": ("obs", "om4", "obs", "cm4"),
    "om4-cm4": ("om4", "cm4"),
}
Mix = Literal["obs", "om4-obs", "om4-cm4-obs", "om4-cm4"]


class ObservationDataConfig(BaseConfig):
    """Prepared archives; relative paths use experiment.data_root."""

    observations: Location = UnresolvedLocation(path="obs/samples")
    daily: Location = UnresolvedLocation(path="obs/daily")
    normalization: Location = UnresolvedLocation(path="obs/normalization")
    om4: Location | None = None
    cm4: Location | None = None

    def local_path(self, name: str, experiment: ExperimentConfig) -> Path:
        location = getattr(self, name)
        if location is None:
            raise ValueError(f"data.{name} is required for this source mix")
        resolved = experiment.resolved_data_root.resolve(location)
        if not isinstance(resolved, LocalLocation):
            raise ValueError(
                "Observation archives currently require local filesystem paths"
            )
        return resolved.path


class ObservationReplayConfig(BaseConfig):
    per_month: int = Field(default=2, ge=1)
    max_age: int = Field(
        default=250, ge=0, description="Maximum age in optimizer updates"
    )


class ObservationTrainConfig(TopLevelConfig):
    """Train the observation-initialized model from YAML and CLI overrides.

    Update counts and cadence count optimizer steps. One microbatch is a monthly
    source window; gradient_accumulation_steps sets the effective batch size.
    """

    experiment: ExperimentConfig
    data: ObservationDataConfig = ObservationDataConfig()
    model: SurfaceInitializedConfig
    pretrain_mix: Mix = "obs"
    main_updates: int = Field(default=2000, ge=1)
    post_training_updates: int = Field(default=1000, ge=0)
    batch_size: Literal[1] = 1
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    fresh_examples: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    # Each schedule receives its phase update budget. Post-training starts a new
    # LR curve while retaining the optimizer moments learned during pretraining.
    scheduler: SchedulerConfig | None = None
    post_training_scheduler: SchedulerConfig | None = None
    post_training_learning_rate: float | None = Field(default=None, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    gradient_clip: float = Field(default=1.0, gt=0)
    save_freq: int = Field(
        default=25, ge=1, description="Optimizer updates between saves"
    )
    validation_freq: int = Field(default=250, ge=1)
    validation_origin: str = "2013-11-01"
    ema_decay: float = Field(default=0.5 ** (1 / 100), ge=0, lt=1)
    faster_decay_at_start: bool = False
    replay: ObservationReplayConfig = ObservationReplayConfig()
    backend: TrainBackendConfig = "auto"
    debug: bool = False
    resume_ckpt_path: str | None = None
    stop_after: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_recipe(self) -> Self:
        if self.fresh_examples > self.gradient_accumulation_steps:
            raise ValueError("fresh_examples exceeds gradient_accumulation_steps")
        for scheduler, rate in (
            (self.scheduler, self.learning_rate),
            (
                self.post_training_scheduler,
                self.post_training_learning_rate or self.learning_rate,
            ),
        ):
            if (
                isinstance(scheduler, LinearCooldownSchedulerConfig)
                and scheduler.final_lr > rate
            ):
                raise ValueError(
                    "Scheduler final_lr must not exceed its phase learning rate"
                )
        for source in MIXES[self.pretrain_mix]:
            if source != "obs" and getattr(self.data, source) is None:
                raise ValueError(
                    f"pretrain_mix={self.pretrain_mix} requires data.{source}"
                )
        return self

    @property
    def total_updates(self) -> int:
        return self.main_updates + self.post_training_updates

    def source_at(self, update: int) -> str:
        if not 0 <= update < self.total_updates:
            raise ValueError("Update outside the training plan")
        cycle = MIXES[self.pretrain_mix]
        return "obs" if update >= self.main_updates else cycle[update % len(cycle)]

    def prepare_output_dirs(self) -> None:
        self.experiment.nets_dir.mkdir(parents=True, exist_ok=True)

    def save_yaml(self, save_path: Path) -> None:
        """Serialize structured Location paths as safe YAML scalar strings.

        Keep this handling local until shared config serialization is generalized.
        """
        save_path.write_text(yaml.safe_dump(self.model_dump(mode="json")))

    def scientific_config(self) -> dict[str, Any]:
        """Operational overrides can change on resume; scientific choices cannot."""
        values = self.model_dump(
            mode="json",
            exclude={
                "experiment",
                "backend",
                "debug",
                "resume_ckpt_path",
                "stop_after",
            },
        )
        values["seed"] = self.experiment.rand_seed
        return values
